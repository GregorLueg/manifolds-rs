//! GPU-accelerated UMAP Adam optimiser. Mirrors
//! `optimise_embedding_adam_parallel` but device-resident throughout: state is
//! uploaded once, three kernels run per epoch (gradient accumulation, Adam
//! moment update and step, edge-schedule advancement) and the embedding is
//! read back once at the end.

#![allow(missing_docs)] // cubecl weirdness

use ann_search_rs::gpu::tensor::GpuTensor;
use ann_search_rs::gpu::{grid_2d, WORKGROUP_SIZE_X};
use cubecl::prelude::*;

use crate::prelude::*;
use crate::training::umap_optimisers::UmapOptimParams;

///////////
// Enums //
///////////

/// Type of UMAP optimiser to use
#[derive(Default)]
pub enum UmapOptimiserGpu {
    /// A GPU-accelerated version of the Adam optimiser
    #[default]
    AdamGpu,
    /// Parallel version of Adam
    AdamParallel,
    /// Adam
    Adam,
    /// Stochastic gradient descent
    Sgd,
}

/// Parse the UMAP Optimiser to use
///
/// ### Params
///
/// * `s` - String defining the optimiser. Choice of `"adam"`, `"adam_parallel"`
///   `"sgd"` or `"adam_gpu"` (default).
///
/// ### Return
///
/// Option of Optimiser
pub fn parse_umap_optimiser_gpu(s: &str) -> Option<UmapOptimiserGpu> {
    match s.to_lowercase().as_str() {
        "adam_gpu" => Some(UmapOptimiserGpu::AdamGpu),
        "adam" => Some(UmapOptimiserGpu::Adam),
        "sgd" => Some(UmapOptimiserGpu::Sgd),
        "adam_parallel" => Some(UmapOptimiserGpu::AdamParallel),
        _ => None,
    }
}

////////////
// Consts //
////////////

/// Symmetric clamp on the per-negative-sample repulsive gradient coefficient.
/// Prevents blow-up when a negative sample lands arbitrarily close to the
/// source point.
const GRAD_CLIP_VAL: f64 = 4.0;

/// Additive epsilon on the repulsive `dist_sq` to keep the denominator finite
/// when a negative sample coincides with the source point.
const GRAD_REP_EPS: f64 = 0.001;

/// Attractive gradients are skipped when the pairwise `dist_sq` falls below
/// this threshold, avoiding a division by zero when two endpoints have
/// collapsed onto each other.
const GRAD_DIST_SQ_THRESHOLD: f64 = 1e-8;

///////////////////
// Host-side CSR //
///////////////////

/// Host-side CSR representation of the symmetrised UMAP graph, ready to be
/// uploaded to the GPU. Each undirected edge `(i, j, w)` with `i < j` appears
/// twice in the CSR arrays: once under node `i` and once under node `j`. Only
/// the smaller-indexed side of an edge schedules negative samples, matching
/// the CPU code.
pub struct UmapCsrGraph<T> {
    /// Number of nodes.
    pub n: usize,
    /// Number of unique undirected edges.
    pub n_edges: usize,
    /// CSR row pointers `[n + 1]`.
    pub node_edge_offsets: Vec<u32>,
    /// Edge index per `(node, edge)` entry `[2 * n_edges]`.
    pub csr_edge_idx: Vec<u32>,
    /// Other endpoint per `(node, edge)` entry `[2 * n_edges]`.
    pub csr_other_node: Vec<u32>,
    /// Sampling period per edge `[n_edges]`. `max_weight / w` for
    /// non-zero-weight edges; a large sentinel value for zero-weight edges so
    /// that they effectively never tick.
    pub epochs_per_sample: Vec<T>,
}

impl<T> UmapCsrGraph<T>
where
    T: ManifoldsFloat,
{
    /// Build a CSR layout from an adjacency-list graph.
    ///
    /// The `i < j` filter deduplicates entries when the input graph is
    /// symmetric, and silently drops self-loops.
    ///
    /// ### Params
    ///
    /// * `graph` - Adjacency list `graph[i] = [(j, w), ...]`. Weights `w` are
    ///   assumed non-negative.
    ///
    /// ### Returns
    ///
    /// A CSR representation on success. `ManifoldsError::NoData` if the graph
    /// has no nodes; `ManifoldsError::NoGraphEdges` if no edges survive the
    /// `i < j` filter.
    pub fn from_graph(graph: &[Vec<(usize, T)>]) -> Result<Self, ManifoldsError> {
        let n = graph.len();
        if n == 0 {
            return Err(ManifoldsError::NoData);
        }

        let mut edges: Vec<(usize, usize, T)> = Vec::new();
        let mut degree = vec![0u32; n];

        for (i, neighbours) in graph.iter().enumerate() {
            for &(j, w) in neighbours {
                if i < j {
                    edges.push((i, j, w));
                    degree[i] += 1;
                    degree[j] += 1;
                }
            }
        }

        if edges.is_empty() {
            return Err(ManifoldsError::NoGraphEdges);
        }

        let n_edges = edges.len();

        let mut node_edge_offsets = vec![0u32; n + 1];
        for i in 0..n {
            node_edge_offsets[i + 1] = node_edge_offsets[i] + degree[i];
        }

        let mut csr_edge_idx = vec![0u32; 2 * n_edges];
        let mut csr_other_node = vec![0u32; 2 * n_edges];

        let mut cursor = node_edge_offsets.clone();
        for (edge_idx, &(i, j, _)) in edges.iter().enumerate() {
            let pos_i = cursor[i] as usize;
            csr_edge_idx[pos_i] = edge_idx as u32;
            csr_other_node[pos_i] = j as u32;
            cursor[i] += 1;

            let pos_j = cursor[j] as usize;
            csr_edge_idx[pos_j] = edge_idx as u32;
            csr_other_node[pos_j] = i as u32;
            cursor[j] += 1;
        }

        let zero = T::zero();
        let one = T::one();
        let large_epoch = T::from(1e8).unwrap();
        let max_weight = edges
            .iter()
            .map(|(_, _, w)| *w)
            .fold(zero, |acc, w| if w > acc { w } else { acc });

        let epochs_per_sample: Vec<T> = edges
            .iter()
            .map(|(_, _, w)| {
                let norm = *w / max_weight;
                if norm > zero {
                    one / norm
                } else {
                    large_epoch
                }
            })
            .collect();

        Ok(Self {
            n,
            n_edges,
            node_edge_offsets,
            csr_edge_idx,
            csr_other_node,
            epochs_per_sample,
        })
    }
}

//////////////////
// Device state //
//////////////////

/// All GPU-resident state for the UMAP optimiser. Built once via
/// [`UmapGpuState::upload`], mutated in place by the per-epoch kernels.
pub struct UmapGpuState<R, T>
where
    R: Runtime,
    T: ManifoldsFloatGpu,
{
    /// Current embedding `[n, n_dim]`, updated every epoch by the Adam step.
    pub embd: GpuTensor<R, T>,
    /// First Adam moment `[n, n_dim]`, updated every epoch.
    pub m: GpuTensor<R, T>,
    /// Second Adam moment `[n, n_dim]`, updated every epoch.
    pub v: GpuTensor<R, T>,
    /// Per-node gradient scratch `[n, n_dim]`, overwritten every epoch.
    pub node_grad: GpuTensor<R, T>,

    /// CSR row pointers `[n + 1]`, uploaded once.
    pub node_edge_offsets: GpuTensor<R, u32>,
    /// CSR edge indices `[2 * n_edges]`, uploaded once.
    pub csr_edge_idx: GpuTensor<R, u32>,
    /// CSR other endpoints `[2 * n_edges]`, uploaded once.
    pub csr_other_node: GpuTensor<R, u32>,

    /// Sampling period per edge `[n_edges]`, uploaded once.
    pub epochs_per_sample: GpuTensor<R, T>,
    /// Edge sampling cursor `[n_edges]`, advanced every epoch for edges that
    /// tick.
    pub epoch_of_next_sample: GpuTensor<R, T>,

    /// Per-node flag `[n]`, `1` if any edge fired this epoch, else `0`.
    /// Overwritten by `umap_grad_accum`; read by `umap_adam_update` to skip
    /// nodes with no active edges.
    pub has_update: GpuTensor<R, u32>,

    /// Number of nodes.
    pub n: usize,
    /// Embedding dimensionality.
    pub n_dim: usize,
    /// Number of unique undirected edges.
    pub n_edges: usize,
    /// Total number of optimisation epochs.
    pub n_epochs: usize,
    /// Per-node partial `sum_d grad[node, d]^2` `[n]`, written by
    /// `umap_grad_norm_sq` and reduced on the host for progress logging.
    pub grad_norm_partial: GpuTensor<R, T>,
}

impl<R, T> UmapGpuState<R, T>
where
    R: Runtime,
    T: ManifoldsFloatGpu,
{
    /// Build device state from CPU inputs. All buffers are uploaded; no
    /// kernels run.
    ///
    /// ### Params
    ///
    /// * `embd` - Initial embedding `[n][n_dim]`, uploaded as a flat
    ///   row-major buffer
    /// * `csr` - Host-side CSR representation of the symmetrised UMAP graph
    /// * `params` - Adam and UMAP hyperparameters
    /// * `client` - CubeCL compute client for the target device
    ///
    /// ### Returns
    ///
    /// Device state on success. `ManifoldsError::NoData` if `embd` is empty.
    pub fn upload(
        embd: &[Vec<T>],
        csr: &UmapCsrGraph<T>,
        params: &UmapOptimParams<T>,
        client: &ComputeClient<R>,
    ) -> Result<Self, ManifoldsError> {
        let n = embd.len();
        if n == 0 {
            return Err(ManifoldsError::NoData);
        }
        let n_dim = embd[0].len();
        let n_edges = csr.n_edges;
        let n_epochs = params.n_epochs;

        let mut embd_flat: Vec<T> = Vec::with_capacity(n * n_dim);
        for point in embd {
            embd_flat.extend_from_slice(point);
        }

        let zeros = vec![T::zero(); n * n_dim];

        Ok(Self {
            embd: GpuTensor::from_slice(&embd_flat, vec![n, n_dim], client),
            m: GpuTensor::from_slice(&zeros, vec![n, n_dim], client),
            v: GpuTensor::from_slice(&zeros, vec![n, n_dim], client),
            node_grad: GpuTensor::from_slice(&zeros, vec![n, n_dim], client),

            node_edge_offsets: GpuTensor::from_slice(&csr.node_edge_offsets, vec![n + 1], client),
            csr_edge_idx: GpuTensor::from_slice(&csr.csr_edge_idx, vec![2 * n_edges], client),
            csr_other_node: GpuTensor::from_slice(&csr.csr_other_node, vec![2 * n_edges], client),

            epochs_per_sample: GpuTensor::from_slice(&csr.epochs_per_sample, vec![n_edges], client),
            // Cursors initialise to one period from time 0 (first sample event).
            epoch_of_next_sample: GpuTensor::from_slice(
                &csr.epochs_per_sample,
                vec![n_edges],
                client,
            ),

            has_update: GpuTensor::from_slice(&vec![0u32; n], vec![n], client),

            grad_norm_partial: GpuTensor::from_slice(&vec![T::zero(); n], vec![n], client),

            n,
            n_dim,
            n_edges,
            n_epochs,
        })
    }
}

//////////
// Hash //
//////////

/// Splitmix-style device-side hash.
///
/// Maps `(seed, node, epoch, edge_local, neg)` to a node index in `[0, n)`.
/// Called in the inner negative-sampling loop of `umap_grad_accum`.
///
/// ### Params
///
/// * `seed` - Random seed for negative sampling, constant across the run
/// * `node` - Source node index
/// * `epoch` - Current epoch index
/// * `edge_local` - Per-node edge counter, incremented for every edge in the
///   node's CSR slice regardless of whether it ticks this epoch
/// * `neg` - Negative-sample index within the current edge, `0..neg_sample_rate`
/// * `n` - Number of nodes; the returned index is reduced modulo `n`
///
/// ### Returns
///
/// A node index in `[0, n)`. The caller is responsible for rejecting the
/// self-hit case `k == node`.
#[cube]
fn gpu_hash_neg(seed: u32, node: u32, epoch: u32, edge_local: u32, neg: u32, n: u32) -> u32 {
    let mut h = seed
        ^ (node * 0x9E3779B1u32)
        ^ (epoch * 0x85EBCA77u32)
        ^ (edge_local * 0xC2B2AE3Du32)
        ^ (neg * 0x27D4EB2Fu32);
    h ^= h >> 16u32;
    h *= 0x7FEB352Du32;
    h ^= h >> 15u32;
    h *= 0x846CA68Bu32;
    h ^= h >> 16u32;
    h % n
}

/////////////
// Kernels //
/////////////

/// Per-node gradient accumulation. One thread per node.
///
/// Each thread walks its CSR slice of edges, accumulates the attractive force
/// into a thread-local gradient buffer, then for each active edge draws
/// `neg_sample_rate` negatives via `gpu_hash_neg` and accumulates repulsive
/// contributions. Writes `node_grad[node]` in full (overwrite, not
/// accumulate) and sets `has_update[node]` to `1` iff at least one edge fired
/// this epoch. No atomics, no shared memory. Reads `epoch_of_next_sample` but
/// never writes it; the schedule update is a separate kernel.
///
/// ### Params
///
/// * `embd` - Current embedding `[n, n_dim]`
/// * `node_edge_offsets` - CSR row pointers `[n + 1]`
/// * `csr_edge_idx` - CSR edge indices `[2 * n_edges]`
/// * `csr_other_node` - CSR other-endpoint indices `[2 * n_edges]`
/// * `epoch_of_next_sample` - Edge sampling cursors `[n_edges]`
/// * `node_grad` - Per-node gradient output `[n, n_dim]`
/// * `has_update` - Per-node active flag output `[n]`
/// * `n` - Number of nodes
/// * `n_dim` - Embedding dimensionality
/// * `epoch` - Current epoch as `u32`
/// * `epoch_f` - Current epoch as `F`, used to compare against
///   `epoch_of_next_sample`
/// * `seed` - Random seed for negative sampling
/// * `neg_sample_rate` - Number of negatives drawn per active edge
/// * `a`, `b` - UMAP curve parameters
/// * `two_a_b` - Precomputed `2 * a * b`
/// * `two_gamma_b` - Precomputed `2 * gamma * b`
/// * `clip_val` - Symmetric clamp on the repulsive gradient coefficient
/// * `rep_eps` - Additive epsilon on repulsive `dist_sq`
/// * `dist_sq_threshold` - Below this, attractive gradient is skipped
/// * `wg_size` - Workgroup size; comptime
/// * `n_dim_ct` - Embedding dimensionality; comptime, matches `n_dim`
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X` -> node
#[cube(launch_unchecked)]
pub fn umap_grad_accum<F: Float + CubeElement>(
    embd: &Tensor<F>,
    node_edge_offsets: &Tensor<u32>,
    csr_edge_idx: &Tensor<u32>,
    csr_other_node: &Tensor<u32>,
    epoch_of_next_sample: &Tensor<F>,
    node_grad: &mut Tensor<F>,
    has_update: &mut Tensor<u32>,
    n: u32,
    n_dim: u32,
    epoch: u32,
    epoch_f: F,
    seed: u32,
    neg_sample_rate: u32,
    a: F,
    b: F,
    two_a_b: F,
    two_gamma_b: F,
    clip_val: F,
    rep_eps: F,
    dist_sq_threshold: F,
    #[comptime] wg_size: u32,
    #[comptime] n_dim_ct: u32,
) {
    let node = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X;
    if node >= n {
        terminate!();
    }

    let base_self = node * n_dim;

    let mut grad = Array::<F>::new(n_dim_ct as usize);
    for d in 0..n_dim_ct {
        grad[d as usize] = F::new(0.0);
    }

    let start = node_edge_offsets[node as usize];
    let end = node_edge_offsets[(node + 1u32) as usize];

    let two = F::new(2.0);

    let mut edge_local: u32 = 0u32;
    let mut active: u32 = 0u32;
    let mut pos = start;
    while pos < end {
        let edge_idx = csr_edge_idx[pos as usize];
        if epoch_of_next_sample[edge_idx as usize] <= epoch_f {
            active = 1u32;
            let other = csr_other_node[pos as usize];
            let base_other = other * n_dim;

            let mut dist_sq = F::new(0.0);
            for d in 0..n_dim_ct {
                let diff = embd[(base_self + d) as usize] - embd[(base_other + d) as usize];
                dist_sq += diff * diff;
            }

            if dist_sq >= dist_sq_threshold {
                let dist_sq_b = F::powf(dist_sq, b);
                let denom = F::new(1.0) + a * dist_sq_b;
                let grad_coeff = two_a_b * dist_sq_b / (dist_sq * denom);

                for d in 0..n_dim_ct {
                    let delta = embd[(base_other + d) as usize] - embd[(base_self + d) as usize];
                    grad[d as usize] += two * grad_coeff * delta;
                }
            }

            let mut neg: u32 = 0u32;
            while neg < neg_sample_rate {
                let k = gpu_hash_neg(seed, node, epoch, edge_local, neg, n);
                if k != node {
                    let base_k = k * n_dim;

                    let mut dist_sq_k = F::new(0.0);
                    for d in 0..n_dim_ct {
                        let diff = embd[(base_self + d) as usize] - embd[(base_k + d) as usize];
                        dist_sq_k += diff * diff;
                    }

                    let dist_sq_safe = dist_sq_k + rep_eps;
                    let dist_sq_b = F::powf(dist_sq_safe, b);
                    let denom = dist_sq_safe * (F::new(1.0) + a * dist_sq_b);
                    let mut grad_coeff = two_gamma_b / denom;
                    if grad_coeff > clip_val {
                        grad_coeff = clip_val;
                    }
                    if grad_coeff < -clip_val {
                        grad_coeff = -clip_val;
                    }

                    for d in 0..n_dim_ct {
                        let delta = embd[(base_self + d) as usize] - embd[(base_k + d) as usize];
                        grad[d as usize] += grad_coeff * delta;
                    }
                }
                neg += 1u32;
            }
        }
        edge_local += 1u32;
        pos += 1u32;
    }

    for d in 0..n_dim_ct {
        node_grad[(base_self + d) as usize] = grad[d as usize];
    }
    has_update[node as usize] = active;
}

/// Adam moment update and embedding step. One thread per `(node, dim)` pair.
///
/// Skips nodes with `has_update == 0` (no active edge this epoch),
/// preserving their `m`, `v` and embedding state — matching the CPU
/// `optimise_embedding_adam_parallel`.
///
/// ### Params
///
/// * `node_grad` - Gradients `[n, n_dim]` from `umap_grad_accum`
/// * `has_update` - Per-node active flag `[n]`
/// * `m` - First Adam moment `[n, n_dim]`, updated in place
/// * `v` - Second Adam moment `[n, n_dim]`, updated in place
/// * `embd` - Embedding `[n, n_dim]`, updated in place
/// * `n_total` - `n * n_dim`
/// * `n_dim` - Embedding dimensionality
/// * `lr_alpha` - Fused `lr * (1 - epoch / n_epochs) * ad_scale`
/// * `epsc` - Bias-corrected `sqrt(1 - beta2^t) * eps`
/// * `one_minus_beta1` - `1 - beta1`
/// * `one_minus_beta2` - `1 - beta2`
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> flat `(node, dim)` index into the `[n * n_dim]`
///   buffers
#[cube(launch_unchecked)]
pub fn umap_adam_update<F: Float + CubeElement>(
    node_grad: &Tensor<F>,
    has_update: &Tensor<u32>,
    m: &mut Tensor<F>,
    v: &mut Tensor<F>,
    embd: &mut Tensor<F>,
    n_total: u32,
    n_dim: u32,
    lr_alpha: F,
    epsc: F,
    one_minus_beta1: F,
    one_minus_beta2: F,
) {
    let i = ABSOLUTE_POS_X;
    if i >= n_total {
        terminate!();
    }

    let node = i / n_dim;
    if has_update[node as usize] == 0u32 {
        terminate!();
    }

    let g = node_grad[i as usize];

    let m_old = m[i as usize];
    let m_new = m_old + one_minus_beta1 * (g - m_old);
    m[i as usize] = m_new;

    let v_old = v[i as usize];
    let v_new = v_old + one_minus_beta2 * (g * g - v_old);
    v[i as usize] = v_new;

    embd[i as usize] += lr_alpha * m_new / (F::sqrt(v_new) + epsc);
}

/// Advance edge sampling cursors for edges that ticked this epoch. One
/// thread per edge.
///
/// ### Params
///
/// * `epochs_per_sample` - Sampling period per edge `[n_edges]`
/// * `epoch_of_next_sample` - Edge sampling cursors `[n_edges]`, advanced in
///   place for edges with `cursor <= epoch_f`
/// * `n_edges` - Number of unique undirected edges
/// * `epoch_f` - Current epoch as `F`
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> edge index
#[cube(launch_unchecked)]
pub fn umap_edge_schedule_update<F: Float + CubeElement>(
    epochs_per_sample: &Tensor<F>,
    epoch_of_next_sample: &mut Tensor<F>,
    n_edges: u32,
    epoch_f: F,
) {
    let e = ABSOLUTE_POS_X;
    if e >= n_edges {
        terminate!();
    }
    if epoch_of_next_sample[e as usize] <= epoch_f {
        epoch_of_next_sample[e as usize] += epochs_per_sample[e as usize];
    }
}

/// Per-node sum of squares of `node_grad`. One thread per node. Writes
/// `partial[node] = sum_d node_grad[node, d]^2`; the host sums the resulting
/// `[n]` buffer to get the squared global gradient norm. Used only for
/// progress logging; not part of the optimisation state.
///
/// ### Params
///
/// * `node_grad` - Per-node gradients `[n, n_dim]` from `umap_grad_accum`
/// * `partial` - Per-node sum-of-squares output `[n]`, overwritten
/// * `n` - Number of nodes
/// * `n_dim` - Embedding dimensionality
/// * `wg_size` - Workgroup size; comptime
/// * `n_dim_ct` - Embedding dimensionality; comptime, matches `n_dim`
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X` -> node
#[cube(launch_unchecked)]
pub fn umap_grad_norm_sq<F: Float + CubeElement>(
    node_grad: &Tensor<F>,
    partial: &mut Tensor<F>,
    n: u32,
    n_dim: u32,
    #[comptime] wg_size: u32,
    #[comptime] n_dim_ct: u32,
) {
    let node = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * wg_size + UNIT_POS_X;
    if node >= n {
        terminate!();
    }
    let base = node * n_dim;
    let mut acc = F::new(0.0);
    for d in 0..n_dim_ct {
        let g = node_grad[(base + d) as usize];
        acc += g * g;
    }
    partial[node as usize] = acc;
}

//////////////
// Launcher //
//////////////

/// Dispatch `umap_grad_accum` for one epoch.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `state` - Device-resident optimiser state
/// * `params` - Adam and UMAP hyperparameters
/// * `epoch` - Current epoch index (0-based)
/// * `seed` - Random seed for negative sampling
pub fn launch_grad_accum<R, T>(
    client: &ComputeClient<R>,
    state: &UmapGpuState<R, T>,
    params: &UmapOptimParams<T>,
    epoch: usize,
    seed: u32,
) where
    R: Runtime,
    T: ManifoldsFloatGpu,
{
    let two = T::from(2.0).unwrap();
    let wg = WORKGROUP_SIZE_X;
    let n_workgroups = (state.n as u32).div_ceil(wg);
    let (gx, gy) = grid_2d(n_workgroups);

    unsafe {
        umap_grad_accum::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(wg),
            state.embd.clone().into_tensor_arg(),
            state.node_edge_offsets.clone().into_tensor_arg(),
            state.csr_edge_idx.clone().into_tensor_arg(),
            state.csr_other_node.clone().into_tensor_arg(),
            state.epoch_of_next_sample.clone().into_tensor_arg(),
            state.node_grad.clone().into_tensor_arg(),
            state.has_update.clone().into_tensor_arg(),
            state.n as u32,
            state.n_dim as u32,
            epoch as u32,
            T::from(epoch).unwrap(),
            seed,
            params.neg_sample_rate as u32,
            params.a,
            params.b,
            two * params.a * params.b,
            two * params.gamma * params.b,
            T::from(GRAD_CLIP_VAL).unwrap(),
            T::from(GRAD_REP_EPS).unwrap(),
            T::from(GRAD_DIST_SQ_THRESHOLD).unwrap(),
            wg,
            state.n_dim as u32,
        );
    }
}

/// Dispatch `umap_adam_update` for one epoch. Computes the per-epoch Adam
/// bias-correction and learning-rate schedule on the host and passes them as
/// scalars.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `state` - Device-resident optimiser state
/// * `params` - Adam and UMAP hyperparameters
/// * `epoch` - Current epoch index (0-based)
pub fn launch_adam_update<R, T>(
    client: &ComputeClient<R>,
    state: &UmapGpuState<R, T>,
    params: &UmapOptimParams<T>,
    epoch: usize,
) where
    R: Runtime,
    T: ManifoldsFloatGpu,
{
    let one = T::one();
    let n_total = (state.n * state.n_dim) as u32;
    let wg = WORKGROUP_SIZE_X;
    let n_workgroups = n_total.div_ceil(wg);
    let (gx, gy) = grid_2d(n_workgroups);

    let lr_alpha = params.lr * (one - T::from(epoch).unwrap() / T::from(state.n_epochs).unwrap());

    let t = T::from(epoch + 1).unwrap();
    let beta1t = num_traits::Float::powf(params.beta1, t);
    let beta2t = num_traits::Float::powf(params.beta2, t);
    let sqrt_b2t1 = num_traits::Float::sqrt(one - beta2t);
    let ad_scale = sqrt_b2t1 / (one - beta1t);
    let epsc = sqrt_b2t1 * params.eps;

    unsafe {
        umap_adam_update::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(wg),
            state.node_grad.clone().into_tensor_arg(),
            state.has_update.clone().into_tensor_arg(),
            state.m.clone().into_tensor_arg(),
            state.v.clone().into_tensor_arg(),
            state.embd.clone().into_tensor_arg(),
            n_total,
            state.n_dim as u32,
            lr_alpha * ad_scale,
            epsc,
            one - params.beta1,
            one - params.beta2,
        );
    }
}

/// Dispatch `umap_edge_schedule_update` for one epoch.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `state` - Device-resident optimiser state
/// * `epoch` - Current epoch index (0-based)
pub fn launch_edge_schedule_update<R, T>(
    client: &ComputeClient<R>,
    state: &UmapGpuState<R, T>,
    epoch: usize,
) where
    R: Runtime,
    T: ManifoldsFloatGpu,
{
    let wg = WORKGROUP_SIZE_X;
    let n_workgroups = (state.n_edges as u32).div_ceil(wg);
    let (gx, gy) = grid_2d(n_workgroups);

    unsafe {
        umap_edge_schedule_update::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(wg),
            state.epochs_per_sample.clone().into_tensor_arg(),
            state.epoch_of_next_sample.clone().into_tensor_arg(),
            state.n_edges as u32,
            T::from(epoch).unwrap(),
        );
    }
}

/// Dispatch `umap_grad_norm_sq`, read the `[n]` partial buffer back and
/// finish the reduction on the host. Synchronous: the readback flushes the
/// command queue, so this should only be called on logging epochs. Reads
/// `state.node_grad` as populated by the most recent `umap_grad_accum`
/// launch; the subsequent `umap_adam_update` does not touch `node_grad`, so
/// calling this after the full per-epoch triple is fine.
///
/// ### Params
///
/// * `client` - CubeCL compute client
/// * `state` - Device-resident optimiser state
///
/// ### Returns
///
/// Global gradient norm `sqrt(sum_{node, d} node_grad[node, d]^2)` on
/// success. Propagates any readback error from `GpuTensor::read`.
pub fn launch_grad_norm<R, T>(
    client: &ComputeClient<R>,
    state: &UmapGpuState<R, T>,
) -> Result<T, ManifoldsError>
where
    R: Runtime,
    T: ManifoldsFloatGpu,
{
    let wg = WORKGROUP_SIZE_X;
    let n_workgroups = (state.n as u32).div_ceil(wg);
    let (gx, gy) = grid_2d(n_workgroups);

    unsafe {
        umap_grad_norm_sq::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(gx, gy, 1),
            CubeDim::new_1d(wg),
            state.node_grad.clone().into_tensor_arg(),
            state.grad_norm_partial.clone().into_tensor_arg(),
            state.n as u32,
            state.n_dim as u32,
            wg,
            state.n_dim as u32,
        );
    }

    let partial = state.grad_norm_partial.clone().read(client)?;
    let sum_sq = partial.iter().copied().fold(T::zero(), |a, b| a + b);
    Ok(num_traits::Float::sqrt(sum_sq))
}

//////////
// Main //
//////////

/// Run the full UMAP Adam optimisation on the GPU. Mirrors
/// `optimise_embedding_adam_parallel` but device-resident throughout: state
/// is uploaded once, three kernels run per epoch (gradient accumulation,
/// Adam step, edge-schedule advancement) and the embedding is read back once
/// at the end.
///
/// ### Params
///
/// * `embd` - Initial embedding `[n][n_dim]`, modified in place with the
///   optimised coordinates on return
/// * `graph` - Adjacency-list representation of the symmetrised UMAP graph.
///   Self-loops are silently dropped; only `i < j` entries are used to build
///   the CSR
/// * `params` - Adam and UMAP hyperparameters
/// * `device` - CubeCL runtime device
/// * `seed` - Random seed for negative sampling. Truncated to `u32` on the
///   device
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// `Ok(())` on success; the optimised embedding is written back into `embd`
/// in place. `ManifoldsError::NoData` if `embd` is empty;
/// `ManifoldsError::NoGraphEdges` if the CSR graph has no surviving edges.
pub fn optimise_embedding_adam_gpu<R, T>(
    embd: &mut [Vec<T>],
    graph: &[Vec<(usize, T)>],
    params: &UmapOptimParams<T>,
    device: R::Device,
    seed: u64,
    verbose: usize,
) -> Result<(), ManifoldsError>
where
    R: Runtime,
    T: ManifoldsFloatGpu,
{
    let n = embd.len();
    if n == 0 {
        return Err(ManifoldsError::NoData);
    }
    let n_dim = embd[0].len();
    let verbosity = parse_verbosity_level(verbose);

    let csr = UmapCsrGraph::from_graph(graph)?;
    let client = R::client(&device);
    let state = UmapGpuState::<R, T>::upload(embd, &csr, params, &client)?;

    // seed fits in u32; upper bits of a u64 seed are unused.
    let seed_u32 = seed as u32;

    if verbosity.normal_verbosity() {
        println!(
            "Running {} epochs with GPU-accelerated Adam optimisation.",
            params.n_epochs
        );
    }

    for epoch in 0..params.n_epochs {
        launch_grad_accum(&client, &state, params, epoch, seed_u32);
        launch_adam_update(&client, &state, params, epoch);
        launch_edge_schedule_update(&client, &state, epoch);

        if verbosity.normal_verbosity() && ((epoch + 1) % 100 == 0 || epoch + 1 == params.n_epochs)
        {
            let gn = launch_grad_norm(&client, &state)?;
            println!(
                " Epoch {} / {}: grad norm {:.2?}",
                epoch + 1,
                params.n_epochs,
                gn
            );
        }
    }

    let final_flat = state.embd.read(&client)?;
    for (i, point) in embd.iter_mut().enumerate() {
        let base = i * n_dim;
        point.copy_from_slice(&final_flat[base..base + n_dim]);
    }

    Ok(())
}

/////////////////////////////
// CPU equivalence testers //
/////////////////////////////

/// Splitmix-style host-side hash matching `gpu_hash_neg` bit-for-bit. Kept in
/// sync manually so the CPU reference reproduces the same negative samples.
#[cfg(test)]
#[inline]
fn cpu_hash_neg(seed: u32, node: u32, epoch: u32, edge_local: u32, neg: u32, n: u32) -> u32 {
    let mut h = seed
        ^ node.wrapping_mul(0x9E3779B1)
        ^ epoch.wrapping_mul(0x85EBCA77)
        ^ edge_local.wrapping_mul(0xC2B2AE3D)
        ^ neg.wrapping_mul(0x27D4EB2F);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846CA68B);
    h ^= h >> 16;
    h % n
}

/// Host-side reference implementation of `umap_grad_accum`. Same iteration
/// order as the kernel so attractive-only runs match bit-for-bit; runs with
/// negatives are FP-tolerant because `powf` and FMAs may compile differently
/// on the GPU.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn cpu_grad_accum<T>(
    embd_flat: &[T],
    csr: &UmapCsrGraph<T>,
    epoch_of_next_sample: &[T],
    n_dim: usize,
    epoch: usize,
    seed: u32,
    neg_sample_rate: usize,
    params: &UmapOptimParams<T>,
) -> Vec<T>
where
    T: ManifoldsFloat,
{
    let n = csr.n;
    let mut grad = vec![T::zero(); n * n_dim];

    let two = T::from(2.0).unwrap();
    let one = T::one();
    let zero = T::zero();
    let dist_sq_threshold = T::from(GRAD_DIST_SQ_THRESHOLD).unwrap();
    let rep_eps = T::from(GRAD_REP_EPS).unwrap();
    let clip_val = T::from(GRAD_CLIP_VAL).unwrap();
    let two_a_b = two * params.a * params.b;
    let two_gamma_b = two * params.gamma * params.b;
    let epoch_t = T::from(epoch).unwrap();

    for node in 0..n {
        let base_self = node * n_dim;
        let start = csr.node_edge_offsets[node] as usize;
        let end = csr.node_edge_offsets[node + 1] as usize;

        let mut edge_local: u32 = 0;
        for pos in start..end {
            let edge_idx = csr.csr_edge_idx[pos] as usize;
            if epoch_of_next_sample[edge_idx] > epoch_t {
                edge_local += 1;
                continue;
            }

            let other = csr.csr_other_node[pos] as usize;
            let base_other = other * n_dim;

            let mut dist_sq = zero;
            for d in 0..n_dim {
                let diff = embd_flat[base_self + d] - embd_flat[base_other + d];
                dist_sq += diff * diff;
            }

            if dist_sq >= dist_sq_threshold {
                let dist_sq_b = dist_sq.powf(params.b);
                let denom = one + params.a * dist_sq_b;
                let grad_coeff = two_a_b * dist_sq_b / (dist_sq * denom);

                for d in 0..n_dim {
                    let delta = embd_flat[base_other + d] - embd_flat[base_self + d];
                    grad[base_self + d] += two * grad_coeff * delta;
                }
            }

            for neg in 0..neg_sample_rate as u32 {
                let k = cpu_hash_neg(seed, node as u32, epoch as u32, edge_local, neg, n as u32)
                    as usize;
                if k == node {
                    continue;
                }
                let base_k = k * n_dim;

                let mut dist_sq_k = zero;
                for d in 0..n_dim {
                    let diff = embd_flat[base_self + d] - embd_flat[base_k + d];
                    dist_sq_k += diff * diff;
                }

                let dist_sq_safe = dist_sq_k + rep_eps;
                let dist_sq_b = dist_sq_safe.powf(params.b);
                let denom = dist_sq_safe * (one + params.a * dist_sq_b);
                let mut grad_coeff = two_gamma_b / denom;
                if grad_coeff > clip_val {
                    grad_coeff = clip_val;
                }
                if grad_coeff < -clip_val {
                    grad_coeff = -clip_val;
                }

                for d in 0..n_dim {
                    let delta = embd_flat[base_self + d] - embd_flat[base_k + d];
                    grad[base_self + d] += grad_coeff * delta;
                }
            }

            edge_local += 1;
        }
    }

    grad
}

/// Host-side reference implementation of `umap_adam_update` for one epoch.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn cpu_adam_update<T>(
    node_grad: &[T],
    has_update: &[u32],
    m: &mut [T],
    v: &mut [T],
    embd: &mut [T],
    n: usize,
    n_dim: usize,
    params: &UmapOptimParams<T>,
    epoch: usize,
    n_epochs: usize,
) where
    T: ManifoldsFloat,
{
    let one = T::one();
    let lr_alpha = params.lr * (one - T::from(epoch).unwrap() / T::from(n_epochs).unwrap());
    let t = T::from(epoch + 1).unwrap();
    let beta1t = params.beta1.powf(t);
    let beta2t = params.beta2.powf(t);
    let sqrt_b2t1 = (one - beta2t).sqrt();
    let ad_scale = sqrt_b2t1 / (one - beta1t);
    let epsc = sqrt_b2t1 * params.eps;
    let one_minus_beta1 = one - params.beta1;
    let one_minus_beta2 = one - params.beta2;
    let fused = lr_alpha * ad_scale;

    for node in 0..n {
        if has_update[node] == 0 {
            continue;
        }
        for d in 0..n_dim {
            let i = node * n_dim + d;
            let g = node_grad[i];

            let m_new = m[i] + one_minus_beta1 * (g - m[i]);
            m[i] = m_new;

            let v_new = v[i] + one_minus_beta2 * (g * g - v[i]);
            v[i] = v_new;

            embd[i] += fused * m_new / (v_new.sqrt() + epsc);
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    // -- CPU / GPU host fixtures --

    type TriangleSetup = (Vec<Vec<f32>>, Vec<Vec<(usize, f32)>>, UmapOptimParams<f32>);

    // Symmetric triangle, weights (0,1)=1.0, (0,2)=0.5, (1,2)=1.0.
    fn triangle_graph() -> Vec<Vec<(usize, f64)>> {
        vec![
            vec![(1, 1.0), (2, 0.5)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 0.5), (1, 1.0)],
        ]
    }

    fn triangle_setup() -> TriangleSetup {
        let embd = vec![vec![0.0_f32, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let graph = vec![
            vec![(1_usize, 1.0_f32), (2, 0.5)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 0.5), (1, 1.0)],
        ];
        let params = UmapOptimParams {
            a: 1.5,
            b: 0.9,
            lr: 1.0,
            gamma: 1.0,
            n_epochs: 10,
            neg_sample_rate: 5,
            min_dist: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-7,
        };
        (embd, graph, params)
    }

    fn build_state(
        device: &WgpuDevice,
    ) -> (
        ComputeClient<WgpuRuntime>,
        UmapCsrGraph<f32>,
        UmapOptimParams<f32>,
        UmapGpuState<WgpuRuntime, f32>,
    ) {
        let client = WgpuRuntime::client(device);
        let (embd, graph, params) = triangle_setup();
        let csr = UmapCsrGraph::from_graph(&graph).unwrap();
        let state =
            UmapGpuState::<WgpuRuntime, f32>::upload(&embd, &csr, &params, &client).unwrap();
        (client, csr, params, state)
    }

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32, ctx: &str) {
        assert_eq!(got.len(), want.len(), "{}: length mismatch", ctx);
        for i in 0..got.len() {
            let d = (got[i] - want[i]).abs();
            assert!(
                d <= tol,
                "{}: elem {}: got {} want {} (diff {})",
                ctx,
                i,
                got[i],
                want[i],
                d,
            );
        }
    }

    // -- CSR --

    #[test]
    fn test_csr_triangle_layout() {
        let g = triangle_graph();
        let csr = UmapCsrGraph::<f64>::from_graph(&g).unwrap();

        assert_eq!(csr.n, 3);
        assert_eq!(csr.n_edges, 3);
        assert_eq!(csr.node_edge_offsets, vec![0, 2, 4, 6]);

        // Edge ordering: (0,1)=edge0, (0,2)=edge1, (1,2)=edge2.
        assert_eq!(csr.csr_edge_idx, vec![0, 1, 0, 2, 1, 2]);
        assert_eq!(csr.csr_other_node, vec![1, 2, 0, 2, 0, 1]);
    }

    #[test]
    fn test_csr_triangle_sampling_periods() {
        let g = triangle_graph();
        let csr = UmapCsrGraph::<f64>::from_graph(&g).unwrap();

        // max_weight = 1.0; epochs_per_sample = max_weight / w.
        assert_eq!(csr.epochs_per_sample, vec![1.0, 2.0, 1.0]);
    }

    // Invariant: degree accumulation must equal 2 * n_edges (each edge
    // contributes to two endpoints).
    #[test]
    fn test_csr_offsets_sum_to_2x_edges() {
        let g = triangle_graph();
        let csr = UmapCsrGraph::<f64>::from_graph(&g).unwrap();
        assert_eq!(
            *csr.node_edge_offsets.last().unwrap() as usize,
            2 * csr.n_edges
        );
    }

    #[test]
    fn test_csr_zero_nodes_error() {
        let g: Vec<Vec<(usize, f64)>> = vec![];
        assert!(matches!(
            UmapCsrGraph::<f64>::from_graph(&g),
            Err(ManifoldsError::NoData)
        ));
    }

    #[test]
    fn test_csr_no_edges_error() {
        let g: Vec<Vec<(usize, f64)>> = vec![vec![], vec![], vec![]];
        assert!(matches!(
            UmapCsrGraph::<f64>::from_graph(&g),
            Err(ManifoldsError::NoGraphEdges)
        ));
    }

    // Self-loops `(i, i, w)` must be dropped by the i < j filter.
    #[test]
    fn test_csr_drops_self_loops() {
        let g: Vec<Vec<(usize, f64)>> = vec![vec![(0, 1.0), (1, 1.0)], vec![(1, 1.0), (0, 1.0)]];
        let csr = UmapCsrGraph::<f64>::from_graph(&g).unwrap();
        assert_eq!(csr.n_edges, 1);
        assert_eq!(csr.csr_other_node, vec![1, 0]);
    }

    // Asymmetric input: only graph[0] specifies edges; still gets a valid CSR.
    #[test]
    fn test_csr_asymmetric_input() {
        let g: Vec<Vec<(usize, f64)>> = vec![vec![(1, 1.0), (2, 0.5)], vec![], vec![]];
        let csr = UmapCsrGraph::<f64>::from_graph(&g).unwrap();
        assert_eq!(csr.n_edges, 2);
        assert_eq!(csr.node_edge_offsets, vec![0, 2, 3, 4]);
    }

    // -- Upload / device state --

    #[test]
    fn test_embedding_roundtrip() {
        let Some(device) = try_device() else { return };
        let (client, _, _, state) = build_state(&device);
        let got = state.embd.clone().read(&client).unwrap();
        let want = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0];
        assert_eq!(got, want);
    }

    #[test]
    fn test_m_v_node_grad_zeroed() {
        let Some(device) = try_device() else { return };
        let (client, _, _, state) = build_state(&device);
        let zeros = vec![0.0_f32; state.n * state.n_dim];
        assert_eq!(state.m.clone().read(&client).unwrap(), zeros);
        assert_eq!(state.v.clone().read(&client).unwrap(), zeros);
        assert_eq!(state.node_grad.clone().read(&client).unwrap(), zeros);
    }

    #[test]
    fn test_csr_arrays_roundtrip() {
        let Some(device) = try_device() else { return };
        let (client, csr, _, state) = build_state(&device);
        assert_eq!(
            state.node_edge_offsets.clone().read(&client).unwrap(),
            csr.node_edge_offsets
        );
        assert_eq!(
            state.csr_edge_idx.clone().read(&client).unwrap(),
            csr.csr_edge_idx
        );
        assert_eq!(
            state.csr_other_node.clone().read(&client).unwrap(),
            csr.csr_other_node
        );
    }

    #[test]
    fn test_edge_schedule_initial_state() {
        let Some(device) = try_device() else { return };
        let (client, csr, _, state) = build_state(&device);
        assert_eq!(
            state.epochs_per_sample.clone().read(&client).unwrap(),
            csr.epochs_per_sample
        );
        // Cursors initialised to one period from time 0.
        assert_eq!(
            state.epoch_of_next_sample.clone().read(&client).unwrap(),
            csr.epochs_per_sample
        );
    }

    // n_dim != 2 must still work; tests the dim handling end-to-end.
    #[test]
    fn test_upload_3d_embedding() {
        let Some(device) = try_device() else { return };
        let client = WgpuRuntime::client(&device);
        let embd = vec![
            vec![0.0_f32, 1.0, 2.0],
            vec![3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0],
        ];
        let (_, graph, mut params) = triangle_setup();
        params.n_epochs = 5;
        let csr = UmapCsrGraph::from_graph(&graph).unwrap();
        let state =
            UmapGpuState::<WgpuRuntime, f32>::upload(&embd, &csr, &params, &client).unwrap();

        assert_eq!(state.n_dim, 3);
        let got = state.embd.clone().read(&client).unwrap();
        assert_eq!(got, vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    // -- Gradient kernel --

    fn run_grad_kernel(
        device: &WgpuDevice,
        embd: &[Vec<f32>],
        graph: &[Vec<(usize, f32)>],
        params: &UmapOptimParams<f32>,
        epoch: usize,
        seed: u32,
    ) -> (Vec<f32>, UmapCsrGraph<f32>, Vec<f32>) {
        let client = WgpuRuntime::client(device);
        let csr = UmapCsrGraph::from_graph(graph).unwrap();
        let state = UmapGpuState::<WgpuRuntime, f32>::upload(embd, &csr, params, &client).unwrap();

        launch_grad_accum(&client, &state, params, epoch, seed);

        let grad = state.node_grad.clone().read(&client).unwrap();
        let next_sample = state.epoch_of_next_sample.clone().read(&client).unwrap();
        (grad, csr, next_sample)
    }

    // No edges have ticked yet (epoch 0, periods >= 1.0) -> gradient is zero.
    #[test]
    fn test_grad_empty_epoch() {
        let Some(device) = try_device() else { return };
        let (embd, graph, params) = triangle_setup();

        let (got, _, _) = run_grad_kernel(&device, &embd, &graph, &params, 0, 42);
        assert_eq!(got, vec![0.0_f32; embd.len() * embd[0].len()]);
    }

    // Attractive only. neg_sample_rate = 0 + an epoch that activates every
    // edge. Bit-exact equality with the CPU reference: each node accumulates
    // serially in the same order on both sides, with no FP reordering.
    #[test]
    fn test_grad_attractive_only_bit_exact() {
        let Some(device) = try_device() else { return };
        let (embd, graph, mut params) = triangle_setup();
        params.neg_sample_rate = 0;

        // max_weight is 1.0, so epochs_per_sample <= 2.0; epoch=5 ticks everything.
        let epoch = 5;
        let (got, csr, next_sample) = run_grad_kernel(&device, &embd, &graph, &params, epoch, 42);

        let embd_flat: Vec<f32> = embd.iter().flatten().copied().collect();
        let want = cpu_grad_accum(
            &embd_flat,
            &csr,
            &next_sample,
            embd[0].len(),
            epoch,
            42,
            params.neg_sample_rate,
            &params,
        );

        assert_close(&got, &want, 0.0, "attractive-only");
    }

    // With negatives, matched hash. FP-tolerant: powf and FMAs may compile
    // differently on GPU.
    #[test]
    fn test_grad_with_negatives_matches_cpu() {
        let Some(device) = try_device() else { return };
        let (embd, graph, params) = triangle_setup();

        let epoch = 5;
        let (got, csr, next_sample) = run_grad_kernel(&device, &embd, &graph, &params, epoch, 42);

        let embd_flat: Vec<f32> = embd.iter().flatten().copied().collect();
        let want = cpu_grad_accum(
            &embd_flat,
            &csr,
            &next_sample,
            embd[0].len(),
            epoch,
            42,
            params.neg_sample_rate,
            &params,
        );

        assert_close(&got, &want, 1e-4, "with-negatives");
    }

    // Single hand-computed active edge. Pair of points at (0,0) and (3,0),
    // one edge, no negatives. Verifies sign and magnitude of the attractive
    // force without depending on the CPU reference.
    #[test]
    fn test_grad_single_edge_hand_computed() {
        let Some(device) = try_device() else { return };
        let embd = vec![vec![0.0_f32, 0.0], vec![3.0, 0.0]];
        let graph = vec![vec![(1_usize, 1.0_f32)], vec![(0, 1.0)]];
        let params = UmapOptimParams::<f32> {
            a: 1.0,
            b: 1.0,
            lr: 1.0,
            gamma: 1.0,
            n_epochs: 10,
            neg_sample_rate: 0,
            min_dist: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-7,
        };

        let (got, _, _) = run_grad_kernel(&device, &embd, &graph, &params, 5, 42);

        // dist_sq = 9, b = 1, a = 1
        // dist_sq_b = 9
        // denom = 1 + 1*9 = 10
        // grad_coeff = 2*1*1 * 9 / (9 * 10) = 0.2
        // For node 0: delta = embd[1] - embd[0] = (3, 0)
        //   grad[0] = 2 * 0.2 * (3, 0) = (1.2, 0)
        // For node 1: delta = embd[0] - embd[1] = (-3, 0)
        //   grad[1] = 2 * 0.2 * (-3, 0) = (-1.2, 0)
        assert_close(&got, &[1.2, 0.0, -1.2, 0.0], 1e-5, "hand-computed");
    }

    // Determinism. Same inputs, same seed, same epoch -> bit-exact result on
    // a second launch.
    #[test]
    fn test_grad_deterministic() {
        let Some(device) = try_device() else { return };
        let (embd, graph, params) = triangle_setup();

        let (got1, _, _) = run_grad_kernel(&device, &embd, &graph, &params, 5, 42);
        let (got2, _, _) = run_grad_kernel(&device, &embd, &graph, &params, 5, 42);
        assert_eq!(got1, got2);
    }

    // -- Adam kernel --

    // Build a state, manually set node_grad / m / v / embd / has_update, run
    // the Adam kernel, return updated (embd, m, v).
    #[allow(clippy::too_many_arguments)]
    fn run_adam(
        device: &WgpuDevice,
        embd: &[f32],
        m: &[f32],
        v: &[f32],
        node_grad: &[f32],
        has_update: &[u32],
        n: usize,
        n_dim: usize,
        params: &UmapOptimParams<f32>,
        epoch: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let client = WgpuRuntime::client(device);

        // Adam kernel never touches CSR or schedule state, so those buffers
        // are just placeholders sized so nothing indexes out of bounds.
        let placeholder_u32 = GpuTensor::from_slice(&[0u32], vec![1], &client);
        let placeholder_f = GpuTensor::from_slice(&[0.0f32], vec![1], &client);

        let state = UmapGpuState::<WgpuRuntime, f32> {
            embd: GpuTensor::from_slice(embd, vec![n, n_dim], &client),
            m: GpuTensor::from_slice(m, vec![n, n_dim], &client),
            v: GpuTensor::from_slice(v, vec![n, n_dim], &client),
            node_grad: GpuTensor::from_slice(node_grad, vec![n, n_dim], &client),
            has_update: GpuTensor::from_slice(has_update, vec![n], &client),

            node_edge_offsets: placeholder_u32.clone(),
            csr_edge_idx: placeholder_u32.clone(),
            csr_other_node: placeholder_u32,
            epochs_per_sample: placeholder_f.clone(),
            epoch_of_next_sample: placeholder_f.clone(),

            grad_norm_partial: placeholder_f.clone(),

            n,
            n_dim,
            n_edges: 0,
            n_epochs: params.n_epochs,
        };

        launch_adam_update(&client, &state, params, epoch);

        (
            state.embd.clone().read(&client).unwrap(),
            state.m.clone().read(&client).unwrap(),
            state.v.clone().read(&client).unwrap(),
        )
    }

    fn make_params() -> UmapOptimParams<f32> {
        UmapOptimParams {
            a: 1.5,
            b: 0.9,
            lr: 1.0,
            gamma: 1.0,
            n_epochs: 10,
            neg_sample_rate: 5,
            min_dist: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-7,
        }
    }

    // has_update = 0 -> embd, m, v all preserved bit-exact.
    #[test]
    fn test_adam_inactive_node_preserved() {
        let Some(device) = try_device() else { return };
        let params = make_params();

        let embd = vec![1.5_f32, -2.5, 7.0, 8.0];
        let m = vec![0.1_f32, 0.2, 0.3, 0.4];
        let v = vec![0.01_f32, 0.02, 0.03, 0.04];
        let grad = vec![5.0_f32, 5.0, 5.0, 5.0]; // non-zero, would update if active
        let has_update = vec![0_u32, 0];

        let (e, mm, vv) = run_adam(&device, &embd, &m, &v, &grad, &has_update, 2, 2, &params, 3);

        assert_eq!(e, embd);
        assert_eq!(mm, m);
        assert_eq!(vv, v);
    }

    // Hand-computed single Adam step at epoch 0.
    #[test]
    fn test_adam_single_step_hand_computed() {
        let Some(device) = try_device() else { return };
        let mut params = make_params();
        params.lr = 1.0;
        params.beta1 = 0.9;
        params.beta2 = 0.999;
        params.eps = 1e-7;
        params.n_epochs = 10;

        let embd = vec![0.0_f32];
        let m = vec![0.0_f32];
        let v = vec![0.0_f32];
        let grad = vec![1.0_f32];
        let has_update = vec![1_u32];

        let (e, mm, vv) = run_adam(&device, &embd, &m, &v, &grad, &has_update, 1, 1, &params, 0);

        // m_new = 0 + (1 - 0.9) * (1 - 0) = 0.1
        // v_new = 0 + (1 - 0.999) * (1 - 0) = 0.001
        // alpha = 1.0 * (1 - 0/10) = 1.0
        // t = 1, beta1^1 = 0.9, beta2^1 = 0.999
        // sqrt(1 - 0.999) = sqrt(0.001) ~= 0.0316228
        // ad_scale = 0.0316228 / (1 - 0.9) = 0.316228
        // epsc = 0.0316228 * 1e-7 ~= 3.16e-9
        // embd += 1.0 * 0.316228 * 0.1 / (sqrt(0.001) + 3.16e-9)
        //       = 0.0316228 / 0.0316228 ~= 1.0
        assert_close(&mm, &[0.1], 1e-6, "m");
        assert_close(&vv, &[0.001], 1e-6, "v");
        assert_close(&e, &[1.0], 1e-4, "embd");
    }

    // Bit-exact match vs CPU reference. Mix of active and inactive nodes.
    #[test]
    fn test_adam_matches_cpu_reference() {
        let Some(device) = try_device() else { return };
        let params = make_params();

        let n = 5;
        let n_dim = 2;
        let embd: Vec<f32> = (0..n * n_dim).map(|i| (i as f32) * 0.5 - 1.0).collect();
        let m: Vec<f32> = (0..n * n_dim).map(|i| (i as f32) * 0.01).collect();
        let v: Vec<f32> = (0..n * n_dim).map(|i| (i as f32) * 0.001 + 1e-6).collect();
        let grad: Vec<f32> = (0..n * n_dim)
            .map(|i| ((i * 7 + 3) % 5) as f32 * 0.1)
            .collect();
        let has_update = vec![1_u32, 0, 1, 1, 0];

        let epoch = 4;

        let (e_gpu, m_gpu, v_gpu) = run_adam(
            &device,
            &embd,
            &m,
            &v,
            &grad,
            &has_update,
            n,
            n_dim,
            &params,
            epoch,
        );

        let mut e_cpu = embd.clone();
        let mut m_cpu = m.clone();
        let mut v_cpu = v.clone();
        cpu_adam_update(
            &grad,
            &has_update,
            &mut m_cpu,
            &mut v_cpu,
            &mut e_cpu,
            n,
            n_dim,
            &params,
            epoch,
            params.n_epochs,
        );

        // FP-tolerant: sqrt and FMAs may compile differently on GPU.
        assert_close(&e_gpu, &e_cpu, 1e-5, "embd");
        assert_close(&m_gpu, &m_cpu, 1e-6, "m");
        assert_close(&v_gpu, &v_cpu, 1e-6, "v");
    }

    #[test]
    fn test_adam_deterministic() {
        let Some(device) = try_device() else { return };
        let params = make_params();

        let n = 3;
        let n_dim = 2;
        let embd = vec![0.5_f32; n * n_dim];
        let m = vec![0.1_f32; n * n_dim];
        let v = vec![0.01_f32; n * n_dim];
        let grad = vec![0.3_f32; n * n_dim];
        let has_update = vec![1_u32; n];

        let (e1, m1, v1) = run_adam(
            &device,
            &embd,
            &m,
            &v,
            &grad,
            &has_update,
            n,
            n_dim,
            &params,
            2,
        );
        let (e2, m2, v2) = run_adam(
            &device,
            &embd,
            &m,
            &v,
            &grad,
            &has_update,
            n,
            n_dim,
            &params,
            2,
        );

        assert_eq!(e1, e2);
        assert_eq!(m1, m2);
        assert_eq!(v1, v2);
    }

    // -- Edge schedule --

    #[test]
    fn test_schedule_active_edges_advance() {
        let Some(device) = try_device() else { return };
        let (client, csr, _, state) = build_state(&device);

        // csr.epochs_per_sample = [1.0, 2.0, 1.0], initial cursors match.
        // At epoch=5, all three edges are active (5 >= 1, 2, 1) -> all advance.
        launch_edge_schedule_update(&client, &state, 5);

        let got = state.epoch_of_next_sample.clone().read(&client).unwrap();
        let want: Vec<f32> = csr.epochs_per_sample.iter().map(|p| p * 2.0).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn test_schedule_inactive_edges_untouched() {
        let Some(device) = try_device() else { return };
        let (client, csr, _, state) = build_state(&device);

        // At epoch=0, no edge has ticked yet (cursors are 1.0, 2.0, 1.0 all > 0).
        launch_edge_schedule_update(&client, &state, 0);

        let got = state.epoch_of_next_sample.clone().read(&client).unwrap();
        assert_eq!(got, csr.epochs_per_sample);
    }

    #[test]
    fn test_schedule_mixed_active_inactive() {
        let Some(device) = try_device() else { return };
        let (client, _, _, state) = build_state(&device);

        // At epoch=1, cursors [1.0, 2.0, 1.0]: edges 0 and 2 tick (1 >= 1),
        // edge 1 doesn't (1 < 2).
        launch_edge_schedule_update(&client, &state, 1);

        let got = state.epoch_of_next_sample.clone().read(&client).unwrap();
        assert_eq!(got, vec![2.0, 2.0, 2.0]);
    }

    // -- Driver loop --

    // Two triangles connected by a weak edge; final embedding should compress
    // each triangle and leave the bridge distance larger. Mirrors
    // test_optimisation_preserves_graph_structure_adam on the CPU side.
    #[test]
    fn test_gpu_driver_preserves_graph_structure() {
        let Some(_device) = try_device() else { return };

        let graph = vec![
            vec![(1, 1.0_f32), (2, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 1.0), (1, 1.0), (3, 0.1)],
            vec![(2, 0.1), (4, 1.0), (5, 1.0)],
            vec![(3, 1.0), (5, 1.0)],
            vec![(3, 1.0), (4, 1.0)],
        ];

        let mut embd = vec![
            vec![0.0_f32, 0.0],
            vec![10.0, 0.0],
            vec![0.0, 10.0],
            vec![10.0, 10.0],
            vec![-5.0, -5.0],
            vec![15.0, 15.0],
        ];

        let mut params = UmapOptimParams::<f32>::default_2d();
        params.n_epochs = 200;

        optimise_embedding_adam_gpu::<WgpuRuntime, f32>(
            &mut embd,
            &graph,
            &params,
            WgpuDevice::DefaultDevice,
            42,
            0,
        )
        .unwrap();

        let dist = |a: &[f32], b: &[f32]| -> f32 {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };

        let intra1 =
            (dist(&embd[0], &embd[1]) + dist(&embd[0], &embd[2]) + dist(&embd[1], &embd[2])) / 3.0;
        let intra2 =
            (dist(&embd[3], &embd[4]) + dist(&embd[3], &embd[5]) + dist(&embd[4], &embd[5])) / 3.0;
        let avg_intra = (intra1 + intra2) / 2.0;

        let inter = [
            dist(&embd[0], &embd[3]),
            dist(&embd[0], &embd[4]),
            dist(&embd[0], &embd[5]),
            dist(&embd[1], &embd[3]),
            dist(&embd[1], &embd[4]),
            dist(&embd[1], &embd[5]),
        ];
        let avg_inter: f32 = inter.iter().sum::<f32>() / inter.len() as f32;

        assert!(
            avg_inter > avg_intra * 1.5,
            "Inter-clique dist ({:.2}) should be > 1.5x intra-clique dist ({:.2})",
            avg_inter,
            avg_intra
        );
    }

    // All coords finite after a full run.
    #[test]
    fn test_gpu_driver_no_nan_or_inf() {
        let Some(_device) = try_device() else { return };

        let (mut embd, graph, params) = triangle_setup();
        optimise_embedding_adam_gpu::<WgpuRuntime, f32>(
            &mut embd,
            &graph,
            &params,
            WgpuDevice::DefaultDevice,
            42,
            0,
        )
        .unwrap();

        for point in &embd {
            for &c in point {
                assert!(c.is_finite(), "non-finite coordinate: {}", c);
            }
        }
    }

    // Same seed -> same result. Confirms nothing sneaks in host-side
    // non-determinism.
    #[test]
    fn test_gpu_driver_deterministic() {
        let Some(_device) = try_device() else { return };

        let (mut embd1, graph, params) = triangle_setup();
        let mut embd2 = embd1.clone();

        optimise_embedding_adam_gpu::<WgpuRuntime, f32>(
            &mut embd1,
            &graph,
            &params,
            WgpuDevice::DefaultDevice,
            42,
            0,
        )
        .unwrap();
        optimise_embedding_adam_gpu::<WgpuRuntime, f32>(
            &mut embd2,
            &graph,
            &params,
            WgpuDevice::DefaultDevice,
            42,
            0,
        )
        .unwrap();

        assert_eq!(embd1, embd2);
    }
}
