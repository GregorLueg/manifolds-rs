//! GPU-accelerated UMAP optimiser. Mirrors `optimise_embedding_adam_parallel`
//! but offloads gradient accumulation, Adam updates and edge-schedule
//! advancement to the GPU.

use ann_search_rs::gpu::tensor::GpuTensor;
use ann_search_rs::gpu::{grid_2d, WORKGROUP_SIZE_X};
use cubecl::prelude::*;

use crate::prelude::*;
use crate::training::umap_optimisers::UmapOptimParams;

///////////////////
// Host-side CSR //
///////////////////

/// Host-side CSR representation of the symmetrised UMAP graph, ready to be
/// uploaded to the GPU. Each undirected edge `(i, j, w)` with `i < j` appears
/// twice in the CSR arrays: once under node `i` (with `is_smaller = 1`) and
/// once under node `j` (with `is_smaller = 0`). Only the smaller-indexed side
/// of an edge schedules negative samples, matching the CPU code.
pub(crate) struct UmapCsrGraph<T> {
    /// Number of nodes
    pub n: usize,
    /// Number of unique undirected edges
    pub n_edges: usize,
    /// CSR row pointers `[n + 1]`
    pub node_edge_offsets: Vec<u32>,
    /// Edge index per `(node, edge)` entry `[2 * n_edges]`
    pub csr_edge_idx: Vec<u32>,
    /// Other endpoint per `(node, edge)` entry `[2 * n_edges]`
    pub csr_other_node: Vec<u32>,
    /// Sampling period per edge `[n_edges]`
    pub epochs_per_sample: Vec<T>,
}

impl<T> UmapCsrGraph<T>
where
    T: ManifoldsFloat,
{
    /// Build CSR layout from an adjacency-list graph.
    ///
    /// The `i < j` filter deduplicates entries when the input graph is
    /// symmetric, and silently drops self-loops.
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

/// All GPU-resident state for the UMAP optimiser. Built once via `upload`,
/// mutated in place by the per-epoch kernels.
pub(crate) struct UmapGpuState<R, T>
where
    R: Runtime,
    T: ManifoldsFloatGpu,
{
    // Mutable, updated every epoch
    pub embd: GpuTensor<R, T>,      // [n, n_dim]
    pub m: GpuTensor<R, T>,         // [n, n_dim]
    pub v: GpuTensor<R, T>,         // [n, n_dim]
    pub node_grad: GpuTensor<R, T>, // [n, n_dim] scratch

    // Immutable CSR, uploaded once
    pub node_edge_offsets: GpuTensor<R, u32>, // [n + 1]
    pub csr_edge_idx: GpuTensor<R, u32>,      // [2 * n_edges]
    pub csr_other_node: GpuTensor<R, u32>,    // [2 * n_edges]

    // Edge schedule
    pub epochs_per_sample: GpuTensor<R, T>,
    pub epoch_of_next_sample: GpuTensor<R, T>,

    // Pre-baked per-epoch schedules
    pub lr_schedule: GpuTensor<R, T>,
    pub ad_scale: GpuTensor<R, T>,
    pub epsc: GpuTensor<R, T>,

    pub has_update: GpuTensor<R, u32>, // [n], 0 or 1

    pub n: usize,
    pub n_dim: usize,
    pub n_edges: usize,
    pub n_epochs: usize,
}

impl<R, T> UmapGpuState<R, T>
where
    R: Runtime,
    T: ManifoldsFloatGpu,
{
    /// Build device state from CPU inputs. All buffers are uploaded; no
    /// kernels run.
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

        // Per-epoch schedules. Formulas mirror optimise_embedding_adam_parallel
        // exactly so a downstream Adam update kernel can read these element-
        // by-element.
        let one = T::one();
        let n_epochs_f = T::from(n_epochs).unwrap();
        let lr_schedule: Vec<T> = (0..n_epochs)
            .map(|e| params.lr * (one - T::from(e).unwrap() / n_epochs_f))
            .collect();

        let (ad_scale, epsc): (Vec<T>, Vec<T>) = (0..n_epochs)
            .map(|epoch| {
                let t = T::from(epoch + 1).unwrap();
                let beta1t = num_traits::Float::powf(params.beta1, t);
                let beta2t = num_traits::Float::powf(params.beta2, t);
                let sqrt_b2t1 = num_traits::Float::sqrt(one - beta2t);
                (sqrt_b2t1 / (one - beta1t), sqrt_b2t1 * params.eps)
            })
            .unzip();

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

            // Cursors initialise to the period itself (first sample event)
            epoch_of_next_sample: GpuTensor::from_slice(
                &csr.epochs_per_sample,
                vec![n_edges],
                client,
            ),

            lr_schedule: GpuTensor::from_slice(&lr_schedule, vec![n_epochs], client),
            ad_scale: GpuTensor::from_slice(&ad_scale, vec![n_epochs], client),
            epsc: GpuTensor::from_slice(&epsc, vec![n_epochs], client),

            has_update: GpuTensor::from_slice(&vec![0u32; n], vec![n], client),

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

// Splitmix-style finaliser. Maps (seed, node, epoch, edge_local, neg) -> [0, n).
// Same constants are reproduced in the kernel below; keep them in sync.
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

/// Per-node gradient accumulation. One thread per node. Each thread walks its
/// CSR slice of edges, accumulates the attractive force into its own gradient
/// buffer, then for each active edge draws `neg_sample_rate` negatives via
/// the shared `(seed, node, epoch, edge_local, neg)` hash and accumulates
/// repulsive contributions.
///
/// Writes `node_grad[node]` in full (overwrite, not accumulate). No atomics,
/// no shared memory. Reads `epoch_of_next_sample` but never writes it; the
/// schedule update is a separate kernel.
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

/// Adam moment update + embedding step. One thread per (node, dim) pair.
/// Skips nodes with `has_update == 0` (no active edge this epoch), preserving
/// their m, v, and embedding state — matching the CPU `optimise_embedding_adam_parallel`.
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> flat (node, dim) index into the [n * n_dim] buffers
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

//////////////
// Launcher //
//////////////

pub(crate) fn launch_grad_accum<R, T>(
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
            T::from(4.0).unwrap(),   // clip_val
            T::from(0.001).unwrap(), // rep_eps
            T::from(1e-8).unwrap(),  // dist_sq_threshold
            wg,
            state.n_dim as u32,
        );
    }
}

/////////////////////////////
// CPU equivalence testers //
/////////////////////////////

#[cfg(test)]
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
    let dist_sq_threshold = T::from(1e-8).unwrap();
    let rep_eps = T::from(0.001).unwrap();
    let clip_val = T::from(4.0).unwrap();
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    // Symmetric triangle, weights (0,1)=1.0, (0,2)=0.5, (1,2)=1.0
    fn triangle_graph() -> Vec<Vec<(usize, f64)>> {
        vec![
            vec![(1, 1.0), (2, 0.5)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 0.5), (1, 1.0)],
        ]
    }

    fn triangle_setup() -> (Vec<Vec<f32>>, Vec<Vec<(usize, f32)>>, UmapOptimParams<f32>) {
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

    #[test]
    fn test_csr_triangle_layout() {
        let g = triangle_graph();
        let csr = UmapCsrGraph::<f64>::from_graph(&g).unwrap();

        assert_eq!(csr.n, 3);
        assert_eq!(csr.n_edges, 3);
        assert_eq!(csr.node_edge_offsets, vec![0, 2, 4, 6]);

        // Edge ordering: (0,1)=edge0, (0,2)=edge1, (1,2)=edge2
        assert_eq!(csr.csr_edge_idx, vec![0, 1, 0, 2, 1, 2]);
        assert_eq!(csr.csr_other_node, vec![1, 2, 0, 2, 0, 1]);
    }

    #[test]
    fn test_csr_triangle_sampling_periods() {
        let g = triangle_graph();
        let csr = UmapCsrGraph::<f64>::from_graph(&g).unwrap();

        // max_weight = 1.0; epochs_per_sample = max_weight / w
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

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

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
        // Cursors initialised to one period from time 0
        assert_eq!(
            state.epoch_of_next_sample.clone().read(&client).unwrap(),
            csr.epochs_per_sample
        );
    }

    #[test]
    fn test_per_epoch_schedules() {
        let Some(device) = try_device() else { return };
        let (client, _, params, state) = build_state(&device);

        let lr_got = state.lr_schedule.clone().read(&client).unwrap();
        let ad_got = state.ad_scale.clone().read(&client).unwrap();
        let epsc_got = state.epsc.clone().read(&client).unwrap();

        let n_epochs_f = params.n_epochs as f32;
        for e in 0..params.n_epochs {
            let lr_want = params.lr * (1.0 - e as f32 / n_epochs_f);
            assert!((lr_got[e] - lr_want).abs() < 1e-6, "lr[{}]", e);

            let t = (e + 1) as f32;
            let sqrt_b2t1 = (1.0 - params.beta2.powf(t)).sqrt();
            let ad_want = sqrt_b2t1 / (1.0 - params.beta1.powf(t));
            let epsc_want = sqrt_b2t1 * params.eps;

            assert!((ad_got[e] - ad_want).abs() < 1e-6, "ad_scale[{}]", e);
            assert!((epsc_got[e] - epsc_want).abs() < 1e-9, "epsc[{}]", e);
        }
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

    // 1. No edges have ticked yet (epoch 0, periods >= 1.0) -> gradient is zero.
    #[test]
    fn test_grad_empty_epoch() {
        let Some(device) = try_device() else { return };
        let (embd, graph, params) = triangle_setup();

        let (got, _, _) = run_grad_kernel(&device, &embd, &graph, &params, 0, 42);
        assert_eq!(got, vec![0.0_f32; embd.len() * embd[0].len()]);
    }

    // 2. Attractive only. neg_sample_rate = 0 + an epoch that activates every
    //    edge. Bit-exact equality with the CPU reference: each node accumulates
    //    serially in the same order on both sides, with no FP reordering.
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

    // 3. With negatives, matched hash. FP-tolerant: powf and FMAs may compile
    //    differently on GPU.
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

    // 4. Single hand-computed active edge. Pair of points at (0,0) and (3,0),
    //    one edge, no negatives. Verifies sign and magnitude of the attractive
    //    force without depending on the CPU reference.
    #[test]
    fn test_grad_single_edge_hand_computed() {
        let Some(device) = try_device() else { return };
        let embd = vec![vec![0.0_f32, 0.0], vec![3.0, 0.0]];
        let graph = vec![vec![(1_usize, 1.0_f32)], vec![(0, 1.0)]];
        let mut params = UmapOptimParams::<f32> {
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
        params.neg_sample_rate = 0;

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

    // 5. Determinism. Same inputs, same seed, same epoch -> bit-exact result on
    //    a second launch.
    #[test]
    fn test_grad_deterministic() {
        let Some(device) = try_device() else { return };
        let (embd, graph, params) = triangle_setup();

        let (got1, _, _) = run_grad_kernel(&device, &embd, &graph, &params, 5, 42);
        let (got2, _, _) = run_grad_kernel(&device, &embd, &graph, &params, 5, 42);
        assert_eq!(got1, got2);
    }
}
