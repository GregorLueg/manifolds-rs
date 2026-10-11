//! Host-side pieces shared by the device-resident UMAP Adam optimisers (cubecl
//! and MLX): the CSR layout of the graph, the gradient constants, the per-epoch
//! Adam schedule, and CPU mirrors of the kernels for the unit tests.

use crate::prelude::*;
use crate::training::umap_optimisers::UmapOptimParams;
#[cfg(test)]
use crate::utils::density::DENS_EPS;

////////////
// Consts //
////////////

/// Symmetric clamp on the per-negative-sample repulsive gradient coefficient.
/// Prevents blow-up when a negative sample lands arbitrarily close to the
/// source point.
pub(crate) const GRAD_CLIP_VAL: f64 = 4.0;

/// Additive epsilon on the repulsive `dist_sq` to keep the denominator finite
/// when a negative sample coincides with the source point.
pub(crate) const GRAD_REP_EPS: f64 = 0.001;

/// Attractive gradients are skipped when the pairwise `dist_sq` falls below
/// this threshold, avoiding a division by zero when two endpoints have
/// collapsed onto each other.
pub(crate) const GRAD_DIST_SQ_THRESHOLD: f64 = 1e-8;

///////////////////
// Host-side CSR //
///////////////////

/// Host-side CSR representation of the symmetrised UMAP graph, ready to be
/// uploaded to the device (cubecl or MLX). Each undirected edge `(i, j, w)`
/// with `i < j` appears twice in the CSR arrays: once under node `i` and once under node `j`. Only
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
    /// Graph weight per edge `[n_edges]`. Only densMAP needs it, to undo the
    /// bias from visiting edges at a rate proportional to their weight.
    pub mu: Vec<T>,
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

        let mu: Vec<T> = edges.iter().map(|(_, _, w)| *w).collect();

        Ok(Self {
            n,
            n_edges,
            node_edge_offsets,
            csr_edge_idx,
            csr_other_node,
            epochs_per_sample,
            mu,
        })
    }
}

//////////////////////
// Adam LR schedule //
//////////////////////

/// Per-epoch Adam scalars, computed on the host.
///
/// Linear learning-rate decay `lr * (1 - epoch / n_epochs)`, fused with the
/// bias correction `sqrt(1 - beta2^t) / (1 - beta1^t)` at `t = epoch + 1`.
///
/// ### Params
///
/// * `params` - Adam and UMAP hyperparameters
/// * `epoch` - Current epoch index (0-based)
///
/// ### Returns
///
/// `(lr_alpha, epsc)`: the fused step size and the bias-corrected
/// `sqrt(1 - beta2^t) * eps`.
pub fn adam_epoch_scalars<T>(params: &UmapOptimParams<T>, epoch: usize) -> (T, T)
where
    T: ManifoldsFloat,
{
    let one = T::one();
    let lr_alpha = params.lr * (one - T::from(epoch).unwrap() / T::from(params.n_epochs).unwrap());
    let t = T::from(epoch + 1).unwrap();
    let beta1t = params.beta1.powf(t);
    let beta2t = params.beta2.powf(t);
    let sqrt_b2t1 = (one - beta2t).sqrt();
    let ad_scale = sqrt_b2t1 / (one - beta1t);
    (lr_alpha * ad_scale, sqrt_b2t1 * params.eps)
}

/////////////////////////////
// CPU equivalence testers //
/////////////////////////////

/// Splitmix-style host-side hash matching the device negative-sampling hash
/// (cubecl `gpu_hash_neg`, MLX `hash_neg`) bit-for-bit. Kept in sync manually
/// so the CPU reference reproduces the same negative samples.
#[cfg(test)]
#[inline]
pub(crate) fn cpu_hash_neg(
    seed: u32,
    node: u32,
    epoch: u32,
    edge_local: u32,
    neg: u32,
    n: u32,
) -> u32 {
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

/// Host-side mirror of the densMAP state the gradient kernel reads.
#[cfg(test)]
pub(crate) struct CpuDensRef<T> {
    /// Embedding log radii `[n]`, from `cpu_dens_radii`.
    pub(crate) re: Vec<T>,
    /// Per-node kernel sums `[n]`, from `cpu_dens_radii`.
    pub(crate) phi_sum: Vec<T>,
    /// Per-node correlation sensitivities `[n]`.
    pub(crate) weight: Vec<T>,
    /// `lambda * mu_tot / (re_std * (n - 1))`.
    pub(crate) coeff: T,
}

/// Host-side reference implementation of `umap_dens_radii`. Same iteration
/// order as the kernel.
#[cfg(test)]
pub(crate) fn cpu_dens_radii<T>(
    embd_flat: &[T],
    csr: &UmapCsrGraph<T>,
    n_dim: usize,
    params: &UmapOptimParams<T>,
) -> (Vec<T>, Vec<T>)
where
    T: ManifoldsFloat,
{
    let n = csr.n;
    let mut re = vec![T::zero(); n];
    let mut phi_sum = vec![T::zero(); n];
    let dens_eps = T::from(DENS_EPS).unwrap();

    for node in 0..n {
        let base_self = node * n_dim;
        let start = csr.node_edge_offsets[node] as usize;
        let end = csr.node_edge_offsets[node + 1] as usize;

        let mut sum_sq = T::zero();
        let mut sum_phi = T::zero();

        for pos in start..end {
            let other = csr.csr_other_node[pos] as usize;
            let base_other = other * n_dim;

            let mut dist_sq = T::zero();
            for d in 0..n_dim {
                let diff = embd_flat[base_self + d] - embd_flat[base_other + d];
                dist_sq += diff * diff;
            }

            let dist_sq_b = dist_sq.powf(params.b);
            let phi = T::one() / (T::one() + params.a * dist_sq_b);

            sum_sq += phi * dist_sq;
            sum_phi += phi;
        }

        re[node] = (dens_eps + sum_sq / sum_phi.max(T::epsilon())).ln();
        phi_sum[node] = sum_phi;
    }

    (re, phi_sum)
}

/// Host-side reference implementation of `umap_grad_accum`. Same iteration
/// order as the kernel so attractive-only runs match bit-for-bit; runs with
/// negatives are FP-tolerant because `powf` and FMAs may compile differently
/// on the GPU.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn cpu_grad_accum<T>(
    embd_flat: &[T],
    csr: &UmapCsrGraph<T>,
    epoch_of_next_sample: &[T],
    n_dim: usize,
    epoch: usize,
    seed: u32,
    neg_sample_rate: usize,
    params: &UmapOptimParams<T>,
    dens: Option<&CpuDensRef<T>>,
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

                if let Some(dens) = dens {
                    let phi = one / denom;
                    let dphi_term = params.a * params.b * dist_sq_b / dist_sq * phi;
                    let common = one - params.b * (one - phi);

                    let dr_self =
                        (phi / dens.phi_sum[node]) * (common / dens.re[node].exp() + dphi_term);
                    let dr_other =
                        (phi / dens.phi_sum[other]) * (common / dens.re[other].exp() + dphi_term);

                    let cor = dens.coeff
                        * (dens.weight[node] * dr_self + dens.weight[other] * dr_other)
                        / csr.mu[edge_idx];
                    let two_cor = two * cor;

                    for d in 0..n_dim {
                        let delta = embd_flat[base_self + d] - embd_flat[base_other + d];
                        let g = (two_cor * delta).max(-clip_val).min(clip_val);
                        grad[base_self + d] += two * g;
                    }
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
pub(crate) fn cpu_adam_update<T>(
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
