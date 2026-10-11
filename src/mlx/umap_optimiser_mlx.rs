//! UMAP / densMAP Adam optimiser on MLX. Same maths, hash and edge schedule as
//! the cubecl version in `training::umap_optimiser_gpu`, so the two agree up to
//! float rounding.
//!
//! MLX kernels write fresh output buffers, never their inputs. That makes the
//! cubecl three-kernel epoch (gradient, Adam step, schedule advance) safe to
//! fuse into one: every thread reads the embedding of epoch `t` and writes
//! epoch `t + 1` into a separate buffer. Each edge cursor is advanced by its
//! lower-indexed endpoint, so it is written exactly once.
//!
//! The host stays at most one epoch ahead of the device: it submits epoch
//! `t` asynchronously, then waits on epoch `t - 1`. Unbounded, it would queue
//! the whole run and hold every epoch's buffers at once.

use crate::mlx::ffi::*;
use crate::prelude::*;
use crate::training::umap_csr::*;
use crate::training::umap_optimisers::UmapOptimParams;
use crate::utils::density::{correlation_stats, DENS_EPS};

////////////
// Consts //
////////////

/// Threads per threadgroup for both kernels. Matches the cubecl optimiser's
/// preferred workgroup; not tuned on MLX.
const UMAP_MLX_THREADGROUP: i32 = 256;

/// Number of entries in the per-epoch f32 parameter buffer, see
/// [`EPOCH_KERNEL_SOURCE`] for the layout.
const N_EPOCH_PARAMS: usize = 13;

/// Epochs between progress lines at normal verbosity.
const LOG_EVERY: usize = 100;

////////////////////
// Kernel sources //
////////////////////

/// Splitmix-style negative-sampling hash, bit-identical to the cubecl
/// `gpu_hash_neg` and the host `cpu_hash_neg`.
const HASH_HEADER: &str = r#"
inline uint hash_neg(uint seed, uint node, uint epoch, uint edge_local, uint neg, uint n) {
    uint h = seed
        ^ (node * 0x9E3779B1u)
        ^ (epoch * 0x85EBCA77u)
        ^ (edge_local * 0xC2B2AE3Du)
        ^ (neg * 0x27D4EB2Fu);
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    return h % n;
}
"#;

/// One full epoch, one thread per node: gradient over the node's CSR slice
/// (attraction, densMAP term, negatives), the Adam step on its coordinates and
/// the schedule advance of the edges it owns.
///
/// Template ints: `N_DIM`, `NEG_RATE`, `DENS` (0 or 1, compiles the density
/// term in or out). `up = [n, epoch, seed]`. `fp = [epoch_f, a, b, 2ab,
/// 2 gamma b, clip, rep_eps, dist_sq_threshold, lr_alpha, epsc, 1 - beta1,
/// 1 - beta2, dens_coeff]`.
///
/// Nodes with no firing edge keep their coordinates and moments, as in the
/// cubecl `has_update` skip. Negatives are gathered before any is consumed so
/// the random row loads are in flight together.
const EPOCH_KERNEL_SOURCE: &str = r#"
    uint node = thread_position_in_grid.x;
    uint n = up[0];
    if (node >= n) return;
    uint epoch = up[1];
    uint seed = up[2];

    float epoch_f = fp[0];
    float a = fp[1];
    float b = fp[2];
    float two_a_b = fp[3];
    float two_gamma_b = fp[4];
    float clip_val = fp[5];
    float rep_eps = fp[6];
    float dsq_thr = fp[7];
    float lr_alpha = fp[8];
    float epsc = fp[9];
    float omb1 = fp[10];
    float omb2 = fp[11];
    float dens_coeff = fp[12];

    uint base_self = node * N_DIM;
    float self_pos[N_DIM];
    float grad[N_DIM];
    for (int d = 0; d < N_DIM; d++) {
        self_pos[d] = embd[base_self + d];
        grad[d] = 0.0f;
    }

    constexpr int NEG_SLOTS = NEG_RATE > 0 ? NEG_RATE : 1;
    uint neg_node[NEG_SLOTS];
    float neg_pos[NEG_SLOTS * N_DIM];

    uint start = node_edge_offsets[node];
    uint end = node_edge_offsets[node + 1];
    bool active = false;
    uint edge_local = 0;
    for (uint pos = start; pos < end; pos++, edge_local++) {
        uint e = csr_edge_idx[pos];
        uint other = csr_other_node[pos];
        float next = epoch_of_next_sample[e];
        bool fires = next <= epoch_f;
        if (node < other) {
            next_out[e] = fires ? next + epochs_per_sample[e] : next;
        }
        if (!fires) continue;
        active = true;

        uint base_other = other * N_DIM;
        float delta[N_DIM];
        float dist_sq = 0.0f;
        for (int d = 0; d < N_DIM; d++) {
            delta[d] = embd[base_other + d] - self_pos[d];
            dist_sq += delta[d] * delta[d];
        }

        if (dist_sq >= dsq_thr) {
            float dist_sq_b = pow(dist_sq, b);
            float denom = 1.0f + a * dist_sq_b;
            float grad_coeff = two_a_b * dist_sq_b / (dist_sq * denom);
            for (int d = 0; d < N_DIM; d++) {
                grad[d] += 2.0f * grad_coeff * delta[d];
            }

            // densMAP, signed along (self - other): ascent on the correlation.
            // The factor 2 mirrors the attractive term, as each undirected
            // edge is visited once per endpoint.
            if (DENS) {
                float phi = 1.0f / denom;
                float dphi_term = a * b * dist_sq_b / dist_sq * phi;
                float common = 1.0f - b * (1.0f - phi);
                float dr_self = (phi / dens_phi_sum[node])
                    * (common / exp(dens_re[node]) + dphi_term);
                float dr_other = (phi / dens_phi_sum[other])
                    * (common / exp(dens_re[other]) + dphi_term);
                float cor = dens_coeff
                    * (dens_weight[node] * dr_self + dens_weight[other] * dr_other)
                    / mu[e];
                float two_cor = 2.0f * cor;
                for (int d = 0; d < N_DIM; d++) {
                    float g = clamp(two_cor * -delta[d], -clip_val, clip_val);
                    grad[d] += 2.0f * g;
                }
            }
        }

        for (int k = 0; k < NEG_RATE; k++) {
            uint kn = hash_neg(seed, node, epoch, edge_local, (uint)k, n);
            neg_node[k] = kn;
            for (int d = 0; d < N_DIM; d++) {
                neg_pos[k * N_DIM + d] = embd[kn * N_DIM + d];
            }
        }
        for (int k = 0; k < NEG_RATE; k++) {
            if (neg_node[k] == node) continue;
            float dist_sq_k = 0.0f;
            for (int d = 0; d < N_DIM; d++) {
                float diff = self_pos[d] - neg_pos[k * N_DIM + d];
                dist_sq_k += diff * diff;
            }
            float dist_sq_safe = dist_sq_k + rep_eps;
            float dist_sq_b = pow(dist_sq_safe, b);
            float denom = dist_sq_safe * (1.0f + a * dist_sq_b);
            float grad_coeff = clamp(two_gamma_b / denom, -clip_val, clip_val);
            for (int d = 0; d < N_DIM; d++) {
                grad[d] += grad_coeff * (self_pos[d] - neg_pos[k * N_DIM + d]);
            }
        }
    }

    for (int d = 0; d < N_DIM; d++) {
        uint i = base_self + d;
        float m_old = m[i];
        float v_old = v[i];
        if (active) {
            float g = grad[d];
            float m_new = m_old + omb1 * (g - m_old);
            float v_new = v_old + omb2 * (g * g - v_old);
            m_out[i] = m_new;
            v_out[i] = v_new;
            embd_out[i] = self_pos[d] + lr_alpha * m_new / (sqrt(v_new) + epsc);
        } else {
            m_out[i] = m_old;
            v_out[i] = v_old;
            embd_out[i] = self_pos[d];
        }
    }
"#;

/// densMAP embedding radii, one thread per node over its full CSR slice.
/// Writes `re = log(dens_eps + sum phi * dsq / sum phi)` and `phi_sum`.
///
/// Template int: `N_DIM`. `up = [n]`. `rp = [a, b, dens_eps, tiny]`, `tiny`
/// floors the kernel sum for nodes left isolated by edge pruning.
const RADII_KERNEL_SOURCE: &str = r#"
    uint node = thread_position_in_grid.x;
    uint n = up[0];
    if (node >= n) return;
    float a = rp[0];
    float b = rp[1];
    float dens_eps = rp[2];
    float tiny = rp[3];

    uint base_self = node * N_DIM;
    float sum_sq = 0.0f;
    float sum_phi = 0.0f;
    for (uint pos = node_edge_offsets[node]; pos < node_edge_offsets[node + 1]; pos++) {
        uint base_other = csr_other_node[pos] * N_DIM;
        float dist_sq = 0.0f;
        for (int d = 0; d < N_DIM; d++) {
            float diff = embd[base_self + d] - embd[base_other + d];
            dist_sq += diff * diff;
        }
        float phi = 1.0f / (1.0f + a * pow(dist_sq, b));
        sum_sq += phi * dist_sq;
        sum_phi += phi;
    }
    re[node] = log(dens_eps + sum_sq / max(sum_phi, tiny));
    phi_sum[node] = sum_phi;
"#;

//////////////////
// Device state //
//////////////////

/// The arrays the fused kernel reads and rewrites every epoch.
struct EpochState {
    /// Embedding `[n * n_dim]`
    embd: Array,
    /// First Adam moment `[n * n_dim]`
    m: Array,
    /// Second Adam moment `[n * n_dim]`
    v: Array,
    /// Edge sampling cursors `[n_edges]`
    epoch_of_next_sample: Array,
}

/// Arrays uploaded once and read by every epoch.
struct GraphArrays {
    /// CSR row pointers `[n + 1]`
    node_edge_offsets: Array,
    /// CSR edge indices `[2 * n_edges]`
    csr_edge_idx: Array,
    /// CSR other endpoints `[2 * n_edges]`
    csr_other_node: Array,
    /// Sampling period per edge `[n_edges]`
    epochs_per_sample: Array,
    /// Graph weight per edge `[n_edges]`, read only by the density term
    mu: Array,
}

/// densMAP arrays the fused kernel reads; `[1]` dummies when the density term
/// is off this epoch, since the `DENS = 0` kernel never touches them.
struct DensArrays {
    /// Embedding log radii `[n]`
    re: Array,
    /// Per-node kernel sums `[n]`
    phi_sum: Array,
    /// Per-node correlation sensitivities `[n]`
    weight: Array,
}

impl DensArrays {
    /// Length-one placeholders for epochs without the density term.
    ///
    /// ### Returns
    ///
    /// Three `[1]` zero arrays
    fn dummy() -> Self {
        Self {
            re: Array::from_f32(&[0.0], &[1]),
            phi_sum: Array::from_f32(&[0.0], &[1]),
            weight: Array::from_f32(&[0.0], &[1]),
        }
    }
}

/// The compiled kernels plus the run constants they are launched with.
struct Kernels {
    /// Fused epoch kernel, [`EPOCH_KERNEL_SOURCE`]
    epoch: MetalKernel,
    /// densMAP radii kernel, [`RADII_KERNEL_SOURCE`]
    radii: MetalKernel,
    /// Number of nodes
    n: usize,
    /// Embedding dimensionality
    n_dim: usize,
    /// Number of unique undirected edges
    n_edges: usize,
}

impl Kernels {
    /// Register both kernels. MLX compiles them on first use.
    ///
    /// ### Params
    ///
    /// * `n` - Number of nodes
    /// * `n_dim` - Embedding dimensionality
    /// * `n_edges` - Number of unique undirected edges
    ///
    /// ### Returns
    ///
    /// The kernel set
    fn new(n: usize, n_dim: usize, n_edges: usize) -> Self {
        let epoch = MetalKernel::new(
            "manifolds_umap_epoch",
            &[
                "embd",
                "m",
                "v",
                "epoch_of_next_sample",
                "node_edge_offsets",
                "csr_edge_idx",
                "csr_other_node",
                "epochs_per_sample",
                "mu",
                "dens_re",
                "dens_phi_sum",
                "dens_weight",
                "up",
                "fp",
            ],
            &["embd_out", "m_out", "v_out", "next_out"],
            HASH_HEADER,
            EPOCH_KERNEL_SOURCE,
        );
        let radii = MetalKernel::new(
            "manifolds_umap_dens_radii",
            &["embd", "node_edge_offsets", "csr_other_node", "up", "rp"],
            &["re", "phi_sum"],
            "",
            RADII_KERNEL_SOURCE,
        );
        Self {
            epoch,
            radii,
            n,
            n_dim,
            n_edges,
        }
    }

    /// Thread grid with one thread per node.
    ///
    /// ### Returns
    ///
    /// `(grid, threadgroup)` for [`MetalKernel::apply`]
    fn node_grid(&self) -> ([i32; 3], [i32; 3]) {
        ([self.n as i32, 1, 1], [UMAP_MLX_THREADGROUP, 1, 1])
    }

    /// Queue one fused epoch.
    ///
    /// ### Params
    ///
    /// * `st` - State of the previous epoch
    /// * `g` - Constant graph arrays
    /// * `dens` - Density arrays, or dummies when `dens_on` is false
    /// * `dens_on` - Compile in the density term
    /// * `neg_rate` - Negatives per firing edge
    /// * `up` - `[n, epoch, seed]`
    /// * `fp` - Per-epoch f32 parameters, see [`EPOCH_KERNEL_SOURCE`]
    /// * `s` - Stream to run on
    ///
    /// ### Returns
    ///
    /// The (lazy) state of the next epoch
    #[allow(clippy::too_many_arguments)]
    fn epoch(
        &self,
        st: &EpochState,
        g: &GraphArrays,
        dens: &DensArrays,
        dens_on: bool,
        neg_rate: usize,
        up: &Array,
        fp: &Array,
        s: &Stream,
    ) -> Result<EpochState, ManifoldsError> {
        let total = [(self.n * self.n_dim) as i32];
        let edges = [self.n_edges as i32];
        let (grid, tg) = self.node_grid();
        let mut out = self
            .epoch
            .apply(
                &[
                    &st.embd,
                    &st.m,
                    &st.v,
                    &st.epoch_of_next_sample,
                    &g.node_edge_offsets,
                    &g.csr_edge_idx,
                    &g.csr_other_node,
                    &g.epochs_per_sample,
                    &g.mu,
                    &dens.re,
                    &dens.phi_sum,
                    &dens.weight,
                    up,
                    fp,
                ],
                &[
                    OutputSpec {
                        shape: &total,
                        dtype: MLX_FLOAT32,
                    },
                    OutputSpec {
                        shape: &total,
                        dtype: MLX_FLOAT32,
                    },
                    OutputSpec {
                        shape: &total,
                        dtype: MLX_FLOAT32,
                    },
                    OutputSpec {
                        shape: &edges,
                        dtype: MLX_FLOAT32,
                    },
                ],
                grid,
                tg,
                &[
                    ("N_DIM", self.n_dim as i32),
                    ("NEG_RATE", neg_rate as i32),
                    ("DENS", dens_on as i32),
                ],
                s,
            )?
            .into_iter();
        let mut next = || out.next().expect("the epoch kernel has four outputs");
        Ok(EpochState {
            embd: next(),
            m: next(),
            v: next(),
            epoch_of_next_sample: next(),
        })
    }

    /// Run the radii kernel on the current embedding and wait for it.
    ///
    /// ### Params
    ///
    /// * `embd` - Current embedding `[n * n_dim]`
    /// * `g` - Constant graph arrays
    /// * `up` - `[n]`
    /// * `rp` - `[a, b, dens_eps, tiny]`
    /// * `s` - Stream to run on
    ///
    /// ### Returns
    ///
    /// Evaluated `(re, phi_sum)`, both `[n]`
    fn radii(
        &self,
        embd: &Array,
        g: &GraphArrays,
        up: &Array,
        rp: &Array,
        s: &Stream,
    ) -> Result<(Array, Array), ManifoldsError> {
        let nodes = [self.n as i32];
        let (grid, tg) = self.node_grid();
        let mut out = self
            .radii
            .apply(
                &[embd, &g.node_edge_offsets, &g.csr_other_node, up, rp],
                &[
                    OutputSpec {
                        shape: &nodes,
                        dtype: MLX_FLOAT32,
                    },
                    OutputSpec {
                        shape: &nodes,
                        dtype: MLX_FLOAT32,
                    },
                ],
                grid,
                tg,
                &[("N_DIM", self.n_dim as i32)],
                s,
            )?
            .into_iter();
        let re = out.next().expect("the radii kernel has two outputs");
        let phi_sum = out.next().expect("the radii kernel has two outputs");
        eval_all(&[&re, &phi_sum], false)?;
        Ok((re, phi_sum))
    }
}

/// Refresh the densMAP state for one epoch.
///
/// Radii on the device, then the correlation statistics on the host in `f64`
/// (summing `n` log-radii in `f32` is exactly the subtract-the-mean pattern
/// that loses its digits), then the per-node weights on the host, which is a
/// single pass over `n` values.
///
/// ### Params
///
/// * `k` - Kernel set
/// * `embd` - Current embedding
/// * `g` - Constant graph arrays
/// * `params` - UMAP parameters, for `a` and `b`
/// * `dens` - Constant density state, already in `f32`
/// * `s` - Stream to run on
///
/// ### Returns
///
/// The density arrays for the fused kernel and `dens_coeff`
fn dens_update(
    k: &Kernels,
    embd: &Array,
    g: &GraphArrays,
    params: &UmapOptimParams<f32>,
    dens: &DensState<f32>,
    s: &Stream,
) -> Result<(DensArrays, f32), ManifoldsError> {
    let up = Array::from_u32(&[k.n as u32], &[1]);
    let rp = Array::from_f32(&[params.a, params.b, DENS_EPS as f32, f32::EPSILON], &[4]);
    let (re, phi_sum) = k.radii(embd, g, &up, &rp, s)?;
    let re_host = re.as_f32()?;
    let (re_mean, re_std, cov) = correlation_stats(re_host, &dens.r, dens.params.var_shift);
    let inv_var = 1.0 / (re_std * re_std);
    let weight: Vec<f32> = dens
        .r
        .iter()
        .zip(re_host)
        .map(|(&r, &re)| r - cov * (re - re_mean) * inv_var)
        .collect();
    let weight = Array::from_f32(&weight, &[k.n as i32]);
    let denom = k.n.saturating_sub(1).max(1) as f32;
    let coeff = dens.params.lambda * dens.mu_tot / (re_std * denom);
    Ok((
        DensArrays {
            re,
            phi_sum,
            weight,
        },
        coeff,
    ))
}

//////////
// Main //
//////////

/// Run the full UMAP (or densMAP) Adam optimisation on MLX.
///
/// Device-resident throughout: graph and state are uploaded once, one fused
/// kernel runs per epoch, and the embedding is read back once at the end.
/// densMAP adds a radii kernel and a host round trip on its active epochs,
/// as on the cubecl path. Computes in `f32` whatever `T` is.
///
/// ### Params
///
/// * `embd` - Initial embedding `[n][n_dim]`, overwritten with the result
/// * `graph` - Adjacency list of the symmetrised UMAP graph. Self-loops are
///   dropped; only `i < j` entries build the CSR
/// * `params` - Adam and UMAP hyperparameters
/// * `dens` - densMAP state, or `None` for plain UMAP
/// * `seed` - Seed for negative sampling, truncated to `u32`
/// * `verbose` - `0` silent, `1` normal, `2` detailed
///
/// ### Returns
///
/// `Ok(())` with `embd` updated in place. `NoData` for an empty embedding,
/// `NoGraphEdges` if no edge survives, `MlxError` if an mlx-c call fails.
pub fn optimise_embedding_adam_mlx<T>(
    embd: &mut [Vec<T>],
    graph: &[Vec<(usize, T)>],
    params: &UmapOptimParams<T>,
    dens: Option<&DensState<T>>,
    seed: u64,
    verbose: usize,
) -> Result<(), ManifoldsError>
where
    T: ManifoldsFloat,
{
    let n = embd.len();
    if n == 0 {
        return Err(ManifoldsError::NoData);
    }
    let n_dim = embd[0].len();
    let verbosity = parse_verbosity_level(verbose);

    let params = params.cast::<f32>();
    let dens = dens.map(|d| d.cast::<f32>());
    let graph_f32: Vec<Vec<(usize, f32)>> = graph
        .iter()
        .map(|edges| {
            edges
                .iter()
                .map(|&(j, w)| (j, w.to_f32().unwrap()))
                .collect()
        })
        .collect();
    let csr = UmapCsrGraph::from_graph(&graph_f32)?;
    let embd_flat: Vec<f32> = embd
        .iter()
        .flat_map(|p| p.iter().map(|x| x.to_f32().unwrap()))
        .collect();

    install_error_handler();
    let s = Stream::default_gpu();
    let k = Kernels::new(n, n_dim, csr.n_edges);

    let total = [(n * n_dim) as i32];
    let edges = [csr.n_edges as i32];
    let csr_len = [2 * csr.n_edges as i32];
    let zeros = vec![0.0f32; n * n_dim];
    let g = GraphArrays {
        node_edge_offsets: Array::from_u32(&csr.node_edge_offsets, &[n as i32 + 1]),
        csr_edge_idx: Array::from_u32(&csr.csr_edge_idx, &csr_len),
        csr_other_node: Array::from_u32(&csr.csr_other_node, &csr_len),
        epochs_per_sample: Array::from_f32(&csr.epochs_per_sample, &edges),
        mu: Array::from_f32(&csr.mu, &edges),
    };
    // cursors start one period in, the first sample event
    let mut st = EpochState {
        embd: Array::from_f32(&embd_flat, &total),
        m: Array::from_f32(&zeros, &total),
        v: Array::from_f32(&zeros, &total),
        epoch_of_next_sample: Array::from_f32(&csr.epochs_per_sample, &edges),
    };

    let two = 2.0f32;
    let mut fp = [0.0f32; N_EPOCH_PARAMS];
    fp[1] = params.a;
    fp[2] = params.b;
    fp[3] = two * params.a * params.b;
    fp[4] = two * params.gamma * params.b;
    fp[5] = GRAD_CLIP_VAL as f32;
    fp[6] = GRAD_REP_EPS as f32;
    fp[7] = GRAD_DIST_SQ_THRESHOLD as f32;
    fp[10] = 1.0 - params.beta1;
    fp[11] = 1.0 - params.beta2;

    if verbosity.normal_verbosity() {
        println!(
            "Running {} epochs with MLX-accelerated Adam optimisation.",
            params.n_epochs
        );
    }

    for epoch in 0..params.n_epochs {
        // radii and weights before the gradient, so both see the same embedding
        let (dens_arrays, dens_coeff) = match dens.as_ref() {
            Some(d) if d.is_active(epoch, params.n_epochs) => {
                let (arrays, coeff) = dens_update(&k, &st.embd, &g, &params, d, &s)?;
                (arrays, Some(coeff))
            }
            _ => (DensArrays::dummy(), None),
        };

        let (lr_alpha, epsc) = adam_epoch_scalars(&params, epoch);
        fp[0] = epoch as f32;
        fp[8] = lr_alpha;
        fp[9] = epsc;
        fp[12] = dens_coeff.unwrap_or(0.0);
        let up = Array::from_u32(&[n as u32, epoch as u32, seed as u32], &[3]);
        let fp_arr = Array::from_f32(&fp, &[N_EPOCH_PARAMS as i32]);

        let next = k.epoch(
            &st,
            &g,
            &dens_arrays,
            dens_coeff.is_some(),
            params.neg_sample_rate,
            &up,
            &fp_arr,
            &s,
        )?;
        eval_all(
            &[&next.embd, &next.m, &next.v, &next.epoch_of_next_sample],
            true,
        )?;
        eval_all(&[&st.embd], false)?;
        st = next;

        if verbosity.normal_verbosity()
            && ((epoch + 1) % LOG_EVERY == 0 || epoch + 1 == params.n_epochs)
        {
            println!(" Epoch {} / {}", epoch + 1, params.n_epochs);
        }
    }

    eval_all(&[&st.embd], false)?;
    let out = st.embd.as_f32()?;
    for (i, point) in embd.iter_mut().enumerate() {
        for (d, coord) in point.iter_mut().enumerate() {
            *coord = T::from(out[i * n_dim + d]).unwrap();
        }
    }
    drop(st);
    drop(g);
    clear_cache()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic symmetric graph: node `i` links to `i + 1 ..= i + 4`
    /// (mod `n`) with LCG weights in `(0.05, 1]`.
    fn ring_graph(n: usize) -> Vec<Vec<(usize, f32)>> {
        let mut state = 12345u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            0.05 + 0.95 * ((state >> 33) as f32 / (1u64 << 31) as f32)
        };
        let mut g = vec![Vec::new(); n];
        for i in 0..n {
            for off in 1..=4 {
                let j = (i + off) % n;
                let w = next();
                g[i].push((j, w));
                g[j].push((i, w));
            }
        }
        g
    }

    /// Deterministic embedding `[n * 2]` spread over roughly `[-5, 5]`.
    fn spread_embd(n: usize) -> Vec<f32> {
        (0..2 * n)
            .map(|i| ((i as f32 * 0.618_034).fract() - 0.5) * 10.0)
            .collect()
    }

    fn params() -> UmapOptimParams<f32> {
        UmapOptimParams {
            a: 1.5,
            b: 0.9,
            lr: 1.0,
            gamma: 1.0,
            n_epochs: 50,
            neg_sample_rate: 5,
            min_dist: 0.1,
            beta1: 0.5,
            beta2: 0.9,
            eps: 1e-7,
        }
    }

    fn dens_setup(n: usize) -> DensState<f32> {
        let r: Vec<f32> = (0..n).map(|i| i as f32 - (n as f32 - 1.0) / 2.0).collect();
        let norm = (r.iter().map(|x| x * x).sum::<f32>() / (n as f32 - 1.0)).sqrt();
        DensState {
            params: DensParams::new(Some(2.0_f32), Some(1.0), Some(0.1)),
            r: r.iter().map(|x| x / norm).collect(),
            mu_sum: vec![2.0_f32; n],
            mu_tot: 2.0 * n as f32,
        }
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32, ctx: &str) {
        assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let d = (g - w).abs();
            assert!(
                d <= tol * (1.0 + w.abs()),
                "{ctx}: elem {i}: got {g} want {w} (diff {d})"
            );
        }
    }

    /// Host state of one epoch: embedding, moments, cursors.
    type HostState = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

    /// Density radii and weights, when the density term ran.
    type DensOut = Option<(Vec<f32>, Vec<f32>)>;

    /// One fused MLX epoch from the given host state. Also returns the
    /// density radii and weights when `dens` is set.
    fn mlx_epoch(
        csr: &UmapCsrGraph<f32>,
        st: &HostState,
        params: &UmapOptimParams<f32>,
        epoch: usize,
        seed: u32,
        dens: Option<&DensState<f32>>,
    ) -> (HostState, DensOut) {
        let n = csr.n;
        let s = Stream::default_gpu();
        let k = Kernels::new(n, 2, csr.n_edges);
        let total = [2 * n as i32];
        let edges = [csr.n_edges as i32];
        let csr_len = [2 * csr.n_edges as i32];
        let g = GraphArrays {
            node_edge_offsets: Array::from_u32(&csr.node_edge_offsets, &[n as i32 + 1]),
            csr_edge_idx: Array::from_u32(&csr.csr_edge_idx, &csr_len),
            csr_other_node: Array::from_u32(&csr.csr_other_node, &csr_len),
            epochs_per_sample: Array::from_f32(&csr.epochs_per_sample, &edges),
            mu: Array::from_f32(&csr.mu, &edges),
        };
        let state = EpochState {
            embd: Array::from_f32(&st.0, &total),
            m: Array::from_f32(&st.1, &total),
            v: Array::from_f32(&st.2, &total),
            epoch_of_next_sample: Array::from_f32(&st.3, &edges),
        };
        let (dens_arrays, coeff) = match dens {
            Some(d) => {
                let (a, c) = dens_update(&k, &state.embd, &g, params, d, &s).unwrap();
                (a, Some(c))
            }
            None => (DensArrays::dummy(), None),
        };
        let (lr_alpha, epsc) = adam_epoch_scalars(params, epoch);
        let fp = [
            epoch as f32,
            params.a,
            params.b,
            2.0 * params.a * params.b,
            2.0 * params.gamma * params.b,
            GRAD_CLIP_VAL as f32,
            GRAD_REP_EPS as f32,
            GRAD_DIST_SQ_THRESHOLD as f32,
            lr_alpha,
            epsc,
            1.0 - params.beta1,
            1.0 - params.beta2,
            coeff.unwrap_or(0.0),
        ];
        let next = k
            .epoch(
                &state,
                &g,
                &dens_arrays,
                coeff.is_some(),
                params.neg_sample_rate,
                &Array::from_u32(&[n as u32, epoch as u32, seed], &[3]),
                &Array::from_f32(&fp, &[N_EPOCH_PARAMS as i32]),
                &s,
            )
            .unwrap();
        eval_all(
            &[&next.embd, &next.m, &next.v, &next.epoch_of_next_sample],
            false,
        )
        .unwrap();
        let dens_out = coeff.map(|_| {
            (
                dens_arrays.re.as_f32().unwrap().to_vec(),
                dens_arrays.weight.as_f32().unwrap().to_vec(),
            )
        });
        (
            (
                next.embd.as_f32().unwrap().to_vec(),
                next.m.as_f32().unwrap().to_vec(),
                next.v.as_f32().unwrap().to_vec(),
                next.epoch_of_next_sample.as_f32().unwrap().to_vec(),
            ),
            dens_out,
        )
    }

    /// The same epoch on the host via the shared CPU mirrors.
    fn cpu_epoch(
        csr: &UmapCsrGraph<f32>,
        st: &HostState,
        params: &UmapOptimParams<f32>,
        epoch: usize,
        seed: u32,
        dens: Option<&CpuDensRef<f32>>,
    ) -> HostState {
        let n = csr.n;
        let epoch_f = epoch as f32;
        let grad = cpu_grad_accum(
            &st.0,
            csr,
            &st.3,
            2,
            epoch,
            seed,
            params.neg_sample_rate,
            params,
            dens,
        );
        let has_update: Vec<u32> = (0..n)
            .map(|node| {
                let s0 = csr.node_edge_offsets[node] as usize;
                let s1 = csr.node_edge_offsets[node + 1] as usize;
                (s0..s1).any(|p| st.3[csr.csr_edge_idx[p] as usize] <= epoch_f) as u32
            })
            .collect();
        let (mut embd, mut m, mut v) = (st.0.clone(), st.1.clone(), st.2.clone());
        cpu_adam_update(
            &grad,
            &has_update,
            &mut m,
            &mut v,
            &mut embd,
            n,
            2,
            params,
            epoch,
            params.n_epochs,
        );
        let next: Vec<f32> =
            st.3.iter()
                .zip(&csr.epochs_per_sample)
                .map(|(&c, &p)| if c <= epoch_f { c + p } else { c })
                .collect();
        (embd, m, v, next)
    }

    /// Non-trivial starting state: spread embedding, non-zero moments.
    fn start_state(csr: &UmapCsrGraph<f32>) -> HostState {
        let n = csr.n;
        let m: Vec<f32> = (0..2 * n).map(|i| (i as f32 * 0.37).sin() * 0.1).collect();
        let v: Vec<f32> = (0..2 * n)
            .map(|i| 0.01 + (i as f32 * 0.11).cos().abs())
            .collect();
        (spread_embd(n), m, v, csr.epochs_per_sample.clone())
    }

    #[test]
    fn test_mlx_epoch_matches_cpu_reference() {
        let csr = UmapCsrGraph::from_graph(&ring_graph(64)).unwrap();
        let params = params();
        let st = start_state(&csr);
        // epoch 3: the heavier edges fire, the light ones do not
        let (got, _) = mlx_epoch(&csr, &st, &params, 3, 42, None);
        let want = cpu_epoch(&csr, &st, &params, 3, 42, None);
        assert_close(&got.0, &want.0, 1e-4, "embd");
        assert_close(&got.1, &want.1, 1e-4, "m");
        assert_close(&got.2, &want.2, 1e-4, "v");
        assert_eq!(got.3, want.3, "edge cursors");
        assert!(got.3 != st.3, "no edge fired, the test proves nothing");
    }

    #[test]
    fn test_mlx_epoch_nothing_fires_is_identity() {
        let csr = UmapCsrGraph::from_graph(&ring_graph(32)).unwrap();
        let st = start_state(&csr);
        // every period is >= 1, so nothing is due at epoch 0
        let (got, _) = mlx_epoch(&csr, &st, &params(), 0, 7, None);
        assert_eq!(got.0, st.0);
        assert_eq!(got.1, st.1);
        assert_eq!(got.2, st.2);
        assert_eq!(got.3, st.3);
    }

    #[test]
    fn test_mlx_epoch_with_density_matches_cpu() {
        let csr = UmapCsrGraph::from_graph(&ring_graph(48)).unwrap();
        let params = params();
        let dens = dens_setup(csr.n);
        let st = start_state(&csr);

        let (got, dens_out) = mlx_epoch(&csr, &st, &params, 3, 42, Some(&dens));
        let (re_mlx, weight_mlx) = dens_out.unwrap();

        let (re, phi_sum) = cpu_dens_radii(&st.0, &csr, 2, &params);
        assert_close(&re_mlx, &re, 1e-4, "dens radii");
        let (re_mean, re_std, cov) = correlation_stats(&re, &dens.r, dens.params.var_shift);
        let inv_var = 1.0 / (re_std * re_std);
        let weight: Vec<f32> = dens
            .r
            .iter()
            .zip(&re)
            .map(|(&r, &e)| r - cov * (e - re_mean) * inv_var)
            .collect();
        assert_close(&weight_mlx, &weight, 1e-3, "dens weights");
        let cpu_dens = CpuDensRef {
            re,
            phi_sum,
            weight,
            coeff: dens.params.lambda * dens.mu_tot / (re_std * (csr.n - 1) as f32),
        };
        let want = cpu_epoch(&csr, &st, &params, 3, 42, Some(&cpu_dens));
        assert_close(&got.0, &want.0, 1e-3, "embd with density");

        // the density arm must actually change something
        let (plain, _) = mlx_epoch(&csr, &st, &params, 3, 42, None);
        let max_diff = got
            .1
            .iter()
            .zip(&plain.1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff > 1e-6, "density term had no effect ({max_diff})");
    }

    #[test]
    fn test_mlx_driver_deterministic_and_finite() {
        let graph = ring_graph(200);
        let start: Vec<Vec<f32>> = spread_embd(200).chunks(2).map(|c| c.to_vec()).collect();
        let run = || {
            let mut e = start.clone();
            optimise_embedding_adam_mlx(&mut e, &graph, &params(), None, 42, 0).unwrap();
            e
        };
        let (a, b) = (run(), run());
        assert_eq!(a, b, "same seed, different result");
        assert!(a.iter().flatten().all(|x| x.is_finite()));
        assert!(a != start, "optimiser did not move anything");
    }

    #[test]
    fn test_mlx_driver_rejects_empty() {
        let mut e: Vec<Vec<f32>> = vec![];
        let res = optimise_embedding_adam_mlx(&mut e, &[], &params(), None, 0, 0);
        assert!(matches!(res, Err(ManifoldsError::NoData)));
    }
}
