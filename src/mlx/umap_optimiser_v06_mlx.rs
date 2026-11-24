//! MLX port of the umap-learn 0.6 Adam ([`crate::training::umap_optimiser_v06`]).
//!
//! Same kernels as the cubecl port in `training::umap_optimiser_v06_gpu`: a
//! per-node gradient walk over the node's CSR row and a per-coordinate Adam
//! step. MLX outputs are fresh buffers, so the gradient kernel writes the
//! row's advanced sampling clocks out rather than updating them in place;
//! every CSR slot belongs to exactly one row, so the clocks are rewritten in
//! full. The negative order, the hard-negative projection order and the
//! force-ratio adaptation run on the host, as on the cubecl path. `f32` only.

use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};

use crate::mlx::ffi::*;
use crate::mlx::umap_optimiser_mlx::{launch_1d, LOG_EVERY};
use crate::prelude::*;
use crate::training::umap_csr::UmapCsrGraph;
use crate::training::umap_optimiser_v06::*;
use crate::training::umap_optimisers::{SplitPowLut, UmapOptimParams};

////////////
// Consts //
////////////

/// Number of entries in the per-epoch f32 parameter buffer, see
/// [`V06_GRAD_SOURCE`] for the layout.
const N_V06_PARAMS: usize = 14;

// The hash modulus is the window or the node count, never above the default
// window, so the 10-bit limb split in [`V06_MULMOD_HEADER`] cannot overflow.
const _: () = assert!(V06_DEFAULT_NEG_SELECTION_RANGE < 1 << 20);

////////////////////
// Kernel sources //
////////////////////

/// `(a * b) mod m` in 32-bit arithmetic for `a, b < m < 2^20`, as the cubecl
/// `v06_mulmod`. Apple GPUs have no native 64-bit integer division, so a
/// `ulong` product and modulo here costs more than the rest of the gradient.
const V06_MULMOD_HEADER: &str = r#"
inline uint v06_mulmod(uint a, uint b, uint m) {
    uint hi = (a * (b >> 10)) % m;
    return ((hi << 10) + a * (b & 1023u)) % m;
}
"#;

/// Gradient of the 0.6 Adam, one thread per node over its CSR row.
///
/// Every entry whose clock has come due pulls the node towards its neighbour
/// and draws its due negatives,
/// `order[(win_start + (p * (epoch + q + 1)) mod modulus) mod n]` for CSR
/// position `p` and draw `q`, each repelling with the tanh soft clip. Writes
/// the node's gradient and the row's clocks in full.
///
/// Template ints: `N_DIM`, `WINDOWED` (0 or 1, negatives from a window
/// around the node's rank). `up = [n, epoch, range, modulus]`. `fp =
/// [epoch_f, a, b, attr_coeff, two_gamma_b, gamma, clip, rep_eps, neg_scale,
/// step, inv_bc2, beta1, beta2, eps]`, shared with the Adam kernel.
///
const V06_GRAD_SOURCE: &str = r#"
    uint node = thread_position_in_grid.x;
    uint n = up[0];
    if (node >= n) return;
    uint epoch = up[1];
    uint range = up[2];
    uint modulus = up[3];

    float epoch_f = fp[0];
    float a = fp[1];
    float b = fp[2];
    float attr_coeff = fp[3];
    float two_gamma_b = fp[4];
    float gamma = fp[5];
    float clip_val = fp[6];
    float rep_eps = fp[7];
    float neg_scale = fp[8];

    uint base_self = node * N_DIM;
    float cur[N_DIM];
    float g[N_DIM];
    for (int d = 0; d < N_DIM; d++) {
        cur[d] = embd[base_self + d];
        g[d] = 0.0f;
    }

    uint win_start = 0;
    if (WINDOWED) {
        uint center = ranks[node];
        uint half_range = range / 2;
        if (center > half_range) win_start = center - half_range;
        if (win_start + range > n) win_start = n - range;
    }

    uint start = offsets[node];
    uint end = offsets[node + 1];
    for (uint p = start; p < end; p++) {
        float ns = next_sample[p];
        float nn = next_neg[p];
        if (ns > epoch_f) {
            next_sample_out[p] = ns;
            next_neg_out[p] = nn;
            continue;
        }
        uint e = edge_idx[p];
        uint base_j = other[p] * N_DIM;

        float dsq = 0.0f;
        for (int d = 0; d < N_DIM; d++) {
            float diff = cur[d] - embd[base_j + d];
            dsq += diff * diff;
        }
        if (dsq > 0.0f) {
            float dsq_b = pow(dsq, b);
            float coeff = -attr_coeff * dsq_b / (dsq * (a * dsq_b + 1.0f));
            for (int d = 0; d < N_DIM; d++) {
                g[d] += coeff * (cur[d] - embd[base_j + d]);
            }
        }
        next_sample_out[p] = ns + epochs_per_sample[e];

        float epn = epochs_per_neg[e];
        float n_neg_f = floor((epoch_f - nn) / epn);
        uint n_neg = n_neg_f > 0.0f ? (uint)n_neg_f : 0u;
        uint p_mod = p % modulus;
        for (uint q = 0; q < n_neg; q++) {
            uint hash = v06_mulmod(p_mod, (epoch + q + 1) % modulus, modulus);
            uint base_k = order[(win_start + hash) % n] * N_DIM;
            float dsq_k = 0.0f;
            for (int d = 0; d < N_DIM; d++) {
                float diff = cur[d] - embd[base_k + d];
                dsq_k += diff * diff;
            }
            if (dsq_k > 0.0f) {
                float coeff = two_gamma_b / ((rep_eps + dsq_k) * (a * pow(dsq_k, b) + 1.0f));
                if (coeff > 0.0f) {
                    float norm = coeff * sqrt(dsq_k);
                    float c = neg_scale * coeff * clip_val * tanh(norm / gamma) / norm;
                    for (int d = 0; d < N_DIM; d++) {
                        g[d] += c * (cur[d] - embd[base_k + d]);
                    }
                }
            }
        }
        next_neg_out[p] = nn + (float)n_neg * epn;
    }

    for (int d = 0; d < N_DIM; d++) {
        grad[base_self + d] = g[d];
    }
"#;

/// Adam step of the 0.6 optimiser, one thread per coordinate. Coordinates
/// with a zero gradient keep their moments and position, as the reference.
///
/// `up` and `fp` as in [`V06_GRAD_SOURCE`].
const V06_ADAM_SOURCE: &str = r#"
    uint i = thread_position_in_grid.x;
    if (i >= up[0] * N_DIM) return;
    float step = fp[9];
    float inv_bc2 = fp[10];
    float beta1 = fp[11];
    float beta2 = fp[12];
    float eps = fp[13];
    float gi = grad[i];
    float m_old = m[i];
    float v_old = v[i];
    float e_old = embd[i];
    if (gi != 0.0f) {
        float mi = beta1 * m_old + (1.0f - beta1) * gi;
        float vi = beta2 * v_old + (1.0f - beta2) * gi * gi;
        m_out[i] = mi;
        v_out[i] = vi;
        embd_out[i] = e_old + step * mi / (sqrt(vi * inv_bc2) + eps);
    } else {
        m_out[i] = m_old;
        v_out[i] = v_old;
        embd_out[i] = e_old;
    }
"#;

//////////////////
// Device state //
//////////////////

/// The arrays every epoch reads and rewrites.
struct V06EpochState {
    /// Embedding `[n * n_dim]`
    embd: Array,
    /// First Adam moment `[n * n_dim]`
    m: Array,
    /// Second Adam moment `[n * n_dim]`
    v: Array,
    /// Next sample epoch per CSR entry `[2 * n_edges]`
    next_sample: Array,
    /// Negatives drawn up to this epoch, per CSR entry `[2 * n_edges]`
    next_neg: Array,
}

/// Arrays uploaded once and read by every epoch.
struct V06GraphArrays {
    /// CSR row pointers `[n + 1]`
    offsets: Array,
    /// Edge index per CSR entry `[2 * n_edges]`
    edge_idx: Array,
    /// Other endpoint per CSR entry `[2 * n_edges]`
    other: Array,
    /// Sampling period per edge `[n_edges]`
    epochs_per_sample: Array,
    /// Negative-sampling period per edge `[n_edges]`
    epochs_per_neg: Array,
}

/// The two compiled kernels plus the sizes they are launched with.
struct V06Kernels {
    /// [`V06_GRAD_SOURCE`]
    grad: MetalKernel,
    /// [`V06_ADAM_SOURCE`]
    adam: MetalKernel,
    /// Number of nodes
    n: usize,
    /// Embedding dimension
    n_dim: usize,
    /// Number of CSR entries, `2 * n_edges`
    n_entries: usize,
}

impl V06Kernels {
    /// Register both kernels. MLX compiles each on first use.
    ///
    /// ### Params
    ///
    /// * `n` - Number of nodes
    /// * `n_dim` - Embedding dimension
    /// * `n_entries` - Number of CSR entries
    ///
    /// ### Returns
    ///
    /// The kernel set
    fn new(n: usize, n_dim: usize, n_entries: usize) -> Self {
        let grad = MetalKernel::new(
            "manifolds_umap_v06_grad",
            &[
                "embd",
                "offsets",
                "edge_idx",
                "other",
                "epochs_per_sample",
                "epochs_per_neg",
                "next_sample",
                "next_neg",
                "order",
                "ranks",
                "up",
                "fp",
            ],
            &["grad", "next_sample_out", "next_neg_out"],
            V06_MULMOD_HEADER,
            V06_GRAD_SOURCE,
        );
        let adam = MetalKernel::new(
            "manifolds_umap_v06_adam",
            &["grad", "m", "v", "embd", "up", "fp"],
            &["embd_out", "m_out", "v_out"],
            "",
            V06_ADAM_SOURCE,
        );
        Self {
            grad,
            adam,
            n,
            n_dim,
            n_entries,
        }
    }

    /// Queue one epoch: gradient, then the Adam step.
    ///
    /// ### Params
    ///
    /// * `st` - State of the previous epoch
    /// * `g` - Constant graph arrays
    /// * `order` - Negative order `[n]`
    /// * `ranks` - Rank of each node in `order` `[n]`, or `[1]` when not
    ///   `windowed`
    /// * `windowed` - Draw negatives from a window around the node's rank
    /// * `up` - `[n, epoch, range, modulus]`
    /// * `fp` - Per-epoch f32 parameters, see [`V06_GRAD_SOURCE`]
    /// * `s` - Stream to run on
    ///
    /// ### Returns
    ///
    /// The (lazy) state of the next epoch
    #[allow(clippy::too_many_arguments)]
    fn epoch(
        &self,
        st: &V06EpochState,
        g: &V06GraphArrays,
        order: &Array,
        ranks: &Array,
        windowed: bool,
        up: &Array,
        fp: &Array,
        s: &Stream,
    ) -> Result<V06EpochState, ManifoldsError> {
        let total = [(self.n * self.n_dim) as i32];
        let entries = [self.n_entries as i32];
        let f32_out = |shape| OutputSpec {
            shape,
            dtype: MLX_FLOAT32,
        };
        let n_dim = ("N_DIM", self.n_dim as i32);

        let (grid, tg) = launch_1d(self.n);
        let mut g_out = self
            .grad
            .apply(
                &[
                    &st.embd,
                    &g.offsets,
                    &g.edge_idx,
                    &g.other,
                    &g.epochs_per_sample,
                    &g.epochs_per_neg,
                    &st.next_sample,
                    &st.next_neg,
                    order,
                    ranks,
                    up,
                    fp,
                ],
                &[f32_out(&total), f32_out(&entries), f32_out(&entries)],
                grid,
                tg,
                &[n_dim, ("WINDOWED", windowed as i32)],
                s,
            )?
            .into_iter();
        let mut next_g = || g_out.next().expect("the gradient kernel has three outputs");
        let (grad, next_sample, next_neg) = (next_g(), next_g(), next_g());

        let (grid, tg) = launch_1d(self.n * self.n_dim);
        let mut a_out = self
            .adam
            .apply(
                &[&grad, &st.m, &st.v, &st.embd, up, fp],
                &[f32_out(&total), f32_out(&total), f32_out(&total)],
                grid,
                tg,
                &[n_dim],
                s,
            )?
            .into_iter();
        let mut next_a = || a_out.next().expect("the Adam kernel has three outputs");
        Ok(V06EpochState {
            embd: next_a(),
            m: next_a(),
            v: next_a(),
            next_sample,
            next_neg,
        })
    }
}

//////////
// Main //
//////////

/// Run the 0.6 Adam layout on MLX.
///
/// Same schedule, negatives and adaptation as [`optimise_layout_v06`]; a seed
/// draws the same orders as the CPU run. Differs only in float rounding
/// (device `pow` against the CPU's lookup table). With hard negatives the
/// embedding comes back to the host every epoch for the projection order, so
/// each epoch blocks; otherwise the host runs one epoch ahead of the device.
///
/// ### Params
///
/// * `embd` - Flat row-major embedding `[n * n_dim]`, modified in place
/// * `n_dim` - Embedding dimension
/// * `csr` - Graph in CSR form
/// * `cfg` - Layout configuration
/// * `seed` - Seed for node orders, projections and the scale adaptation
/// * `verbose` - `0` silent, `1` or `2` print the epoch every 50 epochs
///
/// ### Returns
///
/// `Ok(())`, `ManifoldsError::GraphSizeMismatch` if the graph and the
/// embedding disagree, or `MlxError` if an mlx-c call fails.
fn optimise_layout_v06_mlx(
    embd: &mut [f32],
    n_dim: usize,
    csr: &UmapCsrGraph<f32>,
    cfg: &V06Config<f32>,
    seed: u64,
    verbose: usize,
) -> Result<(), ManifoldsError> {
    let verbosity = parse_verbosity_level(verbose);
    let n = csr.n;
    if embd.len() != n * n_dim {
        return Err(ManifoldsError::GraphSizeMismatch {
            n_graph: n,
            n_embd: embd.len() / n_dim.max(1),
        });
    }

    let small = n <= V06_SMALL_LAYOUT_MAX;
    let hard_negatives = cfg.neg_selection_range < n;
    let windowed = hard_negatives && !small;
    let range = n.min(V06_MIN_NEG_SELECTION_RANGE.max(cfg.neg_selection_range));
    let modulus = if small { n } else { range };
    let mut neg_scale = HardNegScale::new(cfg.neg_selection_range, n);
    let lut = SplitPowLut::new(cfg.b);
    let attr_scale = if small {
        V06_SMALL_ATTRACTION_SCALE as f32
    } else {
        1.0
    };

    let n_entries = csr.csr_edge_idx.len();
    let neg_period_scale = V06_NEG_PERIOD_SCALE as f32 / cfg.neg_sample_rate.max(1) as f32;
    let epochs_per_neg: Vec<f32> = csr
        .epochs_per_sample
        .iter()
        .map(|&e| e * neg_period_scale)
        .collect();
    let next_sample: Vec<f32> = csr
        .csr_edge_idx
        .iter()
        .map(|&e| csr.epochs_per_sample[e as usize])
        .collect();
    let next_neg: Vec<f32> = csr
        .csr_edge_idx
        .iter()
        .map(|&e| epochs_per_neg[e as usize])
        .collect();

    install_error_handler();
    let s = Stream::default_gpu();
    let k = V06Kernels::new(n, n_dim, n_entries);

    let total = [(n * n_dim) as i32];
    let entries = [n_entries as i32];
    let edges = [csr.n_edges as i32];
    let zeros = vec![0.0f32; n * n_dim];
    let g = V06GraphArrays {
        offsets: Array::from_u32(&csr.node_edge_offsets, &[n as i32 + 1]),
        edge_idx: Array::from_u32(&csr.csr_edge_idx, &entries),
        other: Array::from_u32(&csr.csr_other_node, &entries),
        epochs_per_sample: Array::from_f32(&csr.epochs_per_sample, &edges),
        epochs_per_neg: Array::from_f32(&epochs_per_neg, &edges),
    };
    let mut st = V06EpochState {
        embd: Array::from_f32(embd, &total),
        m: Array::from_f32(&zeros, &total),
        v: Array::from_f32(&zeros, &total),
        next_sample: Array::from_f32(&next_sample, &entries),
        next_neg: Array::from_f32(&next_neg, &entries),
    };

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut adapt_rng = SmallRng::seed_from_u64(seed.wrapping_add(0x9e37_79b9_7f4a_7c15));
    let (mut order, mut ranks): (Vec<u32>, Vec<u32>) = if windowed {
        projection_order(embd, n_dim, &mut rng)
    } else {
        ((0..n as u32).collect(), vec![0u32; 1])
    };
    let mut order_arr = Array::from_u32(&order, &[n as i32]);
    let mut ranks_arr = Array::from_u32(&ranks, &[ranks.len() as i32]);

    let alphas = alpha_schedule(cfg.n_epochs, cfg.lr as f64, cfg.good_init);
    let mut fp = [0.0f32; N_V06_PARAMS];
    fp[1] = cfg.a;
    fp[2] = cfg.b;
    fp[3] = attr_scale * 2.0 * cfg.a * cfg.b;
    fp[4] = 2.0 * cfg.gamma * cfg.b;
    fp[5] = cfg.gamma;
    fp[6] = cfg.gamma.min(V06_REP_CLIP_MAX as f32);
    fp[7] = V06_REP_EPS as f32;
    fp[11] = V06_BETA1 as f32;
    fp[12] = V06_BETA2 as f32;
    fp[13] = V06_ADAM_EPS as f32;

    for epoch in 0..cfg.n_epochs {
        if windowed && HardNegScale::due(epoch, cfg.n_epochs) {
            neg_scale.update(negative_force_ratio(
                embd,
                n_dim,
                &order,
                cfg.neg_selection_range.max(1),
                cfg,
                &lut,
                &mut adapt_rng,
            ));
        }

        let t = (epoch + 1) as i32;
        let bc1 = 1.0 - V06_BETA1.powi(t);
        let bc2 = 1.0 - V06_BETA2.powi(t);
        fp[0] = epoch as f32;
        fp[8] = if small { 1.0 } else { neg_scale.scale as f32 };
        fp[9] = (alphas[epoch] / bc1) as f32;
        fp[10] = (1.0 / bc2) as f32;
        let up = Array::from_u32(
            &[n as u32, epoch as u32, range as u32, modulus as u32],
            &[4],
        );
        let fp_arr = Array::from_f32(&fp, &[N_V06_PARAMS as i32]);

        let next = k.epoch(&st, &g, &order_arr, &ranks_arr, windowed, &up, &fp_arr, &s)?;
        let state = [
            &next.embd,
            &next.m,
            &next.v,
            &next.next_sample,
            &next.next_neg,
        ];
        if hard_negatives {
            eval_all(&state, false)?;
            embd.copy_from_slice(next.embd.as_f32()?);
            (order, ranks) = projection_order(embd, n_dim, &mut rng);
            ranks_arr = Array::from_u32(&ranks, &[n as i32]);
        } else {
            eval_all(&state, true)?;
            eval_all(&[&st.embd], false)?;
            order.shuffle(&mut rng);
        }
        order_arr = Array::from_u32(&order, &[n as i32]);
        st = next;

        if verbosity.normal_verbosity()
            && ((epoch + 1) % LOG_EVERY == 0 || epoch + 1 == cfg.n_epochs)
        {
            println!("  Epoch {}/{}", epoch + 1, cfg.n_epochs);
        }
    }

    eval_all(&[&st.embd], false)?;
    embd.copy_from_slice(st.embd.as_f32()?);
    drop(st);
    drop(g);
    clear_cache()
}

/// Optimise a UMAP embedding with the umap-learn 0.6 Adam on MLX.
///
/// Rescales the initialisation with `rescale_isotropic` on the host, then
/// runs the layout on the device. Computes in `f32` whatever `T` is. `beta1`,
/// `beta2` and `eps` in `params` are ignored; the 0.6 constants are fixed.
///
/// ### Params
///
/// * `embd` - Initial embedding `[n][n_dim]`, modified in place
/// * `graph` - Symmetric adjacency list
/// * `params` - Curve, learning rate, repulsion, epochs and negative rate
/// * `good_init` - Use the short warm-up meant for a good initialisation
/// * `seed` - Random seed
/// * `verbose` - `0` silent, `1` or `2` print the epoch every 50 epochs
///
/// ### Returns
///
/// `Ok(())`, `NoData` for an empty embedding, the graph construction error,
/// or `MlxError` if an mlx-c call fails.
pub fn optimise_embedding_adam_v06_mlx<T>(
    embd: &mut [Vec<T>],
    graph: &[Vec<(usize, T)>],
    params: &UmapOptimParams<T>,
    good_init: bool,
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
    let params = params.cast::<f32>();
    let mut flat: Vec<f32> = embd
        .iter()
        .flat_map(|p| p.iter().map(|x| x.to_f32().unwrap()))
        .collect();
    rescale_isotropic(&mut flat, n_dim, params.gamma as f64);

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
    let cfg = V06Config {
        a: params.a,
        b: params.b,
        gamma: params.gamma,
        lr: params.lr,
        n_epochs: params.n_epochs,
        neg_sample_rate: params.neg_sample_rate,
        neg_selection_range: V06_DEFAULT_NEG_SELECTION_RANGE,
        good_init,
    };
    optimise_layout_v06_mlx(&mut flat, n_dim, &csr, &cfg, seed, verbose)?;

    for (point, chunk) in embd.iter_mut().zip(flat.chunks_exact(n_dim)) {
        for (coord, &x) in point.iter_mut().zip(chunk) {
            *coord = T::from(x).unwrap();
        }
    }
    Ok(())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    /// Two rings of `n_per` nodes, each node linked to its `k` ring
    /// neighbours on either side.
    fn two_rings(n_per: usize, k: usize) -> Vec<Vec<(usize, f32)>> {
        let mut graph = vec![Vec::new(); 2 * n_per];
        for c in 0..2 {
            for i in 0..n_per {
                for o in 1..=k {
                    let a = c * n_per + i;
                    let b = c * n_per + (i + o) % n_per;
                    graph[a].push((b, 1.0));
                    graph[b].push((a, 1.0));
                }
            }
        }
        graph
    }

    /// Run CPU and MLX from the same start and seed, return the largest
    /// coordinate gap relative to the embedding's span.
    fn cpu_mlx_gap(n_per: usize, range: usize, n_epochs: usize) -> f32 {
        let graph = two_rings(n_per, 5);
        let csr = UmapCsrGraph::from_graph(&graph).unwrap();
        let p = UmapOptimParams::<f32>::default_2d();
        let cfg = V06Config {
            a: p.a,
            b: p.b,
            gamma: 1.0,
            lr: 1.0,
            n_epochs,
            neg_sample_rate: 5,
            neg_selection_range: range,
            good_init: false,
        };
        let mut rng = SmallRng::seed_from_u64(3);
        let mut cpu: Vec<f32> = (0..4 * n_per)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        rescale_isotropic(&mut cpu, 2, 1.0);
        let mut mlx = cpu.clone();

        optimise_layout_v06(&mut cpu, 2, &csr, &cfg, 11, 0).unwrap();
        optimise_layout_v06_mlx(&mut mlx, 2, &csr, &cfg, 11, 0).unwrap();

        assert!(mlx.iter().all(|x| x.is_finite()));
        let span = cpu.iter().fold(0.0f32, |s, x| s.max(x.abs()));
        let gap = cpu
            .iter()
            .zip(&mlx)
            .fold(0.0f32, |s, (a, b)| s.max((a - b).abs()));
        gap / span
    }

    #[test]
    fn test_v06_mlx_matches_cpu_small_layout() {
        let gap = cpu_mlx_gap(200, V06_DEFAULT_NEG_SELECTION_RANGE, 5);
        assert!(gap < 1e-3, "relative gap {gap}");
    }

    #[test]
    fn test_v06_mlx_matches_cpu_uniform_negatives() {
        let gap = cpu_mlx_gap(1500, V06_DEFAULT_NEG_SELECTION_RANGE, 5);
        assert!(gap < 1e-3, "relative gap {gap}");
    }

    #[test]
    fn test_v06_mlx_matches_cpu_hard_negatives() {
        // a few epochs only: rounding eventually swaps two near-tied ranks in
        // the projection order, after which the two draw different negatives
        let gap = cpu_mlx_gap(1500, 500, 3);
        assert!(gap < 1e-3, "relative gap {gap}");
    }

    #[test]
    fn test_v06_mlx_driver_deterministic_and_finite() {
        let graph = two_rings(150, 4);
        let start: Vec<Vec<f32>> = (0..300)
            .map(|i| {
                vec![
                    (i as f32 * 0.618_034).fract() - 0.5,
                    (i as f32 * 0.37).sin(),
                ]
            })
            .collect();
        let mut p = UmapOptimParams::<f32>::default_2d();
        p.n_epochs = 50;
        let run = || {
            let mut e = start.clone();
            optimise_embedding_adam_v06_mlx(&mut e, &graph, &p, false, 42, 0).unwrap();
            e
        };
        let (a, b) = (run(), run());
        assert_eq!(a, b, "same seed, different result");
        assert!(a.iter().flatten().all(|x| x.is_finite()));
        assert!(a != start, "optimiser did not move anything");
    }

    #[test]
    fn test_v06_mlx_driver_rejects_empty() {
        let mut e: Vec<Vec<f32>> = vec![];
        let p = UmapOptimParams::<f32>::default_2d();
        let res = optimise_embedding_adam_v06_mlx(&mut e, &[], &p, false, 0, 0);
        assert!(matches!(res, Err(ManifoldsError::NoData)));
    }
}
