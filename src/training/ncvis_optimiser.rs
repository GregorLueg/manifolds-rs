//! NCVis layout: Hogwild SGD on a noise-contrastive estimation objective over
//! an unweighted, symmetrised kNN graph.
//!
//! A port of the NCVis+ core (Pedersen, 2026), itself a fork of NCVis
//! (Artemenkov & Panov, 2020). The negatives of each edge are drawn up front
//! and their rows prefetched: with every thread writing to random rows, the
//! load of a negative's coordinates is what the loop waits on.

use rand::{rngs::SmallRng, RngCore, SeedableRng};
use rayon::prelude::*;

use crate::prelude::*;
use crate::training::umap_optimisers::{SplitPowLut, UmapOptimParams};
use crate::utils::prefetch::prefetch_read;

////////////
// Consts //
////////////

/// Default `min_dist` for NCVis. Far below UMAP's 0.1: NCVis+ found it raises
/// local quality at no cost.
pub const NCVIS_MIN_DIST: f64 = 0.001;

/// Default `spread` for NCVis.
pub const NCVIS_SPREAD: f64 = 1.0;

/// Floor on the squared distance before `powf`, so `d2^(b - 1)` stays finite
/// when two points coincide.
const MIN_D2: f64 = 1e-12;

/// Symmetric clamp on each coordinate shift. Keeps a near-coincident pair
/// from flinging a point across the layout.
const MAX_SHIFT: f64 = 4.0;

/// How many edges ahead the edge target's row is prefetched. Taken over from
/// NCVis+, not tuned here.
const PREFETCH_AHEAD: usize = 8;

/// Exponent of the negative-sample ramp: most negatives land in the last
/// epochs. NCVis+ found steeper ramps no better at the same cost.
const RAMP_POWER: i32 = 3;

////////////
// Params //
////////////

/// NCVis optimisation parameters.
#[derive(Clone, Debug)]
pub struct NcvisOptimParams<T> {
    /// Number of epochs. Defaults to `50`.
    pub n_epochs: usize,
    /// Mean number of negative samples per edge over the run, ramped up with
    /// the cube of progress. The time against local quality knob. Defaults to
    /// `10`.
    pub n_negative: T,
    /// Kernel parameter `a` of `q(d) = 1 / (1 + a d^(2b))`
    pub a: T,
    /// Kernel parameter `b` of `q(d) = 1 / (1 + a d^(2b))`
    pub b: T,
    /// Learning rate. Decays as `lr * (1 - (epoch / n_epochs)^2)`. Defaults
    /// to `2`.
    pub lr: T,
    /// Step size of the log normalisation constant `Q`. Defaults to `1`.
    pub lr_q: T,
}

impl<T> Default for NcvisOptimParams<T>
where
    T: ManifoldsFloat,
{
    fn default() -> Self {
        Self::from_min_dist_spread(
            T::from_f64(NCVIS_MIN_DIST).unwrap(),
            T::from_f64(NCVIS_SPREAD).unwrap(),
            None,
            None,
            None,
            None,
        )
    }
}

impl<T> NcvisOptimParams<T>
where
    T: ManifoldsFloat,
{
    /// Create NCVis optimisation parameters with the kernel given directly.
    ///
    /// ### Params
    ///
    /// * `n_epochs` - Number of epochs
    /// * `n_negative` - Mean negative samples per edge
    /// * `a` - Kernel parameter `a`
    /// * `b` - Kernel parameter `b`
    /// * `lr` - Learning rate
    /// * `lr_q` - Step size of the log normalisation constant
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new(n_epochs: usize, n_negative: T, a: T, b: T, lr: T, lr_q: T) -> Self {
        Self {
            n_epochs,
            n_negative,
            a,
            b,
            lr,
            lr_q,
        }
    }

    /// Create NCVis optimisation parameters with `(a, b)` fitted to UMAP's
    /// `min_dist` / `spread` curve.
    ///
    /// ### Params
    ///
    /// * `min_dist` - Minimum distance in the embedding. NCVis default
    ///   [`NCVIS_MIN_DIST`].
    /// * `spread` - Effective scale of the embedded points
    /// * `n_epochs` - Number of epochs. Defaults to `50`.
    /// * `n_negative` - Mean negative samples per edge. Defaults to `10`.
    /// * `lr` - Learning rate. Defaults to `2`.
    /// * `lr_q` - Step size of the log normalisation constant. Defaults to `1`.
    ///
    /// ### Returns
    ///
    /// The parameter set with `a` and `b` fitted.
    pub fn from_min_dist_spread(
        min_dist: T,
        spread: T,
        n_epochs: Option<usize>,
        n_negative: Option<T>,
        lr: Option<T>,
        lr_q: Option<T>,
    ) -> Self {
        let (a, b) = UmapOptimParams::fit_params(min_dist, spread, None);
        Self {
            n_epochs: n_epochs.unwrap_or(50),
            n_negative: n_negative.unwrap_or(T::from_f64(10.0).unwrap()),
            a,
            b,
            lr: lr.unwrap_or(T::from_f64(2.0).unwrap()),
            lr_q: lr_q.unwrap_or(T::one()),
        }
    }
}

///////////
// Graph //
///////////

/// Undirected, unweighted neighbour lists as CSR: node `i`'s neighbours are
/// `targets[offsets[i]..offsets[i + 1]]`.
pub struct NcvisGraph {
    /// Row offsets, length `n + 1`
    pub offsets: Vec<usize>,
    /// Neighbour indices, length `offsets[n]`
    pub targets: Vec<u32>,
}

/// Union of a kNN graph and its transpose, without duplicates or self loops.
///
/// The degree count and scatter run sequentially, the per-row sort and dedup
/// in parallel.
///
/// ### Params
///
/// * `knn_indices` - Neighbour indices per point, `[n][k]`
///
/// ### Returns
///
/// The symmetric CSR graph, or [`ManifoldsError::TooManyVertices`] if `n` does
/// not fit `u32`.
pub fn symmetrise_knn(knn_indices: &[Vec<usize>]) -> Result<NcvisGraph, ManifoldsError> {
    let n = knn_indices.len();
    if n > u32::MAX as usize {
        return Err(ManifoldsError::TooManyVertices { n });
    }

    let mut degree = vec![0usize; n];
    for (i, row) in knn_indices.iter().enumerate() {
        for &j in row {
            if j >= n {
                return Err(ManifoldsError::GraphSizeMismatch {
                    n_graph: j + 1,
                    n_embd: n,
                });
            }
            degree[i] += 1;
            degree[j] += 1;
        }
    }

    let mut offsets = vec![0usize; n + 1];
    for i in 0..n {
        offsets[i + 1] = offsets[i] + degree[i];
    }
    let mut targets = vec![0u32; offsets[n]];
    let mut fill = offsets[..n].to_vec();
    for (i, row) in knn_indices.iter().enumerate() {
        for &j in row {
            targets[fill[i]] = j as u32;
            fill[i] += 1;
            targets[fill[j]] = i as u32;
            fill[j] += 1;
        }
    }

    // sort, dedup and drop self loops per row, in place
    let mut rows: Vec<&mut [u32]> = Vec::with_capacity(n);
    let mut rest = targets.as_mut_slice();
    for i in 0..n {
        let (head, tail) = rest.split_at_mut(degree[i]);
        rows.push(head);
        rest = tail;
    }
    let kept: Vec<usize> = rows
        .into_par_iter()
        .enumerate()
        .map(|(i, row)| {
            row.sort_unstable();
            let mut write = 0;
            for read in 0..row.len() {
                let v = row[read];
                if v as usize != i && (write == 0 || row[write - 1] != v) {
                    row[write] = v;
                    write += 1;
                }
            }
            write
        })
        .collect();

    // compact
    let mut write = 0;
    for i in 0..n {
        let read = offsets[i];
        targets.copy_within(read..read + kept[i], write);
        offsets[i] = write;
        write += kept[i];
    }
    offsets[n] = write;
    targets.truncate(write);
    targets.shrink_to_fit();

    Ok(NcvisGraph { offsets, targets })
}

//////////////
// Schedule //
//////////////

/// Negative samples per edge for each epoch.
///
/// A cubic ramp over training progress, rescaled so its mean is `n_negative`,
/// rounded half to even (as numpy does) and floored at one per epoch.
///
/// ### Params
///
/// * `n_epochs` - Number of epochs
/// * `n_negative` - Target mean negatives per edge
///
/// ### Returns
///
/// One count per epoch.
pub fn noise_plan(n_epochs: usize, n_negative: f64) -> Vec<u32> {
    let ramp: Vec<f64> = if n_epochs > 1 {
        (0..n_epochs)
            .map(|e| (e as f64 / (n_epochs - 1) as f64).powi(RAMP_POWER))
            .collect()
    } else {
        vec![1.0; n_epochs]
    };
    let total: f64 = ramp.iter().sum();
    let scale = n_epochs as f64 * n_negative / total;
    ramp.iter()
        .map(|&r| ((r * scale).round_ties_even() as u32).max(1))
        .collect()
}

///////////////
// Optimiser //
///////////////

/// Run-wide state shared by every chunk of the NCVis kernel.
struct KernelCtx<'a, T: AtomicFloat> {
    /// Flat layout `[n * n_dim]`, written hogwild
    embd: &'a [T::Atomic],
    /// CSR row offsets of the graph
    offsets: &'a [usize],
    /// CSR neighbour indices of the graph
    targets: &'a [u32],
    /// Number of points
    n: usize,
    /// Kernel parameter `a`
    a: T,
    /// `x^b` lookup table for the kernel
    pow_lut: SplitPowLut<T>,
    /// `a == b == 1`: the Cauchy kernel, no `powf`
    cauchy: bool,
    /// `2ab`, the constant of the kernel gradient
    two_ab: T,
    /// Floor on the squared distance before `powf`
    min_d2: T,
    /// Clamp on each coordinate shift
    max_shift: T,
    /// Step size of the log normalisation constant
    lr_q: T,
}

/// Per-epoch state shared by every chunk.
struct EpochCtx<T> {
    /// Learning rate this epoch
    step: T,
    /// `1 / n_noise`
    inv_noise: T,
    /// Negatives per edge this epoch, at least one
    n_noise: usize,
    /// Log normalisation constant at the start of the epoch
    q_log: T,
}

/// One epoch of the NCVis update over a contiguous range of source points.
///
/// Inlined into both call sites so the `n_dim == 2` one sees a constant and
/// unrolls the dim loops.
///
/// ### Params
///
/// * `ctx` - Run-wide state
/// * `ep` - Per-epoch state
/// * `nodes` - Source points this chunk owns
/// * `rng` - The chunk's RNG, kept across epochs
/// * `max_noise` - Largest negative count of the run, sizes the scratch
/// * `n_dim` - Embedding dimensions
///
/// ### Returns
///
/// The chunk's running log normalisation constant at the end of the range.
#[inline(always)]
fn chunk_pass<T>(
    ctx: &KernelCtx<'_, T>,
    ep: &EpochCtx<T>,
    nodes: std::ops::Range<usize>,
    rng: &mut SmallRng,
    max_noise: usize,
    n_dim: usize,
) -> T
where
    T: ManifoldsFloat,
{
    let one = T::one();
    let two = one + one;
    let n_u64 = ctx.n as u64;
    let n_edges = ctx.targets.len();
    let n_noise = ep.n_noise;

    let mut q_local = ep.q_log;
    // others[0] is the edge target, the rest its negatives
    let mut others = vec![0u32; max_noise + 1];
    let mut y_i_buf = vec![T::zero(); n_dim];
    let mut y_o_buf = vec![T::zero(); n_dim];
    let y_i = &mut y_i_buf[..n_dim];
    let y_o = &mut y_o_buf[..n_dim];

    for i in nodes {
        let row_i = &ctx.embd[i * n_dim..(i + 1) * n_dim];
        for d in 0..n_dim {
            y_i[d] = T::load_relaxed(&row_i[d]);
        }

        for edge in ctx.offsets[i]..ctx.offsets[i + 1] {
            if edge + PREFETCH_AHEAD < n_edges {
                let ahead = ctx.targets[edge + PREFETCH_AHEAD] as usize;
                prefetch_read(&ctx.embd[ahead * n_dim]);
            }
            others[0] = ctx.targets[edge];
            for slot in others[1..=n_noise].iter_mut() {
                // multiply-shift draw in [0, n), rejecting i
                let other = loop {
                    let o = ((rng.next_u32() as u64 * n_u64) >> 32) as usize;
                    if o != i {
                        break o;
                    }
                };
                *slot = other as u32;
                prefetch_read(&ctx.embd[other * n_dim]);
            }

            for (s, &other) in others[..=n_noise].iter().enumerate() {
                let base_o = other as usize * n_dim;
                let row_o = &ctx.embd[base_o..base_o + n_dim];
                let mut d2 = T::zero();
                for d in 0..n_dim {
                    y_o[d] = T::load_relaxed(&row_o[d]);
                    let diff = y_o[d] - y_i[d];
                    d2 += diff * diff;
                }

                // kernel_grad = 2ab q d2^(b - 1)
                let (q, kernel_grad) = if ctx.cauchy {
                    let q = one / (one + d2);
                    (q, two * q)
                } else {
                    let d2_safe = d2.max(ctx.min_d2);
                    let d2_pow_b = ctx.pow_lut.get(d2_safe);
                    let q = one / (one + ctx.a * d2_pow_b);
                    (q, ctx.two_ab * q * d2_pow_b / d2_safe)
                };

                // data posterior 1 / (1 + r), noise posterior r / (1 + r): one
                // division for both
                let ratio = q * ep.inv_noise * (-q_local).exp();
                let inv = one / (one + ratio);
                let w = if s == 0 { inv } else { -ratio * inv };
                q_local -= w * ctx.lr_q;
                let w = w * kernel_grad * ep.step;

                for d in 0..n_dim {
                    let shift = ((y_o[d] - y_i[d]) * w)
                        .max(-ctx.max_shift)
                        .min(ctx.max_shift);
                    y_i[d] += shift;
                    T::store_relaxed(&row_o[d], y_o[d] - shift);
                }
            }
        }

        for d in 0..n_dim {
            T::store_relaxed(&row_i[d], y_i[d]);
        }
    }
    q_local
}

/// Optimise an NCVis layout in place.
///
/// Hogwild SGD (Recht et al., 2011) on the noise-contrastive objective of
/// NCVis. Each edge `(i, j)` draws `noise_plan[epoch]` uniform negatives; the
/// edge and its negatives each move both endpoints along their difference,
/// weighted by the posterior of being a data rather than a noise pair under
/// `q(d) = 1 / (1 + a d^(2b))` scaled by `exp(-Q)`. `Q`, the log
/// normalisation constant, is learned alongside the layout.
///
/// The nodes are split into one contiguous chunk per Rayon thread. Each chunk
/// keeps its own RNG across epochs and its own running `Q`, and `Q` is
/// averaged over the chunks at the end of each epoch, as the OpenMP original
/// does per thread. The result is therefore deterministic only on a
/// one-thread pool, and depends on the pool size.
///
/// ### Params
///
/// * `embd` - Initial layout `[n_samples][n_dim]`, modified in place
/// * `graph` - Symmetric unweighted graph from [`symmetrise_knn`]
/// * `params` - Optimisation parameters
/// * `seed` - Random seed for the negative samples
/// * `verbose` - Verbosity level
///
/// ### Returns
///
/// `Ok(())`, or [`ManifoldsError::NoData`] / [`ManifoldsError::NoGraphEdges`]
/// on an empty input.
///
/// ### References
///
/// Artemenkov & Panov, WWW, 2020 (NCVis); Recht et al., NeurIPS, 2011
/// (Hogwild!).
pub fn optimise_ncvis<T>(
    embd: &mut [Vec<T>],
    graph: &NcvisGraph,
    params: &NcvisOptimParams<T>,
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
    let n_edges = graph.targets.len();
    if n_edges == 0 {
        return Err(ManifoldsError::NoGraphEdges);
    }
    let n_dim = embd[0].len();
    let verbosity = parse_verbosity_level(verbose);

    let plan = noise_plan(params.n_epochs, params.n_negative.to_f64().unwrap());
    let max_noise = plan.iter().copied().max().unwrap_or(1) as usize;

    let one = T::one();

    let embd_shared: Vec<T::Atomic> = embd
        .iter()
        .flat_map(|p| p.iter().map(|&v| T::new_atomic(v)))
        .collect();

    let ctx = KernelCtx {
        embd: &embd_shared,
        offsets: &graph.offsets,
        targets: &graph.targets,
        n,
        a: params.a,
        pow_lut: SplitPowLut::new(params.b),
        cauchy: params.a == one && params.b == one,
        two_ab: T::from_f64(2.0).unwrap() * params.a * params.b,
        min_d2: T::from_f64(MIN_D2).unwrap(),
        max_shift: T::from_f64(MAX_SHIFT).unwrap(),
        lr_q: params.lr_q,
    };

    let n_chunks = rayon::current_num_threads().clamp(1, n);
    let chunk_len = n.div_ceil(n_chunks);
    let mut rngs: Vec<SmallRng> = (0..n_chunks)
        .map(|c| SmallRng::seed_from_u64(seed + c as u64))
        .collect();

    let mut q_log = T::zero();
    let n_epochs_t = T::from_usize(params.n_epochs).unwrap();

    for (epoch, &n_noise) in plan.iter().enumerate() {
        let progress = T::from_usize(epoch).unwrap() / n_epochs_t;
        let step = params.lr * (one - progress * progress);
        let inv_noise = one / T::from_u32(n_noise).unwrap();
        let n_noise = n_noise as usize;

        let epoch_ctx = EpochCtx {
            step,
            inv_noise,
            n_noise,
            q_log,
        };
        let q_sum: T = rngs
            .par_iter_mut()
            .enumerate()
            .map(|(c, rng)| {
                let start = c * chunk_len;
                let end = (start + chunk_len).min(n);
                // the literal 2 lets the dim loops unroll on the 2D path
                if n_dim == 2 {
                    chunk_pass(&ctx, &epoch_ctx, start..end, rng, max_noise, 2)
                } else {
                    chunk_pass(&ctx, &epoch_ctx, start..end, rng, max_noise, n_dim)
                }
            })
            .sum();
        q_log = q_sum / T::from_usize(n_chunks).unwrap();

        if verbosity.detailed_verbosity() && (epoch + 1) % 10 == 0 {
            println!(
                "  epoch {}/{} ({} negatives per edge)",
                epoch + 1,
                params.n_epochs,
                n_noise
            );
        }
    }

    for (i, point) in embd.iter_mut().enumerate() {
        for (d, v) in point.iter_mut().enumerate() {
            *v = T::load_relaxed(&embd_shared[i * n_dim + d]);
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
    use approx::assert_relative_eq;

    #[test]
    fn test_noise_plan_ramps_up_with_requested_mean() {
        let plan = noise_plan(50, 10.0);
        assert_eq!(plan.len(), 50);
        assert!(plan.iter().all(|&p| p >= 1));
        assert!(plan.windows(2).all(|w| w[0] <= w[1]));
        let mean = plan.iter().map(|&p| p as f64).sum::<f64>() / 50.0;
        assert_relative_eq!(mean, 10.0, max_relative = 0.05);
    }

    /// Reference `(a, b)` from `scipy.optimize.curve_fit` on the same curve.
    #[test]
    fn test_default_kernel_matches_scipy_fit() {
        let p = NcvisOptimParams::<f64>::default();
        assert_relative_eq!(p.a, 1.9290733953235648, max_relative = 1e-3);
        assert_relative_eq!(p.b, 0.7915045331404773, max_relative = 1e-3);
    }

    #[test]
    fn test_noise_plan_single_epoch() {
        assert_eq!(noise_plan(1, 5.0), vec![5]);
    }

    #[test]
    fn test_symmetrise_knn_is_symmetric_without_dupes_or_self() {
        let knn = vec![vec![1, 2], vec![0, 2], vec![2, 0], vec![0, 1]];
        let g = symmetrise_knn(&knn).unwrap();
        let row = |i: usize| &g.targets[g.offsets[i]..g.offsets[i + 1]];
        assert_eq!(row(0), &[1, 2, 3]);
        assert_eq!(row(1), &[0, 2, 3]);
        assert_eq!(row(2), &[0, 1]);
        assert_eq!(row(3), &[0, 1]);
        for i in 0..4 {
            for &j in row(i) {
                assert!(row(j as usize).contains(&(i as u32)));
            }
        }
    }

    #[test]
    fn test_symmetrise_knn_rejects_out_of_range() {
        let knn = vec![vec![1], vec![5]];
        assert!(symmetrise_knn(&knn).is_err());
    }

    #[test]
    fn test_optimise_ncvis_reproducible_on_one_thread() {
        let n = 200;
        let knn: Vec<Vec<usize>> = (0..n).map(|i| vec![(i + 1) % n, (i + 2) % n]).collect();
        let g = symmetrise_knn(&knn).unwrap();
        let init: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![(i as f64).sin() * 10.0, (i as f64).cos() * 10.0])
            .collect();
        let params = NcvisOptimParams::<f64>::default();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let run = || {
            let mut e = init.clone();
            pool.install(|| optimise_ncvis(&mut e, &g, &params, 42, 0))
                .unwrap();
            e
        };
        let (a, b) = (run(), run());
        assert_eq!(a, b);
        assert!(a.iter().flatten().all(|v| v.is_finite()));
    }
}
