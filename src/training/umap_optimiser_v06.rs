//! Port of the umap-learn 0.6 Adam layout optimiser (`optimizer="adam"` with
//! `compatibility_layout=False`).
//!
//! Differs from the uwot-style Adam in [`crate::training::umap_optimisers`]:
//! textbook Adam constants, a warm-up learning-rate schedule, a tanh soft clip
//! on repulsion, negatives drawn from both endpoints by a deterministic hash
//! into a per-epoch node order, and "hard negatives" on large layouts, i.e.
//! negatives drawn from a window around each node in a random 1-D projection
//! of the embedding. Every epoch is a Jacobi step: gradients are accumulated
//! against the frozen embedding, then all nodes take their Adam step.

use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;

use crate::prelude::*;
use crate::training::umap_csr::UmapCsrGraph;
use crate::training::umap_optimisers::{SplitPowLut, UmapOptimParams};

///////////////
// Constants //
///////////////

/// Adam first-moment decay of the 0.6 optimiser.
pub(crate) const V06_BETA1: f64 = 0.9;

/// Adam second-moment decay of the 0.6 optimiser.
pub(crate) const V06_BETA2: f64 = 0.99;

/// Adam denominator epsilon of the 0.6 optimiser. Large next to textbook
/// Adam, which damps the steps of nodes whose gradients are tiny and noisy.
pub(crate) const V06_ADAM_EPS: f64 = 1e-4;

/// Stretch on the negative-sampling period, so the 0.6 optimiser draws
/// `1 / 1.5` as many negatives per positive as the nominal rate. Both
/// endpoints of an edge draw them, so the total is still above the classic
/// single-endpoint scheme.
pub(crate) const V06_NEG_PERIOD_SCALE: f64 = 1.5;

/// Additive epsilon on the repulsive squared distance.
pub(crate) const V06_REP_EPS: f64 = 0.001;

/// Ceiling on the soft clip of a single repulsive gradient's norm, applied as
/// `min(gamma, V06_REP_CLIP_MAX) * tanh(|g| / gamma)`.
pub(crate) const V06_REP_CLIP_MAX: f64 = 16.0;

/// Layouts with at most this many nodes take the small-layout path: halved
/// attraction and negatives drawn from all nodes. Coarse levels of the
/// recursive initialisation end up here.
pub(crate) const V06_SMALL_LAYOUT_MAX: usize = 1024;

/// Attraction multiplier on the small-layout path.
pub(crate) const V06_SMALL_ATTRACTION_SCALE: f64 = 0.5;

/// Default hard-negative window. Layouts with fewer nodes than this draw
/// negatives from the whole embedding.
pub const V06_DEFAULT_NEG_SELECTION_RANGE: usize = 200_000;

/// Floor on the hard-negative window, so a small configured range still
/// leaves a usable candidate pool.
pub(crate) const V06_MIN_NEG_SELECTION_RANGE: usize = 200;

/// Exponent of the initial hard-negative repulsion scale
/// `(range / n)^V06_NEG_SCALE_EXPONENT`. Local negatives are closer and so
/// repel harder than uniform ones; this scales them back.
const V06_NEG_SCALE_EXPONENT: f64 = 0.75;

/// Source points sampled per adaptation of the hard-negative scale.
const V06_ADAPT_SAMPLES: usize = 128;

/// Negatives per sampled source in the adaptation estimate.
const V06_ADAPT_NEGS_PER_SOURCE: usize = 5;

/// The hard-negative scale is re-estimated every this many epochs, from this
/// epoch on, during the first half of the run.
const V06_ADAPT_EVERY: usize = 10;

/// Quantiles that trim the per-source log force ratios before averaging.
const V06_ADAPT_TRIM: (f64, f64) = (0.1, 0.9);

/// Smoothing of the hard-negative scale update in log space.
const V06_ADAPT_STEP: f64 = 0.1;

/// Bounds on the multiplicative change of one raw scale proposal. Not
/// symmetric in log space; kept as in the reference.
const V06_ADAPT_CHANGE: (f64, f64) = (0.9, 1.1);

/// Bound, as a factor either way, on the scale relative to its prior.
const V06_ADAPT_PRIOR_BAND: f64 = 2.0;

/// Upper bound on the learning-rate warm-up length in epochs.
const V06_WARMUP_MAX: f64 = 100.0;

/// Learning rate at the end of the warm-up.
const V06_WARMUP_FLOOR: f64 = 0.1;

/// Learning rate after warm-up decays linearly from
/// `V06_TAIL_SPAN + V06_TAIL_FLOOR` to `V06_TAIL_FLOOR`.
const V06_TAIL_SPAN: f64 = 0.15;

/// See [`V06_TAIL_SPAN`].
const V06_TAIL_FLOOR: f64 = 0.05;

/// Quantiles whose span sets the robust spread in [`rescale_isotropic`].
const V06_RESCALE_QUANTILES: (f64, f64) = (0.05, 0.95);

////////////
// Config //
////////////

/// Everything one 0.6 layout run needs besides the graph and the embedding.
///
/// The full UMAP run builds it from [`UmapOptimParams`]; the recursive
/// initialisation builds one per coarse level with its own curve, repulsion
/// and window.
#[derive(Clone, Copy, Debug)]
pub(crate) struct V06Config<T> {
    /// Curve parameter `a`
    pub a: T,
    /// Curve parameter `b`
    pub b: T,
    /// Repulsion strength
    pub gamma: T,
    /// Initial learning rate
    pub lr: T,
    /// Number of epochs
    pub n_epochs: usize,
    /// Nominal negatives per positive sample
    pub neg_sample_rate: usize,
    /// Hard-negative window; hard negatives are active when it is smaller
    /// than the number of nodes
    pub neg_selection_range: usize,
    /// Use the short warm-up schedule meant for a good initialisation
    pub good_init: bool,
}

/////////////
// Helpers //
/////////////

/// Sampling clocks of one directed CSR entry.
#[derive(Clone, Copy)]
struct EdgeClock<T> {
    /// Epoch at which the edge is next sampled
    next_sample: T,
    /// Epoch up to which negatives have been drawn for this entry
    next_neg: T,
}

/// Linear-interpolation quantile of an unsorted slice, as numpy's default.
///
/// ### Params
///
/// * `values` - The values; sorted in place
/// * `q` - Quantile in `[0, 1]`
///
/// ### Returns
///
/// The `q` quantile, `0` for an empty slice.
fn quantile_in_place(values: &mut [f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(|a, b| a.total_cmp(b));
    let pos = q * (values.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(values.len() - 1);
    values[lo] + (pos - lo as f64) * (values[hi] - values[lo])
}

/// Learning rate per epoch for the 0.6 Adam.
///
/// A quadratic warm-up decay from `2 * lr` (or `0.5 * lr` after a good
/// initialisation) down to `V06_WARMUP_FLOOR`, then a linear decay from
/// `0.2` to `0.05`. The good-initialisation warm-up is shorter.
///
/// ### Params
///
/// * `n_epochs` - Number of epochs
/// * `lr` - Initial learning rate
/// * `good_init` - Whether the initialisation is already close to the result
///
/// ### Returns
///
/// One learning rate per epoch.
pub(crate) fn alpha_schedule(n_epochs: usize, lr: f64, good_init: bool) -> Vec<f64> {
    let n = n_epochs as f64;
    let (n_warm, start) = if good_init {
        (V06_WARMUP_MAX.min(n / 8.0) as usize, 0.5 * lr)
    } else {
        ((n / 2.0).min(V06_WARMUP_MAX) as usize, 2.0 * lr)
    };
    let tail = (n_epochs - n_warm).max(1) as f64;
    (0..n_epochs)
        .map(|e| {
            if e < n_warm {
                let t = 1.0 - e as f64 / n_warm as f64;
                (start - V06_WARMUP_FLOOR) * t * t + V06_WARMUP_FLOOR
            } else {
                V06_TAIL_SPAN * (1.0 - (e - n_warm) as f64 / tail) + V06_TAIL_FLOOR
            }
        })
        .collect()
}

/// Order the nodes along a random direction and record each node's rank.
///
/// Hard negatives for a node are drawn from a window around its rank, so
/// they are nodes that are close in the embedding along that direction.
///
/// ### Params
///
/// * `embd` - Flat row-major embedding `[n * n_dim]`
/// * `n_dim` - Embedding dimension
/// * `rng` - Random generator for the direction
///
/// ### Returns
///
/// `(order, ranks)`: nodes sorted by projection, and `ranks[order[k]] = k`.
pub(crate) fn projection_order<T>(
    embd: &[T],
    n_dim: usize,
    rng: &mut SmallRng,
) -> (Vec<u32>, Vec<u32>)
where
    T: ManifoldsFloat,
{
    let mut dir: Vec<f64> = (0..n_dim).map(|_| rng.sample(StandardNormal)).collect();
    let norm = dir
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    dir.iter_mut().for_each(|x| *x /= norm);

    let proj: Vec<f64> = embd
        .par_chunks_exact(n_dim)
        .map(|p| {
            p.iter()
                .zip(&dir)
                .map(|(x, d)| x.to_f64().unwrap() * d)
                .sum()
        })
        .collect();

    let mut order: Vec<u32> = (0..proj.len() as u32).collect();
    // ties broken by index keep the order reproducible
    order.par_sort_unstable_by(|&i, &j| {
        proj[i as usize]
            .total_cmp(&proj[j as usize])
            .then(i.cmp(&j))
    });

    let mut ranks = vec![0u32; order.len()];
    for (k, &i) in order.iter().enumerate() {
        ranks[i as usize] = k as u32;
    }
    (order, ranks)
}

/// Start of the hard-negative window centred on `center`, clamped to the
/// node range.
///
/// ### Params
///
/// * `center` - Rank of the source node
/// * `range` - Window size
/// * `n` - Number of nodes
///
/// ### Returns
///
/// First rank of the window.
#[inline(always)]
fn window_start(center: usize, range: usize, n: usize) -> usize {
    let start = center.saturating_sub(range / 2);
    if start + range > n {
        n.saturating_sub(range)
    } else {
        start
    }
}

/// Soft-clipped repulsive gradient coefficient.
///
/// ### Params
///
/// * `dist_sq` - Squared distance, positive
/// * `dist_sq_b` - `dist_sq^b`
/// * `a` - Curve parameter `a`
/// * `two_gamma_b` - `2 * gamma * b`
/// * `gamma` - Repulsion strength
/// * `clip` - `min(gamma, V06_REP_CLIP_MAX)`
/// * `rep_eps` - [`V06_REP_EPS`] in `T`
///
/// ### Returns
///
/// The coefficient on `(y_i - y_k)`, already including the clip factor.
#[inline(always)]
fn repulsion_coeff<T>(
    dist_sq: T,
    dist_sq_b: T,
    a: T,
    two_gamma_b: T,
    gamma: T,
    clip: T,
    rep_eps: T,
) -> T
where
    T: ManifoldsFloat,
{
    let coeff = two_gamma_b / ((rep_eps + dist_sq) * (a * dist_sq_b + T::one()));
    if coeff <= T::zero() {
        return T::zero();
    }
    let grad_norm = coeff * dist_sq.sqrt();
    coeff * clip * (grad_norm / gamma).tanh() / grad_norm
}

/// Ratio of uniform to local resultant repulsion, used to adapt the
/// hard-negative scale.
///
/// For a sample of sources, sums the soft-clipped repulsion from a few
/// uniform negatives and a few window negatives, and returns the trimmed
/// geometric mean of the ratio of the two resultant norms.
///
/// ### Params
///
/// * `embd` - Flat row-major embedding
/// * `n_dim` - Embedding dimension
/// * `order` - Current projection order
/// * `range` - Hard-negative window
/// * `cfg` - Layout configuration
/// * `lut` - `x^b` table
/// * `rng` - Random generator of the adaptation
///
/// ### Returns
///
/// The force ratio, `1` if no sample had both resultants non-zero.
pub(crate) fn negative_force_ratio<T>(
    embd: &[T],
    n_dim: usize,
    order: &[u32],
    range: usize,
    cfg: &V06Config<T>,
    lut: &SplitPowLut<T>,
    rng: &mut SmallRng,
) -> f64
where
    T: ManifoldsFloat,
{
    let n = order.len();
    let n_sources = V06_ADAPT_SAMPLES.min(n);
    let two_gamma_b = T::from_f64(2.0).unwrap() * cfg.gamma * cfg.b;
    let rep_eps = T::from_f64(V06_REP_EPS).unwrap();

    // the reference clips with gamma here, not min(gamma, 16)
    let resultant = |src: usize, cands: &[usize]| -> f64 {
        let mut acc = vec![T::zero(); n_dim];
        let ps = &embd[src * n_dim..(src + 1) * n_dim];
        for &rank in cands {
            let k = order[rank] as usize;
            let pk = &embd[k * n_dim..(k + 1) * n_dim];
            let dist_sq = ps
                .iter()
                .zip(pk)
                .fold(T::zero(), |s, (x, y)| s + (*x - *y) * (*x - *y));
            let coeff = if dist_sq > T::zero() {
                repulsion_coeff(
                    dist_sq,
                    lut.get(dist_sq),
                    cfg.a,
                    two_gamma_b,
                    cfg.gamma,
                    cfg.gamma,
                    rep_eps,
                )
            } else {
                T::zero()
            };
            for d in 0..n_dim {
                acc[d] += coeff * (ps[d] - pk[d]);
            }
        }
        acc.iter()
            .map(|x| x.to_f64().unwrap().powi(2))
            .sum::<f64>()
            .sqrt()
    };

    let mut log_ratios = Vec::with_capacity(n_sources);
    let mut local = vec![0usize; V06_ADAPT_NEGS_PER_SOURCE];
    let mut global = vec![0usize; V06_ADAPT_NEGS_PER_SOURCE];
    for _ in 0..n_sources {
        let src_rank = rng.random_range(0..n);
        let start = src_rank.saturating_sub(range / 2).min(n - range);
        for (l, g) in local.iter_mut().zip(global.iter_mut()) {
            *l = start + rng.random_range(0..range);
            *g = rng.random_range(0..n);
        }
        let src = order[src_rank] as usize;
        let rg = resultant(src, &global);
        let rl = resultant(src, &local);
        if rg > 0.0 && rl > 0.0 {
            log_ratios.push(rg.ln() - rl.ln());
        }
    }
    if log_ratios.is_empty() {
        return 1.0;
    }
    let mut scratch = log_ratios.clone();
    let lower = quantile_in_place(&mut scratch, V06_ADAPT_TRIM.0);
    let upper = quantile_in_place(&mut scratch, V06_ADAPT_TRIM.1);
    let mean = log_ratios
        .iter()
        .map(|x| x.clamp(lower, upper))
        .sum::<f64>()
        / log_ratios.len() as f64;
    mean.exp()
}

/// Repulsion scale of the hard negatives and its force-ratio adaptation.
///
/// Starts at `(range / n)^V06_NEG_SCALE_EXPONENT`. The first adaptation
/// anchors the uniform-to-local force ratio; later ones nudge the scale in
/// log space towards keeping that ratio, bounded around the prior.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HardNegScale {
    /// Current scale on the hard-negative repulsion
    pub scale: f64,
    /// Initial scale, the centre of the allowed band
    prior: f64,
    /// `scale / ratio` at the first adaptation, negative before it
    anchor: f64,
}

impl HardNegScale {
    /// Initial scale for a window of `range` nodes out of `n`.
    ///
    /// ### Params
    ///
    /// * `range` - Configured hard-negative window
    /// * `n` - Number of nodes
    ///
    /// ### Returns
    ///
    /// The scale state before any adaptation.
    pub fn new(range: usize, n: usize) -> Self {
        let scale = (range.min(n) as f64 / n as f64).powf(V06_NEG_SCALE_EXPONENT);
        Self {
            scale,
            prior: scale,
            anchor: -1.0,
        }
    }

    /// Whether the scale is re-estimated at this epoch.
    ///
    /// ### Params
    ///
    /// * `epoch` - Current epoch
    /// * `n_epochs` - Total epochs
    ///
    /// ### Returns
    ///
    /// `true` every `V06_ADAPT_EVERY` epochs during the first half of the run.
    pub fn due(epoch: usize, n_epochs: usize) -> bool {
        (V06_ADAPT_EVERY..=n_epochs / 2).contains(&epoch) && epoch.is_multiple_of(V06_ADAPT_EVERY)
    }

    /// Fold in one force-ratio estimate from [`negative_force_ratio`].
    ///
    /// ### Params
    ///
    /// * `ratio` - Uniform-to-local resultant repulsion ratio
    pub fn update(&mut self, ratio: f64) {
        if self.anchor < 0.0 {
            self.anchor = self.scale / ratio;
            return;
        }
        let raw = (self.anchor * ratio).clamp(
            self.prior / V06_ADAPT_PRIOR_BAND,
            self.prior * V06_ADAPT_PRIOR_BAND,
        );
        let log_change = (raw / self.scale)
            .ln()
            .clamp(V06_ADAPT_CHANGE.0.ln(), V06_ADAPT_CHANGE.1.ln());
        self.scale *= (V06_ADAPT_STEP * log_change).exp();
    }
}

//////////
// Core //
//////////

/// Centre an initialisation and give it the 0.6 robust spread.
///
/// The geometric mean of the per-axis 5th-95th percentile spans is scaled to
/// `log10(n) * 3 * log2(gamma + 1)`, one multiplier for all axes, so the
/// aspect ratio of the initialisation survives.
///
/// ### Params
///
/// * `embd` - Flat row-major embedding `[n * n_dim]`, modified in place
/// * `n_dim` - Embedding dimension
/// * `gamma` - Repulsion strength
pub(crate) fn rescale_isotropic<T>(embd: &mut [T], n_dim: usize, gamma: f64)
where
    T: ManifoldsFloat,
{
    let n = embd.len() / n_dim;
    if n == 0 {
        return;
    }
    let mut mean = vec![0.0f64; n_dim];
    for p in embd.chunks_exact(n_dim) {
        for d in 0..n_dim {
            mean[d] += p[d].to_f64().unwrap();
        }
    }
    mean.iter_mut().for_each(|m| *m /= n as f64);

    let mut log_span_sum = 0.0;
    let mut column = vec![0.0f64; n];
    for d in 0..n_dim {
        for (c, p) in column.iter_mut().zip(embd.chunks_exact(n_dim)) {
            *c = p[d].to_f64().unwrap() - mean[d];
        }
        let lo = quantile_in_place(&mut column, V06_RESCALE_QUANTILES.0);
        let hi = quantile_in_place(&mut column, V06_RESCALE_QUANTILES.1);
        let span = if hi - lo == 0.0 { 1.0 } else { hi - lo };
        log_span_sum += span.ln();
    }
    let characteristic = (log_span_sum / n_dim as f64).exp();
    let target = (n as f64).log10() * 3.0 * (gamma + 1.0).log2();
    let mult = target / characteristic;

    embd.par_chunks_exact_mut(n_dim).for_each(|p| {
        for d in 0..n_dim {
            p[d] = T::from_f64((p[d].to_f64().unwrap() - mean[d]) * mult).unwrap();
        }
    });
}

/// Run the 0.6 Adam layout on a flat embedding.
///
/// Parallel over nodes per epoch. Each node walks its own CSR row: sampled
/// edges pull it towards the neighbour, and each sampled edge draws its own
/// negatives for that node, so both endpoints of an edge repel. Negatives
/// are `order[(p * (epoch + q + 1)) mod range]` (offset into the node's
/// hard-negative window when active) for CSR position `p` and draw `q`; the
/// randomness is in `order`, reshuffled or re-projected every epoch.
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
/// `Ok(())`, or `ManifoldsError::GraphSizeMismatch` if the graph and the
/// embedding disagree on the number of nodes.
///
/// ### References
///
/// McInnes, umap-learn 0.6, `umap/layouts.py`
/// (`optimize_layout_euclidean_single_epoch_adam`).
pub(crate) fn optimise_layout_v06<T>(
    embd: &mut [T],
    n_dim: usize,
    csr: &UmapCsrGraph<T>,
    cfg: &V06Config<T>,
    seed: u64,
    verbose: usize,
) -> Result<(), ManifoldsError>
where
    T: ManifoldsFloat,
{
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
    let range = n.min(V06_MIN_NEG_SELECTION_RANGE.max(cfg.neg_selection_range));
    let mut neg_scale = HardNegScale::new(cfg.neg_selection_range, n);

    let lut = SplitPowLut::new(cfg.b);
    let two = T::from_f64(2.0).unwrap();
    let two_a_b = two * cfg.a * cfg.b;
    let two_gamma_b = two * cfg.gamma * cfg.b;
    let rep_clip = cfg.gamma.min(T::from_f64(V06_REP_CLIP_MAX).unwrap());
    let rep_eps = T::from_f64(V06_REP_EPS).unwrap();
    let attr_scale = if small {
        T::from_f64(V06_SMALL_ATTRACTION_SCALE).unwrap()
    } else {
        T::one()
    };
    let beta1 = T::from_f64(V06_BETA1).unwrap();
    let beta2 = T::from_f64(V06_BETA2).unwrap();
    let adam_eps = T::from_f64(V06_ADAM_EPS).unwrap();

    let neg_period_scale =
        T::from_f64(V06_NEG_PERIOD_SCALE).unwrap() / T::from(cfg.neg_sample_rate.max(1)).unwrap();
    let epochs_per_neg: Vec<T> = csr
        .epochs_per_sample
        .iter()
        .map(|&e| e * neg_period_scale)
        .collect();

    let offsets = &csr.node_edge_offsets;
    let mut clocks: Vec<EdgeClock<T>> = csr
        .csr_edge_idx
        .iter()
        .map(|&e| EdgeClock {
            next_sample: csr.epochs_per_sample[e as usize],
            next_neg: epochs_per_neg[e as usize],
        })
        .collect();
    let mut rows: Vec<&mut [EdgeClock<T>]> = Vec::with_capacity(n);
    let mut rest = clocks.as_mut_slice();
    for i in 0..n {
        let len = (offsets[i + 1] - offsets[i]) as usize;
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(len);
        rows.push(head);
        rest = tail;
    }

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut adapt_rng = SmallRng::seed_from_u64(seed.wrapping_add(0x9e37_79b9_7f4a_7c15));
    let (mut order, mut ranks): (Vec<u32>, Vec<u32>) = if !small && hard_negatives {
        projection_order(embd, n_dim, &mut rng)
    } else {
        ((0..n as u32).collect(), (0..n as u32).collect())
    };

    let alphas = alpha_schedule(cfg.n_epochs, cfg.lr.to_f64().unwrap(), cfg.good_init);
    let mut grads = vec![T::zero(); n * n_dim];
    let mut m = vec![T::zero(); n * n_dim];
    let mut v = vec![T::zero(); n * n_dim];

    for epoch in 0..cfg.n_epochs {
        if !small && hard_negatives && HardNegScale::due(epoch, cfg.n_epochs) {
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

        let epoch_t = T::from(epoch).unwrap();
        let neg_scale_t = if small {
            T::one()
        } else {
            T::from_f64(neg_scale.scale).unwrap()
        };
        let embd_ro: &[T] = embd;
        let order_ro = &order;
        let ranks_ro = &ranks;

        if n_dim == 2 {
            // as the general pass below, with the node's step in registers
            let (pts, _) = embd_ro.as_chunks::<2>();
            let (g2, _) = grads.as_chunks_mut::<2>();
            g2.par_iter_mut()
                .zip(rows.par_iter_mut())
                .enumerate()
                .for_each(|(i, (grad, row_clocks))| {
                    let yi = pts[i];
                    let mut g = [T::zero(); 2];
                    let row_start = offsets[i] as usize;
                    let win_start = if small {
                        0
                    } else {
                        window_start(ranks_ro[i] as usize, range, n)
                    };
                    let modulus = if small { n } else { range };

                    for (local, clock) in row_clocks.iter_mut().enumerate() {
                        if clock.next_sample > epoch_t {
                            continue;
                        }
                        let p = row_start + local;
                        let e = csr.csr_edge_idx[p] as usize;
                        let yj = pts[csr.csr_other_node[p] as usize];
                        let (d0, d1) = (yi[0] - yj[0], yi[1] - yj[1]);
                        let dist_sq = T::zero() + d0 * d0 + d1 * d1;
                        if dist_sq > T::zero() {
                            let dist_sq_b = lut.get(dist_sq);
                            let coeff = -attr_scale * two_a_b * dist_sq_b
                                / (dist_sq * (cfg.a * dist_sq_b + T::one()));
                            g[0] += coeff * d0;
                            g[1] += coeff * d1;
                        }
                        clock.next_sample += csr.epochs_per_sample[e];

                        let n_neg = ((epoch_t - clock.next_neg) / epochs_per_neg[e])
                            .floor()
                            .to_usize()
                            .unwrap_or(0);
                        for q in 0..n_neg {
                            let hash = ((p as u64) * (epoch as u64 + q as u64 + 1) % modulus as u64)
                                as usize;
                            let yk = pts[order_ro[(win_start + hash) % n] as usize];
                            let (d0, d1) = (yi[0] - yk[0], yi[1] - yk[1]);
                            let dist_sq = T::zero() + d0 * d0 + d1 * d1;
                            if dist_sq > T::zero() {
                                let coeff = neg_scale_t
                                    * repulsion_coeff(
                                        dist_sq,
                                        lut.get(dist_sq),
                                        cfg.a,
                                        two_gamma_b,
                                        cfg.gamma,
                                        rep_clip,
                                        rep_eps,
                                    );
                                g[0] += coeff * d0;
                                g[1] += coeff * d1;
                            }
                        }
                        clock.next_neg += T::from(n_neg).unwrap() * epochs_per_neg[e];
                    }
                    *grad = g;
                });
        } else {
            grads
                .par_chunks_exact_mut(n_dim)
                .zip(rows.par_iter_mut())
                .enumerate()
                .for_each(|(i, (grad, row_clocks))| {
                    grad.iter_mut().for_each(|g| *g = T::zero());
                    let base_i = i * n_dim;
                    let current = &embd_ro[base_i..base_i + n_dim];
                    let row_start = offsets[i] as usize;
                    let win_start = if small {
                        0
                    } else {
                        window_start(ranks_ro[i] as usize, range, n)
                    };
                    let modulus = if small { n } else { range };

                    for (local, clock) in row_clocks.iter_mut().enumerate() {
                        if clock.next_sample > epoch_t {
                            continue;
                        }
                        let p = row_start + local;
                        let e = csr.csr_edge_idx[p] as usize;
                        let j = csr.csr_other_node[p] as usize;
                        let other = &embd_ro[j * n_dim..(j + 1) * n_dim];

                        let dist_sq = current
                            .iter()
                            .zip(other)
                            .fold(T::zero(), |s, (x, y)| s + (*x - *y) * (*x - *y));
                        if dist_sq > T::zero() {
                            let dist_sq_b = lut.get(dist_sq);
                            let coeff = -attr_scale * two_a_b * dist_sq_b
                                / (dist_sq * (cfg.a * dist_sq_b + T::one()));
                            for d in 0..n_dim {
                                grad[d] += coeff * (current[d] - other[d]);
                            }
                        }
                        clock.next_sample += csr.epochs_per_sample[e];

                        let n_neg = ((epoch_t - clock.next_neg) / epochs_per_neg[e])
                            .floor()
                            .to_usize()
                            .unwrap_or(0);
                        for q in 0..n_neg {
                            let hash = ((p as u64) * (epoch as u64 + q as u64 + 1) % modulus as u64)
                                as usize;
                            let k = order_ro[(win_start + hash) % n] as usize;
                            let pk = &embd_ro[k * n_dim..(k + 1) * n_dim];
                            let dist_sq = current
                                .iter()
                                .zip(pk)
                                .fold(T::zero(), |s, (x, y)| s + (*x - *y) * (*x - *y));
                            if dist_sq > T::zero() {
                                let coeff = neg_scale_t
                                    * repulsion_coeff(
                                        dist_sq,
                                        lut.get(dist_sq),
                                        cfg.a,
                                        two_gamma_b,
                                        cfg.gamma,
                                        rep_clip,
                                        rep_eps,
                                    );
                                for d in 0..n_dim {
                                    grad[d] += coeff * (current[d] - pk[d]);
                                }
                            }
                        }
                        clock.next_neg += T::from(n_neg).unwrap() * epochs_per_neg[e];
                    }
                });
        }

        let alpha = T::from_f64(alphas[epoch]).unwrap();
        let t = T::from(epoch + 1).unwrap();
        let bc1 = T::one() - beta1.powf(t);
        let bc2 = T::one() - beta2.powf(t);
        embd.par_iter_mut()
            .zip(grads.par_iter())
            .zip(m.par_iter_mut().zip(v.par_iter_mut()))
            .for_each(|((x, &g), (mi, vi))| {
                if g != T::zero() {
                    *mi = beta1 * *mi + (T::one() - beta1) * g;
                    *vi = beta2 * *vi + (T::one() - beta2) * g * g;
                    *x += alpha * (*mi / bc1) / ((*vi / bc2).sqrt() + adam_eps);
                }
            });

        if hard_negatives {
            (order, ranks) = projection_order(embd, n_dim, &mut rng);
        } else {
            order.shuffle(&mut rng);
        }

        if verbosity.normal_verbosity() && ((epoch + 1) % 50 == 0 || epoch + 1 == cfg.n_epochs) {
            println!("  Epoch {}/{}", epoch + 1, cfg.n_epochs);
        }
    }

    Ok(())
}

/// Optimise a UMAP embedding with the umap-learn 0.6 Adam.
///
/// Rescales the initialisation with `rescale_isotropic` first, as the 0.6
/// pipeline does, then runs `optimise_layout_v06`. `beta1`, `beta2` and
/// `eps` in `params` are ignored; the 0.6 constants are fixed.
///
/// ### Params
///
/// * `embd` - Initial embedding `[n][n_dim]`, modified in place
/// * `graph` - Symmetric adjacency list
/// * `params` - Curve, learning rate, repulsion, epochs and negative rate
/// * `good_init` - Use the short warm-up meant for a good initialisation
///   (the recursive initialisation)
/// * `seed` - Random seed
/// * `verbose` - `0` silent, `1` or `2` print the epoch every 50 epochs
///
/// ### Returns
///
/// `Ok(())` or the graph construction error.
pub fn optimise_embedding_adam_v06<T>(
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
    let mut flat: Vec<T> = embd.iter().flatten().copied().collect();

    rescale_isotropic(&mut flat, n_dim, params.gamma.to_f64().unwrap());

    let csr = UmapCsrGraph::from_graph(graph)?;
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
    optimise_layout_v06(&mut flat, n_dim, &csr, &cfg, seed, verbose)?;

    for (point, chunk) in embd.iter_mut().zip(flat.chunks_exact(n_dim)) {
        point.copy_from_slice(chunk);
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

    /// Two rings of `n_per` nodes, each node linked to its `k` ring
    /// neighbours on either side, no edges between the rings.
    fn two_rings(n_per: usize, k: usize) -> Vec<Vec<(usize, f64)>> {
        let n = 2 * n_per;
        let mut graph = vec![Vec::new(); n];
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

    /// Random flat embedding in `[-1, 1]`.
    fn random_flat(n: usize, n_dim: usize, seed: u64) -> Vec<f64> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..n * n_dim)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect()
    }

    /// Mean graph-edge length over the mean distance of random pairs. Rings
    /// laid out from a random start can tangle, so centroid separation is
    /// not a fair check; short edges are.
    fn edge_ratio(embd: &[f64], graph: &[Vec<(usize, f64)>]) -> f64 {
        let dist = |i: usize, j: usize| {
            ((embd[2 * i] - embd[2 * j]).powi(2) + (embd[2 * i + 1] - embd[2 * j + 1]).powi(2))
                .sqrt()
        };
        let (mut edge, mut n_edge) = (0.0, 0usize);
        for (i, nbrs) in graph.iter().enumerate() {
            for &(j, _) in nbrs {
                edge += dist(i, j);
                n_edge += 1;
            }
        }
        let mut rng = SmallRng::seed_from_u64(99);
        let n = graph.len();
        let rand = (0..5000)
            .map(|_| dist(rng.random_range(0..n), rng.random_range(0..n)))
            .sum::<f64>()
            / 5000.0;
        (edge / n_edge as f64) / rand
    }

    fn config(n_epochs: usize, range: usize) -> V06Config<f64> {
        let p = UmapOptimParams::<f64>::default_2d();
        V06Config {
            a: p.a,
            b: p.b,
            gamma: 1.0,
            lr: 1.0,
            n_epochs,
            neg_sample_rate: 5,
            neg_selection_range: range,
            good_init: false,
        }
    }

    #[test]
    fn test_v06_alpha_schedule_matches_reference() {
        let idx = [0, 1, 12, 24, 25, 99, 100, 150, 199];
        let cold = [
            2.0, 1.96219, 1.57136, 1.19744, 1.16875, 0.10019, 0.2, 0.125, 0.0515,
        ];
        let warm = [
            0.5,
            0.46864,
            0.20816,
            0.10064,
            0.2,
            0.136571428571,
            0.135714285714,
            0.092857142857,
            0.050857142857,
        ];
        let s_cold = alpha_schedule(200, 1.0, false);
        let s_warm = alpha_schedule(200, 1.0, true);
        for (k, &i) in idx.iter().enumerate() {
            assert_relative_eq!(s_cold[i], cold[k], epsilon = 1e-9);
            assert_relative_eq!(s_warm[i], warm[k], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_v06_quantile_matches_numpy() {
        let mut v = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        assert_relative_eq!(quantile_in_place(&mut v, 0.05), 1.0, epsilon = 1e-12);
        assert_relative_eq!(quantile_in_place(&mut v, 0.95), 7.95, epsilon = 1e-12);
    }

    #[test]
    fn test_v06_rescale_isotropic_matches_reference() {
        let mut x = vec![
            0., 0., 1., 4., 2., 1., 3., 9., 4., 2., 5., 3., 6., 7., 7., 5.,
        ];
        let expected = [
            -1.3398817777633667,
            -1.483440637588501,
            -0.9570584297180176,
            0.04785292223095894,
            -0.5742350816726685,
            -1.1006171703338623,
            -0.19141168892383575,
            1.9619698524475098,
            0.19141168892383575,
            -0.7177938222885132,
            0.5742350816726685,
            -0.3349704444408417,
            0.9570584297180176,
            1.196323037147522,
            1.3398817777633667,
            0.43067631125450134,
        ];
        rescale_isotropic(&mut x, 2, 1.0);
        for (a, b) in x.iter().zip(expected) {
            assert_relative_eq!(*a, b, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_v06_window_start_clamps() {
        assert_eq!(window_start(0, 10, 100), 0);
        assert_eq!(window_start(50, 10, 100), 45);
        assert_eq!(window_start(98, 10, 100), 90);
        assert_eq!(window_start(3, 100, 100), 0);
    }

    #[test]
    fn test_v06_projection_order_ranks_invert() {
        let embd = random_flat(500, 2, 1);
        let mut rng = SmallRng::seed_from_u64(7);
        let (order, ranks) = projection_order(&embd, 2, &mut rng);
        let mut seen = vec![false; 500];
        for (k, &i) in order.iter().enumerate() {
            assert_eq!(ranks[i as usize] as usize, k);
            seen[i as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn test_v06_reproducible_same_seed() {
        let graph = two_rings(100, 5);
        let csr = UmapCsrGraph::from_graph(&graph).unwrap();
        let cfg = config(50, V06_DEFAULT_NEG_SELECTION_RANGE);
        let mut a = random_flat(200, 2, 3);
        let mut b = a.clone();
        optimise_layout_v06(&mut a, 2, &csr, &cfg, 11, 0).unwrap();
        optimise_layout_v06(&mut b, 2, &csr, &cfg, 11, 0).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_v06_small_layout_separates_rings() {
        let graph = two_rings(200, 5);
        let csr = UmapCsrGraph::from_graph(&graph).unwrap();
        let mut embd = random_flat(400, 2, 5);
        rescale_isotropic(&mut embd, 2, 1.0);
        optimise_layout_v06(
            &mut embd,
            2,
            &csr,
            &config(200, V06_DEFAULT_NEG_SELECTION_RANGE),
            1,
            0,
        )
        .unwrap();
        assert!(embd.iter().all(|x| x.is_finite()));
        assert!(edge_ratio(&embd, &graph) < 0.2);
    }

    #[test]
    fn test_v06_hard_negative_layout_separates_rings() {
        // 3000 nodes with a 500-node window takes the large, hard-negative path
        let graph = two_rings(1500, 5);
        let csr = UmapCsrGraph::from_graph(&graph).unwrap();
        let mut embd = random_flat(3000, 2, 9);
        rescale_isotropic(&mut embd, 2, 1.0);
        optimise_layout_v06(&mut embd, 2, &csr, &config(200, 500), 2, 0).unwrap();
        assert!(embd.iter().all(|x| x.is_finite()));
        assert!(edge_ratio(&embd, &graph) < 0.2);
    }

    #[test]
    fn test_v06_size_mismatch_errors() {
        let graph = two_rings(10, 2);
        let csr = UmapCsrGraph::from_graph(&graph).unwrap();
        let mut embd = vec![0.0; 10];
        assert!(optimise_layout_v06(&mut embd, 2, &csr, &config(5, 100), 0, 0).is_err());
    }
}
