//! Optimisers for PaCMAP and LocalMAP fitting.
//!
//! The sequential variants scatter pair forces onto both endpoints. The parallel
//! variants run a fused per-node gradient-plus-Adam step over per-node CSR
//! pairs, with a dedicated 2D kernel.

use num_traits::{Float, FromPrimitive};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;

use crate::data::pacmap_pairs::PairCsr;
use crate::prelude::*;
use crate::training::*;
use crate::utils::prefetch::prefetch_read;

/////////////
// Globals //
/////////////

/// End of first phase during fitting
const PHASE1_END: usize = 100;
/// End of second phase during fitting
const PHASE2_END: usize = 200;
/// End of third (and last) phase of fitting
const PHASE3_END: usize = 450;
/// Weight for the nearest points in first phase (dominating by mid-near)
const W_NB_PHASE1: f64 = 2.0;
/// Weight for nearest points during the second phase (focussing on near pairs)
const W_NB_PHASE2: f64 = 3.0;
/// Weight for the end of the second phase
const W_MN_PHASE2_END: f64 = 3.0;
/// Weight for nearest points during the third phase (local structure refinement)
const W_NB_PHASE3: f64 = 1.0;
/// Starting weight of the mid-near pairs, decaying across the first phase
const W_MN_PHASE1: f64 = 1000.0;
/// Weight for the far-away points (kept constant during training)
const W_FP: f64 = 1.0;
/// Loss constant `c` of the near pairs
const C_NEAR: f64 = 10.0;
/// Loss constant `c` of the mid-near pairs
const C_MID_NEAR: f64 = 10_000.0;

/// Type of PaCMAP optimiser to use
#[derive(Default)]
pub enum PacMapOptimiser {
    /// Parallel version of Adam
    #[default]
    AdamParallel,
    /// Adam
    Adam,
}

/// Parse the PacMap Optimiser to use
///
/// ### Params
///
/// * `s` - String defining the optimiser. Choice of `"adam"` or
///   `"adam_parallel"`
///
/// ### Returns
///
/// Option of Optimiser
pub fn parse_pacmap_optimiser(s: &str) -> Option<PacMapOptimiser> {
    match s.to_lowercase().as_str() {
        "adam" => Some(PacMapOptimiser::Adam),
        "adam_parallel" => Some(PacMapOptimiser::AdamParallel),
        _ => None,
    }
}

////////////
// Params //
////////////

/// PaCMAP optimisation parameters
#[derive(Clone, Debug)]
pub struct PacmapOptimParams<T> {
    /// Number of optimisation epochs. Paper uses 450.
    pub n_epochs: usize,
    /// Initial learning rate
    pub lr: T,
    /// Adam beta1
    pub beta1: T,
    /// Adam beta2
    pub beta2: T,
    /// Adam epsilon
    pub eps: T,
    /// End of phase 1 (mid-near dominant). Default 100.
    pub phase1_end: usize,
    /// End of phase 2 (mid-near weight held low). Default 200.
    pub phase2_end: usize,
}

impl<T> PacmapOptimParams<T>
where
    T: Float + FromPrimitive,
{
    /// Create a new `PacmapOptimParams` instance.
    ///
    /// All parameters are optional and fall back to the paper's defaults.
    ///
    /// ### Params
    ///
    /// * `n_epochs` - Total number of optimisation epochs across all three
    ///   phases. Default `450` (100 + 100 + 250).
    /// * `lr` - Adam learning rate. Default `1.0`. Adam moves a coordinate by
    ///   about `lr` per epoch at most, so `lr * n_epochs` caps the layout's size.
    ///   Much smaller values leave it compressed, and more so as `n` grows.
    /// * `beta1` - Adam first moment decay. Default `0.9` (standard Adam).
    ///   Unlike UMAP's uwot implementation which uses `0.5`, PaCMAP processes
    ///   all pairs every epoch so standard momentum is appropriate.
    /// * `beta2` - Adam second moment decay. Default `0.999` (standard Adam).
    /// * `eps` - Adam numerical stability constant. Default `1e-7`.
    /// * `phase1_end` - Epoch at which phase 1 ends and phase 2 begins. During
    ///   phase 1 the mid-near weight decays from `1000` to `3`, anchoring global
    ///   structure. Default `100`.
    /// * `phase2_end` - Epoch at which phase 2 ends and phase 3 begins. During
    ///   phase 2 the mid-near weight holds at `3` and the near pair weight rises
    ///   to `3`. Default `200`.
    ///
    /// ### Returns
    ///
    /// `PacmapOptimParams` with all values set to provided arguments or their
    /// defaults.
    pub fn new(
        n_epochs: Option<usize>,
        lr: Option<T>,
        beta1: Option<T>,
        beta2: Option<T>,
        eps: Option<T>,
        phase1_end: Option<usize>,
        phase2_end: Option<usize>,
    ) -> Self {
        Self {
            n_epochs: n_epochs.unwrap_or(PHASE3_END),
            lr: lr.unwrap_or(T::from_f64(1.0).unwrap()),
            beta1: beta1.unwrap_or(T::from_f64(BETA1).unwrap()),
            beta2: beta2.unwrap_or(T::from_f64(BETA2).unwrap()),
            eps: eps.unwrap_or(T::from_f64(EPS).unwrap()),
            phase1_end: phase1_end.unwrap_or(PHASE1_END),
            phase2_end: phase2_end.unwrap_or(PHASE2_END),
        }
    }
}

/// Implementation of the defaults here
impl<T> Default for PacmapOptimParams<T>
where
    T: Float + FromPrimitive,
{
    fn default() -> Self {
        Self::new(None, None, None, None, None, None, None)
    }
}

/////////////
// Helpers //
/////////////

/// Returns `(w_near, w_mid_near, w_further)` for a given epoch.
///
/// Implements the three-phase weight schedule from the PaCMAP paper:
///
/// | Phase | Epochs                    | w_near | w_mid_near      | w_further |
/// |-------|---------------------------|--------|-----------------|-----------|
/// | 1     | `0..phase1_end`           | 2.0    | 1000.0 → 3.0    | 1.0       |
/// | 2     | `phase1_end..phase2_end`  | 3.0    | 3.0             | 1.0       |
/// | 3     | `phase2_end..`            | 1.0    | 0.0             | 1.0       |
///
/// ### Params
///
/// * `epoch` - Current epoch index (zero-based).
/// * `phase1_end` - Epoch at which phase 1 transitions to phase 2.
/// * `phase2_end` - Epoch at which phase 2 transitions to phase 3.
///
/// ### Returns
///
/// Tuple of `(w_near, w_mid_near, w_further)` for the given epoch.
#[inline]
fn phase_weights<T>(epoch: usize, phase1_end: usize, phase2_end: usize) -> (T, T, T)
where
    T: Float + FromPrimitive,
{
    let w_fp = T::from_f64(W_FP).unwrap();
    if epoch < phase1_end {
        let progress = epoch as f64 / phase1_end as f64;
        let w_mn =
            T::from_f64(W_MN_PHASE1 * (1.0 - progress) + W_MN_PHASE2_END * progress).unwrap();
        (T::from_f64(W_NB_PHASE1).unwrap(), w_mn, w_fp)
    } else if epoch < phase2_end {
        (
            T::from_f64(W_NB_PHASE2).unwrap(),
            T::from_f64(W_MN_PHASE2_END).unwrap(),
            w_fp,
        )
    } else {
        (T::from_f64(W_NB_PHASE3).unwrap(), T::zero(), w_fp)
    }
}

/// Attractive loss gradient coefficient for near and mid-near pairs.
///
/// Loss: d' / (d' + c) with d' = 1 + d and d the squared embedding distance, where
/// c=10 for near and c=10000 for mid-near.
/// dLoss/dd = c / (d' + c)^2
/// Gradient w.r.t. embedding coord: 2 * (dLoss/dd) * (y_i - y_j)  [chain rule]
///
/// ### Params
///
/// * `dist_sq` - Squared Euclidean distance between the pair in embedding
///   space.
/// * `c` - Loss scaling constant. Controls how quickly the attractive force
///   saturates with distance. Near pairs use `c=10`, mid-near use `c=10000`.
///
/// ### Returns
///
/// Returns the scalar coefficient; caller multiplies by (y_i - y_j).
#[inline(always)]
fn attract_grad_coeff<T>(dist_sq: T, c: T) -> T
where
    T: Float + FromPrimitive,
{
    let two = T::from_f64(2.0).unwrap();
    let d_tilde = T::one() + dist_sq;
    let denom = c + d_tilde;
    two * c / (denom * denom)
}

/// Repulsive loss gradient coefficient for further pairs.
///
/// Loss: 1 / (1 + d)
/// dLoss/dd = -1 / (1+d)^2
///
/// ### Params
///
/// * `dist_sq` - Squared Euclidean distance between the pair in embedding
///   space.
///
/// ### Returns
///
/// Returns the scalar coefficient (positive; caller negates for repulsion).
#[inline(always)]
fn repel_grad_coeff<T>(dist_sq: T) -> T
where
    T: Float + FromPrimitive,
{
    let two = T::from_f64(2.0).unwrap();
    let d_tilde = T::one() + dist_sq;
    let denom = T::one() + d_tilde; // = 2 + dist_sq
    two / (denom * denom)
}

//////////////
// LocalMAP //
//////////////

/// Further pairs are redrawn every this many epochs in LocalMAP's phase 3.
/// Set by the LocalMAP reference implementation.
const LOCALMAP_RESAMPLE_EVERY: usize = 10;

/// Draws per further-pair slot before the resampler keeps the old pair. brisc
/// counts every kind of rejection towards this, the original only some.
const LOCALMAP_RESAMPLE_TRIALS: usize = 100;

/// Default embedding distance beyond which LocalMAP's phase 3 rejects a
/// further pair. Set by the LocalMAP reference implementation.
pub const LOCALMAP_MAX_DISTANCE: f64 = 10.0;

/// Software prefetch distance in the pair gathers, in entries.
const PREFETCH_AHEAD: usize = 8;

/// LocalMAP's phase 3 settings.
#[derive(Clone, Copy, Debug)]
struct LocalMapPhase3<T> {
    /// Half of `max_distance`; scales the near-pair force by
    /// `half_max_distance / sqrt(1 + d)`.
    half_max_distance: T,
    /// Squared `max_distance`; resampled further pairs must lie within it.
    max_distance_sq: T,
    /// Seed of the further-pair resampler.
    seed: u64,
}

/// Build LocalMAP's phase-3 settings.
///
/// ### Params
///
/// * `max_distance` - Embedding distance cut-off for further pairs.
/// * `seed` - Seed of the further-pair resampler.
///
/// ### Returns
///
/// The settings.
fn localmap_phase3<T: ManifoldsFloat>(max_distance: T, seed: u64) -> LocalMapPhase3<T> {
    LocalMapPhase3 {
        half_max_distance: T::from_f64(0.5).unwrap() * max_distance,
        max_distance_sq: max_distance * max_distance,
        seed,
    }
}

/// Owned partners per node, as the resampler needs them.
///
/// ### Params
///
/// * `pairs` - Pair list grouped by first index, as `construct_pacmap_pairs`
///   returns it.
/// * `n` - Number of nodes.
///
/// ### Returns
///
/// The partners of each node, in pair-list order.
fn own_rows(pairs: &[(usize, usize)], n: usize) -> Vec<Vec<usize>> {
    let mut rows = vec![Vec::new(); n];
    for &(i, j) in pairs {
        rows[i].push(j);
    }
    rows
}

/// Flatten per-node rows back to a pair list grouped by first index.
///
/// ### Params
///
/// * `rows` - Partners of each node.
///
/// ### Returns
///
/// The pairs `(i, j)`.
fn rows_to_pairs(rows: &[Vec<usize>]) -> Vec<(usize, usize)> {
    rows.iter()
        .enumerate()
        .flat_map(|(i, r)| r.iter().map(move |&j| (i, j)))
        .collect()
}

/// Seed of the resampling round after `epoch`. Counts down from `seed` as brisc
/// does, which keeps it clear of the pair construction seeds above it.
///
/// ### Params
///
/// * `seed` - Base seed.
/// * `epoch` - Current epoch.
/// * `n` - Number of nodes.
///
/// ### Returns
///
/// The round's seed; node `i` adds `i`.
fn resample_seed(seed: u64, epoch: usize, n: usize) -> u64 {
    seed.wrapping_sub(((epoch as u64) + 1).wrapping_mul(n as u64))
}

/// Redraw each node's further pairs near it in the embedding.
///
/// Each slot draws up to [`LOCALMAP_RESAMPLE_TRIALS`] candidates and takes
/// the first that is not the node, lies within `max_distance`, is not one of
/// the node's own near pairs and is not in an earlier slot. If none qualifies
/// the old pair stays. One RNG per node, so the result does not depend on the
/// thread count.
///
/// ### Params
///
/// * `embd` - Row-major embedding, `n * n_dim` long.
/// * `n_dim` - Embedding dimensions.
/// * `near_own` - Each node's own near pairs.
/// * `further_own` - Each node's further pairs, redrawn in place.
/// * `max_distance_sq` - Squared distance cut-off.
/// * `seed` - Seed for this round.
///
/// ### References
///
/// Wang et al., arXiv:2412.15426, 2024 (LocalMAP); brisc `embed.pyx`
/// (`sample_further_pairs_nearby`).
fn resample_further_nearby<T>(
    embd: &[T],
    n_dim: usize,
    near_own: &[Vec<usize>],
    further_own: &mut [Vec<usize>],
    max_distance_sq: T,
    seed: u64,
) where
    T: ManifoldsFloat,
{
    let n = further_own.len();
    further_own
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, slots)| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
            let yi = &embd[i * n_dim..(i + 1) * n_dim];
            for s in 0..slots.len() {
                for _ in 0..LOCALMAP_RESAMPLE_TRIALS {
                    let c = rng.random_range(0..n);
                    if c == i {
                        continue;
                    }
                    let yc = &embd[c * n_dim..(c + 1) * n_dim];
                    let d2 = yi
                        .iter()
                        .zip(yc)
                        .fold(T::zero(), |acc, (&a, &b)| acc + (a - b) * (a - b));
                    if d2 > max_distance_sq || near_own[i].contains(&c) || slots[..s].contains(&c) {
                        continue;
                    }
                    slots[s] = c;
                    break;
                }
            }
        });
}

////////////
// Kernel //
////////////

/// Adam constants for one epoch.
#[derive(Clone, Copy)]
struct AdamStep<T> {
    /// Learning rate times the bias correction.
    ad_scale: T,
    /// `1 - beta1`.
    one_minus_b1: T,
    /// `1 - beta2`.
    one_minus_b2: T,
    /// Adam epsilon.
    eps: T,
}

impl<T: ManifoldsFloat> AdamStep<T> {
    /// Constants for the epoch with bias-correction powers `beta1t`, `beta2t`.
    ///
    /// ### Params
    ///
    /// * `params` - Optimisation parameters.
    /// * `beta1t` - `beta1^(epoch + 1)`.
    /// * `beta2t` - `beta2^(epoch + 1)`.
    ///
    /// ### Returns
    ///
    /// The epoch's constants.
    fn new(params: &PacmapOptimParams<T>, beta1t: T, beta2t: T) -> Self {
        Self {
            ad_scale: params.lr * (T::one() - beta2t).sqrt() / (T::one() - beta1t),
            one_minus_b1: T::one() - params.beta1,
            one_minus_b2: T::one() - params.beta2,
            eps: params.eps,
        }
    }

    /// Apply Adam to one coordinate.
    ///
    /// ### Params
    ///
    /// * `x` - Current coordinate.
    /// * `g` - Gradient, signed as a step direction.
    /// * `m` - First moment, updated.
    /// * `v` - Second moment, updated.
    ///
    /// ### Returns
    ///
    /// The new coordinate.
    #[inline(always)]
    fn apply(&self, x: T, g: T, m: &mut T, v: &mut T) -> T {
        *m = *m + self.one_minus_b1 * (g - *m);
        *v = *v + self.one_minus_b2 * (g * g - *v);
        x + self.ad_scale * *m / (v.sqrt() + self.eps)
    }
}

/// Signed near-pair coefficient: PaCMAP's weighted attraction, or LocalMAP's
/// phase-3 attraction scaled by `half_max_distance / sqrt(1 + d)`.
///
/// ### Params
///
/// * `dist_sq` - Squared embedding distance.
/// * `w_nb` - Near-pair weight, ignored in LocalMAP's phase 3.
/// * `half_max_distance` - LocalMAP's phase-3 scaling, `None` otherwise.
///
/// ### Returns
///
/// The coefficient on `y_i - y_j`.
#[inline(always)]
fn near_coeff<T: ManifoldsFloat>(dist_sq: T, w_nb: T, half_max_distance: Option<T>) -> T {
    let c = T::from_f64(C_NEAR).unwrap();
    match half_max_distance {
        None => -(w_nb * attract_grad_coeff(dist_sq, c)),
        Some(h) => -(attract_grad_coeff(dist_sq, c) * h / (T::one() + dist_sq).sqrt()),
    }
}

/// Add one pair type's force on a node.
///
/// ### Params
///
/// * `embd` - Row-major embedding.
/// * `yi` - The node's coordinates.
/// * `n_dim` - Embedding dimensions.
/// * `partners` - All entries of the pair CSR; the node's row is
///   `partners[start..end]`. Passed whole so the prefetch runs into the next
///   row.
/// * `start` - First entry of the row.
/// * `end` - One past the last entry.
/// * `coeff` - Signed force coefficient from the squared distance; the step is
///   `coeff * (y_i - y_j)`.
/// * `grad` - Accumulated step of the node.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn gather<T, F>(
    embd: &[T],
    yi: &[T],
    n_dim: usize,
    partners: &[u32],
    start: usize,
    end: usize,
    coeff: F,
    grad: &mut [T],
) where
    T: ManifoldsFloat,
    F: Fn(T) -> T,
{
    for k in start..end {
        if let Some(&ahead) = partners.get(k + PREFETCH_AHEAD) {
            prefetch_read(&embd[ahead as usize * n_dim]);
        }
        let j = partners[k] as usize * n_dim;
        let yj = &embd[j..j + n_dim];
        let mut dist_sq = T::zero();
        for d in 0..n_dim {
            let diff = yi[d] - yj[d];
            dist_sq += diff * diff;
        }
        let c = coeff(dist_sq);
        for d in 0..n_dim {
            grad[d] += c * (yi[d] - yj[d]);
        }
    }
}

/// [`gather`] for 2D embeddings, with the step held in registers.
///
/// ### Params
///
/// * `embd` - Embedding as 2D points.
/// * `yi` - The node's coordinates.
/// * `partners` - All entries of the pair CSR, as in [`gather`].
/// * `start` - First entry of the row.
/// * `end` - One past the last entry.
/// * `coeff` - Signed force coefficient from the squared distance.
/// * `grad` - Accumulated step of the node.
#[inline(always)]
fn gather_2d<T, F>(
    embd: &[[T; 2]],
    yi: [T; 2],
    partners: &[u32],
    start: usize,
    end: usize,
    coeff: F,
    grad: &mut [T; 2],
) where
    T: ManifoldsFloat,
    F: Fn(T) -> T,
{
    for k in start..end {
        if let Some(&ahead) = partners.get(k + PREFETCH_AHEAD) {
            prefetch_read(&embd[ahead as usize]);
        }
        let yj = embd[partners[k] as usize];
        let (d0, d1) = (yi[0] - yj[0], yi[1] - yj[1]);
        let c = coeff(T::zero() + d0 * d0 + d1 * d1);
        grad[0] += c * d0;
        grad[1] += c * d1;
    }
}

/// One fused epoch: each node gathers its gradient from `old` and applies Adam
/// straight away, writing to `new`, since other nodes still need its old
/// position. 2D embeddings take a copy of the loop built on [`gather_2d`].
///
/// ### Params
///
/// * `old` - Embedding at the start of the epoch.
/// * `new` - Embedding after the epoch, fully overwritten.
/// * `m` - Adam first moments.
/// * `v` - Adam second moments.
/// * `n_dim` - Embedding dimensions.
/// * `near` - Near pairs.
/// * `mid_near` - Mid-near pairs with their weight; `None` skips them.
/// * `further` - Further pairs.
/// * `w_nb` - Near-pair weight.
/// * `half_max_distance` - LocalMAP's phase-3 near scaling, `None` otherwise.
/// * `adam` - Adam constants for this epoch.
#[allow(clippy::too_many_arguments)]
fn fused_epoch<T>(
    old: &[T],
    new: &mut [T],
    m: &mut [T],
    v: &mut [T],
    n_dim: usize,
    near: &PairCsr,
    mid_near: Option<(&PairCsr, T)>,
    further: &PairCsr,
    w_nb: T,
    half_max_distance: Option<T>,
    adam: AdamStep<T>,
) where
    T: ManifoldsFloat,
{
    let c_mn = T::from_f64(C_MID_NEAR).unwrap();
    let w_fp = T::from_f64(W_FP).unwrap();

    if n_dim == 2 {
        let (old2, _) = old.as_chunks::<2>();
        let (new2, _) = new.as_chunks_mut::<2>();
        let (m2, _) = m.as_chunks_mut::<2>();
        let (v2, _) = v.as_chunks_mut::<2>();
        new2.par_iter_mut()
            .zip(m2.par_iter_mut())
            .zip(v2.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((out, m_i), v_i))| {
                let yi = old2[i];
                let row = |csr: &PairCsr| (csr.indptr[i] as usize, csr.indptr[i + 1] as usize);
                let mut grad = [T::zero(); 2];

                let (s, e) = row(near);
                let near_c = |d2| near_coeff(d2, w_nb, half_max_distance);
                gather_2d(old2, yi, &near.indices, s, e, near_c, &mut grad);

                if let Some((mn, w_mn)) = mid_near {
                    let (s, e) = row(mn);
                    let mn_c = |d2| -(w_mn * attract_grad_coeff(d2, c_mn));
                    gather_2d(old2, yi, &mn.indices, s, e, mn_c, &mut grad);
                }

                let (s, e) = row(further);
                let fp_c = |d2| w_fp * repel_grad_coeff(d2);
                gather_2d(old2, yi, &further.indices, s, e, fp_c, &mut grad);

                for d in 0..2 {
                    out[d] = adam.apply(yi[d], grad[d], &mut m_i[d], &mut v_i[d]);
                }
            });
        return;
    }

    new.par_chunks_exact_mut(n_dim)
        .zip(m.par_chunks_exact_mut(n_dim))
        .zip(v.par_chunks_exact_mut(n_dim))
        .enumerate()
        .for_each(|(i, ((grad, m_i), v_i))| {
            let yi = &old[i * n_dim..(i + 1) * n_dim];
            let row = |csr: &PairCsr| (csr.indptr[i] as usize, csr.indptr[i + 1] as usize);
            grad.fill(T::zero());

            let (s, e) = row(near);
            let near_c = |d2| near_coeff(d2, w_nb, half_max_distance);
            gather(old, yi, n_dim, &near.indices, s, e, near_c, grad);

            if let Some((mn, w_mn)) = mid_near {
                let (s, e) = row(mn);
                let mn_c = |d2| -(w_mn * attract_grad_coeff(d2, c_mn));
                gather(old, yi, n_dim, &mn.indices, s, e, mn_c, grad);
            }

            let (s, e) = row(further);
            let fp_c = |d2| w_fp * repel_grad_coeff(d2);
            gather(old, yi, n_dim, &further.indices, s, e, fp_c, grad);

            for d in 0..n_dim {
                grad[d] = adam.apply(yi[d], grad[d], &mut m_i[d], &mut v_i[d]);
            }
        });
}

/// Scatter one pair type's force onto both endpoints of each pair, with a
/// separate loop for 2D embeddings that loads each point as one `[T; 2]`.
///
/// ### Params
///
/// * `embd` - Row-major embedding.
/// * `n_dim` - Embedding dimensions.
/// * `pairs` - Pair list.
/// * `coeff` - Signed force coefficient from the squared distance.
/// * `grads` - Accumulated steps of all nodes.
#[inline(always)]
fn scatter<T, F>(embd: &[T], n_dim: usize, pairs: &[(usize, usize)], coeff: F, grads: &mut [T])
where
    T: ManifoldsFloat,
    F: Fn(T) -> T,
{
    if n_dim == 2 {
        let (pts, _) = embd.as_chunks::<2>();
        let (g2, _) = grads.as_chunks_mut::<2>();
        for (k, &(i, j)) in pairs.iter().enumerate() {
            if let Some(&(_, ahead)) = pairs.get(k + PREFETCH_AHEAD) {
                prefetch_read(&pts[ahead]);
            }
            let (yi, yj) = (pts[i], pts[j]);
            let (d0, d1) = (yi[0] - yj[0], yi[1] - yj[1]);
            let c = coeff(T::zero() + d0 * d0 + d1 * d1);
            g2[i][0] += c * d0;
            g2[i][1] += c * d1;
            g2[j][0] -= c * d0;
            g2[j][1] -= c * d1;
        }
        return;
    }
    for (k, &(i, j)) in pairs.iter().enumerate() {
        if let Some(&(_, ahead)) = pairs.get(k + PREFETCH_AHEAD) {
            prefetch_read(&embd[ahead * n_dim]);
        }
        let (bi, bj) = (i * n_dim, j * n_dim);
        let mut dist_sq = T::zero();
        for d in 0..n_dim {
            let diff = embd[bi + d] - embd[bj + d];
            dist_sq += diff * diff;
        }
        let c = coeff(dist_sq);
        for d in 0..n_dim {
            let delta = embd[bi + d] - embd[bj + d];
            grads[bi + d] += c * delta;
            grads[bj + d] -= c * delta;
        }
    }
}

//////////
// Main //
//////////

/// Shared body of the sequential PaCMAP and LocalMAP optimisers.
///
/// Scatters each pair's force onto both endpoints, then applies Adam to the
/// whole embedding.
///
/// ### Params
///
/// * `embd` - Embedding, shape `[n_samples][n_dim]`. Modified in place.
/// * `pairs` - The three pair sets from `construct_pacmap_pairs`.
/// * `params` - Optimisation parameters.
/// * `localmap` - LocalMAP's phase-3 settings, `None` for PaCMAP.
/// * `verbose` - Verbosity level.
///
/// ### Returns
///
/// `Ok(())` or `NoData` on an empty embedding.
fn sequential_impl<T>(
    embd: &mut [Vec<T>],
    pairs: &PacmapPairs,
    params: &PacmapOptimParams<T>,
    localmap: Option<LocalMapPhase3<T>>,
    verbose: usize,
) -> Result<(), ManifoldsError>
where
    T: ManifoldsFloat,
{
    let n = embd.len();
    if n == 0 {
        return Err(ManifoldsError::NoData);
    }
    let verbosity = parse_verbosity_level(verbose);
    let n_dim = embd[0].len();

    let mut embd_flat: Vec<T> = embd.iter().flatten().copied().collect();
    let mut m = vec![T::zero(); n * n_dim];
    let mut v = vec![T::zero(); n * n_dim];
    let mut grads = vec![T::zero(); n * n_dim];
    let c_mn = T::from_f64(C_MID_NEAR).unwrap();
    let w_fp = T::from_f64(W_FP).unwrap();

    let near_own = localmap.map(|_| own_rows(&pairs.near, n));
    let mut further_own = localmap.map(|_| own_rows(&pairs.further, n));
    let mut further = pairs.further.clone();

    let mut beta1t = params.beta1;
    let mut beta2t = params.beta2;

    for epoch in 0..params.n_epochs {
        let (w_nb, w_mn, _) = phase_weights::<T>(epoch, params.phase1_end, params.phase2_end);
        let phase3 = epoch >= params.phase2_end;
        let half_max = localmap.filter(|_| phase3).map(|l| l.half_max_distance);
        let adam = AdamStep::new(params, beta1t, beta2t);

        grads.fill(T::zero());
        let near_c = |d2| near_coeff(d2, w_nb, half_max);
        scatter(&embd_flat, n_dim, &pairs.near, near_c, &mut grads);
        if w_mn > T::zero() {
            let mn_c = |d2| -(w_mn * attract_grad_coeff(d2, c_mn));
            scatter(&embd_flat, n_dim, &pairs.mid_near, mn_c, &mut grads);
        }
        let fp_c = |d2| w_fp * repel_grad_coeff(d2);
        scatter(&embd_flat, n_dim, &further, fp_c, &mut grads);

        for ((x, &g), (m_i, v_i)) in embd_flat
            .iter_mut()
            .zip(&grads)
            .zip(m.iter_mut().zip(v.iter_mut()))
        {
            *x = adam.apply(*x, g, m_i, v_i);
        }

        beta1t *= params.beta1;
        beta2t *= params.beta2;

        if let (Some(l), Some(near_own), Some(further_own)) =
            (localmap, near_own.as_ref(), further_own.as_mut())
        {
            if phase3 && epoch % LOCALMAP_RESAMPLE_EVERY == 0 {
                resample_further_nearby(
                    &embd_flat,
                    n_dim,
                    near_own,
                    further_own,
                    l.max_distance_sq,
                    resample_seed(l.seed, epoch, n),
                );
                further = rows_to_pairs(further_own);
            }
        }

        if verbosity.normal_verbosity() && ((epoch + 1) % 50 == 0 || epoch + 1 == params.n_epochs) {
            println!("  Epoch {}/{}", epoch + 1, params.n_epochs);
        }
    }

    for (point, row) in embd.iter_mut().zip(embd_flat.chunks_exact(n_dim)) {
        point.copy_from_slice(row);
    }
    Ok(())
}

/// Shared body of the parallel PaCMAP and LocalMAP optimisers.
///
/// Lays the pairs out per node as CSR, then runs [`fused_epoch`] between two
/// embedding buffers that swap every epoch.
///
/// ### Params
///
/// * `embd` - Embedding, shape `[n_samples][n_dim]`. Modified in place.
/// * `pairs` - The three pair sets from `construct_pacmap_pairs`.
/// * `params` - Optimisation parameters.
/// * `localmap` - LocalMAP's phase-3 settings, `None` for PaCMAP.
/// * `verbose` - Verbosity level.
///
/// ### Returns
///
/// `Ok(())`, `NoData` on an empty embedding, or a CSR size error.
fn parallel_impl<T>(
    embd: &mut [Vec<T>],
    pairs: &PacmapPairs,
    params: &PacmapOptimParams<T>,
    localmap: Option<LocalMapPhase3<T>>,
    verbose: usize,
) -> Result<(), ManifoldsError>
where
    T: ManifoldsFloat,
{
    let n = embd.len();
    if n == 0 {
        return Err(ManifoldsError::NoData);
    }
    let verbosity = parse_verbosity_level(verbose);
    let n_dim = embd[0].len();

    let mut old: Vec<T> = embd.iter().flatten().copied().collect();
    let mut new = vec![T::zero(); n * n_dim];
    let mut m = vec![T::zero(); n * n_dim];
    let mut v = vec![T::zero(); n * n_dim];

    let near = PairCsr::from_pairs(&pairs.near, n)?;
    let mid_near = PairCsr::from_pairs(&pairs.mid_near, n)?;
    let mut further = PairCsr::from_pairs(&pairs.further, n)?;

    let near_own = localmap.map(|_| own_rows(&pairs.near, n));
    let mut further_own = localmap.map(|_| own_rows(&pairs.further, n));

    let mut beta1t = params.beta1;
    let mut beta2t = params.beta2;

    for epoch in 0..params.n_epochs {
        let (w_nb, w_mn, _) = phase_weights::<T>(epoch, params.phase1_end, params.phase2_end);
        let phase3 = epoch >= params.phase2_end;
        let half_max = localmap.filter(|_| phase3).map(|l| l.half_max_distance);
        let adam = AdamStep::new(params, beta1t, beta2t);

        fused_epoch(
            &old,
            &mut new,
            &mut m,
            &mut v,
            n_dim,
            &near,
            (w_mn > T::zero()).then_some((&mid_near, w_mn)),
            &further,
            w_nb,
            half_max,
            adam,
        );
        std::mem::swap(&mut old, &mut new);

        beta1t *= params.beta1;
        beta2t *= params.beta2;

        if let (Some(l), Some(near_own), Some(further_own)) =
            (localmap, near_own.as_ref(), further_own.as_mut())
        {
            if phase3 && epoch % LOCALMAP_RESAMPLE_EVERY == 0 {
                resample_further_nearby(
                    &old,
                    n_dim,
                    near_own,
                    further_own,
                    l.max_distance_sq,
                    resample_seed(l.seed, epoch, n),
                );
                further = PairCsr::from_rows(further_own)?;
            }
        }

        if verbosity.normal_verbosity() && ((epoch + 1) % 50 == 0 || epoch + 1 == params.n_epochs) {
            println!("  Epoch {}/{}", epoch + 1, params.n_epochs);
        }
    }

    for (point, row) in embd.iter_mut().zip(old.chunks_exact(n_dim)) {
        point.copy_from_slice(row);
    }
    Ok(())
}

/// Optimise a PaCMAP embedding using Adam.
///
/// ### Params
///
/// * `embd` - Embedding, shape `[n_samples][n_dim]`. Modified in place.
/// * `pairs` - The three pair sets from `construct_pacmap_pairs`.
/// * `params` - Optimisation parameters.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `Ok(())` or `NoData` on an empty embedding.
pub fn optimise_pacmap<T>(
    embd: &mut [Vec<T>],
    pairs: &PacmapPairs,
    params: &PacmapOptimParams<T>,
    verbose: usize,
) -> Result<(), ManifoldsError>
where
    T: ManifoldsFloat,
{
    sequential_impl(embd, pairs, params, None, verbose)
}

/// Optimise a PaCMAP embedding using Adam (parallel).
///
/// Each node gathers its gradient over its own pairs and applies Adam in the
/// same pass, writing to a second embedding buffer.
///
/// ### Params
///
/// * `embd` - Embedding, shape `[n_samples][n_dim]`. Modified in place.
/// * `pairs` - The three pair sets from `construct_pacmap_pairs`.
/// * `params` - Optimisation parameters.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `Ok(())`, `NoData` on an empty embedding, or a CSR size error.
pub fn optimise_pacmap_parallel<T>(
    embd: &mut [Vec<T>],
    pairs: &PacmapPairs,
    params: &PacmapOptimParams<T>,
    verbose: usize,
) -> Result<(), ManifoldsError>
where
    T: ManifoldsFloat,
{
    parallel_impl(embd, pairs, params, None, verbose)
}

/// Optimise a LocalMAP embedding using Adam.
///
/// Phases 1 and 2 are PaCMAP's. In phase 3 the mid-near pairs drop out, the
/// near force is scaled by `0.5 * max_distance / sqrt(1 + d)`, and every
/// 10 epochs the further pairs are redrawn among the
/// points within `max_distance` in the embedding.
///
/// ### Params
///
/// * `embd` - Embedding, shape `[n_samples][n_dim]`. Modified in place.
/// * `pairs` - The three pair sets from `construct_pacmap_pairs`.
/// * `params` - Optimisation parameters.
/// * `max_distance` - Embedding distance cut-off for further pairs.
/// * `seed` - Seed of the further-pair resampler.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `Ok(())` or `NoData` on an empty embedding.
///
/// ### References
///
/// Wang et al., arXiv:2412.15426, 2024
pub fn optimise_localmap<T>(
    embd: &mut [Vec<T>],
    pairs: &PacmapPairs,
    params: &PacmapOptimParams<T>,
    max_distance: T,
    seed: u64,
    verbose: usize,
) -> Result<(), ManifoldsError>
where
    T: ManifoldsFloat,
{
    let l = localmap_phase3(max_distance, seed);
    sequential_impl(embd, pairs, params, Some(l), verbose)
}

/// Optimise a LocalMAP embedding using Adam (parallel).
///
/// As [`optimise_localmap`], with the fused per-node kernel of
/// [`optimise_pacmap_parallel`].
///
/// ### Params
///
/// * `embd` - Embedding, shape `[n_samples][n_dim]`. Modified in place.
/// * `pairs` - The three pair sets from `construct_pacmap_pairs`.
/// * `params` - Optimisation parameters.
/// * `max_distance` - Embedding distance cut-off for further pairs.
/// * `seed` - Seed of the further-pair resampler.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `Ok(())`, `NoData` on an empty embedding, or a CSR size error.
///
/// ### References
///
/// Wang et al., arXiv:2412.15426, 2024
pub fn optimise_localmap_parallel<T>(
    embd: &mut [Vec<T>],
    pairs: &PacmapPairs,
    params: &PacmapOptimParams<T>,
    max_distance: T,
    seed: u64,
    verbose: usize,
) -> Result<(), ManifoldsError>
where
    T: ManifoldsFloat,
{
    let l = localmap_phase3(max_distance, seed);
    parallel_impl(embd, pairs, params, Some(l), verbose)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_pacmap_optimiser {
    use super::*;
    use crate::data::pacmap_pairs::PacmapPairs;

    fn simple_pairs(n: usize) -> PacmapPairs {
        let near = (0..n).map(|i| (i, (i + 1) % n)).collect();
        let mid_near = (0..n).map(|i| (i, (i + 2) % n)).collect();
        let further = (0..n).map(|i| (i, (i + n / 2) % n)).collect();
        PacmapPairs {
            near,
            mid_near,
            further,
        }
    }

    fn simple_embd(n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|i| vec![i as f64 * 2.0, 0.0]).collect()
    }

    fn default_params() -> PacmapOptimParams<f64> {
        PacmapOptimParams::new(Some(50), None, None, None, None, None, None)
    }

    fn total_movement(before: &[Vec<f64>], after: &[Vec<f64>]) -> f64 {
        before
            .iter()
            .zip(after.iter())
            .flat_map(|(b, a)| b.iter().zip(a.iter()).map(|(&x, &y)| (x - y).abs()))
            .sum()
    }

    #[test]
    fn test_sequential_moves_points() {
        let pairs = simple_pairs(6);
        let mut embd = simple_embd(6);
        let initial = embd.clone();
        let _ = optimise_pacmap(&mut embd, &pairs, &default_params(), 0);
        assert!(total_movement(&initial, &embd) > 0.01);
    }

    #[test]
    fn test_parallel_moves_points() {
        let pairs = simple_pairs(6);
        let mut embd = simple_embd(6);
        let initial = embd.clone();
        let _ = optimise_pacmap_parallel(&mut embd, &pairs, &default_params(), 0);
        assert!(total_movement(&initial, &embd) > 0.01);
    }

    #[test]
    fn test_sequential_all_finite() {
        let pairs = simple_pairs(8);
        let mut embd = simple_embd(8);
        let _ = optimise_pacmap(&mut embd, &pairs, &default_params(), 0);
        for point in &embd {
            for &coord in point {
                assert!(coord.is_finite(), "non-finite coordinate: {}", coord);
            }
        }
    }

    #[test]
    fn test_parallel_all_finite() {
        let pairs = simple_pairs(8);
        let mut embd = simple_embd(8);
        let _ = optimise_pacmap_parallel(&mut embd, &pairs, &default_params(), 0);
        for point in &embd {
            for &coord in point {
                assert!(coord.is_finite(), "non-finite coordinate: {}", coord);
            }
        }
    }

    #[test]
    fn test_sequential_reproducible() {
        let pairs = simple_pairs(6);
        let params = default_params();
        let mut embd1 = simple_embd(6);
        let mut embd2 = simple_embd(6);
        let _ = optimise_pacmap(&mut embd1, &pairs, &params, 0);
        let _ = optimise_pacmap(&mut embd2, &pairs, &params, 0);
        assert_eq!(embd1, embd2);
    }

    #[test]
    fn test_parallel_reproducible() {
        let pairs = simple_pairs(6);
        let params = default_params();
        let mut embd1 = simple_embd(6);
        let mut embd2 = simple_embd(6);
        let _ = optimise_pacmap_parallel(&mut embd1, &pairs, &params, 0);
        let _ = optimise_pacmap_parallel(&mut embd2, &pairs, &params, 0);
        assert_eq!(embd1, embd2);
    }

    #[test]
    fn test_empty_embedding_does_not_panic() {
        let pairs = PacmapPairs {
            near: vec![],
            mid_near: vec![],
            further: vec![],
        };
        let mut embd: Vec<Vec<f64>> = vec![];
        let _ = optimise_pacmap(&mut embd, &pairs, &default_params(), 0);
        let _ = optimise_pacmap_parallel(&mut embd, &pairs, &default_params(), 0);
    }

    #[test]
    fn test_phase_weights_respected() {
        // run for only phase 1 epochs and check mid-near pairs drive movement
        // by comparing against a run with zero mid-near pairs
        let n = 10;
        let pairs_full = simple_pairs(n);
        let pairs_no_mn = PacmapPairs {
            near: pairs_full.near.clone(),
            mid_near: vec![],
            further: pairs_full.further.clone(),
        };

        let params = PacmapOptimParams::new(Some(50), None, None, None, None, None, None);

        let mut embd_full = simple_embd(n);
        let mut embd_no_mn = simple_embd(n);
        let initial = simple_embd(n);

        let _ = optimise_pacmap(&mut embd_full, &pairs_full, &params, 0);
        let _ = optimise_pacmap(&mut embd_no_mn, &pairs_no_mn, &params, 0);

        let movement_full = total_movement(&initial, &embd_full);
        let movement_no_mn = total_movement(&initial, &embd_no_mn);

        // mid-near pairs contribute additional force so movement should differ
        assert!(
            (movement_full - movement_no_mn).abs() > 1e-6,
            "mid-near pairs had no effect: full={:.4}, no_mn={:.4}",
            movement_full,
            movement_no_mn
        );
    }

    #[test]
    fn test_sequential_and_parallel_broadly_agree() {
        // both should converge to similar embeddings from the same start,
        // not necessarily identical due to gradient accumulation order
        let pairs = simple_pairs(10);
        let params = PacmapOptimParams::new(Some(200), None, None, None, None, None, None);

        let mut embd_seq = simple_embd(10);
        let mut embd_par = simple_embd(10);

        let _ = optimise_pacmap(&mut embd_seq, &pairs, &params, 0);
        let _ = optimise_pacmap_parallel(&mut embd_par, &pairs, &params, 0);

        let diff = total_movement(&embd_seq, &embd_par);
        let scale = total_movement(&simple_embd(10), &embd_seq);

        assert!(
            diff < scale * 0.1,
            "sequential and parallel diverged too much: diff={:.4}, scale={:.4}",
            diff,
            scale
        );
    }

    #[test]
    fn test_resample_respects_distance_near_and_duplicates() {
        // 1-D line, spacing 1: within distance 3 of node 10 lie 7..=13
        let n = 30;
        let embd: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut near_own = vec![Vec::new(); n];
        near_own[10] = vec![9, 11];
        let mut further_own = vec![vec![0, 1, 2]; n];
        resample_further_nearby(&embd, 1, &near_own, &mut further_own, 9.0, 42);

        let row = &further_own[10];
        for (s, &j) in row.iter().enumerate() {
            assert!(j != 10 && j != 9 && j != 11, "self or near pair kept: {j}");
            assert!((7..=13).contains(&j), "pair {j} beyond max_distance");
            assert!(!row[..s].contains(&j), "duplicate pair {j}");
        }
    }

    #[test]
    fn test_resample_keeps_old_pair_when_nothing_qualifies() {
        // everyone else is far away, so every draw is rejected
        let n = 20;
        let embd: Vec<f64> = (0..n).map(|i| 100.0 * i as f64).collect();
        let near_own = vec![Vec::new(); n];
        let mut further_own = vec![vec![5, 6]; n];
        resample_further_nearby(&embd, 1, &near_own, &mut further_own, 1.0, 42);
        assert!(further_own.iter().all(|r| r == &vec![5, 6]));
    }

    #[test]
    fn test_localmap_reproducible_and_finite() {
        let pairs = simple_pairs(20);
        let params = PacmapOptimParams::new(Some(260), None, None, None, None, None, None);
        let mut a = simple_embd(20);
        let mut b = simple_embd(20);
        optimise_localmap_parallel(&mut a, &pairs, &params, 10.0, 7, 0).unwrap();
        optimise_localmap_parallel(&mut b, &pairs, &params, 10.0, 7, 0).unwrap();
        assert_eq!(a, b);
        assert!(a.iter().flatten().all(|x| x.is_finite()));
    }

    #[test]
    fn test_general_path_matches_2d_path() {
        // a zero third axis gets zero gradient and zero Adam step and adds
        // exact zeros to every distance, so the general n_dim path must
        // reproduce the 2D path bit for bit
        let n = 40;
        let pairs = PacmapPairs {
            near: (0..n).map(|i| (i, (i * 7 + 1) % n)).collect(),
            mid_near: (0..n).map(|i| (i, (i * 11 + 3) % n)).collect(),
            further: (0..n).map(|i| (i, (i * 13 + 5) % n)).collect(),
        };
        let params = PacmapOptimParams::new(Some(260), None, None, None, None, None, None);
        let embd2: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![(i as f64 * 0.37).sin(), (i as f64 * 0.91).cos()])
            .collect();
        let embd3: Vec<Vec<f64>> = embd2.iter().map(|p| vec![p[0], p[1], 0.0]).collect();

        type Run = fn(&mut [Vec<f64>], &PacmapPairs, &PacmapOptimParams<f64>);
        let runs: [Run; 4] = [
            |e, p, o| optimise_pacmap(e, p, o, 0).unwrap(),
            |e, p, o| optimise_pacmap_parallel(e, p, o, 0).unwrap(),
            |e, p, o| optimise_localmap(e, p, o, 10.0, 7, 0).unwrap(),
            |e, p, o| optimise_localmap_parallel(e, p, o, 10.0, 7, 0).unwrap(),
        ];
        for run in runs {
            let mut a = embd2.clone();
            let mut b = embd3.clone();
            run(&mut a, &pairs, &params);
            run(&mut b, &pairs, &params);
            for (pa, pb) in a.iter().zip(&b) {
                assert_eq!(pa[..], pb[..2]);
                assert_eq!(pb[2], 0.0);
            }
        }
    }
}
