//! Optimisers for tSNE fitting. Contains the BarnesHut version from Laurens
//! van der Maaten and the FFT-accelerated Interpolation-based version of tSNE
//! from Linderman et al.

use num_traits::{Float, FromPrimitive};
use rayon::prelude::*;
use thousands::*;

use crate::data::graph::coo_to_adjacency_list;
use crate::data::structures::*;
use crate::prelude::*;
use crate::utils::bh_tree::*;
use crate::utils::density::*;
#[cfg(any(feature = "fft_tsne", feature = "gpu"))]
use crate::utils::math::choose_grid_size;

#[cfg(feature = "fft_tsne")]
use crate::utils::fft::*;

//////////
// tSNE //
//////////

/////////////
// Globals //
/////////////

/// Iteration from when on to switch the tSNE momentum
pub(crate) const TSNE_MOMENTUM_SWITCH_ITER: usize = 250;

/// Initial tSNE momentum
pub(crate) const TSNE_INITIAL_MOMENTUM: f64 = 0.5;

/// Final tSNE momentum
pub(crate) const TSNE_FINAL_MOMENTUM: f64 = 0.8;

/// Minimum tSNE gain
pub(crate) const TSNE_MIN_GAIN: f64 = 0.01;

/// tSNE epsilon
pub(crate) const TSNE_EPS: f64 = 1e-12;

/// Per-point step cap as a fraction of `lr`, floored at `TSNE_MAX_STEP_FLOOR`.
/// The Belkina lr scales as N/12, so a fixed cap forces every step at large N
/// onto the cap and erases gradient direction information; scaling the cap
/// with lr preserves it. At the lr floor (N small), the cap equals 5.
const TSNE_MAX_STEP_FRACTION: f64 = 0.025;
const TSNE_MAX_STEP_FLOOR: f64 = 5.0;

/// Divisor for the default learning rate heuristic (`lr = N / this`).
const TSNE_LR_DIVISOR: f64 = 12.0;

/// Floor for the default learning rate heuristic.
const TSNE_LR_FLOOR: f64 = 200.0;

/// Default tree depth cap for the quick-and-dirty Barnes-Hut optimiser.
/// qdtsne recommends 7 to 10. At n = 20k (five clusters, f32) depth 7 halved
/// the full pipeline time against plain Barnes-Hut (4.0 s vs 8.0 s) with
/// kNN15 preservation 0.223 vs 0.231.
pub(crate) const TSNE_QD_MAX_DEPTH: usize = 7;

/// Cap on `n_boxes` per dimension in the FFT grid. FFT cost per epoch is
/// O(n_boxes^2 log n_boxes); without a cap, n_boxes grows with the embedding
/// span and per-epoch cost blows up. Box width adapts upward once this cap
/// binds, keeping the grid covering the embedding.
#[cfg(any(feature = "fft_tsne", feature = "gpu"))]
pub(crate) const TSNE_FFT_MAX_BOXES: usize = 140;

/// Lower bound on the FFT box width (the original fixed value).
#[cfg(any(feature = "fft_tsne", feature = "gpu"))]
pub(crate) const TSNE_FFT_MIN_BOX_WIDTH: f64 = 1.0;

/// Minimum number of FFT boxes per dimension (FIt-SNE default).
#[cfg(any(feature = "fft_tsne", feature = "gpu"))]
pub(crate) const TSNE_FFT_MIN_INTERVALS: usize = 50;

/// Headroom added to the grid bounds once the box-cap regime is active, so
/// the embedding can move between rebuilds.
#[cfg(any(feature = "fft_tsne", feature = "gpu"))]
pub(crate) const TSNE_FFT_GRID_MARGIN: f64 = 0.3;

////////////////
// Structures //
////////////////

/// t-SNE specific optimization parameters
#[derive(Clone, Debug)]
pub struct TsneOptimParams<T> {
    /// Number of epochs
    pub n_epochs: usize,
    /// Optional learning rate. Defaults to `(N / 12).max(200)`, the
    /// FIt-SNE/Belkina N-invariant heuristic.
    pub lr: Option<T>,
    /// Early exaggeration iters
    pub early_exag_iter: usize,
    /// The factor to exaggerate in the early iterations
    pub early_exag_factor: T,
    /// Optional late stage exaggeration factor. For N >= ~100k a value of ~4
    /// is typically needed to preserve cluster structure that would otherwise
    /// disperse after early exaggeration ends.
    pub late_exag_factor: Option<T>,
    /// The Barnes-Hut theta; relevant if you use `optimise_bh_tsne()`
    pub theta: T,
    /// Interpolation points per box (typically 3); relevant for FFT path.
    pub n_interp_points: usize,
    /// Maximum Barnes-Hut tree depth; relevant for the quick-and-dirty
    /// optimiser (`optimise_qd_tsne()`) only. qdtsne recommends 7 to 10.
    pub max_depth: usize,
}

impl<T> TsneOptimParams<T>
where
    T: Float + FromPrimitive,
{
    /// Generate a new instance
    ///
    /// ### Params
    ///
    /// * `n_epochs` - Number of epochs
    /// * `lr` - Learning rate; `None` uses `(N / 12).max(200)`
    /// * `early_exag_iter` - Number of early exaggeration epochs
    /// * `early_exag_factor` - Early exaggeration factor
    /// * `late_exag_factor` - Optional late exaggeration factor
    /// * `theta` - Barnes-Hut opening parameter
    /// * `n_interp_points` - FFT interpolation points per box; defaults to 3
    /// * `max_depth` - Tree depth cap for the quick-and-dirty optimiser;
    ///   defaults to `TSNE_QD_MAX_DEPTH` (7)
    ///
    /// ### Returns
    ///
    /// The optimiser parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_epochs: usize,
        lr: Option<T>,
        early_exag_iter: usize,
        early_exag_factor: T,
        late_exag_factor: Option<T>,
        theta: T,
        n_interp_points: Option<usize>,
        max_depth: Option<usize>,
    ) -> Self {
        let n_interp_points = n_interp_points.unwrap_or(3);
        let max_depth = max_depth.unwrap_or(TSNE_QD_MAX_DEPTH);

        Self {
            n_epochs,
            lr,
            early_exag_iter,
            early_exag_factor,
            late_exag_factor,
            theta,
            n_interp_points,
            max_depth,
        }
    }

    /// Return the learning rate (explicit, or N-invariant heuristic).
    pub fn get_lr(&self, n_samples: usize) -> T {
        self.lr.unwrap_or_else(|| {
            T::from_f64((n_samples as f64 / TSNE_LR_DIVISOR).max(TSNE_LR_FLOOR)).unwrap()
        })
    }

    /// Late exaggeration factor (defaults to 1.0 when unset).
    pub fn get_late_exag_factor(&self) -> T {
        self.late_exag_factor.unwrap_or(T::one())
    }
}

impl<T> Default for TsneOptimParams<T>
where
    T: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            n_epochs: 1000,
            lr: None,
            early_exag_iter: 250,
            early_exag_factor: T::from_f64(12.0).unwrap(),
            late_exag_factor: None,
            theta: T::from_f64(0.5).unwrap(),
            n_interp_points: 3,
            max_depth: TSNE_QD_MAX_DEPTH,
        }
    }
}

///////////////
// Optimiser //
///////////////

/// Type of optimisation to use for tSNE.
#[derive(Default)]
pub enum TsneOpt {
    /// FFT-accelerated version. Requires the `fft_tsne` feature, which pulls in
    /// FFTW as a system dependency.
    Fft,
    /// BarnesHut-accelerated version. The default, as it is the only variant
    /// available in every build configuration on every platform.
    #[default]
    BarnesHut,
    /// FFT-accelerated version with three kernels (`q`, `q^2 dx`, `q^2 dy`)
    /// against a unit charge instead of the 4-term expansion. One forward and
    /// three inverse transforms instead of four each. Requires `fft_tsne`.
    Fft3Kernel,
    /// Device-resident three-kernel FFT version. Only through `tsne_gpu`;
    /// requires `gpu` but not `fft_tsne`.
    Fft3KernelGpu,
    /// Quick-and-dirty Barnes-Hut (qdtsne): tree depth capped at
    /// `max_depth` plus the leaf approximation, so repulsion is computed once
    /// per leaf rather than once per point.
    BarnesHutQd,
}

/// Parse the tSNE optimiser to use.
///
/// ### Params
///
/// * `s` - String defining the optimiser. Accepts `"barnes hut"`,
///   `"barnes_hut"`, `"barnes-hut"`, `"bh"`, `"fft"`, `"fft_3k"` /
///   `"3-kernel"`, `"fft_3k_gpu"`, or `"bh_qd"` / `"qd"`.
///
/// ### Returns
///
/// `Some(TsneOpt)` if the string matches a known optimiser, `None` otherwise.
pub fn parse_tsne_optimiser(s: &str) -> Option<TsneOpt> {
    match s.to_lowercase().as_str() {
        "barnes hut" | "barnes_hut" | "barnes-hut" | "bh" => Some(TsneOpt::BarnesHut),
        "fft" => Some(TsneOpt::Fft),
        "fft_3k" | "3-kernel" => Some(TsneOpt::Fft3Kernel),
        "fft_3k_gpu" => Some(TsneOpt::Fft3KernelGpu),
        "bh_qd" | "qd" => Some(TsneOpt::BarnesHutQd),
        _ => None,
    }
}

/////////////
// Helpers //
/////////////

/// Adaptive gain update for a single t-SNE parameter (van der Maaten
/// convention).
///
/// Gain increases by 0.2 when the gradient and update disagree in sign,
/// decays by a factor of 0.8 otherwise, and is floored at `min_gain`.
///
/// ### Params
///
/// * `val` - The parameter value to update (modified in place).
/// * `update` - The momentum buffer for this parameter (modified in place).
/// * `gain` - The adaptive gain for this parameter (modified in place).
/// * `grad` - The gradient at the current step.
/// * `lr` - Learning rate.
/// * `momentum` - Momentum coefficient.
/// * `min_gain` - Lower bound on the gain.
#[inline(always)]
fn update_parameter<T>(
    val: &mut T,
    update: &mut T,
    gain: &mut T,
    grad: T,
    lr: T,
    momentum: T,
    min_gain: T,
) where
    T: ManifoldsFloat,
{
    if (grad > T::zero()) != (*update > T::zero()) {
        *gain += T::from_f64(0.2).unwrap();
    } else {
        *gain *= T::from_f64(0.8).unwrap();
    }
    *gain = (*gain).max(min_gain);

    *update = momentum * *update - lr * *gain * grad;
    *val += *update;
}

/// Clip a 2D momentum step to `max_step_norm` and rewrite the point in place.
///
/// If the Euclidean norm of `(u0, u1)` exceeds `max_step_norm`, both
/// components are scaled down proportionally and the point coordinates are
/// recomputed from the previous position.
///
/// ### Params
///
/// * `point` - Mutable slice of length 2 holding the current position
///   (modified in place).
/// * `u0` - x-component of the momentum update (modified in place).
/// * `u1` - y-component of the momentum update (modified in place).
/// * `prev_x` - x-coordinate before the update.
/// * `prev_y` - y-coordinate before the update.
/// * `max_step_norm` - Maximum permitted Euclidean step length.
#[inline(always)]
fn clip_step<T>(point: &mut [T], u0: &mut T, u1: &mut T, prev_x: T, prev_y: T, max_step_norm: T)
where
    T: ManifoldsFloat,
{
    let step_sq = *u0 * *u0 + *u1 * *u1;
    let max_sq = max_step_norm * max_step_norm;
    if step_sq > max_sq {
        let scale = max_step_norm / step_sq.sqrt();
        *u0 *= scale;
        *u1 *= scale;
        point[0] = prev_x + *u0;
        point[1] = prev_y + *u1;
    }
}

/// Compute the per-point step cap from the learning rate.
///
/// Returns `lr * TSNE_MAX_STEP_FRACTION`, floored at `TSNE_MAX_STEP_FLOOR`.
/// Scaling with `lr` rather than using a fixed cap preserves gradient
/// direction at large N where the Belkina heuristic makes `lr` large.
///
/// ### Params
///
/// * `lr` - The learning rate in use.
///
/// ### Returns
///
/// Maximum permitted Euclidean step length per point per epoch.
#[inline]
pub(crate) fn step_cap_from_lr<T: ManifoldsFloat>(lr: T) -> T {
    let lr_f64 = lr.to_f64().unwrap();
    T::from_f64((lr_f64 * TSNE_MAX_STEP_FRACTION).max(TSNE_MAX_STEP_FLOOR)).unwrap()
}

/// Recentre the embedding on the origin.
///
/// Subtracts the per-coordinate mean from every point. The mean is
/// accumulated in `f64` to avoid precision loss at large N when `T` is
/// `f32`. No parallel sum to avoid issues with non-reproducibility given the
/// same seed.
///
/// ### Params
///
/// * `embd` - Mutable slice of 2D points (modified in place).
fn recentre_embedding<T: ManifoldsFloat>(embd: &mut [Vec<T>]) {
    let n = embd.len();
    if n == 0 {
        return;
    }

    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    for p in embd.iter() {
        sum_x += p[0].to_f64().unwrap();
        sum_y += p[1].to_f64().unwrap();
    }

    let n_f64 = n as f64;
    let mean_x = T::from_f64(sum_x / n_f64).unwrap();
    let mean_y = T::from_f64(sum_y / n_f64).unwrap();

    embd.par_iter_mut().for_each(|p| {
        p[0] -= mean_x;
        p[1] -= mean_y;
    });
}

/////////////////////////
// Density (den-SNE)   //
/////////////////////////

/// Per-epoch scratch buffers for the den-SNE density term.
///
/// Allocated once per run, and only when the density term is in use, since
/// these are three extra `[n]` buffers on top of the embedding.
struct DensScratch<T> {
    /// `sum_j phi_ij * ||y_i - y_j||^2` per node.
    re_acc: Vec<T>,
    /// `sum_j phi_ij` per node, the embedding-radius denominator.
    phi_sum: Vec<T>,
    /// `log(eps + re_acc / phi_sum)` per node.
    re: Vec<T>,
}

impl<T> DensScratch<T>
where
    T: ManifoldsFloat,
{
    /// Allocate zeroed buffers for `n` points.
    ///
    /// ### Params
    ///
    /// * `n` - Number of points
    ///
    /// ### Returns
    ///
    /// Zeroed scratch buffers.
    fn new(n: usize) -> Self {
        Self {
            re_acc: vec![T::zero(); n],
            phi_sum: vec![T::zero(); n],
            re: vec![T::zero(); n],
        }
    }
}

/// Epoch-constant scalars of the density gradient.
struct DensGradConsts<T> {
    /// `1 / re_std`.
    w1: T,
    /// `cov / re_std^3`.
    w2: T,
    /// Mean of the embedding log-radii.
    re_mean: T,
    /// `lambda / (n - 1)`, folded in once rather than per edge.
    scale: T,
}

impl<T> DensGradConsts<T>
where
    T: ManifoldsFloat,
{
    /// Derive the epoch constants from the current embedding radii.
    ///
    /// ### Params
    ///
    /// * `re` - Embedding log-radii, `[n]`
    /// * `state` - Constant density state, holding the original radii
    ///
    /// ### Returns
    ///
    /// The scalars used by [`density_gradient`].
    fn new(re: &[T], state: &DensState<T>) -> Self {
        let (re_mean, re_std, cov) = correlation_stats(re, &state.r, state.params.var_shift);
        let n = re.len();
        let denom = T::from_usize(n.saturating_sub(1).max(1)).unwrap();

        Self {
            w1: T::one() / re_std,
            w2: cov / (re_std * re_std * re_std),
            re_mean,
            scale: state.params.lambda / denom,
        }
    }
}

/// Attractive forces for every point, without the density radii.
///
/// ### Params
///
/// * `adj` - Row-major adjacency of the symmetric affinity graph
/// * `pos` - Interleaved positions `[x0, y0, x1, y1, ...]`
/// * `exag_factor` - Current exaggeration factor
/// * `attr` - Output, interleaved attractive force per point, overwritten
fn accumulate_attractive<T>(adj: &[Vec<(usize, T)>], pos: &[T], exag_factor: T, attr: &mut [T])
where
    T: ManifoldsFloat,
{
    attr.par_chunks_exact_mut(2)
        .enumerate()
        .for_each(|(i, out)| {
            let px = pos[2 * i];
            let py = pos[2 * i + 1];

            let mut attr_x = T::zero();
            let mut attr_y = T::zero();
            for &(j, p_val) in &adj[i] {
                let dx = px - pos[2 * j];
                let dy = py - pos[2 * j + 1];
                let dist_sq = dx * dx + dy * dy;
                let q = T::one() / (T::one() + dist_sq);
                let force = p_val * exag_factor * q;
                attr_x += force * dx;
                attr_y += force * dy;
            }

            out[0] = attr_x;
            out[1] = attr_y;
        });
}

/// Attractive forces and embedding local radii in one sweep.
///
/// The Student-t kernel `phi = 1/(1 + dsq)` is already the attractive term's
/// `q_ij`, so the radii ride along on the same loads. Deliberately a separate
/// function from [`accumulate_attractive`] rather than a flag, so the plain
/// tSNE path pays neither the extra arithmetic nor the extra `[n]` buffers.
///
/// Note the radii use `phi` alone: exaggeration scales the attractive force but
/// must not enter the density term, matching the reference.
///
/// ### Params
///
/// * `adj` - Row-major adjacency of the symmetric affinity graph
/// * `pos` - Interleaved positions `[x0, y0, x1, y1, ...]`
/// * `exag_factor` - Current exaggeration factor
/// * `attr` - Output, interleaved attractive force per point, overwritten
/// * `scratch` - Output, radii accumulators and log-radii, overwritten
fn accumulate_attractive_and_radii<T>(
    adj: &[Vec<(usize, T)>],
    pos: &[T],
    exag_factor: T,
    attr: &mut [T],
    scratch: &mut DensScratch<T>,
) where
    T: ManifoldsFloat,
{
    attr.par_chunks_exact_mut(2)
        .zip(scratch.re_acc.par_iter_mut())
        .zip(scratch.phi_sum.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((out, re_acc), phi_sum))| {
            let px = pos[2 * i];
            let py = pos[2 * i + 1];

            let mut attr_x = T::zero();
            let mut attr_y = T::zero();
            let mut sum_sq = T::zero();
            let mut sum_phi = T::zero();

            for &(j, p_val) in &adj[i] {
                let dx = px - pos[2 * j];
                let dy = py - pos[2 * j + 1];
                let dist_sq = dx * dx + dy * dy;
                let q = T::one() / (T::one() + dist_sq);
                let force = p_val * exag_factor * q;
                attr_x += force * dx;
                attr_y += force * dy;

                sum_sq += q * dist_sq;
                sum_phi += q;
            }

            out[0] = attr_x;
            out[1] = attr_y;
            *re_acc = sum_sq;
            *phi_sum = sum_phi;
        });

    embedding_log_radii(&scratch.re_acc, &scratch.phi_sum, &mut scratch.re);
}

/// Density gradient contribution for one point.
///
/// Both endpoints of every edge contribute: moving `y_i` changes `i`'s own
/// radius and each neighbour's, so a single sweep over `i`'s row picks up the
/// whole term. With `phi = 1/(1 + dsq)` the radius derivative collapses to
/// `phi^2 * (1 + exp(-re)) / phi_sum`, the `a = b = 1` case of the densMAP
/// expression.
///
/// ### Params
///
/// * `i` - Point index
/// * `adj_i` - `i`'s row of the adjacency; only the neighbour indices are used
/// * `pos` - Interleaved positions `[x0, y0, x1, y1, ...]`
/// * `r` - Z-scored original log-radii, `[n]`
/// * `scratch` - Embedding radii for the current epoch
/// * `consts` - Epoch-constant scalars
///
/// ### Returns
///
/// The `(x, y)` density gradient for point `i`, already scaled by
/// `lambda / (n - 1)`. Subtract it from the usual tSNE gradient.
#[inline]
fn density_gradient<T>(
    i: usize,
    adj_i: &[(usize, T)],
    pos: &[T],
    r: &[T],
    scratch: &DensScratch<T>,
    consts: &DensGradConsts<T>,
) -> (T, T)
where
    T: ManifoldsFloat,
{
    let one = T::one();
    let floor = T::epsilon();

    let px = pos[2 * i];
    let py = pos[2 * i + 1];

    let weight_i = consts.w1 * r[i] - consts.w2 * (scratch.re[i] - consts.re_mean);
    let inv_phi_sum_i = one / scratch.phi_sum[i].max(floor);
    let tail_i = one + (-scratch.re[i]).exp();

    let mut grad_x = T::zero();
    let mut grad_y = T::zero();

    for &(j, _) in adj_i {
        let dx = px - pos[2 * j];
        let dy = py - pos[2 * j + 1];
        let phi = one / (one + dx * dx + dy * dy);
        let phi_sq = phi * phi;

        let dr_me = phi_sq * inv_phi_sum_i * tail_i;
        let dr_you = phi_sq / scratch.phi_sum[j].max(floor) * (one + (-scratch.re[j]).exp());

        let weight_j = consts.w1 * r[j] - consts.w2 * (scratch.re[j] - consts.re_mean);
        let g = weight_i * dr_me + weight_j * dr_you;

        grad_x += g * dx;
        grad_y += g * dy;
    }

    (grad_x * consts.scale, grad_y * consts.scale)
}

////////////////
// Barnes Hut //
////////////////

/// Optimise a 2D embedding using Barnes-Hut t-SNE.
///
/// Minimises the KL divergence between high-dimensional affinities (`graph`)
/// and low-dimensional Student-t similarities via gradient descent with
/// momentum and adaptive per-parameter gains.
///
/// ### Params
///
/// * `embd` - Initial embedding coordinates, shape `[n_samples][2]`
///   (modified in place).
/// * `params` - Optimisation hyperparameters (epochs, learning rate, momentum
///   schedule, exaggeration, Barnes-Hut theta).
/// * `graph` - Sparse high-dimensional affinities in coordinate-list format.
/// * `dens` - Density-preserving state for den-SNE, or `None` for plain tSNE.
///   When set, the density gradient is applied over the final
///   `dens.params.frac` of the epochs.
/// * `verbose` - Verbosity level: `0` silent, `1` normal, `2` detailed.
///
/// ### References
///
/// van der Maaten, Journal of Machine Learning Research, 2014 (Barnes-Hut).
/// Narayan, Berger & Cho, Nature Biotechnology, 2021 (den-SNE).
pub fn optimise_bh_tsne<T>(
    embd: &mut [Vec<T>],
    params: &TsneOptimParams<T>,
    graph: &CoordinateList<T>,
    dens: Option<&DensState<T>>,
    verbose: usize,
) where
    T: ManifoldsFloat,
{
    optimise_bh_tsne_impl(embd, params, graph, dens, None, verbose);
}

/// Optimise a 2D embedding using quick-and-dirty Barnes-Hut t-SNE.
///
/// Same optimiser as [`optimise_bh_tsne`], but the tree depth is capped at
/// `params.max_depth` and repulsion uses the qdtsne leaf approximation: one
/// traversal per leaf from its centre of mass, plus each point's
/// interaction with the rest of its own leaf.
///
/// ### Params
///
/// * `embd` - Initial embedding coordinates, shape `[n_samples][2]`
///   (modified in place).
/// * `params` - Optimisation hyperparameters; `theta` and `max_depth` control
///   the approximation.
/// * `graph` - Sparse high-dimensional affinities in coordinate-list format.
/// * `dens` - Density-preserving state for den-SNE, or `None` for plain tSNE.
/// * `verbose` - Verbosity level: `0` silent, `1` normal, `2` detailed.
///
/// ### References
///
/// Lun, qdtsne, 2021 (github.com/libscran/qdtsne).
pub fn optimise_qd_tsne<T>(
    embd: &mut [Vec<T>],
    params: &TsneOptimParams<T>,
    graph: &CoordinateList<T>,
    dens: Option<&DensState<T>>,
    verbose: usize,
) where
    T: ManifoldsFloat,
{
    optimise_bh_tsne_impl(embd, params, graph, dens, Some(params.max_depth), verbose);
}

/// Shared body of [`optimise_bh_tsne`] and [`optimise_qd_tsne`].
///
/// ### Params
///
/// * `embd` - Initial embedding coordinates, shape `[n_samples][2]`
///   (modified in place).
/// * `params` - Optimisation hyperparameters.
/// * `graph` - Sparse high-dimensional affinities in coordinate-list format.
/// * `dens` - Density-preserving state for den-SNE, or `None` for plain tSNE.
/// * `max_depth` - `None` for the uncapped tree with per-point traversals;
///   `Some(d)` for a tree capped at depth `d` with the leaf approximation.
/// * `verbose` - Verbosity level: `0` silent, `1` normal, `2` detailed.
fn optimise_bh_tsne_impl<T>(
    embd: &mut [Vec<T>],
    params: &TsneOptimParams<T>,
    graph: &CoordinateList<T>,
    dens: Option<&DensState<T>>,
    max_depth: Option<usize>,
    verbose: usize,
) where
    T: ManifoldsFloat,
{
    let verbosity = parse_verbosity_level(verbose);

    let n = embd.len();
    let n_dim = embd[0].len();
    let lr = params.get_lr(n);

    let initial_momentum = T::from_f64(TSNE_INITIAL_MOMENTUM).unwrap();
    let final_momentum = T::from_f64(TSNE_FINAL_MOMENTUM).unwrap();
    let min_gain = T::from_f64(TSNE_MIN_GAIN).unwrap();
    let max_step_norm = step_cap_from_lr(lr);

    let mut update_flat = vec![T::zero(); n * n_dim];
    let mut gains_flat = vec![T::one(); n * n_dim];
    let mut pos = vec![T::zero(); n * n_dim];
    let mut rep_forces: Vec<(T, T, T)> = vec![(T::zero(), T::zero(), T::zero()); n];

    // attractive forces are staged rather than fused into the update, so the
    // density term can see every point's radius before any point moves.
    let mut attr = vec![T::zero(); n * n_dim];
    let mut dens_scratch = dens.map(|_| DensScratch::<T>::new(n));

    // one tree across all epochs: rebuild reuses its buffers.
    let mut bh_tree = match max_depth {
        Some(d) => BarnesHutTree::with_max_depth(d),
        None => BarnesHutTree::empty(),
    };
    let mut leaf_forces: Vec<(T, T, T)> = Vec::new();

    let adj = coo_to_adjacency_list(graph);

    for epoch in 0..params.n_epochs {
        let momentum = if epoch < TSNE_MOMENTUM_SWITCH_ITER {
            initial_momentum
        } else {
            final_momentum
        };
        let exag_factor = if epoch < params.early_exag_iter {
            params.early_exag_factor
        } else {
            params.get_late_exag_factor()
        };

        // snapshot positions in parallel into the interleaved flat buffer,
        // shared by the tree build and both force passes.
        embd.par_iter()
            .zip(pos.par_chunks_mut(n_dim))
            .for_each(|(p, dst)| {
                dst[0] = p[0];
                dst[1] = p[1];
            });

        bh_tree.rebuild(&pos, None);

        // compute all repulsive forces in one parallel pass, writing into
        // the preallocated rep_forces buffer
        if max_depth.is_some() {
            bh_tree.compute_leaf_forces(params.theta, &mut leaf_forces);
            rep_forces.par_iter_mut().enumerate().for_each(|(i, slot)| {
                *slot = bh_tree.point_force_from_leaf(i, pos[2 * i], pos[2 * i + 1], &leaf_forces);
            });
        } else {
            rep_forces.par_iter_mut().enumerate().for_each_init(
                || Vec::with_capacity(128),
                |stack, (i, slot)| {
                    *slot = bh_tree.compute_repulsive_force(
                        pos[2 * i],
                        pos[2 * i + 1],
                        params.theta,
                        stack,
                    );
                },
            );
        }

        // global normalisation constant, accumulated in f64 (to avoid weirdness)
        let z_total: f64 = rep_forces
            .iter()
            .map(|r| r.2.to_f64().unwrap())
            .sum::<f64>();
        let z_inv = if z_total > TSNE_EPS {
            T::from_f64(1.0 / z_total).unwrap()
        } else {
            T::zero()
        };

        // attractive forces (exact), plus the embedding radii when the density
        // term is live this epoch.
        let dens_ctx = match (dens, dens_scratch.as_mut()) {
            (Some(state), Some(scratch)) if state.is_active(epoch, params.n_epochs) => {
                accumulate_attractive_and_radii(&adj, &pos, exag_factor, &mut attr, scratch);
                Some((state, &*scratch, DensGradConsts::new(&scratch.re, state)))
            }
            _ => {
                accumulate_attractive(&adj, &pos, exag_factor, &mut attr);
                None
            }
        };

        // parameter update + step clip.
        embd.par_iter_mut()
            .zip(update_flat.par_chunks_mut(n_dim))
            .zip(gains_flat.par_chunks_mut(n_dim))
            .enumerate()
            .for_each(|(i, ((point, u_i), g_i))| {
                let px = pos[2 * i];
                let py = pos[2 * i + 1];

                let (rep_x, rep_y, _) = rep_forces[i];

                let (dens_x, dens_y) = match &dens_ctx {
                    Some((state, scratch, consts)) => {
                        density_gradient(i, &adj[i], &pos, &state.r, scratch, consts)
                    }
                    None => (T::zero(), T::zero()),
                };

                let grad_x = attr[2 * i] - rep_x * z_inv - dens_x;
                let grad_y = attr[2 * i + 1] - rep_y * z_inv - dens_y;

                update_parameter(
                    &mut point[0],
                    &mut u_i[0],
                    &mut g_i[0],
                    grad_x,
                    lr,
                    momentum,
                    min_gain,
                );
                update_parameter(
                    &mut point[1],
                    &mut u_i[1],
                    &mut g_i[1],
                    grad_y,
                    lr,
                    momentum,
                    min_gain,
                );

                let (u0, u1) = u_i.split_at_mut(1);
                clip_step(point, &mut u0[0], &mut u1[0], px, py, max_step_norm);
            });

        recentre_embedding(embd);

        if verbosity.normal_verbosity() && (epoch % 50 == 0 || epoch == params.n_epochs - 1) {
            println!(
                " Epoch {}/{} | Z = {}",
                epoch,
                params.n_epochs,
                (z_total.round() as i64).separate_with_underscores()
            );
        }
    }
}

/////////
// FFT //
/////////

/// Flat CSR view of the affinity graph for the attractive pass.
///
/// `u32` indices halve the per-edge traffic against `Vec<(usize, T)>`, which
/// pads to 16 bytes per edge for `f32`.
#[cfg(feature = "fft_tsne")]
struct AttrCsr<T> {
    /// Row offsets, length `n + 1`
    indptr: Vec<usize>,
    /// Neighbour indices, length nnz
    indices: Vec<u32>,
    /// Edge weights `p_ij`, length nnz
    values: Vec<T>,
}

#[cfg(feature = "fft_tsne")]
impl<T: ManifoldsFloat> AttrCsr<T> {
    /// Flatten an adjacency list into CSR.
    ///
    /// ### Params
    ///
    /// * `adj` - Row-major adjacency, `adj[i]` holds `(j, p_ij)` pairs
    ///
    /// ### Returns
    ///
    /// The CSR graph.
    fn from_adjacency(adj: &[Vec<(usize, T)>]) -> Self {
        let mut indptr = Vec::with_capacity(adj.len() + 1);
        indptr.push(0);
        for row in adj {
            indptr.push(indptr.last().unwrap() + row.len());
        }
        let nnz = *indptr.last().unwrap();
        let mut indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        for row in adj {
            for &(j, w) in row {
                indices.push(j as u32);
                values.push(w);
            }
        }
        Self {
            indptr,
            indices,
            values,
        }
    }
}

/// Attractive forces for every point from the CSR graph.
///
/// ### Params
///
/// * `csr` - Affinity graph in CSR
/// * `pos` - Interleaved positions `[x0, y0, x1, y1, ...]`
/// * `exag_factor` - Current exaggeration factor
/// * `attr` - Output, interleaved attractive force per point, overwritten
#[cfg(feature = "fft_tsne")]
fn accumulate_attractive_csr<T>(csr: &AttrCsr<T>, pos: &[T], exag_factor: T, attr: &mut [T])
where
    T: ManifoldsFloat,
{
    attr.par_chunks_exact_mut(2)
        .enumerate()
        .for_each(|(i, out)| {
            let px = pos[2 * i];
            let py = pos[2 * i + 1];
            let (lo, hi) = (csr.indptr[i], csr.indptr[i + 1]);

            let mut attr_x = T::zero();
            let mut attr_y = T::zero();
            for (&j, &p_val) in csr.indices[lo..hi].iter().zip(&csr.values[lo..hi]) {
                let j = j as usize;
                let dx = px - pos[2 * j];
                let dy = py - pos[2 * j + 1];
                let q = T::one() / (T::one() + dx * dx + dy * dy);
                let force = p_val * q;
                attr_x += force * dx;
                attr_y += force * dy;
            }

            out[0] = attr_x * exag_factor;
            out[1] = attr_y * exag_factor;
        });
}

/// Compute FFT grid geometry for a given embedding half-span.
///
/// Below the box cap, `box_width` is fixed at `TSNE_FFT_MIN_BOX_WIDTH` and
/// `n_boxes` grows with the embedding span. Once `n_boxes` would exceed
/// `TSNE_FFT_MAX_BOXES`, the box count is clamped and `box_width` grows
/// instead, keeping the grid covering the embedding plus
/// `TSNE_FFT_GRID_MARGIN` headroom.
///
/// ### Params
///
/// * `half_span` - Half the current embedding extent (max absolute coordinate
///   across both axes).
/// * `min_intervals` - Minimum number of boxes per dimension.
///
/// ### Returns
///
/// `(n_boxes, box_width, grid_half)` where `grid_half` is the half-width of
/// the square grid in embedding coordinates.
#[cfg(any(feature = "fft_tsne", feature = "gpu"))]
pub(crate) fn fft_grid_geometry(half_span: f64, min_intervals: usize) -> (usize, f64, f64) {
    let span = 2.0 * half_span * 1.05;

    let n_boxes_unconstrained = choose_grid_size(0.0, span, TSNE_FFT_MIN_BOX_WIDTH, min_intervals);

    if n_boxes_unconstrained <= TSNE_FFT_MAX_BOXES {
        let half = n_boxes_unconstrained as f64 * TSNE_FFT_MIN_BOX_WIDTH / 2.0;
        (n_boxes_unconstrained, TSNE_FFT_MIN_BOX_WIDTH, half)
    } else {
        let grown_half = half_span * (1.05 + TSNE_FFT_GRID_MARGIN);
        let bw = (grown_half * 2.0 / TSNE_FFT_MAX_BOXES as f64).max(TSNE_FFT_MIN_BOX_WIDTH);
        let half = TSNE_FFT_MAX_BOXES as f64 * bw / 2.0;
        (TSNE_FFT_MAX_BOXES, bw, half)
    }
}

/// Optimise a 2D embedding using FFT-accelerated t-SNE.
///
/// Minimises the KL divergence between high-dimensional affinities (`graph`)
/// and low-dimensional Student-t similarities. Repulsive forces are
/// approximated via an interpolation-based N-body FFT scheme (Linderman et
/// al.), giving O(N) cost per epoch.
///
/// ### Params
///
/// * `embd` - Initial embedding coordinates, shape `[n_samples][2]`
///   (modified in place).
/// * `params` - Optimisation hyperparameters (epochs, learning rate, momentum
///   schedule, exaggeration, interpolation points per box).
/// * `graph` - Sparse high-dimensional affinities in coordinate-list format.
/// * `dens` - Density-preserving state for den-SNE, or `None` for plain tSNE.
///   When set, the density gradient is applied over the final
///   `dens.params.frac` of the epochs.
/// * `verbose` - Verbosity level: `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// `Ok(())` on success, or `Err(ManifoldsError::IncorrectDim)` if the
/// embedding is not 2D.
///
/// ### References
///
/// Linderman et al., Nature Methods, 2019 (FIt-SNE).
/// Narayan, Berger & Cho, Nature Biotechnology, 2021 (den-SNE).
#[cfg(feature = "fft_tsne")]
pub fn optimise_fft_tsne<T>(
    embd: &mut [Vec<T>],
    params: &TsneOptimParams<T>,
    graph: &CoordinateList<T>,
    dens: Option<&DensState<T>>,
    verbose: usize,
) -> Result<(), ManifoldsError>
where
    T: FftwFloat + ManifoldsFloat,
{
    optimise_fft_tsne_impl(embd, params, graph, dens, verbose, false)
}

/// Optimise a 2D embedding using three-kernel FFT-accelerated t-SNE.
///
/// Same optimiser as [`optimise_fft_tsne`], but the repulsion convolves a
/// unit charge grid with three kernels (`q`, `q^2 dx`, `q^2 dy`) instead of
/// four charges with `q^2`. Gives `Z` and the repulsive forces directly, with
/// one forward and three inverse transforms per epoch instead of four each.
///
/// ### Params
///
/// * `embd` - Initial embedding coordinates, shape `[n_samples][2]`
///   (modified in place).
/// * `params` - Optimisation hyperparameters.
/// * `graph` - Sparse high-dimensional affinities in coordinate-list format.
/// * `dens` - Density-preserving state for den-SNE, or `None` for plain tSNE.
/// * `verbose` - Verbosity level: `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// `Ok(())` on success, or `Err(ManifoldsError::IncorrectDim)` if the
/// embedding is not 2D.
#[cfg(feature = "fft_tsne")]
pub fn optimise_fft3k_tsne<T>(
    embd: &mut [Vec<T>],
    params: &TsneOptimParams<T>,
    graph: &CoordinateList<T>,
    dens: Option<&DensState<T>>,
    verbose: usize,
) -> Result<(), ManifoldsError>
where
    T: FftwFloat + ManifoldsFloat,
{
    optimise_fft_tsne_impl(embd, params, graph, dens, verbose, true)
}

/// Shared FFT t-SNE optimiser loop.
///
/// ### Params
///
/// * `embd` - Initial embedding coordinates, shape `[n_samples][2]`
/// * `params` - Optimisation hyperparameters
/// * `graph` - Sparse high-dimensional affinities
/// * `dens` - Density-preserving state, or `None`
/// * `verbose` - Verbosity level
/// * `three_kernel` - Use the three-kernel repulsion instead of the 4-term
///   expansion
///
/// ### Returns
///
/// `Ok(())` on success, or `Err(ManifoldsError::IncorrectDim)` if the
/// embedding is not 2D.
#[cfg(feature = "fft_tsne")]
fn optimise_fft_tsne_impl<T>(
    embd: &mut [Vec<T>],
    params: &TsneOptimParams<T>,
    graph: &CoordinateList<T>,
    dens: Option<&DensState<T>>,
    verbose: usize,
    three_kernel: bool,
) -> Result<(), ManifoldsError>
where
    T: FftwFloat + ManifoldsFloat,
{
    let verbosity = parse_verbosity_level(verbose);

    let n = embd.len();
    let n_dim = embd[0].len();
    let lr = params.get_lr(n);

    if n_dim != 2 {
        return Err(ManifoldsError::IncorrectDim { n_dim });
    }

    // 4-term: potentials of charges (1, x, y, x^2 + y^2); three-kernel:
    // (Z_i, fx_i, fy_i) directly
    let n_terms = if three_kernel { 3 } else { 4 };

    let initial_momentum = T::from_f64(TSNE_INITIAL_MOMENTUM).unwrap();
    let final_momentum = T::from_f64(TSNE_FINAL_MOMENTUM).unwrap();
    let min_gain = T::from_f64(TSNE_MIN_GAIN).unwrap();
    let max_step_norm = step_cap_from_lr(lr);

    let mut uy = vec![vec![T::zero(); n_dim]; n];
    let mut gains = vec![vec![T::one(); n_dim]; n];

    // the adjacency list is kept only for the density helpers shared with
    // Barnes-Hut; the attractive pass reads the leaner CSR.
    let adj_full = coo_to_adjacency_list(graph);
    let csr = AttrCsr::from_adjacency(&adj_full);
    let adj = if dens.is_some() { adj_full } else { Vec::new() };

    // pre-allocated FFT-side buffers and position snapshot.
    let mut charges = vec![T::zero(); if three_kernel { 0 } else { n * n_terms }];
    let mut potentials = vec![T::zero(); n * n_terms];
    let mut xs = vec![T::zero(); n];
    let mut ys = vec![T::zero(); n];

    // the FFT wants xs/ys split, the attractive and density passes an
    // interleaved buffer so each neighbour is one load.
    let mut pos = vec![T::zero(); n * n_dim];
    let mut attr = vec![T::zero(); n * n_dim];
    let mut dens_scratch = dens.map(|_| DensScratch::<T>::new(n));

    let min_intervals = TSNE_FFT_MIN_INTERVALS;
    let mut cached_n_boxes: usize = 0;
    let mut grid: Option<FftGrid<T>> = None;
    let mut workspace: Option<FftWorkspace<T>> = None;
    let mut kernels_3k: Option<FftKernels3k<T>> = None;
    let mut workspace_3k: Option<FftWorkspace3k<T>> = None;

    for epoch in 0..params.n_epochs {
        // snapshot positions in parallel.
        embd.par_iter()
            .zip(xs.par_iter_mut())
            .zip(ys.par_iter_mut())
            .zip(pos.par_chunks_exact_mut(2))
            .for_each(|(((p, x), y), slot)| {
                *x = p[0];
                *y = p[1];
                slot[0] = p[0];
                slot[1] = p[1];
            });

        let mut min_val = xs[0];
        let mut max_val = xs[0];
        for v in xs.iter().chain(ys.iter()) {
            if *v < min_val {
                min_val = *v;
            }
            if *v > max_val {
                max_val = *v;
            }
        }

        let half_span = min_val
            .to_f64()
            .unwrap()
            .abs()
            .max(max_val.to_f64().unwrap().abs());

        let (n_boxes, _box_width, grid_half) = fft_grid_geometry(half_span, min_intervals);

        let needs_rebuild = match grid.as_ref() {
            None => true,
            Some(_) if cached_n_boxes != n_boxes => true,
            Some(g) => {
                let coord_max =
                    g.coord_min + g.box_width * T::from_usize(g.n_boxes_per_dim).unwrap();
                let safe_max = coord_max - g.box_width;
                let safe_min = g.coord_min + g.box_width;
                let max_abs = T::from_f64(half_span).unwrap();
                max_abs >= safe_max || -max_abs <= safe_min
            }
        };

        if needs_rebuild {
            let half = T::from_f64(grid_half).unwrap();
            let new_grid = if three_kernel {
                FftGrid::new_geometry(-half, half, n_boxes, params.n_interp_points)
            } else {
                FftGrid::new(-half, half, n_boxes, params.n_interp_points)
            };
            if three_kernel {
                kernels_3k = Some(FftKernels3k::new(&new_grid));
                if cached_n_boxes != n_boxes {
                    workspace_3k = Some(FftWorkspace3k::new(new_grid.n_fft));
                }
            } else if cached_n_boxes != n_boxes {
                workspace = Some(FftWorkspace::new(new_grid.n_fft));
            }
            grid = Some(new_grid);
            cached_n_boxes = n_boxes;
        }

        let grid_ref = grid.as_ref().unwrap();

        let momentum = if epoch < TSNE_MOMENTUM_SWITCH_ITER {
            initial_momentum
        } else {
            final_momentum
        };
        let exag_factor = if epoch < params.early_exag_iter {
            params.early_exag_factor
        } else {
            params.get_late_exag_factor()
        };

        if three_kernel {
            n_body_fft_2d_3k(
                &xs,
                &ys,
                grid_ref,
                kernels_3k.as_ref().unwrap(),
                workspace_3k.as_mut().unwrap(),
                &mut potentials,
            );
        } else {
            // fill charges.
            charges
                .par_chunks_mut(n_terms)
                .enumerate()
                .for_each(|(i, chunk)| {
                    let x = xs[i];
                    let y = ys[i];
                    chunk[0] = T::one();
                    chunk[1] = x;
                    chunk[2] = y;
                    chunk[3] = x * x + y * y;
                });

            // zero potentials and run the FFT-accelerated convolution.
            for v in potentials.iter_mut() {
                *v = T::zero();
            }
            n_body_fft_2d(
                &xs,
                &ys,
                &charges,
                n_terms,
                grid_ref,
                workspace.as_mut().unwrap(),
                &mut potentials,
            );
        }

        // Z in f64; subtract n to remove the diagonal q_ii = 1 contribution.
        let sum_q: f64 = if three_kernel {
            (0..n)
                .map(|i| potentials[i * n_terms].to_f64().unwrap())
                .sum::<f64>()
                - n as f64
        } else {
            (0..n)
                .map(|i| {
                    let idx = i * n_terms;
                    let phi1 = potentials[idx].to_f64().unwrap();
                    let phi2 = potentials[idx + 1].to_f64().unwrap();
                    let phi3 = potentials[idx + 2].to_f64().unwrap();
                    let phi4 = potentials[idx + 3].to_f64().unwrap();
                    let x = xs[i].to_f64().unwrap();
                    let y = ys[i].to_f64().unwrap();
                    (1.0 + x * x + y * y) * phi1 - 2.0 * (x * phi2 + y * phi3) + phi4
                })
                .sum::<f64>()
                - n as f64
        };

        let sum_q_safe = if sum_q > TSNE_EPS { sum_q } else { 1.0 };

        // attractive forces (exact via sparse graph), plus the embedding radii
        // when the density term is live this epoch.
        let dens_ctx = match (dens, dens_scratch.as_mut()) {
            (Some(state), Some(scratch)) if state.is_active(epoch, params.n_epochs) => {
                accumulate_attractive_and_radii(&adj, &pos, exag_factor, &mut attr, scratch);
                Some((state, &*scratch, DensGradConsts::new(&scratch.re, state)))
            }
            _ => {
                accumulate_attractive_csr(&csr, &pos, exag_factor, &mut attr);
                None
            }
        };

        embd.par_iter_mut()
            .zip(uy.par_iter_mut())
            .zip(gains.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((point, u_i), gains_i))| {
                let x = xs[i];
                let y = ys[i];

                // repulsive forces, normalised in f64. The three-kernel path
                // returns them directly; the 4-term path reconstructs them.
                let pot_idx = i * n_terms;
                let (raw_x, raw_y) = if three_kernel {
                    (
                        potentials[pot_idx + 1].to_f64().unwrap(),
                        potentials[pot_idx + 2].to_f64().unwrap(),
                    )
                } else {
                    let phi1 = potentials[pot_idx].to_f64().unwrap();
                    let phi2 = potentials[pot_idx + 1].to_f64().unwrap();
                    let phi3 = potentials[pot_idx + 2].to_f64().unwrap();
                    let xf = x.to_f64().unwrap();
                    let yf = y.to_f64().unwrap();
                    (xf * phi1 - phi2, yf * phi1 - phi3)
                };

                let rep_x = T::from_f64(raw_x / sum_q_safe).unwrap();
                let rep_y = T::from_f64(raw_y / sum_q_safe).unwrap();

                let (dens_x, dens_y) = match &dens_ctx {
                    Some((state, scratch, consts)) => {
                        density_gradient(i, &adj[i], &pos, &state.r, scratch, consts)
                    }
                    None => (T::zero(), T::zero()),
                };

                let grad_x = attr[2 * i] - rep_x - dens_x;
                let grad_y = attr[2 * i + 1] - rep_y - dens_y;

                update_parameter(
                    &mut point[0],
                    &mut u_i[0],
                    &mut gains_i[0],
                    grad_x,
                    lr,
                    momentum,
                    min_gain,
                );
                update_parameter(
                    &mut point[1],
                    &mut u_i[1],
                    &mut gains_i[1],
                    grad_y,
                    lr,
                    momentum,
                    min_gain,
                );

                let (u0, u1) = u_i.split_at_mut(1);
                clip_step(point, &mut u0[0], &mut u1[0], x, y, max_step_norm);
            });

        recentre_embedding(embd);

        if verbosity.normal_verbosity() && (epoch % 50 == 0 || epoch == params.n_epochs - 1) {
            println!(
                " Epoch {}/{} | Z = {} | n_boxes = {}",
                epoch,
                params.n_epochs,
                (sum_q.round() as i64).separate_with_underscores(),
                n_boxes,
            );
        }
    }

    Ok(())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_tsne_optimiser {
    use super::*;
    use approx::assert_relative_eq;

    fn create_coo_graph(n: usize, edges: &[(usize, usize, f64)]) -> CoordinateList<f64> {
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for &(u, v, w) in edges {
            row_indices.push(u);
            col_indices.push(v);
            values.push(w);

            if u != v {
                row_indices.push(v);
                col_indices.push(u);
                values.push(w);
            }
        }

        CoordinateList {
            row_indices,
            col_indices,
            values,
            n_samples: n,
        }
    }

    #[test]
    fn test_tsne_params_defaults() {
        let params = TsneOptimParams::<f64>::default();
        assert_eq!(params.n_epochs, 1000);
        assert_eq!(params.early_exag_iter, 250);
        assert_relative_eq!(params.early_exag_factor, 12.0);
        assert_relative_eq!(params.theta, 0.5);
    }

    #[test]
    fn test_get_lr_floor_and_scaling() {
        let params = TsneOptimParams::<f64>::default();
        assert_relative_eq!(params.get_lr(100), 200.0);
        assert_relative_eq!(params.get_lr(120_000), 10_000.0);
        let fixed = TsneOptimParams {
            lr: Some(50.0),
            ..TsneOptimParams::default()
        };
        assert_relative_eq!(fixed.get_lr(1_000_000), 50.0);
    }

    #[test]
    fn test_step_cap_scales_with_lr() {
        // At the lr floor (200), the cap equals TSNE_MAX_STEP_FLOOR (= 5).
        let cap_small: f64 = step_cap_from_lr(200.0);
        assert_relative_eq!(cap_small, 5.0);
        // At a large lr, the cap scales with it.
        let cap_large: f64 = step_cap_from_lr(40_000.0);
        assert_relative_eq!(cap_large, 40_000.0 * TSNE_MAX_STEP_FRACTION);
    }

    #[test]
    fn test_bh_tsne_basic_convergence() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        let graph = create_coo_graph(3, &edges);

        let mut embd = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let initial_embd = embd.clone();

        let params = TsneOptimParams {
            n_epochs: 50,
            lr: Some(50.0),
            ..TsneOptimParams::default()
        };

        optimise_bh_tsne(&mut embd, &params, &graph, None, 0);

        for point in &embd {
            for val in point {
                assert!(val.is_finite(), "Embedding contains non-finite values");
            }
        }

        let total_movement: f64 = embd
            .iter()
            .zip(initial_embd.iter())
            .map(|(n, o)| (n[0] - o[0]).powi(2) + (n[1] - o[1]).powi(2))
            .sum();

        assert!(
            total_movement > 0.01,
            "Barnes-Hut t-SNE failed to move points significantly"
        );
    }

    #[test]
    fn test_qd_tsne_basic_convergence() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (3, 4, 1.0)];
        let graph = create_coo_graph(5, &edges);

        let mut embd = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.0],
            vec![2.0, 2.0],
            vec![2.1, 2.0],
        ];
        let initial_embd = embd.clone();

        // depth 1 forces shared leaves, so the own-leaf term is exercised
        let params = TsneOptimParams {
            n_epochs: 50,
            lr: Some(50.0),
            max_depth: 1,
            ..TsneOptimParams::default()
        };

        optimise_qd_tsne(&mut embd, &params, &graph, None, 0);

        for point in &embd {
            for val in point {
                assert!(val.is_finite(), "Embedding contains non-finite values");
            }
        }

        let total_movement: f64 = embd
            .iter()
            .zip(initial_embd.iter())
            .map(|(n, o)| (n[0] - o[0]).powi(2) + (n[1] - o[1]).powi(2))
            .sum();

        assert!(
            total_movement > 0.01,
            "Quick-and-dirty t-SNE failed to move points significantly"
        );
    }

    #[test]
    fn test_qd_tsne_determinism() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0)];
        let graph = create_coo_graph(5, &edges);

        let mut embd1 = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![3.0, 3.0],
            vec![3.1, 3.0],
        ];
        let mut embd2 = embd1.clone();

        let params = TsneOptimParams {
            n_epochs: 50,
            max_depth: 2,
            ..TsneOptimParams::default()
        };

        optimise_qd_tsne(&mut embd1, &params, &graph, None, 0);
        optimise_qd_tsne(&mut embd2, &params, &graph, None, 0);

        for (p1, p2) in embd1.iter().zip(embd2.iter()) {
            assert_relative_eq!(p1[0], p2[0]);
            assert_relative_eq!(p1[1], p2[1]);
        }
    }

    #[test]
    #[cfg(feature = "fft_tsne")]
    fn test_fft_tsne_basic_convergence() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        let graph = create_coo_graph(3, &edges);

        let mut embd = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let initial_embd = embd.clone();

        let params = TsneOptimParams {
            n_epochs: 50,
            lr: Some(50.0),
            n_interp_points: 3,
            ..TsneOptimParams::default()
        };

        let _ = optimise_fft_tsne(&mut embd, &params, &graph, None, 0);

        for point in &embd {
            for val in point {
                assert!(val.is_finite(), "Embedding contains non-finite values");
            }
        }

        let total_movement: f64 = embd
            .iter()
            .zip(initial_embd.iter())
            .map(|(n, o)| (n[0] - o[0]).powi(2) + (n[1] - o[1]).powi(2))
            .sum();

        assert!(
            total_movement > 0.01,
            "FFT t-SNE failed to move points significantly"
        );
    }

    #[test]
    fn test_bh_tsne_determinism() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0)];
        let graph = create_coo_graph(3, &edges);

        let mut embd1 = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut embd2 = embd1.clone();

        let params = TsneOptimParams {
            n_epochs: 50,
            ..TsneOptimParams::default()
        };

        optimise_bh_tsne(&mut embd1, &params, &graph, None, 0);
        optimise_bh_tsne(&mut embd2, &params, &graph, None, 0);

        for (p1, p2) in embd1.iter().zip(embd2.iter()) {
            assert_relative_eq!(p1[0], p2[0]);
            assert_relative_eq!(p1[1], p2[1]);
        }
    }

    #[test]
    fn test_parse_tsne_optimiser() {
        assert!(matches!(
            parse_tsne_optimiser("bh"),
            Some(TsneOpt::BarnesHut)
        ));
        assert!(matches!(
            parse_tsne_optimiser("Barnes_Hut"),
            Some(TsneOpt::BarnesHut)
        ));
        assert!(matches!(parse_tsne_optimiser("fft"), Some(TsneOpt::Fft)));
        assert!(matches!(
            parse_tsne_optimiser("fft_3k"),
            Some(TsneOpt::Fft3Kernel)
        ));
        assert!(matches!(
            parse_tsne_optimiser("3-kernel"),
            Some(TsneOpt::Fft3Kernel)
        ));
        assert!(matches!(
            parse_tsne_optimiser("FFT_3K_GPU"),
            Some(TsneOpt::Fft3KernelGpu)
        ));
        assert!(matches!(
            parse_tsne_optimiser("qd"),
            Some(TsneOpt::BarnesHutQd)
        ));
        assert!(matches!(
            parse_tsne_optimiser("BH_QD"),
            Some(TsneOpt::BarnesHutQd)
        ));
        assert!(parse_tsne_optimiser("fft3").is_none());
    }

    #[test]
    #[cfg(feature = "fft_tsne")]
    fn test_fft3k_tsne_basic_convergence() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        let graph = create_coo_graph(3, &edges);

        let mut embd = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let initial_embd = embd.clone();

        let params = TsneOptimParams {
            n_epochs: 50,
            lr: Some(50.0),
            n_interp_points: 3,
            ..TsneOptimParams::default()
        };

        optimise_fft3k_tsne(&mut embd, &params, &graph, None, 0).unwrap();

        for point in &embd {
            for val in point {
                assert!(val.is_finite(), "Embedding contains non-finite values");
            }
        }

        let total_movement: f64 = embd
            .iter()
            .zip(initial_embd.iter())
            .map(|(n, o)| (n[0] - o[0]).powi(2) + (n[1] - o[1]).powi(2))
            .sum();

        assert!(
            total_movement > 0.01,
            "Three-kernel FFT t-SNE failed to move points significantly"
        );
    }

    #[test]
    #[cfg(feature = "fft_tsne")]
    fn test_fft3k_tsne_determinism() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0)];
        let graph = create_coo_graph(3, &edges);

        let mut embd1 = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut embd2 = embd1.clone();

        let params = TsneOptimParams {
            n_epochs: 50,
            ..TsneOptimParams::default()
        };

        optimise_fft3k_tsne(&mut embd1, &params, &graph, None, 0).unwrap();
        optimise_fft3k_tsne(&mut embd2, &params, &graph, None, 0).unwrap();

        for (p1, p2) in embd1.iter().zip(embd2.iter()) {
            assert_eq!(p1, p2);
        }
    }

    #[test]
    #[cfg(feature = "fft_tsne")]
    fn test_fft3k_tsne_rejects_non_2d() {
        let graph = create_coo_graph(2, &[(0, 1, 1.0)]);
        let mut embd = vec![vec![0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
        let res = optimise_fft3k_tsne(&mut embd, &TsneOptimParams::default(), &graph, None, 0);
        assert!(matches!(
            res,
            Err(ManifoldsError::IncorrectDim { n_dim: 3 })
        ));
    }
}
