//! Optimisers for UMAP fitting. Contains the SGD, a lock-free parallel SGD,
//! Adam and a parallel Adam variant

use rand::{
    rngs::SmallRng,
    {Rng, SeedableRng},
};
use rayon::prelude::*;

use crate::prelude::*;
use crate::training::umap_csr::*;
use crate::training::*;
use crate::utils::density::*;

//////////
// UMAP //
//////////

///////////////
// Constants //
///////////////

/// Minimum embedding distance the default curve is fitted to.
///
/// The single place the default lives, read by every UMAP variant.
pub const DEFAULT_MIN_DIST: f64 = 0.5;

/// Spread the default curve is fitted to. See [`DEFAULT_MIN_DIST`].
pub const DEFAULT_SPREAD: f64 = 1.0;

/// Upper end of the fine `x^b` table in [`SplitPowLut`]. Squared distances of
/// connected points sit well inside it.
const POW_LUT_NEAR_MAX: f64 = 25.0;

/// Upper end of the coarse `x^b` table in [`SplitPowLut`]. Covers the squared
/// distances of random negative samples in a typical 2D layout; anything
/// further falls back to `powf`. Not tuned beyond checking that widening it
/// further does not change the run time.
const POW_LUT_FAR_MAX: f64 = 10_000.0;

/// Entries per `x^b` table. Two tables of this size stay within L2.
const POW_LUT_SIZE: usize = 65_536;

//////////////////////////
// Structures and Enums //
//////////////////////////

/// UMAP optimisation parameters
#[derive(Clone, Debug)]
pub struct UmapOptimParams<T> {
    /// Curve parameter for repulsive force. Fitted from `min_dist` and
    /// `spread` rather than set by hand; `~0.59` at the defaults.
    pub a: T,
    /// Curve parameter for repulsive force. Fitted alongside `a`; `~1.33` at
    /// the defaults.
    pub b: T,
    /// Initial learning rate (typically 1.0)
    pub lr: T,
    /// Parameter to control repulsion force
    pub gamma: T,
    /// Number of optimisation epochs (typically 500)
    pub n_epochs: usize,
    /// Number of negative samples per positive edge (typically 5)
    pub neg_sample_rate: usize,
    /// Minimum distance between points in embedding. Defaults to
    /// [`DEFAULT_MIN_DIST`].
    pub min_dist: T,
    /// Beta1 parameter for Adam optimiser
    pub beta1: T,
    /// Beta2 parameter for Adam optimiser
    pub beta2: T,
    /// Eps for Adam optimiser
    pub eps: T,
}

impl<T> UmapOptimParams<T>
where
    T: ManifoldsFloat,
{
    /// Default parameters for 2D embedding
    ///
    /// `a` and `b` are fitted from `min_dist = 0.5` and `spread = 1.0` rather
    /// than written down. Hardcoding them meant this constructor and
    /// [`UmapOptimParams::from_min_dist_spread`] disagreed about what the
    /// default curve was, and a caller who set `min_dist` got a different `a`
    /// and `b` from one who left it alone at a value that was not even the same
    /// `min_dist`.
    ///
    /// ### Returns
    ///
    /// Self with sensible default parameters for the classical 2D case.
    pub fn default_2d() -> Self {
        Self::from_min_dist_spread(
            T::from_f64(DEFAULT_MIN_DIST).unwrap(),
            T::from_f64(DEFAULT_SPREAD).unwrap(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    /// Params from specified minimum distance and spread
    ///
    /// ### Params
    ///
    /// * `min_dist` - Minimum distance parameter
    /// * `spread` - Effective scale of embedded points
    /// * `lr` - Initial learning rate. Defaults to `1.0`.
    /// * `gamma` - Weight on the repulsive term. Defaults to `1.0`.
    /// * `n_epochs` - Number of optimisation epochs. Defaults to `500`.
    /// * `neg_sample_rate` - Number of negative samples per positive edge.
    ///   Defaults to `5`.
    /// * `beta1` - Adam first-moment decay. Defaults to [`UMAP_BETA1`].
    /// * `beta2` - Adam second-moment decay. Defaults to [`UMAP_BETA2`].
    /// * `eps` - Adam denominator epsilon. Defaults to [`EPS`].
    ///
    /// ### Returns
    ///
    /// Self with `a` and `b` fitted to `min_dist` and `spread`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_min_dist_spread(
        min_dist: T,
        spread: T,
        lr: Option<T>,
        gamma: Option<T>,
        n_epochs: Option<usize>,
        neg_sample_rate: Option<usize>,
        beta1: Option<T>,
        beta2: Option<T>,
        eps: Option<T>,
    ) -> Self {
        // take the Adam-related values
        let beta1 = beta1.unwrap_or(T::from(UMAP_BETA1).unwrap());
        let beta2 = beta2.unwrap_or(T::from(UMAP_BETA2).unwrap());
        let eps = eps.unwrap_or(T::from(EPS).unwrap());
        let n_epochs = n_epochs.unwrap_or(500);
        let neg_sample_rate = neg_sample_rate.unwrap_or(5);
        let lr = lr.unwrap_or(T::one());
        let gamma = gamma.unwrap_or(T::one());

        let (a, b) = Self::fit_params(min_dist, spread, None);
        Self {
            a,
            b,
            lr,
            gamma,
            n_epochs,
            neg_sample_rate,
            min_dist,
            beta1,
            beta2,
            eps,
        }
    }

    /// Fit curve parameters from min_dist and spread
    ///
    /// Fits the UMAP curve `f(x) = 1 / (1 + a * x^(2b))` to `1` below `min_dist`
    /// and `exp(-(x - min_dist) / spread)` above it.
    ///
    /// ### Params
    ///
    /// * `min_dist` - Minimum distance parameter
    /// * `spread` - Effective scale of embedded points
    /// * `n_iter` - Number of optimisation iterations (default: 300)
    ///
    /// ### Returns
    ///
    /// Tuple of `(a, b)` according to the optimisation problem above.
    fn fit_params(min_dist: T, spread: T, n_iter: Option<usize>) -> (T, T) {
        let n_iter = n_iter.unwrap_or(300);
        let n_points = 300;

        // Generate x values from 0 to spread * 3
        let three = T::from_f64(3.0).unwrap();
        let max_x = spread * three;
        let step = max_x / T::from_usize(n_points - 1).unwrap();

        // Generate target y values
        let mut xv = Vec::with_capacity(n_points);
        let mut yv = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let x = step * T::from_usize(i).unwrap();
            let y = if x < min_dist {
                T::one()
            } else {
                (-(x - min_dist) / spread).exp()
            };
            xv.push(x);
            yv.push(y);
        }

        let mut a = T::one();
        let mut b = T::one();
        let two = T::from_f64(2.0).unwrap();

        for _ in 0..n_iter {
            let mut grad_a = T::zero();
            let mut grad_b = T::zero();
            let n_points_t = T::from_usize(n_points).unwrap();

            for i in 0..n_points {
                let x = xv[i];
                if x <= T::zero() {
                    continue;
                }

                let y_target = yv[i];
                let x_2b = x.powf(two * b);
                let denom = T::one() + a * x_2b;
                let pred = T::one() / denom;
                let err = pred - y_target;

                grad_a += err * (-x_2b / (denom * denom));

                let log_x = x.ln();
                grad_b += err * (-two * a * x_2b * log_x / (denom * denom));
            }

            // Normalise gradients and use adaptive learning rate
            grad_a /= n_points_t;
            grad_b /= n_points_t;

            let lr_a = T::from_f64(1.0).unwrap();
            let lr_b = T::from_f64(1.0).unwrap();

            a -= lr_a * grad_a;
            b -= lr_b * grad_b;

            a = a
                .max(T::from_f64(0.001).unwrap())
                .min(T::from_f64(10.0).unwrap());
            b = b
                .max(T::from_f64(0.1).unwrap())
                .min(T::from_f64(2.0).unwrap());
        }

        (a, b)
    }

    /// Cast parameters to a different float type. `usize` fields pass through.
    ///
    /// ### Returns
    ///
    /// `UmapOptimParams<U>` with all float fields converted via `NumCast`.
    pub fn cast<U>(&self) -> UmapOptimParams<U>
    where
        U: ManifoldsFloat,
    {
        let c = |v: T| U::from(v).unwrap();
        UmapOptimParams {
            a: c(self.a),
            b: c(self.b),
            lr: c(self.lr),
            gamma: c(self.gamma),
            n_epochs: self.n_epochs,
            neg_sample_rate: self.neg_sample_rate,
            min_dist: c(self.min_dist),
            beta1: c(self.beta1),
            beta2: c(self.beta2),
            eps: c(self.eps),
        }
    }
}

impl<T> Default for UmapOptimParams<T>
where
    T: ManifoldsFloat,
{
    /// Returns sensible defaults for the optimiser (assuming 2D)
    fn default() -> Self {
        UmapOptimParams::default_2d()
    }
}

/// Type of UMAP optimiser to use
#[derive(Default)]
pub enum UmapOptimiser {
    /// Parallel version of Adam
    #[default]
    AdamParallel,
    /// Adam
    Adam,
    /// Stochastic gradient descent
    Sgd,
    /// Lock-free parallel stochastic gradient descent (hogwild)
    SgdParallel,
    /// Adam on the GPU via MLX (Apple Silicon, `mlx` feature)
    #[cfg(mlx_available)]
    AdamMlx,
}

/// Parse the UMAP Optimiser to use
///
/// ### Params
///
/// * `s` - String defining the optimiser. Choice of `"adam"`, `"adam_parallel"`,
///   `"sgd"` or `"sgd_parallel"`, plus `"adam_mlx"` when the MLX backend is
///   compiled in.
///
/// ### Return
///
/// Option of Optimiser
pub fn parse_umap_optimiser(s: &str) -> Option<UmapOptimiser> {
    match s.to_lowercase().as_str() {
        "adam" => Some(UmapOptimiser::Adam),
        "sgd" => Some(UmapOptimiser::Sgd),
        "sgd_parallel" => Some(UmapOptimiser::SgdParallel),
        "adam_parallel" => Some(UmapOptimiser::AdamParallel),
        #[cfg(mlx_available)]
        "adam_mlx" => Some(UmapOptimiser::AdamMlx),
        _ => None,
    }
}

/// Precomputed constants to avoid repeated calculations
struct OptimConstants<T> {
    /// The a parameter.
    a: T,
    /// The b parameter.
    b: T,
    /// a * b multiplied with 2.
    two_a_b: T,
    /// 2 * gamma b
    two_gamma_b: T,
    /// The clipping value, i.e., `4.0`.
    clip_val: T,
    /// The epsilon value
    eps: T,
}

impl<T> OptimConstants<T>
where
    T: ManifoldsFloat,
{
    /// Generate all of the constants
    ///
    /// ### Params
    ///
    /// * `a` - The a parameter
    /// * `b` - The b parameter
    /// * `gamma` - The repulsion parameter. Usually defaults to `1.0`.
    ///
    /// ### Returns
    ///
    /// Self with all pre-calculated values.
    fn new(a: T, b: T, gamma: T) -> Self {
        let two = T::from_f64(2.0).unwrap();
        Self {
            a,
            b,
            two_a_b: two * a * b,
            two_gamma_b: two * gamma * b,
            clip_val: T::from_f64(GRAD_CLIP_VAL).unwrap(),
            eps: T::from_f64(GRAD_REP_EPS).unwrap(),
        }
    }
}

/////////////
// Helpers //
/////////////

/// Fast Lookup Table (LUT) for expensive power calculations
///
/// Replaces the expensive `powf` calls with a precomputed linear interpolation.
/// This drastically improves SIMD pipelining and reduces math bottlenecks.
struct FastPowLut<T> {
    /// The b-value
    b: T,
    /// Maximum value
    max_val: T,
    /// Inverse step
    inv_step: T,
    /// Table for look-ups
    table: Vec<T>,
}

impl<T> FastPowLut<T>
where
    T: ManifoldsFloat,
{
    /// Creates a new LUT for the function x^b
    ///
    /// ### Params
    ///
    /// * `max_val` - Maximum value
    /// * `size` - Size of the LUT
    fn new(b: T, max_val: f64, size: usize) -> Self {
        let mut table = Vec::with_capacity(size);
        let max_t = T::from(max_val).unwrap();
        let step = max_t / T::from(size - 1).unwrap();

        for i in 0..size {
            let x = step * T::from(i).unwrap();
            table.push(x.powf(b));
        }

        Self {
            b,
            max_val: max_t,
            inv_step: T::one() / step,
            table,
        }
    }

    /// Retrieve approximated x^b using linear interpolation
    ///
    /// ### Params
    ///
    /// * `x` - The distance value
    ///
    /// ### Returns
    ///
    /// x.powf(b) calculated via linear interpolation
    #[inline(always)]
    fn get(&self, x: T) -> T {
        // fallback for extreme distances
        if x >= self.max_val {
            return x.powf(self.b);
        }

        let idx_f = x * self.inv_step;
        let idx = idx_f.to_usize().unwrap_or(0);

        if idx >= self.table.len() - 1 {
            return self.table.last().copied().unwrap();
        }

        // linear interpolation for smooth gradients
        let rem = idx_f - T::from(idx).unwrap();
        let y0 = self.table[idx];
        let y1 = self.table[idx + 1];
        y0 + rem * (y1 - y0)
    }
}

/// Two-range lookup table for `x^b` covering the repulsive distances too.
///
/// [`FastPowLut`] alone spans `[0, POW_LUT_NEAR_MAX]`, where attraction lives,
/// but negative samples land mostly far beyond it and fell back to `powf`. The
/// second, coarser table covers `[POW_LUT_NEAR_MAX, POW_LUT_FAR_MAX]`; its step
/// is wide in absolute terms but small relative to `x` there, so the linear
/// interpolation stays accurate. Beyond that, `powf`.
struct SplitPowLut<T> {
    /// Table over `[0, POW_LUT_NEAR_MAX]`.
    near: FastPowLut<T>,
    /// Table over `[0, POW_LUT_FAR_MAX]`, only read above `POW_LUT_NEAR_MAX`.
    far: FastPowLut<T>,
    /// `POW_LUT_NEAR_MAX` in `T`.
    split: T,
    /// `b == 1`, where `x^b` is `x` and no table is needed.
    b_is_one: bool,
}

impl<T> SplitPowLut<T>
where
    T: ManifoldsFloat,
{
    /// Build both tables for `x^b`.
    ///
    /// ### Params
    ///
    /// * `b` - The exponent
    ///
    /// ### Returns
    ///
    /// The two-range table.
    fn new(b: T) -> Self {
        Self {
            near: FastPowLut::new(b, POW_LUT_NEAR_MAX, POW_LUT_SIZE),
            far: FastPowLut::new(b, POW_LUT_FAR_MAX, POW_LUT_SIZE),
            split: T::from(POW_LUT_NEAR_MAX).unwrap(),
            b_is_one: (b - T::one()).abs() < T::from(1e-10).unwrap(),
        }
    }

    /// Approximate `x^b`.
    ///
    /// ### Params
    ///
    /// * `x` - Non-negative input, a squared distance
    ///
    /// ### Returns
    ///
    /// `x^b`, interpolated from the matching table.
    #[inline(always)]
    fn get(&self, x: T) -> T {
        if self.b_is_one {
            x
        } else if x < self.split {
            self.near.get(x)
        } else {
            self.far.get(x)
        }
    }
}

/// Hint the CPU to pull the cache line holding `x` into L1.
///
/// Only emitted on aarch64, where it was measured to help the scattered
/// negative-sample loads once the embedding outgrows the caches; a no-op
/// elsewhere.
///
/// ### Params
///
/// * `x` - Any reference into the data that will be read soon
#[inline(always)]
fn prefetch_read<A>(x: &A) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `prfm` is a hint; it never faults and writes nothing, and the
    // address comes from a live reference.
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{0}]",
            in(reg) x as *const A,
            options(nostack, readonly, preserves_flags)
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = x;
}

///////////////////////
// Density (densMAP) //
///////////////////////

/// Per-epoch scratch buffers for the densMAP density term.
///
/// Allocated once per run, and only when the density term is in use. The
/// embedding radii are weighted by the UMAP kernel `phi`, which moves every
/// epoch, so unlike `mu_sum` none of this can be precomputed.
struct UmapDensScratch<T> {
    /// `sum_j phi_ij * ||y_i - y_j||^2` per node.
    re_acc: Vec<T>,
    /// `sum_j phi_ij` per node, the embedding-radius denominator.
    phi_sum: Vec<T>,
    /// `log(eps + re_acc / phi_sum)` per node.
    re: Vec<T>,
    /// `R[i] - cov * (re[i] - re_mean) / re_std^2` per node, the per-point
    /// sensitivity of the correlation.
    weight: Vec<T>,
    /// `lambda * mu_tot / (re_std * (n - 1))`, the edge-independent factor of
    /// the per-edge coefficient.
    coeff: T,
}

impl<T> UmapDensScratch<T>
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
            weight: vec![T::zero(); n],
            coeff: T::zero(),
        }
    }
}

/// Accumulate the embedding local radii over every graph edge.
///
/// Walks each node's adjacency row, so the accumulation is node-partitioned and
/// needs no atomics. Deliberately ignores the edge sampling schedule: the radii
/// are a property of the whole graph, and the reference recomputes them over the
/// full edge list every epoch regardless of which edges fire.
///
/// Takes the adjacency list rather than the CSR view so all four CPU
/// optimisers can share it. This runs once per epoch and only over the final
/// `frac` of the run, so the pointer chasing is not worth specialising away.
///
/// ### Params
///
/// * `graph` - Adjacency list, `graph[i]` holding `(neighbour, weight)`
/// * `embd_flat` - Flat row-major embedding `[n * n_dim]`
/// * `n_dim` - Embedding dimensionality
/// * `consts` - Precomputed `a`, `b` and friends
/// * `pow_b` - Computes `x^b`. Supplied by the caller so `phi_sum` here is
///   built from exactly the same approximation the gradient loop uses; mixing
///   the LUT with `powf` would bias every radius
/// * `scratch` - Output, radii accumulators and log-radii, overwritten
fn accumulate_umap_radii<T, F>(
    graph: &[Vec<(usize, T)>],
    embd_flat: &[T],
    n_dim: usize,
    consts: &OptimConstants<T>,
    pow_b: F,
    scratch: &mut UmapDensScratch<T>,
) where
    T: ManifoldsFloat,
    F: Fn(T) -> T + Sync,
{
    scratch
        .re_acc
        .par_iter_mut()
        .zip(scratch.phi_sum.par_iter_mut())
        .enumerate()
        .for_each(|(node_i, (re_acc, phi_sum))| {
            let base_i = node_i * n_dim;

            let mut sum_sq = T::zero();
            let mut sum_phi = T::zero();

            for &(other_node, _) in &graph[node_i] {
                let base_other = other_node * n_dim;

                let mut dist_sq = T::zero();
                for d in 0..n_dim {
                    let diff = embd_flat[base_i + d] - embd_flat[base_other + d];
                    dist_sq += diff * diff;
                }

                let dist_sq_b = pow_b(dist_sq);
                let phi = T::one() / (T::one() + consts.a * dist_sq_b);

                sum_sq += phi * dist_sq;
                sum_phi += phi;
            }

            *re_acc = sum_sq;
            *phi_sum = sum_phi;
        });

    embedding_log_radii(&scratch.re_acc, &scratch.phi_sum, &mut scratch.re);
}

/// Refresh the per-node correlation sensitivities and the shared coefficient.
///
/// ### Params
///
/// * `scratch` - Holds the current `re`; `weight` and `coeff` are overwritten
/// * `state` - Constant density state, holding the original radii and `mu_tot`
fn update_density_weights<T>(scratch: &mut UmapDensScratch<T>, state: &DensState<T>)
where
    T: ManifoldsFloat,
{
    let (re_mean, re_std, cov) = correlation_stats(&scratch.re, &state.r, state.params.var_shift);
    let inv_var = T::one() / (re_std * re_std);

    scratch
        .weight
        .par_iter_mut()
        .zip(state.r.par_iter())
        .zip(scratch.re.par_iter())
        .for_each(|((weight, &r), &re)| {
            *weight = r - cov * (re - re_mean) * inv_var;
        });

    let denom = T::from_usize(scratch.re.len().saturating_sub(1).max(1)).unwrap();
    scratch.coeff = state.params.lambda * state.mu_tot / (re_std * denom);
}

/// Per-edge density coefficient for the densMAP gradient.
///
/// Reuses `dist_sq`, `dist_sq_b` and `denom` from the attractive term, so the
/// only extra transcendentals are the two `exp` calls. Must be called under the
/// existing `dist_sq >= GRAD_DIST_SQ_THRESHOLD` guard: `dist_sq^(b-1)` is
/// `+inf` at zero for the usual `b < 1`, a case the reference does not protect
/// against.
///
/// ### Params
///
/// * `node_i` - The node whose gradient is being accumulated
/// * `other_node` - The neighbour across this edge
/// * `mu_edge` - Graph weight of this edge
/// * `dist_sq` - Squared embedding distance
/// * `dist_sq_b` - `dist_sq^b`, already computed for the attractive term
/// * `denom` - `1 + a * dist_sq^b`, already computed for the attractive term
/// * `consts` - Precomputed `a`, `b` and friends
/// * `scratch` - Current radii, weights and shared coefficient
///
/// ### Returns
///
/// The scalar `grad_cor_coeff`. Multiply by `2 * (y_i - y_other)`, clip, and
/// add to the node gradient.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn density_edge_coeff<T>(
    node_i: usize,
    other_node: usize,
    mu_edge: T,
    dist_sq: T,
    dist_sq_b: T,
    denom: T,
    consts: &OptimConstants<T>,
    scratch: &UmapDensScratch<T>,
) -> T
where
    T: ManifoldsFloat,
{
    let one = T::one();
    let floor = T::epsilon();

    let phi = one / denom;
    // dsq^(b-1) is dsq^b / dsq, so no second powf
    let dphi_term = consts.a * consts.b * dist_sq_b / dist_sq * phi;
    let common = one - consts.b * (one - phi);

    let dr_self = (phi / scratch.phi_sum[node_i].max(floor))
        * (common / scratch.re[node_i].exp() + dphi_term);
    let dr_other = (phi / scratch.phi_sum[other_node].max(floor))
        * (common / scratch.re[other_node].exp() + dphi_term);

    scratch.coeff * (scratch.weight[node_i] * dr_self + scratch.weight[other_node] * dr_other)
        / mu_edge.max(floor)
}

/////////////
// Helpers //
/////////////

/// Fast power version
///
/// For specific versions of b, return quickly the value
#[inline(always)]
fn fast_pow<T: ManifoldsFloat>(x: T, b: T, b_is_one: bool, b_is_half: bool) -> T {
    if b_is_one {
        x
    } else if b_is_half {
        x.sqrt()
    } else {
        x.powf(b)
    }
}

////////////////
// Optimisers //
////////////////

/// Sampling schedule of one directed edge, owned by its source node.
#[derive(Clone, Copy)]
struct EdgeSchedule<T> {
    /// Epoch at which the edge next fires.
    next: T,
    /// Epoch up to which negative samples have been drawn for this edge.
    next_neg: T,
}

/// Optimise UMAP embedding using Stochastic Gradient Descent (SGD)
///
/// Sequential version of [`optimise_embedding_sgd_parallel`]: same code, run
/// over the nodes in order on the calling thread, so it is reproducible for a
/// fixed seed. See there for the algorithm.
///
/// ### Params
///
/// * `embd` - Initial embedding coordinates (modified in place), shape
///   `[n_samples][n_dim]`
/// * `graph` - Adjacency list where `graph[i]` contains (neighbour_idx, weight)
///   pairs
/// * `params` - Optimisation parameters (n_epochs, lr, a, b, gamma,
///   neg_sample_rate)
/// * `dens` - Optional densMAP state. When set, the density gradient is
///   applied over the final `dens.params.frac` of the epochs.
/// * `seed` - Random seed for negative sampling
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `Ok(())` on success, the embedding updated in place. `NoData` for an empty
/// embedding, `NoGraphEdges` for a graph without edges, `TooManyVertices` if
/// the node count does not fit `u32`.
pub fn optimise_embedding_sgd<T>(
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
    sgd_impl(embd, graph, params, dens, seed, verbose, false)
}

/// Optimise UMAP embedding using lock-free parallel SGD (hogwild)
///
/// The umap-learn schedule: every directed edge fires at a rate proportional
/// to its weight, moves both endpoints together, and draws its share of
/// negative samples to push the source away from. Learning rate decays
/// linearly.
///
/// Runs parallel over source nodes with Rayon. Each node owns its outgoing
/// edges, their schedules and its RNG, so those need no synchronisation. The
/// embedding is shared and written without locks: the source node's
/// coordinates are held locally for the whole row and written back once, the
/// neighbour's are written per edge. Concurrent writes to the same point can
/// overwrite each other, the usual hogwild trade, which the sparse UMAP
/// objective tolerates.
///
/// Two departures from umap-learn, both for speed:
///
/// - An edge's negative samples are all evaluated against the same source
///   position and applied as one summed step, instead of moving the source
///   after each. This breaks the serial dependency between samples, so their
///   loads and arithmetic overlap.
/// - `x^b` comes from a lookup table ([`SplitPowLut`]) rather than `powf`.
///
/// With one Rayon thread the result is identical to [`optimise_embedding_sgd`].
/// With more, it is not reproducible run to run for a fixed seed: the RNG
/// streams are fixed, the write interleaving is not.
///
/// ### Params
///
/// * `embd` - Initial embedding coordinates (modified in place), shape
///   `[n_samples][n_dim]`
/// * `graph` - Adjacency list where `graph[i]` contains (neighbour_idx, weight)
///   pairs
/// * `params` - Optimisation parameters (n_epochs, lr, a, b, gamma,
///   neg_sample_rate)
/// * `dens` - Optional densMAP state. When set, the density gradient is
///   applied over the final `dens.params.frac` of the epochs.
/// * `seed` - Random seed for negative sampling
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `Ok(())` on success, the embedding updated in place. `NoData` for an empty
/// embedding, `NoGraphEdges` for a graph without edges, `TooManyVertices` if
/// the node count does not fit `u32`.
///
/// ### References
///
/// Recht et al., NeurIPS, 2011 (Hogwild!); McInnes, Healy & Melville, arXiv,
/// 2018 (UMAP).
pub fn optimise_embedding_sgd_parallel<T>(
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
    sgd_impl(embd, graph, params, dens, seed, verbose, true)
}

/// Shared body of the sequential and the hogwild SGD optimisers.
///
/// ### Params
///
/// * `embd` - Initial embedding coordinates (modified in place)
/// * `graph` - Adjacency list, `graph[i]` holding `(neighbour, weight)`
/// * `params` - Optimisation parameters
/// * `dens` - Optional densMAP state
/// * `seed` - Random seed for negative sampling
/// * `verbose` - Verbosity level
/// * `parallel` - Walk the nodes with Rayon (hogwild) or in order on the
///   calling thread
///
/// ### Returns
///
/// See [`optimise_embedding_sgd_parallel`].
fn sgd_impl<T>(
    embd: &mut [Vec<T>],
    graph: &[Vec<(usize, T)>],
    params: &UmapOptimParams<T>,
    dens: Option<&DensState<T>>,
    seed: u64,
    verbose: usize,
    parallel: bool,
) -> Result<(), ManifoldsError>
where
    T: ManifoldsFloat,
{
    let n = embd.len();
    if n == 0 {
        return Err(ManifoldsError::NoData);
    }
    if n > u32::MAX as usize {
        return Err(ManifoldsError::TooManyVertices { n });
    }
    let n_dim = embd[0].len();
    let verbosity = parse_verbosity_level(verbose);

    let consts = OptimConstants::new(params.a, params.b, params.gamma);
    let pow_lut = SplitPowLut::new(consts.b);

    let zero = T::zero();
    let one = T::one();
    let two = T::from(2.0).unwrap();
    let dist_sq_threshold = T::from(GRAD_DIST_SQ_THRESHOLD).unwrap();
    let large_epoch = T::from(1e8).unwrap();
    let rep_eps = T::from(GRAD_REP_EPS).unwrap();

    // directed edges grouped by source node, as in `graph`
    let n_edges: usize = graph.iter().map(|row| row.len()).sum();
    if n_edges == 0 {
        return Err(ManifoldsError::NoGraphEdges);
    }
    let mut row_ptr: Vec<usize> = Vec::with_capacity(n + 1);
    let mut targets: Vec<u32> = Vec::with_capacity(n_edges);
    let mut weights: Vec<T> = Vec::with_capacity(n_edges);
    row_ptr.push(0);
    for row in graph {
        for &(j, w) in row {
            targets.push(j as u32);
            weights.push(w);
        }
        row_ptr.push(targets.len());
    }

    let max_weight = weights
        .iter()
        .fold(zero, |acc, &w| if w > acc { w } else { acc });

    let neg_sample_rate_t = T::from(params.neg_sample_rate).unwrap();
    let epochs_per_sample: Vec<T> = weights
        .iter()
        .map(|&w| {
            let norm = w / max_weight;
            if norm > zero {
                one / norm
            } else {
                large_epoch
            }
        })
        .collect();

    let mut schedule: Vec<EdgeSchedule<T>> = epochs_per_sample
        .iter()
        .map(|&eps| EdgeSchedule {
            next: eps,
            next_neg: eps / neg_sample_rate_t,
        })
        .collect();

    // one disjoint schedule slice per node, so the owner mutates it lock-free
    let mut node_schedules: Vec<&mut [EdgeSchedule<T>]> = Vec::with_capacity(n);
    let mut rest = schedule.as_mut_slice();
    for i in 0..n {
        let (head, tail) = rest.split_at_mut(row_ptr[i + 1] - row_ptr[i]);
        node_schedules.push(head);
        rest = tail;
    }

    let mut rng_states: Vec<SmallRng> = (0..n)
        .map(|i| SmallRng::seed_from_u64(seed + i as u64))
        .collect();

    let embd_shared: Vec<T::Atomic> = embd
        .iter()
        .flat_map(|point| point.iter().map(|&v| T::new_atomic(v)))
        .collect();

    let mut dens_scratch = dens.map(|_| UmapDensScratch::<T>::new(n));
    let mut dens_snapshot: Vec<T> = if dens.is_some() {
        vec![zero; n * n_dim]
    } else {
        Vec::new()
    };

    let n_epochs_f = T::from(params.n_epochs).unwrap();

    for epoch in 0..params.n_epochs {
        let lr = params.lr * (one - T::from(epoch).unwrap() / n_epochs_f);
        let epoch_t = T::from(epoch).unwrap();

        // densMAP radii from a snapshot at the start of the epoch, as the
        // reference does
        let dens_ctx = match (dens, dens_scratch.as_mut()) {
            (Some(state), Some(scratch)) if state.is_active(epoch, params.n_epochs) => {
                dens_snapshot
                    .par_iter_mut()
                    .zip(embd_shared.par_iter())
                    .for_each(|(dst, src)| *dst = T::load_relaxed(src));
                accumulate_umap_radii(
                    graph,
                    &dens_snapshot,
                    n_dim,
                    &consts,
                    |x| pow_lut.get(x),
                    scratch,
                );
                update_density_weights(scratch, state);
                Some(&*scratch)
            }
            _ => None,
        };

        // scratch: `buf` holds y_i, the other endpoint and the summed
        // repulsive gradient, `negs` the current edge's negative samples
        let new_scratch = || (vec![zero; 3 * n_dim], Vec::<usize>::new());

        let update_node = |(buf, negs): &mut (Vec<T>, Vec<usize>),
                           i: usize,
                           sched: &mut [EdgeSchedule<T>],
                           rng: &mut SmallRng| {
            let (y_i, rest) = buf.split_at_mut(n_dim);
            let (y_other, rep_grad) = rest.split_at_mut(n_dim);
            let base_i = i * n_dim;
            let row_start = row_ptr[i];
            let mut moved = false;

            for d in 0..n_dim {
                y_i[d] = T::load_relaxed(&embd_shared[base_i + d]);
            }

            for (e, st) in sched.iter_mut().enumerate() {
                if st.next > epoch_t {
                    continue;
                }
                moved = true;

                let edge_idx = row_start + e;
                let j = targets[edge_idx] as usize;
                let base_j = j * n_dim;
                let per_neg = epochs_per_sample[edge_idx] / neg_sample_rate_t;

                let mut dist_sq = zero;
                for d in 0..n_dim {
                    y_other[d] = T::load_relaxed(&embd_shared[base_j + d]);
                    let diff = y_i[d] - y_other[d];
                    dist_sq += diff * diff;
                }

                // attractive
                if dist_sq >= dist_sq_threshold {
                    let dist_sq_b = pow_lut.get(dist_sq);
                    let denom = one + consts.a * dist_sq_b;
                    let grad_coeff = consts.two_a_b * dist_sq_b / (dist_sq * denom);

                    // the edge list is directed, so each undirected edge is
                    // visited from both sides and both endpoints move on each
                    // visit, exactly the reference's layout: no extra factor
                    // here, unlike the CSR-based parallel Adam
                    let two_cor = match dens_ctx {
                        Some(scratch) => {
                            two * density_edge_coeff(
                                i,
                                j,
                                weights[edge_idx],
                                dist_sq,
                                dist_sq_b,
                                denom,
                                &consts,
                                scratch,
                            )
                        }
                        None => zero,
                    };

                    for d in 0..n_dim {
                        let delta = y_other[d] - y_i[d];
                        let mut grad_d = (grad_coeff * delta)
                            .max(-consts.clip_val)
                            .min(consts.clip_val);

                        if dens_ctx.is_some() {
                            grad_d += (two_cor * -delta)
                                .max(-consts.clip_val)
                                .min(consts.clip_val);
                        }

                        y_i[d] += grad_d * lr;
                        T::store_relaxed(&embd_shared[base_j + d], y_other[d] - grad_d * lr);
                    }
                }

                st.next += epochs_per_sample[edge_idx];

                // repulsive, via negative sampling. Draw and prefetch all
                // samples first, then evaluate them against the same y_i so
                // they are independent of each other.
                let n_neg_samples = ((epoch_t - st.next_neg) / per_neg)
                    .floor()
                    .to_usize()
                    .unwrap_or(0);

                negs.clear();
                for _ in 0..n_neg_samples {
                    let k = rng.random_range(0..n);
                    prefetch_read(&embd_shared[k * n_dim]);
                    negs.push(k);
                }

                rep_grad.fill(zero);
                for &k in negs.iter() {
                    if k == i {
                        continue;
                    }
                    let base_k = k * n_dim;

                    let mut dist_sq = zero;
                    for d in 0..n_dim {
                        y_other[d] = T::load_relaxed(&embd_shared[base_k + d]);
                        let diff = y_i[d] - y_other[d];
                        dist_sq += diff * diff;
                    }

                    let dist_sq_safe = dist_sq + rep_eps;
                    let dist_sq_b = pow_lut.get(dist_sq_safe);
                    let denom = dist_sq_safe * (one + consts.a * dist_sq_b);
                    let grad_coeff = (consts.two_gamma_b / denom)
                        .max(-consts.clip_val)
                        .min(consts.clip_val);

                    for d in 0..n_dim {
                        rep_grad[d] += grad_coeff * (y_i[d] - y_other[d]);
                    }
                }
                for d in 0..n_dim {
                    y_i[d] += rep_grad[d] * lr;
                }

                st.next_neg += T::from(n_neg_samples).unwrap() * per_neg;
            }

            // skip the write-back when nothing fired, so a stale copy never
            // clobbers updates other threads made to this point
            if moved {
                for d in 0..n_dim {
                    T::store_relaxed(&embd_shared[base_i + d], y_i[d]);
                }
            }
        };

        if parallel {
            node_schedules
                .par_iter_mut()
                .zip(rng_states.par_iter_mut())
                .enumerate()
                .for_each_init(new_scratch, |scratch, (i, (sched, rng))| {
                    update_node(scratch, i, sched, rng)
                });
        } else {
            let mut scratch = new_scratch();
            for (i, (sched, rng)) in node_schedules
                .iter_mut()
                .zip(rng_states.iter_mut())
                .enumerate()
            {
                update_node(&mut scratch, i, sched, rng);
            }
        }

        if verbosity.normal_verbosity() && ((epoch + 1) % 50 == 0 || epoch + 1 == params.n_epochs) {
            println!("  Epoch {}/{}", epoch + 1, params.n_epochs);
        }
    }

    for (i, point) in embd.iter_mut().enumerate() {
        for (d, v) in point.iter_mut().enumerate() {
            *v = T::load_relaxed(&embd_shared[i * n_dim + d]);
        }
    }

    Ok(())
}

/// Optimise UMAP embedding using Adam optimiser (sequential version)
///
/// Implements UMAP optimisation using the Adam adaptive learning rate
/// algorithm:
///
/// - Adaptive edge sampling based on edge weights
/// - First and second moment estimation (momentum and RMSprop)
/// - Bias correction for moment estimates
/// - Bias correction advanced once per epoch, as in the other Adam paths
/// - Negative sampling for repulsive forces
///
/// ### Algorithm
///
/// For each epoch:
///
/// 1. Process edges whose `epoch_of_next_sample` has arrived
/// 2. Compute attractive gradient and apply Adam update to both endpoints
/// 3. Perform negative sampling and apply repulsive Adam updates
/// 4. Advance the bias correction at the end of the epoch
///
/// ### Params
///
/// * `embd` - Initial embedding coordinates (modified in place), shape
///   `[n_samples][n_dim]`
/// * `graph` - Adjacency list where `graph[i]` contains (neighbour_idx, weight)
///   pairs
/// * `params` - Optimisation parameters including Adam hyperparameters (beta1,
///   beta2, eps)
/// * `dens` - Optional density state. Adds the densMAP term to the attraction.
/// * `seed` - Random seed for negative sampling
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Notes
///
/// - Updates are applied immediately (not batched)
/// - Linear learning rate decay schedule
pub fn optimise_embedding_adam<T>(
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
    let mut dens_scratch = dens.map(|_| UmapDensScratch::<T>::new(n));

    let mut embd_flat: Vec<T> = Vec::with_capacity(n * n_dim);
    for point in embd.iter() {
        embd_flat.extend_from_slice(point);
    }

    let consts = OptimConstants::new(params.a, params.b, params.gamma);

    let zero = T::zero();
    let one = T::one();
    let half = T::from(0.5).unwrap();
    let dist_sq_threshold = T::from(1e-8).unwrap();
    let large_epoch = T::from(1e8).unwrap();

    let b_is_one = (consts.b - one).abs() < T::from(1e-10).unwrap();
    let b_is_half = (consts.b - half).abs() < T::from(1e-10).unwrap();

    if n > u32::MAX as usize {
        return Err(ManifoldsError::TooManyVertices { n });
    }

    let mut edges: Vec<(u32, u32, T)> =
        Vec::with_capacity(graph.iter().map(|neighbours| neighbours.len()).sum());
    for (i, neighbours) in graph.iter().enumerate() {
        for &(j, w) in neighbours {
            edges.push((i as u32, j as u32, w));
        }
    }

    if edges.is_empty() {
        return Err(ManifoldsError::NoGraphEdges);
    }

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

    let mut epoch_of_next_sample: Vec<T> = epochs_per_sample.clone();

    let neg_sample_rate_t = T::from(params.neg_sample_rate).unwrap();
    let mut epoch_of_next_neg_sample: Vec<T> = epochs_per_sample
        .iter()
        .map(|eps| *eps / neg_sample_rate_t)
        .collect();

    let n_epochs_f = T::from(params.n_epochs).unwrap();

    let mut m: Vec<T> = vec![zero; n * n_dim];
    let mut v: Vec<T> = vec![zero; n * n_dim];

    let mut rng_states: Vec<SmallRng> = (0..n)
        .map(|i| SmallRng::seed_from_u64(seed + i as u64))
        .collect();

    // Adam parameters matching C++ implementation
    let beta11 = one - params.beta1; // 1 - beta1
    let beta21 = one - params.beta2; // 1 - beta2
    let mut beta1t = params.beta1;
    let mut beta2t = params.beta2;

    for epoch in 0..params.n_epochs {
        // Compute bias-corrected learning rate parameters once per epoch (matching C++ epoch_end)
        let alpha = params.lr * (one - T::from(epoch).unwrap() / n_epochs_f);
        let sqrt_b2t1 = (one - beta2t).sqrt();
        let ad_scale = alpha * sqrt_b2t1 / (one - beta1t);
        let epsc = sqrt_b2t1 * params.eps;

        let epoch_t = T::from(epoch).unwrap();

        // pass 0: embedding local radii from the positions at the start of the
        // epoch, as the reference does. The edge loop below mutates in place.
        let dens_ctx = match (dens, dens_scratch.as_mut()) {
            (Some(state), Some(scratch)) if state.is_active(epoch, params.n_epochs) => {
                accumulate_umap_radii(
                    graph,
                    &embd_flat,
                    n_dim,
                    &consts,
                    |x| fast_pow(x, consts.b, b_is_one, b_is_half),
                    scratch,
                );
                update_density_weights(scratch, state);
                Some(&*scratch)
            }
            _ => None,
        };

        for (edge_idx, &(i, j, _weight)) in edges.iter().enumerate() {
            if epoch_of_next_sample[edge_idx] > epoch_t {
                continue;
            }

            let (i, j) = (i as usize, j as usize);
            let per_neg = epochs_per_sample[edge_idx] / neg_sample_rate_t;

            let base_i = i * n_dim;
            let base_j = j * n_dim;

            let mut dist_sq = zero;
            for d in 0..n_dim {
                let diff = embd_flat[base_i + d] - embd_flat[base_j + d];
                dist_sq += diff * diff;
            }

            if dist_sq >= dist_sq_threshold {
                let dist_sq_b = fast_pow(dist_sq, consts.b, b_is_one, b_is_half);
                let denom = one + consts.a * dist_sq_b;
                let grad_coeff = consts.two_a_b * dist_sq_b / (dist_sq * denom);

                // densMAP. Directed edge list, so both endpoints move on every
                // visit and each undirected edge is seen twice - the reference
                // layout, so no extra factor here.
                let two_cor = match dens_ctx {
                    Some(scratch) => {
                        let cor = density_edge_coeff(
                            i, j, _weight, dist_sq, dist_sq_b, denom, &consts, scratch,
                        );
                        T::from(2.0).unwrap() * cor
                    }
                    None => zero,
                };

                for d in 0..n_dim {
                    let delta = embd_flat[base_j + d] - embd_flat[base_i + d];
                    let mut grad = grad_coeff * delta;

                    if dens_ctx.is_some() {
                        grad += (two_cor * -delta)
                            .max(-consts.clip_val)
                            .min(consts.clip_val);
                    }

                    // Update i (matching C++ compact form)
                    let idx_i = base_i + d;
                    let v_old = v[idx_i];
                    let m_old = m[idx_i];
                    v[idx_i] = v_old + beta21 * (grad * grad - v_old);
                    m[idx_i] = m_old + beta11 * (grad - m_old);
                    embd_flat[idx_i] += ad_scale * m[idx_i] / (v[idx_i].sqrt() + epsc);

                    // Update j (negated gradient)
                    let idx_j = base_j + d;
                    let v_old = v[idx_j];
                    let m_old = m[idx_j];
                    v[idx_j] = v_old + beta21 * (grad * grad - v_old);
                    m[idx_j] = m_old + beta11 * (-grad - m_old);
                    embd_flat[idx_j] += ad_scale * m[idx_j] / (v[idx_j].sqrt() + epsc);
                }
            }

            epoch_of_next_sample[edge_idx] += epochs_per_sample[edge_idx];

            let n_neg_samples = ((epoch_t - epoch_of_next_neg_sample[edge_idx]) / per_neg)
                .floor()
                .to_usize()
                .unwrap_or(0);

            for _ in 0..n_neg_samples {
                let k = rng_states[i].random_range(0..n);
                if k == i {
                    continue;
                }

                let base_k = k * n_dim;

                let mut dist_sq = zero;
                for d in 0..n_dim {
                    let diff = embd_flat[base_i + d] - embd_flat[base_k + d];
                    dist_sq += diff * diff;
                }

                let dist_sq_safe = dist_sq + consts.eps;
                let dist_sq_b = fast_pow(dist_sq_safe, consts.b, b_is_one, b_is_half);
                let denom = dist_sq_safe * (one + consts.a * dist_sq_b);
                let grad_coeff = (consts.two_gamma_b / denom)
                    .max(-consts.clip_val)
                    .min(consts.clip_val);

                for d in 0..n_dim {
                    let delta = embd_flat[base_i + d] - embd_flat[base_k + d];
                    let grad = grad_coeff * delta;

                    let idx = base_i + d;
                    let v_old = v[idx];
                    let m_old = m[idx];
                    v[idx] = v_old + beta21 * (grad * grad - v_old);
                    m[idx] = m_old + beta11 * (grad - m_old);
                    embd_flat[idx] += ad_scale * m[idx] / (v[idx].sqrt() + epsc);
                }
            }

            epoch_of_next_neg_sample[edge_idx] += T::from(n_neg_samples).unwrap() * per_neg;
        }

        // Update bias correction factors for next epoch (matching C++ epoch_end)
        beta1t *= params.beta1;
        beta2t *= params.beta2;

        if verbosity.normal_verbosity() && ((epoch + 1) % 50 == 0 || epoch + 1 == params.n_epochs) {
            println!("  Epoch {}/{}", epoch + 1, params.n_epochs);
        }
    }

    for (i, point) in embd.iter_mut().enumerate() {
        let base = i * n_dim;
        point.copy_from_slice(&embd_flat[base..base + n_dim]);
    }

    Ok(())
}

/// Optimise UMAP embedding using Adam optimiser (parallel batch version)
///
/// Implements uwot's `BatchUpdate` with `NodeWorker` behaviour:
///
/// - Parallelises over nodes using Rayon
/// - Accumulates gradients per node per epoch safely via chunking
/// - Applies Adam updates and edge schedules in parallel
/// - Utilises CSR (Compressed Sparse Row) graph structures for cache locality
/// - Utilises LUTs to avoid expensive float powers
///
/// ### Params
///
/// * `embd` - Initial embedding, modified in place
/// * `graph` - Adjacency list representation
/// * `params` - Includes Adam hyperparameters
/// * `dens` - Density-preserving state for densMAP, or `None` for plain UMAP.
///   When set, the density gradient is applied over the final
///   `dens.params.frac` of the epochs. Negative sampling is untouched either
///   way.
/// * `seed` - Random seed
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### References
///
/// Narayan, Berger & Cho, Nature Biotechnology, 2021 (densMAP).
pub fn optimise_embedding_adam_parallel<T>(
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
    let mut dens_scratch = dens.map(|_| UmapDensScratch::<T>::new(n));

    let mut embd_flat: Vec<T> = Vec::with_capacity(n * n_dim);
    for point in embd.iter() {
        embd_flat.extend_from_slice(point);
    }

    let consts = OptimConstants::new(params.a, params.b, params.gamma);

    // pre-compute a LUT for x^b (mapping squared distances up to 25.0)
    let b_is_one = (consts.b - T::one()).abs() < T::from(1e-10).unwrap();
    let lut = FastPowLut::new(consts.b, 25.0, 65_536);
    let dist_sq_threshold = T::from(GRAD_DIST_SQ_THRESHOLD).unwrap();

    let csr = UmapCsrGraph::from_graph(graph)?;
    let neg_sample_rate_t = T::from(params.neg_sample_rate).unwrap();

    let mut epoch_of_next_sample: Vec<T> = csr.epochs_per_sample.clone();
    let mut epoch_of_next_neg_sample: Vec<T> = csr
        .epochs_per_sample
        .iter()
        .map(|eps| *eps / neg_sample_rate_t)
        .collect();

    let n_epochs_f = T::from(params.n_epochs).unwrap();
    let lr_schedule: Vec<T> = (0..params.n_epochs)
        .map(|e| params.lr * (T::one() - T::from(e).unwrap() / n_epochs_f))
        .collect();

    let mut m: Vec<T> = vec![T::zero(); n * n_dim];
    let mut v: Vec<T> = vec![T::zero(); n * n_dim];

    let bias_corrections: Vec<(T, T)> = (0..params.n_epochs)
        .map(|epoch| {
            let t = T::from(epoch + 1).unwrap();
            let beta1t = params.beta1.powf(t);
            let beta2t = params.beta2.powf(t);
            let sqrt_b2t1 = (T::one() - beta2t).sqrt();
            let ad_scale = sqrt_b2t1 / (T::one() - beta1t);
            let epsc = sqrt_b2t1 * params.eps;
            (ad_scale, epsc)
        })
        .collect();

    let one_minus_beta1 = T::one() - params.beta1;
    let one_minus_beta2 = T::one() - params.beta2;

    let mut node_gradients_all: Vec<T> = vec![T::zero(); n * n_dim];
    let mut node_has_update: Vec<bool> = vec![false; n];

    // stateful RNG instantiated once per thread/node - should be faster ... ?
    let mut node_rngs: Vec<SmallRng> = (0..n)
        .map(|i| SmallRng::seed_from_u64(seed + i as u64))
        .collect();

    for epoch in 0..params.n_epochs {
        let lr = lr_schedule[epoch];
        let epoch_t = T::from(epoch).unwrap();
        let (ad_scale, epsc) = bias_corrections[epoch];

        // reset state
        node_has_update.fill(false);

        // pass 0: embedding local radii over every edge, then the correlation
        // statistics. Reads the same frozen embd_flat as the gradient pass, so
        // both see identical positions.
        let dens_ctx = match (dens, dens_scratch.as_mut()) {
            (Some(state), Some(scratch)) if state.is_active(epoch, params.n_epochs) => {
                accumulate_umap_radii(
                    graph,
                    &embd_flat,
                    n_dim,
                    &consts,
                    |x| if b_is_one { x } else { lut.get(x) },
                    scratch,
                );
                update_density_weights(scratch, state);
                Some(&*scratch)
            }
            _ => None,
        };

        // safely partition gradients per node
        node_gradients_all
            .par_chunks_exact_mut(n_dim)
            .zip(node_has_update.par_iter_mut())
            .zip(node_rngs.par_iter_mut())
            .enumerate()
            .for_each(|(node_i, ((node_grad, has_update), rng))| {
                // Clear old gradients
                for g in node_grad.iter_mut() {
                    *g = T::zero();
                }

                let mut local_has_updates = false;
                let base_i = node_i * n_dim;

                let start_idx = csr.node_edge_offsets[node_i] as usize;
                let end_idx = csr.node_edge_offsets[node_i + 1] as usize;

                for (&edge_idx, &other_node) in csr.csr_edge_idx[start_idx..end_idx]
                    .iter()
                    .zip(&csr.csr_other_node[start_idx..end_idx])
                {
                    let edge_idx = edge_idx as usize;
                    let other_node = other_node as usize;
                    if epoch_of_next_sample[edge_idx] > epoch_t {
                        continue;
                    }

                    local_has_updates = true;
                    let base_other = other_node * n_dim;

                    let mut dist_sq = T::zero();
                    for d in 0..n_dim {
                        let diff = embd_flat[base_i + d] - embd_flat[base_other + d];
                        dist_sq += diff * diff;
                    }

                    if dist_sq >= dist_sq_threshold {
                        let dist_sq_b = if b_is_one { dist_sq } else { lut.get(dist_sq) };
                        let denom = T::one() + consts.a * dist_sq_b;
                        let grad_coeff = consts.two_a_b * dist_sq_b / (dist_sq * denom);
                        let two = T::from(2.0).unwrap();

                        for d in 0..n_dim {
                            let delta = embd_flat[base_other + d] - embd_flat[base_i + d];
                            node_grad[d] += two * grad_coeff * delta;
                        }

                        // densMAP. The outer factor 2 mirrors the attractive
                        // term above: the reference visits each undirected edge
                        // from both COO directions, this CSR walk visits it
                        // once per endpoint. Signed along (y_i - y_other), i.e.
                        // opposite to the attractive delta, since this is
                        // ascent on the correlation.
                        if let Some(scratch) = dens_ctx {
                            let cor = density_edge_coeff(
                                node_i,
                                other_node,
                                csr.mu[edge_idx],
                                dist_sq,
                                dist_sq_b,
                                denom,
                                &consts,
                                scratch,
                            );
                            let two_cor = two * cor;

                            for d in 0..n_dim {
                                let delta = embd_flat[base_i + d] - embd_flat[base_other + d];
                                node_grad[d] += two
                                    * (two_cor * delta).max(-consts.clip_val).min(consts.clip_val);
                            }
                        }
                    }

                    // the smaller endpoint owns the edge's negative samples
                    if node_i < other_node {
                        let n_neg_samples = ((epoch_t - epoch_of_next_neg_sample[edge_idx])
                            / (csr.epochs_per_sample[edge_idx] / neg_sample_rate_t))
                            .floor()
                            .to_usize()
                            .unwrap_or(0);

                        for _ in 0..n_neg_samples {
                            let k = rng.random_range(0..n);
                            if k == node_i {
                                continue;
                            }

                            let base_k = k * n_dim;

                            let mut dist_sq = T::zero();
                            for d in 0..n_dim {
                                let diff = embd_flat[base_i + d] - embd_flat[base_k + d];
                                dist_sq += diff * diff;
                            }

                            let dist_sq_safe = dist_sq + consts.eps;
                            let dist_sq_b = if b_is_one {
                                dist_sq_safe
                            } else {
                                lut.get(dist_sq_safe)
                            };

                            let denom = dist_sq_safe * (T::one() + consts.a * dist_sq_b);
                            let grad_coeff = (consts.two_gamma_b / denom)
                                .max(-consts.clip_val)
                                .min(consts.clip_val);

                            for d in 0..n_dim {
                                let delta = embd_flat[base_i + d] - embd_flat[base_k + d];
                                node_grad[d] += grad_coeff * delta;
                            }
                        }
                    }
                }

                if local_has_updates {
                    *has_update = true;
                }
            });

        // parallelise Adam moments and embedding updates
        node_gradients_all
            .par_chunks_exact_mut(n_dim)
            .zip(m.par_chunks_exact_mut(n_dim))
            .zip(v.par_chunks_exact_mut(n_dim))
            .zip(embd_flat.par_chunks_exact_mut(n_dim))
            .zip(node_has_update.par_iter())
            .for_each(|((((grad, m_node), v_node), embd_node), &has_update)| {
                if !has_update {
                    return;
                }

                for d in 0..n_dim {
                    let g = grad[d];

                    let m_old = m_node[d];
                    m_node[d] += one_minus_beta1 * (g - m_old);

                    let v_old = v_node[d];
                    v_node[d] += one_minus_beta2 * (g * g - v_old);

                    embd_node[d] += lr * ad_scale * m_node[d] / (v_node[d].sqrt() + epsc);
                }
            });

        // parallelise edge schedules
        epoch_of_next_sample
            .par_iter_mut()
            .zip(epoch_of_next_neg_sample.par_iter_mut())
            .zip(csr.epochs_per_sample.par_iter())
            .for_each(|((next_sample, next_neg), &per_sample)| {
                if *next_sample <= epoch_t {
                    let per_neg = per_sample / neg_sample_rate_t;
                    *next_sample += per_sample;

                    let n_neg_samples = ((epoch_t - *next_neg) / per_neg)
                        .floor()
                        .to_usize()
                        .unwrap_or(0);

                    *next_neg += T::from(n_neg_samples).unwrap() * per_neg;
                }
            });

        if verbosity.normal_verbosity() && ((epoch + 1) % 50 == 0 || epoch + 1 == params.n_epochs) {
            println!("  Epoch {}/{}", epoch + 1, params.n_epochs);
        }
    }

    // move flat embeddings back to target
    embd.par_iter_mut().enumerate().for_each(|(i, point)| {
        let base = i * n_dim;
        point.copy_from_slice(&embd_flat[base..base + n_dim]);
    });

    Ok(())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_umap_optimiser {
    use super::*;
    use approx::assert_relative_eq;
    use num_traits::Float;

    #[inline(always)]
    fn squared_dist_flat<T>(embd: &[T], i: usize, j: usize, n_dim: usize) -> T
    where
        T: Float,
    {
        let mut sum = T::zero();
        let base_i = i * n_dim;
        let base_j = j * n_dim;
        for d in 0..n_dim {
            let diff = embd[base_i + d] - embd[base_j + d];
            sum = sum + diff * diff;
        }
        sum
    }

    //////////
    // UMAP //
    //////////

    #[test]
    fn test_optim_params_default_2d() {
        let params = UmapOptimParams::<f64>::default_2d();

        assert_eq!(params.lr, 1.0);
        assert_eq!(params.gamma, 1.0);
        assert_eq!(params.n_epochs, 500);
        assert_eq!(params.neg_sample_rate, 5);
        assert_relative_eq!(params.min_dist, DEFAULT_MIN_DIST, epsilon = 1e-6);
        assert!(params.a > 0.0);
        assert!(params.b > 0.0);
    }

    #[test]
    fn test_default_2d_curve_is_fitted_not_hardcoded() {
        // The contract, rather than the two fitted floats: `default_2d` has to
        // be exactly what a caller gets by asking for the default `min_dist`
        // and `spread` explicitly. Pinning `a` and `b` as literals is what let
        // the two disagree in the first place.
        let implicit = UmapOptimParams::<f64>::default_2d();
        let explicit = UmapOptimParams::<f64>::from_min_dist_spread(
            DEFAULT_MIN_DIST,
            DEFAULT_SPREAD,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        assert_relative_eq!(implicit.a, explicit.a, epsilon = 1e-12);
        assert_relative_eq!(implicit.b, explicit.b, epsilon = 1e-12);
        assert_relative_eq!(implicit.min_dist, explicit.min_dist, epsilon = 1e-12);
    }

    #[test]
    fn test_optim_params_from_min_dist_spread() {
        let params = UmapOptimParams::<f64>::from_min_dist_spread(
            0.1,
            1.0,
            Some(1.0),
            Some(1.0),
            Some(500),
            Some(5),
            None,
            None,
            None,
        );

        assert!(params.a > 0.0);
        assert!(params.b > 0.0);
        assert_eq!(params.lr, 1.0);
        assert_eq!(params.gamma, 1.0);
        assert_eq!(params.n_epochs, 500);
        assert_eq!(params.neg_sample_rate, 5);
        assert_relative_eq!(params.min_dist, 0.1, epsilon = 1e-6);
    }

    #[test]
    fn test_fit_params_constraints() {
        let (a, b) = UmapOptimParams::<f64>::fit_params(0.1, 1.0, None);

        assert!((0.001..=10.0).contains(&a));
        assert!((0.1..=2.0).contains(&b));
    }

    #[test]
    fn test_fit_params_curve_properties() {
        let min_dist = 0.1;
        let spread = 1.0;
        let (a, b) = UmapOptimParams::<f64>::fit_params(min_dist, spread, None);

        // At min_dist, target is 1.0
        let pred_min = 1.0 / (1.0 + a * min_dist.powf(2.0 * b));
        assert!(
            pred_min > 0.9,
            "f(min_dist) = {:.3} should be > 0.9",
            pred_min
        );

        // At 3*spread, target is exp(-(3*spread - min_dist)/spread) ≈ 0.055
        let pred_spread = 1.0 / (1.0 + a * (3.0 * spread).powf(2.0 * b));
        assert!(
            pred_spread < 0.1,
            "f(3*spread) = {:.3} should be < 0.1",
            pred_spread
        );

        // Should be monotonically decreasing
        let mid_point = 1.5 * spread;
        let pred_mid = 1.0 / (1.0 + a * mid_point.powf(2.0 * b));
        assert!(pred_min > pred_mid && pred_mid > pred_spread);
    }

    #[test]
    fn test_squared_dist_basic() {
        let embd = vec![0.0, 0.0, 3.0, 4.0];
        let dist = squared_dist_flat(&embd, 0, 1, 2);
        assert_relative_eq!(dist, 25.0, epsilon = 1e-6);
    }

    #[test]
    fn test_squared_dist_identical_points() {
        let embd = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let dist = squared_dist_flat(&embd, 0, 1, 3);
        assert_relative_eq!(dist, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_optimise_embedding_adam_basic() {
        let graph = vec![
            vec![(1, 1.0), (2, 0.5)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 0.5), (1, 1.0)],
        ];

        let mut embd = vec![vec![0.0, 0.0], vec![5.0, 0.0], vec![0.0, 5.0]];
        let initial_embd = embd.clone();

        let params = UmapOptimParams::default_2d();
        let _ = optimise_embedding_adam(&mut embd, &graph, &params, None, 42, 0);

        let total_movement: f64 = embd
            .iter()
            .zip(initial_embd.iter())
            .map(|(new, old)| {
                new.iter()
                    .zip(old.iter())
                    .map(|(&n, &o)| (n - o).abs())
                    .sum::<f64>()
            })
            .sum();

        assert!(total_movement > 0.01);

        for point in &embd {
            for &coord in point {
                assert!(coord.is_finite());
            }
        }
    }

    #[test]
    fn test_optimise_embedding_adam_parallel_basic() {
        let graph = vec![
            vec![(1, 1.0), (2, 0.5)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 0.5), (1, 1.0)],
        ];

        let mut embd = vec![vec![0.0, 0.0], vec![5.0, 0.0], vec![0.0, 5.0]];
        let initial_embd = embd.clone();

        let params = UmapOptimParams::default_2d();
        let _ = optimise_embedding_adam_parallel(&mut embd, &graph, &params, None, 42, 0);

        let total_movement: f64 = embd
            .iter()
            .zip(initial_embd.iter())
            .map(|(new, old)| {
                new.iter()
                    .zip(old.iter())
                    .map(|(&n, &o)| (n - o).abs())
                    .sum::<f64>()
            })
            .sum();

        assert!(total_movement > 0.01);

        for point in &embd {
            for &coord in point {
                assert!(coord.is_finite());
            }
        }
    }

    #[test]
    fn test_optimise_embedding_empty_graph() {
        let graph: Vec<Vec<(usize, f64)>> = vec![vec![], vec![], vec![]];
        let mut embd = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];

        let params = UmapOptimParams::default_2d();
        let _ = optimise_embedding_adam(&mut embd, &graph, &params, None, 42, 0);

        for point in &embd {
            for &coord in point {
                assert!(coord.is_finite());
            }
        }
    }

    #[test]
    fn test_optimise_embedding_adam_reproducibility() {
        let graph = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
        let mut embd1 = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let mut embd2 = vec![vec![0.0, 0.0], vec![1.0, 0.0]];

        let params = UmapOptimParams {
            a: 1.0,
            b: 1.0,
            lr: 0.5,
            gamma: 1.0,
            n_epochs: 10,
            neg_sample_rate: 2,
            min_dist: 0.1,
            beta1: 0.5,
            beta2: 0.9,
            eps: 1e-7,
        };

        let _ = optimise_embedding_adam(&mut embd1, &graph, &params, None, 42, 0);
        let _ = optimise_embedding_adam(&mut embd2, &graph, &params, None, 42, 0);

        assert_eq!(embd1, embd2);
    }

    #[test]
    fn test_optimise_embedding_adam_parallel_reproducibility() {
        let graph = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
        let mut embd1 = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let mut embd2 = vec![vec![0.0, 0.0], vec![1.0, 0.0]];

        let params = UmapOptimParams {
            a: 1.0,
            b: 1.0,
            lr: 0.5,
            gamma: 1.0,
            n_epochs: 10,
            neg_sample_rate: 2,
            min_dist: 0.1,
            beta1: 0.5,
            beta2: 0.9,
            eps: 1e-7,
        };

        let _ = optimise_embedding_adam_parallel(&mut embd1, &graph, &params, None, 42, 0);
        let _ = optimise_embedding_adam_parallel(&mut embd2, &graph, &params, None, 42, 0);

        assert_eq!(embd1, embd2);
    }

    #[test]
    fn test_optimise_embedding_convergence() {
        let graph = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
        let mut embd = vec![vec![0.0, 0.0], vec![10.0, 0.0]];

        let embd_flat: Vec<f64> = embd.iter().flatten().copied().collect();
        let initial_dist = squared_dist_flat(&embd_flat, 0, 1, 2).sqrt();

        let params = UmapOptimParams {
            a: 1.0,
            b: 1.0,
            lr: 1.0,
            gamma: 1.0,
            n_epochs: 100,
            neg_sample_rate: 2,
            min_dist: 0.1,
            beta1: 0.5,
            beta2: 0.9,
            eps: 1e-7,
        };

        let _ = optimise_embedding_adam(&mut embd, &graph, &params, None, 42, 0);

        let embd_flat: Vec<f64> = embd.iter().flatten().copied().collect();
        let final_dist = squared_dist_flat(&embd_flat, 0, 1, 2).sqrt();

        assert!(final_dist < initial_dist);
    }

    #[test]
    fn test_sgd_vs_adam_both_converge() {
        let graph = vec![
            vec![(1, 1.0), (2, 0.5)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 0.5), (1, 1.0)],
        ];

        let initial_embd = vec![vec![0.0, 0.0], vec![10.0, 0.0], vec![0.0, 10.0]];

        let params = UmapOptimParams {
            a: 1.0,
            b: 1.0,
            lr: 1.0,
            gamma: 1.0,
            n_epochs: 50,
            neg_sample_rate: 2,
            min_dist: 0.1,
            beta1: 0.5,
            beta2: 0.9,
            eps: 1e-7,
        };

        let mut embd_sgd = initial_embd.clone();
        let _ = optimise_embedding_sgd(&mut embd_sgd, &graph, &params, None, 42, 0);

        let mut embd_adam = initial_embd.clone();
        let _ = optimise_embedding_adam(&mut embd_adam, &graph, &params, None, 42, 0);

        let movement_sgd: f64 = embd_sgd
            .iter()
            .zip(initial_embd.iter())
            .map(|(new, old)| {
                new.iter()
                    .zip(old.iter())
                    .map(|(&n, &o)| (n - o).abs())
                    .sum::<f64>()
            })
            .sum();

        let movement_adam: f64 = embd_adam
            .iter()
            .zip(initial_embd.iter())
            .map(|(new, old)| {
                new.iter()
                    .zip(old.iter())
                    .map(|(&n, &o)| (n - o).abs())
                    .sum::<f64>()
            })
            .sum();

        assert!(movement_sgd > 1.0);
        assert!(movement_adam > 1.0);

        for point in embd_sgd.iter().chain(embd_adam.iter()) {
            for &coord in point {
                assert!(coord.is_finite());
            }
        }
    }

    #[test]
    fn test_sgd_adam_adam_parallel_all_converge() {
        let graph = vec![
            vec![(1, 1.0), (2, 0.5)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 0.5), (1, 1.0)],
        ];

        let initial_embd = vec![vec![0.0, 0.0], vec![10.0, 0.0], vec![0.0, 10.0]];

        let params = UmapOptimParams {
            a: 1.0,
            b: 1.0,
            lr: 1.0,
            gamma: 1.0,
            n_epochs: 50,
            neg_sample_rate: 2,
            min_dist: 0.1,
            beta1: 0.5,
            beta2: 0.9,
            eps: 1e-7,
        };

        let mut embd_sgd = initial_embd.clone();
        let _ = optimise_embedding_sgd(&mut embd_sgd, &graph, &params, None, 42, 0);

        let mut embd_adam = initial_embd.clone();
        let _ = optimise_embedding_adam(&mut embd_adam, &graph, &params, None, 42, 0);

        let mut embd_adam_par = initial_embd.clone();
        let _ = optimise_embedding_adam_parallel(&mut embd_adam_par, &graph, &params, None, 42, 0);

        let movement_sgd: f64 = embd_sgd
            .iter()
            .zip(initial_embd.iter())
            .flat_map(|(new, old)| new.iter().zip(old.iter()).map(|(&n, &o)| (n - o).abs()))
            .sum();

        let movement_adam: f64 = embd_adam
            .iter()
            .zip(initial_embd.iter())
            .flat_map(|(new, old)| new.iter().zip(old.iter()).map(|(&n, &o)| (n - o).abs()))
            .sum();

        let movement_adam_par: f64 = embd_adam_par
            .iter()
            .zip(initial_embd.iter())
            .flat_map(|(new, old)| new.iter().zip(old.iter()).map(|(&n, &o)| (n - o).abs()))
            .sum();

        assert!(movement_sgd > 1.0);
        assert!(movement_adam > 1.0);
        assert!(movement_adam_par > 1.0);

        for point in embd_sgd
            .iter()
            .chain(embd_adam.iter())
            .chain(embd_adam_par.iter())
        {
            for &coord in point {
                assert!(coord.is_finite());
            }
        }
    }

    #[test]
    fn test_sgd_reproducibility() {
        let graph = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
        let mut embd1 = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let mut embd2 = vec![vec![0.0, 0.0], vec![1.0, 0.0]];

        let params = UmapOptimParams {
            a: 1.0,
            b: 1.0,
            lr: 0.5,
            gamma: 1.0,
            n_epochs: 10,
            neg_sample_rate: 2,
            min_dist: 0.1,
            beta1: 0.5,
            beta2: 0.9,
            eps: 1e-7,
        };

        let _ = optimise_embedding_sgd(&mut embd1, &graph, &params, None, 42, 0);
        let _ = optimise_embedding_sgd(&mut embd2, &graph, &params, None, 42, 0);

        assert_eq!(embd1, embd2);
    }

    #[test]
    fn test_optimisation_preserves_graph_structure_adam() {
        let graph = vec![
            vec![(1, 1.0), (2, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 1.0), (1, 1.0), (3, 0.1)],
            vec![(2, 0.1), (4, 1.0), (5, 1.0)],
            vec![(3, 1.0), (5, 1.0)],
            vec![(3, 1.0), (4, 1.0)],
        ];

        let mut embd = vec![
            vec![0.0, 0.0],
            vec![10.0, 0.0],
            vec![0.0, 10.0],
            vec![10.0, 10.0],
            vec![-5.0, -5.0],
            vec![15.0, 15.0],
        ];

        let params = UmapOptimParams {
            n_epochs: 200,
            ..UmapOptimParams::default_2d()
        };

        let _ = optimise_embedding_adam(&mut embd, &graph, &params, None, 42, 0);

        let dist = |a: &[f64], b: &[f64]| -> f64 {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };

        let intra_clique1 =
            (dist(&embd[0], &embd[1]) + dist(&embd[0], &embd[2]) + dist(&embd[1], &embd[2])) / 3.0;
        let intra_clique2 =
            (dist(&embd[3], &embd[4]) + dist(&embd[3], &embd[5]) + dist(&embd[4], &embd[5])) / 3.0;
        let avg_intra = (intra_clique1 + intra_clique2) / 2.0;

        let inter_distances = [
            dist(&embd[0], &embd[3]),
            dist(&embd[0], &embd[4]),
            dist(&embd[0], &embd[5]),
            dist(&embd[1], &embd[3]),
            dist(&embd[1], &embd[4]),
            dist(&embd[1], &embd[5]),
        ];
        let avg_inter: f64 = inter_distances.iter().sum::<f64>() / inter_distances.len() as f64;

        assert!(
            avg_inter > avg_intra * 1.5,
            "Inter-clique dist ({:.2}) should be > 1.5x intra-clique dist ({:.2})",
            avg_inter,
            avg_intra
        );
    }

    #[test]
    fn test_optimisation_preserves_graph_structure_adam_parallel() {
        let graph = vec![
            vec![(1, 1.0), (2, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 1.0), (1, 1.0), (3, 0.1)],
            vec![(2, 0.1), (4, 1.0), (5, 1.0)],
            vec![(3, 1.0), (5, 1.0)],
            vec![(3, 1.0), (4, 1.0)],
        ];

        let mut embd = vec![
            vec![0.0, 0.0],
            vec![10.0, 0.0],
            vec![0.0, 10.0],
            vec![10.0, 10.0],
            vec![-5.0, -5.0],
            vec![15.0, 15.0],
        ];

        let params = UmapOptimParams {
            n_epochs: 200,
            ..UmapOptimParams::default_2d()
        };

        let _ = optimise_embedding_adam_parallel(&mut embd, &graph, &params, None, 42, 0);

        let dist = |a: &[f64], b: &[f64]| -> f64 {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };

        let intra_clique1 =
            (dist(&embd[0], &embd[1]) + dist(&embd[0], &embd[2]) + dist(&embd[1], &embd[2])) / 3.0;
        let intra_clique2 =
            (dist(&embd[3], &embd[4]) + dist(&embd[3], &embd[5]) + dist(&embd[4], &embd[5])) / 3.0;
        let avg_intra = (intra_clique1 + intra_clique2) / 2.0;

        let inter_distances = [
            dist(&embd[0], &embd[3]),
            dist(&embd[0], &embd[4]),
            dist(&embd[0], &embd[5]),
            dist(&embd[1], &embd[3]),
            dist(&embd[1], &embd[4]),
            dist(&embd[1], &embd[5]),
        ];
        let avg_inter: f64 = inter_distances.iter().sum::<f64>() / inter_distances.len() as f64;

        assert!(
            avg_inter > avg_intra * 1.5,
            "Inter-clique dist ({:.2}) should be > 1.5x intra-clique dist ({:.2})",
            avg_inter,
            avg_intra
        );
    }

    #[test]
    fn test_optimisation_preserves_graph_structure_sgd() {
        let graph = vec![
            vec![(1, 1.0), (2, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 1.0), (1, 1.0), (3, 0.1)],
            vec![(2, 0.1), (4, 1.0), (5, 1.0)],
            vec![(3, 1.0), (5, 1.0)],
            vec![(3, 1.0), (4, 1.0)],
        ];

        let mut embd = vec![
            vec![0.0, 0.0],
            vec![10.0, 0.0],
            vec![0.0, 10.0],
            vec![10.0, 10.0],
            vec![-5.0, -5.0],
            vec![15.0, 15.0],
        ];

        let params = UmapOptimParams {
            n_epochs: 200,
            ..UmapOptimParams::default_2d()
        };

        let _ = optimise_embedding_sgd(&mut embd, &graph, &params, None, 42, 0);

        let dist = |a: &[f64], b: &[f64]| -> f64 {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };

        let intra_clique1 =
            (dist(&embd[0], &embd[1]) + dist(&embd[0], &embd[2]) + dist(&embd[1], &embd[2])) / 3.0;
        let intra_clique2 =
            (dist(&embd[3], &embd[4]) + dist(&embd[3], &embd[5]) + dist(&embd[4], &embd[5])) / 3.0;
        let avg_intra = (intra_clique1 + intra_clique2) / 2.0;

        let inter_distances = [
            dist(&embd[0], &embd[3]),
            dist(&embd[0], &embd[4]),
            dist(&embd[0], &embd[5]),
            dist(&embd[1], &embd[3]),
            dist(&embd[1], &embd[4]),
            dist(&embd[1], &embd[5]),
        ];
        let avg_inter: f64 = inter_distances.iter().sum::<f64>() / inter_distances.len() as f64;

        assert!(
            avg_inter > avg_intra * 1.5,
            "Inter-clique dist ({:.2}) should be > 1.5x intra-clique dist ({:.2})",
            avg_inter,
            avg_intra
        );
    }

    /// Random symmetric kNN-like graph and random 2D initialisation.
    ///
    /// ### Params
    ///
    /// * `n` - Number of points
    /// * `k` - Out-neighbours drawn per point before symmetrisation
    /// * `seed` - RNG seed
    ///
    /// ### Returns
    ///
    /// `(graph, embedding)` with distinct weights so the edge schedule varies.
    #[allow(clippy::type_complexity)]
    fn random_graph_and_embedding(
        n: usize,
        k: usize,
        seed: u64,
    ) -> (Vec<Vec<(usize, f64)>>, Vec<Vec<f64>>) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut graph: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for i in 0..n {
            for _ in 0..k {
                let j = rng.random_range(0..n);
                if j == i || graph[i].iter().any(|&(o, _)| o == j) {
                    continue;
                }
                let w: f64 = rng.random_range(0.05..1.0);
                graph[i].push((j, w));
                graph[j].push((i, w));
            }
        }
        let embd = (0..n)
            .map(|_| vec![rng.random_range(-10.0..10.0), rng.random_range(-10.0..10.0)])
            .collect();
        (graph, embd)
    }

    /// Run a closure inside a one-thread Rayon pool.
    ///
    /// ### Params
    ///
    /// * `f` - The work to run
    ///
    /// ### Returns
    ///
    /// Whatever `f` returns.
    fn single_threaded<R: Send>(f: impl FnOnce() -> R + Send) -> R {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("one-thread pool")
            .install(f)
    }

    #[test]
    fn test_sgd_parallel_one_thread_matches_sgd() {
        let (graph, embd0) = random_graph_and_embedding(200, 8, 7);
        let params = UmapOptimParams {
            n_epochs: 100,
            ..UmapOptimParams::default_2d()
        };

        let mut embd_seq = embd0.clone();
        optimise_embedding_sgd(&mut embd_seq, &graph, &params, None, 42, 0).unwrap();

        let mut embd_par = embd0.clone();
        single_threaded(|| {
            optimise_embedding_sgd_parallel(&mut embd_par, &graph, &params, None, 42, 0)
        })
        .unwrap();

        assert_eq!(embd_seq, embd_par);
    }

    #[test]
    fn test_sgd_parallel_one_thread_matches_sgd_densmap() {
        let (graph, embd0) = random_graph_and_embedding(200, 8, 11);
        let n = graph.len();
        let mut rng = SmallRng::seed_from_u64(3);
        // z-scored-ish radii; exact values do not matter, only that both paths
        // see the same state
        let r: Vec<f64> = (0..n).map(|_| rng.random_range(-1.5..1.5)).collect();
        let mu_sum: Vec<f64> = graph
            .iter()
            .map(|row| row.iter().map(|e| e.1).sum())
            .collect();
        let mu_tot = mu_sum.iter().sum();
        let state = DensState {
            params: DensParams::new(None, None, None),
            r,
            mu_sum,
            mu_tot,
        };
        let params = UmapOptimParams {
            n_epochs: 60,
            ..UmapOptimParams::default_2d()
        };

        let mut embd_seq = embd0.clone();
        optimise_embedding_sgd(&mut embd_seq, &graph, &params, Some(&state), 42, 0).unwrap();

        let mut embd_par = embd0.clone();
        single_threaded(|| {
            optimise_embedding_sgd_parallel(&mut embd_par, &graph, &params, Some(&state), 42, 0)
        })
        .unwrap();

        assert_eq!(embd_seq, embd_par);
    }

    #[test]
    fn test_sgd_parallel_multi_thread_finite() {
        let (graph, mut embd) = random_graph_and_embedding(2_000, 10, 5);
        let params = UmapOptimParams {
            n_epochs: 100,
            ..UmapOptimParams::default_2d()
        };

        optimise_embedding_sgd_parallel(&mut embd, &graph, &params, None, 42, 0).unwrap();

        assert!(embd.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn test_sgd_parallel_errors() {
        let params = UmapOptimParams::<f64>::default_2d();

        let mut empty: Vec<Vec<f64>> = Vec::new();
        assert!(matches!(
            optimise_embedding_sgd_parallel(&mut empty, &[], &params, None, 42, 0),
            Err(ManifoldsError::NoData)
        ));

        let graph: Vec<Vec<(usize, f64)>> = vec![vec![], vec![]];
        let mut embd = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        assert!(matches!(
            optimise_embedding_sgd_parallel(&mut embd, &graph, &params, None, 42, 0),
            Err(ManifoldsError::NoGraphEdges)
        ));
    }

    #[test]
    fn test_sgd_parallel_f32() {
        let (graph, embd) = random_graph_and_embedding(300, 8, 9);
        let graph: Vec<Vec<(usize, f32)>> = graph
            .into_iter()
            .map(|row| row.into_iter().map(|(j, w)| (j, w as f32)).collect())
            .collect();
        let mut embd: Vec<Vec<f32>> = embd
            .into_iter()
            .map(|p| p.into_iter().map(|v| v as f32).collect())
            .collect();
        let params = UmapOptimParams::<f32> {
            n_epochs: 50,
            ..UmapOptimParams::default_2d()
        };

        optimise_embedding_sgd_parallel(&mut embd, &graph, &params, None, 42, 0).unwrap();

        assert!(embd.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn test_optimisation_preserves_graph_structure_sgd_parallel() {
        let graph = vec![
            vec![(1, 1.0), (2, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 1.0), (1, 1.0), (3, 0.1)],
            vec![(2, 0.1), (4, 1.0), (5, 1.0)],
            vec![(3, 1.0), (5, 1.0)],
            vec![(3, 1.0), (4, 1.0)],
        ];

        let mut embd = vec![
            vec![0.0, 0.0],
            vec![10.0, 0.0],
            vec![0.0, 10.0],
            vec![10.0, 10.0],
            vec![-5.0, -5.0],
            vec![15.0, 15.0],
        ];

        let params = UmapOptimParams {
            n_epochs: 200,
            ..UmapOptimParams::default_2d()
        };

        optimise_embedding_sgd_parallel(&mut embd, &graph, &params, None, 42, 0).unwrap();

        let dist = |a: &[f64], b: &[f64]| -> f64 {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };

        let intra_clique1 =
            (dist(&embd[0], &embd[1]) + dist(&embd[0], &embd[2]) + dist(&embd[1], &embd[2])) / 3.0;
        let intra_clique2 =
            (dist(&embd[3], &embd[4]) + dist(&embd[3], &embd[5]) + dist(&embd[4], &embd[5])) / 3.0;
        let avg_intra = (intra_clique1 + intra_clique2) / 2.0;

        let mut inter = 0.0;
        for &a in &[0, 1] {
            for &b in &[3, 4, 5] {
                inter += dist(&embd[a], &embd[b]);
            }
        }
        let avg_inter = inter / 6.0;

        assert!(
            avg_inter > avg_intra * 1.5,
            "Inter-clique dist ({:.2}) should be > 1.5x intra-clique dist ({:.2})",
            avg_inter,
            avg_intra
        );
    }
}
