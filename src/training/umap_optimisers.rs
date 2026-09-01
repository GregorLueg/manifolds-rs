//! Optimisers for UMAP fitting. Contains the SGD, Adam and a parallel Adam
//! variant

use rand::{
    rngs::SmallRng,
    {Rng, SeedableRng},
};
use rayon::prelude::*;

use crate::prelude::*;
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
/// The single place the default lives. [`UmapOptimParams::default_2d`] and
/// [`crate::UmapParams::new_default_2d`] both read it, so the two cannot drift
/// apart the way they did when each wrote its own.
pub const DEFAULT_MIN_DIST: f64 = 0.5;

/// Spread the default curve is fitted to. See [`DEFAULT_MIN_DIST`].
pub const DEFAULT_SPREAD: f64 = 1.0;

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
    /// * `lr` - Initial learning rate
    /// * `n_epochs` - Number of optimisation epochs (typically 500)
    /// * `neg_sample_rate` - Number of negative samples per positive edge
    ///   (typically 5)
    /// * `beta1` - Optional beta1 parameter for Adam-based optimisations.
    /// * `beta2` - Optional beta2 parameter for Adam-based optimisations.
    /// * `eps` - Optional eps parameter for Adam-based optimisations.
    ///
    /// ### Return
    ///
    /// Self with calculated `a` and `b` parameter according to the
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
    /// Fits the UMAP curve: `f(x) = 1 / (1 + a + x^(2b))` such that
    /// `f(min_dist) ca. 1.0` and `f(spread) ca. 0.0`.
    ///
    /// ### Params
    ///
    /// * `min_dist` - Minimum distance parameter
    /// * `spread` - Effective scale of embedded points
    /// * `lr` - Learning rate for gradient descent (default: 0.1)
    /// * `n_iter` - Number of optimisation iterations (default: 100)
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
}

/// Parse the UMAP Optimiser to use
///
/// ### Params
///
/// * `s` - String defining the optimiser. Choice of `"adam"`, `"adam_parallel"`
///   or `"sgd"`.
///
/// ### Return
///
/// Option of Optimiser
pub fn parse_umap_optimiser(s: &str) -> Option<UmapOptimiser> {
    match s.to_lowercase().as_str() {
        "adam" => Some(UmapOptimiser::Adam),
        "sgd" => Some(UmapOptimiser::Sgd),
        "adam_parallel" => Some(UmapOptimiser::AdamParallel),
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
            clip_val: T::from_f64(4.0).unwrap(),
            eps: T::from_f64(0.001).unwrap(),
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
/// Takes the adjacency list rather than the CSR view so all three CPU
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

/// Optimise UMAP embedding using Stochastic Gradient Descent (SGD)
///
/// Implements the standard UMAP optimisation procedure using SGD with:
///
/// - Adaptive edge sampling based on edge weights (higher weights sampled more
///   frequently)
/// - Negative sampling for repulsive forces
/// - Linear learning rate decay schedule
/// - Per-vertex RNG state for reproducible negative sampling
///
/// ### Algorithm
///
/// For each epoch:
///
/// 1. Process edges whose `epoch_of_next_sample` has arrived
/// 2. Apply attractive force between connected vertices
/// 3. Perform negative sampling: randomly select vertices and apply repulsive
///    forces
/// 4. Update sampling schedules
///
/// ### Params
///
/// * `embd` - Initial embedding coordinates (modified in place), shape
///   `[n_samples][n_dim]`
/// * `graph` - Adjacency list where `graph[i]` contains (neighbour_idx, weight)
///   pairs
/// * `params` - Optimisation parameters (n_epochs, lr, a, b, gamma,
///   neg_sample_rate)
/// * `seed` - Random seed for negative sampling reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Notes
///
/// - Embedding is flattened internally for cache locality
/// - Edge weights are normalised to determine sampling frequency
/// - Higher edge weights result in more frequent sampling
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
    let rep_eps = T::from(0.001).unwrap();

    // fast paths for common b values
    let b_is_one = (consts.b - one).abs() < T::from(1e-10).unwrap();
    let b_is_half = (consts.b - half).abs() < T::from(1e-10).unwrap();

    let mut edges: Vec<(usize, usize, T)> = Vec::new();
    for (i, neighbours) in graph.iter().enumerate() {
        for &(j, w) in neighbours {
            edges.push((i, j, w));
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
    let epochs_per_neg_sample: Vec<T> = epochs_per_sample
        .iter()
        .map(|eps| *eps / neg_sample_rate_t)
        .collect();
    let mut epoch_of_next_neg_sample: Vec<T> = epochs_per_neg_sample.clone();

    let n_epochs_f = T::from(params.n_epochs).unwrap();
    let lr_schedule: Vec<T> = (0..params.n_epochs)
        .map(|e| params.lr * (one - T::from(e).unwrap() / n_epochs_f))
        .collect();

    let mut rng_states: Vec<SmallRng> = (0..n)
        .map(|i| SmallRng::seed_from_u64(seed + i as u64))
        .collect();

    for epoch in 0..params.n_epochs {
        let lr = lr_schedule[epoch];
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

            let base_i = i * n_dim;
            let base_j = j * n_dim;

            // Compute distance squared
            let mut dist_sq = zero;
            for d in 0..n_dim {
                let diff = embd_flat[base_i + d] - embd_flat[base_j + d];
                dist_sq += diff * diff;
            }

            // Attractive force - inlined
            if dist_sq >= dist_sq_threshold {
                // C++ trick: compute d^(2b) once, then divide by d^2 to get d^(2b-2)
                // This avoids computing powf twice
                let dist_sq_b = fast_pow(dist_sq, consts.b, b_is_one, b_is_half);
                let denom = one + consts.a * dist_sq_b;
                let grad_coeff = consts.two_a_b * dist_sq_b / (dist_sq * denom);

                // densMAP. This edge list is directed, so each undirected edge
                // is visited from both sides and both endpoints move on each
                // visit - exactly the reference's layout, hence no extra
                // factor here (unlike the CSR-based parallel variant).
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
                    let mut grad_d = (grad_coeff * delta)
                        .max(-consts.clip_val)
                        .min(consts.clip_val);

                    if dens_ctx.is_some() {
                        grad_d += (two_cor * -delta)
                            .max(-consts.clip_val)
                            .min(consts.clip_val);
                    }

                    embd_flat[base_i + d] += grad_d * lr;
                    embd_flat[base_j + d] -= grad_d * lr;
                }
            }

            epoch_of_next_sample[edge_idx] += epochs_per_sample[edge_idx];

            // Negative sampling
            let n_neg_samples = ((epoch_t - epoch_of_next_neg_sample[edge_idx])
                / epochs_per_neg_sample[edge_idx])
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

                // Repulsive force - inlined
                let dist_sq_safe = dist_sq + rep_eps;
                let dist_sq_b = fast_pow(dist_sq_safe, consts.b, b_is_one, b_is_half);
                let denom = dist_sq_safe * (one + consts.a * dist_sq_b);
                let grad_coeff = (consts.two_gamma_b / denom)
                    .max(-consts.clip_val)
                    .min(consts.clip_val);

                for d in 0..n_dim {
                    let delta = embd_flat[base_i + d] - embd_flat[base_k + d];
                    let grad_d = grad_coeff * delta;
                    embd_flat[base_i + d] += grad_d * lr;
                }
            }

            epoch_of_next_neg_sample[edge_idx] +=
                T::from(n_neg_samples).unwrap() * epochs_per_neg_sample[edge_idx];
        }

        if verbosity.normal_verbosity() && ((epoch + 1) % 50 == 0 || epoch + 1 == params.n_epochs) {
            println!(" Completed epoch {}/{}", epoch + 1, params.n_epochs);
        }
    }

    for (i, point) in embd.iter_mut().enumerate() {
        let base = i * n_dim;
        point.copy_from_slice(&embd_flat[base..base + n_dim]);
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
/// - Per-gradient-step timestep counter for correct bias correction
/// - Negative sampling for repulsive forces
///
/// ### Algorithm
///
/// For each epoch:
///
/// 1. Process edges whose `epoch_of_next_sample` has arrived
/// 2. Compute attractive gradient and apply Adam update to both endpoints
/// 3. Perform negative sampling and apply repulsive Adam updates
/// 4. Bias correction applied before each parameter update
///
/// ### Params
///
/// * `embd` - Initial embedding coordinates (modified in place), shape
///   `[n_samples][n_dim]`
/// * `graph` - Adjacency list where `graph[i]` contains (neighbour_idx, weight)
///   pairs
/// * `params` - Optimisation parameters including Adam hyperparameters (beta1,
///   beta2, eps)
/// * `seed` - Random seed for negative sampling
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Notes
///
/// - Uses per-gradient-step timestep counter (`global_timestep`) for bias
///   correction
/// - Pre-computes bias corrections for first 10,000 timesteps (lookup table)
/// - Each edge processed increments the timestep counter
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

    let mut edges: Vec<(usize, usize, T)> = Vec::new();
    for (i, neighbours) in graph.iter().enumerate() {
        for &(j, w) in neighbours {
            edges.push((i, j, w));
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
    let epochs_per_neg_sample: Vec<T> = epochs_per_sample
        .iter()
        .map(|eps| *eps / neg_sample_rate_t)
        .collect();
    let mut epoch_of_next_neg_sample: Vec<T> = epochs_per_neg_sample.clone();

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

            let n_neg_samples = ((epoch_t - epoch_of_next_neg_sample[edge_idx])
                / epochs_per_neg_sample[edge_idx])
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

            epoch_of_next_neg_sample[edge_idx] +=
                T::from(n_neg_samples).unwrap() * epochs_per_neg_sample[edge_idx];
        }

        // Update bias correction factors for next epoch (matching C++ epoch_end)
        beta1t *= params.beta1;
        beta2t *= params.beta2;

        if verbosity.normal_verbosity() && ((epoch + 1) % 50 == 0 || epoch + 1 == params.n_epochs) {
            println!(" Completed epoch {}/{}", epoch + 1, params.n_epochs);
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

    let mut edges: Vec<(usize, usize, T)> = Vec::new();
    let mut degree = vec![0; n];

    // take only i < j to get the unique undirected edges.
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

    let max_weight =
        edges
            .iter()
            .map(|(_, _, w)| *w)
            .fold(T::zero(), |acc, w| if w > acc { w } else { acc });

    let epochs_per_sample: Vec<T> = edges
        .iter()
        .map(|(_, _, w)| {
            let norm = *w / max_weight;
            if norm > T::zero() {
                T::one() / norm
            } else {
                T::from(1e8).unwrap()
            }
        })
        .collect();

    let mut epoch_of_next_sample: Vec<T> = epochs_per_sample.clone();

    let epochs_per_neg_sample: Vec<T> = epochs_per_sample
        .iter()
        .map(|eps| *eps / T::from(params.neg_sample_rate).unwrap())
        .collect();
    let mut epoch_of_next_neg_sample: Vec<T> = epochs_per_neg_sample.clone();

    let n_epochs_f = T::from(params.n_epochs).unwrap();
    let lr_schedule: Vec<T> = (0..params.n_epochs)
        .map(|e| params.lr * (T::one() - T::from(e).unwrap() / n_epochs_f))
        .collect();

    let mut m: Vec<T> = vec![T::zero(); n * n_dim];
    let mut v: Vec<T> = vec![T::zero(); n * n_dim];

    // flatten graph to CSR layout to kill pointer chasing
    let mut node_edge_offsets = vec![0; n + 1];
    for i in 0..n {
        node_edge_offsets[i + 1] = node_edge_offsets[i] + degree[i];
    }

    // csr_edges stores (edge_idx, is_smaller, other_node_idx)
    let mut csr_edges = vec![(0usize, false, 0usize); edges.len() * 2];
    let mut current_offset = node_edge_offsets.clone();

    for (edge_idx, &(i, j, _)) in edges.iter().enumerate() {
        csr_edges[current_offset[i]] = (edge_idx, true, j);
        current_offset[i] += 1;
        csr_edges[current_offset[j]] = (edge_idx, false, i);
        current_offset[j] += 1;
    }

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

                let start_idx = node_edge_offsets[node_i];
                let end_idx = node_edge_offsets[node_i + 1];
                let node_edges = &csr_edges[start_idx..end_idx];

                for &(edge_idx, is_smaller, other_node) in node_edges {
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

                    if dist_sq >= T::from(1e-8).unwrap() {
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
                                edges[edge_idx].2,
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

                    if is_smaller {
                        let n_neg_samples = ((epoch_t - epoch_of_next_neg_sample[edge_idx])
                            / epochs_per_neg_sample[edge_idx])
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
            .zip(epochs_per_sample.par_iter())
            .zip(epochs_per_neg_sample.par_iter())
            .for_each(|(((next_sample, next_neg), &per_sample), &per_neg)| {
                if *next_sample <= epoch_t {
                    *next_sample += per_sample;

                    let n_neg_samples = ((epoch_t - *next_neg) / per_neg)
                        .floor()
                        .to_usize()
                        .unwrap_or(0);

                    *next_neg += T::from(n_neg_samples).unwrap() * per_neg;
                }
            });

        if verbosity.normal_verbosity() && ((epoch + 1) % 50 == 0 || epoch + 1 == params.n_epochs) {
            println!(" Completed epoch {}/{}", epoch + 1, params.n_epochs);
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
}
