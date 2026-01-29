use core::f64;
use num_traits::{Float, FromPrimitive};
use rand::{
    rngs::SmallRng,
    {Rng, SeedableRng},
};
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::ops::{AddAssign, MulAssign, SubAssign};

//////////
// UMAP //
//////////

/////////////
// Globals //
/////////////

/// Default beta1 value for Adam optimisation
const BETA1: f64 = 0.5;
/// Default beta2 value for Adam optimisation
const BETA2: f64 = 0.9;
/// Default eps for Adam optimisation
const EPS: f64 = 1e-7;

//////////////////////////
// Structures and Enums //
//////////////////////////

/// UMAP optimisation parameters
///
/// ### Fields
///
/// * `a` - Curve parameter for repulsive force (typically ~1.5 for 2D)
/// * `b` - Curve parameter for repulsive force (typically ~0.9 for 2D)
/// * `lr` - Initial learning rate (typically 1.0)
/// * `gamma` - Parameter to control repulsion force
/// * `n_epochs` - Number of optimisation epochs (typically 500)
/// * `neg_sample_rate` - Number of negative samples per positive edge
///   (typically 5)
/// * `min_dist` - Minimum distance between points in embedding (typically 0.1)
/// * `beta1` - beta1 parameter for Adam optimiser
/// * `beta2` - beta2 parameter for Adam optimiser
/// * `eps` - eps for Adam optimiser
#[derive(Clone, Debug)]
pub struct UmapOptimParams<T> {
    pub a: T,
    pub b: T,
    pub lr: T,
    pub gamma: T,
    pub n_epochs: usize,
    pub neg_sample_rate: usize,
    pub min_dist: T,
    pub beta1: T,
    pub beta2: T,
    pub eps: T,
}

impl<T> UmapOptimParams<T>
where
    T: Float + FromPrimitive,
{
    /// Default parameters for 2D embedding
    ///
    /// ### Returns
    ///
    /// Self with sensible default parameters for the classical
    pub fn default_2d() -> Self {
        Self {
            a: T::from_f64(1.5).unwrap(),
            b: T::from_f64(0.9).unwrap(),
            lr: T::one(),
            gamma: T::one(),
            n_epochs: 500,
            neg_sample_rate: 5,
            min_dist: T::from_f64(0.1).unwrap(),
            beta1: T::from(BETA1).unwrap(),
            beta2: T::from(BETA2).unwrap(),
            eps: T::from(EPS).unwrap(),
        }
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
        let beta1 = beta1.unwrap_or(T::from(BETA1).unwrap());
        let beta2 = beta2.unwrap_or(T::from(BETA2).unwrap());
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

                grad_a = grad_a + err * (-x_2b / (denom * denom));

                let log_x = x.ln();
                grad_b = grad_b + err * (-two * a * x_2b * log_x / (denom * denom));
            }

            // Normalise gradients and use adaptive learning rate
            grad_a = grad_a / n_points_t;
            grad_b = grad_b / n_points_t;

            let lr_a = T::from_f64(1.0).unwrap();
            let lr_b = T::from_f64(1.0).unwrap();

            a = a - lr_a * grad_a;
            b = b - lr_b * grad_b;

            a = a
                .max(T::from_f64(0.001).unwrap())
                .min(T::from_f64(10.0).unwrap());
            b = b
                .max(T::from_f64(0.1).unwrap())
                .min(T::from_f64(2.0).unwrap());
        }

        (a, b)
    }
}

impl<T> Default for UmapOptimParams<T>
where
    T: Float + FromPrimitive,
{
    /// Returns sensible defaults for the optimiser (assuming 2D)
    fn default() -> Self {
        UmapOptimParams::default_2d()
    }
}

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

/// Precomputed constants to avoid repeated calculations
///
/// ### Fields
///
/// * `a` - The a parameter.
/// * `b` - The b parameter.
/// * `two_b` - b multiplied with 2.
/// * `four_b` - b multiplied with 4.
/// * `two_a_b` - The product of `2 * a * b`.
/// * `clip_val` - The clipping value, i.e., `4.0`.
/// * `eps` - The epsilon value
struct OptimConstants<T> {
    a: T,
    b: T,
    two_a_b: T,
    two_gamma_b: T,
    clip_val: T,
    eps: T,
}

impl<T: Float + FromPrimitive> OptimConstants<T> {
    /// Generate all of the constants
    ///
    /// ### Params
    ///
    /// * `a` - The a parameter
    /// * `b` - The b parameter
    /// * `gamma` - The repulsion parameter. Usually defaults to `1.0`.
    ///
    /// ###
    ///
    /// Returns
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

#[inline(always)]
fn fast_pow<T: Float>(x: T, b: T, b_is_one: bool, b_is_half: bool) -> T {
    if b_is_one {
        x
    } else if b_is_half {
        x.sqrt()
    } else {
        x.powf(b)
    }
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
///   [n_samples][n_dim]
/// * `graph` - Adjacency list where graph[i] contains (neighbour_idx, weight)
///   pairs
/// * `params` - Optimisation parameters (n_epochs, lr, a, b, gamma,
///   neg_sample_rate)
/// * `seed` - Random seed for negative sampling reproducibility
/// * `verbose` - Controls verbosity
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
    seed: u64,
    verbose: bool,
) where
    T: Float + FromPrimitive + AddAssign + SubAssign,
{
    let n = embd.len();
    if n == 0 {
        return;
    }
    let n_dim = embd[0].len();

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
        return;
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

                for d in 0..n_dim {
                    let delta = embd_flat[base_j + d] - embd_flat[base_i + d];
                    let grad_d = (grad_coeff * delta)
                        .max(-consts.clip_val)
                        .min(consts.clip_val);

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

        if verbose && ((epoch + 1) % 50 == 0 || epoch + 1 == params.n_epochs) {
            println!(" Completed epoch {}/{}", epoch + 1, params.n_epochs);
        }
    }

    for (i, point) in embd.iter_mut().enumerate() {
        let base = i * n_dim;
        point.copy_from_slice(&embd_flat[base..base + n_dim]);
    }
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
///   [n_samples][n_dim]
/// * `graph` - Adjacency list where graph[i] contains (neighbour_idx, weight)
///   pairs
/// * `params` - Optimisation parameters including Adam hyperparameters (beta1,
///   beta2, eps)
/// * `seed` - Random seed for negative sampling
/// * `verbose` - If true, prints progress every 100 epochs
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
    seed: u64,
    verbose: bool,
) where
    T: Float + FromPrimitive + Send + Sync + AddAssign + MulAssign,
{
    let n = embd.len();
    if n == 0 {
        return;
    }
    let n_dim = embd[0].len();

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
        return;
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

                for d in 0..n_dim {
                    let delta = embd_flat[base_j + d] - embd_flat[base_i + d];
                    let grad = grad_coeff * delta;

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

        if verbose && ((epoch + 1) % 50 == 0 || epoch + 1 == params.n_epochs) {
            println!(" Completed epoch {}/{}", epoch + 1, params.n_epochs);
        }
    }

    for (i, point) in embd.iter_mut().enumerate() {
        let base = i * n_dim;
        point.copy_from_slice(&embd_flat[base..base + n_dim]);
    }
}

/// Optimise UMAP embedding using Adam optimiser (parallel batch version)
///
/// Implements uwot's `BatchUpdate` with `NodeWorker` behaviour:
///
/// - Parallelises over nodes (not edges)
/// - Accumulates gradients per node per epoch
/// - Applies Adam updates with per-epoch bias correction (matches uwot)
/// - Single update per node per epoch
///
/// ### Params
///
/// * `embd` - Initial embedding, modified in place
/// * `graph` - Adjacency list representation
/// * `params` - Includes Adam hyperparameters
/// * `seed` - Random seed
/// * `verbose` - Progress reporting
///
/// ### Notes
///
/// - Uses `two_gamma_b` for repulsive gradients (matches uwot)
/// - Bias correction matches uwot's per-epoch approach
/// - Builds bidirectional node-to-edges mapping for efficient parallelisation
pub fn optimise_embedding_adam_parallel<T>(
    embd: &mut [Vec<T>],
    graph: &[Vec<(usize, T)>],
    params: &UmapOptimParams<T>,
    seed: u64,
    verbose: bool,
) where
    T: Float + FromPrimitive + Send + Sync + AddAssign + std::fmt::Display,
{
    let n = embd.len();
    let n_dim = embd[0].len();

    let mut embd_flat: Vec<T> = Vec::with_capacity(n * n_dim);
    for point in embd.iter() {
        embd_flat.extend_from_slice(point);
    }

    let consts = OptimConstants::new(params.a, params.b, params.gamma);

    let mut edges: Vec<(usize, usize, T)> = Vec::new();
    let mut seen: FxHashSet<(usize, usize)> = FxHashSet::default();

    for (i, neighbours) in graph.iter().enumerate() {
        for &(j, w) in neighbours {
            if i < j && !seen.contains(&(i, j)) {
                edges.push((i, j, w));
                seen.insert((i, j));
            }
        }
    }

    if edges.is_empty() {
        return;
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

    let mut node_edges: Vec<Vec<(usize, bool)>> = vec![Vec::new(); n];
    for (edge_idx, &(i, j, _)) in edges.iter().enumerate() {
        node_edges[i].push((edge_idx, true));
        node_edges[j].push((edge_idx, false));
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
    let mut edge_was_sampled: Vec<bool> = vec![false; edges.len()];

    for epoch in 0..params.n_epochs {
        let lr = lr_schedule[epoch];
        let epoch_t = T::from(epoch).unwrap();
        let (ad_scale, epsc) = bias_corrections[epoch];

        node_has_update.fill(false);
        edge_was_sampled.fill(false);

        // HIGHLY unsafe shit
        let gradients_ptr = node_gradients_all.as_mut_ptr() as usize;
        let has_update_ptr = node_has_update.as_mut_ptr() as usize;
        let edge_sampled_ptr = edge_was_sampled.as_mut_ptr() as usize;

        (0..n).into_par_iter().for_each(|node_i| {
            let mut rng = SmallRng::seed_from_u64(
                seed.wrapping_mul(6364136223846793005)
                    .wrapping_add(node_i as u64)
                    .wrapping_add((epoch as u64) << 32),
            );

            let base_i = node_i * n_dim;
            let node_grad = unsafe {
                std::slice::from_raw_parts_mut((gradients_ptr as *mut T).add(base_i), n_dim)
            };

            for g in node_grad.iter_mut() {
                *g = T::zero();
            }

            let mut has_updates = false;

            for &(edge_idx, is_smaller) in &node_edges[node_i] {
                if epoch_of_next_sample[edge_idx] > epoch_t {
                    continue;
                }

                has_updates = true;

                if is_smaller {
                    unsafe {
                        *((edge_sampled_ptr as *mut bool).add(edge_idx)) = true;
                    }
                }

                let (i, j, _) = edges[edge_idx];
                let other_node = if is_smaller { j } else { i };
                let base_other = other_node * n_dim;

                let mut dist_sq = T::zero();
                for d in 0..n_dim {
                    let diff = embd_flat[base_i + d] - embd_flat[base_other + d];
                    dist_sq += diff * diff;
                }

                if dist_sq >= T::from(1e-8).unwrap() {
                    let dist_sq_b = if consts.b == T::one() {
                        dist_sq
                    } else {
                        dist_sq.powf(consts.b)
                    };
                    let denom = T::one() + consts.a * dist_sq_b;
                    let grad_coeff = consts.two_a_b * dist_sq_b / (dist_sq * denom);

                    for d in 0..n_dim {
                        let delta = embd_flat[base_other + d] - embd_flat[base_i + d];
                        node_grad[d] += T::from(2.0).unwrap() * grad_coeff * delta;
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
                        let dist_sq_b = if consts.b == T::one() {
                            dist_sq_safe
                        } else {
                            dist_sq_safe.powf(consts.b)
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

            if has_updates {
                unsafe {
                    *((has_update_ptr as *mut bool).add(node_i)) = true;
                }
            }
        });

        for node_i in 0..n {
            if !node_has_update[node_i] {
                continue;
            }

            let base_i = node_i * n_dim;
            for d in 0..n_dim {
                let idx = base_i + d;
                let g = node_gradients_all[idx];

                let m_old = m[idx];
                m[idx] += one_minus_beta1 * (g - m_old);

                let v_old = v[idx];
                v[idx] += one_minus_beta2 * (g * g - v_old);

                embd_flat[idx] += lr * ad_scale * m[idx] / (v[idx].sqrt() + epsc);
            }
        }

        for (edge_idx, &sampled) in edge_was_sampled.iter().enumerate() {
            if !sampled {
                continue;
            }

            epoch_of_next_sample[edge_idx] += epochs_per_sample[edge_idx];

            let n_neg_samples = ((epoch_t - epoch_of_next_neg_sample[edge_idx])
                / epochs_per_neg_sample[edge_idx])
                .floor()
                .to_usize()
                .unwrap_or(0);

            epoch_of_next_neg_sample[edge_idx] +=
                T::from(n_neg_samples).unwrap() * epochs_per_neg_sample[edge_idx];
        }

        if verbose && ((epoch + 1) % 50 == 0 || epoch + 1 == params.n_epochs) {
            println!(" Completed epoch {}/{}", epoch + 1, params.n_epochs);
        }
    }

    for (i, point) in embd.iter_mut().enumerate() {
        let base = i * n_dim;
        point.copy_from_slice(&embd_flat[base..base + n_dim]);
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_umap_optimiser {
    use super::*;
    use approx::assert_relative_eq;

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

        assert_relative_eq!(params.a, 1.5, epsilon = 1e-6);
        assert_relative_eq!(params.b, 0.9, epsilon = 1e-6);
        assert_eq!(params.lr, 1.0);
        assert_eq!(params.gamma, 1.0);
        assert_eq!(params.n_epochs, 500);
        assert_eq!(params.neg_sample_rate, 5);
        assert_relative_eq!(params.min_dist, 0.1, epsilon = 1e-6);
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
        optimise_embedding_adam(&mut embd, &graph, &params, 42, false);

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
        optimise_embedding_adam_parallel(&mut embd, &graph, &params, 42, false);

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
        optimise_embedding_adam(&mut embd, &graph, &params, 42, false);

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

        optimise_embedding_adam(&mut embd1, &graph, &params, 42, false);
        optimise_embedding_adam(&mut embd2, &graph, &params, 42, false);

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

        optimise_embedding_adam_parallel(&mut embd1, &graph, &params, 42, false);
        optimise_embedding_adam_parallel(&mut embd2, &graph, &params, 42, false);

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

        optimise_embedding_adam(&mut embd, &graph, &params, 42, false);

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
        optimise_embedding_sgd(&mut embd_sgd, &graph, &params, 42, false);

        let mut embd_adam = initial_embd.clone();
        optimise_embedding_adam(&mut embd_adam, &graph, &params, 42, false);

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
        optimise_embedding_sgd(&mut embd_sgd, &graph, &params, 42, false);

        let mut embd_adam = initial_embd.clone();
        optimise_embedding_adam(&mut embd_adam, &graph, &params, 42, false);

        let mut embd_adam_par = initial_embd.clone();
        optimise_embedding_adam_parallel(&mut embd_adam_par, &graph, &params, 42, false);

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

        optimise_embedding_sgd(&mut embd1, &graph, &params, 42, false);
        optimise_embedding_sgd(&mut embd2, &graph, &params, 42, false);

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

        optimise_embedding_adam(&mut embd, &graph, &params, 42, false);

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

        optimise_embedding_adam_parallel(&mut embd, &graph, &params, 42, false);

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

        optimise_embedding_sgd(&mut embd, &graph, &params, 42, false);

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
