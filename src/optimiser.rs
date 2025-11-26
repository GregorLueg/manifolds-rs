use core::f64;
use num_traits::{Float, FromPrimitive};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::ops::AddAssign;

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
/// * `a` - Curve parameter for repulsive force (typically ~1.93 for 2D)
/// * `b` - Curve parameter for repulsive force (typically ~0.79 for 2D)
/// * `lr` - Initial learning rate (typically 1.0)
/// * `n_epochs` - Number of optimisation epochs (typically 500)
/// * `neg_sample_rate` - Number of negative samples per positive edge
///   (typically 5)
/// * `min_dist` - Minimum distance between points in embedding (typically 0.1)
/// * `beta1` -
/// * `beta2` -
/// * `eps` -
#[derive(Clone, Debug)]
pub struct OptimParams<T> {
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

impl<T> OptimParams<T>
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
            a: T::from_f64(1.929).unwrap(),
            b: T::from_f64(0.7915).unwrap(),
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
        lr: T,
        gamma: T,
        n_epochs: usize,
        neg_sample_rate: usize,
        beta1: Option<T>,
        beta2: Option<T>,
        eps: Option<T>,
    ) -> Self {
        // take the Adam-related values
        let beta1 = beta1.unwrap_or(T::from(BETA1).unwrap());
        let beta2 = beta2.unwrap_or(T::from(BETA2).unwrap());
        let eps = eps.unwrap_or(T::from(EPS).unwrap());

        let (a, b) = Self::fit_params(min_dist, spread, None, None);
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
    fn fit_params(min_dist: T, spread: T, lr: Option<T>, n_iter: Option<usize>) -> (T, T) {
        let lr = lr.unwrap_or_else(|| T::from_f64(0.1).unwrap());
        let n_iter = n_iter.unwrap_or(100);

        // high membership at min_dist
        let x0 = min_dist;
        let y0 = T::from_f64(0.95).unwrap();

        // low membership at 3 * spread
        let x1 = spread * T::from(3.0).unwrap();
        let y1 = T::from_f64(0.01).unwrap();

        // non-linear optimisation to find a and b using SGD
        let mut a = T::one();
        let mut b = T::one();

        for _ in 0..n_iter {
            // predictions
            let pred0 = T::one() / (T::one() + a * x0.powf(T::from_f64(2.0).unwrap() * b));
            let pred1 = T::one() / (T::one() + a * x1.powf(T::from_f64(2.0).unwrap() * b));

            let err0 = pred0 - y0;
            let err1 = pred1 - y1;

            // approximate the gradient - horrible formula...
            let grad_a = err0 * x0.powf(T::from_f64(2.0).unwrap() * b)
                / (T::one() + a * x0.powf(T::from_f64(2.0).unwrap() * b)).powi(2)
                + err1 * x1.powf(T::from_f64(2.0).unwrap() * b)
                    / (T::one() + a * x1.powf(T::from_f64(2.0).unwrap() * b)).powi(2);

            let two = T::from_f64(2.0).unwrap();
            let log_x0 = x0.ln();
            let log_x1 = x1.ln();
            let grad_b = err0 * (two * a * x0.powf(two * b) * log_x0)
                / (T::one() + a * x0.powf(two * b)).powi(2)
                + err1 * (two * a * x1.powf(two * b) * log_x1)
                    / (T::one() + a * x1.powf(two * b)).powi(2);

            // update and clamp
            a = a - lr * grad_a;
            b = b - lr * grad_b;

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

impl<T> Default for OptimParams<T>
where
    T: Float + FromPrimitive,
{
    /// Returns sensible defaults for the optimiser (assuming 2D)
    fn default() -> Self {
        OptimParams::default_2d()
    }
}

#[derive(Default)]
pub enum Optimiser {
    /// Adam
    #[default]
    Adam,
    /// Parallel version of Adam
    AdamParallel,
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

/// Parse the Optimiser to use
///
/// ### Params
///
/// * `s` - String defining the optimiser. Choice of `"adam"`, `"adam_parallel"`
///   or `"sgd"`.
///
/// ### Return
///
/// Option of Optimiser
pub fn parse_optimiser(s: &str) -> Option<Optimiser> {
    match s.to_lowercase().as_str() {
        "adam" => Some(Optimiser::Adam),
        "sgd" => Some(Optimiser::Sgd),
        "adam_parallel" => Some(Optimiser::AdamParallel),
        _ => None,
    }
}

/// Compute squared Eucliden distance between two points
///
/// Assumes a flat structure to better cache locality
///
/// ### Params
///
/// * `embd` - The flat embedding structure
/// * `i` - Position of data point i in the embedding
/// * `j` - Position of data point j in the embedding
/// * `n_dim` - Number of dimensions in that embedding
///
/// ### Returns
///
/// Squared distance between two points
#[inline(always)]
fn squared_dist_flat<T: Float>(embd: &[T], i: usize, j: usize, n_dim: usize) -> T {
    let mut sum = T::zero();
    let base_i = i * n_dim;
    let base_j = j * n_dim;
    for d in 0..n_dim {
        let diff = embd[base_i + d] - embd[base_j + d];
        sum = sum + diff * diff;
    }
    sum
}

/// Apply attractive force gradient for a connected edge
///
/// Pulls points `i` and `j` together based on their edge weight.
///
/// The gradient is derived from the UMAP curve: `phi(d) = (1 + a d^{2b})^{-1}`
///
/// ### Params
///
/// * `embd` - Current embedding coordinates (modified in place)
/// * `i` - Source vertex index
/// * `j` - Target vertex index
/// * `n_dim` - Number of dimensions
/// * `lr` - Step size for gradient update
///
/// ### Notes
///
/// * Updates both points symmetrically: `i` moves towards `j`, and `j` moves
///   towards `i`.
/// * Skips update if points are essentially at the same location (dist_square
///   < 1e-8) to avoid numerical instability.
#[inline]
fn apply_attractive_force_flat<T: Float>(
    embd: &mut [T],
    i: usize,
    j: usize,
    n_dim: usize,
    consts: &OptimConstants<T>,
    lr: T,
) {
    let dist_sq = squared_dist_flat(embd, i, j, n_dim);

    if dist_sq < T::from(1e-8).unwrap() {
        return;
    }

    let grad_coeff = if dist_sq > T::zero() {
        let dist_sq_b = dist_sq.powf(consts.b);
        let numerator =
            -consts.a * consts.b * T::from(2.0).unwrap() * dist_sq.powf(consts.b - T::one());
        let denominator = consts.a * dist_sq_b + T::one();
        numerator / denominator
    } else {
        T::zero()
    };

    let base_i = i * n_dim;
    let base_j = j * n_dim;

    for d in 0..n_dim {
        let grad_d = (grad_coeff * (embd[base_i + d] - embd[base_j + d]))
            .max(-consts.clip_val)
            .min(consts.clip_val);

        embd[base_i + d] = embd[base_i + d] + grad_d * lr;
        embd[base_j + d] = embd[base_j + d] - grad_d * lr;
    }
}

/// Apply repulsive force gradient via negative sampling
///
/// Pushes point `i` away from randomly sampled point `j` (which is assumed to
/// be unconnected).
///
/// ### Params
///
/// * `embd` - Current embedding coordinates (modified in place)
/// * `i` - Source vertex index
/// * `j` - Randomly sampled vertex index (negative sample)
/// * `a` - Curve parameter controlling spread
/// * `b` - Curve parameter controlling tail behaviour
/// * `lr` - Step size for gradient update
///
/// ### Notes
///
/// * Only updates point `i` (not `j`), as this is an asymmetric negative
///   sampling step.
/// * Adds a small epsilon (0.001) to the distance to prevent division by zero.
/// * **Clips gradients** to the range [-4.0, 4.0] to prevent the "exploding
///   gradient" problem when points are very close to one another.
#[inline]
fn apply_repulsive_force_flat<T: Float>(
    embd: &mut [T],
    i: usize,
    k: usize,
    n_dim: usize,
    consts: &OptimConstants<T>,
    lr: T,
) {
    let dist_sq = squared_dist_flat(embd, i, k, n_dim);

    let grad_coeff = if dist_sq > T::zero() {
        let dist_sq_b = dist_sq.powf(consts.b);
        let denominator = (T::from(0.001).unwrap() + dist_sq) * (consts.a * dist_sq_b + T::one());
        consts.two_gamma_b / denominator
    } else {
        T::zero()
    };

    let base_i = i * n_dim;
    let base_k = k * n_dim;
    for d in 0..n_dim {
        let grad_d = if grad_coeff > T::zero() {
            (grad_coeff * (embd[base_i + d] - embd[base_k + d]))
                .max(T::from(-consts.clip_val).unwrap())
                .min(T::from(consts.clip_val).unwrap())
        } else {
            T::zero()
        };

        embd[base_i + d] = embd[base_i + d] + grad_d * lr;
    }
}

////////////////////
// Main functions //
////////////////////

/// Optimise UMAP embedding using Stochastic Gradient Descent (SGD)
///
/// Implements the standard UMAP optimization procedure using SGD with:
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
///   [n_vertices][n_dim]
/// * `graph` - Adjacency list where graph[i] contains (neighbour_idx, weight)
///   pairs
/// * `params` - Optimisation parameters (n_epochs, lr, a, b, gamma,
///   neg_sample_rate)
/// * `seed` - Random seed for negative sampling reproducibility
/// * `verbose` - Controls verbosity
///
/// # Notes
///
/// - Embedding is flattened internally for cache locality
/// - Edge weights are normalised to determine sampling frequency
/// - Higher edge weights result in more frequent sampling
pub fn optimise_embedding_sgd<T>(
    embd: &mut [Vec<T>],
    graph: &[Vec<(usize, T)>],
    params: &OptimParams<T>,
    seed: u64,
    verbose: bool,
) where
    T: Float + FromPrimitive + AddAssign,
{
    let n = embd.len();
    let n_dim = embd[0].len();

    // flatten for cache locality
    let mut embd_flat: Vec<T> = Vec::with_capacity(n * n_dim);
    for point in embd.iter() {
        embd_flat.extend_from_slice(point);
    }

    // build edge list
    let mut edges: Vec<(usize, usize, T)> = Vec::new();
    for (i, neighbours) in graph.iter().enumerate() {
        for &(j, w) in neighbours {
            edges.push((i, j, w));
        }
    }

    if edges.is_empty() {
        return;
    }

    // normalise weights for sampling
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

    // linear LR decay
    let lr_schedule: Vec<T> = (0..params.n_epochs)
        .map(|e| params.lr * (T::one() - T::from(e).unwrap() / T::from(params.n_epochs).unwrap()))
        .collect();

    // per-vertex RNG
    let mut rng_states: Vec<StdRng> = (0..n)
        .map(|i| StdRng::seed_from_u64(seed + i as u64))
        .collect();

    let consts = OptimConstants::new(params.a, params.b, params.gamma);

    // main loop
    for epoch in 0..params.n_epochs {
        let lr = lr_schedule[epoch];
        let epoch_t = T::from(epoch).unwrap();

        for (edge_idx, &(i, j, _weight)) in edges.iter().enumerate() {
            if epoch_of_next_sample[edge_idx] > epoch_t {
                continue;
            }

            // apply attractive force
            apply_attractive_force_flat(&mut embd_flat, i, j, n_dim, &consts, lr);

            epoch_of_next_sample[edge_idx] += epochs_per_sample[edge_idx];

            // adaptive negative sampling
            let n_neg_samples = ((epoch_t - epoch_of_next_neg_sample[edge_idx])
                / epochs_per_neg_sample[edge_idx])
                .floor()
                .to_usize()
                .unwrap_or(0);

            for _ in 0..n_neg_samples {
                let k = rng_states[i].random_range(0..n);
                if k == i || k == j {
                    continue;
                }
                apply_repulsive_force_flat(&mut embd_flat, i, k, n_dim, &consts, lr);
            }

            epoch_of_next_neg_sample[edge_idx] +=
                T::from(n_neg_samples).unwrap() * epochs_per_neg_sample[edge_idx];
        }

        if verbose && ((epoch + 1) % 100 == 0 || epoch + 1 == params.n_epochs) {
            println!(" Completed epoch {}/{}", epoch + 1, params.n_epochs);
        }
    }

    // unflatten
    for (i, point) in embd.iter_mut().enumerate() {
        let base = i * n_dim;
        point.copy_from_slice(&embd_flat[base..base + n_dim]);
    }
}

/// Optimise UMAP embedding using Adam optimiser (sequential version)
///
/// Implements UMAP optimization using the Adam adaptive learning rate
/// algorithm:
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
///   [n_vertices][n_dim]
/// * `graph` - Adjacency list where graph[i] contains (neighbour_idx, weight)
///   pairs
/// * `params` - Optimisation parameters including Adam hyperparameters (beta1,
///   beta2, eps)
/// * `seed` - Random seed for negative sampling
/// * `verbose` - If true, prints progress every 100 epochs
///
/// # Implementation Notes
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
    params: &OptimParams<T>,
    seed: u64,
    verbose: bool,
) where
    T: Float + FromPrimitive + Send + Sync + AddAssign,
{
    let n = embd.len();
    let n_dim = embd[0].len();

    let mut embd_flat: Vec<T> = Vec::with_capacity(n * n_dim);
    for point in embd.iter() {
        embd_flat.extend_from_slice(point);
    }

    let consts = OptimConstants::new(params.a, params.b, params.gamma);

    let mut edges: Vec<(usize, usize, T)> = Vec::new();
    for (i, neighbours) in graph.iter().enumerate() {
        for &(j, w) in neighbours {
            edges.push((i, j, w));
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

    let mut rng_states: Vec<StdRng> = (0..n)
        .map(|i| StdRng::seed_from_u64(seed + i as u64))
        .collect();

    // pre-compute bias corrections
    let max_lookup = 10000;
    let mut bias_corr_m_lookup: Vec<T> = Vec::with_capacity(max_lookup);
    let mut bias_corr_v_lookup: Vec<T> = Vec::with_capacity(max_lookup);

    for t in 1..=max_lookup {
        let t_f = T::from(t).unwrap();
        bias_corr_m_lookup.push(T::one() / (T::one() - params.beta1.powf(t_f)));
        bias_corr_v_lookup.push(T::one() / (T::one() - params.beta2.powf(t_f)));
    }

    let mut global_timestep = 0;
    let one_minus_beta1 = T::one() - params.beta1;
    let one_minus_beta2 = T::one() - params.beta2;

    for epoch in 0..params.n_epochs {
        let lr = lr_schedule[epoch];
        let epoch_t = T::from(epoch).unwrap();

        for (edge_idx, &(i, j, _weight)) in edges.iter().enumerate() {
            if epoch_of_next_sample[edge_idx] > epoch_t {
                continue;
            }

            let base_i = i * n_dim;
            let base_j = j * n_dim;

            let mut dist_sq = T::zero();
            for d in 0..n_dim {
                let diff = embd_flat[base_i + d] - embd_flat[base_j + d];
                dist_sq += diff * diff;
            }

            if dist_sq >= T::from(1e-8).unwrap() {
                global_timestep += 1;

                let (bias_corr_m, bias_corr_v) = if global_timestep < max_lookup {
                    (
                        bias_corr_m_lookup[global_timestep - 1],
                        bias_corr_v_lookup[global_timestep - 1],
                    )
                } else {
                    (T::one(), T::one())
                };

                // attractive gradient: -2ab * d^(2b) / (d^2 * (1 + a*d^(2b)))
                let dist_sq_b = dist_sq.powf(consts.b);
                let denom = T::one() + consts.a * dist_sq_b;
                let grad_coeff = consts.two_a_b * dist_sq_b / (dist_sq * denom);

                for d in 0..n_dim {
                    let delta = embd_flat[base_j + d] - embd_flat[base_i + d];
                    let grad = grad_coeff * delta;

                    // Update i
                    let idx_i = base_i + d;
                    m[idx_i] = params.beta1 * m[idx_i] + one_minus_beta1 * grad;
                    v[idx_i] = params.beta2 * v[idx_i] + one_minus_beta2 * grad * grad;
                    embd_flat[idx_i] += lr * (m[idx_i] * bias_corr_m)
                        / ((v[idx_i] * bias_corr_v).sqrt() + params.eps);

                    // Update j
                    let idx_j = base_j + d;
                    m[idx_j] = params.beta1 * m[idx_j] - one_minus_beta1 * grad;
                    v[idx_j] = params.beta2 * v[idx_j] + one_minus_beta2 * grad * grad;
                    embd_flat[idx_j] += lr * (m[idx_j] * bias_corr_m)
                        / ((v[idx_j] * bias_corr_v).sqrt() + params.eps);
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

                global_timestep += 1;
                let (bias_corr_m, bias_corr_v) = if global_timestep < max_lookup {
                    (
                        bias_corr_m_lookup[global_timestep - 1],
                        bias_corr_v_lookup[global_timestep - 1],
                    )
                } else {
                    (T::one(), T::one())
                };

                let base_k = k * n_dim;

                let mut dist_sq = T::zero();
                for d in 0..n_dim {
                    let diff = embd_flat[base_i + d] - embd_flat[base_k + d];
                    dist_sq += diff * diff;
                }

                // Repulsive gradient: 2 * gamma * b / ((0.001 + d^2) * (1 + a*d^(2b)))
                let dist_sq_safe = dist_sq + consts.eps;
                let dist_sq_b = dist_sq_safe.powf(consts.b);
                let denom = dist_sq_safe * (T::one() + consts.a * dist_sq_b);
                let grad_coeff = (consts.two_gamma_b / denom)
                    .max(-consts.clip_val)
                    .min(consts.clip_val);

                for d in 0..n_dim {
                    let delta = embd_flat[base_i + d] - embd_flat[base_k + d];
                    let grad = grad_coeff * delta;

                    let idx = base_i + d;
                    m[idx] = params.beta1 * m[idx] + one_minus_beta1 * grad;
                    v[idx] = params.beta2 * v[idx] + one_minus_beta2 * grad * grad;
                    embd_flat[idx] +=
                        lr * (m[idx] * bias_corr_m) / ((v[idx] * bias_corr_v).sqrt() + params.eps);
                }
            }

            epoch_of_next_neg_sample[edge_idx] +=
                T::from(n_neg_samples).unwrap() * epochs_per_neg_sample[edge_idx];
        }

        if verbose && ((epoch + 1) % 100 == 0 || epoch + 1 == params.n_epochs) {
            println!(" Completed epoch {}/{}", epoch + 1, params.n_epochs);
        }
    }

    for (i, point) in embd.iter_mut().enumerate() {
        let base = i * n_dim;
        point.copy_from_slice(&embd_flat[base..base + n_dim]);
    }
}

/// Optimise UMAP embedding using Adam optimizer (parallel batch version)
///
/// Implements uwot's `BatchUpdate` with `NodeWorker` behavior:
/// - Parallelizes over nodes (not edges)
/// - Accumulates gradients per node per epoch
/// - Applies Adam updates with per-epoch bias correction (matches uwot)
/// - Single update per node per epoch
///
/// # Key Differences from Sequential Adam
///
/// - **Bias correction**: Per-epoch (not per-gradient-step)
///   Matches uwot's Adam::epoch_end() behavior where beta1^t and beta2^t
///   are updated once per epoch and applied to all updates
/// - **Update frequency**: One Adam step per node per epoch
/// - **Parallelization**: Over nodes instead of edges
///
/// # Parameters
///
/// * `embd` - Initial embedding, modified in place
/// * `graph` - Adjacency list representation
/// * `params` - Includes Adam hyperparameters
/// * `seed` - Random seed
/// * `verbose` - Progress reporting
///
/// # Implementation Notes
///
/// - Uses `two_gamma_b` for repulsive gradients (matches uwot)
/// - Bias correction matches uwot's per-epoch approach
/// - Builds bidirectional node-to-edges mapping for efficient parallelization
pub fn optimise_embedding_adam_parallel<T>(
    embd: &mut [Vec<T>],
    graph: &[Vec<(usize, T)>],
    params: &OptimParams<T>,
    seed: u64,
    verbose: bool,
) where
    T: Float + FromPrimitive + Send + Sync + AddAssign,
{
    let n = embd.len();
    let n_dim = embd[0].len();

    let mut embd_flat: Vec<T> = Vec::with_capacity(n * n_dim);
    for point in embd.iter() {
        embd_flat.extend_from_slice(point);
    }

    let consts = OptimConstants::new(params.a, params.b, params.gamma);

    let mut edges: Vec<(usize, usize, T)> = Vec::new();
    for (i, neighbours) in graph.iter().enumerate() {
        for &(j, w) in neighbours {
            edges.push((i, j, w));
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

    // build bidirectional node-to-edges mapping
    let mut node_edges: Vec<Vec<(usize, bool)>> = vec![Vec::new(); n];
    for (edge_idx, &(i, j, _)) in edges.iter().enumerate() {
        node_edges[i].push((edge_idx, true)); // i is head
        node_edges[j].push((edge_idx, false)); // j is tail
    }

    // pre-compute per-epoch bias correction (matching uwot's Adam::epoch_end)
    // beta1t and beta2t track beta1^epoch and beta2^epoch
    let bias_corrections: Vec<(T, T)> = (0..params.n_epochs)
        .map(|epoch| {
            let t = T::from(epoch + 1).unwrap();
            let beta1t = params.beta1.powf(t);
            let beta2t = params.beta2.powf(t);
            let sqrt_b2t1 = (T::one() - beta2t).sqrt();

            // ad_scale and epsc as in uwot's Adam::epoch_end
            let ad_scale = sqrt_b2t1 / (T::one() - beta1t);
            let epsc = sqrt_b2t1 * params.eps;

            (ad_scale, epsc)
        })
        .collect();

    let one_minus_beta1 = T::one() - params.beta1;
    let one_minus_beta2 = T::one() - params.beta2;

    for epoch in 0..params.n_epochs {
        let lr = lr_schedule[epoch];
        let epoch_t = T::from(epoch).unwrap();
        let (ad_scale, epsc) = bias_corrections[epoch];

        // parallel gradient accumulation
        let updates: Vec<(usize, Vec<T>)> = (0..n)
            .into_par_iter()
            .filter_map(|node_i| {
                let mut rng = StdRng::seed_from_u64(
                    seed.wrapping_mul(6364136223846793005)
                        .wrapping_add(node_i as u64)
                        .wrapping_add((epoch as u64) << 32),
                );

                let base_i = node_i * n_dim;
                let mut node_gradients = vec![T::zero(); n_dim];
                let mut has_updates = false;

                for &(edge_idx, is_head) in &node_edges[node_i] {
                    if epoch_of_next_sample[edge_idx] > epoch_t {
                        continue;
                    }

                    has_updates = true;
                    let (i, j, _) = edges[edge_idx];

                    let other_node = if is_head { j } else { i };
                    let base_other = other_node * n_dim;

                    let mut dist_sq = T::zero();
                    for d in 0..n_dim {
                        let diff = embd_flat[base_i + d] - embd_flat[base_other + d];
                        dist_sq += diff * diff;
                    }

                    if dist_sq >= T::from(1e-8).unwrap() {
                        let dist_sq_b = dist_sq.powf(consts.b);
                        let denom = T::one() + consts.a * dist_sq_b;
                        let grad_coeff = consts.two_a_b * dist_sq_b / (dist_sq * denom);

                        for d in 0..n_dim {
                            let delta = embd_flat[base_other + d] - embd_flat[base_i + d];
                            node_gradients[d] += grad_coeff * delta;
                        }
                    }

                    // negative sampling only for head nodes
                    if is_head {
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

                            // repulsive: 2*gamma*b / ((0.001 + d^2) * (1 + a*d^(2b)))
                            let dist_sq_safe = dist_sq + consts.eps;
                            let dist_sq_b = dist_sq_safe.powf(consts.b);
                            let denom = dist_sq_safe * (T::one() + consts.a * dist_sq_b);
                            let grad_coeff = (consts.two_gamma_b / denom)
                                .max(-consts.clip_val)
                                .min(consts.clip_val);

                            for d in 0..n_dim {
                                let delta = embd_flat[base_i + d] - embd_flat[base_k + d];
                                node_gradients[d] += grad_coeff * delta;
                            }
                        }
                    }
                }

                if has_updates {
                    Some((node_i, node_gradients))
                } else {
                    None
                }
            })
            .collect();

        // sequential update application
        for (node_i, node_gradients) in updates {
            let base_i = node_i * n_dim;

            for d in 0..n_dim {
                let idx = base_i + d;
                let g = node_gradients[d];

                // update moments (using in-place trick from uwot)
                let m_old = m[idx];
                m[idx] += one_minus_beta1 * (g - m_old);

                let v_old = v[idx];
                v[idx] += one_minus_beta2 * (g * g - v_old);

                // apply update with per-epoch bias correction
                embd_flat[idx] += lr * ad_scale * m[idx] / (v[idx].sqrt() + epsc);
            }
        }

        // update sampling schedules
        for (edge_idx, &_) in edges.iter().enumerate() {
            if epoch_of_next_sample[edge_idx] <= epoch_t {
                epoch_of_next_sample[edge_idx] += epochs_per_sample[edge_idx];

                let n_neg_samples = ((epoch_t - epoch_of_next_neg_sample[edge_idx])
                    / epochs_per_neg_sample[edge_idx])
                    .floor()
                    .to_usize()
                    .unwrap_or(0);

                epoch_of_next_neg_sample[edge_idx] +=
                    T::from(n_neg_samples).unwrap() * epochs_per_neg_sample[edge_idx];
            }
        }

        if verbose && ((epoch + 1) % 100 == 0 || epoch + 1 == params.n_epochs) {
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
mod test_optimiser {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_optim_params_default_2d() {
        let params = OptimParams::<f64>::default_2d();

        assert_relative_eq!(params.a, 1.929, epsilon = 1e-6);
        assert_relative_eq!(params.b, 0.7915, epsilon = 1e-6);
        assert_eq!(params.lr, 1.0);
        assert_eq!(params.gamma, 1.0);
        assert_eq!(params.n_epochs, 500);
        assert_eq!(params.neg_sample_rate, 5);
        assert_relative_eq!(params.min_dist, 0.1, epsilon = 1e-6);
    }

    #[test]
    fn test_optim_params_from_min_dist_spread() {
        let params =
            OptimParams::<f64>::from_min_dist_spread(0.1, 1.0, 1.0, 1.0, 500, 5, None, None, None);

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
        let (a, b) = OptimParams::<f64>::fit_params(0.1, 1.0, None, None);

        assert!((0.001..=10.0).contains(&a));
        assert!((0.1..=2.0).contains(&b));
    }

    #[test]
    fn test_fit_params_curve_properties() {
        let min_dist = 0.1;
        let spread = 1.0;
        let (a, b) = OptimParams::<f64>::fit_params(min_dist, spread, None, None);

        let pred_min = 1.0 / (1.0 + a * min_dist.powf(2.0 * b));
        assert!(
            pred_min > 0.7,
            "f(min_dist) = {:.3} should be > 0.7",
            pred_min
        );

        let pred_spread = 1.0 / (1.0 + a * (3.0 * spread).powf(2.0 * b));
        assert!(
            pred_spread < 0.3,
            "f(3*spread) = {:.3} should be < 0.3",
            pred_spread
        );

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

        let params = OptimParams::default_2d();
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

        let params = OptimParams::default_2d();
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

        let params = OptimParams::default_2d();
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

        let params = OptimParams {
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

        let params = OptimParams {
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

        let params = OptimParams {
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

        let params = OptimParams {
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

        let params = OptimParams {
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

        let params = OptimParams {
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

        let params = OptimParams {
            n_epochs: 200,
            ..OptimParams::default_2d()
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

        let params = OptimParams {
            n_epochs: 200,
            ..OptimParams::default_2d()
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

        let params = OptimParams {
            n_epochs: 200,
            ..OptimParams::default_2d()
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
