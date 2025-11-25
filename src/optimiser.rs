use core::f64;
use num_traits::{Float, FromPrimitive};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/////////////
// Globals //
/////////////

/// Default beta1 value for Adam optimisation
const BETA1: f64 = 0.9;
/// Default beta2 value for Adam optimisation
const BETA2: f64 = 0.99;
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
    /// * `beta1` -
    /// * `beta2` -
    /// * `eps` -
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
    two_b: T,
    four_b: T,
    two_a_b: T,
    gamma: T,
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
        let four = T::from_f64(4.0).unwrap();
        Self {
            a,
            b,
            two_b: two * b,
            four_b: four * b,
            two_a_b: two * a * b,
            gamma,
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
/// * `s` - String defining the optimiser. Choice of `"adam"` or `"sgd"`.
///
/// ### Return
///
/// Option of Optimiser
pub fn parse_optimiser(s: &str) -> Option<Optimiser> {
    match s.to_lowercase().as_str() {
        "adam" => Some(Optimiser::Adam),
        "sgd" => Some(Optimiser::Sgd),
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

/// Optimise embedding using SGD
///
/// ### Params
///
/// * `embd` - Initial embedding coordinates (modified in place)
/// * `graph` - Adjacency list representation of the high-dimensional graph
/// * `gamma` - Gamma parameter. Regulates the repulsion between points.
/// * `params` - Optimisation parameters
/// * `seed` - Random seed for negative sampling
/// * `verbose` - Controls verbosity of the function
pub fn optimise_embedding_sgd<T>(
    embd: &mut [Vec<T>],
    graph: &[Vec<(usize, T)>],
    params: &OptimParams<T>,
    seed: u64,
    verbose: bool,
) where
    T: Float + FromPrimitive,
{
    let n = embd.len();
    let n_dim = embd[0].len();

    // flatten embedding for cache locality
    let mut embd_flat: Vec<T> = Vec::with_capacity(n * n_dim);
    for point in embd.iter() {
        embd_flat.extend_from_slice(point);
    }

    // build edge list with weights
    let mut edges: Vec<(usize, usize, T)> = Vec::new();
    for (i, neighbours) in graph.iter().enumerate() {
        for &(j, w) in neighbours {
            edges.push((i, j, w));
        }
    }

    if edges.is_empty() {
        return;
    }

    // find max weight for normalisation
    let max_weight =
        edges
            .iter()
            .map(|(_, _, w)| *w)
            .fold(T::zero(), |acc, w| if w > acc { w } else { acc });

    // epochs_per_sample determines sampling frequency (higher weight = more frequent)
    let epochs_per_sample: Vec<T> = edges
        .iter()
        .map(|(_, _, w)| {
            let norm = *w / max_weight;
            if norm > T::zero() {
                T::one() / norm
            } else {
                T::from(1e8).unwrap() // effectively infinite
            }
        })
        .collect();

    let mut epoch_of_next_sample: Vec<T> = epochs_per_sample.clone();

    let epochs_per_neg_sample: Vec<T> = epochs_per_sample
        .iter()
        .map(|eps| *eps / T::from(params.neg_sample_rate).unwrap())
        .collect();
    let mut epoch_of_next_neg_sample: Vec<T> = epochs_per_neg_sample.clone();

    // lr schedule
    let lr_schedule: Vec<T> = (0..params.n_epochs)
        .map(|e| params.lr * (T::one() - T::from(e).unwrap() / T::from(params.n_epochs).unwrap()))
        .collect();

    // RNG state per vertex
    let mut rng_states: Vec<StdRng> = (0..n)
        .map(|i| StdRng::seed_from_u64(seed + i as u64))
        .collect();

    // prepare all of the constants to avoid re-calculations
    let consts = OptimConstants::new(params.a, params.b, params.gamma);

    // main loop
    for epoch in 0..params.n_epochs {
        let lr = lr_schedule[epoch];
        let epoch_t = T::from(epoch).unwrap();

        // only process edges whose epoch_of_next_sample has arrived
        for (edge_idx, &(i, j, _weight)) in edges.iter().enumerate() {
            if epoch_of_next_sample[edge_idx] > epoch_t {
                continue;
            }

            apply_attractive_force_flat(&mut embd_flat, i, j, n_dim, &consts, lr);

            epoch_of_next_sample[edge_idx] =
                epoch_of_next_sample[edge_idx] + epochs_per_sample[edge_idx];

            // adapative negative sampling...
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

            // update negative sampling tracker
            epoch_of_next_neg_sample[edge_idx] = epoch_of_next_neg_sample[edge_idx]
                + T::from(n_neg_samples).unwrap() * epochs_per_neg_sample[edge_idx];
        }

        if verbose && ((epoch + 1) % 100 == 0 || epoch + 1 == params.n_epochs) {
            println!(" Completed epoch {}/{}", epoch + 1, params.n_epochs);
        }
    }

    // unflatten embedding
    for (i, point) in embd.iter_mut().enumerate() {
        let base = i * n_dim;
        point.copy_from_slice(&embd_flat[base..base + n_dim]);
    }
}

/// Optimise embedding using Adam matching uwot/UMAP behaviour
///
/// KEY DIFFERENCES from your version:
/// 1. Edges sampled based on epochs_per_sample (not all edges every epoch)
/// 2. NO gradient normalisation by edge count
/// 3. Negative sampling tracked per edge
/// 4. Bias correction applied to moments before computing update
pub fn optimise_embedding_adam<T>(
    embd: &mut [Vec<T>],
    graph: &[Vec<(usize, T)>],
    params: &OptimParams<T>,
    seed: u64,
    verbose: bool,
) where
    T: Float + FromPrimitive + Send + Sync,
{
    let n = embd.len();
    let n_dim = embd[0].len();

    // Flatten embedding
    let mut embd_flat: Vec<T> = Vec::with_capacity(n * n_dim);
    for point in embd.iter() {
        embd_flat.extend_from_slice(point);
    }

    let consts = OptimConstants::new(params.a, params.b, params.gamma);

    // Build edge list
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

    // Learning rate schedule
    let n_epochs_f = T::from(params.n_epochs).unwrap();
    let lr_schedule: Vec<T> = (0..params.n_epochs)
        .map(|e| params.lr * (T::one() - T::from(e).unwrap() / n_epochs_f))
        .collect();

    // Adam state - PER COORDINATE (not per vertex!)
    let mut m: Vec<T> = vec![T::zero(); n * n_dim];
    let mut v: Vec<T> = vec![T::zero(); n * n_dim];

    // Track timestep per coordinate (for bias correction)
    let mut timesteps: Vec<usize> = vec![0; n * n_dim];

    let mut rng_states: Vec<StdRng> = (0..n)
        .map(|i| StdRng::seed_from_u64(seed + i as u64))
        .collect();

    // Main optimization loop
    for epoch in 0..params.n_epochs {
        let lr = lr_schedule[epoch];
        let epoch_t = T::from(epoch).unwrap();

        // Process edges one at a time (matching uwot's EdgeWorker behavior)
        for (edge_idx, &(i, j, _weight)) in edges.iter().enumerate() {
            if epoch_of_next_sample[edge_idx] > epoch_t {
                continue;
            }

            let base_i = i * n_dim;
            let base_j = j * n_dim;

            // Compute attractive gradient for this edge only
            let mut dist_sq = T::zero();
            for d in 0..n_dim {
                let diff = embd_flat[base_i + d] - embd_flat[base_j + d];
                dist_sq = dist_sq + diff * diff;
            }

            if dist_sq >= T::from(1e-8).unwrap() {
                let dist_sq_b = dist_sq.powf(consts.b);
                let denom = T::one() + consts.a * dist_sq_b;
                let grad_coeff = consts.two_a_b * dist_sq_b / (dist_sq * denom);

                // Update vertices i and j with Adam immediately
                for d in 0..n_dim {
                    let delta = embd_flat[base_j + d] - embd_flat[base_i + d];
                    let grad = grad_coeff * delta;

                    // Update vertex i
                    let idx_i = base_i + d;
                    timesteps[idx_i] += 1;
                    let t = T::from(timesteps[idx_i]).unwrap();

                    m[idx_i] = params.beta1 * m[idx_i] + (T::one() - params.beta1) * grad;
                    v[idx_i] = params.beta2 * v[idx_i] + (T::one() - params.beta2) * grad * grad;

                    let m_hat = m[idx_i] / (T::one() - params.beta1.powf(t));
                    let v_hat = v[idx_i] / (T::one() - params.beta2.powf(t));

                    embd_flat[idx_i] = embd_flat[idx_i] + lr * m_hat / (v_hat.sqrt() + params.eps);

                    // Update vertex j (symmetric)
                    let idx_j = base_j + d;
                    timesteps[idx_j] += 1;
                    let t_j = T::from(timesteps[idx_j]).unwrap();

                    m[idx_j] = params.beta1 * m[idx_j] - (T::one() - params.beta1) * grad;
                    v[idx_j] = params.beta2 * v[idx_j] + (T::one() - params.beta2) * grad * grad;

                    let m_hat_j = m[idx_j] / (T::one() - params.beta1.powf(t_j));
                    let v_hat_j = v[idx_j] / (T::one() - params.beta2.powf(t_j));

                    embd_flat[idx_j] =
                        embd_flat[idx_j] + lr * m_hat_j / (v_hat_j.sqrt() + params.eps);
                }
            }

            epoch_of_next_sample[edge_idx] =
                epoch_of_next_sample[edge_idx] + epochs_per_sample[edge_idx];

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

                let mut dist_sq = T::zero();
                for d in 0..n_dim {
                    let diff = embd_flat[base_i + d] - embd_flat[base_k + d];
                    dist_sq = dist_sq + diff * diff;
                }

                let dist_sq_safe = dist_sq + consts.eps;
                let dist_sq_b = dist_sq_safe.powf(consts.b);
                let denom = dist_sq_safe * (T::one() + consts.a * dist_sq_b);
                let grad_coeff = (consts.two_b / denom)
                    .max(-consts.clip_val)
                    .min(consts.clip_val);

                // Update vertex i with Adam immediately (only i, not k)
                for d in 0..n_dim {
                    let delta = embd_flat[base_i + d] - embd_flat[base_k + d];
                    let grad = grad_coeff * delta;

                    let idx = base_i + d;
                    timesteps[idx] += 1;
                    let t = T::from(timesteps[idx]).unwrap();

                    m[idx] = params.beta1 * m[idx] + (T::one() - params.beta1) * grad;
                    v[idx] = params.beta2 * v[idx] + (T::one() - params.beta2) * grad * grad;

                    let m_hat = m[idx] / (T::one() - params.beta1.powf(t));
                    let v_hat = v[idx] / (T::one() - params.beta2.powf(t));

                    embd_flat[idx] = embd_flat[idx] + lr * m_hat / (v_hat.sqrt() + params.eps);
                }
            }

            epoch_of_next_neg_sample[edge_idx] = epoch_of_next_neg_sample[edge_idx]
                + T::from(n_neg_samples).unwrap() * epochs_per_neg_sample[edge_idx];
        }

        if verbose && ((epoch + 1) % 100 == 0 || epoch + 1 == params.n_epochs) {
            println!(" Completed epoch {}/{}", epoch + 1, params.n_epochs);
        }
    }

    // Unflatten
    for (i, point) in embd.iter_mut().enumerate() {
        let base = i * n_dim;
        point.copy_from_slice(&embd_flat[base..base + n_dim]);
    }
}

///////////
// Tests //
///////////

// #[cfg(test)]
// mod test_optimiser {
//     use super::*;
//     use approx::assert_relative_eq;

//     #[test]
//     fn test_optim_params_default_2d() {
//         let params = OptimParams::<f64>::default_2d();

//         assert_relative_eq!(params.a, 1.929, epsilon = 1e-6);
//         assert_relative_eq!(params.b, 0.7915, epsilon = 1e-6);
//         assert_eq!(params.lr, 1.0);
//         assert_eq!(params.n_epochs, 500);
//         assert_eq!(params.neg_sample_rate, 5);
//         assert_relative_eq!(params.min_dist, 0.1, epsilon = 1e-6);
//     }

//     #[test]
//     fn test_optim_params_from_min_dist_spread() {
//         let params =
//             OptimParams::<f64>::from_min_dist_spread(0.1, 1.0, 1.0, 500, 5, None, None, None);

//         assert!(params.a > 0.0);
//         assert!(params.b > 0.0);
//         assert_eq!(params.lr, 1.0);
//         assert_eq!(params.n_epochs, 500);
//         assert_eq!(params.neg_sample_rate, 5);
//         assert_relative_eq!(params.min_dist, 0.1, epsilon = 1e-6);
//     }

//     #[test]
//     fn test_fit_params_constraints() {
//         let (a, b) = OptimParams::<f64>::fit_params(0.1, 1.0, None, None);

//         // Parameters should be within reasonable bounds
//         assert!((0.001..=10.0).contains(&a));
//         assert!((0.1..=2.0).contains(&b));
//     }

//     #[test]
//     fn test_fit_params_curve_properties() {
//         let min_dist = 0.1;
//         let spread = 1.0;
//         let (a, b) = OptimParams::<f64>::fit_params(min_dist, spread, None, None);

//         // Test that curve satisfies approximate requirements
//         // f(min_dist) should be high (close to 1.0)
//         let pred_min = 1.0 / (1.0 + a * min_dist.powf(2.0 * b));
//         assert!(
//             pred_min > 0.7,
//             "f(min_dist) = {:.3} should be > 0.7",
//             pred_min
//         );

//         // f(3*spread) should be low (close to 0.0)
//         let pred_spread = 1.0 / (1.0 + a * (3.0 * spread).powf(2.0 * b));
//         assert!(
//             pred_spread < 0.3,
//             "f(3*spread) = {:.3} should be < 0.3",
//             pred_spread
//         );

//         // Verify the curve is monotonically decreasing between these points
//         let mid_point = 1.5 * spread;
//         let pred_mid = 1.0 / (1.0 + a * mid_point.powf(2.0 * b));
//         assert!(
//             pred_min > pred_mid && pred_mid > pred_spread,
//             "Curve should be monotonically decreasing"
//         );
//     }

//     #[test]
//     fn test_squared_dist_basic() {
//         let embd = vec![0.0, 0.0, 3.0, 4.0]; // two points flattened
//         let dist = squared_dist_flat(&embd, 0, 1, 2);
//         assert_relative_eq!(dist, 25.0, epsilon = 1e-6); // 3² + 4² = 25
//     }

//     #[test]
//     fn test_squared_dist_identical_points() {
//         let embd = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]; // two identical points flattened
//         let dist = squared_dist_flat(&embd, 0, 1, 3);
//         assert_relative_eq!(dist, 0.0, epsilon = 1e-6);
//     }

//     #[test]
//     fn test_optimise_embedding_basic() {
//         // Simple graph with 3 connected vertices
//         let graph = vec![
//             vec![(1, 1.0), (2, 0.5)],
//             vec![(0, 1.0), (2, 1.0)],
//             vec![(0, 0.5), (1, 1.0)],
//         ];

//         let mut embd = vec![vec![0.0, 0.0], vec![5.0, 0.0], vec![0.0, 5.0]];
//         let initial_embd = embd.clone();

//         // Use default params which have n_epochs = 500
//         let params = OptimParams::default_2d();

//         optimise_embedding_adam(&mut embd, &graph, &params, 42, false);

//         // Check that points moved
//         let total_movement: f64 = embd
//             .iter()
//             .zip(initial_embd.iter())
//             .map(|(new, old)| {
//                 new.iter()
//                     .zip(old.iter())
//                     .map(|(&n, &o)| (n - o).abs())
//                     .sum::<f64>()
//             })
//             .sum();

//         assert!(total_movement > 0.01, "Total movement: {}", total_movement);

//         // All coordinates should still be finite
//         for point in &embd {
//             for &coord in point {
//                 assert!(coord.is_finite());
//             }
//         }
//     }

//     #[test]
//     fn test_optimise_embedding_empty_graph() {
//         let graph: Vec<Vec<(usize, f64)>> = vec![vec![], vec![], vec![]];
//         let mut embd = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];

//         let params = OptimParams::default_2d();

//         optimise_embedding_adam(&mut embd, &graph, &params, 42, false);

//         // With no edges, only negative sampling occurs, so embedding should change
//         for point in &embd {
//             for &coord in point {
//                 assert!(coord.is_finite());
//             }
//         }
//     }

//     #[test]
//     fn test_optimise_embedding_reproducibility() {
//         let graph = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
//         let mut embd1 = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
//         let mut embd2 = vec![vec![0.0, 0.0], vec![1.0, 0.0]];

//         let params = OptimParams {
//             a: 1.0,
//             b: 1.0,
//             lr: 0.5,
//             n_epochs: 10,
//             neg_sample_rate: 2,
//             min_dist: 0.1,
//             beta1: 0.5,
//             beta2: 0.9,
//             eps: 1e-7,
//         };

//         optimise_embedding_adam(&mut embd1, &graph, &params, 42, false);
//         optimise_embedding_adam(&mut embd2, &graph, &params, 42, false);

//         assert_eq!(embd1, embd2);
//     }

//     #[test]
//     fn test_optimise_embedding_convergence() {
//         // Two strongly connected points should end up closer
//         let graph = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
//         let mut embd = vec![vec![0.0, 0.0], vec![10.0, 0.0]];

//         // Compute initial distance using flat representation
//         let embd_flat: Vec<f64> = embd.iter().flatten().copied().collect();
//         let initial_dist = squared_dist_flat(&embd_flat, 0, 1, 2).sqrt();

//         let params = OptimParams {
//             a: 1.0,
//             b: 1.0,
//             lr: 1.0,
//             n_epochs: 100,
//             neg_sample_rate: 2,
//             min_dist: 0.1,
//             beta1: 0.5,
//             beta2: 0.9,
//             eps: 1e-7,
//         };

//         optimise_embedding_adam(&mut embd, &graph, &params, 42, false);

//         // Compute final distance
//         let embd_flat: Vec<f64> = embd.iter().flatten().copied().collect();
//         let final_dist = squared_dist_flat(&embd_flat, 0, 1, 2).sqrt();

//         // Distance should decrease
//         assert!(final_dist < initial_dist);
//     }

//     #[test]
//     fn test_sgd_vs_adam_both_converge() {
//         // Compare parallel SGD vs Adam - both should converge
//         let graph = vec![
//             vec![(1, 1.0), (2, 0.5)],
//             vec![(0, 1.0), (2, 1.0)],
//             vec![(0, 0.5), (1, 1.0)],
//         ];

//         let initial_embd = vec![vec![0.0, 0.0], vec![10.0, 0.0], vec![0.0, 10.0]];

//         let params = OptimParams {
//             a: 1.0,
//             b: 1.0,
//             lr: 1.0,
//             n_epochs: 50,
//             neg_sample_rate: 2,
//             min_dist: 0.1,
//             beta1: 0.5,
//             beta2: 0.9,
//             eps: 1e-7,
//         };

//         // Test SGD
//         let mut embd_sgd = initial_embd.clone();
//         optimise_embedding_sgd(&mut embd_sgd, &graph, &params, 42, false);

//         // Test Adam
//         let mut embd_adam = initial_embd.clone();
//         optimise_embedding_adam(&mut embd_adam, &graph, &params, 42, false);

//         // Both should move points significantly from initial positions
//         let movement_sgd: f64 = embd_sgd
//             .iter()
//             .zip(initial_embd.iter())
//             .map(|(new, old)| {
//                 new.iter()
//                     .zip(old.iter())
//                     .map(|(&n, &o)| (n - o).abs())
//                     .sum::<f64>()
//             })
//             .sum();

//         let movement_adam: f64 = embd_adam
//             .iter()
//             .zip(initial_embd.iter())
//             .map(|(new, old)| {
//                 new.iter()
//                     .zip(old.iter())
//                     .map(|(&n, &o)| (n - o).abs())
//                     .sum::<f64>()
//             })
//             .sum();

//         assert!(movement_sgd > 1.0, "SGD should move points");
//         assert!(movement_adam > 1.0, "Adam should move points");

//         // Both should produce finite results
//         for point in embd_sgd.iter().chain(embd_adam.iter()) {
//             for &coord in point {
//                 assert!(coord.is_finite());
//             }
//         }
//     }

//     #[test]
//     fn test_parallel_sgd_reproducibility() {
//         let graph = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
//         let mut embd1 = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
//         let mut embd2 = vec![vec![0.0, 0.0], vec![1.0, 0.0]];

//         let params = OptimParams {
//             a: 1.0,
//             b: 1.0,
//             lr: 0.5,
//             n_epochs: 10,
//             neg_sample_rate: 2,
//             min_dist: 0.1,
//             beta1: 0.5,
//             beta2: 0.9,
//             eps: 1e-7,
//         };

//         optimise_embedding_sgd(&mut embd1, &graph, &params, 42, false);
//         optimise_embedding_sgd(&mut embd2, &graph, &params, 42, false);

//         // Parallel SGD should be reproducible with same seed
//         assert_eq!(embd1, embd2);
//     }

//     #[test]
//     fn test_optimisation_preserves_graph_structure_adam() {
//         // Create a graph with two distinct cliques that are weakly connected
//         // Clique 1: vertices 0, 1, 2 (strongly connected)
//         // Clique 2: vertices 3, 4, 5 (strongly connected)
//         // Weak bridge: 2 <-> 3

//         let graph = vec![
//             vec![(1, 1.0), (2, 1.0)],           // 0: connected to 1, 2
//             vec![(0, 1.0), (2, 1.0)],           // 1: connected to 0, 2
//             vec![(0, 1.0), (1, 1.0), (3, 0.1)], // 2: connected to 0, 1, weak to 3
//             vec![(2, 0.1), (4, 1.0), (5, 1.0)], // 3: weak to 2, connected to 4, 5
//             vec![(3, 1.0), (5, 1.0)],           // 4: connected to 3, 5
//             vec![(3, 1.0), (4, 1.0)],           // 5: connected to 3, 4
//         ];

//         // Start with random positions (not clustered)
//         let mut embd = vec![
//             vec![0.0, 0.0],
//             vec![10.0, 0.0],
//             vec![0.0, 10.0],
//             vec![10.0, 10.0],
//             vec![-5.0, -5.0],
//             vec![15.0, 15.0],
//         ];

//         let params = OptimParams {
//             n_epochs: 200,
//             ..OptimParams::default_2d()
//         };

//         optimise_embedding_adam(&mut embd, &graph, &params, 42, false);

//         // Helper to compute distance
//         let dist = |a: &[f64], b: &[f64]| -> f64 {
//             ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
//         };

//         // Compute average intra-clique distance
//         let intra_clique1 =
//             (dist(&embd[0], &embd[1]) + dist(&embd[0], &embd[2]) + dist(&embd[1], &embd[2])) / 3.0;
//         let intra_clique2 =
//             (dist(&embd[3], &embd[4]) + dist(&embd[3], &embd[5]) + dist(&embd[4], &embd[5])) / 3.0;
//         let avg_intra = (intra_clique1 + intra_clique2) / 2.0;

//         // Compute average inter-clique distance (excluding the bridge)
//         let inter_distances = [
//             dist(&embd[0], &embd[3]),
//             dist(&embd[0], &embd[4]),
//             dist(&embd[0], &embd[5]),
//             dist(&embd[1], &embd[3]),
//             dist(&embd[1], &embd[4]),
//             dist(&embd[1], &embd[5]),
//         ];
//         let avg_inter: f64 = inter_distances.iter().sum::<f64>() / inter_distances.len() as f64;

//         // Points within cliques should be much closer than points between cliques
//         assert!(
//             avg_inter > avg_intra * 1.5,
//             "Inter-clique dist ({:.2}) should be > 1.5x intra-clique dist ({:.2})",
//             avg_inter,
//             avg_intra
//         );
//     }

//     #[test]
//     fn test_optimisation_preserves_graph_structure_sgd() {
//         // Create a graph with two distinct cliques that are weakly connected
//         // Clique 1: vertices 0, 1, 2 (strongly connected)
//         // Clique 2: vertices 3, 4, 5 (strongly connected)
//         // Weak bridge: 2 <-> 3

//         let graph = vec![
//             vec![(1, 1.0), (2, 1.0)],           // 0: connected to 1, 2
//             vec![(0, 1.0), (2, 1.0)],           // 1: connected to 0, 2
//             vec![(0, 1.0), (1, 1.0), (3, 0.1)], // 2: connected to 0, 1, weak to 3
//             vec![(2, 0.1), (4, 1.0), (5, 1.0)], // 3: weak to 2, connected to 4, 5
//             vec![(3, 1.0), (5, 1.0)],           // 4: connected to 3, 5
//             vec![(3, 1.0), (4, 1.0)],           // 5: connected to 3, 4
//         ];

//         // Start with random positions (not clustered)
//         let mut embd = vec![
//             vec![0.0, 0.0],
//             vec![10.0, 0.0],
//             vec![0.0, 10.0],
//             vec![10.0, 10.0],
//             vec![-5.0, -5.0],
//             vec![15.0, 15.0],
//         ];

//         let params = OptimParams {
//             n_epochs: 200,
//             ..OptimParams::default_2d()
//         };

//         optimise_embedding_sgd(&mut embd, &graph, &params, 42, false);

//         // Helper to compute distance
//         let dist = |a: &[f64], b: &[f64]| -> f64 {
//             ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
//         };

//         // Compute average intra-clique distance
//         let intra_clique1 =
//             (dist(&embd[0], &embd[1]) + dist(&embd[0], &embd[2]) + dist(&embd[1], &embd[2])) / 3.0;
//         let intra_clique2 =
//             (dist(&embd[3], &embd[4]) + dist(&embd[3], &embd[5]) + dist(&embd[4], &embd[5])) / 3.0;
//         let avg_intra = (intra_clique1 + intra_clique2) / 2.0;

//         // Compute average inter-clique distance (excluding the bridge)
//         let inter_distances = [
//             dist(&embd[0], &embd[3]),
//             dist(&embd[0], &embd[4]),
//             dist(&embd[0], &embd[5]),
//             dist(&embd[1], &embd[3]),
//             dist(&embd[1], &embd[4]),
//             dist(&embd[1], &embd[5]),
//         ];
//         let avg_inter: f64 = inter_distances.iter().sum::<f64>() / inter_distances.len() as f64;

//         // Points within cliques should be much closer than points between cliques
//         assert!(
//             avg_inter > avg_intra * 1.5,
//             "Inter-clique dist ({:.2}) should be > 1.5x intra-clique dist ({:.2})",
//             avg_inter,
//             avg_intra
//         );
//     }
// }
