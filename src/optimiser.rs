use core::f64;
use num_traits::{Float, FromPrimitive};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::RwLock;

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
            n_epochs: 500,
            neg_sample_rate: 5,
            min_dist: T::from_f64(0.1).unwrap(),
            beta1: T::from(0.5).unwrap(),
            beta2: T::from(0.9).unwrap(),
            eps: T::from(1e-7).unwrap(),
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
        n_epochs: usize,
        neg_sample_rate: usize,
        beta1: Option<T>,
        beta2: Option<T>,
        eps: Option<T>,
    ) -> Self {
        // take the Adam-related values
        let beta1 = beta1.unwrap_or(T::from(0.5).unwrap());
        let beta2 = beta2.unwrap_or(T::from(0.9).unwrap());
        let eps = eps.unwrap_or(T::from(1e-7).unwrap());

        let (a, b) = Self::fit_params(min_dist, spread, None, None);
        Self {
            a,
            b,
            lr,
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

/// Adam optimiser state for a single point
#[derive(Clone)]
struct AdamState<T> {
    m: Vec<T>, // First moment estimate
    v: Vec<T>, // Second moment estimate
}

impl<T: Float> AdamState<T> {
    fn new(n_dim: usize) -> Self {
        Self {
            m: vec![T::zero(); n_dim],
            v: vec![T::zero(); n_dim],
        }
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
/// ### Params
///
/// * `a` - Point a
/// * `b` - Point b
///
/// ### Returns
///
/// Squared distance between two points
#[inline(always)]
fn squared_dist<T>(a: &[T], b: &[T]) -> T
where
    T: Float,
{
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .fold(T::zero(), |acc, x| acc + x)
}

/// Compute attractive force gradient (without applying it)
///
/// Returns the gradient contribution for vertex `i` from its connection to
/// vertex `j`.
///
/// ### Params
///
/// * `embd` - Current embedding coordinates (read-only - avoid race conditions)
/// * `i` - Source vertex index
/// * `j` - Target vertex index
/// * `w` - Edge weight
/// * `a` - Curve parameter
/// * `b` - Curve parameter
///
/// ### Returns
///
/// Gradient vector for vertex `i`, or None if points are too close
#[inline]
fn compute_attractive_gradient<T>(
    embd: &[Vec<T>],
    i: usize,
    j: usize,
    w: T,
    a: T,
    b: T,
) -> Option<Vec<T>>
where
    T: Float,
{
    let dist_sq = squared_dist(&embd[i], &embd[j]);

    // early return
    if dist_sq < T::from(1e-8).unwrap() {
        return None;
    }

    let dist = dist_sq.sqrt();
    let two_b = T::from(2.0).unwrap() * b;
    let dist_pow = dist.powf(two_b - T::one());

    let two = T::from(2.0).unwrap();
    let grad_coeff = w * two * a * b * dist_pow / (dist * (T::one() + a * dist.powf(two_b)));

    let n_dim = embd[i].len();
    let mut grad = vec![T::zero(); n_dim];
    for d in 0..n_dim {
        let delta = embd[j][d] - embd[i][d];
        grad[d] = grad_coeff * delta;
    }

    Some(grad)
}

/// Compute repulsive force gradient (without applying it)
///
/// Returns the gradient contribution for vertex `i` from repulsion against
/// vertex `j`.
///
/// ### Params
///
/// * `embd` - Current embedding coordinates (read-only - avoid race conditions)
/// * `i` - Source vertex index
/// * `j` - Negative sample vertex index
/// * `a` - Curve parameter
/// * `b` - Curve parameter
///
/// ### Returns
///
/// Gradient vector for vertex `i` (repulsive, so negative)
#[inline]
fn compute_repulsive_gradient<T>(embd: &[Vec<T>], i: usize, j: usize, a: T, b: T) -> Vec<T>
where
    T: Float,
{
    let dist_sq = squared_dist(&embd[i], &embd[j]);
    let dist_sq_safe = dist_sq + T::from(0.001).unwrap();

    let four = T::from(4.0).unwrap();
    let denom = T::one() + a * dist_sq_safe.sqrt().powf(T::from(2.0).unwrap() * b);
    let grad_coeff = b / denom;

    let n_dim = embd[i].len();
    let mut grad = vec![T::zero(); n_dim];
    for d in 0..n_dim {
        let delta = embd[i][d] - embd[j][d]; // Points from j to i (away from j)
        let raw_grad = grad_coeff * delta;
        let clipped = raw_grad.max(-four).min(four);
        grad[d] = clipped;
    }

    grad
}

////////////////////
// Main functions //
////////////////////

/// Optimise embedding using parallel Adam optimiser
///
/// Iteratively applies attractive forces (from connected edges) and repulsive
/// forces (via negative sampling) to optimise the low-dimensional embedding.
/// Uses Adam optimisation with parallel processing for speed.
///
/// ### Params
///
/// * `embd` - Initial embedding coordinates (modified in place)
/// * `graph` - Adjacency list representation of the high-dimensional graph
/// * `params` - Optimisation parameters (a, b, learning rate, epochs, etc.)
/// * `seed` - Random seed for negative sampling
/// * `verbose` - Controls verbosity of the function
///
/// ### Notes
///
/// Processes edges in random order each epoch.
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

    // initialise the adam states
    let mut adam_states: Vec<AdamState<T>> = (0..n).map(|_| AdamState::new(n_dim)).collect();

    // rwlock this
    let embd_lock = RwLock::new(embd);

    for epoch in 0..params.n_epochs {
        let t = T::from(epoch + 1).unwrap(); // Adam timestep (1-indexed)
        let alpha = T::from(epoch as f64 / params.n_epochs as f64).unwrap();
        let lr = params.lr * (T::one() - alpha);

        // calculate gradients in parallel
        let gradients: Vec<Vec<T>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut grad = vec![T::zero(); n_dim];
                let mut rng = StdRng::seed_from_u64(seed + epoch as u64 * n as u64 + i as u64);

                if graph[i].is_empty() {
                    return grad;
                }

                // shuffle edges for the vertex
                let mut edges = graph[i].clone();
                for idx in (1..edges.len()).rev() {
                    let swap_idx = rng.random_range(0..=idx);
                    edges.swap(idx, swap_idx);
                }

                let embd_read = embd_lock.read().unwrap();

                for &(j, w) in &edges {
                    if let Some(attr_grad) =
                        compute_attractive_gradient(&embd_read, i, j, w, params.a, params.b)
                    {
                        for d in 0..n_dim {
                            grad[d] = grad[d] + attr_grad[d];
                        }
                    }

                    for _ in 0..params.neg_sample_rate {
                        let k = rng.random_range(0..n);
                        if k == i {
                            continue;
                        }

                        // Repulsive force gradient using helper function
                        let rep_grad =
                            compute_repulsive_gradient(&embd_read, i, k, params.a, params.b);
                        for d in 0..n_dim {
                            grad[d] = grad[d] + rep_grad[d];
                        }
                    }
                }

                grad
            })
            .collect();

        // apply the updates sequentially
        let mut embd_write = embd_lock.write().unwrap();
        for i in 0..n {
            let state = &mut adam_states[i];

            for d in 0..n_dim {
                // update biased first moment estimate
                state.m[d] =
                    params.beta1 * state.m[d] + (T::one() - params.beta1) * gradients[i][d];

                // update biased second moment estimate
                state.v[d] = params.beta2 * state.v[d]
                    + (T::one() - params.beta2) * gradients[i][d] * gradients[i][d];

                // compute bias-corrected moments
                let m_hat = state.m[d] / (T::one() - params.beta1.powf(t));
                let v_hat = state.v[d] / (T::one() - params.beta2.powf(t));

                // Adam update
                embd_write[i][d] = embd_write[i][d] + lr * m_hat / (v_hat.sqrt() + params.eps);
            }
        }

        // print progress every 100 epochs
        if verbose & ((epoch + 1) % 100 == 0 || epoch + 1 == params.n_epochs) {
            println!(" Completed epoch {}/{}", epoch + 1, params.n_epochs);
        }
    }
}

/// Optimise embedding using parallel vanilla SGD
///
/// Uses the same parallel gradient computation as Adam but applies simple
/// SGD updates without momentum. Useful for comparing optimisation algorithms
/// on equal footing (both parallelised).
///
/// ### Params
///
/// * `embd` - Initial embedding coordinates (modified in place)
/// * `graph` - Adjacency list representation of the high-dimensional graph
/// * `params` - Optimisation parameters
/// * `seed` - Random seed for negative sampling
/// * `verbose` - Controls verbosity of the function
///
/// ### Notes
///
/// This uses parallel gradient computation (same as Adam) but vanilla SGD
/// updates. Adam is likely better, but maybe interesting for comparisons
pub fn optimise_embedding_sgd<T>(
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

    let embd_lock = RwLock::new(embd);

    for epoch in 0..params.n_epochs {
        let alpha = T::from(epoch as f64 / params.n_epochs as f64).unwrap();
        let lr = params.lr * (T::one() - alpha);

        let gradients: Vec<Vec<T>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut grad = vec![T::zero(); n_dim];
                let mut rng = StdRng::seed_from_u64(seed + epoch as u64 * n as u64 + i as u64);

                if graph[i].is_empty() {
                    return grad;
                }

                // shuffle edges for this vertex
                let mut edges = graph[i].clone();
                for idx in (1..edges.len()).rev() {
                    let swap_idx = rng.random_range(0..=idx);
                    edges.swap(idx, swap_idx);
                }

                // compute gradients for this vertex
                let embd_read = embd_lock.read().unwrap();

                for &(j, w) in &edges {
                    // attractive force gradient
                    if let Some(attr_grad) =
                        compute_attractive_gradient(&embd_read, i, j, w, params.a, params.b)
                    {
                        for d in 0..n_dim {
                            grad[d] = grad[d] + attr_grad[d];
                        }
                    }

                    // negative sampling for repulsive forces
                    for _ in 0..params.neg_sample_rate {
                        let k = rng.random_range(0..n);
                        if k == i {
                            continue;
                        }

                        // repulsive force gradient
                        let rep_grad =
                            compute_repulsive_gradient(&embd_read, i, k, params.a, params.b);
                        for d in 0..n_dim {
                            grad[d] = grad[d] + rep_grad[d];
                        }
                    }
                }

                grad
            })
            .collect();

        // Apply vanilla SGD updates
        let mut embd_write = embd_lock.write().unwrap();
        for i in 0..n {
            for d in 0..n_dim {
                // Simple SGD: x += lr * grad
                embd_write[i][d] = embd_write[i][d] + lr * gradients[i][d];
            }
        }

        // print progress every 100 epochs
        if verbose & ((epoch + 1) % 100 == 0 || epoch + 1 == params.n_epochs) {
            println!(" Completed epoch {}/{}", epoch + 1, params.n_epochs);
        }
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
        assert_eq!(params.n_epochs, 500);
        assert_eq!(params.neg_sample_rate, 5);
        assert_relative_eq!(params.min_dist, 0.1, epsilon = 1e-6);
    }

    #[test]
    fn test_optim_params_from_min_dist_spread() {
        let params =
            OptimParams::<f64>::from_min_dist_spread(0.1, 1.0, 1.0, 500, 5, None, None, None);

        assert!(params.a > 0.0);
        assert!(params.b > 0.0);
        assert_eq!(params.lr, 1.0);
        assert_eq!(params.n_epochs, 500);
        assert_eq!(params.neg_sample_rate, 5);
        assert_relative_eq!(params.min_dist, 0.1, epsilon = 1e-6);
    }

    #[test]
    fn test_fit_params_constraints() {
        let (a, b) = OptimParams::<f64>::fit_params(0.1, 1.0, None, None);

        // Parameters should be within reasonable bounds
        assert!((0.001..=10.0).contains(&a));
        assert!((0.1..=2.0).contains(&b));
    }

    #[test]
    fn test_fit_params_curve_properties() {
        let min_dist = 0.1;
        let spread = 1.0;
        let (a, b) = OptimParams::<f64>::fit_params(min_dist, spread, None, None);

        // Test that curve satisfies approximate requirements
        // f(min_dist) should be high (close to 1.0)
        let pred_min = 1.0 / (1.0 + a * min_dist.powf(2.0 * b));
        assert!(
            pred_min > 0.7,
            "f(min_dist) = {:.3} should be > 0.7",
            pred_min
        );

        // f(3*spread) should be low (close to 0.0)
        let pred_spread = 1.0 / (1.0 + a * (3.0 * spread).powf(2.0 * b));
        assert!(
            pred_spread < 0.3,
            "f(3*spread) = {:.3} should be < 0.3",
            pred_spread
        );

        // Verify the curve is monotonically decreasing between these points
        let mid_point = 1.5 * spread;
        let pred_mid = 1.0 / (1.0 + a * mid_point.powf(2.0 * b));
        assert!(
            pred_min > pred_mid && pred_mid > pred_spread,
            "Curve should be monotonically decreasing"
        );
    }

    #[test]
    fn test_squared_dist_basic() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        let dist = squared_dist(&a, &b);
        assert_relative_eq!(dist, 25.0, epsilon = 1e-6); // 3² + 4² = 25
    }

    #[test]
    fn test_squared_dist_identical_points() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];

        let dist = squared_dist(&a, &b);
        assert_relative_eq!(dist, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_optimise_embedding_basic() {
        // Simple graph with 3 connected vertices
        let graph = vec![
            vec![(1, 1.0), (2, 0.5)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 0.5), (1, 1.0)],
        ];

        let mut embd = vec![vec![0.0, 0.0], vec![5.0, 0.0], vec![0.0, 5.0]];
        let initial_embd = embd.clone();

        // Use default params which have n_epochs = 500
        let params = OptimParams::default_2d();

        optimise_embedding_adam(&mut embd, &graph, &params, 42, false);

        // Check that points moved
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

        assert!(total_movement > 0.01, "Total movement: {}", total_movement);

        // All coordinates should still be finite
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

        // With no edges, only negative sampling occurs, so embedding should change
        for point in &embd {
            for &coord in point {
                assert!(coord.is_finite());
            }
        }
    }

    #[test]
    fn test_optimise_embedding_reproducibility() {
        let graph = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
        let mut embd1 = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let mut embd2 = vec![vec![0.0, 0.0], vec![1.0, 0.0]];

        let params = OptimParams {
            a: 1.0,
            b: 1.0,
            lr: 0.5,
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
    fn test_optimise_embedding_convergence() {
        // Two strongly connected points should end up closer
        let graph = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
        let mut embd = vec![vec![0.0, 0.0], vec![10.0, 0.0]];

        let initial_dist = squared_dist(&embd[0], &embd[1]).sqrt();

        let params = OptimParams {
            a: 1.0,
            b: 1.0,
            lr: 1.0,
            n_epochs: 100,
            neg_sample_rate: 2,
            min_dist: 0.1,
            beta1: 0.5,
            beta2: 0.9,
            eps: 1e-7,
        };

        optimise_embedding_adam(&mut embd, &graph, &params, 42, false);

        let final_dist = squared_dist(&embd[0], &embd[1]).sqrt();

        // Distance should decrease
        assert!(final_dist < initial_dist);
    }

    #[test]
    fn test_sgd_vs_adam_both_converge() {
        // Compare parallel SGD vs Adam - both should converge
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
            n_epochs: 50,
            neg_sample_rate: 2,
            min_dist: 0.1,
            beta1: 0.5,
            beta2: 0.9,
            eps: 1e-7,
        };

        // Test SGD
        let mut embd_sgd = initial_embd.clone();
        optimise_embedding_sgd(&mut embd_sgd, &graph, &params, 42, false);

        // Test Adam
        let mut embd_adam = initial_embd.clone();
        optimise_embedding_adam(&mut embd_adam, &graph, &params, 42, false);

        // Both should move points significantly from initial positions
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

        assert!(movement_sgd > 1.0, "SGD should move points");
        assert!(movement_adam > 1.0, "Adam should move points");

        // Both should produce finite results
        for point in embd_sgd.iter().chain(embd_adam.iter()) {
            for &coord in point {
                assert!(coord.is_finite());
            }
        }
    }

    #[test]
    fn test_parallel_sgd_reproducibility() {
        let graph = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
        let mut embd1 = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let mut embd2 = vec![vec![0.0, 0.0], vec![1.0, 0.0]];

        let params = OptimParams {
            a: 1.0,
            b: 1.0,
            lr: 0.5,
            n_epochs: 10,
            neg_sample_rate: 2,
            min_dist: 0.1,
            beta1: 0.5,
            beta2: 0.9,
            eps: 1e-7,
        };

        optimise_embedding_sgd(&mut embd1, &graph, &params, 42, false);
        optimise_embedding_sgd(&mut embd2, &graph, &params, 42, false);

        // Parallel SGD should be reproducible with same seed
        assert_eq!(embd1, embd2);
    }
}
