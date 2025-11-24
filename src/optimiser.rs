use core::f64;

use num_traits::{Float, FromPrimitive};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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
#[derive(Clone, Debug)]
pub struct OptimParams<T> {
    pub a: T,
    pub b: T,
    pub lr: T,
    pub n_epochs: usize,
    pub neg_sample_rate: usize,
    pub min_dist: T,
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
    ///
    /// ### Return
    ///
    /// Self with calculated `a` and `b` parameter according to the
    pub fn from_min_dist_spread(
        min_dist: T,
        spread: T,
        lr: T,
        n_epochs: usize,
        neg_sample_rate: usize,
    ) -> Self {
        let (a, b) = Self::fit_params(min_dist, spread, None, None);
        Self {
            a,
            b,
            lr,
            n_epochs,
            neg_sample_rate,
            min_dist,
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

/// Apply attractive force gradient for a connected edge
///
/// Pulls points i and j together based on their edge weight. Uses the gradient
/// of the negative log-probability under UMAP's curve:
/// `phi(d) = 1/(1 + a * d^(2b))`
///
/// The gradient is: `derivate(-log(phi)) = 2ab * d^(2b-1) / (1 + a * d^(2b))`
///
/// ### Params
///
/// * `embd` - Current embedding coordinates (modified in place)
/// * `i` - Source vertex index
/// * `j` - Target vertex index
/// * `weight` - Edge weight (membership strength)
/// * `a` - Curve parameter controlling spread
/// * `b` - Curve parameter controlling tail behaviour
/// * `lr` - Step size for gradient update
///
/// ### Notes
///
/// Updates both points symmetrically: i moves towards j, j moves away from i.
/// Skips update if points are essentially at the same location (dist² < 1e-8).
#[inline]
fn apply_attractive_force<T>(embd: &mut [Vec<T>], i: usize, j: usize, weight: T, a: T, b: T, lr: T)
where
    T: Float,
{
    let dist_sq = squared_dist(&embd[i], &embd[j]);

    if dist_sq < T::from(1e-8).unwrap() {
        return;
    }

    let dist = dist_sq.sqrt();

    // gradient in UMAP: derivate of -log(phi) with phi 1/(1 + a * d^(2b))
    // = 2ab * d^(2b - 1) / (1 + a*d^(2b) )
    let two_b = T::from(2.0).unwrap() * b;
    let dist_pow = dist.powf(two_b - T::one());
    let grad_coeff = lr * weight * T::from(2.0).unwrap() * a * b * dist_pow
        / (dist * (T::one() + a * dist.powf(two_b)));

    let n_dim = embd[i].len();
    for d in 0..n_dim {
        let delta = embd[j][d] - embd[i][d];
        let update = grad_coeff * delta;

        embd[i][d] = embd[i][d] + update;
        embd[j][d] = embd[j][d] - update;
    }
}

/// Apply repulsive force gradient via negative sampling
///
/// Pushes point i away from randomly sampled point j. Uses the gradient of
/// the log-probability that i and j are NOT connected under UMAP's curve. This
/// function will be used with the negative samples.
///
/// The gradient is: `derivate(-log(phi)) = 2ab * d^(2b-1) / (1 + a * d^(2b))`
///
/// ### Params
///
/// * `embedding` - Current embedding coordinates (modified in place)
/// * `i` - Source vertex index
/// * `j` - Randomly sampled vertex index (negative sample)
/// * `a` - Curve parameter controlling spread
/// * `b` - Curve parameter controlling tail behaviour
/// * `learning_rate` - Step size for gradient update
///
/// ### Notes
///
/// Only updates point i (not j), as this is an asymmetric negative sampling
/// step. Skips update if denominator becomes too small (< 1e-8) to avoid
/// numerical issues
#[inline]
fn apply_repulsive_force<T>(embd: &mut [Vec<T>], i: usize, j: usize, a: T, b: T, lr: T)
where
    T: Float,
{
    let dist_sq = squared_dist(&embd[i], &embd[j]);

    // Add small constant to avoid division by zero (like UMAP Python does)
    let dist_sq_safe = dist_sq + T::from(0.001).unwrap();
    let two = T::from(2.0).unwrap();

    let dist_pow = if b == T::one() {
        dist_sq_safe
    } else {
        dist_sq_safe.powf(two * b)
    };

    let denom = T::one() + a * dist_pow;

    // Gradient: 2b / (d^2 * (1 + a*d^(2b)))
    let grad_coeff = lr * T::from(2.0).unwrap() * b / (dist_sq_safe * denom);

    let n_dim = embd[i].len();
    for d in 0..n_dim {
        let delta = embd[i][d] - embd[j][d];
        embd[i][d] = embd[i][d] - grad_coeff * delta;
    }
}

/// Optimise embedding using stochastic gradient descent
///
/// ### Params
///
/// * `embedding` - Initial embedding to optimise (modified in place).
/// * `graph` - Graph as adjacency list `(neighbour, weight)` for each vertex.
/// * `params` - Optimisation parameters, see `OptimParams`.
/// * `seed` - Random seed for negative sampling.
pub fn optimise_embedding<T>(
    embd: &mut [Vec<T>],
    graph: &[Vec<(usize, T)>],
    params: &OptimParams<T>,
    seed: u64,
) where
    T: Float + FromPrimitive,
{
    let n = embd.len();
    let mut rng = StdRng::seed_from_u64(seed);

    let n_edges = graph.iter().map(|v| v.len()).sum::<usize>();

    // build edge list with probabilities
    let mut edges = Vec::with_capacity(n_edges);
    for (i, neighbours) in graph.iter().enumerate() {
        for &(j, w) in neighbours {
            edges.push((i, j, w))
        }
    }

    let max_weight =
        edges
            .iter()
            .map(|(_, _, w)| *w)
            .fold(T::zero(), |acc, w| if w > acc { w } else { acc });

    // compute epochs per sample (higher weight = more updates)
    let epochs_per_sample = edges
        .iter()
        .map(|(_, _, w)| {
            let norm = (*w / max_weight).to_f64().unwrap();
            if norm > 0.0 {
                params.n_epochs as f64 / norm
            } else {
                f64::INFINITY
            }
        })
        .collect::<Vec<f64>>();

    // learning rate schedule
    let alpha_schedule = (0..params.n_epochs)
        .map(|epoch| {
            let t = T::from_usize(epoch).unwrap() / T::from_usize(params.n_epochs).unwrap();
            params.lr * (T::one() - t)
        })
        .collect::<Vec<T>>();

    // main loop
    let mut epoch_next_sample = vec![0.0; edges.len()];
    for epoch in 0..params.n_epochs {
        let lr = alpha_schedule[epoch];

        // process the positive edges
        for (edge_idx, &(i, j, w)) in edges.iter().enumerate() {
            if epoch_next_sample[edge_idx] > epoch as f64 {
                continue;
            }
            epoch_next_sample[edge_idx] += epochs_per_sample[edge_idx];

            // attractive force
            apply_attractive_force(embd, i, j, w, params.a, params.b, lr);

            // negative sampling for repulsive forces
            for _ in 0..params.neg_sample_rate {
                let k = rng.random_range(0..n);
                if k == i {
                    continue;
                }
                apply_repulsive_force(embd, i, k, params.a, params.b, lr);
            }
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
        let params = OptimParams::<f64>::from_min_dist_spread(0.1, 1.0, 1.0, 500, 5);

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
    fn test_apply_attractive_force() {
        let mut embd = vec![vec![0.0, 0.0], vec![1.0, 0.0]];

        let i = 0;
        let j = 1;
        let weight = 1.0;
        let a = 1.0;
        let b = 1.0;
        let lr = 0.1;

        apply_attractive_force(&mut embd, i, j, weight, a, b, lr);

        // Point i should move towards j (x increases)
        assert!(embd[0][0] > 0.0);
        // Point j should move away from i (x decreases)
        assert!(embd[1][0] < 1.0);
    }

    #[test]
    fn test_apply_attractive_force_zero_distance() {
        let mut embd = vec![vec![0.0, 0.0], vec![0.0, 0.0]];

        let i = 0;
        let j = 1;

        // Should not crash with points at same location
        apply_attractive_force(&mut embd, i, j, 1.0, 1.0, 1.0, 0.1);

        // Points should remain unchanged
        assert_eq!(embd[0], vec![0.0, 0.0]);
        assert_eq!(embd[1], vec![0.0, 0.0]);
    }

    #[test]
    fn test_apply_repulsive_force() {
        let mut embd = vec![vec![0.0, 0.0], vec![0.5, 0.0]];

        let i = 0;
        let j = 1;
        let a = 1.0;
        let b = 1.0;
        let lr = 0.1;

        let initial_pos = embd[0].clone();
        apply_repulsive_force(&mut embd, i, j, a, b, lr);

        assert!(
            embd[0][0] > initial_pos[0],
            "Point moved from {} to {}",
            initial_pos[0],
            embd[0][0]
        );
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

        optimise_embedding(&mut embd, &graph, &params, 42);

        // Check that points moved (due to negative sampling at minimum)
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

        optimise_embedding(&mut embd, &graph, &params, 42);

        // With no edges, only negative sampling occurs, so embedding should change
        // but we can't predict exact values
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
        };

        optimise_embedding(&mut embd1, &graph, &params, 42);
        optimise_embedding(&mut embd2, &graph, &params, 42);

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
        };

        optimise_embedding(&mut embd, &graph, &params, 42);

        let final_dist = squared_dist(&embd[0], &embd[1]).sqrt();

        // Distance should decrease
        assert!(final_dist < initial_dist);
    }
}
