use core::f64;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::iter::Sum;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use thousands::*;

use crate::data::structures::*;
use crate::utils::bh_tree::*;

#[cfg(feature = "fft_tsne")]
use crate::utils::fft::*;

//////////
// tSNE //
//////////

/////////////
// Globals //
/////////////

const TSNE_MOMENTUM_SWITCH_ITER: usize = 250;
const TSNE_INITIAL_MOMENTUM: f64 = 0.5;
const TSNE_FINAL_MOMENTUM: f64 = 0.8;
const TSNE_MIN_GAIN: f64 = 0.01;
const TSNE_EPS: f64 = 1e-12;

////////////////
// Structures //
////////////////

/// t-SNE specific optimization parameters
///
/// ### Fields
///
/// * `n_epochs` - Number of epochs (typically n / 12 or 200 or so)
/// * `lr` - Learning rate
/// * `early_exag_iter` - Early exaggeration iters
/// * `early_exag_factor` - The factor to exaggerate in the early iterations
/// * `theta` - The Barnes-Hut theta; relevant if you use the
///   `optimise_bh_tsne()`
/// * `n_interp_points` - Interpolation points per box (typically 3); relevant
///   if you use `optimise_fft_tsne()`
#[derive(Clone, Debug)]
pub struct TsneOptimParams<T> {
    pub n_epochs: usize,
    pub lr: T,
    pub early_exag_iter: usize,
    pub early_exag_factor: T,
    pub theta: T,
    pub n_interp_points: usize,
}

impl<T> TsneOptimParams<T>
where
    T: Float + FromPrimitive,
{
    /// Generate a new instance
    ///
    /// ### Params
    ///
    /// * `n_epochs` - Number of epochs (typically n / 12 or 200 or so)
    /// * `lr` - Learning rate
    /// * `early_exag_iter` - Early exaggeration iters
    /// * `early_exag_factor` - The factor to exaggerate in the early iterations
    /// * `theta` - The Barnes-Hut theta
    ///
    /// ### Returns
    ///
    /// Initialised self
    pub fn new(
        n_epochs: usize,
        lr: T,
        early_exag_iter: usize,
        early_exag_factor: T,
        theta: T,
        n_interp_points: Option<usize>,
    ) -> Self {
        let n_interp_points = n_interp_points.unwrap_or(3);

        Self {
            n_epochs,
            lr,
            early_exag_iter,
            early_exag_factor,
            theta,
            n_interp_points,
        }
    }
}

impl<T: Float + FromPrimitive> Default for TsneOptimParams<T> {
    fn default() -> Self {
        Self {
            n_epochs: 1000,
            lr: T::from_f64(200.0).unwrap(),
            early_exag_iter: 250,
            early_exag_factor: T::from_f64(12.0).unwrap(),
            theta: T::from_f64(0.5).unwrap(),
            n_interp_points: 3,
        }
    }
}

///////////////
// Optimiser //
///////////////

#[derive(Default)]
pub enum TsneOpt {
    #[default]
    /// FFT-accelerated version
    Fft,
    /// BarnesHut-accelerated version
    BarnesHut,
}

/// Parse the tSNE Optimiser to use
///
/// ### Params
///
/// * `s` - String defining the optimiser. Choice of `"barnes hut" | "bh"` or
///   `"fft"`.
///
/// ### Return
///
/// Option of Optimiser
pub fn parse_tsne_optimiser(s: &str) -> Option<TsneOpt> {
    match s.to_lowercase().as_str() {
        "barnes hut" | "bh" => Some(TsneOpt::BarnesHut),
        "fft" => Some(TsneOpt::Fft),
        _ => None,
    }
}

////////////////
// Barnes Hut //
////////////////

/// Adaptive gain update for t-SNE gradient descent
///
/// Implements per-parameter adaptive learning rates: gains increase when
/// gradient maintains direction, decrease when oscillating.
///
/// ### Params
///
/// * `val` - Current parameter value to update
/// * `update` - Accumulated momentum vector for this parameter
/// * `gain` - Adaptive gain (learning rate multiplier) for this parameter
/// * `grad` - Current gradient for this parameter
/// * `lr` - Base learning rate
/// * `momentum` - Momentum coefficient (typically 0.5 early, 0.8 later)
/// * `min_gain` - Minimum allowed gain value (typically 0.01)
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
    T: Float + FromPrimitive + ToPrimitive,
{
    // adjust gain based on gradient-update alignment
    if (grad > T::zero()) != (*update > T::zero()) {
        *gain = *gain + T::from_f64(0.2).unwrap();
    } else {
        *gain = *gain * T::from_f64(0.8).unwrap();
    }
    *gain = (*gain).max(min_gain);

    // momentum update with adaptive gain
    *update = momentum * *update - lr * *gain * grad;
    *val = *val + *update;
}

/// Optimise 2D embedding using Barnes-Hut t-SNE
///
/// Minimises KL divergence between high-dimensional affinities (graph) and
/// low-dimensional Student-t similarities using gradient descent with momentum
/// and adaptive gains.
///
/// ### Params
///
/// * `embd` - Mutable 2D embedding to optimise in-place
/// * `params` - Optimisation parameters (learning rate, epochs, etc.)
/// * `graph` - Symmetric sparse graph of high-dimensional affinities P_ij
/// * `verbose` - Print progress every 50 epochs
///
/// ### Notes
///
/// For each epoch:
/// 1. Build Barnes-Hut tree from current embedding
/// 2. Compute gradients: ∇C = 4 * (F_attractive - F_repulsive / Z)
///    - F_attractive: exact via sparse graph
///    - F_repulsive: approximated via Barnes-Hut tree
/// 3. Update positions with momentum and adaptive gains
/// 4. Centre embedding every 100 epochs to prevent drift
///
/// Uses early exaggeration (first 250 iterations) and momentum switching.
pub fn optimise_bh_tsne<T>(
    embd: &mut [Vec<T>],
    params: &TsneOptimParams<T>,
    graph: &CoordinateList<T>,
    verbose: bool,
) where
    T: Float + FromPrimitive + Send + Sync + AddAssign + SubAssign + MulAssign + DivAssign + Sum,
{
    let n = embd.len();
    let n_dim = embd[0].len();

    // main paramters
    let initial_momentum = T::from_f64(TSNE_INITIAL_MOMENTUM).unwrap();
    let final_momentum = T::from_f64(TSNE_FINAL_MOMENTUM).unwrap();
    let min_gain = T::from_f64(TSNE_MIN_GAIN).unwrap();
    let eps = T::from_f64(TSNE_EPS).unwrap();

    let mut update_flat = vec![T::zero(); n * n_dim];
    let mut gains_flat = vec![T::one(); n * n_dim];

    // Build adjacency list once (graph is fixed)
    let mut adj: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
    for ((&i, &j), &w) in graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
    {
        adj[i].push((j, w));
    }

    for epoch in 0..params.n_epochs {
        let bh_tree = BarnesHutTree::new(embd);

        let momentum = if epoch < TSNE_MOMENTUM_SWITCH_ITER {
            initial_momentum
        } else {
            final_momentum
        };
        let exag_factor = if epoch < params.early_exag_iter {
            params.early_exag_factor
        } else {
            T::one()
        };

        // Compute forces for all points in parallel
        let results: Vec<(T, T, T, T, T)> = embd
            .par_iter()
            .enumerate()
            .map(|(i, p)| {
                let px = p[0];
                let py = p[1];

                // Repulsive forces (Barnes-Hut approximation)
                let (rep_x, rep_y, partial_z) =
                    bh_tree.compute_repulsive_force(i, px, py, params.theta);

                // Attractive forces (exact via graph)
                let mut attr_x = T::zero();
                let mut attr_y = T::zero();
                if let Some(neighbors) = adj.get(i) {
                    for &(j, p_val) in neighbors {
                        let other = &embd[j];
                        let dx = px - other[0];
                        let dy = py - other[1];
                        let dist_sq = dx * dx + dy * dy;
                        let q = T::one() / (T::one() + dist_sq);
                        let force = p_val * exag_factor * q;
                        attr_x += force * dx;
                        attr_y += force * dy;
                    }
                }
                (attr_x, attr_y, rep_x, rep_y, partial_z)
            })
            .collect();

        // Global normalisation constant Z
        let z_total: T = results
            .iter()
            .map(|t| t.4)
            .fold(T::zero(), |acc, x| acc + x);
        let z_inv = if z_total > eps {
            T::one() / z_total
        } else {
            T::zero()
        };

        // Apply gradient updates (sequential is fine - negligible runtime)
        for i in 0..n {
            let (attr_x, attr_y, rep_x, rep_y, _) = results[i];
            let grad_x = attr_x - rep_x * z_inv;
            let grad_y = attr_y - rep_y * z_inv;

            update_parameter(
                &mut embd[i][0],
                &mut update_flat[i * 2],
                &mut gains_flat[i * 2],
                grad_x,
                params.lr,
                momentum,
                min_gain,
            );

            update_parameter(
                &mut embd[i][1],
                &mut update_flat[i * 2 + 1],
                &mut gains_flat[i * 2 + 1],
                grad_y,
                params.lr,
                momentum,
                min_gain,
            );
        }

        // renormalise to avoid drift (also made parallel)
        let (sum_x, sum_y) = embd
            .iter() // Serial iter
            .fold((T::zero(), T::zero()), |(ax, ay), p| (ax + p[0], ay + p[1]));

        let mean_x = sum_x / T::from_usize(n).unwrap();
        let mean_y = sum_y / T::from_usize(n).unwrap();

        // Parallel subtract is fine (no reduction involved)
        embd.par_iter_mut().for_each(|p| {
            p[0] -= mean_x;
            p[1] -= mean_y;
        });

        if verbose && (epoch % 50 == 0 || epoch == params.n_epochs - 1) {
            println!(
                "Completed Epoch {} out of {} | Z = {}",
                epoch,
                params.n_epochs,
                z_total.to_f32().unwrap().separate_with_underscores()
            );
        }
    }
}

/////////
// FTT //
/////////

/// Optimise 2D embedding using FFT-accelerated t-SNE
///
/// ### Notes
///
/// Refactored to match the C++ "FIt-SNE" implementation logic.
///
/// - Re-calculates grid bounds every iteration for optimal resolution.
/// - Computes Attractive and Repulsive forces in parallel.
/// - Applies momentum and gains in the same parallel pass.
///
/// ### Params
///
/// * `embd` - Mutable 2D embedding to optimise in-place
/// * `params` - Optimisation parameters (learning rate, epochs, etc.)
/// * `graph` - Symmetric sparse graph of high-dimensional affinities P_ij
/// * `verbose` - Print progress every 50 epochs
#[cfg(feature = "fft_tsne")]
pub fn optimise_fft_tsne<T>(
    embd: &mut [Vec<T>],
    params: &TsneOptimParams<T>,
    graph: &CoordinateList<T>,
    verbose: bool,
) where
    T: FftwFloat + AddAssign + SubAssign + MulAssign + DivAssign + Sum + ToPrimitive + Send + Sync,
{
    let n = embd.len();
    let n_dim = embd[0].len();
    assert_eq!(n_dim, 2, "FFT t-SNE only supports 2D output");

    let n_terms = 4;

    // constants matches C++ defaults
    let initial_momentum = T::from_f64(TSNE_INITIAL_MOMENTUM).unwrap();
    let final_momentum = T::from_f64(TSNE_FINAL_MOMENTUM).unwrap();
    let min_gain = T::from_f64(TSNE_MIN_GAIN).unwrap();

    // momentum / update buffer
    let mut uy = vec![vec![T::zero(); n_dim]; n];
    // adaptive gains
    let mut gains = vec![vec![T::one(); n_dim]; n];

    // 1. Convert CoordinateList to Adjacency List for efficient parallel row access
    // This assumes the graph is symmetric.
    let mut adj: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
    for ((&i, &j), &w) in graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
    {
        if i < n {
            adj[i].push((j, w));
        }
    }

    // temp buffers for FFT input to avoid repeated allocation overhead if
    // possible
    let mut charges = vec![T::zero(); n * n_terms];

    for epoch in 0..params.n_epochs {
        // step 1: Dynamic Grid Setup (Matches C++ computeFftGradient logic)

        // extract X and Y for FFT and bounds calculation. we do this copy to
        // satisfy the borrow checker (who doesn't like to satisfy...) and data
        // layout for n_body_fft_2d
        let (xs, ys): (Vec<T>, Vec<T>) = embd.iter().map(|p| (p[0], p[1])).unzip();

        let mut min_val = xs[0];
        let mut max_val = xs[0];
        for v in xs.iter().chain(&ys) {
            if *v < min_val {
                min_val = *v;
            }
            if *v > max_val {
                max_val = *v;
            }
        }

        // determine grid size based on spread
        let n_boxes = choose_grid_size(
            min_val.to_f64().unwrap(),
            max_val.to_f64().unwrap(),
            1.0,
            50,
        );

        // create grid
        let grid = FftGrid::new(min_val, max_val, n_boxes, params.n_interp_points);

        // step 2: FFT potentials (repulsive term pre-calc) - with rayon!
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

        // compute potentials using the refactored function
        let potentials = n_body_fft_2d(&xs, &ys, &charges, n_terms, &grid);

        // compute norm Z (Sum Q)
        // C++: sum_Q += (1 + x^2 + y^2)*phi1 - 2*(x*phi2 + y*phi3) + phi4
        let sum_q: T = (0..n)
            .map(|i| {
                let idx = i * n_terms;
                let phi1 = potentials[idx];
                let phi2 = potentials[idx + 1];
                let phi3 = potentials[idx + 2];
                let phi4 = potentials[idx + 3];
                let x = xs[i];
                let y = ys[i];
                (T::one() + x * x + y * y) * phi1 - (T::one() + T::one()) * (x * phi2 + y * phi3)
                    + phi4
            })
            .sum::<T>()
            - T::from_usize(n).unwrap();

        // step 3: compute forces & updates
        // matches C++ PARALLEL_FOR logic for attractive forces + gradient update

        let momentum = if epoch < TSNE_MOMENTUM_SWITCH_ITER {
            initial_momentum
        } else {
            final_momentum
        };
        let exag_factor = if epoch < params.early_exag_iter {
            params.early_exag_factor
        } else {
            T::one()
        };
        let learning_rate = params.lr;

        // C++ uses max_step_norm to prevent explosions
        let max_step_norm = T::from_f64(5.0).unwrap();

        // update embd in place, but need to read xs/ys (which are copies
        // of old embd) uy and gains are updated in place. bit of magic
        // with rayon.
        embd.par_iter_mut()
            .zip(uy.par_iter_mut())
            .zip(gains.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((point, u_i), gains_i))| {
                let x = xs[i];
                let y = ys[i];

                // attractive forces
                // F_attr = Sum_j P_ij * Q_ij * (y_i - y_j)
                // where Q_ij = 1 / (1 + dist^2)
                let mut attr_x = T::zero();
                let mut attr_y = T::zero();

                for &(j, p_val) in &adj[i] {
                    let other_x = xs[j];
                    let other_y = ys[j];
                    let dx = x - other_x;
                    let dy = y - other_y;
                    let dist_sq = dx * dx + dy * dy;
                    let q_ij = T::one() / (T::one() + dist_sq);

                    // apply exaggeration here
                    let force = p_val * exag_factor * q_ij;

                    attr_x += force * dx;
                    attr_y += force * dy;
                }

                // repulsive forces (approximated via FFT)
                // F_rep_x = (x * phi1 - phi2) / Z
                // F_rep_y = (y * phi1 - phi3) / Z
                let pot_idx = i * n_terms;
                let phi1 = potentials[pot_idx];
                let phi2 = potentials[pot_idx + 1];
                let phi3 = potentials[pot_idx + 2];

                let rep_x = (x * phi1 - phi2) / sum_q;
                let rep_y = (y * phi1 - phi3) / sum_q;

                let grad_x = (attr_x - rep_x) * T::from_f64(4.0).unwrap();
                let grad_y = (attr_y - rep_y) * T::from_f64(4.0).unwrap();

                update_parameter(
                    &mut point[0],
                    &mut u_i[0],
                    &mut gains_i[0],
                    grad_x,
                    learning_rate,
                    momentum,
                    min_gain,
                );

                update_parameter(
                    &mut point[1],
                    &mut u_i[1],
                    &mut gains_i[1],
                    grad_y,
                    learning_rate,
                    momentum,
                    min_gain,
                );

                // clipping logic (in the C++ code)
                let step_sq = u_i[0] * u_i[0] + u_i[1] * u_i[1];
                let max_sq = max_step_norm * max_step_norm;

                if step_sq > max_sq {
                    let scale = max_step_norm / step_sq.sqrt();
                    u_i[0] *= scale;
                    u_i[1] *= scale;

                    point[0] = xs[i] + u_i[0];
                    point[1] = ys[i] + u_i[1];
                }
            });

        // step 4 - recentring
        let (sum_x, sum_y) = embd
            .iter() // Serial iter
            .fold((T::zero(), T::zero()), |(ax, ay), p| (ax + p[0], ay + p[1]));

        let mean_x = sum_x / T::from_usize(n).unwrap();
        let mean_y = sum_y / T::from_usize(n).unwrap();

        // Parallel subtract is fine (no reduction involved)
        embd.par_iter_mut().for_each(|p| {
            p[0] -= mean_x;
            p[1] -= mean_y;
        });

        let sum_q_f64 = sum_q.to_f64().unwrap();

        if verbose && (epoch % 50 == 0 || epoch == params.n_epochs - 1) {
            println!(
                "Epoch {}/{} | Z = {}",
                epoch,
                params.n_epochs,
                sum_q_f64.separate_with_underscores()
            );
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_tsne_optimiser {
    use super::*;
    use approx::assert_relative_eq;

    //////////
    // tSNE //
    //////////

    // Helper to create a symmetric COO graph for t-SNE tests
    fn create_coo_graph(n: usize, edges: &[(usize, usize, f64)]) -> CoordinateList<f64> {
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        // t-SNE usually expects a symmetric P matrix (or graph)
        // We ensure symmetry here manually for the test cases
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
        assert_relative_eq!(params.lr, 200.0);
        assert_eq!(params.early_exag_iter, 250);
        assert_relative_eq!(params.early_exag_factor, 12.0);
        assert_relative_eq!(params.theta, 0.5);
    }

    #[test]
    fn test_bh_tsne_basic_convergence() {
        // Simple triangle graph
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        let graph = create_coo_graph(3, &edges);

        // Initializing points in a line
        let mut embd = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0], // middle
            vec![2.0, 2.0],
        ];
        let initial_embd = embd.clone();

        // Run for a short burst
        let params = TsneOptimParams {
            n_epochs: 50, // Short run
            lr: 50.0,     // Aggressive LR to ensure movement
            ..TsneOptimParams::default()
        };

        optimise_bh_tsne(&mut embd, &params, &graph, false);

        // Check for NaNs
        for point in &embd {
            for val in point {
                assert!(val.is_finite(), "Embedding contains non-finite values");
            }
        }

        // Check for movement
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
    #[cfg(feature = "fft_tsne")]
    fn test_fft_tsne_basic_convergence() {
        // Same setup as BH, ensuring FFT path works
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        let graph = create_coo_graph(3, &edges);

        let mut embd = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let initial_embd = embd.clone();

        let params = TsneOptimParams {
            n_epochs: 50,
            lr: 50.0,
            n_interp_points: 3, // specific to FFT
            ..TsneOptimParams::default()
        };

        optimise_fft_tsne(&mut embd, &params, &graph, false);

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

        optimise_bh_tsne(&mut embd1, &params, &graph, false);
        optimise_bh_tsne(&mut embd2, &params, &graph, false);

        for (p1, p2) in embd1.iter().zip(embd2.iter()) {
            assert_relative_eq!(p1[0], p2[0]);
            assert_relative_eq!(p1[1], p2[1]);
        }
    }
}
