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

/// Default implementation for TsneOptimParams
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

/// Optimise 2D embedding using Barnes-Hut t-SNE.
///
/// Minimises KL divergence between high-dimensional affinities (graph) and
/// low-dimensional Student-t similarities using gradient descent with momentum
/// and adaptive gains.
///
/// ### Notes
///
/// For each epoch:
/// 1. Build Barnes-Hut tree from current embedding
/// 2. Compute gradient: attractive - repulsive / Z
///    - Attractive: exact via sparse graph
///    - Repulsive: approximated via Barnes-Hut tree
/// 3. Update positions with momentum and adaptive gains
/// 4. Centre embedding to prevent drift
///
/// Force computation is split into two parallel passes: first the repulsive
/// forces (which must be summed to get Z), then the attractive forces fused
/// with the parameter update. The adjacency list and update/gains buffers
/// are allocated once outside the epoch loop.
///
/// ### Params
///
/// * `embd` - Mutable 2D embedding to optimise in-place
/// * `params` - Optimisation parameters (learning rate, epochs, etc.)
/// * `graph` - Symmetric sparse graph of high-dimensional affinities P_ij
/// * `verbose` - Print progress every 50 epochs
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

    let initial_momentum = T::from_f64(TSNE_INITIAL_MOMENTUM).unwrap();
    let final_momentum = T::from_f64(TSNE_FINAL_MOMENTUM).unwrap();
    let min_gain = T::from_f64(TSNE_MIN_GAIN).unwrap();
    let eps = T::from_f64(TSNE_EPS).unwrap();

    // --- pre-allocate update and gains buffers ---
    let mut update_flat = vec![T::zero(); n * n_dim];
    let mut gains_flat = vec![T::one(); n * n_dim];

    // --- build adjacency list ONCE ---
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

        // step 1: compute repulsive forces and partial Z in parallel
        // snapshot positions for gradient computation (cannot read embd
        // while updating it in the same pass)
        let xs: Vec<T> = embd.iter().map(|p| p[0]).collect();
        let ys: Vec<T> = embd.iter().map(|p| p[1]).collect();

        // compute all repulsive forces in one parallel pass, storing
        // (rep_x, rep_y, partial_z) per point
        let rep_forces: Vec<(T, T, T)> = (0..n)
            .into_par_iter()
            .map(|i| bh_tree.compute_repulsive_force(i, xs[i], ys[i], params.theta))
            .collect();

        // global normalisation constant
        let z_total: T = rep_forces.iter().map(|r| r.2).fold(T::zero(), |a, b| a + b);
        let z_inv = if z_total > eps {
            T::one() / z_total
        } else {
            T::zero()
        };

        // step 2: compute attractive forces and update in parallel
        embd.par_iter_mut()
            .zip(update_flat.par_chunks_mut(n_dim))
            .zip(gains_flat.par_chunks_mut(n_dim))
            .enumerate()
            .for_each(|(i, ((point, u_i), g_i))| {
                let px = xs[i];
                let py = ys[i];

                let (rep_x, rep_y, _) = rep_forces[i];

                // attractive forces (exact via graph)
                let mut attr_x = T::zero();
                let mut attr_y = T::zero();
                for &(j, p_val) in &adj[i] {
                    let dx = px - xs[j];
                    let dy = py - ys[j];
                    let dist_sq = dx * dx + dy * dy;
                    let q = T::one() / (T::one() + dist_sq);
                    let force = p_val * exag_factor * q;
                    attr_x += force * dx;
                    attr_y += force * dy;
                }

                let grad_x = attr_x - rep_x * z_inv;
                let grad_y = attr_y - rep_y * z_inv;

                update_parameter(
                    &mut point[0],
                    &mut u_i[0],
                    &mut g_i[0],
                    grad_x,
                    params.lr,
                    momentum,
                    min_gain,
                );
                update_parameter(
                    &mut point[1],
                    &mut u_i[1],
                    &mut g_i[1],
                    grad_y,
                    params.lr,
                    momentum,
                    min_gain,
                );
            });

        // step 3: recentring
        let (sum_x, sum_y) = embd
            .iter()
            .fold((T::zero(), T::zero()), |(ax, ay), p| (ax + p[0], ay + p[1]));

        let mean_x = sum_x / T::from_usize(n).unwrap();
        let mean_y = sum_y / T::from_usize(n).unwrap();

        embd.par_iter_mut().for_each(|p| {
            p[0] -= mean_x;
            p[1] -= mean_y;
        });

        if verbose && (epoch % 50 == 0 || epoch == params.n_epochs - 1) {
            println!(
                "Epoch {}/{} | Z = {}",
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

/// Optimise 2D embedding using FFT-accelerated t-SNE.
///
/// ### Notes
///
/// Refactored to match the C++ FIt-SNE implementation logic.
///
/// - Re-calculates grid bounds every iteration for optimal resolution.
/// - Caches the FFT workspace (plans and aligned buffers) across iterations;
///   rebuilds only when the grid dimension changes.
/// - Computes attractive and repulsive forces in parallel.
/// - Applies momentum and gains in the same parallel pass.
/// - Pre-allocates charges, potentials, and adjacency list outside the
///   epoch loop to avoid per-iteration allocation overhead.
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

    let initial_momentum = T::from_f64(TSNE_INITIAL_MOMENTUM).unwrap();
    let final_momentum = T::from_f64(TSNE_FINAL_MOMENTUM).unwrap();
    let min_gain = T::from_f64(TSNE_MIN_GAIN).unwrap();

    let mut uy = vec![vec![T::zero(); n_dim]; n];
    let mut gains = vec![vec![T::one(); n_dim]; n];

    // build adjacency list ONCE... makes it faster...
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

    // pre-allocation
    let mut charges = vec![T::zero(); n * n_terms];
    let mut potentials = vec![T::zero(); n * n_terms];

    // cached grid and workspace
    let mut cached_n_boxes: usize = 0;
    // for pre-allocation - needs to be allowed
    #[allow(unused_assignments)]
    let mut grid: Option<FftGrid<T>> = None;
    let mut workspace: Option<FftWorkspace<T>> = None;

    for epoch in 0..params.n_epochs {
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

        let n_boxes = choose_grid_size(
            min_val.to_f64().unwrap(),
            max_val.to_f64().unwrap(),
            1.0,
            50,
        );

        // only rebuild grid and workspace when grid size changes
        if n_boxes != cached_n_boxes {
            let g = FftGrid::new(min_val, max_val, n_boxes, params.n_interp_points);
            workspace = Some(FftWorkspace::new(g.n_fft));
            grid = Some(g);
            cached_n_boxes = n_boxes;
        } else {
            // grid size unchanged, but bounds shifted — rebuild grid, reuse workspace
            // (kernel depends on relative spacings which are determined by box_width,
            // and box_width = (max-min)/n_boxes, so if n_boxes is the same but
            // min/max changed, we still need a new grid + kernel)
            let g = FftGrid::new(min_val, max_val, n_boxes, params.n_interp_points);
            if g.n_fft != workspace.as_ref().unwrap().n_fft {
                workspace = Some(FftWorkspace::new(g.n_fft));
            }
            grid = Some(g);
        }

        let grid_ref = grid.as_ref().unwrap();
        let ws = workspace.as_mut().unwrap();

        // fill charges
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

        // zero and compute potentials (buffer reused across epochs)
        for v in potentials.iter_mut() {
            *v = T::zero();
        }
        n_body_fft_2d(&xs, &ys, &charges, n_terms, grid_ref, ws, &mut potentials);

        // compute Z
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
        let max_step_norm = T::from_f64(5.0).unwrap();

        embd.par_iter_mut()
            .zip(uy.par_iter_mut())
            .zip(gains.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((point, u_i), gains_i))| {
                let x = xs[i];
                let y = ys[i];

                // attractive forces
                let mut attr_x = T::zero();
                let mut attr_y = T::zero();

                for &(j, p_val) in &adj[i] {
                    let other_x = xs[j];
                    let other_y = ys[j];
                    let dx = x - other_x;
                    let dy = y - other_y;
                    let dist_sq = dx * dx + dy * dy;
                    let q_ij = T::one() / (T::one() + dist_sq);
                    let force = p_val * exag_factor * q_ij;
                    attr_x += force * dx;
                    attr_y += force * dy;
                }

                // repulsive forces (FFT-approximated)
                let pot_idx = i * n_terms;
                let phi1 = potentials[pot_idx];
                let phi2 = potentials[pot_idx + 1];
                let phi3 = potentials[pot_idx + 2];

                let rep_x = (x * phi1 - phi2) / sum_q;
                let rep_y = (y * phi1 - phi3) / sum_q;

                // no factor of 4 — matches C++
                let grad_x = attr_x - rep_x;
                let grad_y = attr_y - rep_y;

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

        // recentring
        let (sum_x, sum_y) = embd
            .iter()
            .fold((T::zero(), T::zero()), |(ax, ay), p| (ax + p[0], ay + p[1]));

        let mean_x = sum_x / T::from_usize(n).unwrap();
        let mean_y = sum_y / T::from_usize(n).unwrap();

        embd.par_iter_mut().for_each(|p| {
            p[0] -= mean_x;
            p[1] -= mean_y;
        });

        if verbose && (epoch % 50 == 0 || epoch == params.n_epochs - 1) {
            let sum_q_f64 = sum_q.to_f64().unwrap();
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
