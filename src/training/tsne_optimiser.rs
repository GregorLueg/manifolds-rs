//! Optimisers for tSNE fitting. Contains the BarnesHut version from Laurens
//! van der Maaten and the FFT-accelerated Interpolation-based version of tSNE
//! from Linderman et al.

use num_traits::{Float, FromPrimitive};
use rayon::prelude::*;
use thousands::*;

use crate::data::structures::*;
use crate::prelude::*;
use crate::utils::bh_tree::*;

#[cfg(feature = "fft_tsne")]
use crate::utils::fft::*;

//////////
// tSNE //
//////////

/////////////
// Globals //
/////////////

/// Iteration from when on to switch the tSNE momentum
const TSNE_MOMENTUM_SWITCH_ITER: usize = 250;

/// Initial tSNE momentum
const TSNE_INITIAL_MOMENTUM: f64 = 0.5;

/// Final tSNE momentum
const TSNE_FINAL_MOMENTUM: f64 = 0.8;

/// Minimum tSNE gain
const TSNE_MIN_GAIN: f64 = 0.01;

/// tSNE epsilon
const TSNE_EPS: f64 = 1e-12;

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

/// Cap on `n_boxes` per dimension in the FFT grid. FFT cost per epoch is
/// O(n_boxes^2 log n_boxes); without a cap, n_boxes grows with the embedding
/// span and per-epoch cost blows up. Box width adapts upward once this cap
/// binds, keeping the grid covering the embedding.
#[cfg(feature = "fft_tsne")]
const TSNE_FFT_MAX_BOXES: usize = 140;

/// Lower bound on the FFT box width (the original fixed value).
#[cfg(feature = "fft_tsne")]
const TSNE_FFT_MIN_BOX_WIDTH: f64 = 1.0;

/// Headroom added to the grid bounds once the box-cap regime is active, so
/// the embedding can move between rebuilds.
#[cfg(feature = "fft_tsne")]
const TSNE_FFT_GRID_MARGIN: f64 = 0.15;

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
}

impl<T> TsneOptimParams<T>
where
    T: Float + FromPrimitive,
{
    /// Generate a new instance
    pub fn new(
        n_epochs: usize,
        lr: Option<T>,
        early_exag_iter: usize,
        early_exag_factor: T,
        late_exag_factor: Option<T>,
        theta: T,
        n_interp_points: Option<usize>,
    ) -> Self {
        let n_interp_points = n_interp_points.unwrap_or(3);

        Self {
            n_epochs,
            lr,
            early_exag_iter,
            early_exag_factor,
            late_exag_factor,
            theta,
            n_interp_points,
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
        }
    }
}

///////////////
// Optimiser //
///////////////

/// Type of optimisation to use for tSNE.
#[derive(Default)]
pub enum TsneOpt {
    #[default]
    /// FFT-accelerated version
    Fft,
    /// BarnesHut-accelerated version
    BarnesHut,
}

/// Parse the tSNE Optimiser to use
pub fn parse_tsne_optimiser(s: &str) -> Option<TsneOpt> {
    match s.to_lowercase().as_str() {
        "barnes hut" | "bh" => Some(TsneOpt::BarnesHut),
        "fft" => Some(TsneOpt::Fft),
        _ => None,
    }
}

/////////////
// Helpers //
/////////////

/// Adaptive gain update for t-SNE gradient descent (van der Maaten convention:
/// gain += 0.2 when grad and update disagree, gain *= 0.8 otherwise; floored
/// at `min_gain`).
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

/// Clip a 2D momentum step in place to `max_step_norm` and rewrite the point.
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
#[inline]
fn step_cap_from_lr<T: ManifoldsFloat>(lr: T) -> T {
    let lr_f64 = lr.to_f64().unwrap();
    T::from_f64((lr_f64 * TSNE_MAX_STEP_FRACTION).max(TSNE_MAX_STEP_FLOOR)).unwrap()
}

/// Recentre embedding on the origin. Sum is accumulated in f64 so that f32
/// storage does not lose precision at large N.
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

////////////////
// Barnes Hut //
////////////////

/// Optimise 2D embedding using Barnes-Hut t-SNE.
///
/// Minimises KL divergence between high-dimensional affinities (graph) and
/// low-dimensional Student-t similarities using gradient descent with momentum
/// and adaptive gains. The adjacency list, gradient buffers and
/// position-snapshot buffers are allocated once outside the epoch loop. Z is
/// accumulated in f64 to avoid precision loss at large N.
pub fn optimise_bh_tsne<T>(
    embd: &mut [Vec<T>],
    params: &TsneOptimParams<T>,
    graph: &CoordinateList<T>,
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
    let mut xs = vec![T::zero(); n];
    let mut ys = vec![T::zero(); n];
    let mut rep_forces: Vec<(T, T, T)> = vec![(T::zero(), T::zero(), T::zero()); n];

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
            params.get_late_exag_factor()
        };

        // snapshot positions in parallel into pre-allocated buffers.
        embd.par_iter()
            .zip(xs.par_iter_mut())
            .zip(ys.par_iter_mut())
            .for_each(|((p, x), y)| {
                *x = p[0];
                *y = p[1];
            });

        // compute all repulsive forces in one parallel pass, writing into
        // the preallocated rep_forces buffer instead of collecting a new Vec.
        rep_forces.par_iter_mut().enumerate().for_each_init(
            || Vec::with_capacity(256),
            |stack, (i, slot)| {
                *slot = bh_tree.compute_repulsive_force(i, xs[i], ys[i], params.theta, stack);
            },
        );

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

        // attractive forces (exact) + parameter update + step clip.
        embd.par_iter_mut()
            .zip(update_flat.par_chunks_mut(n_dim))
            .zip(gains_flat.par_chunks_mut(n_dim))
            .enumerate()
            .for_each(|(i, ((point, u_i), g_i))| {
                let px = xs[i];
                let py = ys[i];

                let (rep_x, rep_y, _) = rep_forces[i];

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

/// Adaptive grid sizing for the FFT path.
///
/// Returns `(n_boxes, box_width, grid_half)`. Below the cap, `box_width` stays
/// at the minimum and `n_boxes` follows the embedding span (rounded to the
/// FFT-friendly sizes in `choose_grid_size`). Above the cap, `n_boxes` stays
/// at `TSNE_FFT_MAX_BOXES` and `box_width` grows so the grid still covers the
/// embedding with margin.
#[cfg(feature = "fft_tsne")]
fn fft_grid_geometry(half_span: f64, min_intervals: usize) -> (usize, f64, f64) {
    let span = 2.0 * half_span * 1.05;
    let unconstrained = choose_grid_size(0.0, span, TSNE_FFT_MIN_BOX_WIDTH, min_intervals);

    if unconstrained <= TSNE_FFT_MAX_BOXES {
        let half = unconstrained as f64 * TSNE_FFT_MIN_BOX_WIDTH / 2.0;
        (unconstrained, TSNE_FFT_MIN_BOX_WIDTH, half)
    } else {
        let grown_half = half_span * (1.05 + TSNE_FFT_GRID_MARGIN);
        let bw = (grown_half * 2.0 / TSNE_FFT_MAX_BOXES as f64).max(TSNE_FFT_MIN_BOX_WIDTH);
        let half = TSNE_FFT_MAX_BOXES as f64 * bw / 2.0;
        (TSNE_FFT_MAX_BOXES, bw, half)
    }
}

/// Optimise 2D embedding using FFT-accelerated t-SNE.
///
/// ### Notes
///
/// - Adjacency list, charges, potentials and the position snapshot (`xs`,
///   `ys`) are allocated once outside the epoch loop.
/// - `n_boxes` is capped at `TSNE_FFT_MAX_BOXES`; once the cap binds,
///   `box_width` grows with the embedding and the grid carries
///   `TSNE_FFT_GRID_MARGIN` headroom so rebuilds happen on growth, not every
///   epoch.
/// - The FFT workspace persists across rebuilds whenever `n_boxes` (and hence
///   `n_fft`) is unchanged.
/// - Z and the repulsive forces are reconstructed in f64 to avoid catastrophic
///   cancellation at large N.
#[cfg(feature = "fft_tsne")]
pub fn optimise_fft_tsne<T>(
    embd: &mut [Vec<T>],
    params: &TsneOptimParams<T>,
    graph: &CoordinateList<T>,
    verbose: usize,
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

    let n_terms = 4;

    let initial_momentum = T::from_f64(TSNE_INITIAL_MOMENTUM).unwrap();
    let final_momentum = T::from_f64(TSNE_FINAL_MOMENTUM).unwrap();
    let min_gain = T::from_f64(TSNE_MIN_GAIN).unwrap();
    let max_step_norm = step_cap_from_lr(lr);

    let mut uy = vec![vec![T::zero(); n_dim]; n];
    let mut gains = vec![vec![T::one(); n_dim]; n];

    // adjacency list built once.
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

    // pre-allocated FFT-side buffers and position snapshot.
    let mut charges = vec![T::zero(); n * n_terms];
    let mut potentials = vec![T::zero(); n * n_terms];
    let mut xs = vec![T::zero(); n];
    let mut ys = vec![T::zero(); n];

    let min_intervals = 50;
    let mut cached_n_boxes: usize = 0;
    let mut grid: Option<FftGrid<T>> = None;
    let mut workspace: Option<FftWorkspace<T>> = None;

    for epoch in 0..params.n_epochs {
        // Snapshot positions in parallel.
        embd.par_iter()
            .zip(xs.par_iter_mut())
            .zip(ys.par_iter_mut())
            .for_each(|((p, x), y)| {
                *x = p[0];
                *y = p[1];
            });

        // Min/max scan; serial scan over 2N values is cheap relative to
        // the per-epoch FFT work.
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

        // Rebuild policy: n_boxes changed (regime crossing or growth in the
        // sub-cap regime), or the cached grid no longer comfortably contains
        // the embedding (cap regime, span exceeded the margin).
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
            let new_grid = FftGrid::new(-half, half, n_boxes, params.n_interp_points);
            if cached_n_boxes != n_boxes {
                workspace = Some(FftWorkspace::new(new_grid.n_fft));
            }
            grid = Some(new_grid);
            cached_n_boxes = n_boxes;
        }

        let grid_ref = grid.as_ref().unwrap();
        let ws = workspace.as_mut().unwrap();

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

        // Fill charges.
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

        // Zero potentials and run the FFT-accelerated convolution.
        for v in potentials.iter_mut() {
            *v = T::zero();
        }
        n_body_fft_2d(&xs, &ys, &charges, n_terms, grid_ref, ws, &mut potentials);

        // Z in f64; subtract n to remove the diagonal q_ii = 1 contribution.
        let sum_q: f64 = (0..n)
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
            - n as f64;

        let sum_q_safe = if sum_q > TSNE_EPS { sum_q } else { 1.0 };

        embd.par_iter_mut()
            .zip(uy.par_iter_mut())
            .zip(gains.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((point, u_i), gains_i))| {
                let x = xs[i];
                let y = ys[i];

                // Attractive forces (exact via sparse graph).
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

                // Repulsive forces reconstructed in f64.
                let pot_idx = i * n_terms;
                let phi1 = potentials[pot_idx].to_f64().unwrap();
                let phi2 = potentials[pot_idx + 1].to_f64().unwrap();
                let phi3 = potentials[pot_idx + 2].to_f64().unwrap();

                let xf = x.to_f64().unwrap();
                let yf = y.to_f64().unwrap();

                let rep_x = T::from_f64((xf * phi1 - phi2) / sum_q_safe).unwrap();
                let rep_y = T::from_f64((yf * phi1 - phi3) / sum_q_safe).unwrap();

                // No factor of 4 here; lr is implicitly calibrated against
                // this gradient definition.
                let grad_x = attr_x - rep_x;
                let grad_y = attr_y - rep_y;

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

        optimise_bh_tsne(&mut embd, &params, &graph, 0);

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

        let _ = optimise_fft_tsne(&mut embd, &params, &graph, 0);

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

        optimise_bh_tsne(&mut embd1, &params, &graph, 0);
        optimise_bh_tsne(&mut embd2, &params, &graph, 0);

        for (p1, p2) in embd1.iter().zip(embd2.iter()) {
            assert_relative_eq!(p1[0], p2[0]);
            assert_relative_eq!(p1[1], p2[1]);
        }
    }
}
