//! Optimiser for PaCMAP fitting.

use num_traits::{Float, FromPrimitive};
use rayon::prelude::*;
use std::ops::{AddAssign, SubAssign};

use crate::prelude::*;
use crate::training::*;

/////////////
// Globals //
/////////////

/// End of first phase during fitting
const PHASE1_END: usize = 100;
/// End of second phase during fitting
const PHASE2_END: usize = 200;
/// End of third (and last) phase of fitting
const PHASE3_END: usize = 450;
/// Weight for the nearest points in first phase (dominating by mid-near)
const W_NB_PHASE1: f64 = 1.0;
/// Weight for nearest points during the second phase (focussing on near pairs)
const W_NB_PHASE2: f64 = 3.0;
/// Weight for nearest points during the third phase (local structure refinement)
const W_NB_PHASE3: f64 = 1.0;
/// Starting point of the medium-near points that decreases then during second phase
const W_MN_PHASE1: f64 = 1000.0;
/// Weight for the far-away points (kept constant during training)
const W_FP: f64 = 1.0;

/// Type of PaCMAP optimiser to use
#[derive(Default)]
pub enum PacMapOptimiser {
    /// Parallel version of Adam
    #[default]
    AdamParallel,
    /// Adam
    Adam,
}

/// Parse the PacMap Optimiser to use
///
/// ### Params
///
/// * `s` - String defining the optimiser. Choice of `"adam"` or
///   `"adam_parallel"`
///
/// ### Return
///
/// Option of Optimiser
pub fn parse_pacmap_optimiser(s: &str) -> Option<PacMapOptimiser> {
    match s.to_lowercase().as_str() {
        "adam" => Some(PacMapOptimiser::Adam),
        "adam_parallel" => Some(PacMapOptimiser::AdamParallel),
        _ => None,
    }
}

////////////
// Params //
////////////

/// PaCMAP optimisation parameters
#[derive(Clone, Debug)]
pub struct PacmapOptimParams<T> {
    /// Number of optimisation epochs. Paper uses 450.
    pub n_epochs: usize,
    /// Initial learning rate
    pub lr: T,
    /// Adam beta1
    pub beta1: T,
    /// Adam beta2
    pub beta2: T,
    /// Adam epsilon
    pub eps: T,
    /// End of phase 1 (mid-near dominant). Default 100.
    pub phase1_end: usize,
    /// End of phase 2 (mid-near weight decays to 0). Default 200.
    pub phase2_end: usize,
}

impl<T> PacmapOptimParams<T>
where
    T: Float + FromPrimitive,
{
    /// Create a new `PacmapOptimParams` instance.
    ///
    /// All parameters are optional and fall back to the paper's defaults.
    ///
    /// ### Params
    ///
    /// * `n_epochs` - Total number of optimisation epochs across all three
    ///   phases. Default `450` (100 + 100 + 250).
    /// * `lr` - Adam learning rate. Default `0.01`. PaCMAP is less sensitive to
    ///   this than UMAP since all pairs are processed every epoch.
    /// * `beta1` - Adam first moment decay. Default `0.9` (standard Adam).
    ///   Unlike UMAP's uwot implementation which uses `0.5`, PaCMAP processes
    ///   all pairs every epoch so standard momentum is appropriate.
    /// * `beta2` - Adam second moment decay. Default `0.999` (standard Adam).
    /// * `eps` - Adam numerical stability constant. Default `1e-7`.
    /// * `phase1_end` - Epoch at which phase 1 ends and phase 2 begins. During
    ///   phase 1 mid-near weight is dominant (`w_mn=1000`), anchoring global
    ///   structure. Default `100`.
    /// * `phase2_end` - Epoch at which phase 2 ends and phase 3 begins. During
    ///   phase 2 the mid-near weight decays linearly to zero and near pair
    ///   weight.
    ///   increases (`w_nb=3`). Default `200`.
    ///
    /// ### Returns
    ///
    /// `PacmapOptimParams` with all values set to provided arguments or their
    /// defaults.
    pub fn new(
        n_epochs: Option<usize>,
        lr: Option<T>,
        beta1: Option<T>,
        beta2: Option<T>,
        eps: Option<T>,
        phase1_end: Option<usize>,
        phase2_end: Option<usize>,
    ) -> Self {
        Self {
            n_epochs: n_epochs.unwrap_or(PHASE3_END),
            lr: lr.unwrap_or(T::from_f64(0.01).unwrap()),
            beta1: beta1.unwrap_or(T::from_f64(BETA1).unwrap()),
            beta2: beta2.unwrap_or(T::from_f64(BETA2).unwrap()),
            eps: eps.unwrap_or(T::from_f64(EPS).unwrap()),
            phase1_end: phase1_end.unwrap_or(PHASE1_END),
            phase2_end: phase2_end.unwrap_or(PHASE2_END),
        }
    }
}

/// Implementation of the defaults here
impl<T> Default for PacmapOptimParams<T>
where
    T: Float + FromPrimitive,
{
    fn default() -> Self {
        Self::new(None, None, None, None, None, None, None)
    }
}

/////////////
// Helpers //
/////////////

/// Returns `(w_near, w_mid_near, w_further)` for a given epoch.
///
/// Implements the three-phase weight schedule from the PaCMAP paper:
///
/// | Phase | Epochs                    | w_near | w_mid_near      | w_further |
/// |-------|---------------------------|--------|-----------------|-----------|
/// | 1     | `0..phase1_end`           | 1.0    | 1000.0          | 1.0       |
/// | 2     | `phase1_end..phase2_end`  | 3.0    | 1000.0 → 0.0    | 1.0       |
/// | 3     | `phase2_end..`            | 1.0    | 0.0             | 1.0       |
///
/// In phase 2 `w_mid_near` decays linearly from `1000.0` to `0.0`, handing
/// off global structure responsibility to the near pairs.
///
/// ### Params
///
/// * `epoch` - Current epoch index (zero-based).
/// * `phase1_end` - Epoch at which phase 1 transitions to phase 2.
/// * `phase2_end` - Epoch at which phase 2 transitions to phase 3.
///
/// ### Returns
///
/// Tuple of `(w_near, w_mid_near, w_further)` for the given epoch.
#[inline]
fn phase_weights<T>(epoch: usize, phase1_end: usize, phase2_end: usize) -> (T, T, T)
where
    T: Float + FromPrimitive,
{
    if epoch < phase1_end {
        (
            T::from_f64(W_NB_PHASE1).unwrap(),
            T::from_f64(W_MN_PHASE1).unwrap(),
            T::from_f64(W_FP).unwrap(),
        )
    } else if epoch < phase2_end {
        let progress = (epoch - phase1_end) as f64 / (phase2_end - phase1_end) as f64;
        let w_mn = T::from_f64(W_MN_PHASE1 * (1.0 - progress)).unwrap();
        (
            T::from_f64(W_NB_PHASE2).unwrap(),
            w_mn,
            T::from_f64(W_FP).unwrap(),
        )
    } else {
        (
            T::from_f64(W_NB_PHASE3).unwrap(),
            T::zero(),
            T::from_f64(W_FP).unwrap(),
        )
    }
}

/// Attractive loss gradient coefficient for near and mid-near pairs.
///
/// Loss: d / (d + c)  where c=10 for near, c=10000 for mid-near
/// dLoss/dd = c / (d + c)^2
/// Gradient w.r.t. embedding coord: (dLoss/dd) * (y_i - y_j) / d  [chain rule]
///
/// ### Params
///
/// * `dist_sq` - Squared Euclidean distance between the pair in embedding
///   space.
/// * `c` - Loss scaling constant. Controls how quickly the attractive force
///   saturates with distance. Near pairs use `c=10`, mid-near use `c=10000`.
///
/// ### Returns
///
/// Returns the scalar coefficient; caller multiplies by (y_i - y_j).
#[inline(always)]
fn attract_grad_coeff<T>(dist_sq: T, c: T) -> T
where
    T: Float + FromPrimitive,
{
    let two = T::from_f64(2.0).unwrap();
    let d = (dist_sq + T::from_f64(1e-10).unwrap()).sqrt();
    let denom = (d + c) * (d + c) * d;
    c / denom * two
}

/// Repulsive loss gradient coefficient for further pairs.
///
/// Loss: 1 / (1 + d)
/// dLoss/dd = -1 / (1+d)^2
///
/// ### Params
///
/// * `dist_sq` - Squared Euclidean distance between the pair in embedding
///   space.
///
/// ### Returns
///
/// Returns the scalar coefficient (positive; caller negates for repulsion).
#[inline(always)]
fn repel_grad_coeff<T>(dist_sq: T) -> T
where
    T: Float + FromPrimitive,
{
    let two = T::from_f64(2.0).unwrap();
    let d = (dist_sq + T::from_f64(1e-10).unwrap()).sqrt();
    let denom = (T::one() + d) * (T::one() + d) * d;
    T::one() / denom * two
}

//////////
// Main //
//////////

/// Optimise a PaCMAP embedding using Adam.
///
/// ### Params
///
/// * `embd` - Embedding, shape [n_samples][n_dim]. Modified in place.
/// * `pairs` - The three pair sets from `construct_pacmap_pairs`.
/// * `params` - Optimisation parameters.
/// * `verbose` - Progress reporting.
pub fn optimise_pacmap<T>(
    embd: &mut [Vec<T>],
    pairs: &PacmapPairs,
    params: &PacmapOptimParams<T>,
    verbose: bool,
) where
    T: ManifoldsFloat,
{
    let n = embd.len();
    if n == 0 {
        return;
    }
    let n_dim = embd[0].len();

    // Flatten embedding for cache locality
    let mut embd_flat: Vec<T> = embd.iter().flatten().copied().collect();

    // Adam moments
    let mut m = vec![T::zero(); n * n_dim];
    let mut v = vec![T::zero(); n * n_dim];

    let one_minus_b1 = T::one() - params.beta1;
    let one_minus_b2 = T::one() - params.beta2;

    let mut beta1t = params.beta1;
    let mut beta2t = params.beta2;

    // Near pair c constant
    let c_near = T::from_f64(10.0).unwrap();
    // Mid-near pair c constant
    let c_mn = T::from_f64(10_000.0).unwrap();

    for epoch in 0..params.n_epochs {
        let (w_nb, w_mn, w_fp) = phase_weights::<T>(epoch, params.phase1_end, params.phase2_end);

        // Bias-corrected Adam scale
        let sqrt_b2t1 = (T::one() - beta2t).sqrt();
        let ad_scale = params.lr * sqrt_b2t1 / (T::one() - beta1t);
        let epsc = sqrt_b2t1 * params.eps;

        // Accumulate per-node gradients
        let mut grads = vec![T::zero(); n * n_dim];

        // Near pairs — attractive
        for &(i, j) in &pairs.near {
            let base_i = i * n_dim;
            let base_j = j * n_dim;

            let mut dist_sq = T::zero();
            for d in 0..n_dim {
                let diff = embd_flat[base_i + d] - embd_flat[base_j + d];
                dist_sq += diff * diff;
            }

            let coeff = w_nb * attract_grad_coeff(dist_sq, c_near);

            for d in 0..n_dim {
                let delta = embd_flat[base_i + d] - embd_flat[base_j + d];
                grads[base_i + d] -= coeff * delta;
                grads[base_j + d] += coeff * delta;
            }
        }

        // Mid-near pairs — attractive (skipped in phase 3 when w_mn == 0)
        if w_mn > T::zero() {
            for &(i, j) in &pairs.mid_near {
                let base_i = i * n_dim;
                let base_j = j * n_dim;

                let mut dist_sq = T::zero();
                for d in 0..n_dim {
                    let diff = embd_flat[base_i + d] - embd_flat[base_j + d];
                    dist_sq += diff * diff;
                }

                let coeff = w_mn * attract_grad_coeff(dist_sq, c_mn);

                for d in 0..n_dim {
                    let delta = embd_flat[base_i + d] - embd_flat[base_j + d];
                    grads[base_i + d] -= coeff * delta;
                    grads[base_j + d] += coeff * delta;
                }
            }
        }

        // Further pairs — repulsive
        for &(i, j) in &pairs.further {
            let base_i = i * n_dim;
            let base_j = j * n_dim;

            let mut dist_sq = T::zero();
            for d in 0..n_dim {
                let diff = embd_flat[base_i + d] - embd_flat[base_j + d];
                dist_sq += diff * diff;
            }

            let coeff = w_fp * repel_grad_coeff(dist_sq);

            for d in 0..n_dim {
                let delta = embd_flat[base_i + d] - embd_flat[base_j + d];
                // repulsive: push i away from j
                grads[base_i + d] += coeff * delta;
                grads[base_j + d] -= coeff * delta;
            }
        }

        // Adam update
        for idx in 0..(n * n_dim) {
            let g = grads[idx];
            let m_old = m[idx];
            let v_old = v[idx];
            m[idx] = m_old + one_minus_b1 * (g - m_old);
            v[idx] = v_old + one_minus_b2 * (g * g - v_old);
            embd_flat[idx] += ad_scale * m[idx] / (v[idx].sqrt() + epsc);
        }

        beta1t *= params.beta1;
        beta2t *= params.beta2;

        if verbose && ((epoch + 1) % 50 == 0 || epoch + 1 == params.n_epochs) {
            println!(" Completed epoch {}/{}", epoch + 1, params.n_epochs);
        }
    }

    // write back
    for (i, point) in embd.iter_mut().enumerate() {
        let base = i * n_dim;
        point.copy_from_slice(&embd_flat[base..base + n_dim]);
    }
}

/// Optimise a PaCMAP embedding using Adam (parallel).
///
/// ### Params
///
/// * `embd` - Embedding, shape [n_samples][n_dim]. Modified in place.
/// * `pairs` - The three pair sets from `construct_pacmap_pairs`.
/// * `params` - Optimisation parameters.
/// * `verbose` - Progress reporting.
pub fn optimise_pacmap_parallel<T>(
    embd: &mut [Vec<T>],
    pairs: &PacmapPairs,
    params: &PacmapOptimParams<T>,
    verbose: bool,
) where
    T: Float + FromPrimitive + AddAssign + Send + Sync + SubAssign,
{
    let n = embd.len();
    if n == 0 {
        return;
    }
    let n_dim = embd[0].len();

    // flatten embedding for cache locality
    let mut embd_flat: Vec<T> = embd.iter().flatten().copied().collect();

    // Adam moments
    let mut m = vec![T::zero(); n * n_dim];
    let mut v = vec![T::zero(); n * n_dim];

    let one_minus_b1 = T::one() - params.beta1;
    let one_minus_b2 = T::one() - params.beta2;

    let mut beta1t = params.beta1;
    let mut beta2t = params.beta2;

    // near pair c constant
    let c_near = T::from_f64(10.0).unwrap();
    // mid-near pair c constant
    let c_mn = T::from_f64(10_000.0).unwrap();

    // collect everything into nodes to be able to accumulate gradients
    // in parallel
    let mut node_near: Vec<Vec<usize>> = vec![vec![]; n];
    let mut node_mn: Vec<Vec<usize>> = vec![vec![]; n];
    let mut node_fp: Vec<Vec<usize>> = vec![vec![]; n];

    for &(i, j) in &pairs.near {
        node_near[i].push(j);
        node_near[j].push(i);
    }
    for &(i, j) in &pairs.mid_near {
        node_mn[i].push(j);
        node_mn[j].push(i);
    }
    for &(i, j) in &pairs.further {
        node_fp[i].push(j);
        node_fp[j].push(i);
    }

    for epoch in 0..params.n_epochs {
        let (w_nb, w_mn, w_fp) = phase_weights::<T>(epoch, params.phase1_end, params.phase2_end);

        // bias-corrected Adam scale
        let sqrt_b2t1 = (T::one() - beta2t).sqrt();
        let ad_scale = params.lr * sqrt_b2t1 / (T::one() - beta1t);
        let epsc = sqrt_b2t1 * params.eps;

        let grads: Vec<T> = (0..n)
            .into_par_iter()
            .flat_map(|i| {
                let base_i = i * n_dim;
                let mut node_grad = vec![T::zero(); n_dim];

                for &j in &node_near[i] {
                    let base_j = j * n_dim;
                    let mut dist_sq = T::zero();
                    for d in 0..n_dim {
                        let diff = embd_flat[base_i + d] - embd_flat[base_j + d];
                        dist_sq += diff * diff;
                    }
                    let coeff = w_nb * attract_grad_coeff(dist_sq, c_near);
                    for d in 0..n_dim {
                        node_grad[d] -= coeff * (embd_flat[base_i + d] - embd_flat[base_j + d]);
                    }
                }

                if w_mn > T::zero() {
                    for &j in &node_mn[i] {
                        let base_j = j * n_dim;
                        let mut dist_sq = T::zero();
                        for d in 0..n_dim {
                            let diff = embd_flat[base_i + d] - embd_flat[base_j + d];
                            dist_sq += diff * diff;
                        }
                        let coeff = w_mn * attract_grad_coeff(dist_sq, c_mn);
                        for d in 0..n_dim {
                            node_grad[d] -= coeff * (embd_flat[base_i + d] - embd_flat[base_j + d]);
                        }
                    }
                }

                for &j in &node_fp[i] {
                    let base_j = j * n_dim;
                    let mut dist_sq = T::zero();
                    for d in 0..n_dim {
                        let diff = embd_flat[base_i + d] - embd_flat[base_j + d];
                        dist_sq += diff * diff;
                    }
                    let coeff = w_fp * repel_grad_coeff(dist_sq);
                    for d in 0..n_dim {
                        node_grad[d] += coeff * (embd_flat[base_i + d] - embd_flat[base_j + d]);
                    }
                }

                node_grad
            })
            .collect();

        // adam update
        for idx in 0..(n * n_dim) {
            let g = grads[idx];
            let m_old = m[idx];
            let v_old = v[idx];
            m[idx] = m_old + one_minus_b1 * (g - m_old);
            v[idx] = v_old + one_minus_b2 * (g * g - v_old);
            embd_flat[idx] += ad_scale * m[idx] / (v[idx].sqrt() + epsc);
        }

        beta1t = beta1t * params.beta1;
        beta2t = beta2t * params.beta2;

        if verbose && ((epoch + 1) % 50 == 0 || epoch + 1 == params.n_epochs) {
            println!(" Completed epoch {}/{}", epoch + 1, params.n_epochs);
        }
    }

    // write back
    for (i, point) in embd.iter_mut().enumerate() {
        let base = i * n_dim;
        point.copy_from_slice(&embd_flat[base..base + n_dim]);
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_pacmap_optimiser {
    use super::*;
    use crate::data::pacmap_pairs::PacmapPairs;

    fn simple_pairs(n: usize) -> PacmapPairs {
        let near = (0..n).map(|i| (i, (i + 1) % n)).collect();
        let mid_near = (0..n).map(|i| (i, (i + 2) % n)).collect();
        let further = (0..n).map(|i| (i, (i + n / 2) % n)).collect();
        PacmapPairs {
            near,
            mid_near,
            further,
        }
    }

    fn simple_embd(n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|i| vec![i as f64 * 2.0, 0.0]).collect()
    }

    fn default_params() -> PacmapOptimParams<f64> {
        PacmapOptimParams::new(Some(50), None, None, None, None, None, None)
    }

    fn total_movement(before: &[Vec<f64>], after: &[Vec<f64>]) -> f64 {
        before
            .iter()
            .zip(after.iter())
            .flat_map(|(b, a)| b.iter().zip(a.iter()).map(|(&x, &y)| (x - y).abs()))
            .sum()
    }

    #[test]
    fn test_sequential_moves_points() {
        let pairs = simple_pairs(6);
        let mut embd = simple_embd(6);
        let initial = embd.clone();
        optimise_pacmap(&mut embd, &pairs, &default_params(), false);
        assert!(total_movement(&initial, &embd) > 0.01);
    }

    #[test]
    fn test_parallel_moves_points() {
        let pairs = simple_pairs(6);
        let mut embd = simple_embd(6);
        let initial = embd.clone();
        optimise_pacmap_parallel(&mut embd, &pairs, &default_params(), false);
        assert!(total_movement(&initial, &embd) > 0.01);
    }

    #[test]
    fn test_sequential_all_finite() {
        let pairs = simple_pairs(8);
        let mut embd = simple_embd(8);
        optimise_pacmap(&mut embd, &pairs, &default_params(), false);
        for point in &embd {
            for &coord in point {
                assert!(coord.is_finite(), "non-finite coordinate: {}", coord);
            }
        }
    }

    #[test]
    fn test_parallel_all_finite() {
        let pairs = simple_pairs(8);
        let mut embd = simple_embd(8);
        optimise_pacmap_parallel(&mut embd, &pairs, &default_params(), false);
        for point in &embd {
            for &coord in point {
                assert!(coord.is_finite(), "non-finite coordinate: {}", coord);
            }
        }
    }

    #[test]
    fn test_sequential_reproducible() {
        let pairs = simple_pairs(6);
        let params = default_params();
        let mut embd1 = simple_embd(6);
        let mut embd2 = simple_embd(6);
        optimise_pacmap(&mut embd1, &pairs, &params, false);
        optimise_pacmap(&mut embd2, &pairs, &params, false);
        assert_eq!(embd1, embd2);
    }

    #[test]
    fn test_parallel_reproducible() {
        let pairs = simple_pairs(6);
        let params = default_params();
        let mut embd1 = simple_embd(6);
        let mut embd2 = simple_embd(6);
        optimise_pacmap_parallel(&mut embd1, &pairs, &params, false);
        optimise_pacmap_parallel(&mut embd2, &pairs, &params, false);
        assert_eq!(embd1, embd2);
    }

    #[test]
    fn test_empty_embedding_does_not_panic() {
        let pairs = PacmapPairs {
            near: vec![],
            mid_near: vec![],
            further: vec![],
        };
        let mut embd: Vec<Vec<f64>> = vec![];
        optimise_pacmap(&mut embd, &pairs, &default_params(), false);
        optimise_pacmap_parallel(&mut embd, &pairs, &default_params(), false);
    }

    #[test]
    fn test_phase_weights_respected() {
        // run for only phase 1 epochs and check mid-near pairs drive movement
        // by comparing against a run with zero mid-near pairs
        let n = 10;
        let pairs_full = simple_pairs(n);
        let pairs_no_mn = PacmapPairs {
            near: pairs_full.near.clone(),
            mid_near: vec![],
            further: pairs_full.further.clone(),
        };

        let params = PacmapOptimParams::new(Some(50), None, None, None, None, None, None);

        let mut embd_full = simple_embd(n);
        let mut embd_no_mn = simple_embd(n);
        let initial = simple_embd(n);

        optimise_pacmap(&mut embd_full, &pairs_full, &params, false);
        optimise_pacmap(&mut embd_no_mn, &pairs_no_mn, &params, false);

        let movement_full = total_movement(&initial, &embd_full);
        let movement_no_mn = total_movement(&initial, &embd_no_mn);

        // mid-near pairs contribute additional force so movement should differ
        assert!(
            (movement_full - movement_no_mn).abs() > 1e-6,
            "mid-near pairs had no effect: full={:.4}, no_mn={:.4}",
            movement_full,
            movement_no_mn
        );
    }

    #[test]
    fn test_sequential_and_parallel_broadly_agree() {
        // both should converge to similar embeddings from the same start,
        // not necessarily identical due to gradient accumulation order
        let pairs = simple_pairs(10);
        let params = PacmapOptimParams::new(Some(200), None, None, None, None, None, None);

        let mut embd_seq = simple_embd(10);
        let mut embd_par = simple_embd(10);

        optimise_pacmap(&mut embd_seq, &pairs, &params, false);
        optimise_pacmap_parallel(&mut embd_par, &pairs, &params, false);

        let diff = total_movement(&embd_seq, &embd_par);
        let scale = total_movement(&simple_embd(10), &embd_seq);

        assert!(
            diff < scale * 0.1,
            "sequential and parallel diverged too much: diff={:.4}, scale={:.4}",
            diff,
            scale
        );
    }
}
