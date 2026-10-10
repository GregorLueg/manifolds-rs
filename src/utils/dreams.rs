//! DREAMS: t-SNE regularised towards a global reference embedding (PCA by
//! default), giving a local-global spectrum controlled by `lambda`.
//!
//! The update follows the reference openTSNE fork, not Equation 2 of the
//! paper. Per epoch, with `alpha = ||Y||_F / ||Y_e||_F` held constant:
//!
//! `y += (1 - lambda) * u - lr * lambda^2 * (2 / n) * (y - alpha * y_e)`
//!
//! where `u` is the usual (momentum, gains, step-capped) t-SNE update. The
//! regulariser gets neither gains nor momentum nor the step cap, and `lambda`
//! enters it twice. The published default of `lambda = 0.15` was tuned on this
//! form, so it is kept as is.
//!
//! The regulariser's `lr` is openTSNE's automatic `n / exaggeration`, not the
//! t-SNE learning rate in use. Our t-SNE runs at `n / 12` throughout, so with
//! the shared `lr` the pull was 12x weaker after early exaggeration and
//! `lambda = 0.5` behaved like the reference's 0.15 (Tasic, n = 23,822).
//!
//! ### References
//!
//! Kury, Kobak & Damrich, Transactions on Machine Learning Research, 2026.

use faer::MatRef;
use rand_distr::{Distribution, StandardNormal};

use crate::data::init::pca_scores;
use crate::errors::ManifoldsError;
use crate::utils::traits::*;

/////////////
// Globals //
/////////////

/// Default DREAMS regularisation strength, the value recommended across all
/// datasets in the paper.
pub const DREAMS_LAMBDA: f64 = 0.15;

/// Standard deviation of the first reference dimension in the initial
/// embedding. Matches the paper's scripts (openTSNE's PCA init scale). Only
/// the first dimension is used for scaling, so the component ratio survives.
const DREAMS_INIT_STD: f64 = 1e-4;

/////////////////
// DreamsState //
/////////////////

/// Constant state of the DREAMS regulariser.
#[derive(Clone, Debug)]
pub struct DreamsState<T> {
    /// Centred reference embedding, interleaved `[x0, y0, x1, y1, ...]`
    reference: Vec<T>,
    /// Frobenius norm of the centred reference, in `f64`
    ref_norm: f64,
    /// Regularisation strength in `[0, 1]`
    lambda: T,
}

/// Epoch-constant scalars of the DREAMS update.
pub(crate) struct DreamsEpoch<T> {
    /// Scale of the reference matched to the current embedding
    alpha: T,
    /// `1 - lambda`, applied to the t-SNE update
    keep: T,
    /// `lr * lambda^2 * 2 / n` with `lr = n / exaggeration`, the regulariser
    /// step coefficient
    coef: T,
}

impl<T> DreamsState<T>
where
    T: ManifoldsFloat,
{
    /// Build the regulariser state from a reference embedding.
    ///
    /// ### Params
    ///
    /// * `lambda` - Regularisation strength in `[0, 1]`
    /// * `reference` - Reference embedding as `[n_samples][2]`; centred here
    ///
    /// ### Returns
    ///
    /// The state, or an error if `lambda` is out of range or the reference
    /// collapses to a point.
    pub fn new(lambda: T, reference: &[Vec<T>]) -> Result<Self, ManifoldsError> {
        let lambda_f64 = lambda.to_f64().unwrap_or(f64::NAN);
        if !(0.0..=1.0).contains(&lambda_f64) {
            return Err(ManifoldsError::DreamsInvalidLambda { lambda: lambda_f64 });
        }

        let centred = centre_rows(reference);
        let ref_norm = centred
            .iter()
            .map(|v| {
                let v = v.to_f64().unwrap();
                v * v
            })
            .sum::<f64>()
            .sqrt();
        if ref_norm <= f64::EPSILON {
            return Err(ManifoldsError::DreamsDegenerateReference);
        }

        Ok(Self {
            reference: centred,
            ref_norm,
            lambda,
        })
    }

    /// Epoch constants from the current positions.
    ///
    /// `||Y||_F` is accumulated sequentially in `f64`, so the result does not
    /// depend on the thread count.
    ///
    /// ### Params
    ///
    /// * `pos` - Interleaved positions `[x0, y0, x1, y1, ...]`, before this
    ///   epoch's update
    /// * `exag_factor` - Exaggeration factor of this epoch; sets the
    ///   regulariser's learning rate `n / exag_factor`
    ///
    /// ### Returns
    ///
    /// The scalars used by [`DreamsState::apply_step`].
    pub(crate) fn epoch_consts(&self, pos: &[T], exag_factor: T) -> DreamsEpoch<T> {
        let y_norm = pos
            .iter()
            .map(|v| {
                let v = v.to_f64().unwrap();
                v * v
            })
            .sum::<f64>()
            .sqrt();
        let lambda = self.lambda.to_f64().unwrap();

        DreamsEpoch {
            alpha: T::from_f64(y_norm / self.ref_norm).unwrap(),
            keep: T::one() - self.lambda,
            // lr * 2 / n with lr = n / exag
            coef: T::from_f64(2.0 * lambda * lambda / exag_factor.to_f64().unwrap()).unwrap(),
        }
    }

    /// Rewrite a point with the DREAMS update.
    ///
    /// Call after the t-SNE update and step clip, so `u` is the clipped
    /// momentum step. The momentum buffer itself stays unscaled, as in the
    /// reference.
    ///
    /// ### Params
    ///
    /// * `i` - Point index
    /// * `point` - Position of length 2, overwritten
    /// * `u` - Clipped t-SNE update of length 2
    /// * `prev_x` - x-coordinate before this epoch's update
    /// * `prev_y` - y-coordinate before this epoch's update
    /// * `ep` - Epoch constants
    #[inline(always)]
    pub(crate) fn apply_step(
        &self,
        i: usize,
        point: &mut [T],
        u: &[T],
        prev_x: T,
        prev_y: T,
        ep: &DreamsEpoch<T>,
    ) {
        let rx = prev_x - ep.alpha * self.reference[2 * i];
        let ry = prev_y - ep.alpha * self.reference[2 * i + 1];
        point[0] = prev_x + ep.keep * u[0] - ep.coef * rx;
        point[1] = prev_y + ep.keep * u[1] - ep.coef * ry;
    }
}

/////////////
// Helpers //
/////////////

/// Centre 2D rows on the origin and interleave them.
///
/// ### Params
///
/// * `rows` - Points as `[n_samples][2]`
///
/// ### Returns
///
/// Centred, interleaved coordinates `[x0, y0, x1, y1, ...]`.
fn centre_rows<T: ManifoldsFloat>(rows: &[Vec<T>]) -> Vec<T> {
    let n = rows.len().max(1) as f64;
    let (sx, sy) = rows.iter().fold((0.0_f64, 0.0_f64), |(sx, sy), r| {
        (sx + r[0].to_f64().unwrap(), sy + r[1].to_f64().unwrap())
    });
    let (mx, my) = (T::from_f64(sx / n).unwrap(), T::from_f64(sy / n).unwrap());

    rows.iter().flat_map(|r| [r[0] - mx, r[1] - my]).collect()
}

/// Build the DREAMS reference, state and initial embedding.
///
/// Without a supplied reference, the first two PC scores are used unscaled.
/// The initial embedding is the centred reference scaled so its first
/// dimension has standard deviation `DREAMS_INIT_STD`, as in the paper.
///
/// ### Params
///
/// * `data` - Input data (samples × features), used for the PCA reference
/// * `reference` - Optional reference embedding as `[2][n_samples]` (the
///   layout every embedding in this crate returns), e.g. an MDS or PHATE
///   result
/// * `lambda` - Regularisation strength in `[0, 1]`
/// * `randomised` - Use randomised SVD for the PCA reference
/// * `seed` - Random seed for the randomised SVD
///
/// ### Returns
///
/// `(initial_embedding, state)`, the embedding as `[n_samples][2]`.
pub fn dreams_setup<T>(
    data: MatRef<T>,
    reference: Option<Vec<Vec<T>>>,
    lambda: T,
    randomised: bool,
    seed: u64,
) -> Result<(Vec<Vec<T>>, DreamsState<T>), ManifoldsError>
where
    T: ManifoldsFloat,
    StandardNormal: Distribution<T>,
{
    let n_samples = data.nrows();

    let rows = match reference {
        Some(r) => {
            let n_ref = r.first().map_or(0, |d| d.len());
            if r.len() != 2 || r.iter().any(|d| d.len() != n_samples) {
                return Err(ManifoldsError::DreamsReferenceMismatch {
                    n_samples,
                    n_dim: r.len(),
                    n_ref,
                });
            }
            (0..n_samples).map(|i| vec![r[0][i], r[1][i]]).collect()
        }
        None => pca_scores(data, 2, randomised, seed)?,
    };

    let state = DreamsState::new(lambda, &rows)?;

    // the state holds the centred reference; scale it by the std of dim 0
    let var_x = state
        .reference
        .iter()
        .step_by(2)
        .map(|v| {
            let v = v.to_f64().unwrap();
            v * v
        })
        .sum::<f64>()
        / n_samples as f64;
    let scale = if var_x > 0.0 {
        DREAMS_INIT_STD / var_x.sqrt()
    } else {
        // the reference has spread (checked above), just none along dim 0
        DREAMS_INIT_STD / (state.ref_norm / (n_samples as f64).sqrt())
    };
    let scale = T::from_f64(scale).unwrap();

    let init = state
        .reference
        .chunks_exact(2)
        .map(|p| vec![p[0] * scale, p[1] * scale])
        .collect();

    Ok((init, state))
}
