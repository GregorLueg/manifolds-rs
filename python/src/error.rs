//! Mapping [`ManifoldsError`] onto Python exceptions.
//!
//! No new error enum: the library already has one, and this crate adds no
//! failure modes of its own beyond what pyo3 raises directly. `ManifoldsError`
//! and `PyErr` are both foreign here, so the `From` impl needs a local newtype.

use manifolds_rs::prelude::ManifoldsError;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;

////////////////
// Exceptions //
////////////////

create_exception!(
    _manifolds,
    ManifoldsRsError,
    PyException,
    "Base class for every error raised by the manifolds Rust core."
);

create_exception!(
    _manifolds,
    ConvergenceError,
    ManifoldsRsError,
    "A spectral decomposition did not converge, or produced fewer eigenpairs than were asked for."
);

/////////////
// Newtype //
/////////////

/// Carries a [`ManifoldsError`] to the FFI boundary.
///
/// A tuple struct, which means it doubles as the conversion function:
/// `.map_err(ManErr)?` in any function returning [`PyResult`].
pub(crate) struct ManErr(
    /// The error the library raised.
    pub ManifoldsError,
);

impl From<ManifoldsError> for ManErr {
    fn from(e: ManifoldsError) -> Self {
        Self(e)
    }
}

impl From<ManErr> for PyErr {
    fn from(e: ManErr) -> PyErr {
        let msg = e.0.to_string();
        match e.0 {
            // Fixable by changing an argument, which is what `ValueError` means
            // to a Python caller. `AnnSearchRsError` is here because every
            // variant reachable from this crate (dimension mismatch,
            // unsupported metric, too few samples for the centroid count) is
            // caller-fixable too; the file-format variants sit behind a feature
            // manifolds-rs does not enable.
            ManifoldsError::PerplexityTooLarge { .. }
            | ManifoldsError::IncorrectDim { .. }
            | ManifoldsError::PowerMustBePositive { .. }
            | ManifoldsError::NoData
            | ManifoldsError::NoGraphEdges
            | ManifoldsError::NotEnoughNeighbours { .. }
            | ManifoldsError::DegenerateLocalRadii
            | ManifoldsError::AnnSearchRsError(_) => PyValueError::new_err(msg),

            // Numerical routines that ran but did not get there. Usually
            // degenerate data rather than a bad argument, so they get their own
            // class: a caller can retry with different parameters, or not.
            ManifoldsError::FaerSvdError
            | ManifoldsError::FaerEigenError
            | ManifoldsError::InsufficientEigenpairs { .. } => ConvergenceError::new_err(msg),

            // Shape and structure invariants of the internal sparse pipeline,
            // the GPU device limits, and the parametric model payload. A caller
            // cannot hand any of these in directly, so reaching one means the
            // pipeline built something wrong.
            _ => ManifoldsRsError::new_err(msg),
        }
    }
}
