//! Flexible matrix inputs.
//!
//! Orientation is fixed throughout the crate: rows are samples, columns are
//! features. [`ManifoldsMatrix`] is the single conversion point that lets every
//! entry point take a faer matrix, an ndarray 2-D array or a pre-flattened
//! row-major buffer without any of the algorithms caring which.
//!
//! The conversion borrows wherever the source can be viewed as a matrix, which
//! is every layout except a non-contiguous ndarray. That matters: the pipeline
//! below these entry points already flattens to row-major inside
//! `ann-search-rs`, so materialising an owned buffer here would add a second
//! full copy of the data to the path that existed before this trait did.
//!
//! Taking `&self` rather than `self` is what keeps the owned cases free: the
//! entry point owns its argument for the whole call, so a `Mat<T>` or a
//! standard-layout `Array2<T>` can simply be borrowed from.

use faer::{Mat, MatRef};

#[cfg(feature = "ndarray")]
use crate::utils::traits::ManifoldsFloat;
#[cfg(feature = "ndarray")]
use ndarray::{Array2, ArrayView2};

//////////////
// MatInput //
//////////////

/// Borrowed-or-owned matrix produced by [`ManifoldsMatrix::to_mat_input`].
///
/// Owning the awkward case keeps every other one free. Bind the result, then
/// call [`MatInput::as_mat_ref`] for the `MatRef` the algorithms want.
pub enum MatInput<'a, T> {
    /// The source could be viewed directly as a samples x features matrix.
    Borrowed(MatRef<'a, T>),
    /// The source layout needed materialising, currently only a
    /// non-contiguous ndarray.
    Owned(Mat<T>),
}

impl<T> MatInput<'_, T> {
    /// Borrow the input as a matrix.
    ///
    /// ### Returns
    ///
    /// A `MatRef` over the samples x features data, valid for as long as the
    /// `MatInput` is held.
    pub fn as_mat_ref(&self) -> MatRef<'_, T> {
        match self {
            MatInput::Borrowed(mat) => *mat,
            MatInput::Owned(mat) => mat.as_ref(),
        }
    }
}

/////////////////////
// ManifoldsMatrix //
/////////////////////

/// Anything that can be handed to an embedding entry point as a
/// samples-by-features matrix.
///
/// The contract is orientation, not layout: rows are samples, columns are
/// features. Every implementation is zero-copy bar a non-contiguous ndarray,
/// which has no single stride pattern a `MatRef` could adopt.
///
/// ### Note
///
/// A bare `&[T]` cannot implement this, since a slice carries no shape. Pass
/// the triple `(data, n_samples, n_features)` instead.
pub trait ManifoldsMatrix<T> {
    /// View the input as a matrix.
    ///
    /// ### Returns
    ///
    /// A [`MatInput`] borrowing the source where the layout already agrees,
    /// and owning a materialised copy where it does not.
    fn to_mat_input(&self) -> MatInput<'_, T>;
}

//////////
// faer //
//////////

impl<T> ManifoldsMatrix<T> for MatRef<'_, T> {
    fn to_mat_input(&self) -> MatInput<'_, T> {
        MatInput::Borrowed(*self)
    }
}

impl<T> ManifoldsMatrix<T> for &Mat<T> {
    fn to_mat_input(&self) -> MatInput<'_, T> {
        MatInput::Borrowed(self.as_ref())
    }
}

impl<T> ManifoldsMatrix<T> for Mat<T> {
    fn to_mat_input(&self) -> MatInput<'_, T> {
        MatInput::Borrowed(self.as_ref())
    }
}

//////////////////
// Flat sources //
//////////////////

/// A row-major buffer with its shape, for callers arriving over FFI who
/// already hold a contiguous array.
///
/// ### Panics
///
/// If `data.len()` is not `n_samples * n_features`.
impl<T> ManifoldsMatrix<T> for (&[T], usize, usize) {
    fn to_mat_input(&self) -> MatInput<'_, T> {
        let (data, n_samples, n_features) = *self;
        assert_eq!(
            data.len(),
            n_samples * n_features,
            "flat input length {} does not match shape {n_samples} x {n_features}",
            data.len()
        );
        MatInput::Borrowed(MatRef::from_row_major_slice(data, n_samples, n_features))
    }
}

/// Owned variant of the flat triple, for callers who built the buffer and hand
/// it straight over.
///
/// ### Panics
///
/// If `data.len()` is not `n_samples * n_features`.
impl<T> ManifoldsMatrix<T> for (Vec<T>, usize, usize) {
    fn to_mat_input(&self) -> MatInput<'_, T> {
        let (data, n_samples, n_features) = (&self.0, self.1, self.2);
        assert_eq!(
            data.len(),
            n_samples * n_features,
            "flat input length {} does not match shape {n_samples} x {n_features}",
            data.len()
        );
        MatInput::Borrowed(MatRef::from_row_major_slice(data, n_samples, n_features))
    }
}

/////////////
// ndarray //
/////////////

/// Contiguous views borrow. A transposed or otherwise strided view is
/// materialised.
#[cfg(feature = "ndarray")]
impl<T> ManifoldsMatrix<T> for ArrayView2<'_, T>
where
    T: ManifoldsFloat,
{
    fn to_mat_input(&self) -> MatInput<'_, T> {
        let (n_samples, n_features) = (self.nrows(), self.ncols());

        match self.to_slice() {
            Some(slice) => {
                MatInput::Borrowed(MatRef::from_row_major_slice(slice, n_samples, n_features))
            }
            None => MatInput::Owned(Mat::from_fn(n_samples, n_features, |i, j| self[(i, j)])),
        }
    }
}

#[cfg(feature = "ndarray")]
impl<T> ManifoldsMatrix<T> for Array2<T>
where
    T: ManifoldsFloat,
{
    fn to_mat_input(&self) -> MatInput<'_, T> {
        let (n_samples, n_features) = (self.nrows(), self.ncols());

        match self.as_slice() {
            Some(slice) => {
                MatInput::Borrowed(MatRef::from_row_major_slice(slice, n_samples, n_features))
            }
            None => MatInput::Owned(Mat::from_fn(n_samples, n_features, |i, j| self[(i, j)])),
        }
    }
}

#[cfg(feature = "ndarray")]
impl<T> ManifoldsMatrix<T> for &Array2<T>
where
    T: ManifoldsFloat,
{
    fn to_mat_input(&self) -> MatInput<'_, T> {
        (*self).to_mat_input()
    }
}
