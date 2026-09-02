//! numpy in, numpy out.
//!
//! Input goes through `PyReadonlyArray2::as_slice`, which feeds the crate's
//! `impl ManifoldsMatrix<T> for (&[T], usize, usize)` directly, so the design
//! matrix is borrowed rather than copied.
//!
//! Output needs a transpose. Every entry point returns `Vec<Vec<T>>` laid out
//! as `[n_dim][n_samples]`, one vector per embedding dimension, and Python
//! wants `(n_samples, n_dim)`. Both the transpose and the flattening happen in
//! [`pack_embedding`], in one pass over an already-allocated buffer.

use numpy::{
    Element, IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

///////////
// Types //
///////////

/// The ragged kNN representation the crate's entry points take.
///
/// Neighbour indices and distances, one inner vector per point, both excluding
/// self. Approximate backends can return a short row, which is why it is ragged
/// rather than a flat buffer with a stride.
pub(crate) type RaggedKnn<T> = (Vec<Vec<usize>>, Vec<Vec<T>>);

/// The same graph on the Python side: two dense `(n, k)` arrays.
pub(crate) type KnnArrays<'py, T> = (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<T>>);

////////////
// Inputs //
////////////

/// Borrow a C-contiguous 2-D array as `(data, n_rows, n_cols)`.
///
/// The returned slice borrows from `a`, so `a` must outlive any
/// `Python::detach` the slice is passed into. Keep this call outside the
/// closure.
///
/// ### Params
///
/// * `a` - Read-only view of a 2-D numpy array.
///
/// ### Returns
///
/// `(data, n_rows, n_cols)`, row-major, in the shape the crate's
/// `ManifoldsMatrix` tuple impl wants. Errors if the array is not
/// C-contiguous; the Python layer runs `np.ascontiguousarray` first, so this is
/// a backstop.
pub(crate) fn flat<'a, T: Element>(
    a: &'a PyReadonlyArray2<'_, T>,
) -> PyResult<(&'a [T], usize, usize)> {
    let shape = a.shape();
    let (n, dim) = (shape[0], shape[1]);
    let data = a.as_slice().map_err(|_| {
        PyValueError::new_err("array must be C-contiguous; use np.ascontiguousarray")
    })?;
    Ok((data, n, dim))
}

/// Rebuild the crate's ragged kNN representation from two dense arrays.
///
/// The core takes `(Vec<Vec<usize>>, Vec<Vec<T>>)` excluding self, which is
/// what every ANN backend in `ann-search-rs` hands back. A dense `(n, k)` pair
/// is how the same graph reaches Python, so the row split happens here. The
/// allocation is `n` small vectors, which is noise next to the optimisation
/// loop that follows.
///
/// ### Params
///
/// * `indices` - `(n_samples, k)` neighbour indices, excluding self.
/// * `distances` - `(n_samples, k)` distances, aligned with `indices`.
/// * `n_samples` - Rows the design matrix has, so a mismatched graph is caught
///   here rather than deep in the graph construction.
///
/// ### Returns
///
/// The ragged pair, or an error naming the mismatch. Negative indices are
/// rejected: `-1` is the padding value on the query side, and a padded kNN
/// graph is not a graph the embedding routines can use.
pub(crate) fn unpack_knn<T>(
    indices: &PyReadonlyArray2<'_, i64>,
    distances: &PyReadonlyArray2<'_, T>,
    n_samples: usize,
) -> PyResult<RaggedKnn<T>>
where
    T: Element + Copy,
{
    let (ind, n_ind, k_ind) = flat(indices)?;
    let (dist, n_dist, k_dist) = flat(distances)?;

    if (n_ind, k_ind) != (n_dist, k_dist) {
        return Err(PyValueError::new_err(format!(
            "knn_indices has shape ({n_ind}, {k_ind}) but knn_distances has ({n_dist}, {k_dist})"
        )));
    }
    if n_ind != n_samples {
        return Err(PyValueError::new_err(format!(
            "knn arrays have {n_ind} rows but X has {n_samples} samples"
        )));
    }
    if let Some(bad) = ind.iter().find(|&&v| v < 0) {
        return Err(PyValueError::new_err(format!(
            "knn_indices contains {bad}; padded rows cannot be used to build an embedding graph"
        )));
    }

    let idx = ind
        .chunks_exact(k_ind)
        .map(|row| row.iter().map(|&v| v as usize).collect())
        .collect();
    let dst = dist.chunks_exact(k_dist).map(|row| row.to_vec()).collect();
    Ok((idx, dst))
}

/////////////
// Outputs //
/////////////

/// Transpose `[n_dim][n_samples]` into a dense `(n_samples, n_dim)` array.
///
/// One allocation and one pass. The inner loop writes down a column of the
/// output, which is the strided side, but `n_dim` is 2 or 3 in every realistic
/// call so the whole row fits in a cache line regardless.
///
/// ### Params
///
/// * `py` - Attached interpreter token.
/// * `embedding` - One vector per embedding dimension, each `n_samples` long.
///
/// ### Returns
///
/// An `(n_samples, n_dim)` array. `into_pyarray` moves the buffer into the
/// numpy object and `reshape` returns a view, so neither copies. Errors if the
/// dimensions came back ragged, which would mean the core broke its own
/// contract.
pub(crate) fn pack_embedding<'py, T>(
    py: Python<'py>,
    embedding: Vec<Vec<T>>,
) -> PyResult<Bound<'py, PyArray2<T>>>
where
    T: Element + Copy + Default,
{
    let n_dim = embedding.len();
    let n = embedding.first().map(Vec::len).unwrap_or(0);
    if embedding.iter().any(|d| d.len() != n) {
        return Err(PyValueError::new_err(
            "core returned embedding dimensions of unequal length",
        ));
    }

    transpose(&embedding).into_pyarray(py).reshape([n, n_dim])
}

/// Interleave `[n_dim][n_samples]` columns into a row-major `(n_samples, n_dim)`
/// buffer.
///
/// ### Params
///
/// * `embedding` - One vector per embedding dimension, all the same length.
///
/// ### Returns
///
/// A row-major buffer of `n_samples * n_dim` elements.
fn transpose<T: Copy + Default>(embedding: &[Vec<T>]) -> Vec<T> {
    let n_dim = embedding.len();
    let n = embedding.first().map(Vec::len).unwrap_or(0);
    let mut out = vec![T::default(); n * n_dim];
    for (d, column) in embedding.iter().enumerate() {
        for (i, &value) in column.iter().enumerate() {
            out[i * n_dim + d] = value;
        }
    }
    out
}

/// Flatten ragged rows into a dense `n * k` buffer, padding short rows.
///
/// One allocation, no reallocation: the buffer is filled with `fill` up front
/// and each row's prefix is written over it. Rows longer than `k` are
/// truncated, which cannot happen today but keeps a future over-returning
/// backend from panicking here.
///
/// ### Params
///
/// * `rows` - One row per point, each holding between 0 and `k` entries.
/// * `k` - Neighbours requested, and the row stride of the output.
/// * `fill` - Value written into slots no row entry reached.
/// * `convert` - Applied to each entry on the way in.
///
/// ### Returns
///
/// A row-major buffer of `rows.len() * k` elements.
fn densify<S, D>(rows: &[Vec<S>], k: usize, fill: D, convert: impl Fn(S) -> D) -> Vec<D>
where
    S: Copy,
    D: Copy,
{
    let mut out = vec![fill; rows.len() * k];
    for (i, row) in rows.iter().enumerate() {
        let slots = &mut out[i * k..(i + 1) * k];
        for (slot, &value) in slots.iter_mut().zip(row.iter().take(k)) {
            *slot = convert(value);
        }
    }
    out
}

/// Pack a computed kNN graph into a dense `(n, k)` index and distance pair.
///
/// Padding is `-1` for indices and `+inf` for distances, the pynndescent and
/// umap convention. `+inf` keeps each row totally ordered, so `argsort` and
/// `min` still hold; NaN would poison both. An exact backend never pads, but an
/// approximate one can return a short row.
///
/// ### Params
///
/// * `py` - Attached interpreter token.
/// * `indices` - Ragged neighbour indices, one row per point.
/// * `distances` - Ragged distances, aligned with `indices`.
/// * `k` - Neighbours requested, and the row stride of both outputs.
///
/// ### Returns
///
/// `(indices, distances)` as `(n, k)` arrays, int64 and `T` respectively.
pub(crate) fn pack_knn<'py, T>(
    py: Python<'py>,
    indices: &[Vec<usize>],
    distances: &[Vec<T>],
    k: usize,
) -> PyResult<KnnArrays<'py, T>>
where
    T: Element + num_traits::Float,
{
    let n = indices.len();
    let idx = densify(indices, k, -1i64, |v| v as i64);
    let dst = densify(distances, k, T::infinity(), |v| v);
    Ok((
        idx.into_pyarray(py).reshape([n, k])?,
        dst.into_pyarray(py).reshape([n, k])?,
    ))
}

/// Copy a faer matrix into a fresh `(n_rows, n_cols)` numpy array.
///
/// Only the synthetic generators need this: they hand back an owned `Mat<f64>`
/// rather than writing into a caller's buffer, and `Mat` is column-major, so
/// there is nothing to borrow and a copy is unavoidable.
///
/// ### Params
///
/// * `py` - Attached interpreter token.
/// * `m` - The matrix to copy.
///
/// ### Returns
///
/// A C-contiguous `(n_rows, n_cols)` float64 array.
pub(crate) fn mat_to_numpy<'py>(
    py: Python<'py>,
    m: &faer::Mat<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (rows, cols) = (m.nrows(), m.ncols());
    let mut out = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            out.push(m[(i, j)]);
        }
    }
    out.into_pyarray(py).reshape([rows, cols])
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_interleaves_dimensions() {
        let embd = vec![vec![1.0f64, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        assert_eq!(transpose(&embd), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_three_dimensions() {
        let embd = vec![vec![1.0f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        assert_eq!(transpose(&embd), vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_transpose_no_samples_gives_no_buffer() {
        let embd: Vec<Vec<f64>> = vec![vec![], vec![]];
        assert!(transpose(&embd).is_empty());
    }
}
