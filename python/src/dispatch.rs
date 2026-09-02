//! Float-type dispatch for the embedding entry points.
//!
//! Every entry point has two arms that are token-identical and differ only in
//! the type inferred for `T`, so each is written once here and expanded twice.
//! A generic helper function could not do this without restating each entry
//! point's trait bounds, and those bounds differ (`tsne` wants `FftwFloat` in
//! an FFT build, the GPU paths want `CubeclFloat`).
//!
//! Note the nesting in every macro: `py.detach(|| pool::run(|| ...))`, never
//! the other way round. `Python<'py>` is not `Send`, so a `pool::run` closure
//! capturing it would not satisfy `Ungil`; and the GIL should be dropped before
//! the rayon fan-out starts regardless.

use pyo3::prelude::*;

////////////
// Macros //
////////////

/// Run one embedding entry point, picking the arm from the design matrix dtype.
///
/// Expands `$run` once per float type. `$build` names a builder in
/// [`crate::params`] and `$run` must be a closure
/// `|data, n, dim, params, knn| -> Result<Vec<Vec<T>>, ManifoldsError>`, where
/// `params` is the built parameter struct and `knn` the unpacked graph.
///
/// The kNN arrays are read inside the arm rather than before it, because the
/// distance array has to match the element type the design matrix picked.
macro_rules! embed_dispatch {
    ($py:ident, $x:ident, $dict:ident, $ki:ident, $kd:ident, $build:ident, $run:expr) => {{
        if let Ok(a) = $x.extract::<::numpy::PyReadonlyArray2<'_, f32>>() {
            crate::dispatch::embed_arm!($py, a, $dict, $ki, $kd, f32, $build, $run)
        } else if let Ok(a) = $x.extract::<::numpy::PyReadonlyArray2<'_, f64>>() {
            crate::dispatch::embed_arm!($py, a, $dict, $ki, $kd, f64, $build, $run)
        } else {
            Err(::pyo3::exceptions::PyTypeError::new_err(
                "X must be a 2-D numpy array of dtype float32 or float64",
            ))
        }
    }};
}

/// One dtype arm of an embedding call.
///
/// Split out of [`embed_dispatch`] only so the body is written once rather than
/// twice; it is not useful on its own.
macro_rules! embed_arm {
    ($py:ident, $a:ident, $dict:ident, $ki:ident, $kd:ident, $t:ty, $build:ident, $run:expr) => {{
        let (data, n, dim) = crate::convert::flat(&$a)?;
        let params = crate::params::$build::<$t>($dict)?;
        let knn = crate::dispatch::knn_arm::<$t>($ki, $kd, n)?;
        let f = $run;
        let embedding = $py
            .detach(|| crate::pool::run(|| f(data, n, dim, &params, knn)))
            .map_err(crate::error::ManErr)?;
        Ok(crate::convert::pack_embedding($py, embedding)?.into_any())
    }};
}

pub(crate) use {embed_arm, embed_dispatch};

/////////////
// Helpers //
/////////////

/// Read the optional precomputed kNN for one dtype arm.
///
/// The distance array must match the design matrix's element type. Casting it
/// here would hide a silent doubling of the caller's memory, so the Python
/// layer does the conversion knowingly and this is the backstop.
///
/// ### Params
///
/// * `indices` - `(n, k)` neighbour indices, or `None`.
/// * `distances` - `(n, k)` distances, or `None`.
/// * `n` - Rows the design matrix has.
///
/// ### Returns
///
/// The ragged pair the core wants, or `None` when no graph was supplied.
/// Errors if only one of the two arrays was given.
pub(crate) fn knn_arm<T>(
    indices: Option<&Bound<'_, PyAny>>,
    distances: Option<&Bound<'_, PyAny>>,
    n: usize,
) -> PyResult<Option<crate::convert::RaggedKnn<T>>>
where
    T: numpy::Element + Copy,
{
    match (indices, distances) {
        (None, None) => Ok(None),
        (Some(i), Some(d)) => {
            let i = i.extract::<numpy::PyReadonlyArray2<'_, i64>>()?;
            let d = d.extract::<numpy::PyReadonlyArray2<'_, T>>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "knn_distances must have the same dtype as X",
                )
            })?;
            crate::convert::unpack_knn(&i, &d, n).map(Some)
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "knn_indices and knn_distances must be given together",
        )),
    }
}
