//! Standalone neighbour search.
//!
//! Every embedding entry point accepts a precomputed graph, and building one
//! here means a caller sweeping parameters pays for the neighbour search once
//! rather than once per embedding. That is the crate's own recommendation, and
//! on anything large the search is the dominant cost.
//!
//! This is the same `run_ann_search` the embeddings call internally, so a graph
//! from here is exactly what they would have built for themselves.

use faer::MatRef;
use manifolds_rs::prelude::run_ann_search;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::ManErr;
use crate::{convert, params, pool};

/// One dtype arm of the neighbour search.
///
/// `run_ann_search` takes a `MatRef` rather than the `ManifoldsMatrix` the
/// embeddings accept, so the borrowed slice is wrapped here. `from_row_major_slice`
/// is a view, not a copy.
macro_rules! knn_arm {
    ($py:ident, $a:ident, $k:ident, $ann:ident, $dict:ident, $t:ty, $seed:ident, $verbose:ident) => {{
        let (data, n, dim) = convert::flat(&$a)?;
        let mut nn = manifolds_rs::prelude::NearestNeighbourParams::<$t>::default();
        if let Some(d) = $dict {
            params::fill_nn(d, &mut nn)?;
        }
        let ann = $ann.to_string();
        let (indices, distances) = $py
            .detach(|| {
                pool::run(|| {
                    let m = MatRef::from_row_major_slice(data, n, dim);
                    run_ann_search(m, $k, ann, &nn, $seed, $verbose)
                })
            })
            .map_err(ManErr)?;
        let (i, d) = convert::pack_knn($py, &indices, &distances, $k)?;
        Ok((i.into_any(), d.into_any()))
    }};
}

/// Build a k-nearest-neighbour graph over `x`.
///
/// ### Params
///
/// * `x` - Samples by features, C-contiguous float32 or float64.
/// * `k` - Neighbours per point, excluding self.
/// * `ann` - Backend: `"exhaustive"`, `"kmknn"`, `"balltree"`, `"annoy"`,
///   `"hnsw"`, `"ivf"` or `"nndescent"`. Validated by the Python layer, since
///   the core falls back to `"kmknn"` on an unrecognised name and only says so
///   on stdout.
/// * `nn_params` - Backend-specific knobs, or `None` for the crate defaults.
/// * `seed` - Fixes anything randomised in the build.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// `(indices, distances)`, both `(n_samples, k)`. Indices are int64 and
/// distances match the dtype of `x`. A row an approximate backend could not
/// fill is padded with `-1` and `+inf`. Note that the distances are whatever
/// the chosen metric returns, squared Euclidean included, and the embedding
/// entry points expect exactly that.
#[pyfunction]
#[pyo3(signature = (x, k, *, ann = "kmknn", nn_params = None, seed = 42, verbose = 0))]
pub fn knn_graph<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    k: usize,
    ann: &str,
    nn_params: Option<&Bound<'py, PyDict>>,
    seed: usize,
    verbose: usize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    if let Ok(a) = x.extract::<PyReadonlyArray2<'_, f32>>() {
        knn_arm!(py, a, k, ann, nn_params, f32, seed, verbose)
    } else if let Ok(a) = x.extract::<PyReadonlyArray2<'_, f64>>() {
        knn_arm!(py, a, k, ann, nn_params, f64, seed, verbose)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "X must be a 2-D numpy array of dtype float32 or float64",
        ))
    }
}
