//! The CPU embedding entry points.
//!
//! Each is the same shape: a design matrix, a parameter dictionary, an optional
//! precomputed kNN graph, a seed and a verbosity level, returning an
//! `(n_samples, n_dim)` array. The dtype of the design matrix picks the
//! element type the whole pipeline runs in.
//!
//! Nothing here validates a string. `ann_type`, `initialisation`, `optimiser`
//! and friends are checked against an allowlist in the Python layer, because
//! the core's parsers fall back to a default and say so on stdout rather than
//! erroring, which is invisible from a notebook.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::dispatch::embed_dispatch;

/// Uniform manifold approximation and projection.
///
/// ### Params
///
/// * `x` - Samples by features, C-contiguous float32 or float64.
/// * `params` - Parameters, as built by the Python layer. See
///   [`crate::params::umap`].
/// * `knn_indices` - Optional `(n, k)` precomputed neighbour indices.
/// * `knn_distances` - Optional `(n, k)` distances, same dtype as `x`.
/// * `seed` - Fixes the initialisation and the negative sampling.
/// * `verbose` - `0` silent, `1` normal, `2` detailed. Goes to the process
///   stdout, not `sys.stdout`.
///
/// ### Returns
///
/// The embedding as an `(n_samples, n_dim)` array of the input's float type.
#[pyfunction]
#[pyo3(signature = (x, params, *, knn_indices = None, knn_distances = None, seed = 42, verbose = 0))]
pub fn umap<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    params: &Bound<'py, PyDict>,
    knn_indices: Option<&Bound<'py, PyAny>>,
    knn_distances: Option<&Bound<'py, PyAny>>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyAny>> {
    embed_dispatch!(
        py,
        x,
        params,
        knn_indices,
        knn_distances,
        umap,
        |data, n, dim, p, knn| manifolds_rs::umap((data, n, dim), knn, p, seed, verbose)
    )
}

/// Density-preserving UMAP.
///
/// ### Params
///
/// * `x` - Samples by features, C-contiguous float32 or float64.
/// * `params` - Parameters, as built by the Python layer. See
///   [`crate::params::densmap`].
/// * `knn_indices` - Optional `(n, k)` precomputed neighbour indices.
/// * `knn_distances` - Optional `(n, k)` distances, same dtype as `x`.
/// * `seed` - Fixes the initialisation and the negative sampling.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// The embedding as an `(n_samples, n_dim)` array of the input's float type.
#[pyfunction]
#[pyo3(signature = (x, params, *, knn_indices = None, knn_distances = None, seed = 42, verbose = 0))]
pub fn densmap<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    params: &Bound<'py, PyDict>,
    knn_indices: Option<&Bound<'py, PyAny>>,
    knn_distances: Option<&Bound<'py, PyAny>>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyAny>> {
    embed_dispatch!(
        py,
        x,
        params,
        knn_indices,
        knn_distances,
        densmap,
        |data, n, dim, p, knn| manifolds_rs::densmap((data, n, dim), knn, p, seed, verbose)
    )
}

/// t-distributed stochastic neighbour embedding.
///
/// ### Params
///
/// * `x` - Samples by features, C-contiguous float32 or float64.
/// * `params` - Parameters, as built by the Python layer. See
///   [`crate::params::tsne`].
/// * `approx` - Repulsive-force approximation: `"barnes_hut"` always, `"fft"`
///   only in a build with the `fft_tsne` feature. The wheel is built without
///   it, since FFTW is a system dependency no manylinux container carries.
/// * `knn_indices` - Optional `(n, k)` precomputed neighbour indices.
/// * `knn_distances` - Optional `(n, k)` distances, same dtype as `x`.
/// * `seed` - Fixes the initialisation.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// The embedding as an `(n_samples, 2)` array of the input's float type. t-SNE
/// here is 2-D only; any other `n_dim` is an error from the core.
#[pyfunction]
#[pyo3(signature = (x, params, *, approx = "barnes_hut", knn_indices = None, knn_distances = None, seed = 42, verbose = 0))]
#[allow(clippy::too_many_arguments)]
pub fn tsne<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    params: &Bound<'py, PyDict>,
    approx: &str,
    knn_indices: Option<&Bound<'py, PyAny>>,
    knn_distances: Option<&Bound<'py, PyAny>>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyAny>> {
    embed_dispatch!(
        py,
        x,
        params,
        knn_indices,
        knn_distances,
        tsne,
        |data, n, dim, p, knn| manifolds_rs::tsne((data, n, dim), knn, p, approx, seed, verbose)
    )
}

/// Density-preserving t-SNE.
///
/// ### Params
///
/// * `x` - Samples by features, C-contiguous float32 or float64.
/// * `params` - Parameters, as built by the Python layer. See
///   [`crate::params::densne`].
/// * `approx` - Repulsive-force approximation. See [`tsne`].
/// * `knn_indices` - Optional `(n, k)` precomputed neighbour indices.
/// * `knn_distances` - Optional `(n, k)` distances, same dtype as `x`.
/// * `seed` - Fixes the initialisation.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// The embedding as an `(n_samples, 2)` array of the input's float type.
#[pyfunction]
#[pyo3(signature = (x, params, *, approx = "barnes_hut", knn_indices = None, knn_distances = None, seed = 42, verbose = 0))]
#[allow(clippy::too_many_arguments)]
pub fn densne<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    params: &Bound<'py, PyDict>,
    approx: &str,
    knn_indices: Option<&Bound<'py, PyAny>>,
    knn_distances: Option<&Bound<'py, PyAny>>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyAny>> {
    embed_dispatch!(
        py,
        x,
        params,
        knn_indices,
        knn_distances,
        densne,
        |data, n, dim, p, knn| manifolds_rs::densne((data, n, dim), knn, p, approx, seed, verbose)
    )
}

/// Potential of heat diffusion for affinity-based transition embedding.
///
/// ### Params
///
/// * `x` - Samples by features, C-contiguous float32 or float64.
/// * `params` - Parameters, as built by the Python layer. See
///   [`crate::params::phate`].
/// * `knn_indices` - Optional `(n, k)` precomputed neighbour indices.
/// * `knn_distances` - Optional `(n, k)` distances, same dtype as `x`.
/// * `seed` - Fixes the initialisation and the landmark sampling.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// The embedding as an `(n_samples, n_dim)` array of the input's float type.
#[pyfunction]
#[pyo3(signature = (x, params, *, knn_indices = None, knn_distances = None, seed = 42, verbose = 0))]
pub fn phate<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    params: &Bound<'py, PyDict>,
    knn_indices: Option<&Bound<'py, PyAny>>,
    knn_distances: Option<&Bound<'py, PyAny>>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyAny>> {
    // `phate` takes its parameters by value where the others borrow, hence the
    // clone. It is a handful of scalars and three strings.
    embed_dispatch!(
        py,
        x,
        params,
        knn_indices,
        knn_distances,
        phate,
        |data, n, dim, p: &_, knn| manifolds_rs::phate(
            (data, n, dim),
            knn,
            Clone::clone(p),
            seed,
            verbose
        )
    )
}

/// Pairwise-controlled manifold approximation and projection.
///
/// ### Params
///
/// * `x` - Samples by features, C-contiguous float32 or float64.
/// * `params` - Parameters, as built by the Python layer. See
///   [`crate::params::pacmap`].
/// * `knn_indices` - Optional `(n, k)` precomputed neighbour indices. Must hold
///   at least `mn_candidate_end` neighbours, since the mid-near window indexes
///   into it.
/// * `knn_distances` - Optional `(n, k)` distances, same dtype as `x`.
/// * `seed` - Fixes the initialisation and the pair sampling.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// The embedding as an `(n_samples, n_dim)` array of the input's float type.
#[pyfunction]
#[pyo3(signature = (x, params, *, knn_indices = None, knn_distances = None, seed = 42, verbose = 0))]
pub fn pacmap<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    params: &Bound<'py, PyDict>,
    knn_indices: Option<&Bound<'py, PyAny>>,
    knn_distances: Option<&Bound<'py, PyAny>>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyAny>> {
    embed_dispatch!(
        py,
        x,
        params,
        knn_indices,
        knn_distances,
        pacmap,
        |data, n, dim, p, knn| manifolds_rs::pacmap((data, n, dim), knn, p, seed, verbose)
    )
}

/// Diffusion maps.
///
/// ### Params
///
/// * `x` - Samples by features, C-contiguous float32 or float64.
/// * `params` - Parameters, as built by the Python layer. See
///   [`crate::params::diffusion_maps`].
/// * `knn_indices` - Optional `(n, k)` precomputed neighbour indices.
/// * `knn_distances` - Optional `(n, k)` distances, same dtype as `x`.
/// * `seed` - Fixes the landmark sampling.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// The embedding as an `(n_samples, n_dim)` array of the input's float type.
#[pyfunction]
#[pyo3(signature = (x, params, *, knn_indices = None, knn_distances = None, seed = 42, verbose = 0))]
pub fn diffusion_maps<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    params: &Bound<'py, PyDict>,
    knn_indices: Option<&Bound<'py, PyAny>>,
    knn_distances: Option<&Bound<'py, PyAny>>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyAny>> {
    // By value here too, as for `phate`.
    embed_dispatch!(
        py,
        x,
        params,
        knn_indices,
        knn_distances,
        diffusion_maps,
        |data, n, dim, p: &_, knn| manifolds_rs::diffusion_maps(
            (data, n, dim),
            knn,
            Clone::clone(p),
            seed,
            verbose
        )
    )
}
