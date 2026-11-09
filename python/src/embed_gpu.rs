//! The GPU embedding entry points.
//!
//! Only the neighbour search and the optimiser (UMAP's Adam, t-SNE's
//! `"fft_3k_gpu"`) run on the device, plus the SVD and k-means of PHATE's spectral
//! landmark selection; the graph construction and the initialisation stay on the
//! CPU. That is the crate's split, not a simplification made here.
//!
//! The device is pinned to wgpu: there is nothing on the Python side to choose
//! a backend with.
//!
//! The design matrix dtype picks `T`, as on the CPU paths. WGSL has no `f64`, so
//! the Python estimators always send `f32`.

use cubecl::wgpu::WgpuDevice;
use cubecl::Device;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::dispatch::{embed_dispatch, embed_dispatch_ref};

/// Fetch the default wgpu device.
///
/// ### Returns
///
/// The adapter wgpu picks for this machine: Metal on macOS, Vulkan or DX12
/// elsewhere.
pub(crate) fn default_device() -> Device {
    WgpuDevice::default().into()
}

///////////////////
// Entry points  //
///////////////////

/// UMAP with a GPU neighbour search and a GPU Adam optimiser.
///
/// ### Params
///
/// * `x` - Samples by features, C-contiguous float32 or float64.
/// * `params` - Parameters, as built by the Python layer. See
///   [`crate::params::umap_gpu`].
/// * `knn_indices` - Optional `(n, k)` precomputed neighbour indices.
/// * `knn_distances` - Optional `(n, k)` distances, same dtype as `x`.
/// * `seed` - Fixes the initialisation and the negative sampling.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// The embedding as an `(n_samples, n_dim)` array, same dtype as `x`.
#[pyfunction]
#[pyo3(signature = (x, params, *, knn_indices = None, knn_distances = None, seed = 42, verbose = 0))]
pub fn umap_gpu<'py>(
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
        umap_gpu,
        |data, n, dim, p, knn| manifolds_rs::umap_gpu(
            (data, n, dim),
            knn,
            p,
            default_device(),
            seed,
            verbose
        )
    )
}

/// PHATE with the neighbour search and spectral landmark selection on the GPU.
///
/// ### Params
///
/// * `x` - Samples by features, C-contiguous float32 or float64.
/// * `params` - Parameters, as built by the Python layer. See
///   [`crate::params::phate_gpu`].
/// * `knn_indices` - Optional `(n, k)` precomputed neighbour indices.
/// * `knn_distances` - Optional `(n, k)` distances, same dtype as `x`.
/// * `seed` - Fixes the sketch, the k-means initialisation and the MDS.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// The embedding as an `(n_samples, n_dim)` array, same dtype as `x`.
#[pyfunction]
#[pyo3(signature = (x, params, *, knn_indices = None, knn_distances = None, seed = 42, verbose = 0))]
pub fn phate_gpu<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    params: &Bound<'py, PyDict>,
    knn_indices: Option<&Bound<'py, PyAny>>,
    knn_distances: Option<&Bound<'py, PyAny>>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyAny>> {
    // `phate_gpu` takes the PHATE parameters by value, hence the clone.
    embed_dispatch!(
        py,
        x,
        params,
        knn_indices,
        knn_distances,
        phate_gpu,
        |data, n, dim, p: &(_, _), knn| manifolds_rs::phate_gpu(
            (data, n, dim),
            knn,
            Clone::clone(&p.0),
            &p.1,
            default_device(),
            seed,
            verbose
        )
    )
}

/// Density-preserving UMAP on the GPU.
///
/// ### Params
///
/// * `x` - Samples by features, C-contiguous float32 or float64.
/// * `params` - Parameters, as built by the Python layer. See
///   [`crate::params::densmap_gpu`].
/// * `knn_indices` - Optional `(n, k)` precomputed neighbour indices.
/// * `knn_distances` - Optional `(n, k)` distances, same dtype as `x`.
/// * `seed` - Fixes the initialisation and the negative sampling.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// The embedding as an `(n_samples, n_dim)` array, same dtype as `x`.
#[pyfunction]
#[pyo3(signature = (x, params, *, knn_indices = None, knn_distances = None, seed = 42, verbose = 0))]
pub fn densmap_gpu<'py>(
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
        densmap_gpu,
        |data, n, dim, p, knn| manifolds_rs::densmap_gpu(
            (data, n, dim),
            knn,
            p,
            default_device(),
            seed,
            verbose
        )
    )
}

/// t-SNE with a GPU neighbour search, and a GPU optimiser for `"fft_3k_gpu"`.
///
/// ### Params
///
/// * `x` - Samples by features, C-contiguous float32 or float64. Must be
///   float32 for `"fft_3k_gpu"`; the Python layer guarantees that.
/// * `params` - Parameters, as built by the Python layer. See
///   [`crate::params::tsne_gpu`].
/// * `approx` - Repulsive-force approximation. `"barnes_hut"`, `"qd"`,
///   `"fft"` and `"fft_3k"` optimise on the CPU, `"fft_3k_gpu"` on the device.
/// * `knn_indices` - Optional `(n, k)` precomputed neighbour indices.
/// * `knn_distances` - Optional `(n, k)` distances, same dtype as `x`.
/// * `seed` - Fixes the initialisation.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// The embedding as an `(n_samples, 2)` array, same dtype as `x`.
#[pyfunction]
#[pyo3(signature = (x, params, *, approx = "barnes_hut", knn_indices = None, knn_distances = None, seed = 42, verbose = 0))]
#[allow(clippy::too_many_arguments)]
pub fn tsne_gpu<'py>(
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
        tsne_gpu,
        |data, n, dim, p, knn| manifolds_rs::tsne_gpu(
            (data, n, dim),
            knn,
            p,
            approx,
            default_device(),
            seed,
            verbose
        )
    )
}

/// DREAMS with a GPU neighbour search, and a GPU optimiser for `"fft_3k_gpu"`.
///
/// ### Params
///
/// * `x` - Samples by features, C-contiguous float32 or float64. Must be
///   float32 for `"fft_3k_gpu"`; the Python layer guarantees that.
/// * `params` - Parameters, as built by the Python layer. See
///   [`crate::params::dreams_gpu`].
/// * `approx` - Repulsive-force approximation. See [`tsne_gpu`].
/// * `reference` - Optional `(n, 2)` reference embedding, same dtype as `x`.
///   `None` uses PCA.
/// * `knn_indices` - Optional `(n, k)` precomputed neighbour indices.
/// * `knn_distances` - Optional `(n, k)` distances, same dtype as `x`.
/// * `seed` - Fixes the PCA initialisation and the optimiser.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// The embedding as an `(n_samples, 2)` array, same dtype as `x`.
#[pyfunction]
#[pyo3(signature = (x, params, *, approx = "barnes_hut", reference = None, knn_indices = None, knn_distances = None, seed = 42, verbose = 0))]
#[allow(clippy::too_many_arguments)]
pub fn dreams_gpu<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    params: &Bound<'py, PyDict>,
    approx: &str,
    reference: Option<&Bound<'py, PyAny>>,
    knn_indices: Option<&Bound<'py, PyAny>>,
    knn_distances: Option<&Bound<'py, PyAny>>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyAny>> {
    embed_dispatch_ref!(
        py,
        x,
        params,
        knn_indices,
        knn_distances,
        reference,
        dreams_gpu,
        |data, n, dim, p, knn, rf| manifolds_rs::dreams_gpu(
            (data, n, dim),
            knn,
            rf,
            p,
            approx,
            default_device(),
            seed,
            verbose
        )
    )
}
