//! The GPU embedding entry points.
//!
//! Only the neighbour search and the Adam update run on the device; the graph
//! construction and the spectral initialisation stay on the CPU. That is the
//! crate's split, not a simplification made here.
//!
//! The runtime is pinned to `WgpuRuntime`. `R` is a type parameter in the
//! library so the kernels can be tested against the CPU backend, but a generic
//! cannot cross into Python: the device type is `R::Device`, and there is
//! nothing on the Python side to choose it with.
//!
//! Unlike the CPU paths these are float32 only. WGSL has no `f64`, and the
//! Python layer casts on the way in rather than letting a float64 array fail
//! somewhere inside a kernel.

use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::ManErr;
use crate::{convert, dispatch, params, pool};

/// The only runtime the bindings expose.
pub(crate) type Rt = WgpuRuntime;

/// Fetch the default wgpu device.
///
/// ### Returns
///
/// The adapter wgpu picks for this machine: Metal on macOS, Vulkan or DX12
/// elsewhere.
pub(crate) fn default_device() -> WgpuDevice {
    WgpuDevice::default()
}

////////////
// Macros //
////////////

/// Run one GPU entry point.
///
/// No dtype dispatch, so this is a plain function body rather than a two-arm
/// macro, but the `py.detach(|| pool::run(...))` nesting rule is the same as
/// [`crate::dispatch`] and matters for the same reason: the CPU half of these
/// pipelines is still a rayon fan-out.
macro_rules! gpu_body {
    ($py:ident, $x:ident, $dict:ident, $ki:ident, $kd:ident, $build:ident, $run:expr) => {{
        let a = $x.extract::<PyReadonlyArray2<'_, f32>>().map_err(|_| {
            ::pyo3::exceptions::PyTypeError::new_err(
                "the GPU paths are float32 only, since WGSL has no f64; pass a \
                     C-contiguous float32 array",
            )
        })?;
        let (data, n, dim) = convert::flat(&a)?;
        let p = params::$build::<f32>($dict)?;
        let knn = dispatch::knn_arm::<f32>($ki, $kd, n)?;
        let f = $run;
        let embedding = $py
            .detach(|| pool::run(|| f(data, n, dim, &p, knn, default_device())))
            .map_err(ManErr)?;
        Ok(convert::pack_embedding($py, embedding)?.into_any())
    }};
}

///////////////////
// Entry points  //
///////////////////

/// UMAP with a GPU neighbour search and a GPU Adam optimiser.
///
/// ### Params
///
/// * `x` - Samples by features, C-contiguous float32.
/// * `params` - Parameters, as built by the Python layer. See
///   [`crate::params::umap_gpu`].
/// * `knn_indices` - Optional `(n, k)` precomputed neighbour indices.
/// * `knn_distances` - Optional `(n, k)` float32 distances.
/// * `seed` - Fixes the initialisation and the negative sampling.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// The embedding as an `(n_samples, n_dim)` float32 array.
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
    gpu_body!(
        py,
        x,
        params,
        knn_indices,
        knn_distances,
        umap_gpu,
        |data, n, dim, p, knn, device| manifolds_rs::umap_gpu::<f32, Rt>(
            (data, n, dim),
            knn,
            p,
            device,
            seed,
            verbose
        )
    )
}

/// Density-preserving UMAP on the GPU.
///
/// ### Params
///
/// * `x` - Samples by features, C-contiguous float32.
/// * `params` - Parameters, as built by the Python layer. See
///   [`crate::params::densmap_gpu`].
/// * `knn_indices` - Optional `(n, k)` precomputed neighbour indices.
/// * `knn_distances` - Optional `(n, k)` float32 distances.
/// * `seed` - Fixes the initialisation and the negative sampling.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// The embedding as an `(n_samples, n_dim)` float32 array.
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
    gpu_body!(
        py,
        x,
        params,
        knn_indices,
        knn_distances,
        densmap_gpu,
        |data, n, dim, p, knn, device| manifolds_rs::densmap_gpu::<f32, Rt>(
            (data, n, dim),
            knn,
            p,
            device,
            seed,
            verbose
        )
    )
}

/// t-SNE with a GPU neighbour search.
///
/// ### Params
///
/// * `x` - Samples by features, C-contiguous float32.
/// * `params` - Parameters, as built by the Python layer. See
///   [`crate::params::tsne_gpu`].
/// * `approx` - Repulsive-force approximation. `"fft"` needs the `fft_tsne`
///   feature, which the wheel is not built with.
/// * `knn_indices` - Optional `(n, k)` precomputed neighbour indices.
/// * `knn_distances` - Optional `(n, k)` float32 distances.
/// * `seed` - Fixes the initialisation.
/// * `verbose` - `0` silent, `1` normal, `2` detailed.
///
/// ### Returns
///
/// The embedding as an `(n_samples, 2)` float32 array.
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
    gpu_body!(
        py,
        x,
        params,
        knn_indices,
        knn_distances,
        tsne_gpu,
        |data, n, dim, p, knn, device| manifolds_rs::tsne_gpu::<f32, Rt>(
            (data, n, dim),
            knn,
            p,
            approx,
            device,
            seed,
            verbose
        )
    )
}
