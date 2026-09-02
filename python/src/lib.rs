//! Python bindings for `manifolds-rs`.
//!
//! Deliberately thin: one function per entry point in the crate, taking a
//! design matrix, a parameter dictionary and an optional precomputed kNN graph.
//! Defaults, argument validation, the string allowlists and the scikit-learn
//! estimator surface all live in the hand-written `manifolds_rs` Python package
//! on top.
//!
//! Three invariants hold throughout, and all three are load-bearing:
//!
//! - Everything expensive runs inside `Python::detach`, so the GIL is dropped
//!   for the whole rayon fan-out. See [`dispatch`] for the nesting rule.
//! - Nothing generic reaches the crate's entry points. The dispatch macros
//!   expand to concrete `f32` / `f64` code, which is what discharges each entry
//!   point's trait bounds without restating any of them.
//! - No default is written down twice. A parameter the caller did not set is
//!   simply absent from the dictionary, and the crate's `Default` impl fills
//!   it. See [`params`].

#![warn(missing_docs)]

use pyo3::prelude::*;

//////////////////////
// Shared plumbing  //
//////////////////////

mod convert;
mod dispatch;
mod error;
mod gpu_probe;
mod params;
mod pool;

/////////////////
// Entry points //
/////////////////

mod datasets;
mod embed;
mod knn;

#[cfg(feature = "gpu")]
mod embed_gpu;

////////////
// Module //
////////////

/// Assemble the extension module.
///
/// ### Params
///
/// * `m` - The module being initialised.
///
/// ### Returns
///
/// Nothing, or the first registration error.
#[pymodule]
fn _manifolds(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Two versions, deliberately. The bindings version on its own line, so a
    // docstring fix does not force a crates.io release; the core version comes
    // from the crate the wheel vendored, which is the only thing that says
    // which numerics are actually inside it.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__core_version__", manifolds_rs::VERSION)?;

    m.add(
        "ManifoldsRsError",
        m.py().get_type::<error::ManifoldsRsError>(),
    )?;
    m.add(
        "ConvergenceError",
        m.py().get_type::<error::ConvergenceError>(),
    )?;

    m.add_function(wrap_pyfunction!(pool::set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(pool::num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_probe::gpu_available, m)?)?;

    m.add_function(wrap_pyfunction!(knn::knn_graph, m)?)?;

    m.add_function(wrap_pyfunction!(datasets::swiss_roll, m)?)?;
    m.add_function(wrap_pyfunction!(datasets::clustered, m)?)?;
    m.add_function(wrap_pyfunction!(datasets::trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(datasets::hierarchical, m)?)?;

    m.add_function(wrap_pyfunction!(embed::umap, m)?)?;
    m.add_function(wrap_pyfunction!(embed::densmap, m)?)?;
    m.add_function(wrap_pyfunction!(embed::tsne, m)?)?;
    m.add_function(wrap_pyfunction!(embed::densne, m)?)?;
    m.add_function(wrap_pyfunction!(embed::phate, m)?)?;
    m.add_function(wrap_pyfunction!(embed::pacmap, m)?)?;
    m.add_function(wrap_pyfunction!(embed::diffusion_maps, m)?)?;

    #[cfg(feature = "gpu")]
    {
        m.add_function(wrap_pyfunction!(embed_gpu::umap_gpu, m)?)?;
        m.add_function(wrap_pyfunction!(embed_gpu::densmap_gpu, m)?)?;
        m.add_function(wrap_pyfunction!(embed_gpu::tsne_gpu, m)?)?;
    }

    Ok(())
}
