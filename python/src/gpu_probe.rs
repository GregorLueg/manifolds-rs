//! Whether this build can actually reach a GPU.
//!
//! Two separate questions, and callers care about both:
//!
//! - Was the extension compiled with the `gpu` feature? Fixed at wheel-build
//!   time; a Python extra cannot change it, because extras only add Python
//!   requirements and never swap the compiled artefact.
//! - Is there an adapter on this machine? Only answerable at runtime, and the
//!   answer is legitimately "no" on a headless box or in a container without
//!   the Vulkan loader.
//!
//! [`gpu_available`] answers both at once, since a caller choosing between
//! `UMAP` and `UMAPGpu` has no use for the distinction.

use pyo3::prelude::*;

/// Whether the GPU entry points can be used here.
///
/// ### Returns
///
/// `True` only when this build has the `gpu` feature **and** wgpu resolves an
/// adapter. Safe to call on any machine.
///
/// ### Note
///
/// Acquiring a client panics rather than erroring when no adapter is found, so
/// the probe catches it. That is sound here because the release profile is
/// pinned to `panic = "unwind"`, which pyo3 requires anyway. The panic hook is
/// silenced for the duration, otherwise merely asking the question prints a
/// backtrace to stderr.
#[pyfunction]
pub fn gpu_available(py: Python<'_>) -> bool {
    py.detach(probe)
}

/// Try to stand up a wgpu client on the default device.
///
/// ### Returns
///
/// `true` if a client came back, `false` if the attempt panicked.
#[cfg(feature = "gpu")]
fn probe() -> bool {
    use cubecl::prelude::Runtime;

    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let ok = std::panic::catch_unwind(|| {
        let device = crate::embed_gpu::default_device();
        let _ = <crate::embed_gpu::Rt as Runtime>::client(&device);
    })
    .is_ok();
    std::panic::set_hook(hook);
    ok
}

/// Stub for builds without the `gpu` feature.
///
/// ### Returns
///
/// Always `false`.
#[cfg(not(feature = "gpu"))]
fn probe() -> bool {
    false
}
