//! Thread-pool control.
//!
//! The core is rayon throughout and uses the global pool, which sizes itself
//! from `RAYON_NUM_THREADS` or the core count. That is the right default, but a
//! caller running several embeddings from a job scheduler needs to cap it, and
//! rayon's own global pool can only be configured once per process.
//!
//! A private pool sidesteps that: [`set_num_threads`] swaps it, and [`run`]
//! installs whichever one is current around every call into the core. With no
//! pool set, [`run`] is a straight call and rayon's global pool applies.

use std::sync::{Arc, OnceLock, RwLock};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::ThreadPool;

///////////
// State //
///////////

/// The pool `run` installs, or `None` for rayon's global one.
///
/// `RwLock` rather than `Mutex` because every embedding call reads it and only
/// an explicit `set_num_threads` writes.
fn pool() -> &'static RwLock<Option<Arc<ThreadPool>>> {
    static POOL: OnceLock<RwLock<Option<Arc<ThreadPool>>>> = OnceLock::new();
    POOL.get_or_init(|| RwLock::new(None))
}

/////////////
// Running //
/////////////

/// Run `f` inside the configured pool.
///
/// Call this inside `Python::detach`, never outside it: the closure runs the
/// whole rayon fan-out, and holding the GIL across it would serialise every
/// worker against the interpreter.
///
/// ### Params
///
/// * `f` - The work, which must be `Send` since it may run on a pool thread.
///
/// ### Returns
///
/// Whatever `f` returns.
pub(crate) fn run<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    let current = pool().read().ok().and_then(|g| g.clone());
    match current {
        Some(p) => p.install(f),
        None => f(),
    }
}

/////////////////
// Python side //
/////////////////

/// Cap the threads the core may use.
///
/// ### Params
///
/// * `n` - Thread count. `0` restores rayon's global pool, which sizes itself
///   from `RAYON_NUM_THREADS` or the core count.
///
/// ### Returns
///
/// Nothing, or a `ValueError` if the pool could not be built.
#[pyfunction]
pub fn set_num_threads(n: usize) -> PyResult<()> {
    let built = if n == 0 {
        None
    } else {
        Some(Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| PyValueError::new_err(format!("could not build thread pool: {e}")))?,
        ))
    };
    let mut guard = pool()
        .write()
        .map_err(|_| PyValueError::new_err("thread pool lock is poisoned"))?;
    *guard = built;
    Ok(())
}

/// Threads the core will use for the next call.
///
/// ### Returns
///
/// The configured cap, or rayon's global pool size when none was set.
#[pyfunction]
pub fn num_threads() -> usize {
    let current = pool().read().ok().and_then(|g| g.clone());
    match current {
        Some(p) => p.current_num_threads(),
        None => rayon::current_num_threads(),
    }
}
