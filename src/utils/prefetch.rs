//! Software prefetch hint for scattered reads in the hot gather loops.

/// Hint the CPU to pull the cache line holding `x` into L1.
///
/// Only emitted on aarch64, where it was measured to help scattered loads
/// once the working set outgrows the caches; a no-op elsewhere.
///
/// ### Params
///
/// * `x` - Any reference into the data that will be read soon
#[inline(always)]
pub(crate) fn prefetch_read<A>(x: &A) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `prfm` is a hint; it never faults and writes nothing, and the
    // address comes from a live reference.
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{0}]",
            in(reg) x as *const A,
            options(nostack, readonly, preserves_flags)
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = x;
}
