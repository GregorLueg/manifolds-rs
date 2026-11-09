//! Software prefetch hint for scattered reads in the hot gather loops.

/// Hint the CPU to pull the cache line holding `x` into L1.
///
/// Emitted on aarch64, where it was measured to help scattered loads once the
/// working set outgrows the caches, and on x86_64 (unmeasured there). A no-op
/// elsewhere.
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
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `prefetcht0` is a hint and never faults; SSE is baseline on x86_64.
    unsafe {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch::<_MM_HINT_T0>(x as *const A as *const i8);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = x;
}
