//! Shared trait implementations for the manifolds-rs crate.

use ann_search_rs::prelude::*;
#[cfg(feature = "gpu")]
use cubecl::frontend::LaunchArg;
use faer_traits::{ComplexField, RealField};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::default::Default;
use std::fmt::Display;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Trait for floating-point types in manifolds-rs. Has common shared trait
/// boundaries
pub trait ManifoldsFloat:
    Float
    + FromPrimitive
    + ToPrimitive
    + Add<Output = Self>
    + AddAssign
    + Div<Output = Self>
    + DivAssign
    + Mul<Output = Self>
    + MulAssign
    + Sub<Output = Self>
    + SubAssign
    + Sync
    + Send
    + SimdDistance
    + Default
    + Sum
    + ComplexField
    + Display
    + RealField
    + AtomicFloat
{
}

impl<T> ManifoldsFloat for T where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Add<Output = Self>
        + AddAssign
        + Div<Output = Self>
        + DivAssign
        + Mul<Output = Self>
        + MulAssign
        + Sub<Output = Self>
        + SubAssign
        + Sync
        + Send
        + SimdDistance
        + Default
        + Sum
        + ComplexField
        + Display
        + RealField
        + AtomicFloat
{
}

/// Float stored in an atomic integer of the same width, for lock-free
/// (hogwild) updates of a shared buffer.
///
/// Only relaxed loads and stores, never read-modify-write: concurrent writers
/// may overwrite each other, which is the point. On x86-64 and aarch64 these
/// compile to plain loads and stores, but unlike racy raw-pointer writes they
/// are defined behaviour.
pub trait AtomicFloat: Sized {
    /// Atomic integer holding the float's bits.
    type Atomic: Send + Sync;

    /// Wrap a value.
    ///
    /// ### Params
    ///
    /// * `v` - Initial value
    ///
    /// ### Returns
    ///
    /// The atomic cell holding `v`.
    fn new_atomic(v: Self) -> Self::Atomic;

    /// Relaxed load.
    ///
    /// ### Params
    ///
    /// * `a` - The atomic cell
    ///
    /// ### Returns
    ///
    /// The value currently stored.
    fn load_relaxed(a: &Self::Atomic) -> Self;

    /// Relaxed store.
    ///
    /// ### Params
    ///
    /// * `a` - The atomic cell
    /// * `v` - Value to store
    fn store_relaxed(a: &Self::Atomic, v: Self);
}

impl AtomicFloat for f32 {
    type Atomic = AtomicU32;

    #[inline(always)]
    fn new_atomic(v: Self) -> Self::Atomic {
        AtomicU32::new(v.to_bits())
    }

    #[inline(always)]
    fn load_relaxed(a: &Self::Atomic) -> Self {
        f32::from_bits(a.load(Ordering::Relaxed))
    }

    #[inline(always)]
    fn store_relaxed(a: &Self::Atomic, v: Self) {
        a.store(v.to_bits(), Ordering::Relaxed)
    }
}

impl AtomicFloat for f64 {
    type Atomic = AtomicU64;

    #[inline(always)]
    fn new_atomic(v: Self) -> Self::Atomic {
        AtomicU64::new(v.to_bits())
    }

    #[inline(always)]
    fn load_relaxed(a: &Self::Atomic) -> Self {
        f64::from_bits(a.load(Ordering::Relaxed))
    }

    #[inline(always)]
    fn store_relaxed(a: &Self::Atomic, v: Self) {
        a.store(v.to_bits(), Ordering::Relaxed)
    }
}

/// Trait for floating-point types in manifolds-rs. Has common shared trait
/// boundaries. This version also supports GPU-accelerated methods via cubecl.
#[cfg(feature = "gpu")]
pub trait ManifoldsFloatGpu: ManifoldsFloat + CubeclFloat + LaunchArg<RuntimeArg = Self> {}

#[cfg(feature = "gpu")]
impl<T> ManifoldsFloatGpu for T where T: ManifoldsFloat + CubeclFloat + LaunchArg<RuntimeArg = T> {}
