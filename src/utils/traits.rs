//! Shared trait implementations for the manifolds-rs crate.

use ann_search_rs::prelude::*;
use faer_traits::{ComplexField, RealField};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::default::Default;
use std::fmt::Display;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

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
{
}
