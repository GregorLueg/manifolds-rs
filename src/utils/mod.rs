//! Contains key utility functions, like BH trees, FFT for tSNE, diffusion and
//! potentials for PHATE, sparse operations across different functions and also
//! macros

pub mod bh_tree;
pub mod diffusions;
pub mod macros;
pub mod math;
pub mod potentials;
pub mod sparse_ops;
pub mod traits;

#[cfg(feature = "fft_tsne")]
pub mod fft;
