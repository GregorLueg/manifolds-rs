//! Errors that can occur in manifolds-rs

use thiserror::Error;

/// Errors that can be returned by manifolds-rs
#[derive(Debug, Error)]
pub enum ManifoldsError {
    // -- tSNE --
    /// Error when perplexity is set to high
    #[error("perplexity ({perplexity}) must be strictly less than the kNN size ({k})")]
    PerplexityTooLarge {
        /// Set perplexity parameter
        perplexity: f64,
        /// Found k-neighbours
        k: usize,
    },
    /// Dimensionality error for tSNE
    #[error("tSNE only supports n_dim = 2. Chosen dim = {n_dim}")]
    IncorrectDim {
        /// Set dimensions for tSNE
        n_dim: usize,
    },
    // -- math errors --
    /// Error for SVDs from faer
    #[error("The faer SVD failed - please verify the data")]
    FaerSvdError,
    /// Error for Eigen decomposition from faer
    #[error("The faer Eigen decomposition failed - please verify the data")]
    FaerEigenError,
}
