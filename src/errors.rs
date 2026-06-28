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
    // -- sparse errors --
    /// Error if the sparse matrix is not of CSR
    #[error("The matrix is not of CSR type. Please double check the inputs")]
    SparseMatrixIsNotCsr,
    // -- sparse errors --
    /// Error for sparse multiplication dimension mismatches
    #[error("The dimensions of the matrix do not support sparse multiplication (matrix a n_col: {n_col_a}; matrix b n_row: {n_row_b})")]
    SparseMatrixMultiplication {
        /// Number of columns in matrix a
        n_col_a: usize,
        /// Number of rows in matrix b
        n_row_b: usize,
    },
    /// The matrix is not square, but should be.
    #[error("The sparse matrix must be square")]
    SpareMatrixMustBeSquare,
    /// Power value is not positive
    #[error("The chosen power must be positive, but is {power}.")]
    PowerMustBePositive {
        /// Chosen power by the user
        power: usize,
    },
    // -- ann-search-rs --
    /// Propagate errors from the ann-search-rs crate
    #[error("Error from the ann-search-rs crate: {0}")]
    AnnSearchRsError(#[from] ann_search_rs::errors::AnnSearchErrors),
    // -- math errors --
    /// Error for SVDs from faer
    #[error("The faer SVD failed - please verify the data")]
    FaerSvdError,
    /// Error for Eigen decomposition from faer
    #[error("The faer Eigen decomposition failed - please verify the data")]
    FaerEigenError,
    // -- input errors --
    /// Error if a square matrix is not square
    #[error("The matrix needs to be square")]
    NotSquareMatrix,

    /// Error if the data is empty
    #[error("Empty data was parsed through - upstream error?")]
    NoData,

    /// Error if the data is empty
    #[error("UMAP: no edges to optimise - upstream error?")]
    NoGraphEdges,

    // -- parametric umap serialisation --
    /// Error when the model bytes cannot be serialised to disk format
    #[error("Failed to serialise parametric UMAP model: {0}")]
    ModelSerialisation(String),
    /// Error when the model bytes cannot be deserialised back into a model
    #[error("Failed to deserialise parametric UMAP model: {0}")]
    ModelDeserialisation(String),
    /// Error when the serialised model has an unsupported schema version
    #[error("Unsupported parametric UMAP model version: {version} (expected {expected})")]
    UnsupportedModelVersion {
        /// Version found in the payload
        version: u32,
        /// Version this build of the crate supports
        expected: u32,
    },
}
