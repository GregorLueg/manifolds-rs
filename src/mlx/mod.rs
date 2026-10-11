//! Experimental MLX backend for the optimisers (Apple Silicon only).
//!
//! Not a cubecl runtime: cubecl has no MLX backend, so each optimiser is
//! re-expressed as custom Metal kernels and MLX ops through mlx-c. f32 only,
//! since MLX has no f64 on the GPU.

mod ffi;
pub mod umap_optimiser_mlx;
