//! The optimisers behind PHATE (MDS), t-SNE, UMAP (including the umap-learn 0.6
//! Adam), PaCMAP, LocalMAP, ForceAtlas2 and NCVis, with the GPU variants of the UMAP
//! (uwot-style and 0.6 Adam) and t-SNE ones.

pub mod fa2_optimiser;
pub mod mds_optimiser;
pub mod ncvis_optimiser;
pub mod pacmap_optimiser;
pub mod tsne_optimiser;
#[cfg(feature = "gpu")]
pub mod tsne_optimiser_gpu;
pub mod umap_csr;
pub mod umap_optimiser_v06;
pub mod umap_optimisers;

#[cfg(feature = "gpu")]
pub mod umap_optimiser_gpu;
#[cfg(feature = "gpu")]
pub mod umap_optimiser_v06_gpu;

/////////////
// Globals //
/////////////

/// Default beta1 value for Adam optimisation (Umap-specific)
pub const UMAP_BETA1: f64 = 0.5;
/// Default beta2 value for Adam optimisation (Umap-specific)
pub const UMAP_BETA2: f64 = 0.9;
/// Default beta1 value for Adam optimisation (classical Adam)
pub const BETA1: f64 = 0.9;
/// Default beta2 value for Adam optimisation (classical Adam)
pub const BETA2: f64 = 0.999;
/// Default eps for Adam optimisation
pub const EPS: f64 = 1e-7;
