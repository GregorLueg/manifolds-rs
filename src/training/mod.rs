//! Contains all of the different optimiser to fit PHATE, tSNE and UMAP.

pub mod mds_optimiser;
pub mod pacmap_optimiser;
pub mod tsne_optimiser;
pub mod umap_optimisers;

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
