//! Re-exports of commonly used types and traits for convenient glob importing.
//!
//! ```rust
//! use manifolds_rs::prelude::*;
//! ```

pub use crate::data::graph::UmapGraphParams;
pub use crate::data::init::{initialise_embedding, parse_initilisation};
pub use crate::data::nearest_neighbours::{
    parse_ann_search, run_ann_search, NearestNeighbourParams,
};
pub use crate::data::pacmap_pairs::PacmapPairs;
pub use crate::data::structures::CoordinateList;
pub use crate::data::synthetic::*;
pub use crate::errors::ManifoldsError;
pub use crate::training::mds_optimiser::{parse_mds_method, MdsMethod};
pub use crate::training::pacmap_optimiser::{parse_pacmap_optimiser, PacmapOptimParams};
pub use crate::training::tsne_optimiser::TsneOptimParams;
pub use crate::training::umap_optimisers::UmapOptimParams;
pub use crate::utils::diffusions::PhateDiffusionParams;
pub use crate::utils::diffusions::{parse_landmark_method, LandmarkMethod, PhateTime};
pub use crate::utils::math::landmark_von_neumann_entropy;
pub use crate::utils::potentials::calculate_potential;
pub use crate::utils::traits::ManifoldsFloat;

#[cfg(feature = "gpu")]
pub use crate::data::nearest_neighbours_gpu::*;
#[cfg(feature = "parametric")]
pub use crate::parametric::parametric_train::TrainParametricParams;
