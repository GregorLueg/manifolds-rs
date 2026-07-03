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
pub use crate::parametric::model::TrainedUmapModel;
#[cfg(feature = "parametric")]
pub use crate::parametric::parametric_train::TrainParametricParams;
#[cfg(feature = "gpu")]
pub use crate::utils::traits::ManifoldsFloatGpu;

///////////
// Types //
///////////

/// The kNN search results in manifolds. If Ok, it's (indices, distances);
/// otherwise a [ManifoldsError].
pub type ManifoldsKnnResults<T> = Result<(Vec<Vec<usize>>, Vec<Vec<T>>), ManifoldsError>;

/// The Umap graph results. If Ok it's
/// `(coordinate_list of umap graph, indices, distances)`
pub type UmapGraphResults<T> =
    Result<(CoordinateList<T>, Vec<Vec<usize>>, Vec<Vec<T>>), ManifoldsError>;

/// The parametric UMAP results. If Ok, it's (embd, trained model); otherwise
/// an error
#[cfg(feature = "parametric")]
pub type ParametricUmapResults<B, T> =
    Result<(Vec<Vec<T>>, TrainedUmapModel<B, T>), ManifoldsError>;

///////////
// Enums //
///////////

/// Enum that controls verbosity
#[derive(Clone, Copy, Debug, Default)]
pub enum Verbosity {
    /// No verbosity at all
    #[default]
    Quiet,
    /// Normal levels of verbosity
    Normal,
    /// Detailed verbosity with increased messages
    Detailed,
}

impl Verbosity {
    /// Returns true if normal or detailed verbosity is set
    pub fn normal_verbosity(&self) -> bool {
        matches!(self, Verbosity::Normal | Verbosity::Detailed)
    }

    /// Returns true if detailed verbosity is set
    pub fn detailed_verbosity(&self) -> bool {
        matches!(self, Verbosity::Detailed)
    }
}

/// Parse verbosity leverl
///
/// ### Params
///
/// * `level` - If `1` returns [Verbosity::Normal], with `2`
///   [Verbosity::Detailed]
///
/// ### Returns
///
/// The desired [Verbosity] level.
pub fn parse_verbosity_level(level: usize) -> Verbosity {
    match level {
        0 => Verbosity::Quiet,
        1 => Verbosity::Normal,
        2 => Verbosity::Detailed,
        _ => Verbosity::Quiet,
    }
}
