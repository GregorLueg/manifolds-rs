//! Contains all of the different optimiser to fit PHATE, tSNE and UMAP.

pub mod mds_optimiser;
pub mod tsne_optimiser;
pub mod umap_optimisers;

use num_traits::Float;

////////////
// Params //
////////////

/// UMAP algorithm parameters
///
/// Controls the fuzzy simplicial set construction and graph symmetrisation.
#[derive(Clone, Debug)]
pub struct UmapGraphParams<T> {
    /// Convergence tolerance for smooth kNN distance binary search (typically
    /// 1e-5). Controls how precisely sigma values are computed.
    pub bandwidth: T,
    /// Number of nearest neighbours assumed to be at distance zero (typically
    /// 1.0). Allows for local manifold structure by treating the nearest
    /// neighbour(s) as having maximal membership strength.
    pub local_connectivity: T,
    /// Balance between fuzzy union and directed graph during symmetrisation
    /// (typically 1.0).
    pub mix_weight: T,
}

impl<T> Default for UmapGraphParams<T>
where
    T: Float,
{
    /// Returns sensible defaults for UMAP
    ///
    /// ### Returns
    ///
    /// * `bandwidth = 1e-5` - Tight convergence for sigma computation
    /// * `local_connectivity = 1.0` - Treat nearest neighbour as connected
    /// * `mix_weight = 1.0` - Standard symmetric fuzzy union
    fn default() -> Self {
        Self {
            local_connectivity: T::from(1.0).unwrap(),
            bandwidth: T::from(1e-5).unwrap(),
            mix_weight: T::from(1.0).unwrap(),
        }
    }
}
