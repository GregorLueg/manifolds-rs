pub mod tsne_optimiser;
pub mod umap_optimisers;

use num_traits::Float;

////////////
// Params //
////////////

/// UMAP algorithm parameters
///
/// Controls the fuzzy simplicial set construction and graph symmetrisation.
///
/// ### Fields
///
/// * `bandwidth` - Convergence tolerance for smooth kNN distance binary search
///   (typically 1e-5). Controls how precisely sigma values are computed.
/// * `local_connectivity` - Number of nearest neighbours assumed to be at
///   distance zero (typically 1.0). Allows for local manifold structure by
///   treating the nearest neighbour(s) as having maximal membership strength.
/// * `mix_weight` - Balance between fuzzy union and directed graph during
///   symmetrisation (typically 1.0).
#[derive(Clone, Debug)]
pub struct UmapGraphParams<T> {
    pub bandwidth: T,
    pub local_connectivity: T,
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
