//! Module containing GPU-accelerated (approximate) nearest neighbour generation
//! functions used in the different embedding methods.

/// Which search algorithm to use for the GPU-accelerated approximate nearest
/// neighbour search. Default is set to IVF GPU.
#[derive(Default)]
pub enum AnnSearchGpu {
    /// IvfGpu
    #[default]
    IvfGpu,
    /// NNDescentGpu
    NNDescentGpu,
    /// Exhaustive
    ExhaustiveGpu,
}

/// Parameters for the nearest neighbour search
#[derive(Debug, Clone)]
pub struct NearestNeighbourParamsGpu<T> {
    /// Distance metric, one of `"euclidean"` or `"cosine"`
    pub dist_metric: String,
    /// IVF-GPU: Number of lists, clusters to use. If not provided, will default
    /// to `sqrt(n)` lists.
    pub n_list: Option<usize>,
    /// IVF-GPU: Number of lists to probes. If not provided, will default to
    /// to `sqrt(n_list)` lists.
    pub n_probes: Option<usize>,
    /// NNDescent-GPU: Final node degree of the CAGRA graph after pruning. If
    /// `None`, defaults to 30.
    pub k: Option<usize>,
    /// NNDescent-GPU: Build node degree. Initial node degree prior to pruning.
    /// If `None`, defaults to `k * 1.5`.
    pub k_build: Option<usize>,
    /// NNDescent-GPU: Number of trees for the initialisation of the kNN graph
    /// prior to NNDescent.
    pub n_tree: Option<usize>,
    /// NNDescent-GPU: Termination criterium for the NNDescent iterations
    pub delta: T,
    /// NNDescent-GPU: Sampling rate for the NNDescent iterations.
    pub rho: Option<T>,
    /// NNDescent-GPU: Beam width during querying. If `None`, will be
    /// automatically determined.
    pub beam_width: Option<usize>,
    /// NNDescent-GPU: Iterations during querying. If `None`, will be
    /// automatically determined
    pub max_beam_iters: Option<usize>,
    /// NNDescent-GPU: Number of entry points during querying to use. If `None`,
    /// will be automatically determined.
    pub n_entry_points: Option<usize>,
}
