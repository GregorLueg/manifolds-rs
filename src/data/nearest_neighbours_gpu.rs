//! Module containing GPU-accelerated (approximate) nearest neighbour generation
//! functions used in the different embedding methods.

use ann_search_rs::cpu::nndescent::NNDescentQuery;
use ann_search_rs::gpu::nndescent_gpu::NNDescentGpu;
use ann_search_rs::prelude::*;
use ann_search_rs::{
    build_exhaustive_index_gpu, build_ivf_index_gpu, build_nndescent_index_gpu,
    query_exhaustive_index_gpu_self, query_ivf_index_gpu_self, query_nndescent_index_gpu_self,
};
use cubecl::prelude::*;
use faer::MatRef;
use rayon::prelude::*;

/////////////
// Helpers //
/////////////

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

impl<T> NearestNeighbourParamsGpu<T> {
    /// Generate a new instance
    ///
    /// ### Params
    ///
    /// General parameters
    ///
    /// * `dist_metric` - One of `"euclidean"` or `"cosine"`
    ///
    /// **IVF-GPU**
    ///
    /// * `n_list` - Number of clusters. Defaults to `sqrt(n)`.
    /// * `n_probes` - Number of clusters to probe during querying. Defaults
    ///   to `sqrt(n_list)`.
    ///
    /// **NNDescent-GPU**
    ///
    /// * `k` - Final node degree after CAGRA pruning. Defaults to `30`.
    /// * `k_build` - Initial node degree before pruning. Defaults to `k * 1.5`.
    /// * `n_tree` - Number of Annoy trees for kNN graph initialisation.
    /// * `delta` - Convergence threshold for NNDescent iterations.
    /// * `rho` - Sampling rate for NNDescent iterations.
    /// * `beam_width` - Beam width during querying. Auto-determined if `None`.
    /// * `max_beam_iters` - Maximum beam search iterations. Auto-determined
    ///   if `None`.
    /// * `n_entry_points` - Number of entry points for querying.
    ///   Auto-determined if `None`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dist_metric: String,
        // ivf-gpu
        n_list: Option<usize>,
        n_probes: Option<usize>,
        // nndescent-gpu
        k: Option<usize>,
        k_build: Option<usize>,
        n_tree: Option<usize>,
        delta: T,
        rho: Option<T>,
        beam_width: Option<usize>,
        max_beam_iters: Option<usize>,
        n_entry_points: Option<usize>,
    ) -> Self {
        Self {
            dist_metric,
            n_list,
            n_probes,
            k,
            k_build,
            n_tree,
            delta,
            rho,
            beam_width,
            max_beam_iters,
            n_entry_points,
        }
    }
}

impl<T> Default for NearestNeighbourParamsGpu<T>
where
    T: AnnSearchFloat,
{
    /// Returns sensible defaults for the GPU-accelerated nearest neighbour
    /// search.
    ///
    /// ### Returns
    ///
    /// Initialised self with sensible default parameters.
    fn default() -> Self {
        Self {
            dist_metric: "euclidean".to_string(),
            // ivf-gpu
            n_list: None,
            n_probes: None,
            // nndescent-gpu
            k: None,
            k_build: None,
            n_tree: None,
            delta: T::from(0.001).unwrap(),
            rho: None,
            beam_width: None,
            max_beam_iters: None,
            n_entry_points: None,
        }
    }
}

/// Parse the GPU ANN search variant to use
///
/// ### Params
///
/// * `s` - String identifying the GPU ANN search. One of `"exhaustive_gpu"`,
///   `"ivf_gpu"` or `"nndescent_gpu"`.
///
/// ### Returns
///
/// `Some(AnnSearchGpu)` if recognised, `None` otherwise.
pub fn parse_ann_search_gpu(s: &str) -> Option<AnnSearchGpu> {
    match s.to_lowercase().as_str() {
        "exhaustive_gpu" | "exhaustive" => Some(AnnSearchGpu::ExhaustiveGpu),
        "ivf_gpu" | "ivf" => Some(AnnSearchGpu::IvfGpu),
        "nndescent_gpu" | "nndescent" => Some(AnnSearchGpu::NNDescentGpu),
        _ => None,
    }
}

//////////
// Main //
//////////

/// Run the GPU-accelerated approximate nearest neighbour search
///
/// Mirrors `run_ann_search` but dispatches to GPU-backed indices. The returned
/// indices and distances exclude self, matching the CPU variant's contract.
///
/// ### Params
///
/// * `data` - The data matrix with samples x features
/// * `k` - Number of neighbours to return (excluding self)
/// * `ann_type` - Which GPU search to use. One of `"exhaustive_gpu"`,
///   `"ivf_gpu"` or `"nndescent_gpu"`.
/// * `params_nn` - Parameters for the GPU nearest neighbour search.
/// * `device` - The GPU device to use.
/// * `seed` - Seed for reproducibility.
/// * `verbose` - Controls verbosity.
///
/// ### Returns
///
/// `(knn_indices, knn_dist)` excluding self.
#[cfg(feature = "gpu")]
pub fn run_ann_search_gpu<T, R>(
    data: MatRef<T>,
    k: usize,
    ann_type: String,
    params_nn: &NearestNeighbourParamsGpu<T>,
    device: R::Device,
    seed: usize,
    verbose: bool,
) -> (Vec<Vec<usize>>, Vec<Vec<T>>)
where
    T: AnnSearchFloat + AnnSearchGpuFloat,
    R: Runtime,
    NNDescentGpu<T, R>: NNDescentQuery<T>,
{
    let ann_search = parse_ann_search_gpu(&ann_type).unwrap_or_default();

    let (knn_indices, knn_dist) = match ann_search {
        AnnSearchGpu::ExhaustiveGpu => {
            let index = build_exhaustive_index_gpu::<T, R>(data, &params_nn.dist_metric, device);

            query_exhaustive_index_gpu_self(&index, k + 1, true, verbose)
        }
        AnnSearchGpu::IvfGpu => {
            let index = build_ivf_index_gpu::<T, R>(
                data,
                params_nn.n_list,
                None,
                &params_nn.dist_metric,
                seed,
                verbose,
                device,
            );

            query_ivf_index_gpu_self(&index, k + 1, params_nn.n_probes, None, true, verbose)
        }
        AnnSearchGpu::NNDescentGpu => {
            let mut index = build_nndescent_index_gpu::<T, R>(
                data,
                &params_nn.dist_metric,
                params_nn.k,
                params_nn.k_build,
                None,
                params_nn.n_tree,
                params_nn.delta.to_f32().map(Some).unwrap_or(None),
                params_nn.rho.map(|r| r.to_f32().unwrap()),
                None,
                seed,
                verbose,
                false,
                device,
            );

            let query_params = CagraGpuSearchParams::new(
                params_nn.beam_width,
                params_nn.max_beam_iters,
                params_nn.n_entry_points,
            );

            query_nndescent_index_gpu_self(&mut index, k + 1, Some(query_params), true)
        }
    };

    let knn_dist = knn_dist.unwrap();

    // remove self (first element) from both indices and distances
    let knn_indices: Vec<Vec<usize>> = knn_indices
        .into_par_iter()
        .map(|mut v| v.drain(1..).collect())
        .collect();

    let knn_dist: Vec<Vec<T>> = knn_dist
        .into_par_iter()
        .map(|mut v| v.drain(1..).collect())
        .collect();

    (knn_indices, knn_dist)
}
