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
use faer::{Mat, MatRef};
use rayon::prelude::*;

use crate::prelude::*;

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
    /// `None`, the ann-search-rs crate defaults to 30. When called via
    /// `construct_tsne_graph_gpu` a `None` value is backfilled to
    /// `3 * perplexity` so the CAGRA graph is sized for the tSNE query.
    pub k: Option<usize>,
    /// NNDescent-GPU: Build node degree. Initial node degree prior to pruning.
    /// If `None`, the ann-search-rs crate defaults to `max(k, floor(1.5 * k))`.
    /// When called via `construct_tsne_graph_gpu` a `None` value is backfilled
    /// to `2 * (3 * perplexity)`.
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

impl<T> NearestNeighbourParamsGpu<T>
where
    T: AnnSearchFloat,
{
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
    /// * `k` - Final node degree after CAGRA pruning. Defaults to `30` in
    ///   ann-search-rs; backfilled to `3 * perplexity` when called via
    ///   `construct_tsne_graph_gpu`.
    /// * `k_build` - Initial node degree before pruning. Defaults to
    ///   `max(k, floor(1.5 * k))` in ann-search-rs; backfilled to
    ///   `2 * (3 * perplexity)` when called via `construct_tsne_graph_gpu`.
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

    /// Cast parameters to a different float type. `usize` fields and
    /// `dist_metric` pass through unchanged.
    ///
    /// ### Returns
    ///
    /// `NearestNeighbourParamsGpu<U>` with all float fields converted via
    /// `NumCast`.
    pub fn cast<U>(&self) -> NearestNeighbourParamsGpu<U>
    where
        U: AnnSearchFloat,
    {
        let c = |v: T| U::from(v).unwrap();
        NearestNeighbourParamsGpu {
            dist_metric: self.dist_metric.clone(),
            n_list: self.n_list,
            n_probes: self.n_probes,
            k: self.k,
            k_build: self.k_build,
            n_tree: self.n_tree,
            delta: c(self.delta),
            rho: self.rho.map(c),
            beam_width: self.beam_width,
            max_beam_iters: self.max_beam_iters,
            n_entry_points: self.n_entry_points,
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

/// Cast matrix to `fp32`
///
/// ### Params
///
/// * `data` - The MatRef to cast to `fp32`.
///
/// ### Returns
///
/// `Mat` in f32
fn cast_matrix_to_fp32<T>(data: MatRef<T>) -> Mat<f32>
where
    T: AnnSearchFloat,
{
    Mat::<f32>::from_fn(data.nrows(), data.ncols(), |i, j| {
        data[(i, j)].to_f32().unwrap()
    })
}

//////////
// Main //
//////////

/// Run the GPU-accelerated approximate nearest neighbour search
///
/// Mirrors `run_ann_search` but dispatches to GPU-backed indices. The returned
/// indices and distances exclude self, matching the CPU variant's contract.
/// The function explicitly casts down to `fp32` internally to run on modern
/// GPUs which rarely support `fp64`.
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
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
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
    verbose: usize,
) -> ManifoldsKnnResults<T>
where
    T: AnnSearchFloat + AnnSearchGpuFloat,
    R: Runtime,
    NNDescentGpu<f32, R>: NNDescentQuery<f32>,
{
    let verbosity = parse_verbosity_level(verbose);

    let ann_search = parse_ann_search_gpu(&ann_type).unwrap_or_else(|| {
        println!("Unrecognised GPU-accelerated approximate nearest neighbour method provided: {:?}. Default to GPU IVF.", ann_type);
        AnnSearchGpu::default()
    });

    // wgpu does not support f64, hence, the kNN is always run in fp32
    let data_fp32 = cast_matrix_to_fp32(data);
    let params_fp32: NearestNeighbourParamsGpu<f32> = params_nn.cast();

    let (knn_indices_raw, knn_dist) = match ann_search {
        AnnSearchGpu::ExhaustiveGpu => {
            let index = build_exhaustive_index_gpu::<f32, R>(
                data_fp32.as_ref(),
                &params_nn.dist_metric,
                device,
            )?;

            query_exhaustive_index_gpu_self(&index, k + 1, true, verbosity.detailed_verbosity())?
        }
        AnnSearchGpu::IvfGpu => {
            let index = build_ivf_index_gpu::<f32, R>(
                data_fp32.as_ref(),
                params_nn.n_list,
                None,
                &params_nn.dist_metric,
                seed,
                verbosity.detailed_verbosity(),
                device,
            )?;

            query_ivf_index_gpu_self(
                &index,
                k + 1,
                params_nn.n_probes,
                None,
                true,
                verbosity.normal_verbosity(),
            )?
        }
        AnnSearchGpu::NNDescentGpu => {
            let mut index = build_nndescent_index_gpu::<f32, R>(
                data_fp32.as_ref(),
                &params_nn.dist_metric,
                params_nn.k,
                params_nn.k_build,
                None,
                params_nn.n_tree,
                Some(params_fp32.delta),
                params_fp32.rho,
                None,
                seed,
                verbosity.detailed_verbosity(),
                false,
                device,
            )?;

            // Mirror CagraGpuSearchParams::from_graph so beam_width/iters scale
            // with the requested k_out and the CAGRA graph degree. Wrapping the
            // params in Some(..) suppresses the crate's own from_graph fallback,
            // so we must backfill here or the raw BEAM_WIDTH=16 default caps
            // the returned neighbours at 16 - 1 = 15 after self-filtering.
            let k_graph = params_nn.k.unwrap_or(30);
            let scaled_bw = (k + 1).max(k_graph).max(16) * 2;
            let query_params = CagraGpuSearchParams::new(
                params_nn.beam_width.or(Some(scaled_bw)),
                params_nn.max_beam_iters.or(Some(scaled_bw * 3)),
                params_nn.n_entry_points,
                None,
            );

            query_nndescent_index_gpu_self(&mut index, k + 1, Some(query_params), true)?
        }
    };

    let knn_dist = knn_dist.unwrap();

    // remove self from indices/distances in a single pass and cast the f32
    // distances back up to T on the way out
    let (knn_indices, knn_dist): (Vec<Vec<usize>>, Vec<Vec<T>>) = knn_indices_raw
        .into_par_iter()
        .zip(knn_dist.into_par_iter())
        .enumerate()
        .map(|(i, (idx, dist))| {
            idx.into_iter()
                .zip(dist)
                .filter(|(j, _)| *j != i)
                .take(k)
                .map(|(j, d)| (j, T::from(d).unwrap()))
                .unzip()
        })
        .unzip();

    Ok((knn_indices, knn_dist))
}
