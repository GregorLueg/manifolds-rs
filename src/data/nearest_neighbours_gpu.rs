//! Module containing GPU-accelerated (approximate) nearest neighbour generation
//! functions used in the different embedding methods.

use ann_search_rs::prelude::*;
use ann_search_rs::{
    build_exhaustive_index_gpu, build_ivf_index_gpu, build_knn_graph_nndescent_gpu,
    query_exhaustive_index_gpu_self, query_ivf_index_gpu_self,
};
#[cfg(mlx_available)]
use ann_search_rs::{
    build_exhaustive_index_mlx, build_ivf_index_mlx, build_knn_graph_mlx,
    query_exhaustive_index_mlx_self, query_ivf_index_mlx_self,
};
use cubecl::Device;
use faer::MatRef;
use rayon::prelude::*;

use crate::data::nearest_neighbours::{cast_matrix_to_fp32, knn_graph_to_rows};
use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Which search algorithm to use for the GPU-accelerated approximate nearest
/// neighbour search. Default is set to NNDescentGPU.
#[derive(Default)]
pub enum AnnSearchGpu {
    /// NNDescentGpu
    #[default]
    NNDescentGpu,
    /// IvfGpu
    IvfGpu,
    /// Exhaustive
    ExhaustiveGpu,
    /// NNDescent on MLX (Apple Silicon)
    #[cfg(mlx_available)]
    NNDescentMlx,
    /// IVF on MLX (Apple Silicon)
    #[cfg(mlx_available)]
    IvfMlx,
    /// Exhaustive on MLX (Apple Silicon)
    #[cfg(mlx_available)]
    ExhaustiveMlx,
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
    /// NNDescent-GPU: Internal working degree of the descent. If `None`, the
    /// ann-search-rs crate defaults to `1.5 * k` of the requested `k`.
    pub k_build: Option<usize>,
    /// NNDescent-GPU: Number of trees for the initialisation of the kNN graph
    /// prior to NNDescent.
    pub n_tree: Option<usize>,
    /// NNDescent-GPU: Termination criterium for the NNDescent iterations
    pub delta: T,
    /// NNDescent-GPU: Sampling rate for the NNDescent iterations.
    pub rho: Option<T>,
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
    /// * `k_build` - Internal working degree of the descent. Defaults to
    ///   `1.5 * k` in ann-search-rs.
    /// * `n_tree` - Number of trees for kNN graph initialisation.
    /// * `delta` - Convergence threshold for NNDescent iterations.
    /// * `rho` - Sampling rate for NNDescent iterations.
    pub fn new(
        dist_metric: String,
        // ivf-gpu
        n_list: Option<usize>,
        n_probes: Option<usize>,
        // nndescent-gpu
        k_build: Option<usize>,
        n_tree: Option<usize>,
        delta: T,
        rho: Option<T>,
    ) -> Self {
        Self {
            dist_metric,
            n_list,
            n_probes,
            k_build,
            n_tree,
            delta,
            rho,
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
            k_build: self.k_build,
            n_tree: self.n_tree,
            delta: c(self.delta),
            rho: self.rho.map(c),
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
            k_build: None,
            n_tree: None,
            delta: T::from(0.001).unwrap(),
            rho: None,
        }
    }
}

/// Parse the GPU ANN search variant to use
///
/// ### Params
///
/// * `s` - String identifying the GPU ANN search. One of `"exhaustive_gpu"`,
///   `"ivf_gpu"` or `"nndescent_gpu"`. With the MLX backend compiled in also
///   `"exhaustive_mlx"`, `"ivf_mlx"` and `"nndescent_mlx"`.
///
/// ### Returns
///
/// `Some(AnnSearchGpu)` if recognised, `None` otherwise.
pub fn parse_ann_search_gpu(s: &str) -> Option<AnnSearchGpu> {
    match s.to_lowercase().as_str() {
        "exhaustive_gpu" | "exhaustive" => Some(AnnSearchGpu::ExhaustiveGpu),
        "ivf_gpu" | "ivf" => Some(AnnSearchGpu::IvfGpu),
        "nndescent_gpu" | "nndescent" => Some(AnnSearchGpu::NNDescentGpu),
        #[cfg(mlx_available)]
        "exhaustive_mlx" => Some(AnnSearchGpu::ExhaustiveMlx),
        #[cfg(mlx_available)]
        "ivf_mlx" => Some(AnnSearchGpu::IvfMlx),
        #[cfg(mlx_available)]
        "nndescent_mlx" => Some(AnnSearchGpu::NNDescentMlx),
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
/// `(knn_indices, knn_dist)` excluding self. Distances are true distances, as
/// in [`run_ann_search`]: the Euclidean backends compute squared distances for
/// speed and the square root is taken before returning.
#[cfg(feature = "gpu")]
pub fn run_ann_search_gpu<T>(
    data: MatRef<T>,
    k: usize,
    ann_type: String,
    params_nn: &NearestNeighbourParamsGpu<T>,
    device: Device,
    seed: usize,
    verbose: usize,
) -> ManifoldsKnnResults<T>
where
    T: AnnSearchFloat + CubeclFloat,
{
    let verbosity = parse_verbosity_level(verbose);

    let ann_search = parse_ann_search_gpu(&ann_type).unwrap_or_else(|| {
        println!("Unrecognised GPU-accelerated approximate nearest neighbour method provided: {:?}. Default to GPU NNDescent.", ann_type);
        AnnSearchGpu::default()
    });

    // wgpu does not support f64, hence, the kNN is always run in fp32
    let data_fp32 = cast_matrix_to_fp32(data);
    let params_fp32: NearestNeighbourParamsGpu<f32> = params_nn.cast();

    let (knn_indices_raw, knn_dist) = match ann_search {
        AnnSearchGpu::ExhaustiveGpu => {
            let index = build_exhaustive_index_gpu::<f32>(
                data_fp32.as_ref(),
                &params_nn.dist_metric,
                device,
                verbosity,
            )?;

            query_exhaustive_index_gpu_self(&index, k + 1, true, verbosity)?
        }
        AnnSearchGpu::IvfGpu => {
            let index = build_ivf_index_gpu::<f32>(
                data_fp32.as_ref(),
                params_nn.n_list,
                None,
                &params_nn.dist_metric,
                seed,
                verbosity,
                device,
            )?;

            query_ivf_index_gpu_self(&index, k + 1, params_nn.n_probes, None, true, verbosity)?
        }
        AnnSearchGpu::NNDescentGpu => {
            let knn_params = NNDescentGpuKnnParams::new(
                k,
                params_nn.k_build,
                None,
                params_nn.n_tree,
                Some(params_fp32.delta),
                params_fp32.rho,
                None,
                false,
                true,
            );
            let graph = build_knn_graph_nndescent_gpu::<f32>(
                data_fp32.as_ref(),
                &params_nn.dist_metric,
                Some(knn_params),
                None,
                seed,
                verbosity,
                device,
            )?;

            knn_graph_to_rows(graph)
        }
        #[cfg(mlx_available)]
        AnnSearchGpu::ExhaustiveMlx => {
            let index =
                build_exhaustive_index_mlx(data_fp32.as_ref(), &params_nn.dist_metric, verbosity)?;

            query_exhaustive_index_mlx_self(&index, k + 1, true, verbosity)?
        }
        #[cfg(mlx_available)]
        AnnSearchGpu::IvfMlx => {
            let index = build_ivf_index_mlx(
                data_fp32.as_ref(),
                params_nn.n_list,
                None,
                &params_nn.dist_metric,
                seed,
                verbosity,
            )?;

            query_ivf_index_mlx_self(&index, k + 1, params_nn.n_probes, None, true, verbosity)?
        }
        #[cfg(mlx_available)]
        AnnSearchGpu::NNDescentMlx => {
            let graph = build_knn_graph_mlx(
                data_fp32.as_ref(),
                &params_nn.dist_metric,
                Some(k),
                params_nn.k_build,
                None,
                params_nn.n_tree,
                Some(params_fp32.delta),
                params_fp32.rho,
                None,
                seed,
                verbosity,
            )?;
            graph.extract_knn(Some(k), false, true)
        }
    };

    let knn_dist = knn_dist.unwrap();

    // The Euclidean backends skip the square root, since it does not change the
    // ordering they sort on. Callers do care; see `run_ann_search`.
    let rooted = metric_returns_squared(&params_nn.dist_metric);

    // remove self from indices/distances in a single pass and cast the f32
    // distances back up to T on the way out
    let (knn_indices, knn_dist): (Vec<Vec<usize>>, Vec<Vec<T>>) = knn_indices_raw
        .into_par_iter()
        .zip(knn_dist.into_par_iter())
        .enumerate()
        .map(|(i, (idx, dist))| {
            // sized up front: a filtered iterator has no size hint and would
            // grow each row to the next power of two
            let mut out_idx = Vec::with_capacity(k);
            let mut out_dist = Vec::with_capacity(k);
            for (j, d) in idx.into_iter().zip(dist) {
                if j == i {
                    continue;
                }
                if out_idx.len() == k {
                    break;
                }
                let d = if rooted { d.sqrt() } else { d };
                out_idx.push(j);
                out_dist.push(T::from(d).unwrap());
            }
            (out_idx, out_dist)
        })
        .unzip();

    Ok((knn_indices, knn_dist))
}
