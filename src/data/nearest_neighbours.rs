//! Module containing (approximate) nearest neighbour generation functions.

use ann_search_rs::cpu::hnsw::{HnswIndex, HnswState};
use ann_search_rs::cpu::nndescent::{NNDescent, NNDescentQuery};
use ann_search_rs::prelude::*;
use ann_search_rs::utils::dist::{parse_ann_dist, Dist};
use ann_search_rs::utils::nndescent_utils::ApplySortedUpdates;

use ann_search_rs::*;
use faer::MatRef;
use num_traits::Float;
use rayon::prelude::*;
use std::default::Default;

use crate::prelude::*;

///////////////
// Constants //
///////////////

/// Graph degree `ann-search-rs` builds an NNDescent index with when given
/// `None`.
///
/// Mirrored rather than imported because the crate does not export it. It only
/// matters as a floor: widening the degree for extraction must never narrow it
/// below what the query path would have built.
const NNDESCENT_DEFAULT_DEGREE: usize = 30;

/////////////
// Helpers //
/////////////

/// Which search algorithm to use for the approximate nearest neighbour search
/// Default is set to Hnsw
#[derive(Default, Clone, Copy, Debug)]
pub enum AnnSearch {
    /// KmKnn
    #[default]
    KmKnn,
    /// HNSW
    Hnsw,
    /// NNDescent
    NNDescent,
    /// Annoy
    Annoy,
    /// IVF
    Ivf,
    /// BallTree
    BallTree,
    /// Exhaustive
    Exhaustive,
    /// Exhaustive on MLX (Apple Silicon, `mlx` feature)
    #[cfg(mlx_available)]
    ExhaustiveMlx,
    /// IVF on MLX (Apple Silicon, `mlx` feature)
    #[cfg(mlx_available)]
    IvfMlx,
    /// NNDescent on MLX (Apple Silicon, `mlx` feature)
    #[cfg(mlx_available)]
    NNDescentMlx,
}

/// Parameters for the nearest neighbour search
#[derive(Debug, Clone)]
pub struct NearestNeighbourParams<T> {
    /// Distance metric, one of `"euclidean"` or `"cosine"`
    pub dist_metric: String,
    /// Annoy: Number of trees to use to build the index. Defaults to `50` like
    /// the `uwot` package.
    pub n_tree: usize,
    /// Annoy: Optional search budget per tree. If not provided, defaults to
    /// `k * n_tree * 20` candidates.
    pub search_budget: Option<usize>,
    /// HNSW: connections per given layer to use
    pub m: usize,
    /// HNSW: construction budget
    pub ef_construction: usize,
    /// HNSW: search budget
    pub ef_search: usize,
    /// NNDescent: diversification probability after generation of the graph.
    pub diversify_prob: T,
    /// NNDescent: convergence criterium. If less than these percentage of
    /// neighbours have been udpated, the algorithm counts as converged.
    pub delta: T,
    /// NNDescent: optional beam search budget for querying. Ignored when
    /// `extract_knn` is set, since no search runs.
    pub ef_budget: Option<usize>,
    /// NNDescent: return the graph the descent already built instead of running
    /// a beam search over it. Defaults to `true`.
    ///
    /// A self-kNN query re-searches a graph that is already a kNN graph, which
    /// is the work the descent just did. Extraction skips it. The build degree
    /// is widened to cover `k` when this is set, since extraction cannot return
    /// more neighbours than the graph holds. No effect on any other backend.
    pub extract_knn: bool,
    /// BallTree: Proportions of N to search in the BallTree
    pub bt_budget: T,
    /// IVF: Number of lists, clusters to use. If not provided, will default
    /// to `sqrt(n)` lists.
    pub n_list: Option<usize>,
    /// IVF: Number of lists to probes. If not provided, will default to
    /// to `sqrt(n_list)` lists.
    pub n_probes: Option<usize>,
}

impl<T> NearestNeighbourParams<T> {
    /// Generate a new instance
    ///
    /// ### Params
    ///
    /// General parameters
    ///
    /// * `dist_metric` - One of `"euclidean"` or `"cosine"`
    ///
    /// **Annoy**
    ///
    /// * `dist_metric` - One of `"euclidean"` or `"cosine"`
    /// * `n_trees` - Number of trees to use to build the index. Defaults to `50`
    ///   like the `uwot` package.
    /// * `search_budget` - Optional search budget. The algorithm will set the
    ///   search budget to `10 * k * n_trees` if not provided.
    ///
    /// **HNSW**
    ///
    /// * `m` - Number of edges to generate per layer.
    /// * `ef_construction` - Budget during the construction of the index.
    /// * `ef_search` - Budget during the search of the index.
    ///
    /// **NN Descent**
    ///
    /// * `delta` - Early termination criterium.
    /// * `diversify_prob` - Diversifying probability at the end of the index
    ///   generation. Generates additional random edges which can improve the
    ///   Recall.
    /// * `ef_budget` - Optional query budget. Ignored when `extract_knn` is set.
    /// * `extract_knn` - Return the built graph rather than searching it.
    ///
    /// **BallTree**
    ///
    /// * `bt_budget` - Budget to use for BallTree
    ///
    /// **IVF**
    ///
    /// * `n_list` - Number of lists to use for IVF
    /// * `n_probes` - Number of lists/clusters to probe during querying
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dist_metric: String,
        // annoy
        n_tree: usize,
        search_budget: Option<usize>,
        // hnsw
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        // nndescent
        diversify_prob: T,
        delta: T,
        ef_budget: Option<usize>,
        extract_knn: bool,
        // balltree
        bt_budget: T,
        // ivf / kmknn
        n_list: Option<usize>,
        n_probes: Option<usize>,
    ) -> Self {
        Self {
            dist_metric,
            n_tree,
            search_budget,
            m,
            ef_construction,
            ef_search,
            diversify_prob,
            delta,
            ef_budget,
            extract_knn,
            bt_budget,
            n_list,
            n_probes,
        }
    }
}

impl<T> Default for NearestNeighbourParams<T>
where
    T: Float,
{
    /// Returns sensible defaults for the approximate nearest neighbour search
    ///
    /// ### Returns
    ///
    /// Initialised self with sensible default parameters.
    fn default() -> Self {
        Self {
            dist_metric: "euclidean".to_string(),
            // annoy
            n_tree: 50,
            search_budget: None,
            // hnsw
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            // nndescent
            diversify_prob: T::from(0.0).unwrap(),
            delta: T::from(0.001).unwrap(),
            ef_budget: None,
            extract_knn: true,
            // balltree
            bt_budget: T::from(0.1).unwrap(),
            // ivf
            n_list: None,
            n_probes: None,
        }
    }
}

/// Whether a metric name resolves to a squared distance in `ann-search-rs`.
///
/// Squared Euclidean is the only one; cosine and Manhattan already come back as
/// true distances.
///
/// ### Params
///
/// * `metric` - Metric name as handed to the ANN backend.
///
/// ### Returns
///
/// `true` when the backend's distances need a square root to become true
/// distances.
#[inline]
pub fn metric_returns_squared(metric: &str) -> bool {
    matches!(parse_ann_dist(metric), Some(Dist::SquaredEuclidean))
}

/// Cast matrix to `fp32` for the f32-only device backends (wgpu, MLX).
///
/// ### Params
///
/// * `data` - The MatRef to cast to `fp32`.
///
/// ### Returns
///
/// `Mat` in f32
#[cfg(any(feature = "gpu", mlx_available))]
pub(crate) fn cast_matrix_to_fp32<T>(data: MatRef<T>) -> faer::Mat<f32>
where
    T: AnnSearchFloat,
{
    faer::Mat::<f32>::from_fn(data.nrows(), data.ncols(), |i, j| {
        data[(i, j)].to_f32().unwrap()
    })
}

/// Cast the f32 distances an MLX backend hands back up to `T`.
///
/// ### Params
///
/// * `knn` - `(knn_indices, optional distances)` from an MLX query
///
/// ### Returns
///
/// The same rows with the distances in `T`.
#[cfg(mlx_available)]
fn cast_mlx_knn<T>(knn: KnnOptionResult<f32>) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
{
    let (indices, dist) = knn?;
    let dist = dist.map(|rows| {
        rows.into_par_iter()
            .map(|row| row.into_iter().map(|d| T::from(d).unwrap()).collect())
            .collect()
    });
    Ok((indices, dist))
}

/// Parse the AnnSearch to use
///
/// ### Params
///
/// * `s` - String defining the ANN search to use. `"exhaustive_mlx"`,
///   `"ivf_mlx"` and `"nndescent_mlx"` are only recognised when the MLX
///   backend is compiled in.
///
/// ### Return
///
/// Option of AnnSearch
pub fn parse_ann_search(s: &str) -> Option<AnnSearch> {
    match s.to_lowercase().as_str() {
        "annoy" => Some(AnnSearch::Annoy),
        "balltree" => Some(AnnSearch::BallTree),
        "exhaustive" => Some(AnnSearch::Exhaustive),
        "hnsw" => Some(AnnSearch::Hnsw),
        "ivf" => Some(AnnSearch::Ivf),
        "kmknn" => Some(AnnSearch::KmKnn),
        "nndescent" => Some(AnnSearch::NNDescent),
        #[cfg(mlx_available)]
        "exhaustive_mlx" => Some(AnnSearch::ExhaustiveMlx),
        #[cfg(mlx_available)]
        "ivf_mlx" => Some(AnnSearch::IvfMlx),
        #[cfg(mlx_available)]
        "nndescent_mlx" => Some(AnnSearch::NNDescentMlx),
        _ => None,
    }
}

//////////
// Main //
//////////

/// Run the approximate nearest neighbour search prior to UMAP
///
/// ### Params
///
/// * `data` - The data with samples x features
/// * `k` - Number of neighbours to return
/// * `ann_type` - Which approximate nearest neighbour search to use. One of
///   `"annoy"`, `"balltree"`, `"exhaustive"`, `"hnsw"`, `"ivf"`, `"kmknn"` or
///   `"nndescent"`. With the MLX backend compiled in also `"exhaustive_mlx"`,
///   `"ivf_mlx"` and `"nndescent_mlx"`, which run in f32 whatever `T` is.
/// * `params_nn` - The parameters for the approximate nearest neighbour search.
/// * `seed` - Seed for reproducibility.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `(knn_indices, knn_dist)` excluding self. Distances are true distances in
/// whatever metric was asked for: the Euclidean backends compute squared
/// distances for speed and this takes the square root before returning, so
/// every metric means the same thing to a caller.
pub fn run_ann_search<T>(
    data: MatRef<T>,
    k: usize,
    ann_type: String,
    params_nn: &NearestNeighbourParams<T>,
    seed: usize,
    verbose: usize,
) -> ManifoldsKnnResults<T>
where
    T: AnnSearchFloat,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let verbosity = parse_verbosity_level(verbose);

    let ann_search = parse_ann_search(&ann_type).unwrap_or_else(|| {
        println!(
            "Unrecognised approximate nearest neighbour method provided: {:?}. Default to KmKnn.",
            ann_type
        );
        AnnSearch::default()
    });

    let (knn_indices, knn_dist) = match ann_search {
        AnnSearch::Annoy => {
            let index = build_annoy_index(data, &params_nn.dist_metric, params_nn.n_tree, seed)?;

            query_annoy_self(
                &index,
                k + 1,
                params_nn.search_budget,
                true,
                verbosity.detailed_verbosity(),
            )?
        }
        AnnSearch::Hnsw => {
            let index = build_hnsw_index(
                data,
                params_nn.m,
                params_nn.ef_construction,
                &params_nn.dist_metric,
                seed,
                verbosity.detailed_verbosity(),
            );

            query_hnsw_self(
                &index,
                k + 1,
                params_nn.ef_search,
                true,
                verbosity.normal_verbosity(),
            )?
        }
        AnnSearch::NNDescent => {
            // Extraction can only hand back what the graph holds, so the degree
            // has to cover the request. `None` leaves the crate's own 30, which
            // is what the query path has always built.
            let graph_k = if params_nn.extract_knn {
                Some((k + 1).max(NNDESCENT_DEFAULT_DEGREE))
            } else {
                None
            };

            let index = build_nndescent_index(
                data,
                &params_nn.dist_metric,
                params_nn.delta,
                params_nn.diversify_prob,
                graph_k,
                None,
                None,
                None,
                seed,
                verbosity.detailed_verbosity(),
            )?;

            if params_nn.extract_knn {
                // `include_self` so the row shape matches the query path and the
                // shared self-removal below applies to both. Asking for the self
                // edge only to drop it costs one prepended entry per row.
                extract_nndescent_knn(&index, Some(k + 1), true, true)?
            } else {
                query_nndescent_self(
                    &index,
                    k + 1,
                    params_nn.ef_budget,
                    true,
                    verbosity.normal_verbosity(),
                )?
            }
        }
        AnnSearch::BallTree => {
            let index = build_balltree_index(data, &params_nn.dist_metric, seed)?;

            let budget = (data.nrows() as f32 * params_nn.bt_budget.to_f32().unwrap()) as usize;

            query_balltree_self(
                &index,
                k + 1,
                Some(budget),
                true,
                verbosity.normal_verbosity(),
            )?
        }
        AnnSearch::Exhaustive => {
            let index = build_exhaustive_index(data, &params_nn.dist_metric);

            query_exhaustive_self(&index, k + 1, true, verbosity.normal_verbosity())?
        }
        AnnSearch::Ivf => {
            let index = build_ivf_index(
                data,
                params_nn.n_list,
                None,
                &params_nn.dist_metric,
                seed,
                verbosity.detailed_verbosity(),
            )?;

            query_ivf_self(
                &index,
                k + 1,
                params_nn.n_probes,
                true,
                verbosity.normal_verbosity(),
            )?
        }
        AnnSearch::KmKnn => {
            let index = build_kmknn_index(
                data,
                &params_nn.dist_metric,
                params_nn.n_list,
                None,
                seed,
                verbosity.detailed_verbosity(),
            )?;

            query_kmknn_self(&index, k + 1, true, verbosity.normal_verbosity())?
        }
        #[cfg(mlx_available)]
        AnnSearch::ExhaustiveMlx => {
            let data_fp32 = cast_matrix_to_fp32(data);
            let index = build_exhaustive_index_mlx(data_fp32.as_ref(), &params_nn.dist_metric)?;

            cast_mlx_knn(query_exhaustive_index_mlx_self(
                &index,
                k + 1,
                true,
                verbosity.detailed_verbosity(),
            ))?
        }
        #[cfg(mlx_available)]
        AnnSearch::IvfMlx => {
            let data_fp32 = cast_matrix_to_fp32(data);
            let index = build_ivf_index_mlx(
                data_fp32.as_ref(),
                params_nn.n_list,
                None,
                &params_nn.dist_metric,
                seed,
                verbosity.detailed_verbosity(),
            )?;

            cast_mlx_knn(query_ivf_index_mlx_self(
                &index,
                k + 1,
                params_nn.n_probes,
                None,
                true,
                verbosity.normal_verbosity(),
            ))?
        }
        #[cfg(mlx_available)]
        AnnSearch::NNDescentMlx => {
            let data_fp32 = cast_matrix_to_fp32(data);
            let graph_k = Some((k + 1).max(NNDESCENT_DEFAULT_DEGREE));
            let delta = params_nn.delta.to_f32();

            if params_nn.extract_knn {
                // The raw graph build skips the CAGRA pruning the index needs
                // for beam search, which extraction never runs.
                let graph = build_knn_graph_mlx(
                    data_fp32.as_ref(),
                    &params_nn.dist_metric,
                    graph_k,
                    None,
                    None,
                    None,
                    delta,
                    None,
                    None,
                    seed,
                    verbosity.detailed_verbosity(),
                )?;
                cast_mlx_knn(Ok(graph.extract_knn(Some(k + 1), true, true)))?
            } else {
                let index = build_nndescent_index_mlx(
                    data_fp32.as_ref(),
                    &params_nn.dist_metric,
                    graph_k,
                    None,
                    None,
                    None,
                    delta,
                    None,
                    None,
                    seed,
                    verbosity.detailed_verbosity(),
                )?;
                cast_mlx_knn(query_nndescent_index_mlx_self(&index, k + 1, None, true))?
            }
        }
    };

    let knn_dist = knn_dist.unwrap();

    // Remove self by index rather than position: the f32 device backends can
    // rank an exact duplicate ahead of the point itself.
    let (knn_indices, mut knn_dist): (Vec<Vec<usize>>, Vec<Vec<T>>) = knn_indices
        .into_par_iter()
        .zip(knn_dist.into_par_iter())
        .enumerate()
        .map(|(i, (idx, dist))| {
            idx.into_iter()
                .zip(dist)
                .filter(|(j, _)| *j != i)
                .take(k)
                .unzip()
        })
        .unzip();

    // The Euclidean backends skip the square root, since it does not change the
    // ordering they sort on and is faster.
    if metric_returns_squared(&params_nn.dist_metric) {
        knn_dist
            .par_iter_mut()
            .for_each(|row| row.iter_mut().for_each(|d| *d = d.sqrt()));
    }

    Ok((knn_indices, knn_dist))
}
