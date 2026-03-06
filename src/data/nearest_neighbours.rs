//! Module containing (approximate) nearest neighbour generation functions.

use ann_search_rs::hnsw::{HnswIndex, HnswState};
use ann_search_rs::nndescent::{ApplySortedUpdates, NNDescent, NNDescentQuery};
use ann_search_rs::prelude::*;

use ann_search_rs::*;
use faer::MatRef;
use num_traits::Float;
use rayon::prelude::*;
use std::default::Default;

/// Which search algorithm to use for the approximate nearest neighbour search
#[derive(Default)]
pub enum AnnSearch {
    /// Annoy
    Annoy,
    /// HNSW
    #[default]
    Hnsw,
    /// NNDescent
    NNDescent,
    /// BallTree
    BallTree,
    /// Exhaustive
    Exhaustive,
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
    /// NNDescent: optional beam search budget for querying.
    pub ef_budget: Option<usize>,
    /// BallTree: Proportions of N to search in the BallTree
    pub bt_budget: T,
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
    /// * `ef_budget` - Optional query budget.
    ///
    /// **BallTree**
    ///
    /// * `bt_budget` - Budget to use for BallTree
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
        // balltree
        bt_budget: T,
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
            bt_budget,
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
            // balltree
            bt_budget: T::from(0.1).unwrap(),
        }
    }
}

/// Parse the AnnSearch to use
///
/// ### Params
///
/// * `s` - String defineing the ANN search to use
///
/// ### Return
///
/// Option of AnnSearch
pub fn parse_ann_search(s: &str) -> Option<AnnSearch> {
    match s.to_lowercase().as_str() {
        "annoy" => Some(AnnSearch::Annoy),
        "hnsw" => Some(AnnSearch::Hnsw),
        "nndescent" => Some(AnnSearch::NNDescent),
        "balltree" => Some(AnnSearch::BallTree),
        "exhaustive" => Some(AnnSearch::Exhaustive),
        _ => None,
    }
}

/// Run the approximate nearest neighbour search prior to UMAP
///
/// ### Params
///
/// * `data` - The data with samples x features
/// * `k` - Number of neighbours to return
/// * `ann_type` - Which approximate nearest neighbour search to use. One of
///   `"annoy"`, `"hnsw"`, `"balltree"` or `"nndesccent"`.
/// * `params_nn` - The parameters for the approximate nearest neighbour search.
/// * `seed` - Seed for reproducibility.
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// `(knn_indices, knn_dist)` including self.
pub fn run_ann_search<T>(
    data: MatRef<T>,
    k: usize,
    ann_type: String,
    params_nn: &NearestNeighbourParams<T>,
    seed: usize,
    verbose: bool,
) -> (Vec<Vec<usize>>, Vec<Vec<T>>)
where
    T: AnnSearchFloat,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let ann_search = parse_ann_search(&ann_type).unwrap_or_default();

    let (knn_indices, knn_dist) = match ann_search {
        AnnSearch::Annoy => {
            let index =
                build_annoy_index(data, params_nn.dist_metric.clone(), params_nn.n_tree, seed);

            query_annoy_index(data, &index, k + 1, params_nn.search_budget, true, verbose)
        }
        AnnSearch::Hnsw => {
            let index = build_hnsw_index(
                data,
                params_nn.m,
                params_nn.ef_construction,
                &params_nn.dist_metric,
                seed,
                verbose,
            );

            query_hnsw_index(data, &index, k + 1, params_nn.ef_search, true, verbose)
        }
        AnnSearch::NNDescent => {
            let index = build_nndescent_index(
                data,
                &params_nn.dist_metric,
                params_nn.delta,
                params_nn.diversify_prob,
                None, // will default to the 30 that is usually used in NNDescent
                None,
                None,
                None,
                seed,
                verbose,
            );

            query_nndescent_index(data, &index, k + 1, params_nn.ef_budget, true, verbose)
        }
        AnnSearch::BallTree => {
            let index = build_balltree_index(data, params_nn.dist_metric.clone(), seed);

            let budget = (data.nrows() as f32 * params_nn.bt_budget.to_f32().unwrap()) as usize;

            query_balltree_index(data, &index, k + 1, Some(budget), true, verbose)
        }
        AnnSearch::Exhaustive => {
            let index = build_exhaustive_index(data, &params_nn.dist_metric);

            query_exhaustive_index(data, &index, k + 1, true, verbose)
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
