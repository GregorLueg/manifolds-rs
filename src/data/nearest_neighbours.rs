use ann_search_rs::hnsw::{HnswIndex, HnswState};
use ann_search_rs::nndescent::{ApplySortedUpdates, NNDescent, NNDescentQuery};
use ann_search_rs::utils::dist::SimdDistance;

use ann_search_rs::*;
use faer::MatRef;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::default::Default;
use std::iter::Sum;

#[derive(Default)]
pub enum AnnSearch {
    #[default]
    /// Annoy
    Annoy,
    /// HNSW
    Hnsw,
    /// NNDescent
    NNDescent,
}

/// Parameters for the nearest neighbour search
///
/// ### Fields
///
/// * `dist_metric` - One of `"euclidean"` or `"cosine"`
///
/// **Annoy**-specific parameter**:
///
/// * `n_trees` - Number of trees to use to build the index. Defaults to `50`
///   like the `uwot` package.
/// * `search_budget` - Multiplier. The algorithm will set the search budget to
///   `10 * k * n_trees`
///
/// **HNSW**-specific parameter:
///
/// * `m` - Number of bidirectional connections per layer. Defaults to 16 based
///   on uwot R package.
/// * `ef_construction` - Size of candidate list during construction.
/// * `ef_search` - Size of candidate list during search (higher = better
///   recall, slower)
///
/// **NNDescent**-specific parameter
///
/// * `diversify_prob` - Diversifying probability at the end of the index
///   generation.
/// * `delta` - Early termination criterium
/// * `ef_budget` - Optional query budget.
#[derive(Debug, Clone)]
pub struct NearestNeighbourParams<T> {
    pub dist_metric: String,
    // annoy
    pub n_tree: usize,
    pub search_budget: Option<usize>,
    // hnsw
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    // nndescent
    pub diversify_prob: T,
    pub delta: T,
    pub ef_budget: Option<usize>,
}

impl<T> NearestNeighbourParams<T> {
    /// Generate a new instance
    ///
    /// ### Params
    ///
    /// * `dist_metric` - One of `"euclidean"` or `"cosine"`
    /// * `n_trees` - Number of trees to use to build the index. Defaults to `50`
    ///   like the `uwot` package.
    /// * `search_budget` - Optional search budget. The algorithm will set the
    ///   search budget to `10 * k * n_trees` if not provided.
    ///
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
///   `"annoy"`, `"hnsw"` or `"nndesccent"`.
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
) -> (Vec<Vec<usize>>, Vec<Vec<T>>)
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Default + Sum + SimdDistance,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let ann_search = parse_ann_search(&ann_type).unwrap_or_default();

    let (knn_indices, knn_dist) = match ann_search {
        AnnSearch::Annoy => {
            let index =
                build_annoy_index(data, params_nn.dist_metric.clone(), params_nn.n_tree, seed);

            query_annoy_index(data, &index, k + 1, params_nn.search_budget, true, false)
        }
        AnnSearch::Hnsw => {
            let index = build_hnsw_index(
                data,
                params_nn.m,
                params_nn.ef_construction,
                &params_nn.dist_metric,
                seed,
                false,
            );

            query_hnsw_index(data, &index, k + 1, params_nn.ef_search, true, false)
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
                false,
            );

            query_nndescent_index(data, &index, k + 1, params_nn.ef_budget, true, false)
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
