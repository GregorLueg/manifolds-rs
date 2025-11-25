use ann_search_rs::hnsw::{HnswIndex, HnswState};
use ann_search_rs::nndescent::{NNDescent, UpdateNeighbours};
use ann_search_rs::*;
use faer::MatRef;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::default::Default;

#[derive(Default)]
pub enum AnnSearch {
    /// Annoy
    #[default]
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
///   `search_budget * k * n_trees`
///
/// **HNSW**-specific parameter:
///
/// * `m` - Number of bidirectional connections per layer. Defaults to 16 based
///   on uwot R package.
/// * `ef_construction` - Size of candidate list during construction.
/// * `ef_search` - Multipler. Size of candidate list during search (higher =
///   better recall, slower). The total search will be `ef_search * k`.
///
/// **NNDescent**-specific parameter
///
/// * `max_iter` - Maximum iterations for the algorithm.
/// * `delta` - Early stop criterium for the algorithm. For example if set to
///   `0.001` the search stops when less than 0.1% of nodes change their
///   neighbours.
/// * `rho` - Sampling rate for the old neighbours. Will adaptively decrease
///   over time.
pub struct NearestNeighbourParams<T> {
    pub dist_metric: String,
    // annoy
    pub n_trees: usize,
    pub search_budget: usize,
    // hnsw
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    // nndescent
    pub max_iter: usize,
    pub delta: T,
    pub rho: T,
}

impl<T> Default for NearestNeighbourParams<T>
where
    T: Float,
{
    /// Returns sensible defaults for the approximate nearest neighbour search
    fn default() -> Self {
        Self {
            dist_metric: "cosine".to_string(),
            n_trees: 50,
            search_budget: 2,
            m: 16,
            ef_construction: 200,
            ef_search: 2,
            max_iter: 25,
            delta: T::from(0.001).unwrap(),
            rho: T::from(1.0).unwrap(),
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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Default,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: UpdateNeighbours<T>,
{
    let ann_search = parse_ann_search(&ann_type).unwrap_or_default();

    let (knn_indices, knn_dist) = match ann_search {
        AnnSearch::Annoy => {
            let index = build_annoy_index(data, params_nn.n_trees, seed);

            query_annoy_index(
                data,
                &index,
                &params_nn.dist_metric,
                k + 1,
                params_nn.search_budget * k * params_nn.n_trees,
                true,
                false,
            )
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

            query_hnsw_index(data, &index, k + 1, params_nn.ef_search * k, true, false)
        }
        AnnSearch::NNDescent => generate_knn_nndescent_with_dist(
            data,
            &params_nn.dist_metric,
            k + 1,
            params_nn.max_iter,
            params_nn.delta,
            params_nn.rho,
            seed,
            false,
            true,
        ),
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
