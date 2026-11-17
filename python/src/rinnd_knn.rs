//! rinnd's NN-Descent as the `"rinnd"` CPU kNN backend.
//!
//! Plugged into the core through [`ExternalKnn`], so the core keeps k
//! resolution, self removal and the Euclidean square root. Lives here because
//! `rinnd-core` is not on crates.io.

use manifolds_rs::prelude::{ExternalKnn, ManifoldsError, ManifoldsKnnResults, Verbosity};
use rinnd_core::distance::{Cosine, Distance, Manhattan, Metric, SquaredEuclidean};
use rinnd_core::index::NNDescentBuilder;

/// The backend handed to every [`manifolds_rs::prelude::NearestNeighbourParams`].
pub(crate) const RINND: ExternalKnn = ExternalKnn {
    name: "rinnd",
    run: rinnd_knn,
};

/// Fewest seeding trees rinnd's default picks.
const RINND_MIN_TREES: usize = 3;

/// Most seeding trees rinnd's default picks.
const RINND_MAX_TREES: usize = 12;

/// Default number of seeding trees: `clamp(round(2 * log10(n)), 3, 12)`.
///
/// rinnd's own default, which follows PyNNDescent. Fewer trees than the
/// ann-search-rs default; rinnd leans on the descent rather than the forest.
///
/// ### Params
///
/// * `n` - Number of samples
///
/// ### Returns
///
/// Number of trees.
fn default_rinnd_trees(n: usize) -> usize {
    ((2.0 * (n as f64).log10()).round() as usize).clamp(RINND_MIN_TREES, RINND_MAX_TREES)
}

/// Build a kNN graph with rinnd, as an [`manifolds_rs::prelude::ExternalKnnFn`].
///
/// ### Params
///
/// * `data` - Row-major samples
/// * `n` - Number of samples
/// * `dim` - Number of features
/// * `k` - Neighbours per row, self included
/// * `metric` - `"euclidean"` / `"l2"`, `"cosine"` or `"manhattan"` / `"l1"`
/// * `n_trees` - Seeding trees, `None` for [`default_rinnd_trees`]
/// * `delta` - Convergence threshold
/// * `seed` - Random seed
/// * `verbosity` - rinnd prints its progress at detailed verbosity
///
/// ### Returns
///
/// Sorted rows of indices and distances, squared for Euclidean. Unfilled slots
/// are dropped, so a row can come back short.
#[allow(clippy::too_many_arguments)]
fn rinnd_knn(
    data: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    metric: &str,
    n_trees: Option<usize>,
    delta: f64,
    seed: usize,
    verbosity: Verbosity,
) -> ManifoldsKnnResults<f32> {
    let builder = NNDescentBuilder::new(data, n, dim)
        .n_neighbors(k)
        .n_trees(n_trees.unwrap_or_else(|| default_rinnd_trees(n)))
        .delta(delta as f32)
        .random_seed(seed as u64)
        .verbose(verbosity.detailed_verbosity());

    match metric.to_lowercase().as_str() {
        "euclidean" | "l2" => Ok(build(
            builder.metric(Metric::SquaredEuclidean),
            SquaredEuclidean,
        )),
        "cosine" => Ok(build(builder.metric(Metric::Cosine), Cosine)),
        "manhattan" | "l1" => Ok(build(builder.metric(Metric::Manhattan), Manhattan)),
        _ => Err(ManifoldsError::MetricNotSupported {
            method: "rinnd",
            metric: metric.to_string(),
        }),
    }
}

/// Run the graph-only build and unpack it into rows.
///
/// ### Params
///
/// * `builder` - Configured builder, metric set
/// * `distance` - Distance matching the builder's metric
///
/// ### Returns
///
/// Rows of indices and distances, `-1` padding dropped.
fn build<D: Distance<f32>>(
    builder: NNDescentBuilder<'_>,
    distance: D,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let graph = builder.build_graph_with_distance(distance, None).graph;
    let k = graph.k.max(1);
    graph
        .indices
        .chunks_exact(k)
        .zip(graph.distances.chunks_exact(k))
        .map(|(idx, dist)| {
            idx.iter()
                .zip(dist)
                .filter(|(&j, _)| j >= 0)
                .map(|(&j, &d)| (j as usize, d))
                .unzip()
        })
        .unzip()
}
