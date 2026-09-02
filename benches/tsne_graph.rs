//! Timing harness for the tSNE graph construction pipeline.
//!
//! Gated behind `large_scale_diagnostics` so nothing here compiles in CI. The
//! kNN search runs once per configuration and is reused across every stage, so
//! the reported numbers are the graph stages alone rather than the ANN search.
//!
//! ```sh
//! cargo bench --features large_scale_diagnostics --bench tsne_graph
//! ```

use ann_search_rs::cpu::hnsw::{HnswIndex, HnswState};
use ann_search_rs::cpu::nndescent::{NNDescent, NNDescentQuery};
use ann_search_rs::utils::nndescent_utils::ApplySortedUpdates;
use faer::Mat;
use manifolds_rs::data::graph::{gaussian_knn_affinities, symmetrise_affinities_tsne};
use manifolds_rs::data::synthetic::generate_clustered_data;
use manifolds_rs::prelude::*;
use std::time::Instant;

/// Binary-search iteration cap, matching the value `construct_tsne_graph`
/// hard-codes.
const MAX_ITER: usize = 200;

/// Entropy convergence tolerance, matching `construct_tsne_graph`.
const TOL: f64 = 1e-5;

/// Dataset sizes to sweep.
const SAMPLE_SIZES: [usize; 3] = [10_000, 50_000, 200_000];

/// Perplexities to sweep. tSNE sizes `k` at `3 * perplexity`, so 30 and 50 give
/// neighbour counts of 90 and 150 rather than the 15/50 the UMAP harness uses.
const PERPLEXITIES: [f64; 2] = [30.0, 50.0];

/// Ambient dimensionality of the synthetic data.
const N_DIM: usize = 50;

/// Number of clusters in the synthetic data.
const N_CLUSTERS: usize = 5;

/// Seed for the synthetic data and the ANN search.
const SEED: usize = 42;

/////////////
// Helpers //
/////////////

/// Narrow a matrix to `f32`.
///
/// ### Params
///
/// * `input` - Source matrix
///
/// ### Returns
///
/// The same matrix in single precision.
fn mat_to_f32(input: &Mat<f64>) -> Mat<f32> {
    Mat::from_fn(input.nrows(), input.ncols(), |i, j| input[(i, j)] as f32)
}

/// Time every stage of the graph pipeline for one dataset and perplexity.
///
/// Runs the ANN search once, then walks `gaussian_knn_affinities` ->
/// `symmetrise_affinities_tsne`, printing the elapsed time and the edge count
/// after each stage.
///
/// ### Params
///
/// * `data` - Data matrix
/// * `perplexity` - Target perplexity; `k` is derived as `3 * perplexity`
/// * `label` - Prefix printed on every line, identifying the configuration
fn bench_pipeline<T>(data: &Mat<T>, perplexity: f64, label: &str)
where
    T: ManifoldsFloat,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let nn_params = NearestNeighbourParams::default();
    let k = (perplexity * 3.0) as usize;

    let start = Instant::now();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "kmknn".to_string(), &nn_params, SEED, 0)
            .expect("ANN search failed");
    println!(
        "{label} ann_search               {:>10.2?}",
        start.elapsed()
    );

    let start = Instant::now();
    let graph = gaussian_knn_affinities(
        &knn_indices,
        &knn_dist,
        T::from_f64(perplexity).unwrap(),
        T::from_f64(TOL).unwrap(),
        MAX_ITER,
    )
    .expect("affinity calibration failed");
    println!(
        "{label} gaussian_affinities      {:>10.2?}  ({} edges)",
        start.elapsed(),
        graph.get_size()
    );

    let start = Instant::now();
    let graph = symmetrise_affinities_tsne(graph);
    println!(
        "{label} symmetrise_affinities    {:>10.2?}  ({} edges)",
        start.elapsed(),
        graph.get_size()
    );
}

//////////
// Main //
//////////

/// Sweep the pipeline in both precisions across every size and perplexity.
fn main() {
    for n_samples in SAMPLE_SIZES {
        let (data, _) = generate_clustered_data(n_samples, N_DIM, N_CLUSTERS, SEED as u64);
        let data_f32 = mat_to_f32(&data);

        for perplexity in PERPLEXITIES {
            let k = (perplexity * 3.0) as usize;
            bench_pipeline(
                &data,
                perplexity,
                &format!("[f64 n={n_samples:>6} k={k:>3}]"),
            );
            bench_pipeline(
                &data_f32,
                perplexity,
                &format!("[f32 n={n_samples:>6} k={k:>3}]"),
            );
        }
    }
}
