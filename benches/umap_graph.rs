//! Timing harness for the UMAP graph construction pipeline.
//!
//! Gated behind `large_scale_diagnostics` so nothing here compiles in CI. The
//! kNN search runs once per configuration and is reused across every stage, so
//! the reported numbers are the graph stages alone rather than the ANN search.
//!
//! ```sh
//! cargo bench --features large_scale_diagnostics --bench umap_graph
//! ```

use ann_search_rs::cpu::hnsw::{HnswIndex, HnswState};
use ann_search_rs::cpu::nndescent::{NNDescent, NNDescentQuery};
use ann_search_rs::utils::nndescent_utils::ApplySortedUpdates;
use faer::Mat;
use manifolds_rs::data::graph::{
    coo_to_adjacency_list, filter_weak_edges, knn_to_coo, smooth_knn_dist, symmetrise_graph,
    UmapGraphParams,
};
use manifolds_rs::data::synthetic::generate_clustered_data;
use manifolds_rs::prelude::*;
use std::time::Instant;

/// Binary-search iterations for the smooth kNN distances, matching the value
/// `construct_umap_graph` hard-codes.
const N_ITER: usize = 64;

/// Epoch count used for the weak-edge threshold, matching the UMAP default for
/// datasets above 10k points.
const N_EPOCHS: usize = 200;

/// Dataset sizes to sweep.
const SAMPLE_SIZES: [usize; 3] = [10_000, 50_000, 200_000];

/// Neighbour counts to sweep. 15 is the UMAP default; 50 is a common choice for
/// larger datasets and makes the per-row inner loops long enough to matter.
const K_VALUES: [usize; 2] = [15, 50];

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

/// Time every stage of the graph pipeline for one dataset and neighbour count.
///
/// Runs the ANN search once, then walks `smooth_knn_dist` -> `knn_to_coo` ->
/// `symmetrise_graph` -> `filter_weak_edges` -> `coo_to_adjacency_list`,
/// printing the elapsed time and the edge count after each stage.
///
/// ### Params
///
/// * `data` - Data matrix
/// * `k` - Number of nearest neighbours
/// * `label` - Prefix printed on every line, identifying the configuration
fn bench_pipeline<T>(data: &Mat<T>, k: usize, label: &str)
where
    T: ManifoldsFloat,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let nn_params = NearestNeighbourParams::default();
    let umap_params = UmapGraphParams::<T>::default();

    let start = Instant::now();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "kmknn".to_string(), &nn_params, SEED, 0)
            .expect("ANN search failed");
    println!("{label} ann_search           {:>10.2?}", start.elapsed());

    let start = Instant::now();
    let (sigma, rho) = smooth_knn_dist(
        &knn_dist,
        knn_dist[0].len(),
        umap_params.local_connectivity,
        umap_params.bandwidth,
        N_ITER,
    );
    println!("{label} smooth_knn_dist      {:>10.2?}", start.elapsed());

    let start = Instant::now();
    let graph = knn_to_coo(&knn_indices, &knn_dist, &sigma, &rho);
    println!(
        "{label} knn_to_coo           {:>10.2?}  ({} edges)",
        start.elapsed(),
        graph.get_size()
    );

    let start = Instant::now();
    let graph = symmetrise_graph(graph, umap_params.mix_weight);
    println!(
        "{label} symmetrise_graph     {:>10.2?}  ({} edges)",
        start.elapsed(),
        graph.get_size()
    );

    let start = Instant::now();
    let graph = filter_weak_edges(graph, N_EPOCHS, 0);
    println!(
        "{label} filter_weak_edges    {:>10.2?}  ({} edges)",
        start.elapsed(),
        graph.get_size()
    );

    let start = Instant::now();
    let adj = coo_to_adjacency_list(&graph);
    println!(
        "{label} coo_to_adjacency     {:>10.2?}  ({} nodes)",
        start.elapsed(),
        adj.len()
    );
}

//////////
// Main //
//////////

/// Sweep the pipeline in both precisions across every size and neighbour count.
fn main() {
    for n_samples in SAMPLE_SIZES {
        let (data, _) = generate_clustered_data(n_samples, N_DIM, N_CLUSTERS, SEED as u64);
        let data_f32 = mat_to_f32(&data);

        for k in K_VALUES {
            bench_pipeline(&data, k, &format!("[f64 n={n_samples:>6} k={k:>2}]"));
            bench_pipeline(&data_f32, k, &format!("[f32 n={n_samples:>6} k={k:>2}]"));
        }
    }
}
