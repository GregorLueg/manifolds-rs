#![allow(clippy::needless_range_loop)] // I like loops ... !

pub mod data;
pub mod parametric;
pub mod training;
pub mod utils;

use ann_search_rs::hnsw::{HnswIndex, HnswState};
use ann_search_rs::nndescent::{NNDescent, NNDescentQuery, UpdateNeighbours};
use faer::traits::{ComplexField, RealField};
use faer::MatRef;
use num_traits::{Float, FromPrimitive};
use rand_distr::{Distribution, StandardNormal};
use std::default::Default;
use std::iter::Sum;
use std::marker::{Send, Sync};
use std::ops::AddAssign;
use std::time::Instant;
use thousands::*;

use crate::data::graph::*;
use crate::data::init::*;
use crate::data::nearest_neighbours::*;
use crate::data::structures::*;
use crate::training::optimiser::*;
use crate::training::UmapParams;

/////////////
// Helpers //
/////////////

/// Helper function to generate the UMAP graph
///
/// ### Params
///
/// * `data` - Input data matrix (samples × features)
/// * `k` - Number of nearest neighbours (typically 15-50).
/// * `ann_type` - Approximate nearest neighbour method: `"annoy"`, `"hnsw"`, or
///   `"nndescent"`. If you provide a weird string, the function will default
///   to `"hnsw"`
/// * `umap_params` - UMAP-specific parameters (bandwidth, local_connectivity,
///   mix_weight)
/// * `nn_params` - Nearest neighbour parameters for nearest neighbour search.
/// * `seed` - Random seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Tuple of (graph, knn_indices, knn_dist) for use in optimisation
pub fn construct_umap_graph<T>(
    data: MatRef<T>,
    k: usize,
    ann_type: String,
    umap_params: &UmapParams<T>,
    nn_params: &NearestNeighbourParams<T>,
    seed: usize,
    verbose: bool,
) -> (SparseGraph<T>, Vec<Vec<usize>>, Vec<Vec<T>>)
where
    T: Float + FromPrimitive + Send + Sync + Default + ComplexField + RealField + Sum + AddAssign,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: UpdateNeighbours<T> + NNDescentQuery<T>,
{
    if verbose {
        println!(
            "Running approximate nearest neighbour search using {}...",
            ann_type
        );
    }

    let start_knn = Instant::now();
    let (knn_indices, knn_dist) = run_ann_search(data, k, ann_type, nn_params, seed);

    if verbose {
        println!("kNN search done in: {:.2?}.", start_knn.elapsed());
        println!("Constructing fuzzy simplicial set...");
    }

    let start_graph = Instant::now();

    let (sigma, rho) = smooth_knn_dist(
        &knn_dist,
        k,
        umap_params.local_connectivity,
        umap_params.bandwidth,
        64,
    );

    let graph = knn_to_coo(&knn_indices, &knn_dist, &sigma, &rho);
    let graph = symmetrise_graph(graph, umap_params.mix_weight);

    if verbose {
        println!(
            "Finalised graph generation in {:.2?}.",
            start_graph.elapsed()
        );
    }

    (graph, knn_indices, knn_dist)
}

////////////////////////
// Main "normal" UMAP //
////////////////////////

/// Run UMAP dimensionality reduction
///
/// Uniform Manifold Approximation and Projection (UMAP) is a manifold learning
/// technique for dimensionality reduction. This implementation follows the
/// standard UMAP algorithm:
///
/// 1. Find k-nearest neighbours using approximate nearest neighbour search
/// 2. Construct fuzzy simplicial set via smooth kNN distances
/// 3. Symmetrise the graph using fuzzy set union
/// 4. Initialise embedding via spectral decomposition
/// 5. Optimise embedding using stochastic gradient descent
///
/// ### Params
///
/// * `data` - Input data matrix (samples × features)
/// * `n_dim` - Target dimensionality (typically 2 or 3)
/// * `k` - Number of nearest neighbours (typically 15-50).
/// * `optimiser` - Which optimiser to use. Choise is `"adam"` or `"sgd"`. If
///   you provide a weird string, the function will default to `"adam"`.
/// * `ann_type` - Approximate nearest neighbour method: `"annoy"`, `"hnsw"`, or
///   `"nndescent"`. If you provide a weird string, the function will default
///   to `"hnsw"`
/// * `initialisation` - The initialisation you wish to use. One of
///   `"spectral"`, `"pca"` or `"random"`.
/// * `umap_params` - UMAP-specific parameters (bandwidth, local_connectivity,
///   mix_weight)
/// * `nn_params` - Optional parameters for nearest neighbour search (uses
///   defaults if None)
/// * `optim_params` - Optional optimisation parameters (uses defaults if None)
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Print progress information
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>` where outer vector has length
/// `n_dim` and inner vectors have length `n_samples`. Each outer element
/// represents one embedding dimension.
///
/// ### Example
///
/// ```ignore
/// use faer::Mat;
/// let data = Mat::from_fn(1000, 128, |_, _| rand::random::<f32>());
/// let embedding = umap(
///     data.as_ref(),
///     2,                          // 2D embedding
///     15,                         // 15 nearest neighbours
///     "adam".into()               // ADAM optimiser
///     "annoy".into(),             // Annoy-based kNN search
///     &UmapParams::default(),     // default parameters for UMAP
///     None,                       // default NN params
///     None,                       // default optim params
///     42,                         // seed
///     true,                       // verbose
/// );
/// // embedding[0] contains x-coordinates for all points
/// // embedding[1] contains y-coordinates for all points
/// ```
#[allow(clippy::too_many_arguments)]
pub fn umap<T>(
    data: MatRef<T>,
    n_dim: usize,
    k: usize,
    optimiser: String,
    ann_type: String,
    initialisation: String,
    umap_params: &UmapParams<T>,
    nn_params: Option<NearestNeighbourParams<T>>,
    optim_params: Option<OptimParams<T>>,
    randomised: bool,
    seed: usize,
    verbose: bool,
) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive + Send + Sync + Default + ComplexField + RealField + Sum + AddAssign,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: UpdateNeighbours<T> + NNDescentQuery<T>,
    StandardNormal: Distribution<T>,
{
    // parse various parameters
    let nn_params = nn_params.unwrap_or_default();
    let optim_params = optim_params.unwrap_or_default();
    let optimiser = parse_optimiser(&optimiser).unwrap_or_default();
    let init_type = parse_initilisation(&initialisation, randomised).unwrap_or_default();

    if verbose {
        println!(
            "Running umap with alpha: {:.4?} and beta: {:.4?}",
            optim_params.a, optim_params.b
        );
    }

    let (graph, _, _) =
        construct_umap_graph(data, k, ann_type, umap_params, &nn_params, seed, verbose);

    if verbose {
        println!(
            "Initialising embedding via {} layout...",
            match init_type {
                #[allow(unused)]
                UmapInit::PcaInit { randomised } => "pca",
                UmapInit::RandomInit => "random",
                UmapInit::SpectralInit => "spectral",
            }
        );
    }

    let start_layout = Instant::now();

    let mut embd = initialise_embedding(&init_type, n_dim, seed as u64, &graph, data);

    let graph = filter_weak_edges(graph, optim_params.n_epochs);
    let graph_adj = coo_to_adjacency_list(&graph);

    if verbose {
        println!(
            "Optimising embedding via {} ({} epochs) on {} edges...",
            match optimiser {
                Optimiser::Adam => "Adam",
                Optimiser::Sgd => "SGD",
                Optimiser::AdamParallel => "Adam (multi-threaded)",
            },
            optim_params.n_epochs,
            graph.col_indices.len().separate_with_underscores()
        );
    }

    match optimiser {
        Optimiser::Adam => {
            optimise_embedding_adam(&mut embd, &graph_adj, &optim_params, seed as u64, verbose)
        }
        Optimiser::Sgd => {
            optimise_embedding_sgd(&mut embd, &graph_adj, &optim_params, seed as u64, verbose);
        }
        Optimiser::AdamParallel => {
            optimise_embedding_adam_parallel(
                &mut embd,
                &graph_adj,
                &optim_params,
                seed as u64,
                verbose,
            );
        }
    }

    let end_layout = start_layout.elapsed();

    if verbose {
        println!(
            "Initialised and optimised embedding in: {:.2?}.",
            end_layout
        );
        println!("UMAP complete!");
    }

    // transpose: from [n_samples][n_dim] to [n_dim][n_samples]
    let n_samples = embd.len();
    let mut transposed = vec![vec![T::zero(); n_samples]; n_dim];

    for sample_idx in 0..n_samples {
        for dim_idx in 0..n_dim {
            transposed[dim_idx][sample_idx] = embd[sample_idx][dim_idx];
        }
    }

    transposed
}

/////////////////////
// Parametric UMAP //
/////////////////////



///////////
// Tests //
///////////

#[cfg(test)]
mod test_umap {
    use super::*;
    use faer::Mat;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn create_data(n_per_cluster: usize, n_dim_input: usize, seed: usize) -> Mat<f32> {
        let mut rng = StdRng::seed_from_u64(seed as u64);

        // create two well-separated clusters in high-dimensional space

        let mut data_vec = Vec::with_capacity(n_per_cluster * 2 * n_dim_input);

        // Cluster 1: centred at origin
        for _ in 0..n_per_cluster {
            for _ in 0..n_dim_input {
                data_vec.push(rng.random::<f32>() * 0.5);
            }
        }

        // Cluster 2: centred at (10, 10, 10, ...)
        for _ in 0..n_per_cluster {
            for _ in 0..n_dim_input {
                data_vec.push(10.0 + rng.random::<f32>() * 0.5);
            }
        }

        Mat::from_fn(n_per_cluster * 2, n_dim_input, |i, j| {
            data_vec[i * n_dim_input + j]
        })
    }

    #[test]
    fn test_umap_separates_clusters_sgd_v1() {
        // seed for this setup
        let seed: usize = 123;

        let n_per_cluster = 150;
        let n_dim_input = 10;

        let data = create_data(n_per_cluster, n_dim_input, seed);

        let embedding = umap(
            data.as_ref(),
            2,
            15,
            "sgd".into(),
            "annoy".into(),
            "pca".into(),
            &UmapParams::default(),
            None,
            None,
            false,
            seed,
            false,
        );

        // Check embedding has correct shape
        assert_eq!(embedding.len(), 2);
        assert_eq!(embedding[0].len(), n_per_cluster * 2);

        // Compute centroids of the two clusters in embedding space
        let mut centroid1 = [0.0f32; 2];
        let mut centroid2 = [0.0f32; 2];

        for i in 0..n_per_cluster {
            centroid1[0] += embedding[0][i];
            centroid1[1] += embedding[1][i];
        }
        for i in n_per_cluster..(n_per_cluster * 2) {
            centroid2[0] += embedding[0][i];
            centroid2[1] += embedding[1][i];
        }

        centroid1[0] /= n_per_cluster as f32;
        centroid1[1] /= n_per_cluster as f32;
        centroid2[0] /= n_per_cluster as f32;
        centroid2[1] /= n_per_cluster as f32;

        // Distance between cluster centroids
        let centroid_dist =
            ((centroid1[0] - centroid2[0]).powi(2) + (centroid1[1] - centroid2[1]).powi(2)).sqrt();

        assert!(
            centroid_dist > 2.0,
            "Clusters should be separated (distance: {:.2})",
            centroid_dist
        );

        // Check spread
        let mean_x: f32 = embedding[0].iter().sum::<f32>() / embedding[0].len() as f32;
        let mean_y: f32 = embedding[1].iter().sum::<f32>() / embedding[1].len() as f32;

        let var_x: f32 = embedding[0]
            .iter()
            .map(|&x| (x - mean_x).powi(2))
            .sum::<f32>()
            / embedding[0].len() as f32;
        let var_y: f32 = embedding[1]
            .iter()
            .map(|&y| (y - mean_y).powi(2))
            .sum::<f32>()
            / embedding[1].len() as f32;

        assert!(
            var_x > 0.5 && var_y > 0.5,
            "Embedding should have spread (var_x: {:.2}, var_y: {:.2})",
            var_x,
            var_y
        );
    }

    #[test]
    fn test_umap_separates_clusters_different_seed_init_sgd() {
        // seed for this setup
        let seed: usize = 456;

        let n_per_cluster = 250;
        let n_dim_input = 10;

        let data = create_data(n_per_cluster, n_dim_input, seed);

        let embedding = umap(
            data.as_ref(),
            2,
            15,
            "sgd".into(),
            "annoy".into(),
            "spectral".into(),
            &UmapParams::default(),
            None,
            None,
            false,
            seed,
            false,
        );

        // Check embedding has correct shape
        assert_eq!(embedding.len(), 2);
        assert_eq!(embedding[0].len(), n_per_cluster * 2);

        // Compute centroids of the two clusters in embedding space
        let mut centroid1 = [0.0f32; 2];
        let mut centroid2 = [0.0f32; 2];

        for i in 0..n_per_cluster {
            centroid1[0] += embedding[0][i];
            centroid1[1] += embedding[1][i];
        }
        for i in n_per_cluster..(n_per_cluster * 2) {
            centroid2[0] += embedding[0][i];
            centroid2[1] += embedding[1][i];
        }

        centroid1[0] /= n_per_cluster as f32;
        centroid1[1] /= n_per_cluster as f32;
        centroid2[0] /= n_per_cluster as f32;
        centroid2[1] /= n_per_cluster as f32;

        // Distance between cluster centroids
        let centroid_dist =
            ((centroid1[0] - centroid2[0]).powi(2) + (centroid1[1] - centroid2[1]).powi(2)).sqrt();

        assert!(
            centroid_dist > 2.0,
            "Clusters should be separated (distance: {:.2})",
            centroid_dist
        );

        // Check spread
        let mean_x: f32 = embedding[0].iter().sum::<f32>() / embedding[0].len() as f32;
        let mean_y: f32 = embedding[1].iter().sum::<f32>() / embedding[1].len() as f32;

        let var_x: f32 = embedding[0]
            .iter()
            .map(|&x| (x - mean_x).powi(2))
            .sum::<f32>()
            / embedding[0].len() as f32;
        let var_y: f32 = embedding[1]
            .iter()
            .map(|&y| (y - mean_y).powi(2))
            .sum::<f32>()
            / embedding[1].len() as f32;

        assert!(
            var_x > 0.5 && var_y > 0.5,
            "Embedding should have spread (var_x: {:.2}, var_y: {:.2})",
            var_x,
            var_y
        );
    }

    #[test]
    fn test_umap_separates_clusters_adam() {
        let seed: usize = 42;

        let n_per_cluster = 150;
        let n_dim_input = 10;

        let data = create_data(n_per_cluster, n_dim_input, seed);

        let embedding = umap(
            data.as_ref(),
            2,
            15,
            "adam".into(),
            "annoy".into(),
            "pca".into(),
            &UmapParams::default(),
            None,
            None,
            false,
            seed,
            false,
        );

        // Check embedding has correct shape
        assert_eq!(embedding.len(), 2);
        assert_eq!(embedding[0].len(), n_per_cluster * 2);

        // Compute centroids of the two clusters in embedding space
        let mut centroid1 = [0.0f32; 2];
        let mut centroid2 = [0.0f32; 2];

        for i in 0..n_per_cluster {
            centroid1[0] += embedding[0][i];
            centroid1[1] += embedding[1][i];
        }
        for i in n_per_cluster..(n_per_cluster * 2) {
            centroid2[0] += embedding[0][i];
            centroid2[1] += embedding[1][i];
        }

        centroid1[0] /= n_per_cluster as f32;
        centroid1[1] /= n_per_cluster as f32;
        centroid2[0] /= n_per_cluster as f32;
        centroid2[1] /= n_per_cluster as f32;

        // Distance between cluster centroids
        let centroid_dist =
            ((centroid1[0] - centroid2[0]).powi(2) + (centroid1[1] - centroid2[1]).powi(2)).sqrt();

        assert!(
            centroid_dist > 2.0,
            "Clusters should be separated (distance: {:.2})",
            centroid_dist
        );

        // Check spread
        let mean_x: f32 = embedding[0].iter().sum::<f32>() / embedding[0].len() as f32;
        let mean_y: f32 = embedding[1].iter().sum::<f32>() / embedding[1].len() as f32;

        let var_x: f32 = embedding[0]
            .iter()
            .map(|&x| (x - mean_x).powi(2))
            .sum::<f32>()
            / embedding[0].len() as f32;
        let var_y: f32 = embedding[1]
            .iter()
            .map(|&y| (y - mean_y).powi(2))
            .sum::<f32>()
            / embedding[1].len() as f32;

        assert!(
            var_x > 0.5 && var_y > 0.5,
            "Embedding should have spread (var_x: {:.2}, var_y: {:.2})",
            var_x,
            var_y
        );
    }

    #[test]
    fn test_umap_separates_clusters_different_seed_init_adam() {
        let seed: usize = 1005;

        let n_per_cluster = 150;
        let n_dim_input = 10;

        let data = create_data(n_per_cluster, n_dim_input, seed);

        let embedding = umap(
            data.as_ref(),
            2,
            15,
            "adam".into(),
            "hnsw".into(),
            "pca".into(),
            &UmapParams::default(),
            None,
            None,
            false,
            seed,
            false,
        );

        // Check embedding has correct shape
        assert_eq!(embedding.len(), 2);
        assert_eq!(embedding[0].len(), n_per_cluster * 2);

        // Compute centroids of the two clusters in embedding space
        let mut centroid1 = [0.0f32; 2];
        let mut centroid2 = [0.0f32; 2];

        for i in 0..n_per_cluster {
            centroid1[0] += embedding[0][i];
            centroid1[1] += embedding[1][i];
        }
        for i in n_per_cluster..(n_per_cluster * 2) {
            centroid2[0] += embedding[0][i];
            centroid2[1] += embedding[1][i];
        }

        centroid1[0] /= n_per_cluster as f32;
        centroid1[1] /= n_per_cluster as f32;
        centroid2[0] /= n_per_cluster as f32;
        centroid2[1] /= n_per_cluster as f32;

        // Distance between cluster centroids
        let centroid_dist =
            ((centroid1[0] - centroid2[0]).powi(2) + (centroid1[1] - centroid2[1]).powi(2)).sqrt();

        assert!(
            centroid_dist > 2.0,
            "Clusters should be separated (distance: {:.2})",
            centroid_dist
        );

        // Check spread
        let mean_x: f32 = embedding[0].iter().sum::<f32>() / embedding[0].len() as f32;
        let mean_y: f32 = embedding[1].iter().sum::<f32>() / embedding[1].len() as f32;

        let var_x: f32 = embedding[0]
            .iter()
            .map(|&x| (x - mean_x).powi(2))
            .sum::<f32>()
            / embedding[0].len() as f32;
        let var_y: f32 = embedding[1]
            .iter()
            .map(|&y| (y - mean_y).powi(2))
            .sum::<f32>()
            / embedding[1].len() as f32;

        assert!(
            var_x > 0.5 && var_y > 0.5,
            "Embedding should have spread (var_x: {:.2}, var_y: {:.2})",
            var_x,
            var_y
        );
    }

    #[test]
    fn test_umap_separates_clusters_adam_par() {
        let seed: usize = 42;

        let n_per_cluster = 150;
        let n_dim_input = 10;

        let data = create_data(n_per_cluster, n_dim_input, seed);

        let embedding = umap(
            data.as_ref(),
            2,
            15,
            "adam_parallel".into(),
            "annoy".into(),
            "pca".into(),
            &UmapParams::default(),
            None,
            None,
            false,
            seed,
            false,
        );

        // Check embedding has correct shape
        assert_eq!(embedding.len(), 2);
        assert_eq!(embedding[0].len(), n_per_cluster * 2);

        // Compute centroids of the two clusters in embedding space
        let mut centroid1 = [0.0f32; 2];
        let mut centroid2 = [0.0f32; 2];

        for i in 0..n_per_cluster {
            centroid1[0] += embedding[0][i];
            centroid1[1] += embedding[1][i];
        }
        for i in n_per_cluster..(n_per_cluster * 2) {
            centroid2[0] += embedding[0][i];
            centroid2[1] += embedding[1][i];
        }

        centroid1[0] /= n_per_cluster as f32;
        centroid1[1] /= n_per_cluster as f32;
        centroid2[0] /= n_per_cluster as f32;
        centroid2[1] /= n_per_cluster as f32;

        // Distance between cluster centroids
        let centroid_dist =
            ((centroid1[0] - centroid2[0]).powi(2) + (centroid1[1] - centroid2[1]).powi(2)).sqrt();

        assert!(
            centroid_dist > 2.0,
            "Clusters should be separated (distance: {:.2})",
            centroid_dist
        );

        // Check spread
        let mean_x: f32 = embedding[0].iter().sum::<f32>() / embedding[0].len() as f32;
        let mean_y: f32 = embedding[1].iter().sum::<f32>() / embedding[1].len() as f32;

        let var_x: f32 = embedding[0]
            .iter()
            .map(|&x| (x - mean_x).powi(2))
            .sum::<f32>()
            / embedding[0].len() as f32;
        let var_y: f32 = embedding[1]
            .iter()
            .map(|&y| (y - mean_y).powi(2))
            .sum::<f32>()
            / embedding[1].len() as f32;

        assert!(
            var_x > 0.5 && var_y > 0.5,
            "Embedding should have spread (var_x: {:.2}, var_y: {:.2})",
            var_x,
            var_y
        );
    }
}
