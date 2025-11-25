#![allow(clippy::needless_range_loop)] // I like loops ... !

pub mod data_gen;
pub mod data_struct;
pub mod init;
pub mod macros;
pub mod nearest_neighbours;
pub mod optimiser;
pub mod utils_math;

use ann_search_rs::hnsw::{HnswIndex, HnswState};
use ann_search_rs::nndescent::{NNDescent, UpdateNeighbours};
use faer::MatRef;
use num_traits::{Float, FromPrimitive};
use std::default::Default;
use std::marker::{Send, Sync};

use crate::data_gen::*;
use crate::init::*;
use crate::nearest_neighbours::*;
use crate::optimiser::*;

////////////
// Params //
////////////

/// UMAP algorithm parameters
///
/// Controls the fuzzy simplicial set construction and graph symmetrisation.
///
/// ### Fields
///
/// * `bandwidth` - Convergence tolerance for smooth kNN distance binary search
///   (typically 1e-5). Controls how precisely sigma values are computed.
/// * `local_connectivity` - Number of nearest neighbours assumed to be at
///   distance zero (typically 1.0). Allows for local manifold structure by
///   treating the nearest neighbour(s) as having maximal membership strength.
/// * `mix_weight` - Balance between fuzzy union and directed graph during
///   symmetrisation (typically 1.0).
#[derive(Clone, Debug)]
pub struct UmapParams<T> {
    pub bandwidth: T,
    pub local_connectivity: T,
    pub mix_weight: T,
}

impl<T> Default for UmapParams<T>
where
    T: Float,
{
    /// Returns sensible defaults for UMAP
    ///
    /// ### Returns
    ///
    /// * `bandwidth = 1e-5` - Tight convergence for sigma computation
    /// * `local_connectivity = 1.0` - Treat nearest neighbour as connected
    /// * `mix_weight = 1.0` - Standard symmetric fuzzy union
    fn default() -> Self {
        Self {
            local_connectivity: T::from(1.0).unwrap(),
            bandwidth: T::from(1e-5).unwrap(),
            mix_weight: T::from(1.0).unwrap(),
        }
    }
}

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
///   to `"annoy"`
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
///     2,              // 2D embedding
///     15,             // 15 nearest neighbours
///     "adam".into()   // ADAM optimiser
///     "hnsw".into(),  // HNSW index
///     &UmapParams::default(),
///     None,           // default NN params
///     None,           // default optim params
///     42,             // seed
///     true,           // verbose
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
    umap_params: &UmapParams<T>,
    nn_params: Option<NearestNeighbourParams<T>>,
    optim_params: Option<OptimParams<T>>,
    seed: usize,
    verbose: bool,
) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive + Send + Sync + Default,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: UpdateNeighbours<T>,
{
    let nn_params = nn_params.unwrap_or_default();
    let optim_params = optim_params.unwrap_or_default();
    let optimiser = parse_optimiser(&optimiser).unwrap_or_default();

    if verbose {
        println!("Running approximate nearest neighbour search...");
    }

    let (knn_indices, knn_dist) = run_ann_search(data, k, ann_type, &nn_params, seed, verbose);

    if verbose {
        println!("Constructing fuzzy simplicial set...");
    }

    let (sigma, rho) = smooth_knn_dist(
        &knn_dist,
        k,
        umap_params.local_connectivity,
        umap_params.bandwidth,
        64,
    );

    let graph = knn_to_coo(&knn_indices, &knn_dist, &sigma, &rho);

    let graph = symmetrise_graph(graph, umap_params.mix_weight);

    let graph = filter_weak_edges(graph, optim_params.n_epochs);

    let graph_adj = coo_to_adjacency_list(&graph);

    if verbose {
        println!("Initialising embedding via spectral layout...");
    }

    let mut embd = spectral_layout(&graph, n_dim, seed as u64);

    if verbose {
        println!(
            "Optimising embedding via {} ({} epochs)...",
            match optimiser {
                Optimiser::Adam => "Adam",
                Optimiser::Sgd => "SGD",
            },
            optim_params.n_epochs
        );
    }

    match optimiser {
        Optimiser::Adam => {
            optimise_embedding_adam(&mut embd, &graph_adj, &optim_params, seed as u64, verbose)
        }
        Optimiser::Sgd => {
            optimise_embedding_sgd(&mut embd, &graph_adj, &optim_params, seed as u64, verbose);
        }
    }

    if verbose {
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

///////////
// Tests //
///////////

#[cfg(test)]
mod test_umap {
    use super::*;
    use faer::Mat;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn create_data(n_per_cluster: usize, n_dim_input: usize, seed: u64) -> Mat<f32> {
        let mut rng = StdRng::seed_from_u64(seed);

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
    fn test_umap_separates_clusters_sgd() {
        let n_per_cluster = 50;
        let n_dim_input = 10;

        let data = create_data(n_per_cluster, n_dim_input, 12456);

        let embedding = umap(
            data.as_ref(),
            2,
            15,
            "sgd".into(),
            "hnsw".into(),
            &UmapParams::default(),
            None,
            None,
            42,
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
    fn test_umap_separates_clusters_different_seed_sgd() {
        let n_per_cluster = 50;
        let n_dim_input = 10;

        let data = create_data(n_per_cluster, n_dim_input, 42);

        let embedding = umap(
            data.as_ref(),
            2,
            15,
            "sgd".into(),
            "hnsw".into(),
            &UmapParams::default(),
            None,
            None,
            42,
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
        let n_per_cluster = 50;
        let n_dim_input = 10;

        let data = create_data(n_per_cluster, n_dim_input, 12456);

        let embedding = umap(
            data.as_ref(),
            2,
            15,
            "adam".into(),
            "hnsw".into(),
            &UmapParams::default(),
            None,
            None,
            42,
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
    fn test_umap_separates_clusters_different_seed_adam() {
        let n_per_cluster = 50;
        let n_dim_input = 10;

        let data = create_data(n_per_cluster, n_dim_input, 42);

        let embedding = umap(
            data.as_ref(),
            2,
            15,
            "adam".into(),
            "hnsw".into(),
            &UmapParams::default(),
            None,
            None,
            42,
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
