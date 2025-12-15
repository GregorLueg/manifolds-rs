#![allow(clippy::needless_range_loop)] // I like loops ... !

pub mod data;
pub mod parametric;
pub mod training;
pub mod utils;

use ann_search_rs::{
    hnsw::{HnswIndex, HnswState},
    nndescent::{NNDescent, NNDescentQuery, UpdateNeighbours},
};
use burn::tensor::{backend::AutodiffBackend, Element};
use faer::{
    traits::{ComplexField, RealField},
    MatRef,
};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand_distr::{Distribution, StandardNormal};
use std::{
    default::Default,
    iter::Sum,
    marker::{Send, Sync},
    ops::AddAssign,
    time::Instant,
};
use thousands::*;

use crate::data::graph::*;
use crate::data::init::*;
use crate::data::nearest_neighbours::*;
use crate::data::structures::*;
use crate::parametric::model::*;
use crate::training::optimiser::*;
use crate::training::parametric_train::*;
use crate::training::*;

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
    umap_params: &UmapGraphParams<T>,
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

/// Main Config structure with all of the possible sub configurations
///
/// ### Fields
///
/// * `n_dim` - How many dimensions to return
/// * `k` - Number of neighbours
/// * `optimiser` - Which optimiser to use. Defaults to `"adam_parallel"`.
/// * `ann_type` - Which of the possible approximate nearest neighbour searches
///   to use. Defaults to `"hnsw"`.
/// * `initialisation` - Which embedding initialisation to use. Defaults to
///   spectral clustering.
/// * `nn_params` - Nearest neighbour parameters.
/// * `optim_params` - The optimiser parameters.
/// * `umap_graph_params` - The graph parameters for the generation of the
///   graph structure.
/// * `randomised` - If initialisation is set to PCA, shall randomised PCA be
///   used.
#[derive(Debug, Clone)]
pub struct UmapParams<T> {
    n_dim: usize,
    k: usize,
    optimiser: String,
    ann_type: String,
    initialisation: String,
    nn_params: NearestNeighbourParams<T>,
    umap_graph_params: UmapGraphParams<T>,
    optim_params: OptimParams<T>,
    randomised: bool,
}

impl<T> UmapParams<T>
where
    T: Float + FromPrimitive,
{
    /// Generate new UMAP parameters
    ///
    /// This function will generate new UMAP parameters and has a lot of
    /// options that give fine-grained control. If everything is set to `None`,
    /// sensible (hopefully) defaults will be provided.
    ///
    /// ### Params
    ///
    /// * `n_dim` - How many dimensions to return. Default `2`.
    /// * `k` - How many neighbours to consider. Default `15`.
    /// * `optimiser` - Which optimiser to use. Default `"adam_parallel"`.
    /// * `ann_type` - Which approximate nearest neighbour search algorithm
    ///   to use. Defaults to `"hnsw"`.
    /// * `initialisation` - Which initialisation of the embedding to use.
    ///   Defaults to `"spectral"`.
    /// * `nn_params` - Further nearest neighbour parameters.
    /// * `optim_params` - Further optimiser parameters.
    /// * `umap_graph_params` - Further UMAP graph generation parameters
    /// * `randomised` - If initialisation is set to `"PCA"`, shall randomised
    ///   SVD be used. Defaults to `false`.
    ///
    /// ### Returns
    ///
    /// Hopefully sensible standard parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_dim: Option<usize>,
        k: Option<usize>,
        optimiser: Option<String>,
        ann_type: Option<String>,
        initialisation: Option<String>,
        nn_params: Option<NearestNeighbourParams<T>>,
        optim_params: Option<OptimParams<T>>,
        umap_graph_params: Option<UmapGraphParams<T>>,
        randomised: Option<bool>,
    ) -> Self {
        // sensible defaults
        let n_dim = n_dim.unwrap_or(2);
        let k = k.unwrap_or(15);
        let optimiser = optimiser.unwrap_or("adam_parallel".to_string());
        let ann_type = ann_type.unwrap_or("hnsw".to_string());
        let initialisation = initialisation.unwrap_or("spectral".to_string());
        let nn_params = nn_params.unwrap_or_default();
        let optim_params = optim_params.unwrap_or_default();
        let umap_graph_params = umap_graph_params.unwrap_or_default();
        let randomised = randomised.unwrap_or(false);

        Self {
            n_dim,
            k,
            optimiser,
            ann_type,
            initialisation,
            nn_params,
            optim_params,
            umap_graph_params,
            randomised,
        }
    }

    /// Default 2D parameters
    ///
    /// This function will generate new UMAP parameters and has a lot of
    /// options that give fine-grained control. If everything is set to `None`,
    /// sensible (hopefully) defaults will be provided.
    ///
    /// ### Params
    ///
    /// * `n_dim` - How many dimensions to return. Default `2`.
    /// * `k` - How many neighbours to consider. Default `15`.
    /// * `min_dist` - Minimum distance between the data points. Defaults to
    ///   `0.1`.
    /// * `spread` - Spread paramter. Defaults to `1.0`.
    ///
    /// ### Returns
    ///
    /// Hopefully sensible standard parameters for standard 2D visualisation.
    pub fn default_2d(
        n_dim: Option<usize>,
        k: Option<usize>,
        min_dist: Option<T>,
        spread: Option<T>,
    ) -> Self {
        let n_dim = n_dim.unwrap_or(2);
        let k = k.unwrap_or(15);
        let min_dist = min_dist.unwrap_or(T::from_f64(0.1).unwrap());
        let spread = spread.unwrap_or(T::from_f64(1.0).unwrap());

        Self {
            n_dim,
            k,
            optimiser: "adam_parallel".into(),
            ann_type: "hnsw".into(),
            initialisation: "spectral".into(),
            nn_params: NearestNeighbourParams::default(),
            optim_params: OptimParams::from_min_dist_spread(
                min_dist, spread, None, None, None, None, None, None, None,
            ),
            umap_graph_params: UmapGraphParams::default(),
            randomised: false,
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
/// * `umap_params` - The UMAP parameters.
/// * `seed` - Seed for reproducibility.
/// * `verbose` - Controls verbosity of the function.
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
/// );
/// // embedding[0] contains x-coordinates for all points
/// // embedding[1] contains y-coordinates for all points
/// ```
pub fn umap<T>(
    data: MatRef<T>,
    umap_params: &UmapParams<T>,
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
    let init_type = parse_initilisation(&umap_params.initialisation, umap_params.randomised)
        .unwrap_or_default();
    let optimiser = parse_optimiser(&umap_params.optimiser).unwrap_or_default();

    if verbose {
        println!(
            "Running umap with alpha: {:.4?} and beta: {:.4?}",
            umap_params.optim_params.a, umap_params.optim_params.b
        );
    }

    let (graph, _, _) = construct_umap_graph(
        data,
        umap_params.k,
        umap_params.ann_type.clone(),
        &umap_params.umap_graph_params,
        &umap_params.nn_params,
        seed,
        verbose,
    );

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

    let mut embd = initialise_embedding(&init_type, umap_params.n_dim, seed as u64, &graph, data);

    let graph = filter_weak_edges(graph, umap_params.optim_params.n_epochs);
    let graph_adj = coo_to_adjacency_list(&graph);

    if verbose {
        println!(
            "Optimising embedding via {} ({} epochs) on {} edges...",
            match optimiser {
                Optimiser::Adam => "Adam",
                Optimiser::Sgd => "SGD",
                Optimiser::AdamParallel => "Adam (multi-threaded)",
            },
            umap_params.optim_params.n_epochs,
            graph.col_indices.len().separate_with_underscores()
        );
    }

    match optimiser {
        Optimiser::Adam => optimise_embedding_adam(
            &mut embd,
            &graph_adj,
            &umap_params.optim_params,
            seed as u64,
            verbose,
        ),
        Optimiser::Sgd => {
            optimise_embedding_sgd(
                &mut embd,
                &graph_adj,
                &umap_params.optim_params,
                seed as u64,
                verbose,
            );
        }
        Optimiser::AdamParallel => {
            optimise_embedding_adam_parallel(
                &mut embd,
                &graph_adj,
                &umap_params.optim_params,
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
    let mut transposed = vec![vec![T::zero(); n_samples]; umap_params.n_dim];

    for sample_idx in 0..n_samples {
        for dim_idx in 0..umap_params.n_dim {
            transposed[dim_idx][sample_idx] = embd[sample_idx][dim_idx];
        }
    }

    transposed
}

/////////////////////
// Parametric UMAP //
/////////////////////

/// Stores the parameters for parametric UMAP via neural nets
///
/// * `n_dim` - How many dimensions to return
/// * `k` - Number of neighbours
/// * `ann_type` - Which of the possible approximate nearest neighbour searches
///   to use. Defaults to `"hnsw"`.
/// * `hidden_layers` - Vector of usizes for the hidden layers in the MLP.
/// * `nn_params` - Nearest neighbour parameters.
/// * `umap_graph_params` - The graph parameters for the generation of the
///   graph structure.
/// * `train_param` - Train parameters for the neural network.
#[derive(Debug, Clone)]
pub struct ParametricUmapParams<T> {
    n_dim: usize,
    k: usize,
    ann_type: String,
    hidden_layers: Vec<usize>,
    nn_params: NearestNeighbourParams<T>,
    umap_graph_params: UmapGraphParams<T>,
    train_param: TrainParametricParams<T>,
}

impl<T> ParametricUmapParams<T>
where
    T: Float + FromPrimitive + Element,
{
    /// Generate new parametric UMAP parameters
    ///
    /// Provides fine-grained control over all parametric UMAP settings.
    /// If parameters are set to `None`, sensible defaults will be provided.
    ///
    /// ### Params
    ///
    /// * `n_dim` - Number of embedding dimensions. Default `2`.
    /// * `k` - Number of nearest neighbours. Default `15`.
    /// * `ann_type` - Approximate nearest neighbour algorithm. Default `"hnsw"`.
    /// * `hidden_layers` - Hidden layer sizes for MLP. Default `vec![128, 128, 128]`.
    /// * `nn_params` - Nearest neighbour parameters. Default uses sensible values.
    /// * `umap_graph_params` - UMAP graph parameters. Default uses sensible values.
    /// * `train_param` - Training parameters. Default uses sensible values.
    ///
    /// ### Returns
    ///
    /// Configured `ParametricUmapParams` instance
    pub fn new(
        n_dim: Option<usize>,
        k: Option<usize>,
        ann_type: Option<String>,
        hidden_layers: Option<Vec<usize>>,
        nn_params: Option<NearestNeighbourParams<T>>,
        umap_graph_params: Option<UmapGraphParams<T>>,
        train_param: Option<TrainParametricParams<T>>,
    ) -> Self {
        let n_dim = n_dim.unwrap_or(2);
        let k = k.unwrap_or(15);
        let ann_type = ann_type.unwrap_or("hnsw".to_string());
        let hidden_layers = hidden_layers.unwrap_or(vec![128, 128, 128]);
        let nn_params = nn_params.unwrap_or_default();
        let umap_graph_params = umap_graph_params.unwrap_or_default();
        let train_param = train_param.unwrap_or_default();

        Self {
            n_dim,
            k,
            ann_type,
            hidden_layers,
            nn_params,
            umap_graph_params,
            train_param,
        }
    }

    /// Default parameters for 2D parametric UMAP
    ///
    /// Generates sensible defaults for standard 2D visualisation using
    /// parametric UMAP with a neural network encoder.
    ///
    /// ### Params
    ///
    /// * `n_dim` - Number of embedding dimensions. Default `2`.
    /// * `k` - Number of nearest neighbours. Default `15`.
    /// * `min_dist` - Minimum distance between embedded points. Default `0.1`.
    /// * `spread` - Effective scale of embedded points. Default `1.0`.
    /// * `corr_weight` -
    ///
    /// ### Returns
    ///
    /// Configured `ParametricUmapParams` suitable for 2D visualisation
    pub fn default_2d(
        n_dim: Option<usize>,
        k: Option<usize>,
        min_dist: Option<T>,
        spread: Option<T>,
        corr_weight: Option<T>,
    ) -> Self {
        let n_dim = n_dim.unwrap_or(2);
        let k = k.unwrap_or(15);
        let min_dist = min_dist.unwrap_or(T::from_f64(0.1).unwrap());
        let spread = spread.unwrap_or(T::from_f64(1.0).unwrap());
        let corr_weight = corr_weight.unwrap_or(T::from_f64(0.0).unwrap());

        Self {
            n_dim,
            k,
            ann_type: "hnsw".to_string(),
            hidden_layers: vec![128, 128, 128],
            nn_params: NearestNeighbourParams::default(),
            umap_graph_params: UmapGraphParams::default(),
            train_param: TrainParametricParams::from_min_dist_spread(
                min_dist,
                spread,
                corr_weight,
                None,
                None,
                None,
                None,
            ),
        }
    }
}

/// Run parametric UMAP dimensionality reduction
///
/// Parametric UMAP learns a neural network encoder that maps high-dimensional
/// data to a low-dimensional embedding space. Unlike standard UMAP, this
/// approach provides an explicit parametric mapping that can be applied to
/// new data points without retraining.
///
/// The algorithm follows these steps:
///
/// 1. Find k-nearest neighbours using approximate nearest neighbour search
/// 2. Construct fuzzy simplicial set via smooth kNN distances
/// 3. Symmetrise the graph using fuzzy set union
/// 4. Train an MLP encoder to preserve the graph structure using UMAP loss
/// 5. Return embeddings by passing all data through the trained encoder
///
/// ### Params
///
/// * `data` - Input data matrix (samples × features)
/// * `umap_params` - Configuration parameters for parametric UMAP
/// * `device` - Burn backend device for neural network training
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Whether to print progress information
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
/// use burn::backend::ndarray::{NdArray, NdArrayDevice};
/// use burn::backend::Autodiff;
/// use faer::Mat;
///
/// let data = Mat::from_fn(1000, 128, |_, _| rand::random::<f64>());
/// let params = ParametricUmapParams::default_2d(None, None, None, None, None, None, None);
/// let device = NdArrayDevice::Cpu;
///
/// let embedding = parametric_umap::<f64, Autodiff<NdArray>>(
///     data.as_ref(),
///     &params,
///     &device,
///     42,
///     true,
/// );
/// // embedding[0] contains x-coordinates for all points
/// // embedding[1] contains y-coordinates for all points
/// ```
pub fn parametric_umap<T, B>(
    data: MatRef<T>,
    umap_params: &ParametricUmapParams<T>,
    device: &B::Device,
    seed: usize,
    verbose: bool,
) -> Vec<Vec<T>>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Default
        + ComplexField
        + RealField
        + Sum
        + AddAssign
        + Element,
    B: AutodiffBackend,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: UpdateNeighbours<T> + NNDescentQuery<T>,
{
    // parse various parameters
    let nn_params = umap_params.nn_params.clone();

    if verbose {
        println!(
            "Running umap with alpha: {:.4?} and beta: {:.4?}",
            ToPrimitive::to_f32(&umap_params.train_param.a).unwrap(),
            ToPrimitive::to_f32(&umap_params.train_param.b).unwrap()
        );
    }

    let (graph, _, _) = construct_umap_graph(
        data,
        umap_params.k,
        umap_params.ann_type.clone(),
        &umap_params.umap_graph_params,
        &nn_params,
        seed,
        verbose,
    );

    let model_params = UmapMlpConfig::from_params(
        data.ncols(),
        umap_params.hidden_layers.clone(),
        umap_params.n_dim,
    );

    let embd: Vec<Vec<T>> = train_parametric_umap::<B, T>(
        data,
        graph,
        &model_params,
        &umap_params.train_param,
        device,
        seed,
        verbose,
    );

    // transpose: from [n_samples][n_dim] to [n_dim][n_samples]
    let n_samples = embd.len();
    let mut transposed = vec![vec![T::zero(); n_samples]; umap_params.n_dim];

    for sample_idx in 0..n_samples {
        for dim_idx in 0..umap_params.n_dim {
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
            &UmapParams::new(
                None,
                None,
                Some("sgd".into()),
                None,
                Some("pca".into()),
                None,
                None,
                None,
                None,
            ),
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
    fn test_umap_separates_clusters_different_seed_init_sgd() {
        // seed for this setup
        let seed: usize = 456;

        let n_per_cluster = 250;
        let n_dim_input = 10;

        let data = create_data(n_per_cluster, n_dim_input, seed);

        let embedding = umap(
            data.as_ref(),
            &UmapParams::new(
                None,
                None,
                Some("sgd".into()),
                None,
                Some("pca".into()),
                None,
                None,
                None,
                None,
            ),
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
        let seed: usize = 42;

        let n_per_cluster = 150;
        let n_dim_input = 10;

        let data = create_data(n_per_cluster, n_dim_input, seed);

        let embedding = umap(
            data.as_ref(),
            &UmapParams::new(
                None,
                None,
                Some("adam".into()),
                None,
                Some("pca".into()),
                None,
                None,
                None,
                None,
            ),
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
    fn test_umap_separates_clusters_different_seed_init_adam() {
        let seed: usize = 1005;

        let n_per_cluster = 150;
        let n_dim_input = 10;

        let data = create_data(n_per_cluster, n_dim_input, seed);

        let embedding = umap(
            data.as_ref(),
            &UmapParams::new(
                None,
                None,
                Some("adam".into()),
                None,
                Some("pca".into()),
                None,
                None,
                None,
                None,
            ),
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
    fn test_umap_separates_clusters_adam_par() {
        let seed: usize = 42;

        let n_per_cluster = 150;
        let n_dim_input = 10;

        let data = create_data(n_per_cluster, n_dim_input, seed);

        let embedding = umap(
            data.as_ref(),
            &UmapParams::new(
                None,
                None,
                Some("adam_par".into()),
                None,
                Some("pca".into()),
                None,
                None,
                None,
                None,
            ),
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
