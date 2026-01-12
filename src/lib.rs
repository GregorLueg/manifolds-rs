#![allow(clippy::needless_range_loop)] // I like loops ... !

pub mod data;
pub mod training;
pub mod utils;

#[cfg(feature = "parametric")]
pub mod parametric;

use ann_search_rs::hnsw::{HnswIndex, HnswState};
use ann_search_rs::nndescent::{ApplySortedUpdates, NNDescent, NNDescentQuery};
use ann_search_rs::utils::dist::SimdDistance;
use burn::tensor::{backend::AutodiffBackend, Element};
use faer::traits::{ComplexField, RealField};
use faer::MatRef;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand_distr::{Distribution, StandardNormal};
use std::{
    default::Default,
    iter::Sum,
    marker::{Send, Sync},
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
    time::Instant,
};
use thousands::*;

use crate::data::graph::*;
use crate::data::init::*;
use crate::data::nearest_neighbours::*;
use crate::data::structures::*;
use crate::parametric::model::*;
use crate::parametric::parametric_train::*;
use crate::training::optimiser::*;
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
#[allow(clippy::too_many_arguments)]
pub fn construct_umap_graph<T>(
    data: MatRef<T>,
    k: usize,
    ann_type: String,
    umap_params: &UmapGraphParams<T>,
    nn_params: &NearestNeighbourParams<T>,
    n_epochs: usize,
    seed: usize,
    verbose: bool,
) -> (SparseGraph<T>, Vec<Vec<usize>>, Vec<Vec<T>>)
where
    T: Float
        + FromPrimitive
        + Send
        + Sync
        + Default
        + ComplexField
        + RealField
        + Sum
        + AddAssign
        + SimdDistance,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
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
        knn_dist[0].len(),
        umap_params.local_connectivity,
        umap_params.bandwidth,
        64,
    );

    let graph = knn_to_coo(&knn_indices, &knn_dist, &sigma, &rho);
    let graph = symmetrise_graph(graph, umap_params.mix_weight);
    let graph = filter_weak_edges(graph, n_epochs, verbose);

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
    T: Float
        + FromPrimitive
        + Send
        + Sync
        + Default
        + ComplexField
        + RealField
        + Sum
        + AddAssign
        + SimdDistance
        + std::fmt::Display,
    HnswIndex<T>: HnswState<T>,
    StandardNormal: Distribution<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    // parse various parameters
    let init_type = parse_initilisation(&umap_params.initialisation, umap_params.randomised)
        .unwrap_or_default();
    let optimiser = parse_optimiser(&umap_params.optimiser).unwrap_or_default();

    if verbose {
        println!(
            "Running umap with alpha: {:.2?} and beta: {:.2?}",
            umap_params.optim_params.a, umap_params.optim_params.b
        );
    }

    let (graph, _, _) = construct_umap_graph(
        data,
        umap_params.k,
        umap_params.ann_type.clone(),
        &umap_params.umap_graph_params,
        &umap_params.nn_params,
        umap_params.optim_params.n_epochs,
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

//////////
// tSNE //
//////////

/// Main configuration for t-SNE dimensionality reduction
///
/// ### Fields
///
/// * `n_dim` - Number of output dimensions (typically 2)
/// * `perplexity` - Perplexity parameter controlling neighbourhood size
///   (typical: 5-50)
/// * `ann_type` - Approximate nearest neighbour method: "hnsw" or "nndescent"
/// * `initialisation` - Embedding initialisation method: "pca", "random", or
///   "spectral"
/// * `nn_params` - Nearest neighbour search parameters
/// * `optim_params` - Optimization parameters (learning rate, epochs, early
///   exaggeration, theta)
/// * `randomised_init` - Use randomised SVD for PCA initialisation
#[derive(Debug, Clone)]
pub struct TsneParams<T> {
    pub n_dim: usize,
    pub perplexity: T,
    pub ann_type: String,
    pub initialisation: String,
    pub nn_params: NearestNeighbourParams<T>,
    pub optim_params: TsneOptimParams<T>,
    pub randomised_init: bool,
}

impl<T> TsneParams<T>
where
    T: Float + FromPrimitive,
{
    /// Create new t-SNE parameters with sensible defaults
    ///
    /// ### Params
    ///
    /// * `n_dim` - Number of output dimensions. Default: 2
    /// * `perplexity` - Perplexity parameter. Default: 30.0
    /// * `lr` - Learning rate. Default: 200.0
    /// * `n_epochs` - Number of optimization epochs. Default: 1000
    /// * `ann_type` - ANN algorithm: "hnsw" or "nndescent". Default: "hnsw"
    /// * `theta` - Barnes-Hut approximation parameter. Default: 0.5
    ///
    /// ### Returns
    ///
    /// `TsneParams` with sensible defaults for standard t-SNE
    pub fn new(
        n_dim: Option<usize>,
        perplexity: Option<T>,
        lr: Option<T>,
        n_epochs: Option<usize>,
        ann_type: Option<String>,
        theta: Option<T>,
    ) -> Self {
        let n_dim = n_dim.unwrap_or(2);
        let perplexity = perplexity.unwrap_or_else(|| T::from_f64(30.0).unwrap());
        let lr = lr.unwrap_or_else(|| T::from_f64(200.0).unwrap());
        let n_epochs = n_epochs.unwrap_or(1000);
        let ann_type = ann_type.unwrap_or_else(|| "hnsw".to_string());
        let theta = theta.unwrap_or_else(|| T::from_f64(0.5).unwrap());

        Self {
            n_dim,
            perplexity,
            ann_type,
            initialisation: "pca".to_string(),
            nn_params: NearestNeighbourParams::default(),
            optim_params: TsneOptimParams {
                n_epochs,
                lr,
                early_exag_iter: 250,
                early_exag_factor: T::from_f64(12.0).unwrap(),
                theta,
            },
            randomised_init: true,
        }
    }
}

/// Construct affinity graph for t-SNE from high-dimensional data
///
/// Performs the following steps:
/// 1. Runs k-nearest neighbour search where k = 3 × perplexity
/// 2. Computes Gaussian affinities P(j|i) via binary search for target entropy
/// 3. Symmetrises to joint probabilities: P_ij = (P(j|i) + P(i|j)) / 2N
///
/// # Arguments
///
/// * `data` - Input data matrix (samples × features)
/// * `perplexity` - Target perplexity (effective number of neighbours,
///   typical: 5-50)
/// * `ann_type` - ANN algorithm: "hnsw" or "nndescent"
/// * `nn_params` - Nearest neighbour search parameters
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Print progress information
///
/// # Returns
///
/// Tuple of:
/// - `SparseGraph<T>` containing symmetric joint probabilities P_ij
/// - `Vec<Vec<usize>>` k-nearest neighbour indices for each point
/// - `Vec<Vec<T>>` k-nearest neighbour distances for each point
///
/// # Notes
///
/// The k value is automatically set to `3 × perplexity`, clamped between 10 and
/// n-1. This is standard practice in t-SNE implementations.
pub fn construct_tsne_graph<T>(
    data: MatRef<T>,
    perplexity: T,
    ann_type: String,
    nn_params: &NearestNeighbourParams<T>,
    seed: usize,
    verbose: bool,
) -> (SparseGraph<T>, Vec<Vec<usize>>, Vec<Vec<T>>)
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Send
        + Sync
        + Default
        + ComplexField
        + RealField
        + Sum
        + AddAssign
        + SimdDistance,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    // t-SNE rule of thumb: k = 3 * perplexity
    let k_float = perplexity * T::from_f64(3.0).unwrap();
    let k = k_float.to_usize().unwrap().max(5).min(data.nrows() - 1);

    if verbose {
        println!("Running kNN search (k={}) using {}...", k, ann_type);
    }

    let start_knn = Instant::now();
    let (knn_indices, knn_dist) = run_ann_search(data, k, ann_type, nn_params, seed);

    if verbose {
        println!("kNN search done in: {:.2?}.", start_knn.elapsed());
        println!("Computing Gaussian affinities and symmetrising...");
    }

    let start_graph = Instant::now();

    // 1. compute Conditional Probs P(j|i)
    let directed_graph = gaussian_knn_affinities(
        &knn_indices,
        &knn_dist,
        perplexity,
        T::from_f64(1e-5).unwrap(),
        200,
    );

    // 2. symmetrise to Joint Probs P_ij
    let graph = symmetrise_affinities_tsne(directed_graph);

    if verbose {
        println!(
            "Finalised graph generation in {:.2?}.",
            start_graph.elapsed()
        );
    }

    (graph, knn_indices, knn_dist)
}

/// Run Barnes-Hut t-SNE dimensionality reduction
///
/// t-Distributed Stochastic Neighbour Embedding (t-SNE) is a technique for
/// visualising high-dimensional data by reducing it to 2 or 3 dimensions.
/// This implementation uses the Barnes-Hut approximation for O(N log N)
/// complexity.
///
/// ### Algorithm
///
/// 1. Construct high-dimensional affinity graph via Gaussian kernels
///    - k-NN search with k = 3 × perplexity
///    - Binary search for precision to match target perplexity
///    - Symmetrise to joint probabilities P_ij
/// 2. Initialise low-dimensional embedding (typically via PCA)
/// 3. Optimise embedding via gradient descent
///    - Attractive forces: exact computation from graph
///    - Repulsive forces: Barnes-Hut approximation
///    - Early exaggeration (first 250 iterations)
///    - Momentum switching at iteration 250
///
/// ### Params
///
/// * `data` - Input data matrix (samples × features)
/// * `params` - t-SNE parameters controlling algorithm behaviour
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Print progress information
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>` where outer vector has length
/// `n_dim` and inner vectors have length `n_samples`. Each outer element
/// represents one embedding dimension.
///
/// # Example
///
/// ```ignore
/// use faer::Mat;
/// let data = Mat::from_fn(1000, 128, |_, _| rand::random::<f32>());
/// let params = TsneParams::new(None, None, None, None, None, None);
/// let embedding = tsne(data.as_ref(), &params, 42, true);
/// // embedding[0] contains x-coordinates for all points
/// // embedding[1] contains y-coordinates for all points
/// ```
///
/// # References
///
/// - van der Maaten & Hinton (2008): "Visualizing Data using t-SNE"
/// - van der Maaten (2014): "Accelerating t-SNE using Tree-Based Algorithms"
pub fn tsne<T>(data: MatRef<T>, params: &TsneParams<T>, seed: usize, verbose: bool) -> Vec<Vec<T>>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Send
        + Sync
        + Default
        + ComplexField
        + RealField
        + Sum
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + SimdDistance
        + std::fmt::Display
        + std::fmt::Debug,
    HnswIndex<T>: HnswState<T>,
    StandardNormal: Distribution<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    // 1. graph construction
    let (graph, _, _) = construct_tsne_graph(
        data,
        params.perplexity,
        params.ann_type.clone(),
        &params.nn_params,
        seed,
        verbose,
    );

    // 2. initialise embedding
    let init_type = parse_initilisation(&params.initialisation, params.randomised_init)
        .unwrap_or(UmapInit::PcaInit { randomised: true });

    if verbose {
        println!("Initialising embedding via PCA...");
    }

    let mut embd = initialise_embedding(&init_type, params.n_dim, seed as u64, &graph, data);

    // 3. optimise
    if verbose {
        println!(
            "Optimising via Barnes-Hut t-SNE ({} epochs)...",
            params.optim_params.n_epochs
        );
    }

    let start_optim = Instant::now();

    optimise_bh_tsne(&mut embd, &params.optim_params, &graph, verbose);

    if verbose {
        println!("Optimisation complete in {:.2?}.", start_optim.elapsed());
    }

    // 4. transpose output: [n_samples][n_dim] → [n_dim][n_samples]
    let n_samples = embd.len();
    let mut transposed = vec![vec![T::zero(); n_samples]; params.n_dim];

    for sample_idx in 0..n_samples {
        for dim_idx in 0..params.n_dim {
            transposed[dim_idx][sample_idx] = embd[sample_idx][dim_idx];
        }
    }

    transposed
}

/////////////////////
// Parametric UMAP //
/////////////////////

#[cfg(feature = "parametric")]
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

#[cfg(feature = "parametric")]
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

#[cfg(feature = "parametric")]
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
        + Element
        + SimdDistance,
    B: AutodiffBackend,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    // parse various parameters
    let nn_params = umap_params.nn_params.clone();

    if verbose {
        println!(
            "Running parametric umap with alpha: {:.2?} and beta: {:.2?}",
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
        umap_params.train_param.n_epochs,
        seed,
        verbose,
    );

    let model_params = UmapMlpConfig::from_params(
        data.ncols(),
        umap_params.hidden_layers.clone(),
        umap_params.n_dim,
    );

    let (embd, _) = train_parametric_umap::<B, T>(
        data,
        graph,
        &model_params,
        &umap_params.train_param,
        device,
        seed,
        verbose,
    );

    embd
}

#[cfg(feature = "parametric")]
/// Train the parametric UMAP model and return it
///
/// ### Params
///
/// * `data` - Input data matrix (samples × features)
/// * `umap_params` - Configuration parameters for parametric UMAP
/// * `device` - Burn backend device for neural network training
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Whether to print progress information
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
/// 5. Returns the trained MLP encoder that can be used now also on new data
///
/// ### Returns
///
/// Returns the `TrainedUmapModel` for further usage.
pub fn train_parametric_umap_model<'a, T, B>(
    data: MatRef<T>,
    umap_params: &ParametricUmapParams<T>,
    device: &'a B::Device,
    seed: usize,
    verbose: bool,
) -> TrainedUmapModel<'a, B, T>
where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Default
        + ComplexField
        + RealField
        + Sum
        + AddAssign
        + Element
        + SimdDistance,
    B: AutodiffBackend,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    // parse various parameters
    let nn_params = umap_params.nn_params.clone();

    if verbose {
        println!(
            "Training parametric umap model with alpha: {:.2?} and beta: {:.2?}",
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
        umap_params.train_param.n_epochs,
        seed,
        verbose,
    );

    let model_params = UmapMlpConfig::from_params(
        data.ncols(),
        umap_params.hidden_layers.clone(),
        umap_params.n_dim,
    );

    let (_, trained_model) = train_parametric_umap::<B, T>(
        data,
        graph,
        &model_params,
        &umap_params.train_param,
        device,
        seed,
        verbose,
    );

    trained_model
}

#[cfg(test)]
mod umap_full_tests {
    use super::*;
    use faer::Mat;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use rustc_hash::FxHashMap;

    /// Create a synthetic dataset with well-separated clusters
    ///
    /// Creates 5 clusters in high-dimensional space:
    /// - Cluster 0: centred at origin
    /// - Cluster 1: centred at (20, 0, 0, ...)
    /// - Cluster 2: centred at (0, 20, 0, ...)
    /// - Cluster 3: centred at (0, 0, 20, ...)
    /// - Cluster 4: centred at (10, 10, 10, ...)
    ///
    /// Each cluster has tight spread (std = 0.5) so they're clearly separated
    fn create_diagnostic_data(
        n_per_cluster: usize,
        n_dim: usize,
        seed: u64,
    ) -> (Mat<f64>, Vec<usize>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_total = n_per_cluster * 5;

        let mut data_vec = Vec::with_capacity(n_total * n_dim);
        let mut labels = Vec::with_capacity(n_total);

        // Define cluster centres
        let centres = [
            vec![0.0; n_dim], // Cluster 0: origin
            (0..n_dim)
                .map(|i| if i == 0 { 20.0 } else { 0.0 })
                .collect::<Vec<_>>(), // Cluster 1
            (0..n_dim)
                .map(|i| if i == 1 { 20.0 } else { 0.0 })
                .collect::<Vec<_>>(), // Cluster 2
            (0..n_dim)
                .map(|i| if i == 2 { 20.0 } else { 0.0 })
                .collect::<Vec<_>>(), // Cluster 3
            vec![10.0; n_dim], // Cluster 4: all 10s
        ];

        for (cluster_id, centre) in centres.iter().enumerate() {
            for _ in 0..n_per_cluster {
                for dim in 0..n_dim {
                    // Add Gaussian noise with std=0.5 around centre
                    let noise: f64 = rng.random::<f64>() * 0.5 - 0.25;
                    data_vec.push(centre[dim] + noise);
                }
                labels.push(cluster_id);
            }
        }

        let data = Mat::from_fn(n_total, n_dim, |i, j| data_vec[i * n_dim + j]);

        (data, labels)
    }

    /// Test 1: Verify kNN search finds correct neighbours
    #[test]
    fn umap_integration_01_knn_correctness() {
        let (data, labels) = create_diagnostic_data(50, 10, 42);
        let k = 15;

        let nn_params = NearestNeighbourParams::default();
        let (knn_indices, knn_dist) =
            run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42);

        println!("\n=== DIAGNOSTIC 1: kNN Search Correctness ===");
        println!(
            "Data shape: {} samples, {} features",
            data.nrows(),
            data.ncols()
        );
        println!("Requested k = {} neighbours", k);
        println!("Returned {} neighbours per point", knn_indices[0].len());

        // Check that kNN doesn't include self
        let mut self_in_neighbours = 0;
        for (i, neighbours) in knn_indices.iter().enumerate() {
            if neighbours.contains(&i) {
                self_in_neighbours += 1;
                if self_in_neighbours == 1 {
                    println!(
                        "WARNING: Point {} has itself in neighbours: {:?}",
                        i, neighbours
                    );
                }
            }
        }

        if self_in_neighbours > 0 {
            println!(
                "ERROR: {} points have themselves in their neighbours!",
                self_in_neighbours
            );
        } else {
            println!("✓ No point has itself in neighbours (correct)");
        }

        // check that neighbours are mostly from same cluster
        let mut intra_cluster_ratio = 0.0;
        for (i, neighbours) in knn_indices.iter().enumerate() {
            let my_label = labels[i];
            let same_cluster = neighbours
                .iter()
                .filter(|&&j| labels[j] == my_label)
                .count();
            intra_cluster_ratio += same_cluster as f64 / neighbours.len() as f64;
        }
        intra_cluster_ratio /= knn_indices.len() as f64;

        println!(
            "Average intra-cluster neighbor ratio: {:.2}%",
            intra_cluster_ratio * 100.0
        );

        assert!(
            intra_cluster_ratio > 0.8,
            "kNN should find mostly same-cluster neighbours, got {:.2}",
            intra_cluster_ratio
        );

        // check distance statistics
        let all_dists: Vec<f64> = knn_dist.iter().flatten().copied().collect();
        let min_dist = all_dists.iter().copied().fold(f64::INFINITY, f64::min);
        let max_dist = all_dists.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean_dist = all_dists.iter().sum::<f64>() / all_dists.len() as f64;

        println!(
            "Distance statistics: min = {:.3}, mean = {:.3}, max = {:.3}",
            min_dist, mean_dist, max_dist
        );

        assert!(min_dist > 0.0, "Minimum distance should be > 0 (no self)");
    }

    /// Test 2: Verify smooth_knn_dist produces reasonable sigma/rho
    #[test]
    fn umap_integration_02_smooth_knn_dist() {
        let (data, _labels) = create_diagnostic_data(50, 10, 42);
        let k = 15;

        let nn_params = NearestNeighbourParams::default();
        let (_, knn_dist) = run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42);

        println!("\n=== DIAGNOSTIC 2: smooth_knn_dist Values ===");

        let umap_params = UmapGraphParams::default();
        let (sigma, rho) = smooth_knn_dist(
            &knn_dist,
            knn_dist[0].len(),
            umap_params.local_connectivity,
            umap_params.bandwidth,
            64,
        );

        println!("Sigma statistics:");
        let min_sigma = sigma.iter().copied().fold(f64::INFINITY, f64::min);
        let max_sigma = sigma.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean_sigma = sigma.iter().sum::<f64>() / sigma.len() as f64;
        println!(
            "  min = {:.6}, mean = {:.6}, max = {:.6}",
            min_sigma, mean_sigma, max_sigma
        );

        println!("Rho statistics:");
        let min_rho = rho.iter().copied().fold(f64::INFINITY, f64::min);
        let max_rho = rho.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean_rho = rho.iter().sum::<f64>() / rho.len() as f64;
        let zero_rho = rho.iter().filter(|&&r| r == 0.0).count();
        println!(
            "  min = {:.6}, mean = {:.6}, max = {:.6}",
            min_rho, mean_rho, max_rho
        );
        println!("  Points with rho=0: {} / {}", zero_rho, rho.len());

        // Critical checks
        assert!(min_sigma > 0.0, "All sigma values should be > 0");
        assert!(
            mean_sigma > 0.01,
            "Mean sigma seems too small: {}",
            mean_sigma
        );

        if zero_rho > 0 {
            println!(
                "WARNING: {} points have rho = 0 (first neighbor at distance 0!)",
                zero_rho
            );
            println!("This suggests self is still in the neighbor list!");
        }

        assert_eq!(
            zero_rho, 0,
            "No point should have rho=0 (would mean self is in neighbours)"
        );

        // Check that rho values are reasonable (should be smallest distance to actual neighbor)
        for i in 0..knn_dist.len() {
            let expected_rho = knn_dist[i][0]; // First neighbor distance
            let actual_rho = rho[i];

            assert!(
                (expected_rho - actual_rho).abs() < 1e-6,
                "Point {}: rho = {:.6} but first neighbor is at distance {:.6}",
                i,
                actual_rho,
                expected_rho
            );
        }
        println!("✓ All rho values correctly match first neighbor distance");
    }

    /// Test 3: Verify graph construction creates strong intra-cluster edges
    #[test]
    fn umap_integration_03_graph_connectivity() {
        let (data, labels) = create_diagnostic_data(50, 10, 42);
        let k = 15;

        let nn_params = NearestNeighbourParams::default();
        let umap_params = UmapGraphParams::default();

        let (graph, _, _) = construct_umap_graph(
            data.as_ref(),
            k,
            "hnsw".to_string(),
            &umap_params,
            &nn_params,
            500,
            42,
            false,
        );

        println!("\n=== DIAGNOSTIC 3: Graph Connectivity ===");
        println!("Graph has {} edges", graph.values.len());

        // Build adjacency list
        let n = graph.n_vertices;
        let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
        for ((&i, &j), &w) in graph
            .row_indices
            .iter()
            .zip(&graph.col_indices)
            .zip(&graph.values)
        {
            adj[i].push((j, w));
        }

        // Check connectivity within each cluster
        for cluster_id in 0..5 {
            let cluster_points: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, &l)| l == cluster_id)
                .map(|(i, _)| i)
                .collect();

            println!(
                "\nCluster {} ({} points):",
                cluster_id,
                cluster_points.len()
            );

            // BFS to check if cluster is connected
            let mut visited = vec![false; n];
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(cluster_points[0]);
            visited[cluster_points[0]] = true;
            let mut reachable = 1;

            while let Some(node) = queue.pop_front() {
                for &(neighbor, _) in &adj[node] {
                    if !visited[neighbor] && cluster_points.contains(&neighbor) {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                        reachable += 1;
                    }
                }
            }

            println!(
                "  Reachable within cluster: {} / {}",
                reachable,
                cluster_points.len()
            );

            // Compute average edge weights within cluster
            let mut intra_weights = Vec::new();
            let mut inter_weights = Vec::new();

            for &i in &cluster_points {
                for &(j, w) in &adj[i] {
                    if cluster_points.contains(&j) {
                        intra_weights.push(w);
                    } else {
                        inter_weights.push(w);
                    }
                }
            }

            if !intra_weights.is_empty() {
                let avg_intra = intra_weights.iter().sum::<f64>() / intra_weights.len() as f64;
                let min_intra = intra_weights.iter().copied().fold(f64::INFINITY, f64::min);
                let max_intra = intra_weights
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                println!(
                    "  Intra-cluster edges: min = {:.6}, avg = {:.6}, max = {:.6}",
                    min_intra, avg_intra, max_intra
                );
            }

            if !inter_weights.is_empty() {
                let avg_inter = inter_weights.iter().sum::<f64>() / inter_weights.len() as f64;
                let min_inter = inter_weights.iter().copied().fold(f64::INFINITY, f64::min);
                let max_inter = inter_weights
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                println!(
                    "  Inter-cluster edges: min = {:.6}, avg = {:.6}, max = {:.6}",
                    min_inter, avg_inter, max_inter
                );
            }

            assert_eq!(
                reachable,
                cluster_points.len(),
                "Cluster {} is fragmented! Only {} / {} points reachable",
                cluster_id,
                reachable,
                cluster_points.len()
            );
        }
    }

    /// Test 4: Verify initialisation doesn't pre-split clusters
    #[test]
    fn umap_integration_04_initialisation() {
        let (data, labels) = create_diagnostic_data(50, 10, 42);

        let umap_params = UmapGraphParams::default();
        let nn_params = NearestNeighbourParams::default();

        let (graph, _, _) = construct_umap_graph(
            data.as_ref(),
            15,
            "hnsw".to_string(),
            &umap_params,
            &nn_params,
            500,
            42,
            false,
        );

        println!("\n=== DIAGNOSTIC 4: Initialisation Quality ===");

        // Test each initialisation method INCLUDING PCA
        for init_name in &["spectral", "random", "pca"] {
            let init_type = parse_initilisation(init_name, false).unwrap();
            let embedding = initialise_embedding(&init_type, 2, 42, &graph, data.as_ref());

            println!("\n{} initialisation:", init_name);

            // Check coordinate range
            let coords: Vec<f64> = embedding.iter().flat_map(|p| p.iter().copied()).collect();
            let min_coord = coords.iter().copied().fold(f64::INFINITY, f64::min);
            let max_coord = coords.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let range = max_coord - min_coord;
            println!(
                "  Coordinate range: [{:.3}, {:.3}] (span: {:.3})",
                min_coord, max_coord, range
            );

            // Check if clusters are separated
            let mut cluster_centres: FxHashMap<usize, (f64, f64, usize)> = FxHashMap::default();

            for (i, &label) in labels.iter().enumerate() {
                let entry = cluster_centres.entry(label).or_insert((0.0, 0.0, 0));
                entry.0 += embedding[i][0];
                entry.1 += embedding[i][1];
                entry.2 += 1;
            }

            // Compute centroids
            let mut centroids: Vec<(usize, f64, f64)> = Vec::new();
            for (label, (sum_x, sum_y, count)) in cluster_centres {
                centroids.push((label, sum_x / count as f64, sum_y / count as f64));
            }

            // Check pairwise distances between centroids
            let mut min_centroid_dist = f64::INFINITY;
            let mut max_centroid_dist = f64::NEG_INFINITY;
            for i in 0..centroids.len() {
                for j in (i + 1)..centroids.len() {
                    let dist = ((centroids[i].1 - centroids[j].1).powi(2)
                        + (centroids[i].2 - centroids[j].2).powi(2))
                    .sqrt();
                    min_centroid_dist = min_centroid_dist.min(dist);
                    max_centroid_dist = max_centroid_dist.max(dist);
                }
            }

            println!(
                "  Inter-cluster centroid distances: min = {:.3}, max = {:.3}",
                min_centroid_dist, max_centroid_dist
            );

            // Check spread within each cluster
            let mut avg_intra_dist = 0.0;
            for (label, cx, cy) in &centroids {
                let points: Vec<usize> = labels
                    .iter()
                    .enumerate()
                    .filter(|(_, &l)| l == *label)
                    .map(|(i, _)| i)
                    .collect();

                let intra_dist: f64 = points
                    .iter()
                    .map(|&i| {
                        ((embedding[i][0] - cx).powi(2) + (embedding[i][1] - cy).powi(2)).sqrt()
                    })
                    .sum::<f64>()
                    / points.len() as f64;

                avg_intra_dist += intra_dist;
            }
            avg_intra_dist /= 5.0;
            println!("  Average intra-cluster distance: {:.3}", avg_intra_dist);

            // Critical assertions
            assert!(
                range > 1.0,
                "{} initialization has insufficient spread: range = {:.3} (need > 1.0)",
                init_name,
                range
            );

            // For spectral and random, we expect decent separation
            if init_name != &"pca" {
                assert!(
                    min_centroid_dist > 0.1 || max_centroid_dist > 2.0,
                    "{} initialisation has poor initial separation: min = {:.3}, max = {:.3}",
                    init_name,
                    min_centroid_dist,
                    max_centroid_dist
                );
            }
        }
    }

    /// Test 5: Check optimisation with different optimisers
    #[test]
    fn umap_integration_05_optimisation_quality() {
        let (data, labels) = create_diagnostic_data(50, 10, 123);

        println!("\n=== DIAGNOSTIC 5: Optimisation Quality ===");

        // Test all init+optimizer combinations
        let configs = vec![
            ("spectral", "adam"),
            ("spectral", "sgd"),
            ("spectral", "adam_parallel"),
            ("pca", "adam"),
            ("pca", "sgd"),
            ("pca", "adam_parallel"),
            ("random", "adam"),
            ("random", "sgd"),
            ("random", "adam_parallel"),
        ];

        for (init, opt) in configs {
            println!("\n--- Testing: init = {}, optimiser = {} ---", init, opt);

            let params = UmapParams::new(
                Some(2),
                Some(15),
                Some(opt.to_string()),
                None,
                Some(init.to_string()),
                None,
                None,
                None,
                None,
            );

            let embedding = umap(data.as_ref(), &params, 42, false);

            // Check that coordinates are finite
            let mut has_nan = false;
            let mut has_inf = false;
            for i in 0..embedding[0].len() {
                if embedding[0][i].is_nan() || embedding[1][i].is_nan() {
                    has_nan = true;
                }
                if embedding[0][i].is_infinite() || embedding[1][i].is_infinite() {
                    has_inf = true;
                }
            }

            assert!(!has_nan, "Embedding contains NaN values!");
            assert!(!has_inf, "Embedding contains infinite values!");

            // Check cluster separation
            let mut cluster_centres: FxHashMap<usize, (f64, f64, usize)> = FxHashMap::default();

            for (i, &label) in labels.iter().enumerate() {
                let entry = cluster_centres.entry(label).or_insert((0.0, 0.0, 0));
                entry.0 += embedding[0][i];
                entry.1 += embedding[1][i];
                entry.2 += 1;
            }

            let mut centroids: Vec<(usize, f64, f64)> = Vec::new();
            for (label, (sum_x, sum_y, count)) in cluster_centres {
                centroids.push((label, sum_x / count as f64, sum_y / count as f64));
            }

            // Check if any cluster is fragmented
            for cluster_id in 0..5 {
                let points: Vec<usize> = labels
                    .iter()
                    .enumerate()
                    .filter(|(_, &l)| l == cluster_id)
                    .map(|(i, _)| i)
                    .collect();

                // Check connectivity in 2D space (threshold = 3.0)
                let threshold = 3.0;

                let mut visited = vec![false; points.len()];
                let mut queue = std::collections::VecDeque::new();
                queue.push_back(0);
                visited[0] = true;
                let mut reachable = 1;

                while let Some(idx) = queue.pop_front() {
                    let pi = points[idx];

                    for (other_idx, &other_i) in points.iter().enumerate() {
                        if !visited[other_idx] {
                            let dist = ((embedding[0][pi] - embedding[0][other_i]).powi(2)
                                + (embedding[1][pi] - embedding[1][other_i]).powi(2))
                            .sqrt();

                            if dist < threshold {
                                visited[other_idx] = true;
                                queue.push_back(other_idx);
                                reachable += 1;
                            }
                        }
                    }
                }

                let connectivity_ratio = reachable as f64 / points.len() as f64;
                println!(
                    "  Cluster {}: {}/{} points connected ({:.1}%)",
                    cluster_id,
                    reachable,
                    points.len(),
                    connectivity_ratio * 100.0
                );

                assert!(
                    connectivity_ratio > 0.85,
                    "Cluster {} is fragmented with init={}, opt={}! Only {:.1}% connected",
                    cluster_id,
                    init,
                    opt,
                    connectivity_ratio * 100.0
                );
            }

            // Check minimum inter-cluster distance
            let mut min_inter_dist = f64::INFINITY;
            let mut avg_inter_dist = 0.0;
            let mut count = 0;

            for i in 0..centroids.len() {
                for j in (i + 1)..centroids.len() {
                    let dist = ((centroids[i].1 - centroids[j].1).powi(2)
                        + (centroids[i].2 - centroids[j].2).powi(2))
                    .sqrt();
                    min_inter_dist = min_inter_dist.min(dist);
                    avg_inter_dist += dist;
                    count += 1;
                }
            }
            avg_inter_dist /= count as f64;

            println!(
                "  Inter-cluster distances: min = {:.3}, avg = {:.3}",
                min_inter_dist, avg_inter_dist
            );

            assert!(
                min_inter_dist > 0.5,
                "Clusters too close with init = {}, opt = {}: min dist = {:.3}",
                init,
                opt,
                min_inter_dist
            );

            // Check average intra-cluster compactness
            let mut avg_intra_dist = 0.0;
            for (cluster_id, cx, cy) in &centroids {
                let points: Vec<usize> = labels
                    .iter()
                    .enumerate()
                    .filter(|(_, &l)| l == *cluster_id)
                    .map(|(i, _)| i)
                    .collect();

                let intra_dist: f64 = points
                    .iter()
                    .map(|&i| {
                        ((embedding[0][i] - cx).powi(2) + (embedding[1][i] - cy).powi(2)).sqrt()
                    })
                    .sum::<f64>()
                    / points.len() as f64;

                avg_intra_dist += intra_dist;
            }
            avg_intra_dist /= 5.0;

            println!("  Average intra-cluster distance: {:.3}", avg_intra_dist);

            // Quality metric: inter-cluster distance should be >> intra-cluster distance
            let separation_ratio = min_inter_dist / avg_intra_dist;
            println!("  Separation ratio (inter/intra): {:.2}", separation_ratio);

            assert!(
                separation_ratio > 0.3,
                "Poor separation with init = {}, opt = {}: ratio = {:.2}",
                init,
                opt,
                separation_ratio
            );
        }

        println!("\n✓ All init+optimiser combinations produced valid embeddings!");
    }

    /// Test 6: Compare optimisation consistency across runs
    #[test]
    fn umap_integration_06_reproducibility() {
        let (data, _) = create_diagnostic_data(50, 10, 42);

        println!("\n=== DIAGNOSTIC 6: Reproducibility ===");

        // Run UMAP twice with same seed
        let params = UmapParams::new(
            Some(2),
            Some(15),
            Some("adam_parallel".to_string()),
            None,
            Some("spectral".to_string()),
            None,
            None,
            None,
            None,
        );

        let embedding1 = umap(data.as_ref(), &params, 42, false);
        let embedding2 = umap(data.as_ref(), &params, 42, false);

        // Check if embeddings are identical
        let mut max_diff = 0.0;
        for i in 0..embedding1[0].len() {
            for dim in 0..2 {
                let diff = (embedding1[dim][i] - embedding2[dim][i]).abs();
                max_diff = max_diff.max(diff);
            }
        }

        println!(
            "Maximum coordinate difference between runs: {:.10}",
            max_diff
        );

        assert!(
            max_diff < 1e-6,
            "UMAP should be reproducible with same seed, but max diff = {}",
            max_diff
        );

        println!("✓ UMAP is reproducible with same seed");
    }
}

#[cfg(test)]
mod parametric_integration_tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::backend::Autodiff;
    use faer::Mat;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use rustc_hash::FxHashMap;

    type TestBackend = Autodiff<NdArray<f64>>;

    fn create_test_data(n_per_cluster: usize, n_dim: usize, seed: u64) -> (Mat<f64>, Vec<usize>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_total = n_per_cluster * 5;

        let mut data_vec = Vec::with_capacity(n_total * n_dim);
        let mut labels = Vec::with_capacity(n_total);

        let centres = [
            vec![0.0; n_dim],
            (0..n_dim)
                .map(|i| if i == 0 { 20.0 } else { 0.0 })
                .collect::<Vec<_>>(),
            (0..n_dim)
                .map(|i| if i == 1 { 20.0 } else { 0.0 })
                .collect::<Vec<_>>(),
            (0..n_dim)
                .map(|i| if i == 2 { 20.0 } else { 0.0 })
                .collect::<Vec<_>>(),
            vec![10.0; n_dim],
        ];

        for (cluster_id, centre) in centres.iter().enumerate() {
            for _ in 0..n_per_cluster {
                for dim in 0..n_dim {
                    let noise: f64 = rng.random::<f64>() * 0.5 - 0.25;
                    data_vec.push(centre[dim] + noise);
                }
                labels.push(cluster_id);
            }
        }

        let data = Mat::from_fn(n_total, n_dim, |i, j| data_vec[i * n_dim + j]);
        (data, labels)
    }

    fn fast_test_params_custom(
        n_dim: Option<usize>,
        n_neighbours: Option<usize>,
        min_dist: Option<f64>,
        spread: Option<f64>,
        hidden_layers: Vec<usize>,
        corr_weight: Option<f64>,
    ) -> ParametricUmapParams<f64> {
        let n_dim = n_dim.unwrap_or(2);
        let n_neighbours = n_neighbours.unwrap_or(15);
        let min_dist = min_dist.unwrap_or(0.1);
        let corr_weight = corr_weight.unwrap_or(0.0);
        let spread = spread.unwrap_or(1.0);

        let fit_params = TrainParametricParams::from_min_dist_spread(
            min_dist,
            spread,
            corr_weight,
            None,
            Some(10),
            Some(50),
            None,
        );

        ParametricUmapParams::new(
            Some(n_dim),
            Some(n_neighbours),
            Some("annoy".into()),
            Some(hidden_layers),
            None,
            None,
            Some(fit_params),
        )
    }

    fn fast_test_params() -> ParametricUmapParams<f64> {
        fast_test_params_custom(Some(2), Some(15), Some(0.1), Some(1.0), vec![32], Some(0.0))
    }

    #[test]
    fn parametric_01_comprehensive_quality() {
        let (data, labels) = create_test_data(20, 10, 42);
        let device = NdArrayDevice::Cpu;

        println!("\n=== PARAMETRIC TEST 1: Comprehensive Quality ===");
        println!("Data: {} samples, {} features", data.nrows(), data.ncols());
        println!("Training: 10 epochs, 32 hidden units, batch size 50");

        let params = fast_test_params();
        let embedding =
            parametric_umap::<f64, TestBackend>(data.as_ref(), &params, &device, 42, false);

        // Basic shape checks
        assert_eq!(embedding.len(), 2, "Should have 2 dimensions");
        assert_eq!(embedding[0].len(), 100, "Should have 100 samples");
        println!(
            "✓ Embedding shape: {} dimensions × {} samples",
            embedding.len(),
            embedding[0].len()
        );

        // Check for finite values
        let mut has_nan = false;
        let mut has_inf = false;
        for dim in 0..2 {
            for i in 0..embedding[dim].len() {
                if embedding[dim][i].is_nan() {
                    has_nan = true;
                }
                if embedding[dim][i].is_infinite() {
                    has_inf = true;
                }
            }
        }
        assert!(!has_nan, "Embedding contains NaN values");
        assert!(!has_inf, "Embedding contains infinite values");
        println!("✓ All coordinates are finite");

        // Coordinate statistics
        for dim in 0..2 {
            let min = embedding[dim].iter().copied().fold(f64::INFINITY, f64::min);
            let max = embedding[dim]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mean = embedding[dim].iter().sum::<f64>() / embedding[dim].len() as f64;
            println!(
                "  Dim {}: min = {:.3}, mean = {:.3}, max = {:.3}",
                dim, min, mean, max
            );
        }

        // Compute cluster centroids
        let mut cluster_centres: FxHashMap<usize, (f64, f64, usize)> = FxHashMap::default();
        for (i, &label) in labels.iter().enumerate() {
            let entry = cluster_centres.entry(label).or_insert((0.0, 0.0, 0));
            entry.0 += embedding[0][i];
            entry.1 += embedding[1][i];
            entry.2 += 1;
        }

        let mut centroids: Vec<(usize, f64, f64)> = Vec::new();
        for (label, (sum_x, sum_y, count)) in cluster_centres {
            centroids.push((label, sum_x / count as f64, sum_y / count as f64));
        }

        // Check inter-cluster separation
        println!("\nCluster analysis:");
        let mut min_inter_dist = f64::INFINITY;
        let mut max_inter_dist = f64::NEG_INFINITY;
        let mut avg_inter_dist = 0.0;
        let mut count = 0;

        for i in 0..centroids.len() {
            for j in (i + 1)..centroids.len() {
                let dist = ((centroids[i].1 - centroids[j].1).powi(2)
                    + (centroids[i].2 - centroids[j].2).powi(2))
                .sqrt();
                min_inter_dist = min_inter_dist.min(dist);
                max_inter_dist = max_inter_dist.max(dist);
                avg_inter_dist += dist;
                count += 1;
            }
        }
        avg_inter_dist /= count as f64;

        println!(
            "  Inter-cluster distances: min = {:.3}, avg = {:.3}, max = {:.3}",
            min_inter_dist, avg_inter_dist, max_inter_dist
        );

        assert!(
            min_inter_dist > 0.5,
            "Clusters too close: min distance = {:.3}",
            min_inter_dist
        );

        // Check cluster connectivity
        for cluster_id in 0..5 {
            let points: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, &l)| l == cluster_id)
                .map(|(i, _)| i)
                .collect();

            let threshold = 3.0;
            let mut visited = vec![false; points.len()];
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(0);
            visited[0] = true;
            let mut reachable = 1;

            while let Some(idx) = queue.pop_front() {
                let pi = points[idx];

                for (other_idx, &other_i) in points.iter().enumerate() {
                    if !visited[other_idx] {
                        let dist = ((embedding[0][pi] - embedding[0][other_i]).powi(2)
                            + (embedding[1][pi] - embedding[1][other_i]).powi(2))
                        .sqrt();

                        if dist < threshold {
                            visited[other_idx] = true;
                            queue.push_back(other_idx);
                            reachable += 1;
                        }
                    }
                }
            }

            let connectivity = reachable as f64 / points.len() as f64;
            println!(
                "  Cluster {}: {}/{} connected ({:.1}%)",
                cluster_id,
                reachable,
                points.len(),
                connectivity * 100.0
            );

            assert!(
                connectivity > 0.85,
                "Cluster {} fragmented: only {:.1}% connected",
                cluster_id,
                connectivity * 100.0
            );
        }

        // Compute intra-cluster compactness
        let mut avg_intra_dist = 0.0;
        for (cluster_id, cx, cy) in &centroids {
            let points: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, &l)| l == *cluster_id)
                .map(|(i, _)| i)
                .collect();

            let intra_dist: f64 = points
                .iter()
                .map(|&i| ((embedding[0][i] - cx).powi(2) + (embedding[1][i] - cy).powi(2)).sqrt())
                .sum::<f64>()
                / points.len() as f64;

            avg_intra_dist += intra_dist;
        }
        avg_intra_dist /= 5.0;

        let separation_ratio = min_inter_dist / avg_intra_dist;
        println!("  Average intra-cluster distance: {:.3}", avg_intra_dist);
        println!("  Separation ratio (inter/intra): {:.2}", separation_ratio);

        assert!(
            separation_ratio > 0.3,
            "Poor separation: ratio = {:.2}",
            separation_ratio
        );

        println!("✓ All quality checks passed");
    }

    #[test]
    fn parametric_02_different_dimensions() {
        let (data, _) = create_test_data(15, 10, 42);
        let device = NdArrayDevice::Cpu;

        println!("\n=== PARAMETRIC TEST 2: Different Output Dimensions ===");

        for n_dim in [2, 3, 5] {
            println!("\nTesting {} dimensions...", n_dim);

            let params = fast_test_params_custom(Some(n_dim), None, None, None, vec![32], None);
            let embedding =
                parametric_umap::<f64, TestBackend>(data.as_ref(), &params, &device, 42, false);

            assert_eq!(embedding.len(), n_dim, "Should have {} dimensions", n_dim);

            for dim in 0..n_dim {
                assert_eq!(
                    embedding[dim].len(),
                    75,
                    "Dimension {} should have 75 samples",
                    dim
                );

                let has_non_finite = embedding[dim].iter().any(|&x| !x.is_finite());
                assert!(!has_non_finite, "Dimension {} has non-finite values", dim);
            }

            println!("  ✓ {} dimensions: all finite, correct shape", n_dim);
        }
    }

    #[test]
    fn parametric_03_different_architectures() {
        let (data, _) = create_test_data(15, 10, 42);
        let device = NdArrayDevice::Cpu;

        println!("\n=== PARAMETRIC TEST 3: Different Network Architectures ===");

        let layer_configs = vec![vec![32], vec![64, 32], vec![128, 64, 32]];

        for hidden_layers in layer_configs {
            println!("\nTesting architecture: {:?}", hidden_layers);

            let params =
                fast_test_params_custom(None, None, None, None, hidden_layers.clone(), None);

            let embedding =
                parametric_umap::<f64, TestBackend>(data.as_ref(), &params, &device, 42, false);

            assert_eq!(embedding.len(), 2);
            assert_eq!(embedding[0].len(), 75);

            let has_non_finite = embedding[0]
                .iter()
                .chain(&embedding[1])
                .any(|&x| !x.is_finite());
            assert!(
                !has_non_finite,
                "Architecture {:?} produced non-finite values",
                hidden_layers
            );

            println!("  ✓ Architecture {:?}: all finite", hidden_layers);
        }
    }

    #[test]
    fn parametric_04_correlation_loss() {
        let (data, _) = create_test_data(15, 10, 42);
        let device = NdArrayDevice::Cpu;

        println!("\n=== PARAMETRIC TEST 4: Correlation Loss ===");

        let params = fast_test_params_custom(None, None, None, None, vec![32], Some(0.5));
        let embedding =
            parametric_umap::<f64, TestBackend>(data.as_ref(), &params, &device, 42, false);

        assert_eq!(embedding.len(), 2);
        assert_eq!(embedding[0].len(), 75);

        let has_non_finite = embedding[0]
            .iter()
            .chain(&embedding[1])
            .any(|&x| !x.is_finite());
        assert!(
            !has_non_finite,
            "Correlation loss produced non-finite values"
        );

        println!("  ✓ Correlation loss (λ=0.5): all finite");
    }

    #[test]
    fn parametric_05_min_dist_spread() {
        let (data, _) = create_test_data(15, 10, 42);
        let device = NdArrayDevice::Cpu;

        println!("\n=== PARAMETRIC TEST 5: min_dist and spread ===");

        let configs = vec![(0.1, 1.0), (0.5, 1.0), (0.1, 2.0)];

        for (min_dist, spread) in configs {
            println!("\nTesting min_dist={}, spread={}...", min_dist, spread);

            let params =
                fast_test_params_custom(None, None, Some(min_dist), Some(spread), vec![32], None);

            let embedding =
                parametric_umap::<f64, TestBackend>(data.as_ref(), &params, &device, 42, false);

            let has_non_finite = embedding[0]
                .iter()
                .chain(&embedding[1])
                .any(|&x| !x.is_finite());
            assert!(
                !has_non_finite,
                "min_dist={}, spread={} produced non-finite values",
                min_dist, spread
            );

            println!("  ✓ min_dist={}, spread={}: all finite", min_dist, spread);
        }
    }

    #[test]
    fn parametric_06_small_dataset() {
        let data = Mat::from_fn(10, 5, |i, j| (i as f64 + j as f64) * 0.1);
        let device = NdArrayDevice::Cpu;

        println!("\n=== PARAMETRIC TEST 6: Small Dataset ===");
        println!("Data: {} samples, {} features", data.nrows(), data.ncols());

        let params = fast_test_params_custom(None, Some(5), None, None, vec![32], None);

        let embedding =
            parametric_umap::<f64, TestBackend>(data.as_ref(), &params, &device, 42, false);

        assert_eq!(embedding.len(), 2);
        assert_eq!(embedding[0].len(), 10);

        let has_non_finite = embedding[0]
            .iter()
            .chain(&embedding[1])
            .any(|&x| !x.is_finite());
        assert!(!has_non_finite, "Small dataset produced non-finite values");

        println!("  ✓ Small dataset (10 samples): all finite");
    }
}

// #[cfg(test)]
// mod parametric_integration_tests {
//     use super::*;
//     use burn::backend::ndarray::{NdArray, NdArrayDevice};
//     use burn::backend::Autodiff;
//     use faer::Mat;
//     use rand::{rngs::StdRng, Rng, SeedableRng};
//     use rustc_hash::FxHashMap;

//     type TestBackend = Autodiff<NdArray<f64>>;

//     fn create_test_data(n_per_cluster: usize, n_dim: usize, seed: u64) -> (Mat<f64>, Vec<usize>) {
//         let mut rng = StdRng::seed_from_u64(seed);
//         let n_total = n_per_cluster * 5;

//         let mut data_vec = Vec::with_capacity(n_total * n_dim);
//         let mut labels = Vec::with_capacity(n_total);

//         let centres = [
//             vec![0.0; n_dim],
//             (0..n_dim)
//                 .map(|i| if i == 0 { 20.0 } else { 0.0 })
//                 .collect::<Vec<_>>(),
//             (0..n_dim)
//                 .map(|i| if i == 1 { 20.0 } else { 0.0 })
//                 .collect::<Vec<_>>(),
//             (0..n_dim)
//                 .map(|i| if i == 2 { 20.0 } else { 0.0 })
//                 .collect::<Vec<_>>(),
//             vec![10.0; n_dim],
//         ];

//         for (cluster_id, centre) in centres.iter().enumerate() {
//             for _ in 0..n_per_cluster {
//                 for dim in 0..n_dim {
//                     let noise: f64 = rng.random::<f64>() * 0.5 - 0.25;
//                     data_vec.push(centre[dim] + noise);
//                 }
//                 labels.push(cluster_id);
//             }
//         }

//         let data = Mat::from_fn(n_total, n_dim, |i, j| data_vec[i * n_dim + j]);
//         (data, labels)
//     }

//     fn fast_test_params_custom(
//         n_dim: Option<usize>,
//         n_neighbours: Option<usize>,
//         min_dist: Option<f64>,
//         spread: Option<f64>,
//         hidden_layers: Vec<usize>,
//         corr_weight: Option<f64>,
//     ) -> ParametricUmapParams<f64> {
//         let n_dim = n_dim.unwrap_or(2);
//         let n_neighbours = n_neighbours.unwrap_or(15);
//         let min_dist = min_dist.unwrap_or(0.1);
//         let corr_weight = corr_weight.unwrap_or(0.0);

//         let fit_params = TrainParametricParams::from_min_dist_spread(
//             min_dist,
//             spread,
//             corr_weight,
//             None,
//             Some(10),
//             Some(50),
//             None,
//         );

//         ParametricUmapParams::new(
//             Some(n_dim),
//             Some(n_neighbours),
//             Some("annoy".into()),
//             Some(hidden_layers),
//             None,
//             None,
//             Some(fit_params),
//         )
//     }

//     fn fast_test_params() -> ParametricUmapParams<f64> {
//         fast_test_params_custom(2, 15, 0.1, 1.0, vec![32], 0.0)
//     }

//     #[test]
//     fn parametric_01_comprehensive_quality() {
//         let (data, labels) = create_test_data(20, 10, 42);
//         let device = NdArrayDevice::Cpu;

//         println!("\n=== PARAMETRIC TEST 1: Comprehensive Quality ===");
//         println!("Data: {} samples, {} features", data.nrows(), data.ncols());
//         println!("Training: 10 epochs, 32 hidden units, batch size 50");

//         let params = fast_test_params();
//         let embedding =
//             parametric_umap::<f64, TestBackend>(data.as_ref(), &params, &device, 42, false);

//         // Basic shape checks
//         assert_eq!(embedding.len(), 2, "Should have 2 dimensions");
//         assert_eq!(embedding[0].len(), 100, "Should have 100 samples");
//         println!(
//             "✓ Embedding shape: {} dimensions × {} samples",
//             embedding.len(),
//             embedding[0].len()
//         );

//         // Check for finite values
//         let mut has_nan = false;
//         let mut has_inf = false;
//         for dim in 0..2 {
//             for i in 0..embedding[dim].len() {
//                 if embedding[dim][i].is_nan() {
//                     has_nan = true;
//                 }
//                 if embedding[dim][i].is_infinite() {
//                     has_inf = true;
//                 }
//             }
//         }
//         assert!(!has_nan, "Embedding contains NaN values");
//         assert!(!has_inf, "Embedding contains infinite values");
//         println!("✓ All coordinates are finite");

//         // Coordinate statistics
//         for dim in 0..2 {
//             let min = embedding[dim].iter().copied().fold(f64::INFINITY, f64::min);
//             let max = embedding[dim]
//                 .iter()
//                 .copied()
//                 .fold(f64::NEG_INFINITY, f64::max);
//             let mean = embedding[dim].iter().sum::<f64>() / embedding[dim].len() as f64;
//             println!(
//                 "  Dim {}: min = {:.3}, mean = {:.3}, max = {:.3}",
//                 dim, min, mean, max
//             );
//         }

//         // Compute cluster centroids
//         let mut cluster_centres: FxHashMap<usize, (f64, f64, usize)> = FxHashMap::default();
//         for (i, &label) in labels.iter().enumerate() {
//             let entry = cluster_centres.entry(label).or_insert((0.0, 0.0, 0));
//             entry.0 += embedding[0][i];
//             entry.1 += embedding[1][i];
//             entry.2 += 1;
//         }

//         let mut centroids: Vec<(usize, f64, f64)> = Vec::new();
//         for (label, (sum_x, sum_y, count)) in cluster_centres {
//             centroids.push((label, sum_x / count as f64, sum_y / count as f64));
//         }

//         // Check inter-cluster separation
//         println!("\nCluster analysis:");
//         let mut min_inter_dist = f64::INFINITY;
//         let mut max_inter_dist = f64::NEG_INFINITY;
//         let mut avg_inter_dist = 0.0;
//         let mut count = 0;

//         for i in 0..centroids.len() {
//             for j in (i + 1)..centroids.len() {
//                 let dist = ((centroids[i].1 - centroids[j].1).powi(2)
//                     + (centroids[i].2 - centroids[j].2).powi(2))
//                 .sqrt();
//                 min_inter_dist = min_inter_dist.min(dist);
//                 max_inter_dist = max_inter_dist.max(dist);
//                 avg_inter_dist += dist;
//                 count += 1;
//             }
//         }
//         avg_inter_dist /= count as f64;

//         println!(
//             "  Inter-cluster distances: min = {:.3}, avg = {:.3}, max = {:.3}",
//             min_inter_dist, avg_inter_dist, max_inter_dist
//         );

//         assert!(
//             min_inter_dist > 0.5,
//             "Clusters too close: min distance = {:.3}",
//             min_inter_dist
//         );

//         // Check cluster connectivity
//         for cluster_id in 0..5 {
//             let points: Vec<usize> = labels
//                 .iter()
//                 .enumerate()
//                 .filter(|(_, &l)| l == cluster_id)
//                 .map(|(i, _)| i)
//                 .collect();

//             let threshold = 3.0;
//             let mut visited = vec![false; points.len()];
//             let mut queue = std::collections::VecDeque::new();
//             queue.push_back(0);
//             visited[0] = true;
//             let mut reachable = 1;

//             while let Some(idx) = queue.pop_front() {
//                 let pi = points[idx];

//                 for (other_idx, &other_i) in points.iter().enumerate() {
//                     if !visited[other_idx] {
//                         let dist = ((embedding[0][pi] - embedding[0][other_i]).powi(2)
//                             + (embedding[1][pi] - embedding[1][other_i]).powi(2))
//                         .sqrt();

//                         if dist < threshold {
//                             visited[other_idx] = true;
//                             queue.push_back(other_idx);
//                             reachable += 1;
//                         }
//                     }
//                 }
//             }

//             let connectivity = reachable as f64 / points.len() as f64;
//             println!(
//                 "  Cluster {}: {}/{} connected ({:.1}%)",
//                 cluster_id,
//                 reachable,
//                 points.len(),
//                 connectivity * 100.0
//             );

//             assert!(
//                 connectivity > 0.85,
//                 "Cluster {} fragmented: only {:.1}% connected",
//                 cluster_id,
//                 connectivity * 100.0
//             );
//         }

//         // Compute intra-cluster compactness
//         let mut avg_intra_dist = 0.0;
//         for (cluster_id, cx, cy) in &centroids {
//             let points: Vec<usize> = labels
//                 .iter()
//                 .enumerate()
//                 .filter(|(_, &l)| l == *cluster_id)
//                 .map(|(i, _)| i)
//                 .collect();

//             let intra_dist: f64 = points
//                 .iter()
//                 .map(|&i| ((embedding[0][i] - cx).powi(2) + (embedding[1][i] - cy).powi(2)).sqrt())
//                 .sum::<f64>()
//                 / points.len() as f64;

//             avg_intra_dist += intra_dist;
//         }
//         avg_intra_dist /= 5.0;

//         let separation_ratio = min_inter_dist / avg_intra_dist;
//         println!("  Average intra-cluster distance: {:.3}", avg_intra_dist);
//         println!("  Separation ratio (inter/intra): {:.2}", separation_ratio);

//         assert!(
//             separation_ratio > 0.3,
//             "Poor separation: ratio = {:.2}",
//             separation_ratio
//         );

//         println!("✓ All quality checks passed");
//     }

//     // Note: Reproducibility test skipped for NdArray backend
//     // NdArray doesn't support deterministic seeding for neural network training
//     // For reproducibility testing, use LibTorch backend with LibTorch::seed()

//     #[test]
//     fn parametric_02_different_dimensions() {
//         let (data, _) = create_test_data(15, 10, 42);
//         let device = NdArrayDevice::Cpu;

//         println!("\n=== PARAMETRIC TEST 2: Different Output Dimensions ===");

//         for n_dim in [2, 3, 5] {
//             println!("\nTesting {} dimensions...", n_dim);

//             let params = fast_test_params_custom(n_dim, 15, vec![32], None, None, None);
//             let embedding =
//                 parametric_umap::<f64, TestBackend>(data.as_ref(), &params, &device, 42, false);

//             assert_eq!(embedding.len(), n_dim, "Should have {} dimensions", n_dim);

//             for dim in 0..n_dim {
//                 assert_eq!(
//                     embedding[dim].len(),
//                     75,
//                     "Dimension {} should have 75 samples",
//                     dim
//                 );

//                 let has_non_finite = embedding[dim].iter().any(|&x| !x.is_finite());
//                 assert!(!has_non_finite, "Dimension {} has non-finite values", dim);
//             }

//             println!("  ✓ {} dimensions: all finite, correct shape", n_dim);
//         }
//     }

//     #[test]
//     fn parametric_03_different_architectures() {
//         let (data, _) = create_test_data(15, 10, 42);
//         let device = NdArrayDevice::Cpu;

//         println!("\n=== PARAMETRIC TEST 3: Different Network Architectures ===");

//         let layer_configs = vec![vec![64], vec![128, 64], vec![256, 128, 64]];

//         for hidden_layers in layer_configs {
//             println!("\nTesting architecture: {:?}", hidden_layers);

//             let params = fast_test_params_custom(2, 15, hidden_layers.clone(), None, None, None);

//             let embedding =
//                 parametric_umap::<f64, TestBackend>(data.as_ref(), &params, &device, 42, false);

//             assert_eq!(embedding.len(), 2);
//             assert_eq!(embedding[0].len(), 75);

//             let has_non_finite = embedding[0]
//                 .iter()
//                 .chain(&embedding[1])
//                 .any(|&x| !x.is_finite());
//             assert!(
//                 !has_non_finite,
//                 "Architecture {:?} produced non-finite values",
//                 hidden_layers
//             );

//             println!("  ✓ Architecture {:?}: all finite", hidden_layers);
//         }
//     }

//     #[test]
//     fn parametric_04_correlation_loss() {
//         let (data, _) = create_test_data(15, 10, 42);
//         let device = NdArrayDevice::Cpu;

//         println!("\n=== PARAMETRIC TEST 4: Correlation Loss ===");

//         let params = fast_test_params_custom(2, 15, vec![32], None, None, Some(0.5));
//         let embedding =
//             parametric_umap::<f64, TestBackend>(data.as_ref(), &params, &device, 42, false);

//         assert_eq!(embedding.len(), 2);
//         assert_eq!(embedding[0].len(), 75);

//         let has_non_finite = embedding[0]
//             .iter()
//             .chain(&embedding[1])
//             .any(|&x| !x.is_finite());
//         assert!(
//             !has_non_finite,
//             "Correlation loss produced non-finite values"
//         );

//         println!("  ✓ Correlation loss (λ=0.5): all finite");
//     }

//     #[test]
//     fn parametric_05_min_dist_spread() {
//         let (data, _) = create_test_data(15, 10, 42);
//         let device = NdArrayDevice::Cpu;

//         println!("\n=== PARAMETRIC TEST 5: min_dist and spread ===");

//         let configs = vec![(0.1, 1.0), (0.5, 1.0), (0.1, 2.0)];

//         for (min_dist, spread) in configs {
//             println!("\nTesting min_dist={}, spread={}...", min_dist, spread);

//             let params =
//                 fast_test_params_custom(2, 15, vec![32], Some(min_dist), Some(spread), None);

//             let embedding =
//                 parametric_umap::<f64, TestBackend>(data.as_ref(), &params, &device, 42, false);

//             let has_non_finite = embedding[0]
//                 .iter()
//                 .chain(&embedding[1])
//                 .any(|&x| !x.is_finite());
//             assert!(
//                 !has_non_finite,
//                 "min_dist={}, spread={} produced non-finite values",
//                 min_dist, spread
//             );

//             println!("  ✓ min_dist={}, spread={}: all finite", min_dist, spread);
//         }
//     }

//     #[test]
//     fn parametric_06_small_dataset() {
//         let data = Mat::from_fn(10, 5, |i, j| (i as f64 + j as f64) * 0.1);
//         let device = NdArrayDevice::Cpu;

//         println!("\n=== PARAMETRIC TEST 6: Small Dataset ===");
//         println!("Data: {} samples, {} features", data.nrows(), data.ncols());

//         let params = fast_test_params_custom(2, 5, vec![32], None, None, None);

//         let embedding =
//             parametric_umap::<f64, TestBackend>(data.as_ref(), &params, &device, 42, false);

//         assert_eq!(embedding.len(), 2);
//         assert_eq!(embedding[0].len(), 10);

//         let has_non_finite = embedding[0]
//             .iter()
//             .chain(&embedding[1])
//             .any(|&x| !x.is_finite());
//         assert!(!has_non_finite, "Small dataset produced non-finite values");

//         println!("  ✓ Small dataset (10 samples): all finite");
//     }
// }
