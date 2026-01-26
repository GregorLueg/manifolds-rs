#![allow(clippy::needless_range_loop)] // I like loops ... !

pub mod data;
pub mod prelude;
pub mod training;
pub mod utils;

#[cfg(feature = "parametric")]
pub mod parametric;

use ann_search_rs::hnsw::{HnswIndex, HnswState};
use ann_search_rs::nndescent::{ApplySortedUpdates, NNDescent, NNDescentQuery};
use ann_search_rs::utils::dist::SimdDistance;
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

#[cfg(feature = "parametric")]
use burn::tensor::{backend::AutodiffBackend, Element};

use crate::data::graph::*;
use crate::data::init::*;
use crate::data::nearest_neighbours::*;
use crate::data::structures::*;
use crate::training::optimiser::*;
use crate::training::*;
use crate::utils::fft::FftwFloat;

#[cfg(feature = "parametric")]
use crate::parametric::model::*;
#[cfg(feature = "parametric")]
use crate::parametric::parametric_train::*;

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
    init_range: Option<T>,
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
    ///   to use. Defaults to `"annoy"`.
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
        init_range: Option<T>,
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
            init_range,
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
            init_range: None,
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
    let init_type = parse_initilisation(
        &umap_params.initialisation,
        umap_params.randomised,
        umap_params.init_range,
    )
    .unwrap_or(EmbdInit::RandomInit { range: None });
    let optimiser = parse_umap_optimiser(&umap_params.optimiser).unwrap_or_default();

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
            umap_params.initialisation
        );
    }

    let start_layout = Instant::now();

    let mut embd = initialise_embedding(&init_type, umap_params.n_dim, seed as u64, &graph, data);

    let graph_adj = coo_to_adjacency_list(&graph);

    if verbose {
        println!(
            "Optimising embedding via {} ({} epochs) on {} edges...",
            match optimiser {
                UmapOptimiser::Adam => "Adam",
                UmapOptimiser::Sgd => "SGD",
                UmapOptimiser::AdamParallel => "Adam (multi-threaded)",
            },
            umap_params.optim_params.n_epochs,
            graph.col_indices.len().separate_with_underscores()
        );
    }

    match optimiser {
        UmapOptimiser::Adam => optimise_embedding_adam(
            &mut embd,
            &graph_adj,
            &umap_params.optim_params,
            seed as u64,
            verbose,
        ),
        UmapOptimiser::Sgd => {
            optimise_embedding_sgd(
                &mut embd,
                &graph_adj,
                &umap_params.optim_params,
                seed as u64,
                verbose,
            );
        }
        UmapOptimiser::AdamParallel => {
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
    pub init_range: Option<T>,
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_dim: Option<usize>,
        perplexity: Option<T>,
        init_range: Option<T>,
        lr: Option<T>,
        n_epochs: Option<usize>,
        ann_type: Option<String>,
        theta: Option<T>,
        n_interp_points: Option<usize>,
    ) -> Self {
        let n_dim = n_dim.unwrap_or(2);
        let perplexity = perplexity.unwrap_or_else(|| T::from_f64(30.0).unwrap());
        let lr = lr.unwrap_or_else(|| T::from_f64(200.0).unwrap());
        let n_epochs = n_epochs.unwrap_or(1000);
        let ann_type = ann_type.unwrap_or_else(|| "hnsw".to_string());
        let theta = theta.unwrap_or_else(|| T::from_f64(0.5).unwrap());
        let n_interp_points = n_interp_points.unwrap_or(3);

        Self {
            n_dim,
            perplexity,
            ann_type,
            initialisation: "pca".to_string(),
            init_range,
            nn_params: NearestNeighbourParams::default(),
            optim_params: TsneOptimParams {
                n_epochs,
                lr,
                early_exag_iter: 250,
                early_exag_factor: T::from_f64(12.0).unwrap(),
                theta,
                n_interp_points,
            },
            randomised_init: true,
        }
    }
}

/// Construct affinity graph for t-SNE from high-dimensional data
///
/// Performs the following steps:
///
/// 1. Runs k-nearest neighbour search where k = 3 × perplexity
/// 2. Computes Gaussian affinities P(j|i) via binary search for target entropy
/// 3. Symmetrises to joint probabilities: P_ij = (P(j|i) + P(i|j)) / 2N
///
/// ### Params
///
/// * `data` - Input data matrix (samples × features)
/// * `perplexity` - Target perplexity (effective number of neighbours,
///   typical: 5-50)
/// * `ann_type` - ANN algorithm: `"annoy"` (default), `"hnsw"` or `"nndescent"`
/// * `nn_params` - Nearest neighbour search parameters
/// * `seed` - Random seed for reproducibility
/// * `verbose` - Print progress information
///
/// ### Returns
///
/// Tuple of:
///
/// - `SparseGraph<T>` containing symmetric joint probabilities P_ij
/// - `Vec<Vec<usize>>` k-nearest neighbour indices for each point
/// - `Vec<Vec<T>>` k-nearest neighbour distances for each point
///
/// # Notes
///
/// The k value is automatically set to `3 × perplexity`, clamped between 5 and
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
        // euclidean is already squared; cosine not
        nn_params.dist_metric == "euclidean",
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
///    - Repulsive forces: Barnes-Hut approximation or FFT-accelerated
///      interpolation.
///    - Early exaggeration (first 250 iterations)
///    - Momentum switching at iteration 250
///
/// ### Params
///
/// * `data` - Input data matrix (samples × features)
/// * `params` - t-SNE parameters controlling algorithm behaviour
/// * `approx_type` - Type of approximation to use for repulsive forces.
///   Options: `"barnes_hut" | "bh"`, `"fft"`
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
/// let params = TsneParams::new(None, None, None, None, None, None);
/// let embedding = tsne(data.as_ref(), &params, 42, true);
/// // embedding[0] contains x-coordinates for all points
/// // embedding[1] contains y-coordinates for all points
/// ```
///
/// ### References
///
/// - van der Maaten & Hinton (2008): "Visualizing Data using t-SNE"
/// - van der Maaten (2014): "Accelerating t-SNE using Tree-Based Algorithms"
/// - Linderman et al. (2019): " Fast interpolation-based t-SNE for improved
///   visualization of single-cell RNA-seq data"
pub fn tsne<T>(
    data: MatRef<T>,
    params: &TsneParams<T>,
    approx_type: &str,
    seed: usize,
    verbose: bool,
) -> Vec<Vec<T>>
where
    T: Float
        + FftwFloat
        + Default
        + ComplexField
        + RealField
        + Sum
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + SimdDistance,
    HnswIndex<T>: HnswState<T>,
    StandardNormal: Distribution<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    assert!(
        params.n_dim == 2,
        "At the moment, this tSNE implementation only supports n_dim = 2"
    );

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
    let init_type = parse_initilisation(
        &params.initialisation,
        params.randomised_init,
        params.init_range,
    )
    .unwrap_or(EmbdInit::PcaInit {
        randomised: false,
        range: Some(T::from_f64(1e-4).unwrap()),
    });

    if verbose {
        println!("Initialising embedding via PCA...");
    }

    let mut embd = initialise_embedding(&init_type, params.n_dim, seed as u64, &graph, data);

    // parse the optimisation type
    let tsne_approx = parse_tsne_optimiser(approx_type).unwrap_or_default();

    // 3. optimise
    let start_optim = Instant::now();
    match tsne_approx {
        TsneOpt::BarnesHut => {
            if verbose {
                println!(
                    "Optimising via Barnes-Hut t-SNE ({} epochs)...",
                    params.optim_params.n_epochs
                );
            }

            optimise_bh_tsne(&mut embd, &params.optim_params, &graph, verbose);
        }
        TsneOpt::Fft => {
            if verbose {
                println!(
                    "Optimising via Fast Fourier transformation Interpolation-based t-SNE ({} epochs)...",
                    params.optim_params.n_epochs
                );
            }

            optimise_fft_tsne(&mut embd, &params.optim_params, &graph, verbose);
        }
    }

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
