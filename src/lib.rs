//! Dimensionality reduction algorithms including UMAP, t-SNE, PHATE, Diffusion
//! Maps and PacMAP.
//!
//! Provides both standard and approximate nearest-neighbour-based graph
//! construction, multiple optimisers, and (optionally) parametric UMAP via a
//! neural network encoder.
//!
//! Additionally, optional GPU-accelerated versions (in terms of kNN search)
//! can be used when the right feature flags are active.

#![allow(clippy::needless_range_loop)] // I like loops ... !
#![warn(missing_docs)]

pub mod data;
pub mod errors;
pub mod prelude;
pub mod training;
pub mod utils;

#[cfg(feature = "parametric")]
pub mod parametric;

use ann_search_rs::cpu::hnsw::{HnswIndex, HnswState};
use ann_search_rs::cpu::nndescent::{ApplySortedUpdates, NNDescent, NNDescentQuery};
use ann_search_rs::utils::dist::parse_ann_dist;
use faer::MatRef;
use rand_distr::{Distribution, StandardNormal};
use std::{default::Default, time::Instant};
use thousands::*;

#[cfg(feature = "parametric")]
use burn::tensor::{backend::AutodiffBackend, Element};
#[cfg(feature = "parametric")]
use num_traits::ToPrimitive;

#[cfg(feature = "gpu")]
use ann_search_rs::gpu::nndescent_gpu::NNDescentGpu;
#[cfg(feature = "gpu")]
use ann_search_rs::gpu::traits_gpu::AnnSearchGpuFloat;
#[cfg(feature = "gpu")]
use cubecl::prelude::*;

use crate::data::graph::*;
use crate::data::init::*;
use crate::data::nearest_neighbours::*;
use crate::data::pacmap_pairs::*;
use crate::data::structures::*;
use crate::prelude::*;
use crate::training::mds_optimiser::*;
use crate::training::pacmap_optimiser::{
    optimise_pacmap, optimise_pacmap_parallel, PacMapOptimiser,
};
use crate::training::tsne_optimiser::*;
use crate::training::umap_optimisers::*;
use crate::utils::diffusions::*;
use crate::utils::math::compute_largest_eigenpairs_lanczos;
use crate::utils::potentials::compute_potential_distances;
use crate::utils::sparse_ops::matrix_power;

#[cfg(feature = "parametric")]
use crate::parametric::model::*;
#[cfg(feature = "parametric")]
use crate::parametric::parametric_train::*;
#[cfg(feature = "gpu")]
use crate::training::umap_optimiser_gpu::*;
#[cfg(feature = "fft_tsne")]
use crate::utils::fft::FftwFloat;

///////////
// Types //
///////////

/// Type for the pre-computed kNN
///
/// ### Fields
///
/// * `0` - Should be the indices of the nearest neighbours excluding self
/// * `1` - Should be the distances to the nearest neighbours excluding self
pub type PreComputedKnn<T> = Option<(Vec<Vec<usize>>, Vec<Vec<T>>)>;

/// Result of the tSNE graph
///
/// ### Fields
///
/// * `0` - The coordinate list
/// * `1` - The nearest neighbours
/// * `2` - The distances to the nearest neighbours
///
/// ### Errors
///
/// Potential errors from the generation of the graph
pub type TsneGraph<T> = Result<(CoordinateList<T>, Vec<Vec<usize>>, Vec<Vec<T>>), ManifoldsError>;

//////////
// Umap //
//////////

/// Main Config structure with all of the possible sub configurations
#[derive(Debug, Clone)]
pub struct UmapParams<T> {
    /// How many dimensions to return
    pub n_dim: usize,
    /// Number of neighbours
    pub k: usize,
    /// Which optimiser to use. Defaults to `"adam_parallel"`.
    pub optimiser: String,
    /// (Approximate) Nearest neighbour method. One of `"exhaustive"`, `"ivf"`,
    /// `"hnsw"`, `"nndescent"`, `"annoy"`, `"kmknn"` or `"balltree"`.
    pub ann_type: String,
    /// Which embedding initialisation to use. Defaults to spectral clustering.
    pub initialisation: String,
    /// Optional initialisation range to use
    pub init_range: Option<T>,
    /// Nearest neighbour parameters.
    pub nn_params: NearestNeighbourParams<T>,
    /// Parameters for the UMAP graph generation.
    pub umap_graph_params: UmapGraphParams<T>,
    /// Parameters to use for the UMAP optimiser.
    pub optim_params: UmapOptimParams<T>,
    /// Shall randomised SVC be used for PCA-based embedding
    pub randomised: bool,
}

impl<T> Default for UmapParams<T>
where
    T: ManifoldsFloat,
{
    fn default() -> Self {
        Self {
            n_dim: 2,
            k: 15,
            optimiser: "adam_parallel".to_string(),
            ann_type: "kmknn".to_string(),
            initialisation: "spectral".to_string(),
            init_range: None,
            nn_params: NearestNeighbourParams::default(),
            optim_params: UmapOptimParams::default(),
            umap_graph_params: UmapGraphParams::default(),
            randomised: false,
        }
    }
}

impl<T> UmapParams<T>
where
    T: ManifoldsFloat,
{
    /// Generate new UMAP parameters with full control over every field.
    ///
    /// This constructor exposes every field directly. For sensible defaults,
    /// use [`UmapParams::default`] or [`UmapParams::new_default_2d`] instead.
    ///
    /// ### Params
    ///
    /// * `n_dim` - How many dimensions to return.
    /// * `k` - How many neighbours to consider.
    /// * `optimiser` - Which optimiser to use, e.g. `"adam_parallel"`.
    /// * `ann_type` - (Approximate) nearest neighbour method. One of
    ///   `"exhaustive"`, `"ivf"`, `"hnsw"`, `"nndescent"`, `"annoy"`,
    ///   `"kmknn"` or `"balltree"`.
    /// * `initialisation` - Which embedding initialisation to use, e.g.
    ///   `"spectral"`.
    /// * `init_range` - Optional initialisation range.
    /// * `nn_params` - Nearest neighbour parameters.
    /// * `optim_params` - Parameters for the UMAP optimiser.
    /// * `umap_graph_params` - Parameters for the UMAP graph generation.
    /// * `randomised` - Whether randomised SVD is used for PCA-based embedding.
    ///
    /// ### Returns
    ///
    /// A fully specified set of UMAP parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_dim: usize,
        k: usize,
        optimiser: String,
        ann_type: String,
        initialisation: String,
        init_range: Option<T>,
        nn_params: NearestNeighbourParams<T>,
        optim_params: UmapOptimParams<T>,
        umap_graph_params: UmapGraphParams<T>,
        randomised: bool,
    ) -> Self {
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

    /// Default parameters for standard 2D visualisation.
    ///
    /// Returns the default parameters but lets you tune `min_dist` and
    /// `spread`, which feed into the optimiser parameters and control how
    /// tightly points are packed in the embedding.
    ///
    /// ### Params
    ///
    /// * `min_dist` - Minimum distance between data points. Defaults to `0.5`.
    /// * `spread` - Spread parameter. Defaults to `1.0`.
    ///
    /// ### Returns
    ///
    /// Hopefully sensible standard parameters for 2D visualisation.
    pub fn new_default_2d(min_dist: Option<T>, spread: Option<T>) -> Self {
        let min_dist = min_dist.unwrap_or(T::from_f64(0.5).unwrap());
        let spread = spread.unwrap_or(T::from_f64(1.0).unwrap());

        Self {
            optim_params: UmapOptimParams::from_min_dist_spread(
                min_dist, spread, None, None, None, None, None, None, None,
            ),
            ..Self::default()
        }
    }
}

/// Helper function to generate the UMAP graph
///
/// ### Params
///
/// * `data` - Input data matrix (samples × features)
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances. Needs
///   to be a tuple of `(Vec<Vec<usize>>, Vec<Vec<T>>)` with indices and
///   distances excluding self.
/// * `k` - Number of nearest neighbours (typically 15-50).
/// * `ann_type` - (Approximate) Nearest neighbour method: `"exhaustive"`,
///   `"kmknn"`, `"balltree"`, `"annoy"`, `"hnsw"`, or `"nndescent"`. If you
///   provide a weird string, the function will default to `"kmknn"`
/// * `umap_params` - UMAP-specific parameters (bandwidth, local_connectivity,
///   mix_weight)
/// * `nn_params` - Nearest neighbour parameters for nearest neighbour search.
/// * `seed` - Random seed
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Tuple of (graph, knn_indices, knn_dist) for use in optimisation
#[allow(clippy::too_many_arguments)]
pub fn construct_umap_graph<T>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    k: usize,
    ann_type: String,
    umap_params: &UmapGraphParams<T>,
    nn_params: &NearestNeighbourParams<T>,
    n_epochs: usize,
    seed: usize,
    verbose: usize,
) -> UmapGraphResults<T>
where
    T: ManifoldsFloat,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let verbosity = parse_verbosity_level(verbose);

    let (knn_indices, knn_dist) = match precomputed_knn {
        Some((indices, distances)) => {
            if verbosity.normal_verbosity() {
                println!("Using precomputed kNN graph...");
            }
            (indices, distances)
        }
        None => {
            if verbosity.normal_verbosity() {
                println!(
                    "Running (approximate) nearest neighbour search using {}...",
                    ann_type
                );
            }
            let start_knn = Instant::now();
            let result = run_ann_search(data, k, ann_type, nn_params, seed, verbose)?;
            if verbosity.normal_verbosity() {
                println!("kNN search done in: {:.2?}.", start_knn.elapsed());
            }
            result
        }
    };

    if verbosity.normal_verbosity() {
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

    if verbosity.normal_verbosity() {
        println!(
            "Finalised graph generation in {:.2?}.",
            start_graph.elapsed()
        );
    }

    Ok((graph, knn_indices, knn_dist))
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
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances. Needs
///   to be a tuple of `(Vec<Vec<usize>>, Vec<Vec<T>>)` with indices and
///   distances excluding self.
/// * `umap_params` - The UMAP parameters.
/// * `seed` - Seed for reproducibility.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>` where outer vector has length
/// `n_dim` and inner vectors have length `n_samples`. Each outer element
/// represents one embedding dimension.
pub fn umap<T>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    umap_params: &UmapParams<T>,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
    HnswIndex<T>: HnswState<T>,
    StandardNormal: Distribution<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let verbosity = parse_verbosity_level(verbose);

    // parse various parameters
    let init_type = parse_initilisation(
        &umap_params.initialisation,
        umap_params.randomised,
        umap_params.init_range,
    )
    .unwrap_or_else(|| {
        println!(
            "Unknown initialisation provided: {:?}. Defaulting to PCA.",
            umap_params.initialisation,
        );
        EmbdInit::PcaInit {
            range: None,
            randomised: true,
        }
    });
    let optimiser = parse_umap_optimiser(&umap_params.optimiser).unwrap_or_default();

    if verbosity.normal_verbosity() {
        println!(
            "Running umap with alpha: {:.2?} and beta: {:.2?}",
            umap_params.optim_params.a, umap_params.optim_params.b
        );
    }

    let (graph, _, _) = construct_umap_graph(
        data,
        precomputed_knn,
        umap_params.k,
        umap_params.ann_type.clone(),
        &umap_params.umap_graph_params,
        &umap_params.nn_params,
        umap_params.optim_params.n_epochs,
        seed,
        verbose,
    )?;

    if verbosity.normal_verbosity() {
        println!(
            "Initialising embedding via {} layout...",
            umap_params.initialisation
        );
    }

    let start_layout = Instant::now();

    let mut embd = initialise_embedding(&init_type, umap_params.n_dim, seed as u64, &graph, data)?;

    let graph_adj = coo_to_adjacency_list(&graph);

    if verbosity.normal_verbosity() {
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
        )?,
        UmapOptimiser::Sgd => {
            optimise_embedding_sgd(
                &mut embd,
                &graph_adj,
                &umap_params.optim_params,
                seed as u64,
                verbose,
            )?;
        }
        UmapOptimiser::AdamParallel => {
            optimise_embedding_adam_parallel(
                &mut embd,
                &graph_adj,
                &umap_params.optim_params,
                seed as u64,
                verbose,
            )?;
        }
    }

    let end_layout = start_layout.elapsed();

    if verbosity.normal_verbosity() {
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

    Ok(transposed)
}

//////////
// tSNE //
//////////

/// Main configuration for t-SNE dimensionality reduction
#[derive(Debug, Clone)]
pub struct TsneParams<T> {
    ///  Number of output dimensions (typically 2)
    pub n_dim: usize,
    /// Perplexity parameter controlling neighbourhood size (typical: 5-50)
    pub perplexity: T,
    /// (Approximate) Nearest neighbour method. One of `"exhaustive"`, `"ivf"`,
    /// `"hnsw"`, `"nndescent"`, `"annoy"`, `"kmknn"` or `"balltree"`.
    pub ann_type: String,
    /// Embedding initialisation method: `"pca"`, `"random"`, or `"spectral"`
    pub initialisation: String,
    /// Optional initialisation range
    pub init_range: Option<T>,
    /// Nearest neighbour parameters
    pub nn_params: NearestNeighbourParams<T>,
    /// tSNE optimisation parameters
    pub optim_params: TsneOptimParams<T>,
    /// Use randomised SVD for PCA initialisation
    pub randomised_init: bool,
}

impl<T> Default for TsneParams<T>
where
    T: ManifoldsFloat,
{
    fn default() -> Self {
        Self {
            n_dim: 2,
            perplexity: T::from_f64(30.0).unwrap(),
            ann_type: "kmknn".to_string(),
            initialisation: "pca".to_string(),
            init_range: None,
            nn_params: NearestNeighbourParams::default(),
            optim_params: TsneOptimParams {
                n_epochs: 1000,
                lr: None,
                early_exag_iter: 250,
                early_exag_factor: T::from_f64(12.0).unwrap(),
                late_exag_factor: None,
                theta: T::from_f64(0.5).unwrap(),
                n_interp_points: 3,
            },
            randomised_init: true,
        }
    }
}

impl<T> TsneParams<T>
where
    T: ManifoldsFloat,
{
    /// Create new t-SNE parameters with full control over every field.
    ///
    /// This constructor exposes every field directly, including the
    /// optimiser parameters. For sensible defaults, use
    /// [`TsneParams::default`] or [`TsneParams::new_default_2d`] instead.
    ///
    /// ### Params
    ///
    /// * `n_dim` - Number of output dimensions.
    /// * `perplexity` - Perplexity parameter controlling neighbourhood size
    ///   (typical: 5-50).
    /// * `ann_type` - (Approximate) nearest neighbour method. One of
    ///   `"exhaustive"`, `"ivf"`, `"hnsw"`, `"nndescent"`, `"annoy"`,
    ///   `"kmknn"` or `"balltree"`.
    /// * `initialisation` - Embedding initialisation method: `"pca"`,
    ///   `"random"`, or `"spectral"`.
    /// * `init_range` - Optional initialisation range to fix the initial
    ///   embedding between certain values.
    /// * `nn_params` - Nearest neighbour parameters.
    /// * `optim_params` - t-SNE optimisation parameters. These cover the
    ///   learning rate, number of epochs, early/late exaggeration, the
    ///   Barnes-Hut `theta`, and the number of FFT interpolation points.
    /// * `randomised_init` - Whether randomised SVD is used for PCA
    ///   initialisation.
    ///
    /// ### Returns
    ///
    /// A fully specified set of t-SNE parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_dim: usize,
        perplexity: T,
        ann_type: String,
        initialisation: String,
        init_range: Option<T>,
        nn_params: NearestNeighbourParams<T>,
        optim_params: TsneOptimParams<T>,
        randomised_init: bool,
    ) -> Self {
        Self {
            n_dim,
            perplexity,
            ann_type,
            initialisation,
            init_range,
            nn_params,
            optim_params,
            randomised_init,
        }
    }

    /// Default parameters for standard 2D visualisation.
    ///
    /// Returns the default parameters but lets you tune `perplexity`, the
    /// characteristic t-SNE knob controlling neighbourhood size.
    ///
    /// ### Params
    ///
    /// * `perplexity` - Perplexity parameter (typical: 5-50). Defaults to
    ///   `30.0`.
    ///
    /// ### Returns
    ///
    /// Hopefully sensible standard parameters for 2D visualisation.
    pub fn new_default_2d(perplexity: Option<T>) -> Self {
        let perplexity = perplexity.unwrap_or_else(|| T::from_f64(30.0).unwrap());

        Self {
            perplexity,
            ..Self::default()
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
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances. Needs
///   to be a tuple of `(Vec<Vec<usize>>, Vec<Vec<T>>)` with indices and
///   distances excluding self.
/// * `perplexity` - Target perplexity (effective number of neighbours,
///   typical: 5-50)
/// * `ann_type` - (Approximate) Nearest neighbour method: `"exhaustive"`,
///   `"kmknn"`, `"balltree"`, `"annoy"`, `"hnsw"`, or `"nndescent"`. If you
///   provide a weird string, the function will default to `"kmknn"`
/// * `nn_params` - Nearest neighbour search parameters
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Tuple of:
///
/// - `CoordinateList<T>` containing symmetric joint probabilities P_ij
/// - `Vec<Vec<usize>>` k-nearest neighbour indices for each point
/// - `Vec<Vec<T>>` k-nearest neighbour distances for each point
///
/// # Notes
///
/// The k value is automatically set to `3 × perplexity`, clamped between 5 and
/// n-1. This is standard practice in t-SNE implementations.
pub fn construct_tsne_graph<T>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    perplexity: T,
    ann_type: String,
    nn_params: &NearestNeighbourParams<T>,
    seed: usize,
    verbose: usize,
) -> TsneGraph<T>
where
    T: ManifoldsFloat,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let verbosity = parse_verbosity_level(verbose);

    let (knn_indices, knn_dist) = match precomputed_knn {
        Some((indices, distances)) => {
            if verbosity.normal_verbosity() {
                println!("Using precomputed kNN graph...");
            }
            (indices, distances)
        }
        None => {
            let k_float = perplexity * T::from_f64(3.0).unwrap();
            let k = k_float.to_usize().unwrap().max(5).min(data.nrows() - 1);

            if verbosity.normal_verbosity() {
                println!("Running kNN search (k={}) using {}...", k, ann_type);
            }

            let start_knn = Instant::now();
            let result = run_ann_search(data, k, ann_type, nn_params, seed, verbose)?;

            if verbosity.normal_verbosity() {
                println!("kNN search done in: {:.2?}.", start_knn.elapsed());
            }

            result
        }
    };

    if verbosity.normal_verbosity() {
        println!("Computing Gaussian affinities and symmetrising...");
    }

    let start_graph = Instant::now();

    let directed_graph = gaussian_knn_affinities(
        &knn_indices,
        &knn_dist,
        perplexity,
        T::from_f64(1e-5).unwrap(),
        200,
        nn_params.dist_metric == "euclidean",
    )?;

    let graph = symmetrise_affinities_tsne(directed_graph);

    if verbosity.normal_verbosity() {
        println!(
            "Finalised graph generation in {:.2?}.",
            start_graph.elapsed()
        );
    }

    Ok((graph, knn_indices, knn_dist))
}

/// Run t-SNE dimensionality reduction
///
/// t-Distributed Stochastic Neighbour Embedding (t-SNE) is a technique for
/// visualising high-dimensional data by reducing it to 2 dimensions. This
/// version supports both Barnes-Hut approximation and FFT-accelerated
/// interpolation for repulsive forces.
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
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances. Needs
///   to be a tuple of `(Vec<Vec<usize>>, Vec<Vec<T>>)` with indices and
///   distances excluding self.
/// * `params` - t-SNE parameters controlling algorithm behaviour
/// * `approx_type` - Type of approximation to use for repulsive forces.
///   Options: `"barnes_hut" | "bh"`, `"fft"`
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>` where outer vector has length
/// `n_dim` and inner vectors have length `n_samples`. Each outer element
/// represents one embedding dimension.
///
/// ### References
///
/// - van der Maaten & Hinton (2008): "Visualizing Data using t-SNE"
/// - van der Maaten (2014): "Accelerating t-SNE using Tree-Based Algorithms"
/// - Linderman et al. (2019): " Fast interpolation-based t-SNE for improved
///   visualization of single-cell RNA-seq data"
#[cfg(feature = "fft_tsne")]
pub fn tsne<T>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    params: &TsneParams<T>,
    approx_type: &str,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat + FftwFloat,
    HnswIndex<T>: HnswState<T>,
    StandardNormal: Distribution<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    if params.n_dim != 2 {
        return Err(ManifoldsError::IncorrectDim {
            n_dim: params.n_dim,
        });
    }

    let verbosity = parse_verbosity_level(verbose);

    // 1. graph construction
    let (graph, _, _) = construct_tsne_graph(
        data,
        precomputed_knn,
        params.perplexity,
        params.ann_type.clone(),
        &params.nn_params,
        seed,
        verbose,
    )?;

    if verbosity.normal_verbosity() {
        println!("Initialising embedding via {}...", &params.initialisation);
    }

    // 2. initialise embedding
    let init_type = parse_initilisation(
        &params.initialisation,
        params.randomised_init,
        params.init_range,
    )
    .unwrap_or_else(|| {
        println!(
            "Unknown initialisation provided: {:?}. Defaulting to PCA.",
            params.initialisation,
        );
        EmbdInit::PcaInit {
            range: Some(T::from_f64(1e-2).unwrap()),
            randomised: true,
        }
    });

    let mut embd = initialise_embedding(&init_type, params.n_dim, seed as u64, &graph, data)?;

    // parse the optimisation type
    let tsne_approx = parse_tsne_optimiser(approx_type).unwrap_or_default();

    // 3. optimise
    let start_optim = Instant::now();
    match tsne_approx {
        TsneOpt::BarnesHut => {
            if verbosity.normal_verbosity() {
                println!(
                    "Optimising via Barnes-Hut t-SNE ({} epochs)...",
                    params.optim_params.n_epochs
                );
            }
            optimise_bh_tsne(&mut embd, &params.optim_params, &graph, verbose);
        }
        #[cfg(feature = "fft_tsne")]
        TsneOpt::Fft => {
            if verbosity.normal_verbosity() {
                println!(
                    "Optimising via FFT Interpolation-based t-SNE ({} epochs)...",
                    params.optim_params.n_epochs
                );
            }
            let _ = optimise_fft_tsne(&mut embd, &params.optim_params, &graph, verbose);
        }
    }

    if verbosity.normal_verbosity() {
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

    Ok(transposed)
}

/// Run t-SNE dimensionality reduction
///
/// t-Distributed Stochastic Neighbour Embedding (t-SNE) is a technique for
/// visualising high-dimensional data by reducing it to 2 dimensions. This
/// version supports both Barnes-Hut approximation and FFT-accelerated
/// interpolation for repulsive forces.
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
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances. Needs
///   to be a tuple of `(Vec<Vec<usize>>, Vec<Vec<T>>)` with indices and
///   distances excluding self.
/// * `params` - t-SNE parameters controlling algorithm behaviour
/// * `approx_type` - Type of approximation to use for repulsive forces.
///   Options: `"barnes_hut" | "bh"`, `"fft"`
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>` where outer vector has length
/// `n_dim` and inner vectors have length `n_samples`. Each outer element
/// represents one embedding dimension.
///
/// ### References
///
/// - van der Maaten & Hinton (2008): "Visualizing Data using t-SNE"
/// - van der Maaten (2014): "Accelerating t-SNE using Tree-Based Algorithms"
/// - Linderman et al. (2019): " Fast interpolation-based t-SNE for improved
///   visualization of single-cell RNA-seq data"
#[cfg(not(feature = "fft_tsne"))]
pub fn tsne<T>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    params: &TsneParams<T>,
    approx_type: &str,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
    HnswIndex<T>: HnswState<T>,
    StandardNormal: Distribution<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    if params.n_dim != 2 {
        return Err(ManifoldsError::IncorrectDim {
            n_dim: params.n_dim,
        });
    }
    let verbosity = parse_verbosity_level(verbose);

    // 1. graph construction
    let (graph, _, _) = construct_tsne_graph(
        data,
        precomputed_knn,
        params.perplexity,
        params.ann_type.clone(),
        &params.nn_params,
        seed,
        verbose,
    )?;

    // 2. initialise embedding
    if verbosity.normal_verbosity() {
        println!("Initialising embedding via {}...", &params.initialisation);
    }

    let init_type = parse_initilisation(
        &params.initialisation,
        params.randomised_init,
        params.init_range,
    )
    .unwrap_or_else(|| {
        println!(
            "Unknown initialisation provided: {:?}. Defaulting to PCA.",
            params.initialisation,
        );
        EmbdInit::PcaInit {
            range: Some(T::from_f64(1e-2).unwrap()),
            randomised: true,
        }
    });

    let mut embd = initialise_embedding(&init_type, params.n_dim, seed as u64, &graph, data)?;

    // parse the optimisation type
    let tsne_approx = parse_tsne_optimiser(approx_type).unwrap_or_default();

    // 3. optimise
    let start_optim = Instant::now();
    match tsne_approx {
        TsneOpt::BarnesHut => {
            if verbosity.normal_verbosity() {
                println!(
                    "Optimising via Barnes-Hut t-SNE ({} epochs)...",
                    params.optim_params.n_epochs
                );
            }
            optimise_bh_tsne(&mut embd, &params.optim_params, &graph, verbose);
        }
        #[cfg(not(feature = "fft_tsne"))]
        TsneOpt::Fft => {
            panic!("FFT-accelerated t-SNE not available. Recompile with 'fft_tsne' feature or use 'barnes_hut' approximation.");
        }
    }

    if verbosity.normal_verbosity() {
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

    Ok(transposed)
}

///////////
// PHATE //
///////////

/// PHATE parameters
#[derive(Debug, Clone)]
pub struct PhateParams<T> {
    /// Number of output dimensions (default: 2)
    pub n_dim: usize,
    /// Number of neighbours to use
    pub k: usize,
    /// (Approximate) Nearest neighbour method. One of `"exhaustive"`, `"ivf"`,
    /// `"hnsw"`, `"nndescent"`, `"annoy"`, `"kmknn"`, or `"balltree"`.
    pub ann_type: String,
    /// Nearest neighbour search parameters to use
    pub ann_params: NearestNeighbourParams<T>,
    /// Diffusion parameters to use
    pub diffusion_params: PhateDiffusionParams<T>,
    /// Which MDS implementation to use
    pub mds_method: String,
    /// Optional number of iterations for MDS fitting
    pub mds_iter: Option<usize>,
    /// Shall randomised SVD be used for PCA-based initialisation
    pub randomised: bool,
}

impl<T> Default for PhateParams<T>
where
    T: ManifoldsFloat,
{
    fn default() -> Self {
        let diffusion_params = PhateDiffusionParams::new(
            Some(T::from_f64(40.0).unwrap()),
            T::from_f64(1.0).unwrap(),
            T::from_f64(1e-4).unwrap(),
            "average".to_string(),
            None,
            "spectral".to_string(),
            None,
            None,
            None,
            T::from_f64(1.0).unwrap(),
        );

        Self {
            n_dim: 2,
            k: 5,
            ann_type: "kmknn".to_string(),
            ann_params: NearestNeighbourParams::default(),
            diffusion_params,
            mds_method: "sgd_dense".to_string(),
            mds_iter: None,
            randomised: true,
        }
    }
}

impl<T> PhateParams<T>
where
    T: ManifoldsFloat,
{
    /// Create new PHATE parameters with full control over every field.
    ///
    /// This constructor exposes every field directly, including the
    /// diffusion parameters. For sensible defaults, use
    /// [`PhateParams::default`] or [`PhateParams::new_default_2d`] instead.
    ///
    /// ### Params
    ///
    /// * `n_dim` - Number of output dimensions.
    /// * `k` - Number of nearest neighbours.
    /// * `ann_type` - (Approximate) nearest neighbour method. One of
    ///   `"exhaustive"`, `"ivf"`, `"hnsw"`, `"nndescent"`, `"annoy"`,
    ///   `"kmknn"` or `"balltree"`.
    /// * `ann_params` - Nearest neighbour search parameters.
    /// * `diffusion_params` - Diffusion parameters. These cover the alpha
    ///   decay, kernel bandwidth scaling, graph symmetry, landmarks, SVD
    ///   components, diffusion time selection, and the informational distance
    ///   constant `gamma`.
    /// * `mds_method` - Which MDS implementation to use, e.g. `"sgd_dense"`.
    /// * `mds_iter` - Optional number of iterations for MDS fitting; `None`
    ///   uses the MDS default.
    /// * `randomised` - Whether randomised SVD is used for PCA-based
    ///   initialisation.
    ///
    /// ### Returns
    ///
    /// A fully specified set of PHATE parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_dim: usize,
        k: usize,
        ann_type: String,
        ann_params: NearestNeighbourParams<T>,
        diffusion_params: PhateDiffusionParams<T>,
        mds_method: String,
        mds_iter: Option<usize>,
        randomised: bool,
    ) -> Self {
        Self {
            n_dim,
            k,
            ann_type,
            ann_params,
            diffusion_params,
            mds_method,
            mds_iter,
            randomised,
        }
    }

    /// Default parameters for standard 2D visualisation.
    ///
    /// Returns the default parameters but lets you tune `k`, the number of
    /// nearest neighbours used to build the affinity graph.
    ///
    /// ### Params
    ///
    /// * `k` - Number of nearest neighbours. Defaults to `5`.
    ///
    /// ### Returns
    ///
    /// Hopefully sensible standard parameters for 2D visualisation.
    pub fn new_default_2d(k: Option<usize>) -> Self {
        let k = k.unwrap_or(5);

        Self {
            k,
            ..Self::default()
        }
    }
}

/////////////////////
// PHATE diffusion //
/////////////////////

/// Build the PHATE diffusion operator from high-dimensional data
///
/// Runs kNN search (or uses precomputed results), computes alpha decay
/// affinities, and constructs either a full or landmark diffusion operator
/// depending on `phate_params.n_landmarks`.
///
/// ### Params
///
/// * `data` - Input data matrix (samples × features)
/// * `k` - Number of nearest neighbours for graph construction
/// * `precomputed_knn` - Precomputed kNN indices and distances as
///   `Some((Vec<Vec<usize>>, Vec<Vec<T>>))`, or `None` to run search
///   internally. Indices and distances must exclude self.
/// * `ann_type` - (Approximate) Nearest neighbour method: `"exhaustive"`,
///   `"kmknn"`, `"balltree"`, `"annoy"`, `"hnsw"`, or `"nndescent"`. If you
///   provide a weird string, the function will default to `"kmknn"`
/// * `nn_params` - Nearest neighbour search parameters
/// * `phate_params` - Full PHATE parameter struct (uses `k`, `decay`,
///   `bandwidth_scale`, `thresh`, `symmetrise`, `n_landmarks`,
///   `landmark_mode`)
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `PhateDiffusion::Full { operator }` when `n_landmarks` is `None` or
/// \>= N, otherwise `PhateDiffusion::Landmark { landmarks }` containing
/// the compressed landmark operator and interpolation matrices.
#[allow(clippy::too_many_arguments)]
pub fn construct_phate_diffusion<T>(
    data: MatRef<T>,
    k: usize,
    precomputed_knn: PreComputedKnn<T>,
    ann_type: &str,
    nn_params: &NearestNeighbourParams<T>,
    phate_params: &PhateParams<T>,
    seed: usize,
    verbose: usize,
) -> Result<PhateDiffusion<T>, ManifoldsError>
where
    T: ManifoldsFloat,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let verbosity = parse_verbosity_level(verbose);

    let (knn_indices, knn_dist) = match precomputed_knn {
        Some((indices, distances)) => {
            if verbosity.normal_verbosity() {
                println!("Using precomputed kNN graph...");
            }
            (indices, distances)
        }
        None => {
            if verbosity.normal_verbosity() {
                println!(
                    "Running (approximate) nearest neighbour search using {}...",
                    ann_type
                );
            }
            let start_knn = Instant::now();
            let result = run_ann_search(data, k, ann_type.to_string(), nn_params, seed, verbose)?;
            if verbosity.normal_verbosity() {
                println!("kNN search done in: {:.2?}.", start_knn.elapsed());
            }
            result
        }
    };

    if verbosity.normal_verbosity() {
        println!("Calculating alpha decay affinities");
    }
    let start_alpha_affinities = Instant::now();

    let graph = phate_alpha_decay_affinities(
        &knn_indices,
        &knn_dist,
        phate_params.k,
        phate_params.diffusion_params.decay,
        phate_params.diffusion_params.bandwidth_scale,
        phate_params.diffusion_params.thresh,
        &phate_params.diffusion_params.graph_symmetry,
        nn_params.dist_metric == "euclidean",
    );

    if verbosity.normal_verbosity() {
        println!(
            "Alpha decay affinity calculations done in: {:.2?}.",
            start_alpha_affinities.elapsed()
        );
    }

    let affinity = coo_to_csr(&graph);
    let diffusion_op = build_diffusion_operator(&affinity);

    match phate_params.diffusion_params.n_landmarks {
        None => Ok(PhateDiffusion::Full {
            operator: diffusion_op,
        }),
        Some(n_landmarks) if n_landmarks >= data.nrows() => Ok(PhateDiffusion::Full {
            operator: diffusion_op,
        }),
        Some(n_landmarks) => {
            if verbosity.normal_verbosity() {
                println!(" Building {} landmarks...", n_landmarks);
            }
            let start_landmarks = Instant::now();
            let landmarks = PhateLandmarks::build(
                data,
                &affinity,
                &diffusion_op,
                n_landmarks,
                &phate_params.diffusion_params.landmark_method,
                &nn_params.dist_metric,
                seed,
                Some(100),
                verbose,
            )?;
            if verbosity.normal_verbosity() {
                println!(
                    " Landmarks generated in: {:.2?}.",
                    start_landmarks.elapsed()
                );
            }
            Ok(PhateDiffusion::Landmark { landmarks })
        }
    }
}

/// Run PHATE dimensionality reduction
///
/// Potential of Heat-diffusion for Affinity-based Transition Embedding
/// (PHATE) learns a low-dimensional embedding that preserves the
/// diffusion geometry of the data.
///
/// ### Algorithm
///
/// 1. Build affinity graph via alpha decay kernel on kNN distances
/// 2. Construct row-stochastic diffusion operator P = D^{-1} K
/// 3. Determine diffusion time t (knee of Von Neumann entropy, or fixed)
/// 4. Compute diffusion potential: log transformation of P^t by default
/// 5. Embed via MDS on pairwise potential distances
///
/// Steps 2–4 optionally operate on a compressed landmark representation
/// (N × L instead of N × N) when `phate_params.n_landmarks` is set.
/// Pairwise distances and MDS (step 5) always run in the full N × N space;
/// however, to avoid holding the full N x N matrix in memory, you can
/// use a streaming version of MDS that computes the distances on the fly.
///
/// ### Params
///
/// * `data` - Input data matrix (samples × features)
/// * `precomputed_knn` - Precomputed kNN indices and distances as
///   `Some((Vec<Vec<usize>>, Vec<Vec<T>>))`, or `None` to run search
///   internally. Indices and distances must exclude self.
/// * `phate_params` - PHATE parameters. See `PhateParams` for full
///   documentation of each field.
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>` where the outer vector has
/// length `n_dim` and each inner vector has length `n_samples`. Each
/// outer element represents one embedding dimension.
///
/// ### References
///
/// - Moon et al. (2019): "Visualizing Structure and Transitions in
///   High-Dimensional Biological Data" (Nature Biotechnology)
pub fn phate<T>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    phate_params: PhateParams<T>,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
    StandardNormal: Distribution<T>,
{
    let start_phate = Instant::now();
    let verbosity = parse_verbosity_level(verbose);

    let phate_diffusion = construct_phate_diffusion(
        data,
        phate_params.k,
        precomputed_knn,
        &phate_params.ann_type,
        &phate_params.ann_params,
        &phate_params,
        seed,
        verbose,
    )?;

    let start_t = Instant::now();
    let t = match phate_params.diffusion_params.t {
        PhateTime::Auto { t_max } => {
            if verbosity.normal_verbosity() {
                println!("Finding optimal t (t_max={})...", t_max);
            }
            match &phate_diffusion {
                PhateDiffusion::Landmark { landmarks } => landmarks.find_optimal_t(t_max),
                PhateDiffusion::Full { operator } => {
                    let entropy = landmark_von_neumann_entropy(operator, t_max)?;
                    Ok(find_knee_point(&entropy))
                }
            }
        }
        PhateTime::Fixed(t) => Ok(t),
    }?;
    if verbosity.normal_verbosity() {
        println!("Identified t = {} in {:.2?}.", t, start_t.elapsed());
    }

    let mds_method = parse_mds_method(&phate_params.mds_method).unwrap_or_default();
    let dist = parse_ann_dist(&phate_params.ann_params.dist_metric).unwrap_or_default();
    let mds_params = MdsOptimParams::new(
        data.nrows(),
        phate_params.randomised,
        phate_params.mds_iter,
        None,
    );

    let start_embed = Instant::now();

    let embedding = match phate_diffusion {
        PhateDiffusion::Full { operator } => {
            if verbosity.normal_verbosity() {
                println!("Powering diffusion operator...");
            }
            let powered = matrix_power(&operator, t)?;
            let potential = calculate_potential(&powered, 1, phate_params.diffusion_params.gamma)?;

            if verbosity.normal_verbosity() {
                println!(
                    "Potential shape: {} × {} - calculated in {:.2?}.",
                    potential.shape().0,
                    potential.shape().1,
                    start_embed.elapsed()
                );
            }

            let res = match mds_method {
                MdsMethod::ClassicMds => {
                    if verbosity.normal_verbosity() {
                        println!("Computing pairwise distances, running classic MDS...");
                    }
                    let distances = compute_potential_distances(&potential, &dist);
                    classic_mds(&distances, phate_params.n_dim, mds_params.randomised, seed)
                }
                MdsMethod::SgdDense => {
                    if verbosity.normal_verbosity() {
                        println!("Computing pairwise distances, running SGD-MDS...");
                    }
                    let distances = compute_potential_distances(&potential, &dist);
                    sgd_mds(
                        &distances,
                        phate_params.n_dim,
                        &mds_params,
                        None,
                        seed,
                        verbose,
                    )
                }
            }?;

            res
        }
        PhateDiffusion::Landmark { landmarks } => {
            if verbosity.normal_verbosity() {
                println!(
                    "Powering landmark operator ({} landmarks)...",
                    landmarks.get_n_landmarks()
                );
            }
            let landmark_powered = landmarks.power(t)?;
            let landmark_potential =
                calculate_potential(&landmark_powered, 1, phate_params.diffusion_params.gamma)?;

            if verbosity.normal_verbosity() {
                println!(
                    "Landmark potential shape: {} × {} - calculated in {:.2?}.",
                    landmark_potential.shape().0,
                    landmark_potential.shape().1,
                    start_embed.elapsed()
                );
                println!("Computing landmark pairwise distances...");
            }

            let landmark_distances = compute_potential_distances(&landmark_potential, &dist);

            let landmark_mds_params = MdsOptimParams::new(
                landmarks.get_n_landmarks(),
                phate_params.randomised,
                None,
                None,
            );

            if verbosity.normal_verbosity() {
                println!("Running MDS on landmarks...");
            }

            let landmark_embedding = match mds_method {
                MdsMethod::ClassicMds => classic_mds(
                    &landmark_distances,
                    phate_params.n_dim,
                    landmark_mds_params.randomised,
                    seed,
                ),
                _ => sgd_mds(
                    &landmark_distances,
                    phate_params.n_dim,
                    &landmark_mds_params,
                    None,
                    seed,
                    verbose,
                ),
            }?;

            if verbosity.normal_verbosity() {
                println!("Interpolating to full N points via Nyström...");
            }
            landmarks.interpolate_embedding(&landmark_embedding)
        }
    };

    if verbosity.normal_verbosity() {
        println!("Ran MDS in {:.2?}.", start_embed.elapsed());
        println!("Finished running PHATE in {:.2?}.", start_phate.elapsed());
    }

    let n_samples = embedding.len();
    let mut transposed = vec![vec![T::zero(); n_samples]; phate_params.n_dim];
    for i in 0..n_samples {
        for d in 0..phate_params.n_dim {
            transposed[d][i] = embedding[i][d];
        }
    }

    Ok(transposed)
}

////////////
// PaCMAP //
////////////

////////////
// Params //
////////////

/// Parameters for PaCMAP dimensionality reduction.
#[derive(Debug, Clone)]
pub struct PacmapParams<T> {
    /// Output dimensionality. Default 2.
    pub n_dim: usize,
    /// Number of near neighbours. Default 10 (paper default; lower than UMAP's
    /// 15 since PaCMAP is less sensitive to k).
    pub k: usize,
    /// (Approximate) Nearest neighbour method. One of `"exhaustive"`, `"ivf"`,
    /// `"hnsw"`, `"nndescent"`, `"annoy"`, `"kmknn"` or `"balltree"`.
    pub ann_type: String,
    /// Which optimiser to use. Options are `"adam"` and `"adam_parallel"`
    pub optimiser_type: String,
    /// Mid-near pairs per point. Default 2.
    pub n_mid_near: usize,
    /// Further (random) pairs per point. Default 2.
    pub n_further: usize,
    /// Start index into kNN list for mid-near candidate window. Default 4
    /// (skip the 4 nearest).
    pub mn_candidate_start: usize,
    /// End index into kNN list for mid-near candidate window. Default 50.
    /// Requires k >= this value.
    pub mn_candidate_end: usize,
    /// Embedding initialisation. Default `"pca"`. PCA is strongly recommended
    /// for PaCMAP as random init degrades global structure.
    pub initialisation: String,
    /// Nearest neighbour search parameters.
    pub nn_params: NearestNeighbourParams<T>,
    /// Optimiser parameters.
    pub optim_params: PacmapOptimParams<T>,
}

impl<T> Default for PacmapParams<T>
where
    T: ManifoldsFloat,
{
    fn default() -> Self {
        Self {
            n_dim: 2,
            k: 50,
            ann_type: "kmknn".to_string(),
            optimiser_type: "adam_parallel".to_string(),
            n_mid_near: 2,
            n_further: 2,
            mn_candidate_start: 4,
            mn_candidate_end: 50,
            initialisation: "pca".to_string(),
            nn_params: NearestNeighbourParams::default(),
            optim_params: PacmapOptimParams::default(),
        }
    }
}

impl<T> PacmapParams<T>
where
    T: ManifoldsFloat,
{
    /// Generate a new instance of the PaCMAP parameters with full control over
    /// every field.
    ///
    /// For sensible defaults, use [`PacmapParams::default`] or
    /// [`PacmapParams::new_default_2d`] instead.
    ///
    /// Note: `k` is clamped to at least `mn_candidate_end`, since the mid-near
    /// candidate window indexes into the kNN list and must not run past its
    /// end.
    ///
    /// ### Params
    ///
    /// * `n_dim` - Number of dimensions for the embedding.
    /// * `k` - Number of near neighbours. Clamped to at least
    ///   `mn_candidate_end`.
    /// * `ann_type` - (Approximate) nearest neighbour method. One of
    ///   `"exhaustive"`, `"ivf"`, `"hnsw"`, `"nndescent"`, `"annoy"`,
    ///   `"kmknn"` or `"balltree"`.
    /// * `optimiser_type` - Which optimiser to use. Options are `"adam"` and
    ///   `"adam_parallel"`.
    /// * `n_mid_near` - Mid-near pairs per point.
    /// * `n_further` - Further (random) pairs per point.
    /// * `mn_candidate_start` - Start index into the kNN list for the mid-near
    ///   candidate window.
    /// * `mn_candidate_end` - End index into the kNN list for the mid-near
    ///   candidate window. Requires `k >= this value`.
    /// * `initialisation` - Embedding initialisation. PCA is strongly
    ///   recommended for PaCMAP as random init degrades global structure.
    /// * `nn_params` - Nearest neighbour search parameters.
    /// * `optim_params` - Optimiser parameters.
    ///
    /// ### Returns
    ///
    /// A fully specified set of PaCMAP parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_dim: usize,
        k: usize,
        ann_type: String,
        optimiser_type: String,
        n_mid_near: usize,
        n_further: usize,
        mn_candidate_start: usize,
        mn_candidate_end: usize,
        initialisation: String,
        nn_params: NearestNeighbourParams<T>,
        optim_params: PacmapOptimParams<T>,
    ) -> Self {
        let k = k.max(mn_candidate_end);

        Self {
            n_dim,
            k,
            ann_type,
            optimiser_type,
            n_mid_near,
            n_further,
            mn_candidate_start,
            mn_candidate_end,
            initialisation,
            nn_params,
            optim_params,
        }
    }

    /// Default parameters for standard 2D visualisation.
    ///
    /// Returns the default parameters but lets you tune `k`, the number of
    /// near neighbours. Note that `k` is clamped to at least
    /// `mn_candidate_end` (default `50`).
    ///
    /// ### Params
    ///
    /// * `k` - Number of near neighbours. Defaults to `50`.
    ///
    /// ### Returns
    ///
    /// Hopefully sensible standard parameters for 2D visualisation.
    pub fn new_default_2d(k: Option<usize>) -> Self {
        let default = Self::default();
        let k = k.unwrap_or(default.k).max(default.mn_candidate_end);

        Self { k, ..default }
    }
}

//////////
// Main //
//////////

/// Run PaCMAP dimensionality reduction.
///
/// Pairwise Controlled Manifold Approximation (PaCMAP) is a dimensionality
/// reduction method that explicitly balances local and global structure
/// preservation via three pair types and a phased optimisation schedule.
///
/// 1. Find k-nearest neighbours
/// 2. Construct near, mid-near, and further pairs
/// 3. Initialise embedding (PCA strongly recommended)
/// 4. Optimise via three-phase Adam with pair-type weight schedule
///
/// ### Params
///
/// * `data` - Input data matrix (samples × features).
/// * `precomputed_knn` - Optional precomputed kNN. Must have been computed
///   with k >= `params.mn_candidate_end` to support mid-near sampling.
/// * `params` - PaCMAP parameters.
/// * `seed` - Seed for reproducibility.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding as `Vec<Vec<T>>` with shape `[n_dim][n_samples]`.
pub fn pacmap<T>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    params_pacmap: &PacmapParams<T>,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
    HnswIndex<T>: HnswState<T>,
    StandardNormal: Distribution<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let n_samples = data.nrows();

    let verbosity = parse_verbosity_level(verbose);

    let (knn_indices, _) = match precomputed_knn {
        Some((indices, distances)) => {
            if verbosity.normal_verbosity() {
                println!("Using precomputed kNN graph...");
            }
            (indices, distances)
        }
        None => {
            if verbosity.normal_verbosity() {
                println!(
                    "Running (approximate) nearest neighbour search using {} (k={})...",
                    params_pacmap.ann_type, params_pacmap.k
                );
            }
            let start_knn = Instant::now();
            let result = run_ann_search(
                data,
                params_pacmap.k,
                params_pacmap.ann_type.clone(),
                &params_pacmap.nn_params,
                seed,
                verbose,
            )?;
            if verbosity.normal_verbosity() {
                println!("kNN search done in: {:.2?}.", start_knn.elapsed());
            }
            result
        }
    };

    if verbosity.normal_verbosity() {
        println!("Constructing PaCMAP pairs...");
    }

    let start_pairs = Instant::now();

    let pairs: PacmapPairs = construct_pacmap_pairs(
        &knn_indices,
        params_pacmap.n_mid_near,
        params_pacmap.n_further,
        params_pacmap.mn_candidate_start,
        params_pacmap.mn_candidate_end,
        seed as u64,
    );

    let end_pairs = start_pairs.elapsed();

    if verbosity.normal_verbosity() {
        println!(
            "Pairs generated in {:.2?}: {} near, {} mid-near, {} further.",
            end_pairs,
            pairs.near.len().separate_with_underscores(),
            pairs.mid_near.len().separate_with_underscores(),
            pairs.further.len().separate_with_underscores()
        );
    }

    let init_type =
        parse_initilisation(&params_pacmap.initialisation, true, None).unwrap_or_else(|| {
            println!(
                "Unknown initialisation provided: {:?}. Defaulting to PCA.",
                params_pacmap.initialisation,
            );
            EmbdInit::PcaInit {
                range: None,
                randomised: true,
            }
        });

    if verbosity.normal_verbosity() {
        println!(
            "Initialising embedding via {} layout...",
            params_pacmap.initialisation
        );
    }

    let dummy_graph = knn_to_coo_unweighted(&knn_indices);
    let mut embd = initialise_embedding(
        &init_type,
        params_pacmap.n_dim,
        seed as u64,
        &dummy_graph,
        data,
    )?;

    let optimiser = parse_pacmap_optimiser(&params_pacmap.optimiser_type).unwrap_or_default();

    let start_optim = Instant::now();

    match optimiser {
        PacMapOptimiser::AdamParallel => {
            let _ =
                optimise_pacmap_parallel(&mut embd, &pairs, &params_pacmap.optim_params, verbose);
        }
        PacMapOptimiser::Adam => {
            let _ = optimise_pacmap(&mut embd, &pairs, &params_pacmap.optim_params, verbose);
        }
    }

    let end_optim = start_optim.elapsed();

    if verbosity.normal_verbosity() {
        println!("Optimisation done in {:.2?}.", end_optim);
        println!("PaCMAP complete!");
    }

    // transpose from [n_samples][n_dim] to [n_dim][n_samples]
    let mut transposed = vec![vec![T::zero(); n_samples]; params_pacmap.n_dim];
    for sample_idx in 0..n_samples {
        for dim_idx in 0..params_pacmap.n_dim {
            transposed[dim_idx][sample_idx] = embd[sample_idx][dim_idx];
        }
    }

    Ok(transposed)
}

////////////////////
// Diffusion maps //
////////////////////

/// Parameters for classical diffusion maps (Coifman & Lafon, 2006).
#[derive(Debug, Clone)]
pub struct DiffusionMapsParams<T> {
    /// Output embedding dimensionality (default: 2)
    pub n_dim: usize,
    /// Number of nearest neighbours used to build the graph
    pub k: usize,
    /// ANN algorithm name (see `NearestNeighbourParams`)
    pub ann_type: String,
    /// Nearest neighbour search parameters
    pub ann_params: NearestNeighbourParams<T>,
    /// Multiplicative factor applied to the adaptive kernel bandwidth
    pub bandwidth_scale: T,
    /// Sparsity threshold applied to kernel entries
    pub thresh: T,
    /// Graph symmetrisation: `"add"`, `"multiply"`, `"mnn"` or `"none"`
    pub graph_symmetry: String,
    /// Anisotropic density-correction exponent in [0, 1]. 0 gives the
    /// normalised graph Laplacian, 0.5 the Fokker-Planck operator, 1 the
    /// Laplace-Beltrami operator.
    pub alpha_norm: T,
    /// Diffusion time: `Auto` picks the VNE knee, `Fixed(t)` uses `t` directly.
    pub t: PhateTime,
    /// Landmark count. `None` or `>= n` runs full DM.
    pub n_landmarks: Option<usize>,
    /// `"random"`, `"spectral"` or `"density"`
    pub landmark_method: String,
    /// Components for spectral landmark selection (ignored otherwise)
    pub n_svd: Option<usize>,
}

impl<T> Default for DiffusionMapsParams<T>
where
    T: ManifoldsFloat,
{
    fn default() -> Self {
        Self {
            n_dim: 2,
            k: 5,
            ann_type: "kmknn".to_string(),
            ann_params: NearestNeighbourParams::default(),
            bandwidth_scale: T::from_f64(1.0).unwrap(),
            thresh: T::from_f64(1e-4).unwrap(),
            graph_symmetry: "add".to_string(),
            alpha_norm: T::from_f64(1.0).unwrap(),
            t: parse_phate_time(None, PHATE_MAX_T),
            n_landmarks: None,
            landmark_method: "spectral".to_string(),
            n_svd: None,
        }
    }
}

impl<T> DiffusionMapsParams<T>
where
    T: ManifoldsFloat,
{
    /// Construct a new `DiffusionMapsParams` with full control over every
    /// field.
    ///
    /// For sensible defaults, use [`DiffusionMapsParams::default`] or
    /// [`DiffusionMapsParams::new_default_2d`] instead.
    ///
    /// ### Params
    ///
    /// * `n_dim` - Output embedding dimensionality.
    /// * `k` - Number of nearest neighbours for graph construction.
    /// * `ann_type` - ANN algorithm: `"exhaustive"`, `"kmknn"`, `"balltree"`,
    ///   `"annoy"`, `"hnsw"`, or `"nndescent"`.
    /// * `ann_params` - Nearest neighbour search parameters.
    /// * `bandwidth_scale` - Multiplicative factor on the adaptive kernel
    ///   bandwidth.
    /// * `thresh` - Kernel entries below this value are set to zero.
    /// * `graph_symmetry` - Symmetrisation method: `"add"`, `"multiply"`,
    ///   `"mnn"` or `"none"`.
    /// * `alpha_norm` - Anisotropic normalisation exponent in [0, 1]. 0 gives
    ///   the normalised graph Laplacian, 0.5 the Fokker-Planck operator, 1 the
    ///   Laplace-Beltrami operator.
    /// * `t` - Diffusion time: `Auto` picks the VNE knee, `Fixed(t)` uses `t`
    ///   directly.
    /// * `n_landmarks` - Number of landmarks. `None` or `>= n` runs full DM.
    /// * `landmark_method` - `"random"`, `"spectral"`, or `"density"`.
    /// * `n_svd` - SVD components for spectral landmark selection. Ignored
    ///   otherwise.
    ///
    /// ### Returns
    ///
    /// A fully specified set of diffusion maps parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_dim: usize,
        k: usize,
        ann_type: String,
        ann_params: NearestNeighbourParams<T>,
        bandwidth_scale: T,
        thresh: T,
        graph_symmetry: String,
        alpha_norm: T,
        t: PhateTime,
        n_landmarks: Option<usize>,
        landmark_method: String,
        n_svd: Option<usize>,
    ) -> Self {
        Self {
            n_dim,
            k,
            ann_type,
            ann_params,
            bandwidth_scale,
            thresh,
            graph_symmetry,
            alpha_norm,
            t,
            n_landmarks,
            landmark_method,
            n_svd,
        }
    }

    /// Default parameters for standard 2D visualisation.
    ///
    /// Returns the default parameters but lets you tune `k`, the number of
    /// nearest neighbours used to build the graph.
    ///
    /// ### Params
    ///
    /// * `k` - Number of nearest neighbours. Defaults to `5`.
    ///
    /// ### Returns
    ///
    /// Hopefully sensible standard parameters for 2D visualisation.
    pub fn new_default_2d(k: Option<usize>) -> Self {
        let k = k.unwrap_or(5);

        Self {
            k,
            ..Self::default()
        }
    }
}

/// Full or landmark diffusion maps operator.
///
/// `Full` is used when no landmarks are requested or when the landmark count
/// meets or exceeds the data size. `Landmark` delegates eigendecomposition and
/// Nystroem extension to `DiffusionMapsLandmarks`.
pub enum DiffusionMapsOperator<T>
where
    T: ManifoldsFloat,
{
    /// Full N×N symmetric diffusion operator.
    Full {
        /// Symmetric diffusion operator P_sym = D^{-1/2} K D^{-1/2}
        p_sym: CompressedSparseData<T>,
        /// Square-root of node degrees; used to recover right eigenvectors
        /// of the row-stochastic operator from the symmetric ones
        sqrt_degrees: Vec<T>,
    },
    /// Landmark-based operator for large data sets.
    Landmark {
        /// Landmark operator ready for eigendecomposition and Nystroem extension
        landmarks: DiffusionMapsLandmarks<T>,
    },
}

/// Build the diffusion maps operator from raw data.
///
/// Runs kNN search, builds a Gaussian kernel via alpha-decay, applies
/// anisotropic normalisation, then returns either a full or landmark operator
/// depending on `dm_params.n_landmarks`. Falls back to full if
/// `n_landmarks >= n`.
///
/// ### Params
///
/// * `data` - Input data matrix (N × features)
/// * `k` - Number of nearest neighbours
/// * `precomputed_knn` - Optional precomputed kNN; skips search if provided
/// * `ann_type` - ANN algorithm name
/// * `nn_params` - Nearest neighbour search parameters
/// * `dm_params` - Diffusion maps parameters
/// * `seed` - RNG seed
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `DiffusionMapsOperator` (full or landmark)
#[allow(clippy::too_many_arguments)]
pub fn construct_diffusion_maps_operator<T>(
    data: MatRef<T>,
    k: usize,
    precomputed_knn: PreComputedKnn<T>,
    ann_type: &str,
    nn_params: &NearestNeighbourParams<T>,
    dm_params: &DiffusionMapsParams<T>,
    seed: usize,
    verbose: usize,
) -> Result<DiffusionMapsOperator<T>, ManifoldsError>
where
    T: ManifoldsFloat,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let use_full = match dm_params.n_landmarks {
        None => true,
        Some(n) => n >= data.nrows(),
    };

    let verbosity = parse_verbosity_level(verbose);

    let needs_affinity =
        use_full || matches!(dm_params.landmark_method.as_str(), "spectral" | "density");

    let affinity = if needs_affinity {
        let (knn_indices, knn_dist) = match precomputed_knn {
            Some((indices, distances)) => {
                if verbosity.normal_verbosity() {
                    println!("Using precomputed kNN graph...");
                }
                (indices, distances)
            }
            None => {
                if verbosity.normal_verbosity() {
                    println!(
                        "Running (approximate) nearest neighbour search using {}...",
                        ann_type
                    );
                }
                let start_knn = Instant::now();
                let res = run_ann_search(data, k, ann_type.to_string(), nn_params, seed, verbose)?;
                if verbosity.normal_verbosity() {
                    println!("kNN search done in {:.2?}.", start_knn.elapsed());
                }
                res
            }
        };

        if verbosity.normal_verbosity() {
            println!("Building Gaussian kernel affinities");
        }
        let graph = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dist,
            dm_params.k,
            Some(T::from_f64(2.0).unwrap()),
            dm_params.bandwidth_scale,
            dm_params.thresh,
            &dm_params.graph_symmetry,
            nn_params.dist_metric == "euclidean",
        );
        Some(coo_to_csr(&graph))
    } else {
        if verbosity.normal_verbosity() {
            println!("Skipping full affinity (random landmarks).");
        }
        None
    };

    if use_full {
        let kernel = affinity.unwrap();
        let kernel_norm = apply_anisotropic_normalisation(&kernel, dm_params.alpha_norm)?;
        let (p_sym, sqrt_degrees) = build_symmetric_diffusion_operator(&kernel_norm)?;
        return Ok(DiffusionMapsOperator::Full {
            p_sym,
            sqrt_degrees,
        });
    }

    let n_landmarks = dm_params.n_landmarks.unwrap();
    if verbosity.normal_verbosity() {
        println!(" Building {} landmarks...", n_landmarks);
    }
    let start_l = Instant::now();
    let landmarks = DiffusionMapsLandmarks::build(
        data,
        affinity.as_ref(),
        n_landmarks,
        &dm_params.landmark_method,
        &nn_params.dist_metric,
        dm_params.alpha_norm,
        dm_params.k,
        dm_params.bandwidth_scale,
        dm_params.thresh,
        &dm_params.graph_symmetry,
        seed,
        dm_params.n_svd,
        verbose,
    )?;
    if verbosity.normal_verbosity() {
        println!(" Landmarks built in {:.2?}.", start_l.elapsed());
    }
    Ok(DiffusionMapsOperator::Landmark { landmarks })
}

/// Run diffusion maps end-to-end.
///
/// 1. kNN graph on the raw data
/// 2. Gaussian kernel via alpha-decay with decay = 2 (adaptive bandwidth)
/// 3. Anisotropic (alpha) normalisation for density correction
/// 4. Symmetric diffusion operator P_sym = D^{-1/2} K D^{-1/2}
/// 5. Top (n_dim + 1) eigenpairs of P_sym via Lanczos
/// 6. Drop the trivial eigenvalue (= 1) and scale each non-trivial eigenvector
///    phi_k by lambda_k^t
///
/// With landmarks, steps 4–6 run on the L×L operator and are Nystroem-extended
/// back to N points.
///
/// ### Params
///
/// * `data` - Input data matrix (N × features)
/// * `precomputed_knn` - Optional precomputed kNN; skips search if provided.
///   Must have been computed with k >= `dm_params.k`.
/// * `dm_params` - Diffusion maps parameters
/// * `seed` - RNG seed
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding as `Vec<Vec<T>>` with shape `[n_dim][n_samples]`
pub fn diffusion_maps<T>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    dm_params: DiffusionMapsParams<T>,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let start_dm = Instant::now();

    let verbosity = parse_verbosity_level(verbose);

    let op = construct_diffusion_maps_operator(
        data,
        dm_params.k,
        precomputed_knn,
        &dm_params.ann_type,
        &dm_params.ann_params,
        &dm_params,
        seed,
        verbose,
    )?;

    let embedding: Vec<Vec<T>> = match op {
        DiffusionMapsOperator::Full {
            p_sym,
            sqrt_degrees,
        } => {
            let t = match dm_params.t {
                PhateTime::Auto { t_max } => {
                    if verbosity.normal_verbosity() {
                        println!("Finding optimal t (t_max={})...", t_max);
                    }
                    let entropy = landmark_von_neumann_entropy(&p_sym, t_max)?;
                    find_knee_point(&entropy)
                }
                PhateTime::Fixed(t) => t,
            };
            if verbosity.normal_verbosity() {
                println!("Using t = {}.", t);
                println!(
                    "Computing top {} eigenpairs of P_sym...",
                    dm_params.n_dim + 1
                );
            }
            let start_eig = Instant::now();
            let n_ask = (dm_params.n_dim + 5).min(sqrt_degrees.len() - 1);
            let (evals, evecs) = compute_largest_eigenpairs_lanczos(&p_sym, n_ask, seed as u64)?;

            if verbosity.normal_verbosity() {
                println!("Eigendecomposition done in {:.2?}.", start_eig.elapsed());
            }

            let n = sqrt_degrees.len();
            let mut embedding = vec![vec![T::zero(); dm_params.n_dim]; n];
            for comp_idx in 1..=dm_params.n_dim {
                let lambda = T::from_f32(evals[comp_idx]).unwrap();
                let lambda_t = lambda.powi(t as i32);

                let mut max_abs = 0.0f32;
                let mut sign = 1.0f32;
                for i in 0..n {
                    let v = evecs[i][comp_idx];
                    if v.abs() > max_abs {
                        max_abs = v.abs();
                        sign = if v >= 0.0 { 1.0 } else { -1.0 };
                    }
                }
                for i in 0..n {
                    let u = T::from_f32(evecs[i][comp_idx] * sign).unwrap();
                    embedding[i][comp_idx - 1] = lambda_t * u / sqrt_degrees[i];
                }
            }
            embedding
        }
        DiffusionMapsOperator::Landmark { landmarks } => {
            let t = match dm_params.t {
                PhateTime::Auto { t_max } => {
                    if verbosity.normal_verbosity() {
                        println!("Finding optimal t on landmarks (t_max={})...", t_max);
                    }
                    landmarks.find_optimal_t(t_max)?
                }
                PhateTime::Fixed(t) => t,
            };
            if verbosity.detailed_verbosity() {
                println!(
                    "Using t = {}. Eigendecomposing {}x{} landmark operator...",
                    t,
                    landmarks.get_n_landmarks(),
                    landmarks.get_n_landmarks()
                );
            }
            let start_eig = Instant::now();
            let (evals, evecs) = landmarks.eigendecompose(dm_params.n_dim, seed as u64)?;

            if verbosity.detailed_verbosity() {
                println!("Eigendecomposition done in {:.2?}.", start_eig.elapsed());
            }
            if verbosity.normal_verbosity() {
                println!("Nystroem-extending to full data...");
            }
            let (landmark_embedding, lambdas) =
                landmarks.compute_landmark_embedding(&evals, &evecs, dm_params.n_dim, t);
            landmarks.nystrom_extend(&landmark_embedding, &lambdas)
        }
    };

    if verbosity.normal_verbosity() {
        println!("Diffusion maps finished in {:.2?}.", start_dm.elapsed());
    }

    let n = embedding.len();
    let mut transposed = vec![vec![T::zero(); n]; dm_params.n_dim];
    for i in 0..n {
        for d in 0..dm_params.n_dim {
            transposed[d][i] = embedding[i][d];
        }
    }

    Ok(transposed)
}

/////////////////////
// Parametric UMAP //
/////////////////////

#[cfg(feature = "parametric")]
/// Stores the parameters for parametric UMAP via neural nets
#[derive(Debug, Clone)]
pub struct ParametricUmapParams<T> {
    /// How many dimensions to return
    pub n_dim: usize,
    /// Number of neighbours
    pub k: usize,
    /// Which of the possible approximate nearest neighbour searches to use.
    /// Defaults to `"hnsw"`.
    pub ann_type: String,
    /// Vector of usizes for the hidden layers in the MLP.
    pub hidden_layers: Vec<usize>,
    /// Nearest neighbour parameters.
    pub nn_params: NearestNeighbourParams<T>,
    /// The graph parameters for the generation of the graph structure.
    pub umap_graph_params: UmapGraphParams<T>,
    /// Train parameters for the neural network.
    pub train_param: TrainParametricParams<T>,
}

#[cfg(feature = "parametric")]
impl<T> Default for ParametricUmapParams<T>
where
    T: ManifoldsFloat + Element,
{
    fn default() -> Self {
        Self {
            n_dim: 2,
            k: 15,
            ann_type: "hnsw".to_string(),
            hidden_layers: vec![128, 128, 128],
            nn_params: NearestNeighbourParams::default(),
            umap_graph_params: UmapGraphParams::default(),
            train_param: TrainParametricParams::default(),
        }
    }
}

#[cfg(feature = "parametric")]
impl<T> ParametricUmapParams<T>
where
    T: ManifoldsFloat + Element,
{
    /// Generate new parametric UMAP parameters with full control over every
    /// field.
    ///
    /// For sensible defaults, use [`ParametricUmapParams::default`] or
    /// [`ParametricUmapParams::new_default_2d`] instead.
    ///
    /// ### Params
    ///
    /// * `n_dim` - Number of embedding dimensions.
    /// * `k` - Number of nearest neighbours.
    /// * `ann_type` - (Approximate) nearest neighbour method. One of
    ///   `"exhaustive"`, `"ivf"`, `"hnsw"`, `"nndescent"`, `"annoy"`,
    ///   `"kmknn"` or `"balltree"`.
    /// * `hidden_layers` - Hidden layer sizes for the MLP encoder.
    /// * `nn_params` - Nearest neighbour parameters.
    /// * `umap_graph_params` - UMAP graph generation parameters.
    /// * `train_param` - Training parameters for the neural network.
    ///
    /// ### Returns
    ///
    /// A fully specified set of parametric UMAP parameters.
    pub fn new(
        n_dim: usize,
        k: usize,
        ann_type: String,
        hidden_layers: Vec<usize>,
        nn_params: NearestNeighbourParams<T>,
        umap_graph_params: UmapGraphParams<T>,
        train_param: TrainParametricParams<T>,
    ) -> Self {
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

    /// Default parameters for standard 2D parametric UMAP.
    ///
    /// Returns the default parameters but lets you tune `min_dist`, `spread`
    /// and `corr_weight`, which feed into the training parameters.
    ///
    /// ### Params
    ///
    /// * `min_dist` - Minimum distance between embedded points. Defaults to
    ///   `0.1`.
    /// * `spread` - Effective scale of embedded points. Defaults to `1.0`.
    /// * `corr_weight` - Correlation loss weight. Defaults to `0.0`.
    ///
    /// ### Returns
    ///
    /// Hopefully sensible standard parameters for 2D visualisation.
    pub fn new_default_2d(min_dist: Option<T>, spread: Option<T>, corr_weight: Option<T>) -> Self {
        let min_dist = min_dist.unwrap_or(T::from_f64(0.1).unwrap());
        let spread = spread.unwrap_or(T::from_f64(1.0).unwrap());
        let corr_weight = corr_weight.unwrap_or(T::from_f64(0.0).unwrap());

        Self {
            train_param: TrainParametricParams::from_min_dist_spread(
                min_dist,
                spread,
                corr_weight,
                None,
                None,
                None,
                None,
            ),
            ..Self::default()
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
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances. Needs
///   to be a tuple of `(Vec<Vec<usize>>, Vec<Vec<T>>)` with indices and
///   distances excluding self.
/// * `umap_params` - Configuration parameters for parametric UMAP
/// * `device` - Burn backend device for neural network training
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>` where outer vector has length
/// `n_dim` and inner vectors have length `n_samples`. Each outer element
/// represents one embedding dimension.
pub fn parametric_umap<T, B>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    umap_params: &ParametricUmapParams<T>,
    device: &B::Device,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat + Element,
    B: AutodiffBackend,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    // parse various parameters
    let nn_params = umap_params.nn_params.clone();

    let verbosity = parse_verbosity_level(verbose);

    if verbosity.normal_verbosity() {
        println!(
            "Running parametric umap with alpha: {:.2?} and beta: {:.2?}",
            ToPrimitive::to_f32(&umap_params.train_param.a).unwrap(),
            ToPrimitive::to_f32(&umap_params.train_param.b).unwrap()
        );
    }

    let (graph, _, _) = construct_umap_graph(
        data,
        precomputed_knn,
        umap_params.k,
        umap_params.ann_type.clone(),
        &umap_params.umap_graph_params,
        &nn_params,
        umap_params.train_param.n_epochs,
        seed,
        verbose,
    )?;

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
        verbosity.normal_verbosity(),
    );

    Ok(embd)
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
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
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
/// Returns a tuple of embedding and the `TrainedUmapModel` for further usage.
pub fn train_parametric_umap_model<T, B>(
    data: MatRef<T>,
    umap_params: &ParametricUmapParams<T>,
    device: &B::Device,
    seed: usize,
    verbose: usize,
) -> ParametricUmapResults<B, T>
where
    T: ManifoldsFloat + Element,
    B: AutodiffBackend,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    // parse various parameters
    let nn_params = umap_params.nn_params.clone();
    let verbosity = parse_verbosity_level(verbose);

    if verbosity.normal_verbosity() {
        println!(
            "Training parametric umap model with alpha: {:.2?} and beta: {:.2?}",
            ToPrimitive::to_f32(&umap_params.train_param.a).unwrap(),
            ToPrimitive::to_f32(&umap_params.train_param.b).unwrap()
        );
    }

    let (graph, _, _) = construct_umap_graph(
        data,
        None,
        umap_params.k,
        umap_params.ann_type.clone(),
        &umap_params.umap_graph_params,
        &nn_params,
        umap_params.train_param.n_epochs,
        seed,
        verbose,
    )?;

    let model_params = UmapMlpConfig::from_params(
        data.ncols(),
        umap_params.hidden_layers.clone(),
        umap_params.n_dim,
    );

    let (embd, trained_model) = train_parametric_umap::<B, T>(
        data,
        graph,
        &model_params,
        &umap_params.train_param,
        device,
        seed,
        verbosity.normal_verbosity(),
    );

    Ok((embd, trained_model))
}

/////////
// GPU //
/////////

//////////////
// UMAP GPU //
//////////////

/// UMAP parameters for GPU-accelerated nearest neighbour search
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct UmapParamsGpu<T> {
    /// How many dimensions to return
    pub n_dim: usize,
    /// Number of neighbours
    pub k: usize,
    /// Which optimiser to use. Defaults to `"adam_gpu"`.
    pub optimiser: String,
    /// Which GPU nearest neighbour search to use. One of `"exhaustive_gpu"`,
    /// `"ivf_gpu"` or `"nndescent_gpu"`. Defaults to `"ivf_gpu"`.
    pub ann_type: String,
    /// Which embedding initialisation to use. Defaults to spectral clustering.
    pub initialisation: String,
    /// Optional initialisation range
    pub init_range: Option<T>,
    /// GPU nearest neighbour parameters
    pub nn_params: NearestNeighbourParamsGpu<T>,
    /// Parameters for UMAP graph generation
    pub umap_graph_params: UmapGraphParams<T>,
    /// Parameters for the UMAP optimiser
    pub optim_params: UmapOptimParams<T>,
    /// Use randomised SVD for PCA-based initialisation
    pub randomised: bool,
}

#[cfg(feature = "gpu")]
impl<T> Default for UmapParamsGpu<T>
where
    T: ManifoldsFloat,
{
    fn default() -> Self {
        Self {
            n_dim: 2,
            k: 15,
            optimiser: "adam_gpu".to_string(),
            ann_type: "nndescent".to_string(),
            initialisation: "spectral".to_string(),
            init_range: None,
            nn_params: NearestNeighbourParamsGpu::default(),
            umap_graph_params: UmapGraphParams::default(),
            optim_params: UmapOptimParams::default(),
            randomised: false,
        }
    }
}

#[cfg(feature = "gpu")]
impl<T> UmapParamsGpu<T>
where
    T: ManifoldsFloat,
{
    /// Generate new GPU UMAP parameters with full control over every field.
    ///
    /// This constructor exposes every field directly. For sensible defaults,
    /// use [`UmapParamsGpu::default`] or [`UmapParamsGpu::new_default_2d`]
    /// instead.
    ///
    /// ### Params
    ///
    /// * `n_dim` - How many dimensions to return.
    /// * `k` - How many neighbours to consider.
    /// * `optimiser` - Which optimiser to use, e.g. `"adam_parallel"`.
    /// * `ann_type` - Which GPU ANN search to use. One of `"exhaustive_gpu"`,
    ///   `"ivf_gpu"` or `"nndescent_gpu"`.
    /// * `initialisation` - Which embedding initialisation to use, e.g.
    ///   `"spectral"`.
    /// * `init_range` - Optional initialisation range.
    /// * `nn_params` - GPU nearest neighbour parameters.
    /// * `umap_graph_params` - Parameters for the UMAP graph generation.
    /// * `optim_params` - Parameters for the UMAP optimiser.
    /// * `randomised` - Whether randomised SVD is used for PCA-based
    ///   initialisation.
    ///
    /// ### Returns
    ///
    /// A fully specified set of GPU UMAP parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_dim: usize,
        k: usize,
        optimiser: String,
        ann_type: String,
        initialisation: String,
        init_range: Option<T>,
        nn_params: NearestNeighbourParamsGpu<T>,
        umap_graph_params: UmapGraphParams<T>,
        optim_params: UmapOptimParams<T>,
        randomised: bool,
    ) -> Self {
        Self {
            n_dim,
            k,
            optimiser,
            ann_type,
            initialisation,
            init_range,
            nn_params,
            umap_graph_params,
            optim_params,
            randomised,
        }
    }

    /// Default parameters for standard 2D visualisation.
    ///
    /// Returns the default parameters but lets you tune `min_dist` and
    /// `spread`, which feed into the optimiser parameters and control how
    /// tightly points are packed in the embedding.
    ///
    /// ### Params
    ///
    /// * `min_dist` - Minimum distance between data points. Defaults to `0.5`.
    /// * `spread` - Spread parameter. Defaults to `1.0`.
    ///
    /// ### Returns
    ///
    /// Hopefully sensible standard parameters for 2D visualisation with GPU
    /// kNN search.
    pub fn new_default_2d(min_dist: Option<T>, spread: Option<T>) -> Self {
        let min_dist = min_dist.unwrap_or(T::from_f64(0.5).unwrap());
        let spread = spread.unwrap_or(T::from_f64(1.0).unwrap());

        Self {
            optim_params: UmapOptimParams::from_min_dist_spread(
                min_dist, spread, None, None, None, None, None, None, None,
            ),
            ..Self::default()
        }
    }
}

/// Construct the UMAP graph using GPU-accelerated nearest neighbour search
///
/// Identical to `construct_umap_graph` except the kNN search runs on a GPU
/// device via `run_ann_search_gpu`. All downstream graph construction
/// (smooth kNN distances, symmetrisation, edge filtering) remains on CPU.
///
/// ### Params
///
/// * `data` - Input data matrix (samples x features)
/// * `precomputed_knn` - Optional precomputed kNN (indices, distances)
///   excluding self.
/// * `k` - Number of nearest neighbours.
/// * `ann_type` - GPU ANN method: `"exhaustive_gpu"`, `"ivf_gpu"` or
///   `"nndescent_gpu"`.
/// * `umap_params` - UMAP graph parameters (bandwidth, local_connectivity,
///   mix_weight).
/// * `nn_params` - GPU nearest neighbour search parameters.
/// * `n_epochs` - Number of optimisation epochs (used for edge filtering).
/// * `device` - The GPU device to use.
/// * `seed` - Random seed.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Tuple of (graph, knn_indices, knn_dist).
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn construct_umap_graph_gpu<T, R>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    k: usize,
    ann_type: String,
    umap_params: &UmapGraphParams<T>,
    nn_params: &NearestNeighbourParamsGpu<T>,
    n_epochs: usize,
    device: R::Device,
    seed: usize,
    verbose: usize,
) -> UmapGraphResults<T>
where
    T: ManifoldsFloat + AnnSearchGpuFloat,
    R: Runtime,
    NNDescentGpu<T, R>: NNDescentQuery<T>,
{
    let verbosity = parse_verbosity_level(verbose);

    let (knn_indices, knn_dist) = match precomputed_knn {
        Some((indices, distances)) => {
            if verbosity.normal_verbosity() {
                println!("Using precomputed kNN graph...");
            }
            (indices, distances)
        }
        None => {
            if verbosity.normal_verbosity() {
                println!("Running GPU nearest neighbour search using {}...", ann_type);
            }
            let start_knn = Instant::now();
            let result =
                run_ann_search_gpu::<T, R>(data, k, ann_type, nn_params, device, seed, verbose)?;
            if verbosity.normal_verbosity() {
                println!("GPU kNN search done in: {:.2?}.", start_knn.elapsed());
            }
            result
        }
    };

    if verbosity.normal_verbosity() {
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

    if verbosity.normal_verbosity() {
        println!(
            "Finalised graph generation in {:.2?}.",
            start_graph.elapsed()
        );
    }

    Ok((graph, knn_indices, knn_dist))
}

/// Run UMAP with GPU-accelerated nearest neighbour search
///
/// Identical to `umap` except the kNN graph is constructed on the GPU.
/// Embedding initialisation and optimisation remain on the CPU (the parallel
/// Adam optimiser is already extremely fast for the 2D embedding updates).
///
/// ### Params
///
/// * `data` - Input data matrix (samples x features)
/// * `precomputed_knn` - Optional precomputed kNN (indices, distances)
///   excluding self.
/// * `umap_params` - GPU UMAP parameters.
/// * `device` - The GPU device to use.
/// * `seed` - Random seed.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>` where outer vector has length
/// `n_dim` and inner vectors have length `n_samples`.
#[cfg(feature = "gpu")]
pub fn umap_gpu<T, R>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    umap_params: &UmapParamsGpu<T>,
    device: R::Device,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat + AnnSearchGpuFloat,
    R: Runtime,
    StandardNormal: Distribution<T>,
    NNDescentGpu<T, R>: NNDescentQuery<T>,
{
    let verbosity = parse_verbosity_level(verbose);

    let init_type = parse_initilisation(
        &umap_params.initialisation,
        umap_params.randomised,
        umap_params.init_range,
    )
    .unwrap_or_else(|| {
        println!(
            "Unknown initialisation provided: {:?}. Defaulting to PCA.",
            umap_params.initialisation,
        );
        EmbdInit::PcaInit {
            range: None,
            randomised: true,
        }
    });
    let optimiser = parse_umap_optimiser_gpu(&umap_params.optimiser).unwrap_or_else(|| {
        println!(
            "Unknown optimiser string provided ({:?}). Defaulting to GPU-accelerated Adam",
            umap_params.optimiser
        );
        UmapOptimiserGpu::default()
    });

    if verbosity.normal_verbosity() {
        println!(
            "Running umap (GPU kNN) with alpha: {:.2?} and beta: {:.2?}",
            umap_params.optim_params.a, umap_params.optim_params.b
        );
    }

    let (graph, _, _) = construct_umap_graph_gpu::<T, R>(
        data,
        precomputed_knn,
        umap_params.k,
        umap_params.ann_type.clone(),
        &umap_params.umap_graph_params,
        &umap_params.nn_params,
        umap_params.optim_params.n_epochs,
        device.clone(),
        seed,
        verbose,
    )?;

    if verbosity.normal_verbosity() {
        println!(
            "Initialising embedding via {} layout...",
            umap_params.initialisation
        );
    }

    let start_layout = Instant::now();

    let mut embd = initialise_embedding(&init_type, umap_params.n_dim, seed as u64, &graph, data)?;

    let graph_adj = coo_to_adjacency_list(&graph);

    if verbosity.normal_verbosity() {
        println!(
            "Optimising embedding via {} ({} epochs) on {} edges...",
            match optimiser {
                UmapOptimiserGpu::Adam => "Adam",
                UmapOptimiserGpu::Sgd => "SGD",
                UmapOptimiserGpu::AdamParallel => "Adam (multi-threaded)",
                UmapOptimiserGpu::AdamGpu => "Adam (GPU-accelerated)",
            },
            umap_params.optim_params.n_epochs,
            graph.col_indices.len().separate_with_underscores()
        );
    }

    match optimiser {
        UmapOptimiserGpu::Adam => optimise_embedding_adam(
            &mut embd,
            &graph_adj,
            &umap_params.optim_params,
            seed as u64,
            verbose,
        )?,
        UmapOptimiserGpu::Sgd => {
            optimise_embedding_sgd(
                &mut embd,
                &graph_adj,
                &umap_params.optim_params,
                seed as u64,
                verbose,
            )?;
        }
        UmapOptimiserGpu::AdamParallel => {
            optimise_embedding_adam_parallel(
                &mut embd,
                &graph_adj,
                &umap_params.optim_params,
                seed as u64,
                verbose,
            )?;
        }
        UmapOptimiserGpu::AdamGpu => {
            // downcast to f32 for GPU here...
            let mut embd_f32: Vec<Vec<f32>> = embd
                .iter()
                .map(|p| p.iter().map(|&x| x.to_f32().unwrap()).collect())
                .collect();
            let graph_adj_f32: Vec<Vec<(usize, f32)>> = graph_adj
                .iter()
                .map(|edges| {
                    edges
                        .iter()
                        .map(|&(j, w)| (j, w.to_f32().unwrap()))
                        .collect()
                })
                .collect();
            let params_f32 = umap_params.optim_params.cast::<f32>();

            optimise_embedding_adam_gpu::<R, f32>(
                &mut embd_f32,
                &graph_adj_f32,
                &params_f32,
                device,
                seed as u64,
                verbose,
            )?;

            for (i, point) in embd.iter_mut().enumerate() {
                for (j, coord) in point.iter_mut().enumerate() {
                    *coord = T::from(embd_f32[i][j]).unwrap();
                }
            }
        }
    }

    if verbosity.normal_verbosity() {
        println!(
            "Initialised and optimised embedding in: {:.2?}.",
            start_layout.elapsed()
        );
        println!("UMAP (GPU) complete!");
    }

    let n_samples = embd.len();
    let mut transposed = vec![vec![T::zero(); n_samples]; umap_params.n_dim];

    for sample_idx in 0..n_samples {
        for dim_idx in 0..umap_params.n_dim {
            transposed[dim_idx][sample_idx] = embd[sample_idx][dim_idx];
        }
    }

    Ok(transposed)
}

//////////////
// tSNE GPU //
//////////////

/// t-SNE parameters for GPU-accelerated nearest neighbour search
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct TsneParamsGpu<T> {
    /// Number of output dimensions (typically 2)
    pub n_dim: usize,
    /// Perplexity parameter controlling neighbourhood size (typical: 5-50)
    pub perplexity: T,
    /// Which GPU ANN search to use. One of `"exhaustive_gpu"`, `"ivf_gpu"` or
    /// `"nndescent_gpu"`. Defaults to `"ivf_gpu"`.
    pub ann_type: String,
    /// Embedding initialisation method: `"pca"`, `"random"`, or `"spectral"`
    pub initialisation: String,
    /// Optional initialisation range
    pub init_range: Option<T>,
    /// GPU nearest neighbour parameters
    pub nn_params: NearestNeighbourParamsGpu<T>,
    /// tSNE optimisation parameters
    pub optim_params: TsneOptimParams<T>,
    /// Use randomised SVD for PCA initialisation
    pub randomised_init: bool,
}

#[cfg(feature = "gpu")]
impl<T> Default for TsneParamsGpu<T>
where
    T: ManifoldsFloat,
{
    fn default() -> Self {
        Self {
            n_dim: 2,
            perplexity: T::from_f64(30.0).unwrap(),
            ann_type: "ivf_gpu".to_string(),
            initialisation: "pca".to_string(),
            init_range: None,
            nn_params: NearestNeighbourParamsGpu::default(),
            optim_params: TsneOptimParams {
                n_epochs: 1000,
                lr: None,
                early_exag_iter: 250,
                early_exag_factor: T::from_f64(12.0).unwrap(),
                late_exag_factor: None,
                theta: T::from_f64(0.5).unwrap(),
                n_interp_points: 3,
            },
            randomised_init: true,
        }
    }
}

#[cfg(feature = "gpu")]
impl<T> TsneParamsGpu<T>
where
    T: ManifoldsFloat,
{
    /// Create new GPU t-SNE parameters with full control over every field.
    ///
    /// This constructor exposes every field directly, including the
    /// optimiser parameters. For sensible defaults, use
    /// [`TsneParamsGpu::default`] or [`TsneParamsGpu::new_default_2d`] instead.
    ///
    /// ### Params
    ///
    /// * `n_dim` - Number of output dimensions.
    /// * `perplexity` - Perplexity parameter controlling neighbourhood size
    ///   (typical: 5-50).
    /// * `ann_type` - GPU ANN algorithm. One of `"exhaustive_gpu"`,
    ///   `"ivf_gpu"` or `"nndescent_gpu"`.
    /// * `initialisation` - Embedding initialisation method: `"pca"`,
    ///   `"random"`, or `"spectral"`.
    /// * `init_range` - Optional initialisation range.
    /// * `nn_params` - GPU nearest neighbour parameters.
    /// * `optim_params` - t-SNE optimisation parameters. These cover the
    ///   learning rate, number of epochs, early/late exaggeration, the
    ///   Barnes-Hut `theta`, and the number of FFT interpolation points.
    /// * `randomised_init` - Whether randomised SVD is used for PCA
    ///   initialisation.
    ///
    /// ### Returns
    ///
    /// A fully specified set of GPU t-SNE parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_dim: usize,
        perplexity: T,
        ann_type: String,
        initialisation: String,
        init_range: Option<T>,
        nn_params: NearestNeighbourParamsGpu<T>,
        optim_params: TsneOptimParams<T>,
        randomised_init: bool,
    ) -> Self {
        Self {
            n_dim,
            perplexity,
            ann_type,
            initialisation,
            init_range,
            nn_params,
            optim_params,
            randomised_init,
        }
    }

    /// Default parameters for standard 2D visualisation.
    ///
    /// Returns the default parameters but lets you tune `perplexity`, the
    /// characteristic t-SNE knob controlling neighbourhood size.
    ///
    /// ### Params
    ///
    /// * `perplexity` - Perplexity parameter (typical: 5-50). Defaults to
    ///   `30.0`.
    ///
    /// ### Returns
    ///
    /// Hopefully sensible standard parameters for 2D visualisation with GPU
    /// kNN search.
    pub fn new_default_2d(perplexity: Option<T>) -> Self {
        let perplexity = perplexity.unwrap_or_else(|| T::from_f64(30.0).unwrap());

        Self {
            perplexity,
            ..Self::default()
        }
    }
}

/// Construct affinity graph for t-SNE using GPU-accelerated kNN search
///
/// Identical to `construct_tsne_graph` except the kNN runs on the GPU.
/// Gaussian affinity computation and symmetrisation remain on CPU.
///
/// ### Params
///
/// * `data` - Input data matrix (samples × features)
/// * `precomputed_knn` - Precomputed kNN (indices, distances) excluding self
/// * `perplexity` - Target perplexity
/// * `ann_type` - GPU ANN method: `"exhaustive_gpu"`, `"ivf_gpu"` or
///   `"nndescent_gpu"`
/// * `nn_params` - GPU nearest neighbour search parameters
/// * `device` - GPU device to use
/// * `seed` - Random seed
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Tuple of (symmetric affinity graph, knn_indices, knn_dist)
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn construct_tsne_graph_gpu<T, R>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    perplexity: T,
    ann_type: String,
    nn_params: &NearestNeighbourParamsGpu<T>,
    device: R::Device,
    seed: usize,
    verbose: usize,
) -> TsneGraph<T>
where
    T: ManifoldsFloat + AnnSearchGpuFloat,
    R: Runtime,
    NNDescentGpu<T, R>: NNDescentQuery<T>,
{
    let verbosity = parse_verbosity_level(verbose);

    let (knn_indices, knn_dist) = match precomputed_knn {
        Some((indices, distances)) => {
            if verbosity.normal_verbosity() {
                println!("Using precomputed kNN graph...");
            }
            (indices, distances)
        }
        None => {
            let k_float = perplexity * T::from_f64(3.0).unwrap();
            let k = k_float.to_usize().unwrap().max(5).min(data.nrows() - 1);

            if verbosity.normal_verbosity() {
                println!("Running GPU kNN search (k={}) using {}...", k, ann_type);
            }

            let start_knn = Instant::now();
            let result =
                run_ann_search_gpu::<T, R>(data, k, ann_type, nn_params, device, seed, verbose)?;

            if verbosity.normal_verbosity() {
                println!("GPU kNN search done in: {:.2?}.", start_knn.elapsed());
            }

            result
        }
    };

    if verbosity.normal_verbosity() {
        println!("Computing Gaussian affinities and symmetrising...");
    }

    let start_graph = Instant::now();

    let directed_graph = gaussian_knn_affinities(
        &knn_indices,
        &knn_dist,
        perplexity,
        T::from_f64(1e-5).unwrap(),
        200,
        nn_params.dist_metric == "euclidean",
    )?;

    let graph = symmetrise_affinities_tsne(directed_graph);

    if verbosity.normal_verbosity() {
        println!(
            "Finalised graph generation in {:.2?}.",
            start_graph.elapsed()
        );
    }

    Ok((graph, knn_indices, knn_dist))
}

/// Run t-SNE with GPU-accelerated nearest neighbour search (FFT build)
///
/// Identical to `tsne` except the kNN graph is constructed on the GPU.
/// Optimisation (Barnes-Hut or FFT) stays on CPU.
///
/// ### Params
///
/// * `data` - Input data matrix (samples × features)
/// * `precomputed_knn` - Optional precomputed kNN, indices and distances
///   excluding self
/// * `params` - GPU t-SNE parameters
/// * `approx_type` - Repulsive-force approximation: `"barnes_hut" | "bh"` or
///   `"fft"`
/// * `device` - GPU device to use
/// * `seed` - Random seed
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding as `Vec<Vec<T>>` with shape `[n_dim][n_samples]`.
#[cfg(all(feature = "gpu", feature = "fft_tsne"))]
pub fn tsne_gpu<T, R>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    params: &TsneParamsGpu<T>,
    approx_type: &str,
    device: R::Device,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat + AnnSearchGpuFloat + FftwFloat,
    R: Runtime,
    StandardNormal: Distribution<T>,
    NNDescentGpu<T, R>: NNDescentQuery<T>,
{
    if params.n_dim != 2 {
        return Err(ManifoldsError::IncorrectDim {
            n_dim: params.n_dim,
        });
    }

    let verbosity = parse_verbosity_level(verbose);

    let (graph, _, _) = construct_tsne_graph_gpu::<T, R>(
        data,
        precomputed_knn,
        params.perplexity,
        params.ann_type.clone(),
        &params.nn_params,
        device,
        seed,
        verbose,
    )?;

    if verbosity.normal_verbosity() {
        println!("Initialising embedding via {}...", &params.initialisation);
    }

    let init_type = parse_initilisation(
        &params.initialisation,
        params.randomised_init,
        params.init_range,
    )
    .unwrap_or_else(|| {
        println!(
            "Unknown initialisation provided: {:?}. Defaulting to PCA.",
            params.initialisation,
        );
        EmbdInit::PcaInit {
            range: Some(T::from_f64(1e-2).unwrap()),
            randomised: true,
        }
    });

    let mut embd = initialise_embedding(&init_type, params.n_dim, seed as u64, &graph, data)?;

    let tsne_approx = parse_tsne_optimiser(approx_type).unwrap_or_default();

    let start_optim = Instant::now();
    match tsne_approx {
        TsneOpt::BarnesHut => {
            if verbosity.normal_verbosity() {
                println!(
                    "Optimising via Barnes-Hut t-SNE ({} epochs)...",
                    params.optim_params.n_epochs
                );
            }
            optimise_bh_tsne(&mut embd, &params.optim_params, &graph, verbose);
        }
        TsneOpt::Fft => {
            if verbosity.normal_verbosity() {
                println!(
                    "Optimising via FFT Interpolation-based t-SNE ({} epochs)...",
                    params.optim_params.n_epochs
                );
            }
            optimise_fft_tsne(&mut embd, &params.optim_params, &graph, verbose)?;
        }
    }

    if verbosity.normal_verbosity() {
        println!("Optimisation complete in {:.2?}.", start_optim.elapsed());
    }

    let n_samples = embd.len();
    let mut transposed = vec![vec![T::zero(); n_samples]; params.n_dim];

    for sample_idx in 0..n_samples {
        for dim_idx in 0..params.n_dim {
            transposed[dim_idx][sample_idx] = embd[sample_idx][dim_idx];
        }
    }

    Ok(transposed)
}

/// Run t-SNE with GPU-accelerated nearest neighbour search (non-FFT build)
///
/// Identical to `tsne` except the kNN graph is constructed on the GPU.
/// Barnes-Hut optimisation stays on CPU. Calling with `approx_type = "fft"`
/// panics; recompile with the `fft_tsne` feature for FFT support.
///
/// ### Params
///
/// * `data` - Input data matrix (samples × features)
/// * `precomputed_knn` - Optional precomputed kNN, indices and distances
///   excluding self
/// * `params` - GPU t-SNE parameters
/// * `approx_type` - Repulsive-force approximation: `"barnes_hut" | "bh"`
/// * `device` - GPU device to use
/// * `seed` - Random seed
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding as `Vec<Vec<T>>` with shape `[n_dim][n_samples]`.
#[cfg(all(feature = "gpu", not(feature = "fft_tsne")))]
pub fn tsne_gpu<T, R>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    params: &TsneParamsGpu<T>,
    approx_type: &str,
    device: R::Device,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat + AnnSearchGpuFloat,
    R: Runtime,
    StandardNormal: Distribution<T>,
    NNDescentGpu<T, R>: NNDescentQuery<T>,
{
    if params.n_dim != 2 {
        return Err(ManifoldsError::IncorrectDim {
            n_dim: params.n_dim,
        });
    }

    let verbosity = parse_verbosity_level(verbose);

    let (graph, _, _) = construct_tsne_graph_gpu::<T, R>(
        data,
        precomputed_knn,
        params.perplexity,
        params.ann_type.clone(),
        &params.nn_params,
        device,
        seed,
        verbose,
    )?;

    if verbosity.normal_verbosity() {
        println!("Initialising embedding via {}...", &params.initialisation);
    }

    let init_type = parse_initilisation(
        &params.initialisation,
        params.randomised_init,
        params.init_range,
    )
    .unwrap_or_else(|| {
        println!(
            "Unknown initialisation provided: {:?}. Defaulting to PCA.",
            params.initialisation,
        );
        EmbdInit::PcaInit {
            range: Some(T::from_f64(1e-2).unwrap()),
            randomised: true,
        }
    });

    let mut embd = initialise_embedding(&init_type, params.n_dim, seed as u64, &graph, data)?;

    let tsne_approx = parse_tsne_optimiser(approx_type).unwrap_or_default();

    let start_optim = Instant::now();
    match tsne_approx {
        TsneOpt::BarnesHut => {
            if verbosity.normal_verbosity() {
                println!(
                    "Optimising via Barnes-Hut t-SNE ({} epochs)...",
                    params.optim_params.n_epochs
                );
            }
            optimise_bh_tsne(&mut embd, &params.optim_params, &graph, verbose);
        }
        TsneOpt::Fft => {
            panic!("FFT-accelerated t-SNE not available. Recompile with 'fft_tsne' feature or use 'barnes_hut' approximation.");
        }
    }

    if verbosity.normal_verbosity() {
        println!("Optimisation complete in {:.2?}.", start_optim.elapsed());
    }

    let n_samples = embd.len();
    let mut transposed = vec![vec![T::zero(); n_samples]; params.n_dim];

    for sample_idx in 0..n_samples {
        for dim_idx in 0..params.n_dim {
            transposed[dim_idx][sample_idx] = embd[sample_idx][dim_idx];
        }
    }

    Ok(transposed)
}
