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
    n_dim: usize,
    /// Number of neighbours
    k: usize,
    /// Which optimiser to use. Defaults to `"adam_parallel"`.
    optimiser: String,
    /// (Approximate) Nearest neighbour method. One of `"exhaustive"`, `"ivf"`,
    /// `"hnsw"`, `"nndescent"`, `"annoy"`, `"kmknn"` or `"balltree"`.
    ann_type: String,
    /// Which embedding initialisation to use. Defaults to spectral clustering.
    initialisation: String,
    /// Optional initialisation range to use
    init_range: Option<T>,
    /// Nearest neighbour parameters.
    nn_params: NearestNeighbourParams<T>,
    /// Parameters for the UMAP graph generation.
    umap_graph_params: UmapGraphParams<T>,
    /// Parameters to use for the UMAP optimiser.
    optim_params: UmapOptimParams<T>,
    /// Shall randomised SVC be used for PCA-based embedding
    randomised: bool,
}

impl<T> UmapParams<T>
where
    T: ManifoldsFloat,
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
    /// * `ann_type` - (Approximate) Nearest neighbour method: `"exhaustive"`,
    ///   `"kmknn"`, `"balltree"`, `"annoy"`, `"hnsw"`, or `"nndescent"`. If you
    ///   provide a weird string, the function will default to `"kmknn"`
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
        optim_params: Option<UmapOptimParams<T>>,
        umap_graph_params: Option<UmapGraphParams<T>>,
        randomised: Option<bool>,
    ) -> Self {
        // sensible defaults
        let n_dim = n_dim.unwrap_or(2);
        let k = k.unwrap_or(15);
        let optimiser = optimiser.unwrap_or("adam_parallel".to_string());
        let ann_type = ann_type.unwrap_or("kmknn".to_string());
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
    ///   `0.5`.
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
        let min_dist = min_dist.unwrap_or(T::from_f64(0.5).unwrap());
        let spread = spread.unwrap_or(T::from_f64(1.0).unwrap());

        Self {
            n_dim,
            k,
            optimiser: "adam_parallel".into(),
            ann_type: "kmknn".into(),
            initialisation: "spectral".into(),
            init_range: None,
            nn_params: NearestNeighbourParams::default(),
            optim_params: UmapOptimParams::from_min_dist_spread(
                min_dist, spread, None, None, None, None, None, None, None,
            ),
            umap_graph_params: UmapGraphParams::default(),
            randomised: false,
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
/// * `verbose` - Controls verbosity
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
    verbose: bool,
) -> (CoordinateList<T>, Vec<Vec<usize>>, Vec<Vec<T>>)
where
    T: ManifoldsFloat,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let (knn_indices, knn_dist) = match precomputed_knn {
        Some((indices, distances)) => {
            if verbose {
                println!("Using precomputed kNN graph...");
            }
            (indices, distances)
        }
        None => {
            if verbose {
                println!(
                    "Running approximate nearest neighbour search using {}...",
                    ann_type
                );
            }
            let start_knn = Instant::now();
            let result = run_ann_search(data, k, ann_type, nn_params, seed, verbose);
            if verbose {
                println!("kNN search done in: {:.2?}.", start_knn.elapsed());
            }
            result
        }
    };

    if verbose {
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
/// * `verbose` - Controls verbosity of the function.
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
    verbose: bool,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
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
        precomputed_knn,
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

    let mut embd = initialise_embedding(&init_type, umap_params.n_dim, seed as u64, &graph, data)?;

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

impl<T> TsneParams<T>
where
    T: ManifoldsFloat,
{
    /// Create new t-SNE parameters with sensible defaults
    ///
    /// ### Params
    ///
    /// * `n_dim` - Number of output dimensions. Default: 2
    /// * `perplexity` - Perplexity parameter. Default: 30.0
    /// * `init_range` - Optional initialisation range to fix the initial
    ///   embedding between certain values.
    /// * `lr` - Learning rate. Default: 200.0
    /// * `n_epochs` - Number of optimization epochs. Default: 1000
    /// * `ann_type` - (Approximate) Nearest neighbour method: `"exhaustive"`,
    ///   `"kmknn"`, `"balltree"`, `"annoy"`, `"hnsw"`, or `"nndescent"`. If you
    ///   provide a weird string, the function will default to `"kmknn"`
    /// * `theta` - Barnes-Hut approximation parameter. Default: 0.5
    /// * `n_interp_points` - Number of interpolation points for the FFT version
    ///   of the optimiser.
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
        let ann_type = ann_type.unwrap_or_else(|| "kmknn".to_string());
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
/// * `verbose` - Print progress information
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
    verbose: bool,
) -> TsneGraph<T>
where
    T: ManifoldsFloat,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let (knn_indices, knn_dist) = match precomputed_knn {
        Some((indices, distances)) => {
            if verbose {
                println!("Using precomputed kNN graph...");
            }
            (indices, distances)
        }
        None => {
            let k_float = perplexity * T::from_f64(3.0).unwrap();
            let k = k_float.to_usize().unwrap().max(5).min(data.nrows() - 1);

            if verbose {
                println!("Running kNN search (k={}) using {}...", k, ann_type);
            }

            let start_knn = Instant::now();
            let result = run_ann_search(data, k, ann_type, nn_params, seed, verbose);

            if verbose {
                println!("kNN search done in: {:.2?}.", start_knn.elapsed());
            }

            result
        }
    };

    if verbose {
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

    if verbose {
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
/// * `verbose` - Print progress information
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
    verbose: bool,
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

    if verbose {
        println!("Initialising embedding via {}...", &params.initialisation);
    }

    // 2. initialise embedding
    let init_type = parse_initilisation(
        &params.initialisation,
        params.randomised_init,
        params.init_range,
    )
    .unwrap_or(EmbdInit::PcaInit {
        randomised: true,
        range: Some(T::from_f64(1e-2).unwrap()),
    });

    let mut embd = initialise_embedding(&init_type, params.n_dim, seed as u64, &graph, data)?;

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
        #[cfg(feature = "fft_tsne")]
        TsneOpt::Fft => {
            if verbose {
                println!(
                    "Optimising via FFT Interpolation-based t-SNE ({} epochs)...",
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
/// * `verbose` - Print progress information
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
    verbose: bool,
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
    if verbose {
        println!("Initialising embedding via {}...", &params.initialisation);
    }

    let init_type = parse_initilisation(
        &params.initialisation,
        params.randomised_init,
        params.init_range,
    )
    .unwrap_or(EmbdInit::PcaInit {
        randomised: false,
        range: Some(T::from_f64(1e-2).unwrap()),
    });

    let mut embd = initialise_embedding(&init_type, params.n_dim, seed as u64, &graph, data)?;

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
        #[cfg(not(feature = "fft_tsne"))]
        TsneOpt::Fft => {
            panic!("FFT-accelerated t-SNE not available. Recompile with 'fft_tsne' feature or use 'barnes_hut' approximation.");
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

impl<T> PhateParams<T>
where
    T: ManifoldsFloat,
{
    /// Create new PHATE parameters with sensible defaults
    ///
    /// ### Params
    ///
    /// * `n_dim` - Number of output dimensions (default: 2)
    /// * `k` - Number of nearest neighbours (default: 5)
    /// * `ann_type` - (Approximate) Nearest neighbour method: `"exhaustive"`,
    ///   `"kmknn"`, `"balltree"`, `"annoy"`, `"hnsw"`, or `"nndescent"`. If you
    ///   provide a weird string, the function will default to `"kmknn"`
    /// * `decay` - Alpha decay parameter controlling kernel bandwidth (default:
    ///   40.0)
    /// * `bandwidth_scale` - Scaling factor for the kernel bandwidth (default:
    ///   1.0)
    /// * `graph_symmetry` - Method for symmetrising the affinity graph:
    ///   `"average"` or `"max"` (default: `"average"`)
    /// * `t_max` - Maximum number of diffusion steps for optimal `t` selection
    /// * `gamma` - Informational distance constant; `1.0` gives PHATE, `-1.0`
    ///   gives MDS (default: 1.0)
    /// * `n_landmarks` - Number of landmark points for large-scale
    ///   approximation; `None` disables landmarks
    /// * `landmark_method`: Method for selecting landmarks: `"spectral"`,
    ///   `"random"` or `"density"`
    /// * `n_svd` - Number of SVD components to retain during diffusion
    ///   (default: determined by diffusion params)
    /// * `t_custom` - Fixed diffusion time `t`; overrides automatic selection
    ///   if provided
    /// * `mds_method` - MDS implementation to use (default: `"sgd_dense"`)
    /// * `mds_iter` - Number of iterations for MDS fitting; `None` uses the MDS
    ///   default
    /// * `randomised` - Whether to use randomised SVD for PCA-based initialisation (default: `true`)
    ///
    /// ### Returns
    ///
    /// `PhateParams` with sensible defaults for standard PHATE
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_dim: Option<usize>,
        k: Option<usize>,
        ann_type: Option<String>,
        decay: Option<T>,
        bandwidth_scale: Option<T>,
        graph_symmetry: Option<String>,
        t_max: Option<usize>,
        gamma: Option<T>,
        n_landmarks: Option<usize>,
        landmark_method: Option<String>,
        n_svd: Option<usize>,
        t_custom: Option<usize>,
        mds_method: Option<String>,
        mds_iter: Option<usize>,
        randomised: Option<bool>,
    ) -> Self {
        let phate_diffusion_params = PhateDiffusionParams::new(
            Some(decay.unwrap_or_else(|| T::from_f64(40.0).unwrap())),
            bandwidth_scale.unwrap_or_else(|| T::from_f64(1.0).unwrap()),
            T::from_f64(1e-4).unwrap(),
            graph_symmetry.unwrap_or("average".to_string()),
            n_landmarks,
            landmark_method.unwrap_or("spectral".to_string()),
            n_svd,
            t_max,
            t_custom,
            gamma.unwrap_or_else(|| T::from_f64(1.0).unwrap()),
        );

        Self {
            n_dim: n_dim.unwrap_or(2),
            // knn
            k: k.unwrap_or(5),
            ann_type: ann_type.unwrap_or_else(|| "kmknn".to_string()),
            ann_params: NearestNeighbourParams::default(),
            // diffusion
            diffusion_params: phate_diffusion_params,
            // mds
            mds_method: mds_method.unwrap_or_else(|| "sgd_dense".to_string()),
            mds_iter,
            randomised: randomised.unwrap_or(true),
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
/// * `verbose` - Print progress information
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
    verbose: bool,
) -> Result<PhateDiffusion<T>, ManifoldsError>
where
    T: ManifoldsFloat,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let (knn_indices, knn_dist) = match precomputed_knn {
        Some((indices, distances)) => {
            if verbose {
                println!("Using precomputed kNN graph...");
            }
            (indices, distances)
        }
        None => {
            if verbose {
                println!(
                    "Running approximate nearest neighbour search using {}...",
                    ann_type
                );
            }
            let start_knn = Instant::now();
            let result = run_ann_search(data, k, ann_type.to_string(), nn_params, seed, verbose);
            if verbose {
                println!("kNN search done in: {:.2?}.", start_knn.elapsed());
            }
            result
        }
    };

    if verbose {
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

    if verbose {
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
            if verbose {
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
            if verbose {
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
/// * `verbose` - Print progress information
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
    verbose: bool,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
    StandardNormal: Distribution<T>,
{
    let start_phate = Instant::now();

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
            if verbose {
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
    if verbose {
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
            if verbose {
                println!("Powering diffusion operator...");
            }
            let powered = matrix_power(&operator, t);
            let potential = calculate_potential(&powered, 1, phate_params.diffusion_params.gamma);

            if verbose {
                println!(
                    "Potential shape: {} × {} - calculated in {:.2?}.",
                    potential.shape().0,
                    potential.shape().1,
                    start_embed.elapsed()
                );
            }

            let res = match mds_method {
                MdsMethod::ClassicMds => {
                    if verbose {
                        println!("Computing pairwise distances, running classic MDS...");
                    }
                    let distances = compute_potential_distances(&potential, &dist);
                    classic_mds(&distances, phate_params.n_dim, mds_params.randomised, seed)
                }
                MdsMethod::SgdDense => {
                    if verbose {
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
            if verbose {
                println!(
                    "Powering landmark operator ({} landmarks)...",
                    landmarks.get_n_landmarks()
                );
            }
            let landmark_powered = landmarks.power(t);
            let landmark_potential =
                calculate_potential(&landmark_powered, 1, phate_params.diffusion_params.gamma);

            if verbose {
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

            if verbose {
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

            if verbose {
                println!("Interpolating to full N points via Nyström...");
            }
            landmarks.interpolate_embedding(&landmark_embedding)
        }
    };

    if verbose {
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

impl<T> PacmapParams<T>
where
    T: ManifoldsFloat,
{
    /// Generate a new instance of the PaCMAP parameters
    ///
    /// ### Params
    ///
    /// * `n_dim` - Number of dimensions for the embedding
    /// * `k` - Number of near neighbours. Default 10 (paper default; lower than
    ///   UMAP's 15 since PaCMAP is less sensitive to k). The function will use
    ///   `mn_candidate_end` neighbours for in the k-nearest neighbour searches
    ///   generally speaking.
    /// * `ann_type` - (Approximate) Nearest neighbour method: `"exhaustive"`,
    ///   `"kmknn"`, `"balltree"`, `"annoy"`, `"hnsw"`, or `"nndescent"`. If you
    ///   provide a weird string, the function will default to `"kmknn"`
    /// * `optimiser_type` - Which optimiser to use. Options are `"adam"` and
    ///   `"adam_parallel"`. Defaults to the parallel version.
    /// * `n_mid_near` - Mid-near pairs per point. Default 2.
    /// * `n_further` - Start index into kNN list for mid-near candidate window. ]
    ///   Default 4 (skip the 4 nearest).
    /// * `mn_candidate_start` - Start index into kNN list for mid-near
    ///   candidate window. Default 4 (skip the 4 nearest).
    /// * `mn_candidate_end` - End index into kNN list for mid-near candidate
    ///   window. Default 50. Requires k >= this value.
    /// * `initialisation` - Embedding initialisation. Default `"pca"`. PCA is
    ///   strongly recommended for PaCMAP as random init degrades global
    ///   structure.
    /// * `nn_params` - Nearest neighbour search parameters.
    /// * `optim_params` - Optimiser parameters.
    ///
    /// ### Returns
    ///
    /// Initialised self
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_dim: Option<usize>,
        k: Option<usize>,
        ann_type: Option<String>,
        optimiser_type: Option<String>,
        n_mid_near: Option<usize>,
        n_further: Option<usize>,
        mn_candidate_start: Option<usize>,
        mn_candidate_end: Option<usize>,
        initialisation: Option<String>,
        nn_params: Option<NearestNeighbourParams<T>>,
        optim_params: Option<PacmapOptimParams<T>>,
    ) -> Self {
        let mn_candidate_end = mn_candidate_end.unwrap_or(50);
        let k = k.unwrap_or(10).max(mn_candidate_end);

        Self {
            n_dim: n_dim.unwrap_or(2),
            k,
            ann_type: ann_type.unwrap_or("kmknn".to_string()),
            optimiser_type: optimiser_type.unwrap_or("adam_parallel".to_string()),
            n_mid_near: n_mid_near.unwrap_or(2),
            n_further: n_further.unwrap_or(2),
            mn_candidate_start: mn_candidate_start.unwrap_or(4),
            mn_candidate_end,
            initialisation: initialisation.unwrap_or("pca".to_string()),
            nn_params: nn_params.unwrap_or_default(),
            optim_params: optim_params.unwrap_or_default(),
        }
    }
}

/// Default implementation of PaCMAP
impl<T> Default for PacmapParams<T>
where
    T: ManifoldsFloat,
{
    fn default() -> Self {
        Self::new(
            None, None, None, None, None, None, None, None, None, None, None,
        )
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
/// * `verbose` - Controls verbosity.
///
/// ### Returns
///
/// Embedding as `Vec<Vec<T>>` with shape `[n_dim][n_samples]`.
pub fn pacmap<T>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    params_pacmap: &PacmapParams<T>,
    seed: usize,
    verbose: bool,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
    HnswIndex<T>: HnswState<T>,
    StandardNormal: Distribution<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let n_samples = data.nrows();

    let (knn_indices, _) = match precomputed_knn {
        Some((indices, distances)) => {
            if verbose {
                println!("Using precomputed kNN graph...");
            }
            (indices, distances)
        }
        None => {
            if verbose {
                println!(
                    "Running approximate nearest neighbour search using {} (k={})...",
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
            );
            if verbose {
                println!("kNN search done in: {:.2?}.", start_knn.elapsed());
            }
            result
        }
    };

    if verbose {
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

    if verbose {
        println!(
            "Pairs generated in {:.2?}: {} near, {} mid-near, {} further.",
            end_pairs,
            pairs.near.len().separate_with_underscores(),
            pairs.mid_near.len().separate_with_underscores(),
            pairs.further.len().separate_with_underscores()
        );
    }

    let init_type = parse_initilisation(&params_pacmap.initialisation, true, None).unwrap_or(
        EmbdInit::PcaInit {
            randomised: true,
            range: None,
        },
    );

    if verbose {
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
            optimise_pacmap_parallel(&mut embd, &pairs, &params_pacmap.optim_params, verbose);
        }
        PacMapOptimiser::Adam => {
            optimise_pacmap(&mut embd, &pairs, &params_pacmap.optim_params, verbose);
        }
    }

    let end_optim = start_optim.elapsed();

    if verbose {
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

impl<T> DiffusionMapsParams<T>
where
    T: ManifoldsFloat,
{
    /// Construct a new `DiffusionMapsParams`.
    ///
    /// All parameters are optional; `None` falls back to the stated defaults.
    ///
    /// ### Params
    ///
    /// * `n_dim` - Output embedding dimensionality. Default 2.
    /// * `k` - Number of nearest neighbours for graph construction. Default 5.
    /// * `ann_type` - ANN algorithm: `"exhaustive"`, `"kmknn"`, `"balltree"`,
    ///   `"annoy"`, `"hnsw"`, or `"nndescent"`. Default `"kmknn"`.
    /// * `bandwidth_scale` - Multiplicative factor on the adaptive kernel
    ///   bandwidth. Default 1.0.
    /// * `thresh` - Kernel entries below this value are set to zero. Default
    ///   1e-4.
    /// * `graph_symmetry` - Symmetrisation method: `"add"`, `"multiply"`,
    ///   `"mnn"` or `"none"`. Default `"add"`.
    /// * `alpha_norm` - Anisotropic normalisation exponent in [0, 1]. Default
    ///   1.0.
    /// * `t_max` - Maximum diffusion steps for VNE-based optimal t. Default
    ///   100.
    /// * `t_custom` - If provided, fixes t to this value instead of
    ///   auto-detecting.
    /// * `n_landmarks` - Number of landmarks. `None` or `>= n` runs full DM.
    /// * `landmark_method` - `"random"`, `"spectral"`, or `"density"`. Default
    ///   `"spectral"`.
    /// * `n_svd` - SVD components for spectral landmark selection. Ignored
    ///   otherwise.
    ///
    /// ### Returns
    ///
    /// Initialised `DiffusionMapsParams`
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_dim: Option<usize>,
        k: Option<usize>,
        ann_type: Option<String>,
        bandwidth_scale: Option<T>,
        thresh: Option<T>,
        graph_symmetry: Option<String>,
        alpha_norm: Option<T>,
        t_max: Option<usize>,
        t_custom: Option<usize>,
        n_landmarks: Option<usize>,
        landmark_method: Option<String>,
        n_svd: Option<usize>,
    ) -> Self {
        let t_max = t_max.unwrap_or(PHATE_MAX_T);
        let t = parse_phate_time(t_custom, t_max);

        Self {
            n_dim: n_dim.unwrap_or(2),
            k: k.unwrap_or(5),
            ann_type: ann_type.unwrap_or_else(|| "kmknn".to_string()),
            ann_params: NearestNeighbourParams::default(),
            bandwidth_scale: bandwidth_scale.unwrap_or_else(|| T::from_f64(1.0).unwrap()),
            thresh: thresh.unwrap_or_else(|| T::from_f64(1e-4).unwrap()),
            graph_symmetry: graph_symmetry.unwrap_or_else(|| "add".to_string()),
            alpha_norm: alpha_norm.unwrap_or_else(|| T::from_f64(1.0).unwrap()),
            t,
            n_landmarks,
            landmark_method: landmark_method.unwrap_or_else(|| "spectral".to_string()),
            n_svd,
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
/// * `verbose` - Print progress messages
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
    verbose: bool,
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

    let needs_affinity =
        use_full || matches!(dm_params.landmark_method.as_str(), "spectral" | "density");

    let affinity = if needs_affinity {
        let (knn_indices, knn_dist) = match precomputed_knn {
            Some((indices, distances)) => {
                if verbose {
                    println!("Using precomputed kNN graph...");
                }
                (indices, distances)
            }
            None => {
                if verbose {
                    println!("Running ANN search using {}...", ann_type);
                }
                let start_knn = Instant::now();
                let res = run_ann_search(data, k, ann_type.to_string(), nn_params, seed, verbose);
                if verbose {
                    println!("kNN search done in {:.2?}.", start_knn.elapsed());
                }
                res
            }
        };

        if verbose {
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
        if verbose {
            println!("Skipping full affinity (random landmarks).");
        }
        None
    };

    if use_full {
        let kernel = affinity.unwrap();
        let kernel_norm = apply_anisotropic_normalisation(&kernel, dm_params.alpha_norm);
        let (p_sym, sqrt_degrees) = build_symmetric_diffusion_operator(&kernel_norm);
        return Ok(DiffusionMapsOperator::Full {
            p_sym,
            sqrt_degrees,
        });
    }

    let n_landmarks = dm_params.n_landmarks.unwrap();
    if verbose {
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
    if verbose {
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
/// * `verbose` - Print progress messages
///
/// ### Returns
///
/// Embedding as `Vec<Vec<T>>` with shape `[n_dim][n_samples]`
pub fn diffusion_maps<T>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    dm_params: DiffusionMapsParams<T>,
    seed: usize,
    verbose: bool,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let start_dm = Instant::now();

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
                    if verbose {
                        println!("Finding optimal t (t_max={})...", t_max);
                    }
                    let entropy = landmark_von_neumann_entropy(&p_sym, t_max)?;
                    find_knee_point(&entropy)
                }
                PhateTime::Fixed(t) => t,
            };
            if verbose {
                println!("Using t = {}.", t);
                println!(
                    "Computing top {} eigenpairs of P_sym...",
                    dm_params.n_dim + 1
                );
            }
            let start_eig = Instant::now();
            let n_ask = (dm_params.n_dim + 5).min(sqrt_degrees.len() - 1);
            let (evals, evecs) = compute_largest_eigenpairs_lanczos(&p_sym, n_ask, seed as u64)?;

            if verbose {
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
                    if verbose {
                        println!("Finding optimal t on landmarks (t_max={})...", t_max);
                    }
                    landmarks.find_optimal_t(t_max)?
                }
                PhateTime::Fixed(t) => t,
            };
            if verbose {
                println!(
                    "Using t = {}. Eigendecomposing {}x{} landmark operator...",
                    t,
                    landmarks.get_n_landmarks(),
                    landmarks.get_n_landmarks()
                );
            }
            let start_eig = Instant::now();
            let (evals, evecs) = landmarks.eigendecompose(dm_params.n_dim, seed as u64)?;
            if verbose {
                println!("Eigendecomposition done in {:.2?}.", start_eig.elapsed());
                println!("Nystroem-extending to full data...");
            }
            let (landmark_embedding, lambdas) =
                landmarks.compute_landmark_embedding(&evals, &evecs, dm_params.n_dim, t);
            landmarks.nystrom_extend(&landmark_embedding, &lambdas)
        }
    };

    if verbose {
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
impl<T> ParametricUmapParams<T>
where
    T: ManifoldsFloat + Element,
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
    /// * `ann_type` - (Approximate) Nearest neighbour method: `"exhaustive"`,
    ///   `"kmknn"`, `"balltree"`, `"annoy"`, `"hnsw"`, or `"nndescent"`. If you
    ///   provide a weird string, the function will default to `"kmknn"`
    /// * `hidden_layers` - Hidden layer sizes for MLP. Default
    ///   `vec![128, 128, 128]`.
    /// * `nn_params` - Nearest neighbour parameters. Default uses sensible
    ///   values.
    /// * `umap_graph_params` - UMAP graph parameters. Default uses sensible
    ///   values.
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
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances. Needs
///   to be a tuple of `(Vec<Vec<usize>>, Vec<Vec<T>>)` with indices and
///   distances excluding self.
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
pub fn parametric_umap<T, B>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    umap_params: &ParametricUmapParams<T>,
    device: &B::Device,
    seed: usize,
    verbose: bool,
) -> Vec<Vec<T>>
where
    T: ManifoldsFloat + Element,
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
        precomputed_knn,
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
/// Returns a tuple of embedding and the `TrainedUmapModel` for further usage.
pub fn train_parametric_umap_model<T, B>(
    data: MatRef<T>,
    umap_params: &ParametricUmapParams<T>,
    device: &B::Device,
    seed: usize,
    verbose: bool,
) -> (Vec<Vec<T>>, TrainedUmapModel<B, T>)
where
    T: ManifoldsFloat + Element,
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
        None,
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

    let (embd, trained_model) = train_parametric_umap::<B, T>(
        data,
        graph,
        &model_params,
        &umap_params.train_param,
        device,
        seed,
        verbose,
    );

    (embd, trained_model)
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
    /// Which optimiser to use. Defaults to `"adam_parallel"`.
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
impl<T> UmapParamsGpu<T>
where
    T: ManifoldsFloat,
{
    /// Generate new GPU UMAP parameters
    ///
    /// ### Params
    ///
    /// * `n_dim` - How many dimensions to return. Default `2`.
    /// * `k` - How many neighbours to consider. Default `15`.
    /// * `optimiser` - Which optimiser to use. Default `"adam_parallel"`.
    /// * `ann_type` - Which GPU ANN search to use. One of `"exhaustive_gpu"`,
    ///   `"ivf_gpu"` or `"nndescent_gpu"`. Default `"ivf_gpu"`.
    /// * `initialisation` - Embedding initialisation. Default `"spectral"`.
    /// * `init_range` - Optional initialisation range.
    /// * `nn_params` - GPU nearest neighbour parameters.
    /// * `optim_params` - Optimiser parameters.
    /// * `umap_graph_params` - UMAP graph generation parameters.
    /// * `randomised` - Use randomised SVD for PCA init. Default `false`.
    ///
    /// ### Returns
    ///
    /// Configured `UmapParamsGpu` with sensible defaults.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_dim: Option<usize>,
        k: Option<usize>,
        optimiser: Option<String>,
        ann_type: Option<String>,
        initialisation: Option<String>,
        init_range: Option<T>,
        nn_params: Option<NearestNeighbourParamsGpu<T>>,
        optim_params: Option<UmapOptimParams<T>>,
        umap_graph_params: Option<UmapGraphParams<T>>,
        randomised: Option<bool>,
    ) -> Self {
        Self {
            n_dim: n_dim.unwrap_or(2),
            k: k.unwrap_or(15),
            optimiser: optimiser.unwrap_or("adam_parallel".to_string()),
            ann_type: ann_type.unwrap_or("ivf_gpu".to_string()),
            initialisation: initialisation.unwrap_or("spectral".to_string()),
            init_range,
            nn_params: nn_params.unwrap_or_default(),
            optim_params: optim_params.unwrap_or_default(),
            umap_graph_params: umap_graph_params.unwrap_or_default(),
            randomised: randomised.unwrap_or(false),
        }
    }

    /// Default 2D parameters for GPU UMAP
    ///
    /// ### Params
    ///
    /// * `n_dim` - Number of dimensions. Default `2`.
    /// * `k` - Number of neighbours. Default `15`.
    /// * `min_dist` - Minimum distance. Default `0.5`.
    /// * `spread` - Spread parameter. Default `1.0`.
    ///
    /// ### Returns
    ///
    /// Sensible defaults for 2D visualisation with GPU kNN search.
    pub fn default_2d(
        n_dim: Option<usize>,
        k: Option<usize>,
        min_dist: Option<T>,
        spread: Option<T>,
    ) -> Self {
        let min_dist = min_dist.unwrap_or(T::from_f64(0.5).unwrap());
        let spread = spread.unwrap_or(T::from_f64(1.0).unwrap());

        Self {
            n_dim: n_dim.unwrap_or(2),
            k: k.unwrap_or(15),
            optimiser: "adam_parallel".into(),
            ann_type: "ivf_gpu".into(),
            initialisation: "spectral".into(),
            init_range: None,
            nn_params: NearestNeighbourParamsGpu::default(),
            optim_params: UmapOptimParams::from_min_dist_spread(
                min_dist, spread, None, None, None, None, None, None, None,
            ),
            umap_graph_params: UmapGraphParams::default(),
            randomised: false,
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
/// * `verbose` - Controls verbosity.
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
    verbose: bool,
) -> (CoordinateList<T>, Vec<Vec<usize>>, Vec<Vec<T>>)
where
    T: ManifoldsFloat + AnnSearchGpuFloat,
    R: Runtime,
    NNDescentGpu<T, R>: NNDescentQuery<T>,
{
    let (knn_indices, knn_dist) = match precomputed_knn {
        Some((indices, distances)) => {
            if verbose {
                println!("Using precomputed kNN graph...");
            }
            (indices, distances)
        }
        None => {
            if verbose {
                println!("Running GPU nearest neighbour search using {}...", ann_type);
            }
            let start_knn = Instant::now();
            let result =
                run_ann_search_gpu::<T, R>(data, k, ann_type, nn_params, device, seed, verbose);
            if verbose {
                println!("GPU kNN search done in: {:.2?}.", start_knn.elapsed());
            }
            result
        }
    };

    if verbose {
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
/// * `verbose` - Controls verbosity.
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
    verbose: bool,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat + AnnSearchGpuFloat,
    R: Runtime,
    StandardNormal: Distribution<T>,
    NNDescentGpu<T, R>: NNDescentQuery<T>,
{
    let init_type = parse_initilisation(
        &umap_params.initialisation,
        umap_params.randomised,
        umap_params.init_range,
    )
    .unwrap_or(EmbdInit::RandomInit { range: None });
    let optimiser = parse_umap_optimiser(&umap_params.optimiser).unwrap_or_default();

    if verbose {
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
        device,
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

    let mut embd = initialise_embedding(&init_type, umap_params.n_dim, seed as u64, &graph, data)?;

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

    if verbose {
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
impl<T> TsneParamsGpu<T>
where
    T: ManifoldsFloat,
{
    /// Create new GPU t-SNE parameters with sensible defaults
    ///
    /// ### Params
    ///
    /// * `n_dim` - Number of output dimensions. Default: 2
    /// * `perplexity` - Perplexity parameter. Default: 30.0
    /// * `init_range` - Optional initialisation range
    /// * `lr` - Learning rate. Default: 200.0
    /// * `n_epochs` - Number of optimisation epochs. Default: 1000
    /// * `ann_type` - GPU ANN algorithm: `"exhaustive_gpu"`, `"ivf_gpu"` or
    ///   `"nndescent_gpu"`. Default: `"ivf_gpu"`
    /// * `theta` - Barnes-Hut approximation parameter. Default: 0.5
    /// * `n_interp_points` - Number of interpolation points for the FFT
    ///   version of the optimiser.
    ///
    /// ### Returns
    ///
    /// `TsneParamsGpu` with sensible defaults for GPU-accelerated t-SNE
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
        let ann_type = ann_type.unwrap_or_else(|| "ivf_gpu".to_string());
        let theta = theta.unwrap_or_else(|| T::from_f64(0.5).unwrap());
        let n_interp_points = n_interp_points.unwrap_or(3);

        Self {
            n_dim,
            perplexity,
            ann_type,
            initialisation: "pca".to_string(),
            init_range,
            nn_params: NearestNeighbourParamsGpu::default(),
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
/// * `verbose` - Controls verbosity
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
    verbose: bool,
) -> TsneGraph<T>
where
    T: ManifoldsFloat + AnnSearchGpuFloat,
    R: Runtime,
    NNDescentGpu<T, R>: NNDescentQuery<T>,
{
    let (knn_indices, knn_dist) = match precomputed_knn {
        Some((indices, distances)) => {
            if verbose {
                println!("Using precomputed kNN graph...");
            }
            (indices, distances)
        }
        None => {
            let k_float = perplexity * T::from_f64(3.0).unwrap();
            let k = k_float.to_usize().unwrap().max(5).min(data.nrows() - 1);

            if verbose {
                println!("Running GPU kNN search (k={}) using {}...", k, ann_type);
            }

            let start_knn = Instant::now();
            let result =
                run_ann_search_gpu::<T, R>(data, k, ann_type, nn_params, device, seed, verbose);

            if verbose {
                println!("GPU kNN search done in: {:.2?}.", start_knn.elapsed());
            }

            result
        }
    };

    if verbose {
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

    if verbose {
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
/// * `verbose` - Controls verbosity
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
    verbose: bool,
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

    if verbose {
        println!("Initialising embedding via {}...", &params.initialisation);
    }

    let init_type = parse_initilisation(
        &params.initialisation,
        params.randomised_init,
        params.init_range,
    )
    .unwrap_or(EmbdInit::PcaInit {
        randomised: params.randomised_init,
        range: Some(T::from_f64(1e-2).unwrap()),
    });

    let mut embd = initialise_embedding(&init_type, params.n_dim, seed as u64, &graph, data)?;

    let tsne_approx = parse_tsne_optimiser(approx_type).unwrap_or_default();

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
                    "Optimising via FFT Interpolation-based t-SNE ({} epochs)...",
                    params.optim_params.n_epochs
                );
            }
            optimise_fft_tsne(&mut embd, &params.optim_params, &graph, verbose);
        }
    }

    if verbose {
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
/// * `verbose` - Controls verbosity
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
    verbose: bool,
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

    if verbose {
        println!("Initialising embedding via {}...", &params.initialisation);
    }

    let init_type = parse_initilisation(
        &params.initialisation,
        params.randomised_init,
        params.init_range,
    )
    .unwrap_or(EmbdInit::PcaInit {
        randomised: params.randomised_init,
        range: Some(T::from_f64(1e-2).unwrap()),
    });

    let mut embd = initialise_embedding(&init_type, params.n_dim, seed as u64, &graph, data)?;

    let tsne_approx = parse_tsne_optimiser(approx_type).unwrap_or_default();

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
            panic!("FFT-accelerated t-SNE not available. Recompile with 'fft_tsne' feature or use 'barnes_hut' approximation.");
        }
    }

    if verbose {
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
