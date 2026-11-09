//! Dimensionality reduction algorithms: UMAP, densMAP, t-SNE, den-SNE, DREAMS,
//! PHATE, diffusion maps, PaCMAP and ForceAtlas2, plus parametric UMAP via a
//! neural network encoder with the `parametric` feature.
//!
//! The `gpu` feature adds `*_gpu` entry points: kNN search on the device for all
//! of them, with the UMAP Adam update, the three-kernel FFT t-SNE and PHATE's
//! spectral landmark selection there as well. The `mlx` feature adds the same
//! stages on MLX through string keys on the existing entry points.

#![allow(clippy::needless_range_loop)] // I like loops ... !
#![warn(missing_docs)]

pub mod data;
pub mod errors;
pub mod prelude;
pub mod training;
pub mod utils;

#[cfg(mlx_available)]
pub mod mlx;
#[cfg(feature = "parametric")]
pub mod parametric;

use ann_search_rs::cpu::hnsw::{HnswIndex, HnswState};
use ann_search_rs::cpu::nndescent::{NNDescent, NNDescentQuery};
use ann_search_rs::utils::dist::{parse_ann_dist, Dist};
use ann_search_rs::utils::matrix_to_flat;
use ann_search_rs::utils::nndescent_utils::ApplySortedUpdates;
use faer::MatRef;
use rand_distr::{Distribution, StandardNormal};
use std::{default::Default, time::Instant};
use thousands::*;

#[cfg(feature = "parametric")]
use burn::tensor::Element;
#[cfg(feature = "parametric")]
use num_traits::ToPrimitive;

#[cfg(feature = "gpu")]
use cubecl::Device;

use crate::data::graph::*;
use crate::data::init::*;
use crate::data::nearest_neighbours::*;
use crate::data::pacmap_pairs::*;
use crate::data::structures::*;
use crate::prelude::*;
use crate::training::fa2_optimiser::*;
use crate::training::mds_optimiser::*;
use crate::training::pacmap_optimiser::{
    optimise_pacmap, optimise_pacmap_parallel, PacMapOptimiser,
};
use crate::training::tsne_optimiser::*;
use crate::training::umap_optimisers::*;
use crate::utils::density::{DENSMAP_LAMBDA, DENSNE_LAMBDA};
use crate::utils::diffusions::*;
use crate::utils::dreams::{dreams_setup, DREAMS_LAMBDA};
use crate::utils::math::{compute_largest_eigenpairs_lanczos, von_neumann_entropy_f64};
use crate::utils::potentials::{
    compute_potential_distances_with_fill, dense_potential_rows, pairwise_row_distances,
    potential_of_zero,
};
use crate::utils::sparse_ops::matrix_power;

#[cfg(mlx_available)]
use crate::mlx::tsne_optimiser_mlx::*;
#[cfg(mlx_available)]
use crate::mlx::umap_optimiser_mlx::*;
#[cfg(feature = "parametric")]
use crate::parametric::model::*;
#[cfg(feature = "parametric")]
use crate::parametric::parametric_train::*;
#[cfg(feature = "gpu")]
use crate::training::tsne_optimiser_gpu::*;
#[cfg(feature = "gpu")]
use crate::training::umap_optimiser_gpu::*;
#[cfg(feature = "fft_tsne")]
use crate::utils::fft::FftFloat;

///////////////
// Constants //
///////////////

/// Whether the MLX optimisers were compiled in: the `mlx` feature is on and
/// ann-search-rs found mlx-c at build time.
pub const MLX_AVAILABLE: bool = cfg!(mlx_available);

/// Version of this crate, as declared in `Cargo.toml`.
///
/// Exposed so a wrapper built against it can report which numerics it actually
/// vendored, independently of its own version number.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

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
    /// Defaults to `2`.
    pub n_dim: usize,
    /// Number of neighbours
    /// Defaults to `15`.
    pub k: usize,
    /// Which optimiser to use: `"adam_parallel"` (default), `"adam"`, `"sgd"`,
    /// `"sgd_parallel"`, or `"adam_mlx"` with the `mlx` feature.
    /// `"sgd_parallel"` is lock-free (hogwild) and so not reproducible run to
    /// run with more than one thread.
    pub optimiser: String,
    /// (Approximate) Nearest neighbour method. One of `"exhaustive"`, `"ivf"`,
    /// `"hnsw"`, `"nndescent"`, `"annoy"`, `"kmknn"` or `"balltree"`.
    /// The `mlx` feature adds `"exhaustive_mlx"`, `"ivf_mlx"` and `"nndescent_mlx"`.
    /// Defaults to `"nndescent"`.
    pub ann_type: String,
    /// Which embedding initialisation to use: `"spectral"` (default), `"pca"`
    /// or `"random"`.
    pub initialisation: String,
    /// Optional initialisation range to use
    pub init_range: Option<T>,
    /// Nearest neighbour parameters.
    pub nn_params: NearestNeighbourParams<T>,
    /// Parameters for the UMAP graph generation.
    pub graph_params: UmapGraphParams<T>,
    /// Parameters to use for the UMAP optimiser.
    pub optim_params: UmapOptimParams<T>,
    /// Shall randomised SVD be used for PCA-based embedding
    /// Defaults to `false`.
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
            ann_type: "nndescent".to_string(),
            initialisation: "spectral".to_string(),
            init_range: None,
            nn_params: NearestNeighbourParams::default(),
            optim_params: UmapOptimParams::default(),
            graph_params: UmapGraphParams::default(),
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
    ///   The `mlx` feature adds `"exhaustive_mlx"`, `"ivf_mlx"` and `"nndescent_mlx"`.
    /// * `initialisation` - Which embedding initialisation to use, e.g.
    ///   `"spectral"`.
    /// * `init_range` - Optional initialisation range.
    /// * `nn_params` - Nearest neighbour parameters.
    /// * `optim_params` - Parameters for the UMAP optimiser.
    /// * `graph_params` - Parameters for the UMAP graph generation.
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
        graph_params: UmapGraphParams<T>,
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
            graph_params,
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
        let min_dist = min_dist.unwrap_or(T::from_f64(DEFAULT_MIN_DIST).unwrap());
        let spread = spread.unwrap_or(T::from_f64(DEFAULT_SPREAD).unwrap());

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
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances. Needs
///   to be a tuple of `(Vec<Vec<usize>>, Vec<Vec<T>>)` with indices and
///   distances excluding self, sorted by distance. A wider graph is cut to `k`,
///   with a warning on stderr.
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
    data: impl ManifoldsMatrix<T>,
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
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    let verbosity = parse_verbosity_level(verbose);

    let (knn_indices, knn_dist) = match precomputed_knn {
        Some((indices, distances)) => {
            if verbosity.normal_verbosity() {
                println!(" Using precomputed kNN graph");
            }
            truncate_knn(indices, distances, k)
        }
        None => {
            if verbosity.normal_verbosity() {
                println!(" kNN search via {} (k = {}) ...", ann_type, k);
            }
            let start_knn = Instant::now();
            let result =
                run_ann_search(data, k, ann_type, nn_params, seed, ann_verbose(verbosity))?;
            if verbosity.normal_verbosity() {
                println!(" ... kNN search in {:.2?}", start_knn.elapsed());
            }
            result
        }
    };

    if verbosity.normal_verbosity() {
        println!(" Building fuzzy simplicial set ...");
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
        println!(" ... graph in {:.2?}", start_graph.elapsed());
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
/// 5. Optimise embedding with the chosen optimiser, parallel Adam by default
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
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
///
/// ### References
///
/// McInnes, Healy & Melville (2018): "UMAP: Uniform Manifold Approximation and
/// Projection for Dimension Reduction", arXiv
pub fn umap<T>(
    data: impl ManifoldsMatrix<T>,
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
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    umap_inner(data, precomputed_knn, umap_params, None, seed, verbose)
}

/// Shared body of [`umap`] and [`densmap`].
///
/// Split out so the density-preserving variant is a parameter rather than a
/// second copy of the pipeline. `dens_params` is the only difference between
/// the two entry points.
///
/// ### Params
///
/// * `data` - Input data matrix (samples × features)
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances
/// * `umap_params` - UMAP parameters
/// * `dens_params` - Density knobs for densMAP, or `None` for plain UMAP
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>`, `[n_dim][n_samples]`.
fn umap_inner<T>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    umap_params: &UmapParams<T>,
    dens_params: Option<DensParams<T>>,
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
    let name = if dens_params.is_some() {
        "densMAP"
    } else {
        "UMAP"
    };
    let start_run = Instant::now();
    if verbosity.normal_verbosity() {
        println!(
            "Running {} on {} samples ...",
            name,
            data.nrows().separate_with_underscores()
        );
    }

    // parse various parameters
    let init_type = parse_initialisation(
        &umap_params.initialisation,
        umap_params.randomised,
        umap_params.init_range,
    )
    .unwrap_or_else(|| {
        eprintln!(
            "Warning: unknown initialisation provided: {:?}. Defaulting to PCA.",
            umap_params.initialisation,
        );
        EmbdInit::PcaInit {
            range: None,
            randomised: true,
        }
    });
    let optimiser = parse_umap_optimiser(&umap_params.optimiser).unwrap_or_default();

    if verbosity.detailed_verbosity() {
        println!(
            " alpha = {:.2?}, beta = {:.2?}",
            umap_params.optim_params.a, umap_params.optim_params.b
        );
    }

    let (graph, knn_indices, knn_dist) = construct_umap_graph(
        data,
        precomputed_knn,
        umap_params.k,
        umap_params.ann_type.clone(),
        &umap_params.graph_params,
        &umap_params.nn_params,
        umap_params.optim_params.n_epochs,
        seed,
        verbose,
    )?;

    // density state, while the kNN results are still around. `DensState::new` takes
    // true distances and squares them itself.
    let dens_state = match dens_params {
        Some(dens_params) => {
            if verbosity.normal_verbosity() {
                println!(
                    " Computing local radii (lambda = {:.3}) ...",
                    dens_params.lambda.to_f64().unwrap_or(f64::NAN)
                );
            }
            Some(DensState::new(
                dens_params,
                &graph,
                &knn_indices,
                &knn_dist,
            )?)
        }
        None => None,
    };

    // the kNN lists can be gigabytes at scale; nothing below needs them
    drop(knn_indices);
    drop(knn_dist);

    if verbosity.normal_verbosity() {
        println!(" Initialising via {} ...", umap_params.initialisation);
    }

    let start_init = Instant::now();

    let mut embd = initialise_embedding(&init_type, umap_params.n_dim, seed as u64, &graph, data)?;

    if verbosity.normal_verbosity() {
        println!(" ... initialised in {:.2?}", start_init.elapsed());
    }

    let start_layout = Instant::now();

    let graph_adj = coo_to_adjacency_list(&graph);
    let n_edges = graph.col_indices.len();
    // the optimisers only read the adjacency list; at scale the COO is GBs
    drop(graph);

    if verbosity.normal_verbosity() {
        println!(
            " Optimising via {}, {} epochs on {} edges ...",
            match optimiser {
                UmapOptimiser::Adam => "Adam",
                UmapOptimiser::Sgd => "SGD",
                UmapOptimiser::SgdParallel => "SGD (multi-threaded, lock-free)",
                UmapOptimiser::AdamParallel => "Adam (multi-threaded)",
                #[cfg(mlx_available)]
                UmapOptimiser::AdamMlx => "Adam (MLX)",
            },
            umap_params.optim_params.n_epochs,
            n_edges.separate_with_underscores()
        );
    }

    match optimiser {
        UmapOptimiser::Adam => optimise_embedding_adam(
            &mut embd,
            &graph_adj,
            &umap_params.optim_params,
            dens_state.as_ref(),
            seed as u64,
            verbose,
        )?,
        UmapOptimiser::Sgd => {
            optimise_embedding_sgd(
                &mut embd,
                &graph_adj,
                &umap_params.optim_params,
                dens_state.as_ref(),
                seed as u64,
                verbose,
            )?;
        }
        UmapOptimiser::SgdParallel => {
            optimise_embedding_sgd_parallel(
                &mut embd,
                &graph_adj,
                &umap_params.optim_params,
                dens_state.as_ref(),
                seed as u64,
                verbose,
            )?;
        }
        UmapOptimiser::AdamParallel => {
            optimise_embedding_adam_parallel(
                &mut embd,
                &graph_adj,
                &umap_params.optim_params,
                dens_state.as_ref(),
                seed as u64,
                verbose,
            )?;
        }
        #[cfg(mlx_available)]
        UmapOptimiser::AdamMlx => {
            optimise_embedding_adam_mlx(
                &mut embd,
                &graph_adj,
                &umap_params.optim_params,
                dens_state.as_ref(),
                seed as u64,
                verbose,
            )?;
        }
    }

    let end_layout = start_layout.elapsed();

    if verbosity.normal_verbosity() {
        println!(" ... optimised in {:.2?}", end_layout);
        println!("... finished {} in {:.2?}", name, start_run.elapsed());
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
    /// Number of output dimensions. Anything but 2 returns
    /// [`ManifoldsError::IncorrectDim`].
    pub n_dim: usize,
    /// Perplexity parameter controlling neighbourhood size (typical: 5-50)
    /// Defaults to `30.0`.
    pub perplexity: T,
    /// (Approximate) Nearest neighbour method. One of `"exhaustive"`, `"ivf"`,
    /// `"hnsw"`, `"nndescent"`, `"annoy"`, `"kmknn"` or `"balltree"`.
    /// The `mlx` feature adds `"exhaustive_mlx"`, `"ivf_mlx"` and `"nndescent_mlx"`.
    /// Defaults to `"ivf"`.
    pub ann_type: String,
    /// Embedding initialisation method: `"pca"`, `"random"`, or `"spectral"`
    /// Defaults to `"pca"`.
    pub initialisation: String,
    /// Optional initialisation range
    pub init_range: Option<T>,
    /// Nearest neighbour parameters
    pub nn_params: NearestNeighbourParams<T>,
    /// tSNE optimisation parameters
    pub optim_params: TsneOptimParams<T>,
    /// Use randomised SVD for PCA initialisation
    /// Defaults to `true`.
    pub randomised: bool,
}

impl<T> Default for TsneParams<T>
where
    T: ManifoldsFloat,
{
    fn default() -> Self {
        Self {
            n_dim: 2,
            perplexity: T::from_f64(30.0).unwrap(),
            ann_type: "ivf".to_string(),
            initialisation: "pca".to_string(),
            init_range: None,
            nn_params: NearestNeighbourParams::default(),
            optim_params: TsneOptimParams::default(),
            randomised: true,
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
    ///   The `mlx` feature adds `"exhaustive_mlx"`, `"ivf_mlx"` and `"nndescent_mlx"`.
    /// * `initialisation` - Embedding initialisation method: `"pca"`,
    ///   `"random"`, or `"spectral"`.
    /// * `init_range` - Optional initialisation range to fix the initial
    ///   embedding between certain values.
    /// * `nn_params` - Nearest neighbour parameters.
    /// * `optim_params` - t-SNE optimisation parameters. These cover the
    ///   learning rate, number of epochs, early/late exaggeration, the
    ///   Barnes-Hut `theta`, and the number of FFT interpolation points.
    /// * `randomised` - Whether randomised SVD is used for PCA
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
        randomised: bool,
    ) -> Self {
        Self {
            n_dim,
            perplexity,
            ann_type,
            initialisation,
            init_range,
            nn_params,
            optim_params,
            randomised,
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
        let default = Self::default();

        Self {
            perplexity: perplexity.unwrap_or(default.perplexity),
            ..default
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
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
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
/// ### Notes
///
/// The k value is automatically set to `3 × perplexity`, clamped between 5 and
/// n-1. This is standard practice in t-SNE implementations.
pub fn construct_tsne_graph<T>(
    data: impl ManifoldsMatrix<T>,
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
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    let verbosity = parse_verbosity_level(verbose);

    let (knn_indices, knn_dist) = match precomputed_knn {
        Some((indices, distances)) => {
            if verbosity.normal_verbosity() {
                println!(" Using precomputed kNN graph");
            }
            (indices, distances)
        }
        None => {
            let k_float = perplexity * T::from_f64(3.0).unwrap();
            let k = k_float.to_usize().unwrap().max(5).min(data.nrows() - 1);

            if verbosity.normal_verbosity() {
                println!(" kNN search via {} (k = {}) ...", ann_type, k);
            }

            let start_knn = Instant::now();
            let result =
                run_ann_search(data, k, ann_type, nn_params, seed, ann_verbose(verbosity))?;

            if verbosity.normal_verbosity() {
                println!(" ... kNN search in {:.2?}", start_knn.elapsed());
            }

            result
        }
    };

    if verbosity.normal_verbosity() {
        println!(" Building Gaussian affinities ...");
    }

    let start_graph = Instant::now();

    let directed_graph = gaussian_knn_affinities(
        &knn_indices,
        &knn_dist,
        perplexity,
        T::from_f64(1e-5).unwrap(),
        200,
    )?;

    let graph = symmetrise_affinities_tsne(directed_graph);

    if verbosity.normal_verbosity() {
        println!(" ... graph in {:.2?}", start_graph.elapsed());
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
///    - Early exaggeration for `optim_params.early_exag_iter` iterations
///    - Momentum switching at iteration 250
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances. Needs
///   to be a tuple of `(Vec<Vec<usize>>, Vec<Vec<T>>)` with indices and
///   distances excluding self.
/// * `params` - t-SNE parameters controlling algorithm behaviour
/// * `approx_type` - Type of approximation to use for repulsive forces.
///   Options: `"barnes_hut" | "bh"`, `"bh_qd" | "qd"`, `"fft"`,
///   `"fft_3k" | "3-kernel"`, and with the `mlx` feature `"fft_3k_mlx"`
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
/// ### Errors
///
/// [`ManifoldsError::IncorrectDim`] if `n_dim` is not 2 and
/// [`ManifoldsError::TsneOptimiserNeedsGpu`] for `"fft_3k_gpu"`, which only
/// [`tsne_gpu`] can run. Without the `fft_tsne` feature, `"fft"` and `"fft_3k"`
/// return [`ManifoldsError::TsneOptimiserNeedsFft`].
///
/// ### References
///
/// - van der Maaten & Hinton (2008): "Visualizing Data using t-SNE"
/// - van der Maaten (2014): "Accelerating t-SNE using Tree-Based Algorithms"
/// - Linderman et al. (2019): "Fast interpolation-based t-SNE for improved
///   visualization of single-cell RNA-seq data"
#[cfg(feature = "fft_tsne")]
pub fn tsne<T>(
    data: impl ManifoldsMatrix<T>,
    precomputed_knn: PreComputedKnn<T>,
    params: &TsneParams<T>,
    approx_type: &str,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat + FftFloat,
    HnswIndex<T>: HnswState<T>,
    StandardNormal: Distribution<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    tsne_inner(
        data,
        precomputed_knn,
        params,
        approx_type,
        None,
        None,
        None,
        seed,
        verbose,
    )
}

/// Shared body of [`tsne`] and [`densne`].
///
/// Split out so the density-preserving variant is a parameter rather than a
/// second copy of the pipeline. `dens_params` is the only difference between
/// the two entry points.
///
/// ### Params
///
/// * `data` - Input data matrix (samples × features)
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances
/// * `params` - t-SNE parameters
/// * `approx_type` - `"barnes_hut" | "bh"`, `"fft"` or `"fft_3k" | "3-kernel"`
/// * `dens_params` - Density knobs for den-SNE, or `None` for plain t-SNE
/// * `dreams_lambda` - DREAMS regularisation strength, or `None` for no
///   regulariser. When set, the embedding is initialised from the reference
///   and `params.initialisation` is ignored
/// * `dreams_reference` - DREAMS reference embedding `[2][n_samples]`;
///   `None` uses PCA. Ignored without `dreams_lambda`
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>`, `[n_dim][n_samples]`.
#[cfg(feature = "fft_tsne")]
#[allow(clippy::too_many_arguments)]
fn tsne_inner<T>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    params: &TsneParams<T>,
    approx_type: &str,
    dens_params: Option<DensParams<T>>,
    dreams_lambda: Option<T>,
    dreams_reference: Option<Vec<Vec<T>>>,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat + FftFloat,
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
    let name = match (dens_params.is_some(), dreams_lambda.is_some()) {
        (true, _) => "densNE",
        (_, true) => "DREAMS",
        _ => "t-SNE",
    };
    let start_run = Instant::now();
    if verbosity.normal_verbosity() {
        println!(
            "Running {} on {} samples ...",
            name,
            data.nrows().separate_with_underscores()
        );
    }

    // 1. graph construction
    let (graph, knn_indices, knn_dist) = construct_tsne_graph(
        data,
        precomputed_knn,
        params.perplexity,
        params.ann_type.clone(),
        &params.nn_params,
        seed,
        verbose,
    )?;

    // 1b. density state, while the kNN results are still around. `DensState::new` takes
    // true distances and squares them itself.
    let dens_state = match dens_params {
        Some(dens_params) => {
            if verbosity.normal_verbosity() {
                println!(
                    " Computing local radii (lambda = {:.3}) ...",
                    dens_params.lambda.to_f64().unwrap_or(f64::NAN)
                );
            }
            Some(DensState::new(
                dens_params,
                &graph,
                &knn_indices,
                &knn_dist,
            )?)
        }
        None => None,
    };

    // the kNN lists can be gigabytes at scale; nothing below needs them
    drop(knn_indices);
    drop(knn_dist);

    if verbosity.normal_verbosity() {
        println!(" Initialising via {} ...", params.initialisation);
    }

    // 2. initialise embedding
    let init_type =
        parse_initialisation(&params.initialisation, params.randomised, params.init_range)
            .unwrap_or_else(|| {
                eprintln!(
                    "Warning: unknown initialisation provided: {:?}. Defaulting to PCA.",
                    params.initialisation,
                );
                EmbdInit::PcaInit {
                    range: Some(T::from_f64(1e-2).unwrap()),
                    randomised: true,
                }
            });

    let start_init = Instant::now();

    // DREAMS initialises from its own reference, so the regulariser and the
    // starting layout agree.
    let (mut embd, dreams_state) = match dreams_lambda {
        Some(lambda) => {
            let (embd, state) = dreams_setup(
                data,
                dreams_reference,
                lambda,
                params.randomised,
                seed as u64,
            )?;
            (embd, Some(state))
        }
        None => (
            initialise_embedding(&init_type, params.n_dim, seed as u64, &graph, data)?,
            None,
        ),
    };

    if verbosity.normal_verbosity() {
        println!(" ... initialised in {:.2?}", start_init.elapsed());
    }

    // parse the optimisation type
    let tsne_approx = parse_tsne_optimiser(approx_type).unwrap_or_else(|| {
        eprintln!(
            "Warning: unrecognised tSNE approximation provided: {:?}. Default to Barnes-Hut.",
            approx_type
        );
        TsneOpt::default()
    });

    // 3. optimise
    let start_optim = Instant::now();
    match tsne_approx {
        TsneOpt::BarnesHut => {
            if verbosity.normal_verbosity() {
                println!(
                    " Optimising via Barnes-Hut t-SNE ({} epochs) ...",
                    params.optim_params.n_epochs
                );
            }
            optimise_bh_tsne(
                &mut embd,
                &params.optim_params,
                &graph,
                dens_state.as_ref(),
                dreams_state.as_ref(),
                verbose,
            );
        }
        TsneOpt::BarnesHutQd => {
            if verbosity.normal_verbosity() {
                println!(
                    " Optimising via quick-and-dirty Barnes-Hut t-SNE ({} epochs, max depth {}) ...",
                    params.optim_params.n_epochs, params.optim_params.max_depth
                );
            }
            optimise_qd_tsne(
                &mut embd,
                &params.optim_params,
                &graph,
                dens_state.as_ref(),
                dreams_state.as_ref(),
                verbose,
            );
        }
        #[cfg(feature = "fft_tsne")]
        TsneOpt::Fft => {
            if verbosity.normal_verbosity() {
                println!(
                    " Optimising via FFT Interpolation-based t-SNE ({} epochs) ...",
                    params.optim_params.n_epochs
                );
            }
            optimise_fft_tsne(
                &mut embd,
                &params.optim_params,
                &graph,
                dens_state.as_ref(),
                dreams_state.as_ref(),
                verbose,
            )?;
        }
        #[cfg(feature = "fft_tsne")]
        TsneOpt::Fft3Kernel => {
            if verbosity.normal_verbosity() {
                println!(
                    " Optimising via three-kernel FFT Interpolation-based t-SNE ({} epochs) ...",
                    params.optim_params.n_epochs
                );
            }
            optimise_fft3k_tsne(
                &mut embd,
                &params.optim_params,
                &graph,
                dens_state.as_ref(),
                dreams_state.as_ref(),
                verbose,
            )?;
        }
        TsneOpt::Fft3KernelGpu => {
            return Err(ManifoldsError::TsneOptimiserNeedsGpu {
                name: approx_type.to_string(),
            });
        }
        #[cfg(mlx_available)]
        TsneOpt::Fft3KernelMlx => {
            if dens_state.is_some() {
                return Err(ManifoldsError::DensityNotSupported {
                    name: approx_type.to_string(),
                });
            }
            if verbosity.normal_verbosity() {
                println!(
                    " Optimising via MLX three-kernel FFT t-SNE ({} epochs) ...",
                    params.optim_params.n_epochs
                );
            }
            optimise_fft3k_tsne_mlx(
                &mut embd,
                &params.optim_params,
                &graph,
                dreams_state.as_ref(),
                verbose,
            )?;
        }
    }

    if verbosity.normal_verbosity() {
        println!(" ... optimised in {:.2?}", start_optim.elapsed());
        println!("... finished {} in {:.2?}", name, start_run.elapsed());
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
///    - Early exaggeration for `optim_params.early_exag_iter` iterations
///    - Momentum switching at iteration 250
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances. Needs
///   to be a tuple of `(Vec<Vec<usize>>, Vec<Vec<T>>)` with indices and
///   distances excluding self.
/// * `params` - t-SNE parameters controlling algorithm behaviour
/// * `approx_type` - Type of approximation to use for repulsive forces.
///   Options: `"barnes_hut" | "bh"`, `"bh_qd" | "qd"`, and with the `mlx`
///   feature `"fft_3k_mlx"`. `"fft"` and `"fft_3k"` need `fft_tsne`.
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
/// ### Errors
///
/// [`ManifoldsError::IncorrectDim`] if `n_dim` is not 2 and
/// [`ManifoldsError::TsneOptimiserNeedsGpu`] for `"fft_3k_gpu"`, which only
/// [`tsne_gpu`] can run. Without the `fft_tsne` feature, `"fft"` and `"fft_3k"`
/// return [`ManifoldsError::TsneOptimiserNeedsFft`].
///
/// ### References
///
/// - van der Maaten & Hinton (2008): "Visualizing Data using t-SNE"
/// - van der Maaten (2014): "Accelerating t-SNE using Tree-Based Algorithms"
/// - Linderman et al. (2019): "Fast interpolation-based t-SNE for improved
///   visualization of single-cell RNA-seq data"
#[cfg(not(feature = "fft_tsne"))]
pub fn tsne<T>(
    data: impl ManifoldsMatrix<T>,
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
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    tsne_inner(
        data,
        precomputed_knn,
        params,
        approx_type,
        None,
        None,
        None,
        seed,
        verbose,
    )
}

/// Shared body of [`tsne`] and [`densne`].
///
/// Split out so the density-preserving variant is a parameter rather than a
/// second copy of the pipeline. `dens_params` is the only difference between
/// the two entry points.
///
/// ### Params
///
/// * `data` - Input data matrix (samples × features)
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances
/// * `params` - t-SNE parameters
/// * `approx_type` - `"barnes_hut" | "bh"`; `"fft"` and `"fft_3k"` return
///   [`ManifoldsError::TsneOptimiserNeedsFft`] without the `fft_tsne` feature
/// * `dens_params` - Density knobs for den-SNE, or `None` for plain t-SNE
/// * `dreams_lambda` - DREAMS regularisation strength, or `None` for no
///   regulariser. When set, the embedding is initialised from the reference
///   and `params.initialisation` is ignored
/// * `dreams_reference` - DREAMS reference embedding `[2][n_samples]`;
///   `None` uses PCA. Ignored without `dreams_lambda`
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>`, `[n_dim][n_samples]`.
#[cfg(not(feature = "fft_tsne"))]
#[allow(clippy::too_many_arguments)]
fn tsne_inner<T>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    params: &TsneParams<T>,
    approx_type: &str,
    dens_params: Option<DensParams<T>>,
    dreams_lambda: Option<T>,
    dreams_reference: Option<Vec<Vec<T>>>,
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
    let name = match (dens_params.is_some(), dreams_lambda.is_some()) {
        (true, _) => "densNE",
        (_, true) => "DREAMS",
        _ => "t-SNE",
    };
    let start_run = Instant::now();
    if verbosity.normal_verbosity() {
        println!(
            "Running {} on {} samples ...",
            name,
            data.nrows().separate_with_underscores()
        );
    }

    // 1. graph construction
    let (graph, knn_indices, knn_dist) = construct_tsne_graph(
        data,
        precomputed_knn,
        params.perplexity,
        params.ann_type.clone(),
        &params.nn_params,
        seed,
        verbose,
    )?;

    // 1b. density state, while the kNN results are still around. `DensState::new` takes
    // true distances and squares them itself.
    let dens_state = match dens_params {
        Some(dens_params) => {
            if verbosity.normal_verbosity() {
                println!(
                    " Computing local radii (lambda = {:.3}) ...",
                    dens_params.lambda.to_f64().unwrap_or(f64::NAN)
                );
            }
            Some(DensState::new(
                dens_params,
                &graph,
                &knn_indices,
                &knn_dist,
            )?)
        }
        None => None,
    };

    // the kNN lists can be gigabytes at scale; nothing below needs them
    drop(knn_indices);
    drop(knn_dist);

    // 2. initialise embedding
    if verbosity.normal_verbosity() {
        println!(" Initialising via {} ...", &params.initialisation);
    }

    let init_type =
        parse_initialisation(&params.initialisation, params.randomised, params.init_range)
            .unwrap_or_else(|| {
                eprintln!(
                    "Warning: unknown initialisation provided: {:?}. Defaulting to PCA.",
                    params.initialisation,
                );
                EmbdInit::PcaInit {
                    range: Some(T::from_f64(1e-2).unwrap()),
                    randomised: true,
                }
            });

    let start_init = Instant::now();

    // DREAMS initialises from its own reference, so the regulariser and the
    // starting layout agree.
    let (mut embd, dreams_state) = match dreams_lambda {
        Some(lambda) => {
            let (embd, state) = dreams_setup(
                data,
                dreams_reference,
                lambda,
                params.randomised,
                seed as u64,
            )?;
            (embd, Some(state))
        }
        None => (
            initialise_embedding(&init_type, params.n_dim, seed as u64, &graph, data)?,
            None,
        ),
    };

    if verbosity.normal_verbosity() {
        println!(" ... initialised in {:.2?}", start_init.elapsed());
    }

    // parse the optimisation type
    let tsne_approx = parse_tsne_optimiser(approx_type).unwrap_or_else(|| {
        eprintln!(
            "Warning: unrecognised tSNE approximation provided: {:?}. Default to Barnes-Hut.",
            approx_type
        );
        TsneOpt::default()
    });

    // 3. optimise
    let start_optim = Instant::now();
    match tsne_approx {
        TsneOpt::BarnesHut => {
            if verbosity.normal_verbosity() {
                println!(
                    " Optimising via Barnes-Hut t-SNE ({} epochs) ...",
                    params.optim_params.n_epochs
                );
            }
            optimise_bh_tsne(
                &mut embd,
                &params.optim_params,
                &graph,
                dens_state.as_ref(),
                dreams_state.as_ref(),
                verbose,
            );
        }
        TsneOpt::BarnesHutQd => {
            if verbosity.normal_verbosity() {
                println!(
                    " Optimising via quick-and-dirty Barnes-Hut t-SNE ({} epochs, max depth {}) ...",
                    params.optim_params.n_epochs, params.optim_params.max_depth
                );
            }
            optimise_qd_tsne(
                &mut embd,
                &params.optim_params,
                &graph,
                dens_state.as_ref(),
                dreams_state.as_ref(),
                verbose,
            );
        }
        #[cfg(not(feature = "fft_tsne"))]
        TsneOpt::Fft | TsneOpt::Fft3Kernel => {
            return Err(ManifoldsError::TsneOptimiserNeedsFft {
                name: approx_type.to_string(),
            });
        }
        TsneOpt::Fft3KernelGpu => {
            return Err(ManifoldsError::TsneOptimiserNeedsGpu {
                name: approx_type.to_string(),
            });
        }
        #[cfg(mlx_available)]
        TsneOpt::Fft3KernelMlx => {
            if dens_state.is_some() {
                return Err(ManifoldsError::DensityNotSupported {
                    name: approx_type.to_string(),
                });
            }
            if verbosity.normal_verbosity() {
                println!(
                    " Optimising via MLX three-kernel FFT t-SNE ({} epochs) ...",
                    params.optim_params.n_epochs
                );
            }
            optimise_fft3k_tsne_mlx(
                &mut embd,
                &params.optim_params,
                &graph,
                dreams_state.as_ref(),
                verbose,
            )?;
        }
    }

    if verbosity.normal_verbosity() {
        println!(" ... optimised in {:.2?}", start_optim.elapsed());
        println!("... finished {} in {:.2?}", name, start_run.elapsed());
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

/////////////
// densMAP //
/////////////

////////////
// Params //
////////////

/// Parameters for densMAP, the density-preserving variant of UMAP.
#[derive(Debug, Clone)]
pub struct DensmapParams<T> {
    /// The underlying UMAP parameters. Unchanged from plain UMAP.
    pub umap_params: UmapParams<T>,
    /// Density-preservation knobs. Defaults to [`DENSMAP_LAMBDA`].
    pub dens_params: DensParams<T>,
}

impl<T> Default for DensmapParams<T>
where
    T: ManifoldsFloat,
{
    fn default() -> Self {
        Self {
            umap_params: UmapParams::default(),
            dens_params: DensParams::densmap_default(),
        }
    }
}

impl<T> DensmapParams<T>
where
    T: ManifoldsFloat,
{
    /// Full-control constructor.
    ///
    /// ### Params
    ///
    /// * `umap_params` - The underlying UMAP parameters
    /// * `dens_params` - Density-preservation knobs
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new(umap_params: UmapParams<T>, dens_params: DensParams<T>) -> Self {
        Self {
            umap_params,
            dens_params,
        }
    }

    /// 2D defaults with the knobs most worth tuning exposed.
    ///
    /// ### Params
    ///
    /// * `min_dist` - Minimum distance between embedded points. Defaults to the
    ///   UMAP default
    /// * `spread` - Effective scale of embedded points. Defaults to the UMAP
    ///   default
    /// * `lambda` - Density weight. Defaults to [`DENSMAP_LAMBDA`]. Larger
    ///   values preserve density more aggressively at the cost of cluster
    ///   separation
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new_default_2d(min_dist: Option<T>, spread: Option<T>, lambda: Option<T>) -> Self {
        Self {
            umap_params: UmapParams::new_default_2d(min_dist, spread),
            dens_params: DensParams::new(
                Some(lambda.unwrap_or_else(|| T::from_f64(DENSMAP_LAMBDA).unwrap())),
                None,
                None,
            ),
        }
    }
}

//////////
// Main //
//////////

/// Run densMAP, the density-preserving variant of UMAP
///
/// Identical to [`umap`] except for an extra term in the gradient that
/// maximises the Pearson correlation between the log local radius of each point
/// in the original space and in the embedding. Plain UMAP is free to render a
/// tight cluster and a diffuse one at the same size; densMAP is not.
///
/// ### Algorithm
///
/// 1. Build the fuzzy simplicial set exactly as UMAP does
/// 2. Compute the constant original log local radii
///    `log(sum_j mu_ij d_ij^2 / sum_j mu_ij)` and z-score them
/// 3. Optimise as UMAP, but over the final `dens_params.frac` of the epochs
///    also recompute the embedding log radii each epoch (weighted by the UMAP
///    kernel `1 / (1 + a d^2b)`) and add `-lambda * dCorr/dy` to the gradient
///
/// The density term touches only the attractive part of the gradient; negative
/// sampling is untouched. Works with all four CPU optimisers (`"sgd"`,
/// `"sgd_parallel"`, `"adam"`, `"adam_parallel"`).
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances. Needs
///   to be a tuple of `(Vec<Vec<usize>>, Vec<Vec<T>>)` with indices and
///   distances excluding self.
/// * `params` - densMAP parameters
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>` where the outer vector has length
/// `n_dim` and the inner vectors length `n_samples`.
///
/// ### Errors
///
/// [`ManifoldsError::DegenerateLocalRadii`] if every point has the same local
/// radius, which leaves the correlation undefined. Otherwise the same errors as
/// [`umap`].
///
/// ### References
///
/// Narayan, Berger & Cho (2021): "Assessing single-cell transcriptomic
/// variability through density-preserving data visualization", Nature
/// Biotechnology
pub fn densmap<T>(
    data: impl ManifoldsMatrix<T>,
    precomputed_knn: PreComputedKnn<T>,
    params: &DensmapParams<T>,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
    HnswIndex<T>: HnswState<T>,
    StandardNormal: Distribution<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    umap_inner(
        data,
        precomputed_knn,
        &params.umap_params,
        Some(params.dens_params),
        seed,
        verbose,
    )
}

/////////////
// den-SNE //
/////////////

////////////
// Params //
////////////

/// Parameters for den-SNE, the density-preserving variant of t-SNE.
#[derive(Debug, Clone)]
pub struct DensneParams<T> {
    /// The underlying t-SNE parameters. Unchanged from plain t-SNE.
    pub tsne_params: TsneParams<T>,
    /// Density-preservation knobs. Defaults to [`DENSNE_LAMBDA`].
    pub dens_params: DensParams<T>,
}

impl<T> Default for DensneParams<T>
where
    T: ManifoldsFloat,
{
    fn default() -> Self {
        Self {
            tsne_params: TsneParams::default(),
            dens_params: DensParams::densne_default(),
        }
    }
}

impl<T> DensneParams<T>
where
    T: ManifoldsFloat,
{
    /// Full-control constructor.
    ///
    /// ### Params
    ///
    /// * `tsne_params` - The underlying t-SNE parameters
    /// * `dens_params` - Density-preservation knobs
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new(tsne_params: TsneParams<T>, dens_params: DensParams<T>) -> Self {
        Self {
            tsne_params,
            dens_params,
        }
    }

    /// 2D defaults with the two knobs most worth tuning exposed.
    ///
    /// ### Params
    ///
    /// * `perplexity` - Target perplexity. Defaults to the t-SNE default of 30
    /// * `lambda` - Density weight. Defaults to [`DENSNE_LAMBDA`]. Larger values
    ///   preserve density more aggressively at the cost of cluster separation
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new_default_2d(perplexity: Option<T>, lambda: Option<T>) -> Self {
        Self {
            tsne_params: TsneParams::new_default_2d(perplexity),
            dens_params: DensParams::new(
                Some(lambda.unwrap_or_else(|| T::from_f64(DENSNE_LAMBDA).unwrap())),
                None,
                None,
            ),
        }
    }
}

//////////
// Main //
//////////

/// Run den-SNE, the density-preserving variant of t-SNE
///
/// Identical to [`tsne`] except for an extra term in the gradient that
/// maximises the Pearson correlation between the log local radius of each point
/// in the original space and in the embedding. Plain t-SNE is free to render a
/// tight cluster and a diffuse one at the same size; den-SNE is not.
///
/// ### Algorithm
///
/// 1. Construct the high-dimensional affinity graph exactly as t-SNE does
/// 2. Compute the constant original log local radii
///    `log(sum_j P_ij d_ij^2 / sum_j P_ij)` and z-score them
/// 3. Optimise as t-SNE, but over the final `dens_params.frac` of the epochs
///    also recompute the embedding log radii each epoch (weighted by the
///    Student-t kernel, not by `P`) and add `-lambda * dCorr/dy` to the
///    gradient
///
/// The density term touches only the attractive part of the gradient; the
/// Barnes-Hut and FFT repulsion are untouched.
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances. Needs
///   to be a tuple of `(Vec<Vec<usize>>, Vec<Vec<T>>)` with indices and
///   distances excluding self.
/// * `params` - den-SNE parameters
/// * `approx_type` - Type of approximation to use for repulsive forces.
///   Options: `"barnes_hut" | "bh"`, `"bh_qd" | "qd"`, `"fft"`,
///   `"fft_3k" | "3-kernel"`
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>` where the outer vector has length
/// `n_dim` and the inner vectors length `n_samples`.
///
/// ### Errors
///
/// [`ManifoldsError::DegenerateLocalRadii`] if every point has the same local
/// radius, which leaves the correlation undefined. Otherwise the same errors as
/// [`tsne`].
///
/// ### References
///
/// Narayan, Berger & Cho (2021): "Assessing single-cell transcriptomic
/// variability through density-preserving data visualization", Nature
/// Biotechnology
#[cfg(feature = "fft_tsne")]
pub fn densne<T>(
    data: impl ManifoldsMatrix<T>,
    precomputed_knn: PreComputedKnn<T>,
    params: &DensneParams<T>,
    approx_type: &str,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat + FftFloat,
    HnswIndex<T>: HnswState<T>,
    StandardNormal: Distribution<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    tsne_inner(
        data,
        precomputed_knn,
        &params.tsne_params,
        approx_type,
        Some(params.dens_params),
        None,
        None,
        seed,
        verbose,
    )
}

/// Run den-SNE, the density-preserving variant of t-SNE
///
/// Identical to [`tsne`] except for an extra term in the gradient that
/// maximises the Pearson correlation between the log local radius of each point
/// in the original space and in the embedding. Plain t-SNE is free to render a
/// tight cluster and a diffuse one at the same size; den-SNE is not.
///
/// ### Algorithm
///
/// 1. Construct the high-dimensional affinity graph exactly as t-SNE does
/// 2. Compute the constant original log local radii
///    `log(sum_j P_ij d_ij^2 / sum_j P_ij)` and z-score them
/// 3. Optimise as t-SNE, but over the final `dens_params.frac` of the epochs
///    also recompute the embedding log radii each epoch (weighted by the
///    Student-t kernel, not by `P`) and add `-lambda * dCorr/dy` to the
///    gradient
///
/// The density term touches only the attractive part of the gradient; the
/// Barnes-Hut repulsion is untouched.
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances. Needs
///   to be a tuple of `(Vec<Vec<usize>>, Vec<Vec<T>>)` with indices and
///   distances excluding self.
/// * `params` - den-SNE parameters
/// * `approx_type` - Type of approximation to use for repulsive forces.
///   Options: `"barnes_hut" | "bh"`, `"bh_qd" | "qd"`. `"fft"` and `"fft_3k"`
///   require the `fft_tsne` feature
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>` where the outer vector has length
/// `n_dim` and the inner vectors length `n_samples`.
///
/// ### Errors
///
/// [`ManifoldsError::DegenerateLocalRadii`] if every point has the same local
/// radius, which leaves the correlation undefined. Otherwise the same errors as
/// [`tsne`].
///
/// ### References
///
/// Narayan, Berger & Cho (2021): "Assessing single-cell transcriptomic
/// variability through density-preserving data visualization", Nature
/// Biotechnology
#[cfg(not(feature = "fft_tsne"))]
pub fn densne<T>(
    data: impl ManifoldsMatrix<T>,
    precomputed_knn: PreComputedKnn<T>,
    params: &DensneParams<T>,
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
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    tsne_inner(
        data,
        precomputed_knn,
        &params.tsne_params,
        approx_type,
        Some(params.dens_params),
        None,
        None,
        seed,
        verbose,
    )
}

////////////
// DREAMS //
////////////

/// Parameters for DREAMS, t-SNE regularised towards a global reference
/// embedding.
#[derive(Debug, Clone)]
pub struct DreamsParams<T> {
    /// The underlying t-SNE parameters. `initialisation` is ignored: DREAMS
    /// always starts from its reference embedding.
    pub tsne_params: TsneParams<T>,
    /// Regularisation strength in `[0, 1]`. `0` is plain t-SNE, `1` the
    /// reference embedding. Defaults to [`DREAMS_LAMBDA`].
    pub lambda: T,
}

impl<T> Default for DreamsParams<T>
where
    T: ManifoldsFloat,
{
    fn default() -> Self {
        Self {
            tsne_params: TsneParams::default(),
            lambda: T::from_f64(DREAMS_LAMBDA).unwrap(),
        }
    }
}

impl<T> DreamsParams<T>
where
    T: ManifoldsFloat,
{
    /// Full-control constructor.
    ///
    /// ### Params
    ///
    /// * `tsne_params` - The underlying t-SNE parameters
    /// * `lambda` - Regularisation strength in `[0, 1]`
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new(tsne_params: TsneParams<T>, lambda: T) -> Self {
        Self {
            tsne_params,
            lambda,
        }
    }

    /// 2D defaults with the two knobs most worth tuning exposed.
    ///
    /// ### Params
    ///
    /// * `perplexity` - Target perplexity. Defaults to the t-SNE default of 30
    /// * `lambda` - Regularisation strength. Defaults to [`DREAMS_LAMBDA`].
    ///   Larger values pull harder towards the reference layout
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new_default_2d(perplexity: Option<T>, lambda: Option<T>) -> Self {
        Self {
            tsne_params: TsneParams::new_default_2d(perplexity),
            lambda: lambda.unwrap_or_else(|| T::from_f64(DREAMS_LAMBDA).unwrap()),
        }
    }
}

/// Run DREAMS, t-SNE regularised towards a global reference embedding
///
/// Adds a quadratic pull towards a scaled reference embedding (PCA unless one
/// is supplied) to every t-SNE epoch. `lambda = 0` is plain t-SNE, `lambda = 1`
/// recovers the reference; the default 0.15 keeps t-SNE's local structure and
/// most of PCA's global layout.
///
/// ### Algorithm
///
/// 1. Construct the high-dimensional affinity graph exactly as t-SNE does
/// 2. Take the reference `Y_e`: the first two PC scores (component ratio
///    preserved) or the supplied embedding, centred
/// 3. Initialise from `Y_e`, scaled so its first dimension has std `1e-4`
/// 4. Optimise as t-SNE, but every epoch scale the t-SNE update by
///    `1 - lambda` and add `-lr * lambda^2 * (2/n) * (y - alpha * y_e)`, with
///    `alpha = ||Y||_F / ||Y_e||_F` and `lr = n / exaggeration` (openTSNE's
///    automatic rate, independent of the t-SNE learning rate). This is the
///    reference implementation's update, which differs from Equation 2 of the
///    paper; see [`crate::utils::dreams`]
///
/// `params.tsne_params.initialisation` is ignored. The regulariser is O(n) per
/// epoch on top of the Barnes-Hut or FFT repulsion.
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances. Needs
///   to be a tuple of `(Vec<Vec<usize>>, Vec<Vec<T>>)` with indices and
///   distances excluding self.
/// * `reference` - Optional global reference embedding as `[2][n_samples]`,
///   the layout every embedding in this crate returns (e.g. an MDS or PHATE
///   result). `None` uses PCA.
/// * `params` - DREAMS parameters
/// * `approx_type` - Type of approximation to use for repulsive forces.
///   Options: `"barnes_hut" | "bh"`, `"bh_qd" | "qd"`, with the `fft_tsne`
///   feature `"fft"`, `"fft_3k" | "3-kernel"`, and with the `mlx` feature
///   `"fft_3k_mlx"`
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>` where the outer vector has length
/// `n_dim` and the inner vectors length `n_samples`.
///
/// ### Errors
///
/// [`ManifoldsError::DreamsInvalidLambda`] if `lambda` is outside `[0, 1]`,
/// [`ManifoldsError::DreamsReferenceMismatch`] if the reference is not
/// `2 x n_samples`, [`ManifoldsError::DreamsDegenerateReference`] if it
/// collapses to a point. Otherwise the same errors as [`tsne`].
///
/// ### References
///
/// Kury, Kobak & Damrich (2026): "DREAMS: Preserving both Local and Global
/// Structure in Dimensionality Reduction", Transactions on Machine Learning
/// Research
#[cfg(feature = "fft_tsne")]
pub fn dreams<T>(
    data: impl ManifoldsMatrix<T>,
    precomputed_knn: PreComputedKnn<T>,
    reference: Option<Vec<Vec<T>>>,
    params: &DreamsParams<T>,
    approx_type: &str,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat + FftFloat,
    HnswIndex<T>: HnswState<T>,
    StandardNormal: Distribution<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    tsne_inner(
        data,
        precomputed_knn,
        &params.tsne_params,
        approx_type,
        None,
        Some(params.lambda),
        reference,
        seed,
        verbose,
    )
}

/// Run DREAMS, t-SNE regularised towards a global reference embedding
///
/// Adds a quadratic pull towards a scaled reference embedding (PCA unless one
/// is supplied) to every t-SNE epoch. `lambda = 0` is plain t-SNE, `lambda = 1`
/// recovers the reference; the default 0.15 keeps t-SNE's local structure and
/// most of PCA's global layout.
///
/// ### Algorithm
///
/// 1. Construct the high-dimensional affinity graph exactly as t-SNE does
/// 2. Take the reference `Y_e`: the first two PC scores (component ratio
///    preserved) or the supplied embedding, centred
/// 3. Initialise from `Y_e`, scaled so its first dimension has std `1e-4`
/// 4. Optimise as t-SNE, but every epoch scale the t-SNE update by
///    `1 - lambda` and add `-lr * lambda^2 * (2/n) * (y - alpha * y_e)`, with
///    `alpha = ||Y||_F / ||Y_e||_F` and `lr = n / exaggeration` (openTSNE's
///    automatic rate, independent of the t-SNE learning rate). This is the
///    reference implementation's update, which differs from Equation 2 of the
///    paper; see [`crate::utils::dreams`]
///
/// `params.tsne_params.initialisation` is ignored. The regulariser is O(n) per
/// epoch on top of the Barnes-Hut or FFT repulsion.
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances. Needs
///   to be a tuple of `(Vec<Vec<usize>>, Vec<Vec<T>>)` with indices and
///   distances excluding self.
/// * `reference` - Optional global reference embedding as `[2][n_samples]`,
///   the layout every embedding in this crate returns (e.g. an MDS or PHATE
///   result). `None` uses PCA.
/// * `params` - DREAMS parameters
/// * `approx_type` - Type of approximation to use for repulsive forces.
///   Options: `"barnes_hut" | "bh"`, `"bh_qd" | "qd"`, with the `fft_tsne`
///   feature `"fft"`, `"fft_3k" | "3-kernel"`, and with the `mlx` feature
///   `"fft_3k_mlx"`
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>` where the outer vector has length
/// `n_dim` and the inner vectors length `n_samples`.
///
/// ### Errors
///
/// [`ManifoldsError::DreamsInvalidLambda`] if `lambda` is outside `[0, 1]`,
/// [`ManifoldsError::DreamsReferenceMismatch`] if the reference is not
/// `2 x n_samples`, [`ManifoldsError::DreamsDegenerateReference`] if it
/// collapses to a point. Otherwise the same errors as [`tsne`].
///
/// ### References
///
/// Kury, Kobak & Damrich (2026): "DREAMS: Preserving both Local and Global
/// Structure in Dimensionality Reduction", Transactions on Machine Learning
/// Research
#[cfg(not(feature = "fft_tsne"))]
pub fn dreams<T>(
    data: impl ManifoldsMatrix<T>,
    precomputed_knn: PreComputedKnn<T>,
    reference: Option<Vec<Vec<T>>>,
    params: &DreamsParams<T>,
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
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    tsne_inner(
        data,
        precomputed_knn,
        &params.tsne_params,
        approx_type,
        None,
        Some(params.lambda),
        reference,
        seed,
        verbose,
    )
}

/////////////////
// ForceAtlas2 //
/////////////////

/// Main configuration for ForceAtlas2 on a kNN graph
#[derive(Debug, Clone)]
pub struct Fa2Params<T> {
    /// Number of neighbours
    /// Defaults to `15`.
    pub k: usize,
    /// (Approximate) Nearest neighbour method. One of `"exhaustive"`, `"ivf"`,
    /// `"hnsw"`, `"nndescent"`, `"annoy"`, `"kmknn"` or `"balltree"`.
    /// The `mlx` feature adds `"exhaustive_mlx"`, `"ivf_mlx"` and `"nndescent_mlx"`.
    /// Defaults to `"nndescent"`.
    pub ann_type: String,
    /// Embedding initialisation method: `"spectral"`, `"pca"` or `"random"`
    /// Defaults to `"spectral"`.
    pub initialisation: String,
    /// Optional initialisation range
    pub init_range: Option<T>,
    /// Use randomised SVD for PCA initialisation
    /// Defaults to `false`.
    pub randomised: bool,
    /// Nearest neighbour parameters
    pub nn_params: NearestNeighbourParams<T>,
    /// Parameters for the fuzzy simplicial set the layout runs on.
    /// `mix_weight` must stay at 1; lower values leave the graph directed.
    pub graph_params: UmapGraphParams<T>,
    /// ForceAtlas2 optimisation parameters
    pub optim_params: Fa2OptimParams<T>,
}

impl<T> Default for Fa2Params<T>
where
    T: ManifoldsFloat,
{
    fn default() -> Self {
        Self {
            k: 15,
            ann_type: "nndescent".to_string(),
            initialisation: "spectral".to_string(),
            init_range: None,
            randomised: false,
            nn_params: NearestNeighbourParams::default(),
            graph_params: UmapGraphParams::default(),
            optim_params: Fa2OptimParams::default(),
        }
    }
}

impl<T> Fa2Params<T>
where
    T: ManifoldsFloat,
{
    /// Create new ForceAtlas2 parameters with full control over every field.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours.
    /// * `ann_type` - (Approximate) nearest neighbour method. One of
    ///   `"exhaustive"`, `"ivf"`, `"hnsw"`, `"nndescent"`, `"annoy"`,
    ///   `"kmknn"` or `"balltree"`.
    ///   The `mlx` feature adds `"exhaustive_mlx"`, `"ivf_mlx"` and `"nndescent_mlx"`.
    /// * `initialisation` - Embedding initialisation method: `"spectral"`,
    ///   `"pca"` or `"random"`.
    /// * `init_range` - Optional initialisation range.
    /// * `randomised` - Whether randomised SVD is used for PCA
    ///   initialisation.
    /// * `nn_params` - Nearest neighbour parameters.
    /// * `graph_params` - Parameters for the fuzzy simplicial set.
    /// * `optim_params` - ForceAtlas2 optimisation parameters.
    ///
    /// ### Returns
    ///
    /// A fully specified set of ForceAtlas2 parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        k: usize,
        ann_type: String,
        initialisation: String,
        init_range: Option<T>,
        randomised: bool,
        nn_params: NearestNeighbourParams<T>,
        graph_params: UmapGraphParams<T>,
        optim_params: Fa2OptimParams<T>,
    ) -> Self {
        Self {
            k,
            ann_type,
            initialisation,
            init_range,
            randomised,
            nn_params,
            graph_params,
            optim_params,
        }
    }

    /// Default parameters for standard 2D visualisation.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours. Defaults to `15`.
    ///
    /// ### Returns
    ///
    /// Hopefully sensible standard parameters for 2D visualisation.
    pub fn new_default_2d(k: Option<usize>) -> Self {
        let default = Self::default();

        Self {
            k: k.unwrap_or(default.k),
            ..default
        }
    }
}

/// Construct the graph ForceAtlas2 lays out from high-dimensional data.
///
/// Runs the kNN search (or takes the pre-computed one), then builds the UMAP
/// fuzzy simplicial set. Unlike UMAP there is no weak-edge filter, since FA2
/// visits every edge each epoch rather than sampling them.
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances. Needs
///   to be a tuple of `(Vec<Vec<usize>>, Vec<Vec<T>>)` with indices and
///   distances excluding self, sorted by distance. A wider graph is cut to `k`,
///   with a warning on stderr.
/// * `k` - Number of nearest neighbours.
/// * `ann_type` - (Approximate) nearest neighbour method.
/// * `graph_params` - Fuzzy simplicial set parameters.
/// * `nn_params` - Nearest neighbour parameters.
/// * `seed` - Random seed.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Tuple of (graph, knn_indices, knn_dist)
#[allow(clippy::too_many_arguments)]
pub fn construct_fa2_graph<T>(
    data: impl ManifoldsMatrix<T>,
    precomputed_knn: PreComputedKnn<T>,
    k: usize,
    ann_type: String,
    graph_params: &UmapGraphParams<T>,
    nn_params: &NearestNeighbourParams<T>,
    seed: usize,
    verbose: usize,
) -> UmapGraphResults<T>
where
    T: ManifoldsFloat,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    let verbosity = parse_verbosity_level(verbose);

    let (knn_indices, knn_dist) = match precomputed_knn {
        Some((indices, distances)) => {
            if verbosity.normal_verbosity() {
                println!(" Using precomputed kNN graph");
            }
            truncate_knn(indices, distances, k)
        }
        None => {
            if verbosity.normal_verbosity() {
                println!(" kNN search via {} (k = {}) ...", ann_type, k);
            }
            let start_knn = Instant::now();
            let result =
                run_ann_search(data, k, ann_type, nn_params, seed, ann_verbose(verbosity))?;
            if verbosity.normal_verbosity() {
                println!(" ... kNN search in {:.2?}", start_knn.elapsed());
            }
            result
        }
    };

    if verbosity.normal_verbosity() {
        println!(" Building fuzzy simplicial set ...");
    }

    let start_graph = Instant::now();

    let (sigma, rho) = smooth_knn_dist(
        &knn_dist,
        knn_dist[0].len(),
        graph_params.local_connectivity,
        graph_params.bandwidth,
        64,
    );
    let graph = knn_to_coo(&knn_indices, &knn_dist, &sigma, &rho);
    let graph = symmetrise_graph(graph, graph_params.mix_weight);

    if verbosity.normal_verbosity() {
        println!(" ... graph in {:.2?}", start_graph.elapsed());
    }

    Ok((graph, knn_indices, knn_dist))
}

/// Shared tail of [`forceatlas2`] and [`forceatlas2_from_graph`]: run the
/// optimiser and transpose to `[2][n_samples]`.
///
/// ### Params
///
/// * `embd` - Initial embedding, `[n_samples][2]`
/// * `graph` - Symmetric weighted graph
/// * `optim_params` - ForceAtlas2 optimisation parameters
/// * `approx_type` - Repulsion approximation, `"barnes_hut" | "bh"`
/// * `verbose` - Verbosity level
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>`, `[2][n_samples]`.
fn fa2_optimise<T>(
    mut embd: Vec<Vec<T>>,
    graph: &CoordinateList<T>,
    optim_params: &Fa2OptimParams<T>,
    approx_type: &str,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
{
    let verbosity = parse_verbosity_level(verbose);

    let fa2_approx = parse_fa2_optimiser(approx_type).unwrap_or_else(|| {
        eprintln!(
            "Warning: unrecognised ForceAtlas2 approximation provided: {:?}. Default to Barnes-Hut.",
            approx_type
        );
        Fa2Opt::default()
    });

    let start_optim = Instant::now();
    match fa2_approx {
        Fa2Opt::BarnesHut => {
            if verbosity.normal_verbosity() {
                println!(
                    " Optimising via Barnes-Hut ForceAtlas2 ({} epochs) ...",
                    optim_params.n_epochs
                );
            }
            optimise_fa2(&mut embd, optim_params, graph, verbose)?;
        }
    }

    if verbosity.normal_verbosity() {
        println!(" ... optimised in {:.2?}", start_optim.elapsed());
    }

    let n_samples = embd.len();
    let mut transposed = vec![vec![T::zero(); n_samples]; 2];
    for (sample_idx, point) in embd.iter().enumerate() {
        transposed[0][sample_idx] = point[0];
        transposed[1][sample_idx] = point[1];
    }

    Ok(transposed)
}

/// Run ForceAtlas2 on the kNN graph of high-dimensional data
///
/// Builds the UMAP fuzzy simplicial set (as scanpy's `draw_graph` does) and
/// lays it out with ForceAtlas2: `1/d` repulsion between all node pairs with
/// masses `1 + degree`, linear (or LinLog) attraction along edges, gravity
/// towards the origin, and Gephi's adaptive per-node speed.
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances. Needs
///   to be a tuple of `(Vec<Vec<usize>>, Vec<Vec<T>>)` with indices and
///   distances excluding self.
/// * `params` - ForceAtlas2 parameters
/// * `approx_type` - Repulsion approximation: `"barnes_hut" | "bh"`
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>`, `[2][n_samples]`.
///
/// ### Errors
///
/// [`ManifoldsError::Fa2InvalidParam`] if `graph_params.mix_weight` is not 1, since
/// the layout needs a symmetric graph.
///
/// ### References
///
/// Jacomy et al., PLoS ONE, 2014 (ForceAtlas2)
pub fn forceatlas2<T>(
    data: impl ManifoldsMatrix<T>,
    precomputed_knn: PreComputedKnn<T>,
    params: &Fa2Params<T>,
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
    // mix_weight < 1 keeps a directed term in `symmetrise_graph`, and FA2
    // needs a symmetric graph
    if params.graph_params.mix_weight != T::one() {
        return Err(ManifoldsError::Fa2InvalidParam {
            param: "graph_params.mix_weight",
            value: params.graph_params.mix_weight.to_f64().unwrap_or(f64::NAN),
            requirement: "must be 1 (FA2 needs the symmetric fuzzy union)",
        });
    }

    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    let verbosity = parse_verbosity_level(verbose);
    let name = "ForceAtlas2";
    let start_run = Instant::now();
    if verbosity.normal_verbosity() {
        println!(
            "Running {} on {} samples ...",
            name,
            data.nrows().separate_with_underscores()
        );
    }

    let (graph, _, _) = construct_fa2_graph(
        data,
        precomputed_knn,
        params.k,
        params.ann_type.clone(),
        &params.graph_params,
        &params.nn_params,
        seed,
        verbose,
    )?;

    if verbosity.normal_verbosity() {
        println!(" Initialising via {} ...", params.initialisation);
    }

    let init_type =
        parse_initialisation(&params.initialisation, params.randomised, params.init_range)
            .unwrap_or_else(|| {
                eprintln!(
                    "Warning: unknown initialisation provided: {:?}. Defaulting to spectral.",
                    params.initialisation,
                );
                EmbdInit::SpectralInit { range: None }
            });

    let start_init = Instant::now();
    let embd = initialise_embedding(&init_type, 2, seed as u64, &graph, data)?;
    if verbosity.normal_verbosity() {
        println!(" ... initialised in {:.2?}", start_init.elapsed());
    }

    let res = fa2_optimise(embd, &graph, &params.optim_params, approx_type, verbose)?;
    if verbosity.normal_verbosity() {
        println!("... finished {} in {:.2?}", name, start_run.elapsed());
    }

    Ok(res)
}

/// Run ForceAtlas2 on a caller-supplied graph
///
/// For graphs that do not come from this crate's kNN pipeline, e.g. an SNN
/// graph built in R.
///
/// ### Params
///
/// * `graph` - Symmetric weighted graph in COO format, no self-loops needed
///   (they are dropped)
/// * `init` - Optional initial embedding, `[2][n_samples]` (the layout this
///   function returns). `None` draws a random layout in `[-10, 10]`.
/// * `optim_params` - ForceAtlas2 optimisation parameters
/// * `approx_type` - Repulsion approximation: `"barnes_hut" | "bh"`
/// * `seed` - Random seed for the random layout
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>`, `[2][n_samples]`, or an error if
/// the graph is asymmetric or does not match `init`.
///
/// ### References
///
/// Jacomy et al., PLoS ONE, 2014 (ForceAtlas2)
pub fn forceatlas2_from_graph<T>(
    graph: &CoordinateList<T>,
    init: Option<Vec<Vec<T>>>,
    optim_params: &Fa2OptimParams<T>,
    approx_type: &str,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
{
    let n = graph.n_samples;

    let embd = match init {
        Some(init) => {
            if init.len() != 2 {
                return Err(ManifoldsError::IncorrectDim { n_dim: init.len() });
            }
            if let Some(bad) = init.iter().map(|row| row.len()).find(|&len| len != n) {
                return Err(ManifoldsError::GraphSizeMismatch {
                    n_graph: n,
                    n_embd: bad,
                });
            }
            (0..n).map(|i| vec![init[0][i], init[1][i]]).collect()
        }
        None => random_layout(n, 2, seed as u64, None),
    };

    fa2_optimise(embd, graph, optim_params, approx_type, verbose)
}

///////////
// PHATE //
///////////

/// PHATE parameters
#[derive(Debug, Clone)]
pub struct PhateParams<T> {
    /// Number of output dimensions (default: 2)
    pub n_dim: usize,
    /// Neighbours that set the kernel bandwidth. The kNN search runs wider, see
    /// [`phate_search_width`].
    /// Defaults to `5`.
    pub k: usize,
    /// (Approximate) Nearest neighbour method. One of `"exhaustive"`, `"ivf"`,
    /// `"hnsw"`, `"nndescent"`, `"annoy"`, `"kmknn"`, or `"balltree"`.
    /// The `mlx` feature adds `"exhaustive_mlx"`, `"ivf_mlx"` and `"nndescent_mlx"`.
    /// Defaults to `"ivf"`.
    pub ann_type: String,
    /// Nearest neighbour search parameters to use
    pub nn_params: NearestNeighbourParams<T>,
    /// Diffusion parameters to use
    pub diffusion_params: PhateDiffusionParams<T>,
    /// Which MDS implementation to use: `"sgd_dense"` (default, also `"dense"`) or
    /// `"classic"`.
    pub mds_method: String,
    /// Optional number of iterations for MDS fitting
    pub mds_iter: Option<usize>,
    /// Shall randomised SVD be used for PCA-based initialisation
    /// Defaults to `true`.
    pub randomised: bool,
}

impl<T> Default for PhateParams<T>
where
    T: ManifoldsFloat,
{
    fn default() -> Self {
        Self {
            n_dim: 2,
            k: 5,
            ann_type: "ivf".to_string(),
            nn_params: NearestNeighbourParams::default(),
            diffusion_params: PhateDiffusionParams::default(),
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
    ///   The `mlx` feature adds `"exhaustive_mlx"`, `"ivf_mlx"` and `"nndescent_mlx"`.
    /// * `nn_params` - Nearest neighbour search parameters.
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
        nn_params: NearestNeighbourParams<T>,
        diffusion_params: PhateDiffusionParams<T>,
        mds_method: String,
        mds_iter: Option<usize>,
        randomised: bool,
    ) -> Self {
        Self {
            n_dim,
            k,
            ann_type,
            nn_params,
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
        let default = Self::default();

        Self {
            k: k.unwrap_or(default.k),
            ..default
        }
    }
}

/////////////////////
// PHATE diffusion //
/////////////////////

/// Reject a metric the potential distances and the landmark selection cannot use.
///
/// Some kNN backends do search in Manhattan distance, so without this the run only
/// fails after the search.
///
/// ### Params
///
/// * `method` - Name of the method, for the error message
/// * `metric` - The distance metric asked for
///
/// ### Returns
///
/// Nothing, or [`ManifoldsError::MetricNotSupported`] for Manhattan.
fn check_diffusion_metric(method: &'static str, metric: &str) -> Result<(), ManifoldsError> {
    match parse_ann_dist(metric) {
        Some(Dist::Manhattan) => Err(ManifoldsError::MetricNotSupported {
            method,
            metric: metric.to_string(),
        }),
        _ => Ok(()),
    }
}

/// Build the PHATE diffusion operator from high-dimensional data
///
/// Runs kNN search (or uses precomputed results), computes alpha decay
/// affinities, and constructs either a full or landmark diffusion operator
/// depending on `phate_params.n_landmarks`.
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `k` - Neighbours that set the kernel bandwidth
/// * `precomputed_knn` - Precomputed kNN indices and distances as
///   `Some((Vec<Vec<usize>>, Vec<Vec<T>>))`, or `None` to run search
///   internally. Indices and distances must exclude self. Every neighbour given
///   joins the kernel while its affinity clears `thresh`, so make the graph
///   [`phate_search_width`] wide, as the internal search is.
/// * `ann_type` - (Approximate) Nearest neighbour method: `"exhaustive"`,
///   `"kmknn"`, `"balltree"`, `"annoy"`, `"hnsw"`, or `"nndescent"`. If you
///   provide a weird string, the function will default to `"kmknn"`
/// * `nn_params` - Nearest neighbour search parameters
/// * `phate_params` - Full PHATE parameter struct. Reads `k` and every field of
///   `diffusion_params` bar `t` and `gamma`.
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
    data: impl ManifoldsMatrix<T>,
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
    check_diffusion_metric("PHATE", &nn_params.dist_metric)?;

    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    let verbosity = parse_verbosity_level(verbose);

    let (knn_indices, knn_dist) = match precomputed_knn {
        Some((indices, distances)) => {
            if verbosity.normal_verbosity() {
                println!(" Using precomputed kNN graph");
            }
            (indices, distances)
        }
        None => {
            if verbosity.normal_verbosity() {
                println!(
                    " kNN search via {} (k = {}) ...",
                    ann_type,
                    phate_search_width(k)
                );
            }
            let start_knn = Instant::now();
            let result = run_ann_search(
                data,
                phate_search_width(k),
                ann_type.to_string(),
                nn_params,
                seed,
                ann_verbose(verbosity),
            )?;
            if verbosity.normal_verbosity() {
                println!(" ... kNN search in {:.2?}", start_knn.elapsed());
            }
            result
        }
    };

    phate_diffusion_from_knn(
        data,
        &knn_indices,
        &knn_dist,
        &nn_params.dist_metric,
        phate_params,
        &SpectralBackend::Cpu,
        seed,
        verbose,
    )
}

/// Build the PHATE diffusion operator from a finished kNN graph.
///
/// ### Params
///
/// * `data` - Input data as samples x features
/// * `knn_indices` - Neighbour indices, excluding self
/// * `knn_dist` - Neighbour distances, excluding self
/// * `dist_metric` - Distance metric of the graph
/// * `phate_params` - Full PHATE parameter struct
/// * `backend` - Where spectral landmark selection runs
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// See [`construct_phate_diffusion`].
#[allow(clippy::too_many_arguments)]
fn phate_diffusion_from_knn<T>(
    data: MatRef<T>,
    knn_indices: &[Vec<usize>],
    knn_dist: &[Vec<T>],
    dist_metric: &str,
    phate_params: &PhateParams<T>,
    backend: &SpectralBackend,
    seed: usize,
    verbose: usize,
) -> Result<PhateDiffusion<T>, ManifoldsError>
where
    T: ManifoldsFloat,
{
    let verbosity = parse_verbosity_level(verbose);
    let (landmark_method, backend) =
        resolve_landmark_backend(&phate_params.diffusion_params.landmark_method, backend);

    if verbosity.normal_verbosity() {
        println!(" Building alpha-decay affinities ...");
    }
    let start_alpha_affinities = Instant::now();

    let graph = phate_alpha_decay_affinities(
        knn_indices,
        knn_dist,
        phate_params.k,
        phate_params.diffusion_params.decay,
        phate_params.diffusion_params.bandwidth_scale,
        phate_params.diffusion_params.thresh,
        &phate_params.diffusion_params.graph_symmetry,
    );

    if verbosity.normal_verbosity() {
        println!(
            " ... affinities in {:.2?}",
            start_alpha_affinities.elapsed()
        );
    }

    let affinity = add_unit_diagonal(&coo_to_csr(&graph));
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
                println!(" Building {} landmarks ...", n_landmarks);
            }
            let start_landmarks = Instant::now();
            let landmarks = PhateLandmarks::build_with_backend(
                data,
                &affinity,
                &diffusion_op,
                n_landmarks,
                landmark_method,
                dist_metric,
                seed,
                phate_params.diffusion_params.n_svd,
                &backend,
                verbose,
            )?;
            if verbosity.normal_verbosity() {
                println!(" ... landmarks in {:.2?}", start_landmarks.elapsed());
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
/// By default steps 3 to 5 run on a compressed landmark operator, with
/// [`PHATE_DEFAULT_LANDMARKS`] landmarks: the potential distances and the MDS are
/// L x L and the result is interpolated back to all samples. Setting
/// `diffusion_params.n_landmarks` to `None`, or to at least the number of samples,
/// runs the full operator instead, whose distance matrix is a dense N x N.
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `precomputed_knn` - Precomputed kNN indices and distances as
///   `Some((Vec<Vec<usize>>, Vec<Vec<T>>))`, or `None` to run search
///   internally. Indices and distances must exclude self.
/// * `phate_params` - PHATE parameters. See `PhateParams` for full
///   documentation of each field. `nn_params.dist_metric` has to be Euclidean or
///   cosine; Manhattan returns [`ManifoldsError::MetricNotSupported`].
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
    data: impl ManifoldsMatrix<T>,
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
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    let verbosity = parse_verbosity_level(verbose);
    let name = "PHATE";
    let start_run = Instant::now();
    if verbosity.normal_verbosity() {
        println!(
            "Running {} on {} samples ...",
            name,
            data.nrows().separate_with_underscores()
        );
    }

    let phate_diffusion = construct_phate_diffusion(
        data,
        phate_params.k,
        precomputed_knn,
        &phate_params.ann_type,
        &phate_params.nn_params,
        &phate_params,
        seed,
        verbose,
    )?;

    phate_from_diffusion(
        data.nrows(),
        phate_diffusion,
        &phate_params,
        start_run,
        seed,
        verbose,
    )
}

/// Run PHATE with spectral landmark selection and kNN search on a cubecl device.
///
/// Identical to [`phate`] except for where those two stages run: the randomised
/// SVD and the k-means on the device in `f32`, the rest on the CPU. Only the
/// spectral landmark method uses the device beyond the kNN search.
/// `landmark_method = "spectral_mlx"` keeps the search on wgpu and runs the SVD
/// and the k-means on MLX.
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `precomputed_knn` - Precomputed kNN indices and distances, excluding self,
///   or `None` to search on the device.
/// * `phate_params` - PHATE parameters. `ann_type` is read as a GPU key, see
///   [`run_ann_search_gpu`]; its `nn_params` are not used, bar the `dist_metric`,
///   which is overwritten by the one of the `nn_params` argument.
/// * `nn_params` - GPU nearest neighbour parameters
/// * `device` - The GPU device to use
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>` where the outer vector has length
/// `n_dim` and each inner vector has length `n_samples`.
#[cfg(feature = "gpu")]
pub fn phate_gpu<T>(
    data: impl ManifoldsMatrix<T>,
    precomputed_knn: PreComputedKnn<T>,
    mut phate_params: PhateParams<T>,
    nn_params: &NearestNeighbourParamsGpu<T>,
    device: Device,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloatGpu,
    StandardNormal: Distribution<T>,
{
    check_diffusion_metric("PHATE", &nn_params.dist_metric)?;

    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    let verbosity = parse_verbosity_level(verbose);
    let start_run = Instant::now();
    phate_params.nn_params.dist_metric = nn_params.dist_metric.clone();

    let (knn_indices, knn_dist) = match precomputed_knn {
        Some(knn) => knn,
        None => {
            let start_knn = Instant::now();
            let result = run_ann_search_gpu::<T>(
                data,
                phate_search_width(phate_params.k),
                phate_params.ann_type.clone(),
                nn_params,
                device.clone(),
                seed,
                ann_verbose(verbosity),
            )?;
            if verbosity.normal_verbosity() {
                println!(" ... kNN search in {:.2?}", start_knn.elapsed());
            }
            result
        }
    };

    let phate_diffusion = phate_diffusion_from_knn(
        data,
        &knn_indices,
        &knn_dist,
        &nn_params.dist_metric,
        &phate_params,
        &SpectralBackend::Wgpu(device),
        seed,
        verbose,
    )?;

    phate_from_diffusion(
        data.nrows(),
        phate_diffusion,
        &phate_params,
        start_run,
        seed,
        verbose,
    )
}

/// Everything in PHATE after the diffusion operator: diffusion time, potential,
/// MDS and, for landmarks, the interpolation back to all samples.
///
/// ### Params
///
/// * `n_obs` - Number of samples
/// * `phate_diffusion` - Full or landmark diffusion operator
/// * `phate_params` - PHATE parameters
/// * `start_run` - Start of the whole run, for the verbose total
/// * `seed` - Random seed for reproducibility
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// See [`phate`].
fn phate_from_diffusion<T>(
    n_obs: usize,
    phate_diffusion: PhateDiffusion<T>,
    phate_params: &PhateParams<T>,
    start_run: Instant,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
    StandardNormal: Distribution<T>,
{
    let verbosity = parse_verbosity_level(verbose);
    let name = "PHATE";

    let start_t = Instant::now();
    let t = match phate_params.diffusion_params.t {
        PhateTime::Auto { t_max } => {
            if verbosity.normal_verbosity() {
                println!(" Finding optimal t (t_max = {}) ...", t_max);
            }
            match &phate_diffusion {
                PhateDiffusion::Landmark { landmarks } => landmarks.find_optimal_t(t_max),
                PhateDiffusion::Full { operator } => {
                    let entropy = von_neumann_entropy_f64(operator, t_max)?;
                    find_knee_point(&entropy)
                }
            }
        }
        PhateTime::Fixed(t) => Ok(t),
    }?;
    if verbosity.normal_verbosity() {
        println!(" ... t = {} in {:.2?}", t, start_t.elapsed());
    }

    let mds_method = parse_mds_method(&phate_params.mds_method).unwrap_or_default();
    let dist = parse_ann_dist(&phate_params.nn_params.dist_metric).unwrap_or_default();
    let mds_params =
        MdsOptimParams::new(n_obs, phate_params.randomised, phate_params.mds_iter, None);

    let start_potential = Instant::now();
    let potential_fill = potential_of_zero(phate_params.diffusion_params.gamma);
    let start_mds;

    let embedding = match phate_diffusion {
        PhateDiffusion::Full { operator } => {
            if verbosity.normal_verbosity() {
                println!(" Powering diffusion operator ...");
            }
            let powered = matrix_power(&operator, t)?;
            let potential = calculate_potential(&powered, 1, phate_params.diffusion_params.gamma)?;

            if verbosity.normal_verbosity() {
                println!(
                    " ... potential ({} x {}) in {:.2?}",
                    potential.shape().0,
                    potential.shape().1,
                    start_potential.elapsed()
                );
            }

            start_mds = Instant::now();
            let distances =
                compute_potential_distances_with_fill(&potential, &dist, potential_fill);

            let res = match mds_method {
                MdsMethod::ClassicMds => {
                    if verbosity.normal_verbosity() {
                        println!(" Running classic MDS ...");
                    }
                    classic_mds(&distances, phate_params.n_dim, mds_params.randomised, seed)
                }
                MdsMethod::SgdDense => {
                    if verbosity.normal_verbosity() {
                        println!(" Running SGD-MDS ...");
                    }
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
                    " Powering landmark operator ({} landmarks) ...",
                    landmarks.get_n_landmarks()
                );
            }
            let landmark_potential = dense_potential_rows(
                &landmarks.power_dense(t)?,
                phate_params.diffusion_params.gamma,
            );

            if verbosity.normal_verbosity() {
                println!(
                    " ... landmark potential ({0} x {0}) in {1:.2?}",
                    landmark_potential.len(),
                    start_potential.elapsed()
                );
                println!(" Computing landmark distances ...");
            }

            start_mds = Instant::now();
            let landmark_distances = pairwise_row_distances(&landmark_potential, &dist);

            let landmark_mds_params = MdsOptimParams::new(
                landmarks.get_n_landmarks(),
                phate_params.randomised,
                phate_params.mds_iter,
                None,
            );

            if verbosity.normal_verbosity() {
                println!(" Running MDS on landmarks ...");
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
                println!(" Interpolating to all points via Nyström ...");
            }
            landmarks.interpolate_embedding(&landmark_embedding)
        }
    };

    if verbosity.normal_verbosity() {
        println!(" ... MDS in {:.2?}", start_mds.elapsed());
        println!("... finished {} in {:.2?}", name, start_run.elapsed());
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
    /// Defaults to `2`.
    pub n_dim: usize,
    /// (Approximate) Nearest neighbour method. One of `"exhaustive"`, `"ivf"`,
    /// `"hnsw"`, `"nndescent"`, `"annoy"`, `"kmknn"` or `"balltree"`.
    /// The `mlx` feature adds `"exhaustive_mlx"`, `"ivf_mlx"` and `"nndescent_mlx"`.
    /// Defaults to `"ivf"`.
    pub ann_type: String,
    /// Which optimiser to use. Options are `"adam"` and `"adam_parallel"`
    pub optimiser: String,
    /// Near pairs per point (attractive). The kNN search runs 50 wider, and the
    /// pairs are the best of those by locally scaled distance. Default 10.
    pub n_near: usize,
    /// Mid-near pairs per point (attractive, global structure). Default 5.
    pub n_mid_near: usize,
    /// Further (random) pairs per point (repulsive). Default 20.
    pub n_further: usize,
    /// Embedding initialisation. Default `"pca"`. PCA is strongly recommended
    /// for PaCMAP as random init degrades global structure.
    /// Defaults to `"pca"`.
    pub initialisation: String,
    /// Scale of the initial embedding. For PCA, the multiplier on the scores.
    pub range: Option<T>,
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
            // the search runs at n_near + 50, where IVF scales better
            ann_type: "ivf".to_string(),
            optimiser: "adam_parallel".to_string(),
            n_near: 10,
            n_mid_near: 5,
            n_further: 20,
            initialisation: "pca".to_string(),
            range: Some(T::from_f64(PACMAP_PCA_SCALE).unwrap()),
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
    /// ### Params
    ///
    /// * `n_dim` - Number of dimensions for the embedding.
    /// * `ann_type` - (Approximate) nearest neighbour method. One of
    ///   `"exhaustive"`, `"ivf"`, `"hnsw"`, `"nndescent"`, `"annoy"`,
    ///   `"kmknn"` or `"balltree"`.
    ///   The `mlx` feature adds `"exhaustive_mlx"`, `"ivf_mlx"` and `"nndescent_mlx"`.
    /// * `optimiser` - Which optimiser to use. Options are `"adam"` and
    ///   `"adam_parallel"`.
    /// * `n_near` - Near pairs per point (attractive). The kNN search runs 50
    ///   wider, and the pairs are the best of those by locally scaled distance.
    /// * `n_mid_near` - Mid-near pairs per point (attractive, global structure).
    /// * `n_further` - Further (random) pairs per point (repulsive).
    /// * `initialisation` - Embedding initialisation. PCA is strongly
    ///   recommended for PaCMAP as random init degrades global structure.
    /// * `range` - Scale of the initial embedding. For PCA, the multiplier on the
    ///   scores.
    /// * `nn_params` - Nearest neighbour search parameters.
    /// * `optim_params` - Optimiser parameters.
    ///
    /// ### Returns
    ///
    /// A fully specified set of PaCMAP parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_dim: usize,
        ann_type: String,
        optimiser: String,
        n_near: usize,
        n_mid_near: usize,
        n_further: usize,
        initialisation: String,
        range: Option<T>,
        nn_params: NearestNeighbourParams<T>,
        optim_params: PacmapOptimParams<T>,
    ) -> Self {
        Self {
            n_dim,
            ann_type,
            optimiser,
            n_near,
            n_mid_near,
            n_further,
            initialisation,
            range,
            nn_params,
            optim_params,
        }
    }

    /// Default parameters for standard 2D visualisation.
    ///
    /// Returns the default parameters but lets you tune `n_near`, the number
    /// of near pairs per point.
    ///
    /// ### Params
    ///
    /// * `n_near` - Number of near pairs. Defaults to `10`.
    ///
    /// ### Returns
    ///
    /// Hopefully sensible standard parameters for 2D visualisation.
    pub fn new_default_2d(n_near: Option<usize>) -> Self {
        let default = Self::default();
        let n_near = n_near.unwrap_or(default.n_near);
        Self { n_near, ..default }
    }
}

/// Scale of the PaCMAP PCA initialisation. Set by the reference implementation.
const PACMAP_PCA_SCALE: f64 = 0.01;

/// Above this many features the PaCMAP reference works on this many PCA scores
/// directly and skips its global min-max normalisation.
const PACMAP_NORMALISE_MAX_DIM: usize = 100;

/// PCA initialisation for PaCMAP.
///
/// Takes the PC scores times a small constant, so the ratio between the
/// components survives, unlike [`pca_layout`], which sets every component to the
/// same spread. That ratio is what the mid-near phase starts from.
///
/// The layout is sensitive to the accuracy of these PCs: a rank 2 randomised SVD
/// leaves enough of PC3 in PC2 to end up in a visibly different embedding. So
/// the scores are exact up to [`PACMAP_NORMALISE_MAX_DIM`] features, and above
/// that the leading ones of a randomised SVD of that rank.
///
/// ### Params
///
/// * `data` - Input data, samples x features
/// * `data_flat` - The same data, row-major
/// * `n_dim` - Number of embedding dimensions
/// * `scale` - Multiplier on the scores
/// * `seed` - Random seed
///
/// ### Returns
///
/// The initial embedding, shape `[n_samples][n_dim]`.
fn pacmap_pca_init<T>(
    data: MatRef<T>,
    data_flat: &[T],
    n_dim: usize,
    scale: T,
    seed: u64,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
{
    let exact = data.ncols() <= PACMAP_NORMALISE_MAX_DIM;
    let mut scores = if exact {
        pca_scores(data, n_dim, false, seed)?
    } else {
        pca_scores(data, PACMAP_NORMALISE_MAX_DIM, true, seed)?
    };
    scores.iter_mut().for_each(|row| row.truncate(n_dim));

    let spread = if exact {
        let (min, max) = data_flat
            .iter()
            .fold((T::infinity(), T::neg_infinity()), |(lo, hi), &x| {
                (lo.min(x), hi.max(x))
            });
        max - min
    } else {
        T::one()
    };
    let factor = if spread > T::zero() {
        scale / spread
    } else {
        scale
    };

    scores
        .iter_mut()
        .for_each(|row| row.iter_mut().for_each(|v| *v *= factor));
    Ok(scores)
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
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `precomputed_knn` - Optional precomputed kNN, with k >= `params.n_near`,
///   sorted by distance. The near pairs are ranked from what is given, so make
///   it `params.n_near + 50` wide, as the internal search is.
/// * `params` - PaCMAP parameters.
/// * `seed` - Seed for reproducibility.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding as `Vec<Vec<T>>` with shape `[n_dim][n_samples]`.
///
/// ### Errors
///
/// [`ManifoldsError::NoData`] on an empty precomputed graph and
/// [`ManifoldsError::NotEnoughNeighbours`] if it is narrower than `params.n_near`.
///
/// ### References
///
/// Wang et al. (2021): "Understanding How Dimension Reduction Tools Work: An
/// Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data
/// Visualization", JMLR
pub fn pacmap<T>(
    data: impl ManifoldsMatrix<T>,
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
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    let n_samples = data.nrows();

    let verbosity = parse_verbosity_level(verbose);
    let name = "PaCMAP";
    let start_run = Instant::now();
    if verbosity.normal_verbosity() {
        println!(
            "Running {} on {} samples ...",
            name,
            data.nrows().separate_with_underscores()
        );
    }

    let (knn_indices, knn_dists) = match precomputed_knn {
        Some((indices, distances)) => {
            if indices.is_empty() {
                return Err(ManifoldsError::NoData);
            }
            if indices[0].len() < params_pacmap.n_near {
                return Err(ManifoldsError::NotEnoughNeighbours {
                    n_provided: indices[0].len(),
                    n_near: params_pacmap.n_near,
                });
            }
            if verbosity.normal_verbosity() {
                println!(" Using precomputed kNN graph");
            }
            (indices, distances)
        }
        None => {
            let search_width =
                (params_pacmap.n_near + PACMAP_EXTRA_NEIGHBOURS).min(n_samples.saturating_sub(1));
            if verbosity.normal_verbosity() {
                println!(
                    " kNN search via {} (k = {}) ...",
                    params_pacmap.ann_type, search_width
                );
            }
            let start_knn = Instant::now();
            let result = run_ann_search(
                data,
                search_width,
                params_pacmap.ann_type.clone(),
                &params_pacmap.nn_params,
                seed,
                ann_verbose(verbosity),
            )?;
            if verbosity.normal_verbosity() {
                println!(" ... kNN search in {:.2?}", start_knn.elapsed());
            }
            result
        }
    };

    if verbosity.normal_verbosity() {
        println!(" Building PaCMAP pairs ...");
    }

    let start_pairs = Instant::now();

    let (data_flat, _, dim) = matrix_to_flat(data);
    let metric = parse_ann_dist(&params_pacmap.nn_params.dist_metric).unwrap_or_default();
    let near_neighbours = rank_near_neighbours(&knn_indices, &knn_dists, params_pacmap.n_near);

    let pairs: PacmapPairs = construct_pacmap_pairs(
        &data_flat,
        dim,
        &near_neighbours,
        params_pacmap.n_near,
        params_pacmap.n_mid_near,
        params_pacmap.n_further,
        &metric,
        seed as u64,
    );

    let end_pairs = start_pairs.elapsed();

    if verbosity.normal_verbosity() {
        println!(
            " ... pairs in {:.2?}: {} near, {} mid-near, {} further",
            end_pairs,
            pairs.near.len().separate_with_underscores(),
            pairs.mid_near.len().separate_with_underscores(),
            pairs.further.len().separate_with_underscores()
        );
    }

    let init_type = parse_initialisation(&params_pacmap.initialisation, true, params_pacmap.range)
        .unwrap_or_else(|| {
            eprintln!(
                "Warning: unknown initialisation provided: {:?}. Defaulting to PCA.",
                params_pacmap.initialisation,
            );
            EmbdInit::PcaInit {
                range: Some(T::from_f64(PACMAP_PCA_SCALE).unwrap()),
                randomised: true,
            }
        });

    if verbosity.normal_verbosity() {
        println!(" Initialising via {} ...", params_pacmap.initialisation);
    }

    let start_init = Instant::now();

    let mut embd = match init_type {
        EmbdInit::PcaInit { range, .. } => pacmap_pca_init(
            data,
            &data_flat,
            params_pacmap.n_dim,
            range.unwrap_or_else(|| T::from_f64(PACMAP_PCA_SCALE).unwrap()),
            seed as u64,
        )?,
        _ => {
            let dummy_graph = knn_to_coo_unweighted(&knn_indices);
            initialise_embedding(
                &init_type,
                params_pacmap.n_dim,
                seed as u64,
                &dummy_graph,
                data,
            )?
        }
    };

    if verbosity.normal_verbosity() {
        println!(" ... initialised in {:.2?}", start_init.elapsed());
    }

    let optimiser = parse_pacmap_optimiser(&params_pacmap.optimiser).unwrap_or_default();

    if verbosity.normal_verbosity() {
        println!(
            " Optimising via Adam ({} epochs) ...",
            params_pacmap.optim_params.n_epochs
        );
    }

    let start_optim = Instant::now();

    match optimiser {
        PacMapOptimiser::AdamParallel => {
            optimise_pacmap_parallel(&mut embd, &pairs, &params_pacmap.optim_params, verbose)?;
        }
        PacMapOptimiser::Adam => {
            optimise_pacmap(&mut embd, &pairs, &params_pacmap.optim_params, verbose)?;
        }
    }

    let end_optim = start_optim.elapsed();

    if verbosity.normal_verbosity() {
        println!(" ... optimised in {:.2?}", end_optim);
        println!("... finished {} in {:.2?}", name, start_run.elapsed());
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
    /// Defaults to `5`.
    pub k: usize,
    /// (Approximate) Nearest neighbour method. One of `"exhaustive"`, `"ivf"`,
    /// `"hnsw"`, `"nndescent"`, `"annoy"`, `"kmknn"` or `"balltree"`.
    /// The `mlx` feature adds `"exhaustive_mlx"`, `"ivf_mlx"` and `"nndescent_mlx"`.
    /// Defaults to `"nndescent"`.
    pub ann_type: String,
    /// Nearest neighbour search parameters
    pub nn_params: NearestNeighbourParams<T>,
    /// Multiplicative factor applied to the adaptive kernel bandwidth
    /// Defaults to `1.0`.
    pub bandwidth_scale: T,
    /// Sparsity threshold applied to kernel entries
    /// Defaults to `1e-4`.
    pub thresh: T,
    /// Graph symmetrisation: `"add"`, `"multiply"`, `"mnn"` or `"none"`
    /// Defaults to `"add"`.
    pub graph_symmetry: String,
    /// Anisotropic density-correction exponent in [0, 1]. 0 gives the
    /// normalised graph Laplacian, 0.5 the Fokker-Planck operator, 1 the
    /// Laplace-Beltrami operator.
    /// Defaults to `1.0`.
    pub alpha_norm: T,
    /// Diffusion time: `Auto` picks the VNE knee, `Fixed(t)` uses `t` directly.
    pub t: PhateTime,
    /// Landmark count. `None` or `>= n` runs full DM.
    ///
    /// Keep `k` a small fraction of the landmark count. The landmark kernel
    /// connects each landmark to its `k` nearest landmarks, so when `k` covers a
    /// large share of them the kernel turns broad, the eigenvalues drop, and
    /// `lambda^t` shrinks the later coordinates towards a line. On small data,
    /// prefer the full operator.
    pub n_landmarks: Option<usize>,
    /// `"random"`, `"spectral"` or `"density"`
    /// Defaults to `"spectral"`.
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
            ann_type: "nndescent".to_string(),
            nn_params: NearestNeighbourParams::default(),
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
    /// * `nn_params` - Nearest neighbour search parameters.
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
    ///   Keep `k` a small fraction of it, see [`DiffusionMapsParams::n_landmarks`].
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
        nn_params: NearestNeighbourParams<T>,
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
            nn_params,
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
        let default = Self::default();

        Self {
            k: k.unwrap_or(default.k),
            ..default
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
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
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
    data: impl ManifoldsMatrix<T>,
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
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    let use_full = match dm_params.n_landmarks {
        None => true,
        Some(n) => n >= data.nrows(),
    };
    if !use_full {
        check_diffusion_metric("Landmark diffusion maps", &nn_params.dist_metric)?;
    }

    let verbosity = parse_verbosity_level(verbose);

    let needs_affinity =
        use_full || matches!(dm_params.landmark_method.as_str(), "spectral" | "density");

    let affinity = if needs_affinity {
        let (knn_indices, knn_dist) = match precomputed_knn {
            Some((indices, distances)) => {
                if verbosity.normal_verbosity() {
                    println!(" Using precomputed kNN graph");
                }
                (indices, distances)
            }
            None => {
                if verbosity.normal_verbosity() {
                    println!(" kNN search via {} (k = {}) ...", ann_type, k);
                }
                let start_knn = Instant::now();
                let res = run_ann_search(
                    data,
                    k,
                    ann_type.to_string(),
                    nn_params,
                    seed,
                    ann_verbose(verbosity),
                )?;
                if verbosity.normal_verbosity() {
                    println!(" ... kNN search in {:.2?}", start_knn.elapsed());
                }
                res
            }
        };

        if verbosity.normal_verbosity() {
            println!(" Building Gaussian kernel affinities ...");
        }
        let graph = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dist,
            dm_params.k,
            Some(T::from_f64(2.0).unwrap()),
            dm_params.bandwidth_scale,
            dm_params.thresh,
            &dm_params.graph_symmetry,
        );
        Some(coo_to_csr(&graph))
    } else {
        if verbosity.detailed_verbosity() {
            println!(" Skipping full affinity (random landmarks)");
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
        println!(" Building {} landmarks ...", n_landmarks);
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
        println!(" ... landmarks in {:.2?}", start_l.elapsed());
    }
    Ok(DiffusionMapsOperator::Landmark { landmarks })
}

/// Run diffusion maps end-to-end.
///
/// 1. kNN graph on the raw data
/// 2. Gaussian kernel via alpha-decay with decay = 2 (adaptive bandwidth)
/// 3. Anisotropic (alpha) normalisation for density correction
/// 4. Symmetric diffusion operator P_sym = D^{-1/2} K D^{-1/2}
/// 5. Top eigenpairs of P_sym via Lanczos; the dense path asks for
///    (n_dim + 5) to give the solve some headroom, the landmark path for
///    (n_dim + 1)
/// 6. Drop the trivial eigenvalue (= 1) and scale each non-trivial eigenvector
///    phi_k by lambda_k^t
///
/// With landmarks, steps 4–6 run on the L×L operator and are Nystroem-extended
/// back to N points.
///
/// Degenerate data (duplicate points, a disconnected landmark set) can leave
/// Lanczos with fewer than `n_dim + 1` converged eigenpairs, which surfaces as
/// [`ManifoldsError::InsufficientEigenpairs`] rather than a truncated
/// embedding.
///
/// The kernel has no diagonal, so a point can lose every edge: `"multiply"` and
/// `"mnn"` drop one-sided edges, and a high `thresh` drops weak ones under any
/// symmetry. Such a point has no diffusion coordinate, which surfaces as
/// [`ManifoldsError::IsolatedPoints`]. An auto `t` with `t_max` below 3 returns
/// [`ManifoldsError::TMaxTooSmall`].
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `precomputed_knn` - Optional precomputed kNN; skips search if provided.
///   The `dm_params.k`-th neighbour sets the bandwidth and every neighbour given
///   joins the kernel while its affinity clears `thresh`. Unlike UMAP, densMAP
///   and ForceAtlas2, a wider graph is not cut to `k`. A list shorter than `k`
///   takes the bandwidth from its last neighbour.
/// * `dm_params` - Diffusion maps parameters
/// * `seed` - RNG seed
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding as `Vec<Vec<T>>` with shape `[n_dim][n_samples]`
///
/// ### References
///
/// Coifman & Lafon (2006): "Diffusion maps", Applied and Computational Harmonic
/// Analysis
pub fn diffusion_maps<T>(
    data: impl ManifoldsMatrix<T>,
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
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    let verbosity = parse_verbosity_level(verbose);
    let name = "diffusion maps";
    let start_run = Instant::now();
    if verbosity.normal_verbosity() {
        println!(
            "Running {} on {} samples ...",
            name,
            data.nrows().separate_with_underscores()
        );
    }

    let op = construct_diffusion_maps_operator(
        data,
        dm_params.k,
        precomputed_knn,
        &dm_params.ann_type,
        &dm_params.nn_params,
        &dm_params,
        seed,
        verbose,
    )?;

    let embedding: Vec<Vec<T>> = match op {
        DiffusionMapsOperator::Full {
            p_sym,
            sqrt_degrees,
        } => {
            let start_t = Instant::now();
            let t = match dm_params.t {
                PhateTime::Auto { t_max } => {
                    if verbosity.normal_verbosity() {
                        println!(" Finding optimal t (t_max = {}) ...", t_max);
                    }
                    let entropy = landmark_von_neumann_entropy(&p_sym, t_max)?;
                    find_knee_point(&entropy)?
                }
                PhateTime::Fixed(t) => t,
            };
            if verbosity.normal_verbosity() {
                println!(" ... t = {} in {:.2?}", t, start_t.elapsed());
            }
            let n_ask = (dm_params.n_dim + 5).min(sqrt_degrees.len() - 1);
            if verbosity.normal_verbosity() {
                println!(" Computing top {} eigenpairs ...", n_ask);
            }
            let start_eig = Instant::now();
            let (evals, evecs) = compute_largest_eigenpairs_lanczos(&p_sym, n_ask, seed as u64)?;

            // Lanczos can converge fewer pairs than asked for on a degenerate
            // operator; the loop below indexes up to `n_dim` unconditionally
            if evals.len() <= dm_params.n_dim {
                return Err(ManifoldsError::InsufficientEigenpairs {
                    found: evals.len(),
                    requested: dm_params.n_dim + 1,
                });
            }

            if verbosity.normal_verbosity() {
                println!(" ... eigendecomposition in {:.2?}", start_eig.elapsed());
            }

            let n = sqrt_degrees.len();
            let mut embedding = vec![vec![T::zero(); dm_params.n_dim]; n];
            for comp_idx in 1..=dm_params.n_dim {
                let lambda = T::from_f64(evals[comp_idx]).unwrap();
                let lambda_t = lambda.powi(t as i32);

                let mut max_abs = 0.0f64;
                let mut sign = 1.0f64;
                for i in 0..n {
                    let v = evecs[i][comp_idx];
                    if v.abs() > max_abs {
                        max_abs = v.abs();
                        sign = if v >= 0.0 { 1.0 } else { -1.0 };
                    }
                }
                for i in 0..n {
                    let u = T::from_f64(evecs[i][comp_idx] * sign).unwrap();
                    embedding[i][comp_idx - 1] = lambda_t * u / sqrt_degrees[i];
                }
            }
            embedding
        }
        DiffusionMapsOperator::Landmark { landmarks } => {
            let start_t = Instant::now();
            let t = match dm_params.t {
                PhateTime::Auto { t_max } => {
                    if verbosity.normal_verbosity() {
                        println!(" Finding optimal t on landmarks (t_max = {}) ...", t_max);
                    }
                    landmarks.find_optimal_t(t_max)?
                }
                PhateTime::Fixed(t) => t,
            };
            if verbosity.normal_verbosity() {
                println!(" ... t = {} in {:.2?}", t, start_t.elapsed());
                println!(
                    " Eigendecomposing {} x {} landmark operator ...",
                    landmarks.get_n_landmarks(),
                    landmarks.get_n_landmarks()
                );
            }
            let start_eig = Instant::now();
            let (evals, evecs) = landmarks.eigendecompose(dm_params.n_dim, seed as u64)?;

            // as above: `compute_landmark_embedding` indexes up to `n_dim`
            if evals.len() <= dm_params.n_dim {
                return Err(ManifoldsError::InsufficientEigenpairs {
                    found: evals.len(),
                    requested: dm_params.n_dim + 1,
                });
            }

            if verbosity.normal_verbosity() {
                println!(" ... eigendecomposition in {:.2?}", start_eig.elapsed());
            }
            if verbosity.normal_verbosity() {
                println!(" Extending to all points via Nyström ...");
            }
            let (landmark_embedding, lambdas) =
                landmarks.compute_landmark_embedding(&evals, &evecs, dm_params.n_dim, t);
            landmarks.nystrom_extend(&landmark_embedding, &lambdas)
        }
    };

    if verbosity.normal_verbosity() {
        println!("... finished {} in {:.2?}", name, start_run.elapsed());
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
    /// Defaults to `2`.
    pub n_dim: usize,
    /// Number of neighbours
    /// Defaults to `15`.
    pub k: usize,
    /// Which of the possible approximate nearest neighbour searches to use.
    /// Defaults to `"nndescent"`.
    pub ann_type: String,
    /// Vector of usizes for the hidden layers in the MLP.
    /// Defaults to `[128, 128, 128]`.
    pub hidden_layers: Vec<usize>,
    /// Nearest neighbour parameters.
    pub nn_params: NearestNeighbourParams<T>,
    /// The graph parameters for the generation of the graph structure.
    pub graph_params: UmapGraphParams<T>,
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
            ann_type: "nndescent".to_string(),
            hidden_layers: vec![128, 128, 128],
            nn_params: NearestNeighbourParams::default(),
            graph_params: UmapGraphParams::default(),
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
    ///   The `mlx` feature adds `"exhaustive_mlx"`, `"ivf_mlx"` and `"nndescent_mlx"`.
    /// * `hidden_layers` - Hidden layer sizes for the MLP encoder.
    /// * `nn_params` - Nearest neighbour parameters.
    /// * `graph_params` - UMAP graph generation parameters.
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
        graph_params: UmapGraphParams<T>,
        train_param: TrainParametricParams<T>,
    ) -> Self {
        Self {
            n_dim,
            k,
            ann_type,
            hidden_layers,
            nn_params,
            graph_params,
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
        let min_dist = min_dist.unwrap_or(T::from_f64(DEFAULT_MIN_DIST).unwrap());
        let spread = spread.unwrap_or(T::from_f64(DEFAULT_SPREAD).unwrap());
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
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `precomputed_knn` - Precomputed k-nearest neighbours and distances. Needs
///   to be a tuple of `(Vec<Vec<usize>>, Vec<Vec<T>>)` with indices and
///   distances excluding self.
/// * `umap_params` - Configuration parameters for parametric UMAP
/// * `device` - Burn device for neural network training; switched to autodiff
///   internally
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
/// Sainburg, McInnes & Gentner (2021): "Parametric UMAP Embeddings for
/// Representation and Semisupervised Learning", Neural Computation
pub fn parametric_umap<T>(
    data: impl ManifoldsMatrix<T>,
    precomputed_knn: PreComputedKnn<T>,
    umap_params: &ParametricUmapParams<T>,
    device: &burn::tensor::Device,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat + Element,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    // parse various parameters
    let nn_params = umap_params.nn_params.clone();

    let verbosity = parse_verbosity_level(verbose);
    let name = "parametric UMAP";
    let start_run = Instant::now();
    if verbosity.normal_verbosity() {
        println!(
            "Running {} on {} samples ...",
            name,
            data.nrows().separate_with_underscores()
        );
    }

    if verbosity.detailed_verbosity() {
        println!(
            " alpha = {:.2?}, beta = {:.2?}",
            ToPrimitive::to_f32(&umap_params.train_param.a).unwrap(),
            ToPrimitive::to_f32(&umap_params.train_param.b).unwrap()
        );
    }

    let (graph, _, _) = construct_umap_graph(
        data,
        precomputed_knn,
        umap_params.k,
        umap_params.ann_type.clone(),
        &umap_params.graph_params,
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

    let (embd, _) = train_parametric_umap::<T>(
        data,
        graph,
        &model_params,
        &umap_params.train_param,
        device,
        seed,
        verbosity,
    );

    if verbosity.normal_verbosity() {
        println!("... finished {} in {:.2?}", name, start_run.elapsed());
    }

    Ok(embd)
}

#[cfg(feature = "parametric")]
/// Train the parametric UMAP model and return it
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `umap_params` - Configuration parameters for parametric UMAP
/// * `device` - Burn device for neural network training; switched to autodiff
///   internally
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
pub fn train_parametric_umap_model<T>(
    data: impl ManifoldsMatrix<T>,
    umap_params: &ParametricUmapParams<T>,
    device: &burn::tensor::Device,
    seed: usize,
    verbose: usize,
) -> ParametricUmapResults<T>
where
    T: ManifoldsFloat + Element,
    HnswIndex<T>: HnswState<T>,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
{
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    // parse various parameters
    let nn_params = umap_params.nn_params.clone();
    let verbosity = parse_verbosity_level(verbose);
    let name = "parametric UMAP";
    let start_run = Instant::now();
    if verbosity.normal_verbosity() {
        println!(
            "Running {} on {} samples ...",
            name,
            data.nrows().separate_with_underscores()
        );
    }

    if verbosity.detailed_verbosity() {
        println!(
            " alpha = {:.2?}, beta = {:.2?}",
            ToPrimitive::to_f32(&umap_params.train_param.a).unwrap(),
            ToPrimitive::to_f32(&umap_params.train_param.b).unwrap()
        );
    }

    let (graph, _, _) = construct_umap_graph(
        data,
        None,
        umap_params.k,
        umap_params.ann_type.clone(),
        &umap_params.graph_params,
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

    let (embd, trained_model) = train_parametric_umap::<T>(
        data,
        graph,
        &model_params,
        &umap_params.train_param,
        device,
        seed,
        verbosity,
    );

    if verbosity.normal_verbosity() {
        println!("... finished {} in {:.2?}", name, start_run.elapsed());
    }

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
    /// Defaults to `2`.
    pub n_dim: usize,
    /// Number of neighbours
    /// Defaults to `15`.
    pub k: usize,
    /// Which optimiser to use: `"adam_gpu"` (default), `"adam"`, `"sgd"`,
    /// `"sgd_parallel"`, `"adam_parallel"`, or `"adam_mlx"` with the `mlx`
    /// feature.
    pub optimiser: String,
    /// Which GPU nearest neighbour search to use. One of `"exhaustive_gpu"`,
    /// `"ivf_gpu"` or `"nndescent_gpu"`. Defaults to `"nndescent_gpu"`.
    pub ann_type: String,
    /// Which embedding initialisation to use: `"spectral"` (default), `"pca"`
    /// or `"random"`.
    pub initialisation: String,
    /// Optional initialisation range
    pub init_range: Option<T>,
    /// GPU nearest neighbour parameters
    pub nn_params: NearestNeighbourParamsGpu<T>,
    /// Parameters for UMAP graph generation
    pub graph_params: UmapGraphParams<T>,
    /// Parameters for the UMAP optimiser
    pub optim_params: UmapOptimParams<T>,
    /// Use randomised SVD for PCA-based initialisation
    /// Defaults to `false`.
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
            ann_type: "nndescent_gpu".to_string(),
            initialisation: "spectral".to_string(),
            init_range: None,
            nn_params: NearestNeighbourParamsGpu::default(),
            graph_params: UmapGraphParams::default(),
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
    /// * `graph_params` - Parameters for the UMAP graph generation.
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
        graph_params: UmapGraphParams<T>,
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
            graph_params,
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
        let min_dist = min_dist.unwrap_or(T::from_f64(DEFAULT_MIN_DIST).unwrap());
        let spread = spread.unwrap_or(T::from_f64(DEFAULT_SPREAD).unwrap());

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
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
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
pub fn construct_umap_graph_gpu<T>(
    data: impl ManifoldsMatrix<T>,
    precomputed_knn: PreComputedKnn<T>,
    k: usize,
    ann_type: String,
    umap_params: &UmapGraphParams<T>,
    nn_params: &NearestNeighbourParamsGpu<T>,
    n_epochs: usize,
    device: Device,
    seed: usize,
    verbose: usize,
) -> UmapGraphResults<T>
where
    T: ManifoldsFloatGpu,
{
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    let verbosity = parse_verbosity_level(verbose);

    let (knn_indices, knn_dist) = match precomputed_knn {
        Some((indices, distances)) => {
            if verbosity.normal_verbosity() {
                println!(" Using precomputed kNN graph");
            }
            truncate_knn(indices, distances, k)
        }
        None => {
            if verbosity.normal_verbosity() {
                println!(" kNN search via {} (k = {}) ...", ann_type, k);
            }
            let start_knn = Instant::now();
            let result = run_ann_search_gpu::<T>(
                data,
                k,
                ann_type,
                nn_params,
                device,
                seed,
                ann_verbose(verbosity),
            )?;
            if verbosity.normal_verbosity() {
                println!(" ... kNN search in {:.2?}", start_knn.elapsed());
            }
            result
        }
    };

    if verbosity.normal_verbosity() {
        println!(" Building fuzzy simplicial set ...");
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
        println!(" ... graph in {:.2?}", start_graph.elapsed());
    }

    Ok((graph, knn_indices, knn_dist))
}

/// Run UMAP with GPU-accelerated nearest neighbour search
///
/// Identical to `umap` except the kNN graph is constructed on the GPU and, with
/// the default `"adam_gpu"` optimiser, the Adam update runs there too. Graph
/// construction and the initialisation stay on the CPU.
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
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
pub fn umap_gpu<T>(
    data: impl ManifoldsMatrix<T>,
    precomputed_knn: PreComputedKnn<T>,
    umap_params: &UmapParamsGpu<T>,
    device: Device,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloatGpu,
    StandardNormal: Distribution<T>,
{
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    umap_gpu_inner::<T>(
        data,
        precomputed_knn,
        umap_params,
        None,
        device,
        seed,
        verbose,
    )
}

/// Shared body of [`umap_gpu`] and [`densmap_gpu`].
///
/// Split out so the density-preserving variant is a parameter rather than a
/// second copy of the pipeline. `dens_params` is the only difference between
/// the two entry points.
///
/// ### Params
///
/// * `data` - Input data matrix (samples x features)
/// * `precomputed_knn` - Optional precomputed kNN (indices, distances)
/// * `umap_params` - GPU UMAP parameters
/// * `dens_params` - Density knobs for densMAP, or `None` for plain UMAP
/// * `device` - The GPU device to use
/// * `seed` - Random seed
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>`, `[n_dim][n_samples]`.
#[cfg(feature = "gpu")]
fn umap_gpu_inner<T>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    umap_params: &UmapParamsGpu<T>,
    dens_params: Option<DensParams<T>>,
    device: Device,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloatGpu,
    StandardNormal: Distribution<T>,
{
    let verbosity = parse_verbosity_level(verbose);
    let name = if dens_params.is_some() {
        "densMAP (GPU)"
    } else {
        "UMAP (GPU)"
    };
    let start_run = Instant::now();
    if verbosity.normal_verbosity() {
        println!(
            "Running {} on {} samples ...",
            name,
            data.nrows().separate_with_underscores()
        );
    }

    let init_type = parse_initialisation(
        &umap_params.initialisation,
        umap_params.randomised,
        umap_params.init_range,
    )
    .unwrap_or_else(|| {
        eprintln!(
            "Warning: unknown initialisation provided: {:?}. Defaulting to PCA.",
            umap_params.initialisation,
        );
        EmbdInit::PcaInit {
            range: None,
            randomised: true,
        }
    });
    let optimiser = parse_umap_optimiser_gpu(&umap_params.optimiser).unwrap_or_else(|| {
        eprintln!(
            "Warning: unknown optimiser string provided ({:?}). Defaulting to GPU-accelerated Adam",
            umap_params.optimiser
        );
        UmapOptimiserGpu::default()
    });

    if verbosity.detailed_verbosity() {
        println!(
            " alpha = {:.2?}, beta = {:.2?}",
            umap_params.optim_params.a, umap_params.optim_params.b
        );
    }

    let (graph, knn_indices, knn_dist) = construct_umap_graph_gpu::<T>(
        data,
        precomputed_knn,
        umap_params.k,
        umap_params.ann_type.clone(),
        &umap_params.graph_params,
        &umap_params.nn_params,
        umap_params.optim_params.n_epochs,
        device.clone(),
        seed,
        verbose,
    )?;

    // density state, while the kNN results are still around. `DensState::new` takes
    // true distances and squares them itself.
    let dens_state = match dens_params {
        Some(dens_params) => {
            if verbosity.normal_verbosity() {
                println!(
                    " Computing local radii (lambda = {:.3}) ...",
                    dens_params.lambda.to_f64().unwrap_or(f64::NAN)
                );
            }
            Some(DensState::new(
                dens_params,
                &graph,
                &knn_indices,
                &knn_dist,
            )?)
        }
        None => None,
    };

    // the kNN lists can be gigabytes at scale; nothing below needs them
    drop(knn_indices);
    drop(knn_dist);

    if verbosity.normal_verbosity() {
        println!(" Initialising via {} ...", umap_params.initialisation);
    }

    let start_init = Instant::now();

    let mut embd = initialise_embedding(&init_type, umap_params.n_dim, seed as u64, &graph, data)?;

    if verbosity.normal_verbosity() {
        println!(" ... initialised in {:.2?}", start_init.elapsed());
    }

    let start_layout = Instant::now();

    let graph_adj = coo_to_adjacency_list(&graph);
    let n_edges = graph.col_indices.len();
    // the optimisers only read the adjacency list; at scale the COO is GBs
    drop(graph);

    if verbosity.normal_verbosity() {
        println!(
            " Optimising via {}, {} epochs on {} edges ...",
            match optimiser {
                UmapOptimiserGpu::Adam => "Adam",
                UmapOptimiserGpu::Sgd => "SGD",
                UmapOptimiserGpu::SgdParallel => "SGD (multi-threaded, lock-free)",
                UmapOptimiserGpu::AdamParallel => "Adam (multi-threaded)",
                UmapOptimiserGpu::AdamGpu => "Adam (GPU-accelerated)",
                #[cfg(mlx_available)]
                UmapOptimiserGpu::AdamMlx => "Adam (MLX)",
            },
            umap_params.optim_params.n_epochs,
            n_edges.separate_with_underscores()
        );
    }

    match optimiser {
        UmapOptimiserGpu::Adam => optimise_embedding_adam(
            &mut embd,
            &graph_adj,
            &umap_params.optim_params,
            dens_state.as_ref(),
            seed as u64,
            verbose,
        )?,
        UmapOptimiserGpu::Sgd => {
            optimise_embedding_sgd(
                &mut embd,
                &graph_adj,
                &umap_params.optim_params,
                dens_state.as_ref(),
                seed as u64,
                verbose,
            )?;
        }
        UmapOptimiserGpu::SgdParallel => {
            optimise_embedding_sgd_parallel(
                &mut embd,
                &graph_adj,
                &umap_params.optim_params,
                dens_state.as_ref(),
                seed as u64,
                verbose,
            )?;
        }
        UmapOptimiserGpu::AdamParallel => {
            optimise_embedding_adam_parallel(
                &mut embd,
                &graph_adj,
                &umap_params.optim_params,
                dens_state.as_ref(),
                seed as u64,
                verbose,
            )?;
        }
        #[cfg(mlx_available)]
        UmapOptimiserGpu::AdamMlx => {
            optimise_embedding_adam_mlx(
                &mut embd,
                &graph_adj,
                &umap_params.optim_params,
                dens_state.as_ref(),
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
            let dens_f32 = dens_state.as_ref().map(|state| state.cast::<f32>());

            optimise_embedding_adam_gpu::<f32>(
                &mut embd_f32,
                &graph_adj_f32,
                &params_f32,
                dens_f32.as_ref(),
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
        println!(" ... optimised in {:.2?}", start_layout.elapsed());
        println!("... finished {} in {:.2?}", name, start_run.elapsed());
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

/////////////////
// densMAP GPU //
/////////////////

/// Parameters for GPU-accelerated densMAP.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct DensmapParamsGpu<T> {
    /// The underlying GPU UMAP parameters. Unchanged from plain UMAP.
    pub umap_params: UmapParamsGpu<T>,
    /// Density-preservation knobs. Defaults to [`DENSMAP_LAMBDA`].
    pub dens_params: DensParams<T>,
}

#[cfg(feature = "gpu")]
impl<T> Default for DensmapParamsGpu<T>
where
    T: ManifoldsFloat,
{
    fn default() -> Self {
        Self {
            umap_params: UmapParamsGpu::default(),
            dens_params: DensParams::densmap_default(),
        }
    }
}

#[cfg(feature = "gpu")]
impl<T> DensmapParamsGpu<T>
where
    T: ManifoldsFloat,
{
    /// Full-control constructor.
    ///
    /// ### Params
    ///
    /// * `umap_params` - The underlying GPU UMAP parameters
    /// * `dens_params` - Density-preservation knobs
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new(umap_params: UmapParamsGpu<T>, dens_params: DensParams<T>) -> Self {
        Self {
            umap_params,
            dens_params,
        }
    }

    /// 2D defaults with the knobs most worth tuning exposed.
    ///
    /// ### Params
    ///
    /// * `min_dist` - Minimum distance between embedded points. Defaults to the
    ///   UMAP default
    /// * `spread` - Effective scale of embedded points. Defaults to the UMAP
    ///   default
    /// * `lambda` - Density weight. Defaults to [`DENSMAP_LAMBDA`]
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new_default_2d(min_dist: Option<T>, spread: Option<T>, lambda: Option<T>) -> Self {
        Self {
            umap_params: UmapParamsGpu::new_default_2d(min_dist, spread),
            dens_params: DensParams::new(
                Some(lambda.unwrap_or_else(|| T::from_f64(DENSMAP_LAMBDA).unwrap())),
                None,
                None,
            ),
        }
    }
}

/// Run GPU-accelerated densMAP, the density-preserving variant of UMAP
///
/// Identical to [`umap_gpu`] except for an extra term in the gradient that
/// maximises the Pearson correlation between the log local radius of each point
/// in the original space and in the embedding.
///
/// With the default `"adam_gpu"` optimiser the density term runs on the device
/// too: one extra kernel computes the embedding radii, and a second writes the
/// per-node correlation sensitivities. Between them the radii are read back so
/// the correlation statistics can be accumulated in `f64` on the host, since
/// summing `n` log-radii in `f32` is exactly the subtract-the-mean pattern that
/// loses its significant digits. That readback is a sync point, but it only
/// happens over the final `dens_params.frac` of the epochs.
///
/// The other optimiser choices (`"sgd"`, `"sgd_parallel"`, `"adam"`,
/// `"adam_parallel"`) run the CPU density path; only the kNN search is
/// GPU-accelerated for those.
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `precomputed_knn` - Optional precomputed kNN (indices, distances)
///   excluding self.
/// * `params` - GPU densMAP parameters.
/// * `device` - The GPU device to use.
/// * `seed` - Random seed.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding coordinates as `Vec<Vec<T>>` where the outer vector has length
/// `n_dim` and the inner vectors length `n_samples`.
///
/// ### Errors
///
/// [`ManifoldsError::DegenerateLocalRadii`] if every point has the same local
/// radius, which leaves the correlation undefined. Otherwise the same errors as
/// [`umap_gpu`].
///
/// ### References
///
/// Narayan, Berger & Cho (2021): "Assessing single-cell transcriptomic
/// variability through density-preserving data visualization", Nature
/// Biotechnology
#[cfg(feature = "gpu")]
pub fn densmap_gpu<T>(
    data: impl ManifoldsMatrix<T>,
    precomputed_knn: PreComputedKnn<T>,
    params: &DensmapParamsGpu<T>,
    device: Device,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloatGpu,
    StandardNormal: Distribution<T>,
{
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    umap_gpu_inner::<T>(
        data,
        precomputed_knn,
        &params.umap_params,
        Some(params.dens_params),
        device,
        seed,
        verbose,
    )
}

//////////////
// tSNE GPU //
//////////////

/// t-SNE parameters for GPU-accelerated nearest neighbour search
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct TsneParamsGpu<T> {
    /// Number of output dimensions. Anything but 2 returns
    /// [`ManifoldsError::IncorrectDim`].
    pub n_dim: usize,
    /// Perplexity parameter controlling neighbourhood size (typical: 5-50)
    /// Defaults to `30.0`.
    pub perplexity: T,
    /// Which GPU ANN search to use. One of `"exhaustive_gpu"`, `"ivf_gpu"` or
    /// `"nndescent_gpu"`. Defaults to `"ivf_gpu"`.
    pub ann_type: String,
    /// Embedding initialisation method: `"pca"`, `"random"`, or `"spectral"`
    /// Defaults to `"pca"`.
    pub initialisation: String,
    /// Optional initialisation range
    pub init_range: Option<T>,
    /// GPU nearest neighbour parameters
    pub nn_params: NearestNeighbourParamsGpu<T>,
    /// tSNE optimisation parameters
    pub optim_params: TsneOptimParams<T>,
    /// Use randomised SVD for PCA initialisation
    /// Defaults to `true`.
    pub randomised: bool,
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
            optim_params: TsneOptimParams::default(),
            randomised: true,
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
    /// * `randomised` - Whether randomised SVD is used for PCA
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
        randomised: bool,
    ) -> Self {
        Self {
            n_dim,
            perplexity,
            ann_type,
            initialisation,
            init_range,
            nn_params,
            optim_params,
            randomised,
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
        let default = Self::default();

        Self {
            perplexity: perplexity.unwrap_or(default.perplexity),
            ..default
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
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
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
pub fn construct_tsne_graph_gpu<T>(
    data: impl ManifoldsMatrix<T>,
    precomputed_knn: PreComputedKnn<T>,
    perplexity: T,
    ann_type: String,
    nn_params: &NearestNeighbourParamsGpu<T>,
    device: Device,
    seed: usize,
    verbose: usize,
) -> TsneGraph<T>
where
    T: ManifoldsFloatGpu,
{
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    let verbosity = parse_verbosity_level(verbose);

    let (knn_indices, knn_dist) = match precomputed_knn {
        Some((indices, distances)) => {
            if verbosity.normal_verbosity() {
                println!(" Using precomputed kNN graph");
            }
            (indices, distances)
        }
        None => {
            let k_float = perplexity * T::from_f64(3.0).unwrap();
            let k = k_float.to_usize().unwrap().max(5).min(data.nrows() - 1);

            if verbosity.normal_verbosity() {
                println!(" kNN search via {} (k = {}) ...", ann_type, k);
            }

            let start_knn = Instant::now();
            let result = run_ann_search_gpu::<T>(
                data,
                k,
                ann_type,
                nn_params,
                device,
                seed,
                ann_verbose(verbosity),
            )?;

            if verbosity.normal_verbosity() {
                println!(" ... kNN search in {:.2?}", start_knn.elapsed());
            }

            result
        }
    };

    if verbosity.normal_verbosity() {
        println!(" Building Gaussian affinities ...");
    }

    let start_graph = Instant::now();

    let directed_graph = gaussian_knn_affinities(
        &knn_indices,
        &knn_dist,
        perplexity,
        T::from_f64(1e-5).unwrap(),
        200,
    )?;

    let graph = symmetrise_affinities_tsne(directed_graph);

    if verbosity.normal_verbosity() {
        println!(" ... graph in {:.2?}", start_graph.elapsed());
    }

    Ok((graph, knn_indices, knn_dist))
}

/// Parameters for GPU DREAMS. The GPU twin of [`DreamsParams`].
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct DreamsParamsGpu<T> {
    /// The underlying GPU t-SNE parameters. `initialisation` is ignored:
    /// DREAMS always starts from its reference embedding.
    pub tsne_params: TsneParamsGpu<T>,
    /// Regularisation strength in `[0, 1]`. Defaults to [`DREAMS_LAMBDA`].
    pub lambda: T,
}

#[cfg(feature = "gpu")]
impl<T> Default for DreamsParamsGpu<T>
where
    T: ManifoldsFloat,
{
    fn default() -> Self {
        Self {
            tsne_params: TsneParamsGpu::default(),
            lambda: T::from_f64(DREAMS_LAMBDA).unwrap(),
        }
    }
}

#[cfg(feature = "gpu")]
impl<T> DreamsParamsGpu<T>
where
    T: ManifoldsFloat,
{
    /// Default parameters for standard 2D visualisation.
    ///
    /// ### Params
    ///
    /// * `perplexity` - Perplexity of the underlying t-SNE. Defaults to `30.0`.
    /// * `lambda` - Regularisation strength in `[0, 1]`. Defaults to
    ///   [`DREAMS_LAMBDA`].
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new_default_2d(perplexity: Option<T>, lambda: Option<T>) -> Self {
        Self {
            tsne_params: TsneParamsGpu::new_default_2d(perplexity),
            lambda: lambda.unwrap_or_else(|| T::from_f64(DREAMS_LAMBDA).unwrap()),
        }
    }
}

#[cfg(feature = "gpu")]
impl<T> DreamsParamsGpu<T>
where
    T: ManifoldsFloat,
{
    /// Full-control constructor.
    ///
    /// ### Params
    ///
    /// * `tsne_params` - The underlying GPU t-SNE parameters
    /// * `lambda` - Regularisation strength in `[0, 1]`
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new(tsne_params: TsneParamsGpu<T>, lambda: T) -> Self {
        Self {
            tsne_params,
            lambda,
        }
    }
}

/// Run t-SNE with GPU-accelerated nearest neighbour search (FFT build)
///
/// Identical to `tsne` except the kNN graph is constructed on the GPU. The
/// optimiser stays on the CPU unless `approx_type` is `"fft_3k_gpu"`.
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `precomputed_knn` - Optional precomputed kNN, indices and distances
///   excluding self
/// * `params` - GPU t-SNE parameters
/// * `approx_type` - Repulsive-force approximation: `"barnes_hut" | "bh"`,
///   `"bh_qd" | "qd"`, `"fft"`, `"fft_3k" | "3-kernel"`, `"fft_3k_gpu"`
///   (optimiser on the GPU as well) or, with the `mlx` feature,
///   `"fft_3k_mlx"` (optimiser on MLX, kNN still on wgpu)
/// * `device` - GPU device to use
/// * `seed` - Random seed
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding as `Vec<Vec<T>>` with shape `[n_dim][n_samples]`.
#[cfg(all(feature = "gpu", feature = "fft_tsne"))]
pub fn tsne_gpu<T>(
    data: impl ManifoldsMatrix<T>,
    precomputed_knn: PreComputedKnn<T>,
    params: &TsneParamsGpu<T>,
    approx_type: &str,
    device: Device,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloatGpu + FftFloat,
    StandardNormal: Distribution<T>,
{
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    tsne_gpu_inner::<T>(
        data,
        precomputed_knn,
        params,
        approx_type,
        None,
        None,
        device,
        seed,
        verbose,
    )
}

/// Shared body of [`tsne_gpu`] and [`dreams_gpu`].
///
/// ### Params
///
/// * `data` - Input data matrix (samples × features)
/// * `precomputed_knn` - Optional precomputed kNN, indices and distances
///   excluding self
/// * `params` - GPU t-SNE parameters
/// * `approx_type` - Repulsive-force approximation, as for [`tsne_gpu`]
/// * `dreams_lambda` - DREAMS regularisation strength, or `None` for no
///   regulariser. When set, the embedding is initialised from the reference
///   and `params.initialisation` is ignored
/// * `dreams_reference` - DREAMS reference embedding `[2][n_samples]`;
///   `None` uses PCA. Ignored without `dreams_lambda`
/// * `device` - GPU device to use
/// * `seed` - Random seed
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding as `Vec<Vec<T>>` with shape `[n_dim][n_samples]`.
#[cfg(all(feature = "gpu", feature = "fft_tsne"))]
#[allow(clippy::too_many_arguments)]
fn tsne_gpu_inner<T>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    params: &TsneParamsGpu<T>,
    approx_type: &str,
    dreams_lambda: Option<T>,
    dreams_reference: Option<Vec<Vec<T>>>,
    device: Device,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloatGpu + FftFloat,
    StandardNormal: Distribution<T>,
{
    if params.n_dim != 2 {
        return Err(ManifoldsError::IncorrectDim {
            n_dim: params.n_dim,
        });
    }

    let verbosity = parse_verbosity_level(verbose);
    let name = if dreams_lambda.is_some() {
        "DREAMS (GPU)"
    } else {
        "t-SNE (GPU)"
    };
    let start_run = Instant::now();
    if verbosity.normal_verbosity() {
        println!(
            "Running {} on {} samples ...",
            name,
            data.nrows().separate_with_underscores()
        );
    }

    let (graph, _, _) = construct_tsne_graph_gpu::<T>(
        data,
        precomputed_knn,
        params.perplexity,
        params.ann_type.clone(),
        &params.nn_params,
        device.clone(),
        seed,
        verbose,
    )?;

    if verbosity.normal_verbosity() {
        println!(" Initialising via {} ...", params.initialisation);
    }

    let init_type =
        parse_initialisation(&params.initialisation, params.randomised, params.init_range)
            .unwrap_or_else(|| {
                eprintln!(
                    "Warning: unknown initialisation provided: {:?}. Defaulting to PCA.",
                    params.initialisation,
                );
                EmbdInit::PcaInit {
                    range: Some(T::from_f64(1e-2).unwrap()),
                    randomised: true,
                }
            });

    let start_init = Instant::now();

    // DREAMS initialises from its own reference, so the regulariser and the
    // starting layout agree.
    let (mut embd, dreams_state) = match dreams_lambda {
        Some(lambda) => {
            let (embd, state) = dreams_setup(
                data,
                dreams_reference,
                lambda,
                params.randomised,
                seed as u64,
            )?;
            (embd, Some(state))
        }
        None => (
            initialise_embedding(&init_type, params.n_dim, seed as u64, &graph, data)?,
            None,
        ),
    };

    if verbosity.normal_verbosity() {
        println!(" ... initialised in {:.2?}", start_init.elapsed());
    }

    let tsne_approx = parse_tsne_optimiser(approx_type).unwrap_or_else(|| {
        eprintln!(
            "Warning: unrecognised tSNE approximation provided: {:?}. Default to Barnes-Hut.",
            approx_type
        );
        TsneOpt::default()
    });

    let start_optim = Instant::now();
    match tsne_approx {
        TsneOpt::BarnesHut => {
            if verbosity.normal_verbosity() {
                println!(
                    " Optimising via Barnes-Hut t-SNE ({} epochs) ...",
                    params.optim_params.n_epochs
                );
            }
            optimise_bh_tsne(
                &mut embd,
                &params.optim_params,
                &graph,
                None,
                dreams_state.as_ref(),
                verbose,
            );
        }
        TsneOpt::BarnesHutQd => {
            if verbosity.normal_verbosity() {
                println!(
                    " Optimising via quick-and-dirty Barnes-Hut t-SNE ({} epochs, max depth {}) ...",
                    params.optim_params.n_epochs, params.optim_params.max_depth
                );
            }
            optimise_qd_tsne(
                &mut embd,
                &params.optim_params,
                &graph,
                None,
                dreams_state.as_ref(),
                verbose,
            );
        }
        TsneOpt::Fft => {
            if verbosity.normal_verbosity() {
                println!(
                    " Optimising via FFT Interpolation-based t-SNE ({} epochs) ...",
                    params.optim_params.n_epochs
                );
            }
            optimise_fft_tsne(
                &mut embd,
                &params.optim_params,
                &graph,
                None,
                dreams_state.as_ref(),
                verbose,
            )?;
        }
        TsneOpt::Fft3Kernel => {
            if verbosity.normal_verbosity() {
                println!(
                    " Optimising via three-kernel FFT Interpolation-based t-SNE ({} epochs) ...",
                    params.optim_params.n_epochs
                );
            }
            optimise_fft3k_tsne(
                &mut embd,
                &params.optim_params,
                &graph,
                None,
                dreams_state.as_ref(),
                verbose,
            )?;
        }
        TsneOpt::Fft3KernelGpu => {
            if verbosity.normal_verbosity() {
                println!(
                    " Optimising via GPU three-kernel FFT t-SNE ({} epochs) ...",
                    params.optim_params.n_epochs
                );
            }
            optimise_fft3k_tsne_gpu::<T>(
                &mut embd,
                &params.optim_params,
                &graph,
                dreams_state.as_ref(),
                device,
                verbose,
            )?;
        }
        #[cfg(mlx_available)]
        TsneOpt::Fft3KernelMlx => {
            if verbosity.normal_verbosity() {
                println!(
                    " Optimising via MLX three-kernel FFT t-SNE ({} epochs) ...",
                    params.optim_params.n_epochs
                );
            }
            optimise_fft3k_tsne_mlx(
                &mut embd,
                &params.optim_params,
                &graph,
                dreams_state.as_ref(),
                verbose,
            )?;
        }
    }

    if verbosity.normal_verbosity() {
        println!(" ... optimised in {:.2?}", start_optim.elapsed());
        println!("... finished {} in {:.2?}", name, start_run.elapsed());
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

/// Run DREAMS with GPU-accelerated nearest neighbour search
///
/// Same as [`dreams`], with the kNN graph built on the GPU. Pass
/// `approx_type = "fft_3k_gpu"` to run the optimiser, regulariser included,
/// on the GPU as well; the other approximations optimise on the CPU.
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `precomputed_knn` - Optional precomputed kNN, indices and distances
///   excluding self
/// * `reference` - Optional global reference embedding as `[2][n_samples]`.
///   `None` uses PCA.
/// * `params` - GPU DREAMS parameters
/// * `approx_type` - Repulsive-force approximation, as for [`tsne_gpu`]
/// * `device` - GPU device to use
/// * `seed` - Random seed
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding as `Vec<Vec<T>>` with shape `[n_dim][n_samples]`.
///
/// ### Errors
///
/// The DREAMS errors listed for [`dreams`], otherwise the same errors as
/// [`tsne_gpu`].
///
/// ### References
///
/// Kury, Kobak & Damrich (2026): "DREAMS: Preserving both Local and Global
/// Structure in Dimensionality Reduction", Transactions on Machine Learning
/// Research
#[cfg(all(feature = "gpu", feature = "fft_tsne"))]
#[allow(clippy::too_many_arguments)]
pub fn dreams_gpu<T>(
    data: impl ManifoldsMatrix<T>,
    precomputed_knn: PreComputedKnn<T>,
    reference: Option<Vec<Vec<T>>>,
    params: &DreamsParamsGpu<T>,
    approx_type: &str,
    device: Device,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloatGpu + FftFloat,
    StandardNormal: Distribution<T>,
{
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    tsne_gpu_inner::<T>(
        data,
        precomputed_knn,
        &params.tsne_params,
        approx_type,
        Some(params.lambda),
        reference,
        device,
        seed,
        verbose,
    )
}

/// Run t-SNE with GPU-accelerated nearest neighbour search (non-FFT build)
///
/// Identical to `tsne` except the kNN graph is constructed on the GPU.
/// Barnes-Hut optimisation stays on CPU; `"fft_3k_gpu"` runs the optimiser on
/// the GPU and needs no `fft_tsne`. Calling with `approx_type = "fft"` or
/// `"fft_3k"` returns [`ManifoldsError::TsneOptimiserNeedsFft`]; recompile with the
/// `fft_tsne` feature for CPU FFT support.
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `precomputed_knn` - Optional precomputed kNN, indices and distances
///   excluding self
/// * `params` - GPU t-SNE parameters
/// * `approx_type` - Repulsive-force approximation: `"barnes_hut" | "bh"`,
///   `"bh_qd" | "qd"`, `"fft_3k_gpu"` or, with the `mlx` feature,
///   `"fft_3k_mlx"`
/// * `device` - GPU device to use
/// * `seed` - Random seed
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding as `Vec<Vec<T>>` with shape `[n_dim][n_samples]`.
#[cfg(all(feature = "gpu", not(feature = "fft_tsne")))]
pub fn tsne_gpu<T>(
    data: impl ManifoldsMatrix<T>,
    precomputed_knn: PreComputedKnn<T>,
    params: &TsneParamsGpu<T>,
    approx_type: &str,
    device: Device,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloatGpu,
    StandardNormal: Distribution<T>,
{
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    tsne_gpu_inner::<T>(
        data,
        precomputed_knn,
        params,
        approx_type,
        None,
        None,
        device,
        seed,
        verbose,
    )
}

/// Shared body of [`tsne_gpu`] and [`dreams_gpu`].
///
/// ### Params
///
/// * `data` - Input data matrix (samples × features)
/// * `precomputed_knn` - Optional precomputed kNN, indices and distances
///   excluding self
/// * `params` - GPU t-SNE parameters
/// * `approx_type` - Repulsive-force approximation, as for [`tsne_gpu`]
/// * `dreams_lambda` - DREAMS regularisation strength, or `None` for no
///   regulariser. When set, the embedding is initialised from the reference
///   and `params.initialisation` is ignored
/// * `dreams_reference` - DREAMS reference embedding `[2][n_samples]`;
///   `None` uses PCA. Ignored without `dreams_lambda`
/// * `device` - GPU device to use
/// * `seed` - Random seed
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding as `Vec<Vec<T>>` with shape `[n_dim][n_samples]`.
#[cfg(all(feature = "gpu", not(feature = "fft_tsne")))]
#[allow(clippy::too_many_arguments)]
fn tsne_gpu_inner<T>(
    data: MatRef<T>,
    precomputed_knn: PreComputedKnn<T>,
    params: &TsneParamsGpu<T>,
    approx_type: &str,
    dreams_lambda: Option<T>,
    dreams_reference: Option<Vec<Vec<T>>>,
    device: Device,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloatGpu,
    StandardNormal: Distribution<T>,
{
    if params.n_dim != 2 {
        return Err(ManifoldsError::IncorrectDim {
            n_dim: params.n_dim,
        });
    }

    let verbosity = parse_verbosity_level(verbose);
    let name = if dreams_lambda.is_some() {
        "DREAMS (GPU)"
    } else {
        "t-SNE (GPU)"
    };
    let start_run = Instant::now();
    if verbosity.normal_verbosity() {
        println!(
            "Running {} on {} samples ...",
            name,
            data.nrows().separate_with_underscores()
        );
    }

    let (graph, _, _) = construct_tsne_graph_gpu::<T>(
        data,
        precomputed_knn,
        params.perplexity,
        params.ann_type.clone(),
        &params.nn_params,
        device.clone(),
        seed,
        verbose,
    )?;

    if verbosity.normal_verbosity() {
        println!(" Initialising via {} ...", &params.initialisation);
    }

    let init_type =
        parse_initialisation(&params.initialisation, params.randomised, params.init_range)
            .unwrap_or_else(|| {
                eprintln!(
                    "Warning: unknown initialisation provided: {:?}. Defaulting to PCA.",
                    params.initialisation,
                );
                EmbdInit::PcaInit {
                    range: Some(T::from_f64(1e-2).unwrap()),
                    randomised: true,
                }
            });

    let start_init = Instant::now();

    // DREAMS initialises from its own reference, so the regulariser and the
    // starting layout agree.
    let (mut embd, dreams_state) = match dreams_lambda {
        Some(lambda) => {
            let (embd, state) = dreams_setup(
                data,
                dreams_reference,
                lambda,
                params.randomised,
                seed as u64,
            )?;
            (embd, Some(state))
        }
        None => (
            initialise_embedding(&init_type, params.n_dim, seed as u64, &graph, data)?,
            None,
        ),
    };

    if verbosity.normal_verbosity() {
        println!(" ... initialised in {:.2?}", start_init.elapsed());
    }

    let tsne_approx = parse_tsne_optimiser(approx_type).unwrap_or_else(|| {
        eprintln!(
            "Warning: unrecognised tSNE approximation provided: {:?}. Default to Barnes-Hut.",
            approx_type
        );
        TsneOpt::default()
    });

    let start_optim = Instant::now();
    match tsne_approx {
        TsneOpt::BarnesHut => {
            if verbosity.normal_verbosity() {
                println!(
                    " Optimising via Barnes-Hut t-SNE ({} epochs) ...",
                    params.optim_params.n_epochs
                );
            }
            optimise_bh_tsne(
                &mut embd,
                &params.optim_params,
                &graph,
                None,
                dreams_state.as_ref(),
                verbose,
            );
        }
        TsneOpt::BarnesHutQd => {
            if verbosity.normal_verbosity() {
                println!(
                    " Optimising via quick-and-dirty Barnes-Hut t-SNE ({} epochs, max depth {}) ...",
                    params.optim_params.n_epochs, params.optim_params.max_depth
                );
            }
            optimise_qd_tsne(
                &mut embd,
                &params.optim_params,
                &graph,
                None,
                dreams_state.as_ref(),
                verbose,
            );
        }
        TsneOpt::Fft | TsneOpt::Fft3Kernel => {
            return Err(ManifoldsError::TsneOptimiserNeedsFft {
                name: approx_type.to_string(),
            });
        }
        TsneOpt::Fft3KernelGpu => {
            if verbosity.normal_verbosity() {
                println!(
                    " Optimising via GPU three-kernel FFT t-SNE ({} epochs) ...",
                    params.optim_params.n_epochs
                );
            }
            optimise_fft3k_tsne_gpu::<T>(
                &mut embd,
                &params.optim_params,
                &graph,
                dreams_state.as_ref(),
                device,
                verbose,
            )?;
        }
        #[cfg(mlx_available)]
        TsneOpt::Fft3KernelMlx => {
            if verbosity.normal_verbosity() {
                println!(
                    " Optimising via MLX three-kernel FFT t-SNE ({} epochs) ...",
                    params.optim_params.n_epochs
                );
            }
            optimise_fft3k_tsne_mlx(
                &mut embd,
                &params.optim_params,
                &graph,
                dreams_state.as_ref(),
                verbose,
            )?;
        }
    }

    if verbosity.normal_verbosity() {
        println!(" ... optimised in {:.2?}", start_optim.elapsed());
        println!("... finished {} in {:.2?}", name, start_run.elapsed());
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

/// Run DREAMS with GPU-accelerated nearest neighbour search
///
/// Same as [`dreams`], with the kNN graph built on the GPU. Pass
/// `approx_type = "fft_3k_gpu"` to run the optimiser, regulariser included,
/// on the GPU as well; the other approximations optimise on the CPU.
///
/// ### Params
///
/// * `data` - Input data as samples x features. Accepts a faer matrix, an
///   ndarray 2-D array (with the `ndarray` feature) or a row-major
///   `(&[T], n_samples, n_features)` tuple. See [`ManifoldsMatrix`].
/// * `precomputed_knn` - Optional precomputed kNN, indices and distances
///   excluding self
/// * `reference` - Optional global reference embedding as `[2][n_samples]`.
///   `None` uses PCA.
/// * `params` - GPU DREAMS parameters
/// * `approx_type` - Repulsive-force approximation, as for [`tsne_gpu`]
/// * `device` - GPU device to use
/// * `seed` - Random seed
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Embedding as `Vec<Vec<T>>` with shape `[n_dim][n_samples]`.
///
/// ### Errors
///
/// The DREAMS errors listed for [`dreams`], otherwise the same errors as
/// [`tsne_gpu`].
///
/// ### References
///
/// Kury, Kobak & Damrich (2026): "DREAMS: Preserving both Local and Global
/// Structure in Dimensionality Reduction", Transactions on Machine Learning
/// Research
#[cfg(all(feature = "gpu", not(feature = "fft_tsne")))]
#[allow(clippy::too_many_arguments)]
pub fn dreams_gpu<T>(
    data: impl ManifoldsMatrix<T>,
    precomputed_knn: PreComputedKnn<T>,
    reference: Option<Vec<Vec<T>>>,
    params: &DreamsParamsGpu<T>,
    approx_type: &str,
    device: Device,
    seed: usize,
    verbose: usize,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloatGpu,
    StandardNormal: Distribution<T>,
{
    let data_input = data.to_mat_input();
    let data = data_input.as_mat_ref();
    tsne_gpu_inner::<T>(
        data,
        precomputed_knn,
        &params.tsne_params,
        approx_type,
        Some(params.lambda),
        reference,
        device,
        seed,
        verbose,
    )
}
