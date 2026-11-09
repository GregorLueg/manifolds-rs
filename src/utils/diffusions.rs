//! Diffusion methods for PHATE and diffusion maps

use ann_search_rs::cpu::hnsw::{HnswIndex, HnswState};
use ann_search_rs::cpu::nndescent::{NNDescent, NNDescentQuery};
#[cfg(feature = "gpu")]
use ann_search_rs::gpu::k_means_gpu::{k_means_gpu_flat, KMeansGpuParams};
#[cfg(mlx_available)]
use ann_search_rs::mlx::k_means_mlx::k_means_clusters_mlx;
use ann_search_rs::prelude::KMeansTrainingParams;
use ann_search_rs::utils::dist::{parse_ann_dist, Dist};
use ann_search_rs::utils::k_means_utils::{assign_all_parallel, train_centroids};
use ann_search_rs::utils::nndescent_utils::ApplySortedUpdates;
use faer::{Mat, MatRef};
use faer_traits::ComplexField;
use num_traits::Float;
use rand::prelude::Distribution;
use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_distr::weighted::WeightedIndex;
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashSet};
use std::ops::{AddAssign, Range};
use std::time::Instant;

use crate::data::graph::phate_alpha_decay_affinities;
use crate::data::structures::*;
#[cfg(mlx_available)]
use crate::mlx::sparse_svd_mlx::symmetric_svd_vectors_mlx;
use crate::prelude::*;
use crate::utils::math::*;
use crate::utils::sparse_ops::*;
#[cfg(feature = "gpu")]
use crate::utils::sparse_svd_gpu::{symmetric_svd_vectors_gpu, TSMM_K_MAX};

/////////////
// Globals //
/////////////

/// Maximum iterations to use for powering the Markov transition matrix during
/// PHATE
pub const PHATE_MAX_T: usize = 100;

/// Multiple of `k + 1` that PHATE searches. The alpha-decay kernel stays above
/// its threshold for points just past the `k`-th neighbour, and in high dimensions
/// there are many of those, so a graph cut at `k` loses a good part of the kernel.
/// Matches the first search pass of the graphtools reference, without its adaptive
/// extension for rows still inside the kernel radius.
const PHATE_SEARCH_MULTIPLIER: usize = 6;

/// Default number of landmarks. Full PHATE needs an N x N dense distance matrix,
/// which stops being practical long before single cell sizes.
pub const PHATE_DEFAULT_LANDMARKS: usize = 2048;

/// Default number of singular vectors for spectral landmark selection, in PHATE
/// and diffusion maps alike.
pub const LANDMARK_DEFAULT_N_SVD: usize = 100;

/// Power iterations for spectral landmark selection. The spectrum of a kNN affinity
/// is close to flat, so two iterations leave the leading subspace unconverged, and
/// k-means then has little to separate. Each iteration costs two sparse products and
/// a QR.
const SPECTRAL_LANDMARK_POWER_ITER: usize = 7;

/// Oversampling of the randomised SVD for spectral landmark selection. Same as
/// the default of `sparse_randomised_svd`, not tuned.
const SPECTRAL_LANDMARK_OVERSAMPLING: usize = 10;

/// k-means iterations for spectral landmark selection. Fewer iterations shift the
/// diffusion time the landmark operator picks.
const SPECTRAL_LANDMARK_KMEANS_ITERS: usize = 30;

/// Shortest curve the knee search accepts. It fits a line either side of an
/// interior breakpoint, and a curve needs three values to have one.
const MIN_KNEE_CURVE_LEN: usize = 3;

///////////
// PHATE //
///////////

////////////
// Params //
////////////

/// Parameters for the diffusion process
#[derive(Debug, Clone)]
pub struct PhateDiffusionParams<T> {
    /// Decay exponent alpha (default: 40). If None, returns binary connectivity.
    pub decay: Option<T>,
    /// Multiplicative factor for bandwidth (default: 1.0)
    pub bandwidth_scale: T,
    /// Threshold below which affinities are set to 0 (default: 1e-4, for
    /// sparsity)
    pub thresh: T,
    /// symmetrisation method: "add" (default) for (K+K^T)/2, "multiply" for the
    /// element-wise product of K and K^T, "mnn" for the mutual nearest neighbour
    /// form, "none" for asymmetric.
    pub graph_symmetry: String,
    /// Number of landmarks, [`PHATE_DEFAULT_LANDMARKS`] by default. `None`, or a
    /// value of at least the number of samples, runs the full operator.
    pub n_landmarks: Option<usize>,
    /// Which landmark method to use: `"spectral"` (default), `"random"` or
    /// `"density"`. The `mlx` feature adds `"spectral_mlx"`.
    pub landmark_method: String,
    /// Singular vectors for spectral landmark selection. Defaults to
    /// [`LANDMARK_DEFAULT_N_SVD`].
    pub n_svd: Option<usize>,
    /// Enum describing how to power the matrix. Auto or user-defined.
    pub t: PhateTime,
    /// Gamma parameter to control the informational distance between data
    /// points. Between `[-1.0, 1.0]`
    pub gamma: T,
}

impl<T> Default for PhateDiffusionParams<T>
where
    T: ManifoldsFloat,
{
    /// The graphtools kernel on [`PHATE_DEFAULT_LANDMARKS`] spectral landmarks,
    /// with the diffusion time from the VNE knee.
    fn default() -> Self {
        Self {
            decay: Some(T::from_f64(40.0).unwrap()),
            bandwidth_scale: T::one(),
            thresh: T::from_f64(1e-4).unwrap(),
            graph_symmetry: "add".to_string(),
            n_landmarks: Some(PHATE_DEFAULT_LANDMARKS),
            landmark_method: "spectral".to_string(),
            n_svd: None,
            t: PhateTime::default(),
            gamma: T::one(),
        }
    }
}

impl<T> PhateDiffusionParams<T> {
    /// Generate new PhateDiffusionParams
    ///
    /// ### Params
    ///
    /// * `decay` - Decay exponent alpha. If None, returns binary connectivity.
    /// * `bandwidth_scale` - Multiplicative factor for bandwidth.
    /// * `thresh` - Threshold below which affinities are set to 0, for sparsity.
    /// * `graph_symmetry` - symmetrisation method: "add" for (K+K^T)/2, "multiply"
    ///   for the element-wise product of K and K^T, "mnn" for the mutual nearest
    ///   neighbour form, "none" for asymmetric.
    /// * `n_landmarks` - Number of landmarks. `None`, or a value of at least the
    ///   number of samples, runs the full operator.
    /// * `landmark_method` - Which landmark method to use.
    /// * `n_svd` - Singular vectors for spectral landmark selection. `None` gives
    ///   [`LANDMARK_DEFAULT_N_SVD`].
    /// * `t_max` - Largest diffusion time the VNE knee search considers. `None`
    ///   gives [`PHATE_MAX_T`].
    /// * `t_custom` - Fixed diffusion time. `None` picks the VNE knee.
    /// * `gamma` - Informational distance constant, in `[-1.0, 1.0]`. `1.0` is the
    ///   log potential.
    ///
    /// ### Returns
    ///
    /// Initialised self
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        decay: Option<T>,
        bandwidth_scale: T,
        thresh: T,
        graph_symmetry: String,
        n_landmarks: Option<usize>,
        landmark_method: String,
        n_svd: Option<usize>,
        t_max: Option<usize>,
        t_custom: Option<usize>,
        gamma: T,
    ) -> Self {
        let t_max = t_max.unwrap_or(PHATE_MAX_T);

        let t = parse_phate_time(t_custom, t_max);

        Self {
            decay,
            bandwidth_scale,
            thresh,
            graph_symmetry,
            n_landmarks,
            landmark_method,
            n_svd,
            t,
            gamma,
        }
    }
}

/// Width of the kNN search PHATE runs for a given `k`.
///
/// ### Params
///
/// * `k` - Neighbours that set the kernel bandwidth.
///
/// ### Returns
///
/// Neighbours to search for, excluding self.
pub fn phate_search_width(k: usize) -> usize {
    (k + 1) * PHATE_SEARCH_MULTIPLIER - 1
}

/// Put ones on the diagonal of a kernel, so a walker can stay where it is.
///
/// Each point sits at distance zero from itself, so this is the value the
/// alpha-decay kernel gives it.
///
/// ### Params
///
/// * `matrix` - Kernel in CSR format with sorted columns.
///
/// ### Returns
///
/// The kernel with a unit diagonal, columns still sorted.
pub fn add_unit_diagonal<T>(matrix: &CompressedSparseData<T>) -> CompressedSparseData<T>
where
    T: ManifoldsFloat,
{
    let n = matrix.nrows();
    let capacity = matrix.data.len() + n;
    let mut data = Vec::with_capacity(capacity);
    let mut indices = Vec::with_capacity(capacity);
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0);

    for i in 0..n {
        let mut placed = false;
        for idx in matrix.indptr[i]..matrix.indptr[i + 1] {
            let j = matrix.indices[idx];
            if j == i {
                continue;
            }
            if !placed && j > i {
                indices.push(i);
                data.push(T::one());
                placed = true;
            }
            indices.push(j);
            data.push(matrix.data[idx]);
        }
        if !placed {
            indices.push(i);
            data.push(T::one());
        }
        indptr.push(data.len());
    }

    CompressedSparseData::new_csr(&data, &indices, &indptr, matrix.shape)
}

/// Symmetric normalisation of a kernel: `K(x, y) / sqrt(d(x) * d(y))`.
///
/// Spectral landmark selection needs this rather than the raw kernel. The leading
/// singular vectors of a raw kernel sit on its highest degree points, so nearly
/// every other point is close to zero in that space.
///
/// ### Params
///
/// * `matrix` - Kernel in CSR format.
///
/// ### Returns
///
/// The normalised kernel, with the same sparsity pattern.
fn symmetric_normalise<T>(matrix: &CompressedSparseData<T>) -> CompressedSparseData<T>
where
    T: ManifoldsFloat,
{
    let degrees: Vec<f64> = (0..matrix.nrows())
        .into_par_iter()
        .map(|i| {
            matrix.data[matrix.indptr[i]..matrix.indptr[i + 1]]
                .iter()
                .map(|v| v.to_f64().unwrap())
                .sum()
        })
        .collect();

    let data: Vec<T> = (0..matrix.nrows())
        .into_par_iter()
        .flat_map_iter(|i| {
            let degrees = &degrees;
            (matrix.indptr[i]..matrix.indptr[i + 1]).map(move |idx| {
                let denom = (degrees[i] * degrees[matrix.indices[idx]]).sqrt();
                let scaled = if denom > 0.0 {
                    matrix.data[idx].to_f64().unwrap() / denom
                } else {
                    0.0
                };
                T::from_f64(scaled).unwrap()
            })
        })
        .collect();

    CompressedSparseData {
        data,
        indices: matrix.indices.clone(),
        indptr: matrix.indptr.clone(),
        cs_type: matrix.cs_type.clone(),
        shape: matrix.shape,
    }
}

/// Build the row-stochastic Diffusion Operator P
///
/// Computes P = D^-1 * K, where D is the degree matrix (row sums).
/// This normalises the kernel so that each row sums to 1.0, representing
/// transition probabilities.
///
/// ### Params
///
/// * `matrix` - Input Kernel matrix in CSR format
///
/// ### Returns
///
/// Row-normalised CSR matrix
pub fn build_diffusion_operator<T>(matrix: &CompressedSparseData<T>) -> CompressedSparseData<T>
where
    T: ManifoldsFloat,
{
    if !matrix.cs_type.is_csr() {
        panic!("Diffusion operator requires CSR format input");
    }
    let (rows, _) = matrix.shape();
    // calculate degrees
    let degrees: Vec<T> = (0..rows)
        .into_par_iter()
        .map(|i| {
            let start = matrix.indptr[i];
            let end = matrix.indptr[i + 1];
            let mut sum = T::zero();
            for idx in start..end {
                sum += matrix.data[idx];
            }
            sum
        })
        .collect();

    // row-normalise the data
    let norm_data: Vec<T> = (0..rows)
        .into_par_iter()
        .flat_map(|i| {
            let start = matrix.indptr[i];
            let end = matrix.indptr[i + 1];
            let deg = degrees[i];

            let mut row_vals = Vec::with_capacity(end - start);

            if deg > T::zero() {
                for idx in start..end {
                    row_vals.push(matrix.data[idx] / deg);
                }
            } else {
                // Handle disconnected nodes / zero-degree rows
                // Usually keep as zero or set self-loop to 1?
                // PHATE generally assumes a connected component or handles zeros gracefully.
                for idx in start..end {
                    row_vals.push(matrix.data[idx]);
                }
            }
            row_vals
        })
        .collect();

    CompressedSparseData::new_csr(&norm_data, &matrix.indices, &matrix.indptr, matrix.shape())
}

////////////////////////
// Landmark diffusion //
////////////////////////

///////////
// Enums //
///////////

/// Enum representing different landmark diffusion methods.
pub enum PhateDiffusion<T>
where
    T: ComplexField + Float,
{
    /// Full diffusion using all nodes.
    Full {
        /// The full operator
        operator: CompressedSparseData<T>,
    },
    /// Landmark diffusion using a subset of nodes.
    Landmark {
        /// The landmark operator
        landmarks: PhateLandmarks<T>,
    },
}

/// Enum representing different time diffusion methods.
#[derive(Debug, Clone)]
pub enum PhateTime {
    /// Find optimal via VNE (default: 100)
    Auto {
        /// Maximum number of iterations to test
        t_max: usize,
    },
    /// Use specific t
    Fixed(usize),
}

/// Default implementation for PhateTime.
impl Default for PhateTime {
    fn default() -> Self {
        PhateTime::Auto { t_max: PHATE_MAX_T }
    }
}

/// Parse the Phate t value based on the provided values
///
/// ### Params
///
/// * `t_custom` - If provided, it will use this value.
/// * `t_max` - Maximum number for auto-detection
///
/// ### Returns
///
/// PhateTime
pub fn parse_phate_time(t_custom: Option<usize>, t_max: usize) -> PhateTime {
    match t_custom {
        Some(t) => PhateTime::Fixed(t),
        None => PhateTime::Auto { t_max },
    }
}

/////////////
// Helpers //
/////////////

/// Where the SVD and k-means of PHATE's spectral landmark selection run.
#[derive(Clone, Default)]
pub enum SpectralBackend {
    /// On the CPU. The SVD runs in `f64`.
    #[default]
    Cpu,
    /// On a cubecl device, in `f32`.
    #[cfg(feature = "gpu")]
    Wgpu(cubecl::Device),
    /// On MLX, in `f32`.
    #[cfg(mlx_available)]
    Mlx,
}

/// Split a PHATE landmark method into the method and where it runs.
///
/// `"spectral_mlx"` is `"spectral"` on MLX. Without the MLX backend compiled in it
/// warns and stays on `backend`, as the other `_mlx` names fall back.
///
/// ### Params
///
/// * `method` - Landmark method as the caller named it
/// * `backend` - Backend of the entry point
///
/// ### Returns
///
/// The method to parse and the backend to run it on
pub(crate) fn resolve_landmark_backend<'a>(
    method: &'a str,
    backend: &SpectralBackend,
) -> (&'a str, SpectralBackend) {
    if !method.eq_ignore_ascii_case("spectral_mlx") {
        return (method, backend.clone());
    }
    #[cfg(mlx_available)]
    {
        ("spectral", SpectralBackend::Mlx)
    }
    #[cfg(not(mlx_available))]
    {
        eprintln!("Warning: \"spectral_mlx\" needs the MLX backend, which this build lacks. Using \"spectral\".");
        ("spectral", backend.clone())
    }
}

/// Right singular vectors for the spectral landmark selection.
///
/// ### Params
///
/// * `a_sym` - Symmetrically normalised affinity
/// * `n_svd` - Target rank
/// * `seed` - Seed for the sketch
/// * `backend` - Where to run. The wgpu path falls back to the CPU for a sketch
///   wider than its shared-memory staging holds.
///
/// ### Returns
///
/// The vectors, `n x n_svd`
fn spectral_vectors<T>(
    a_sym: &CompressedSparseData<T>,
    n_svd: usize,
    seed: usize,
    backend: &SpectralBackend,
) -> Result<Mat<f64>, ManifoldsError>
where
    T: ManifoldsFloat,
{
    let cpu = || {
        sparse_randomised_svd(
            a_sym,
            n_svd,
            seed as u64,
            Some(SPECTRAL_LANDMARK_OVERSAMPLING),
            Some(SPECTRAL_LANDMARK_POWER_ITER),
        )
        .map(|svd| svd.v)
    };

    match backend {
        SpectralBackend::Cpu => cpu(),
        #[cfg(feature = "gpu")]
        SpectralBackend::Wgpu(device) => {
            if n_svd + SPECTRAL_LANDMARK_OVERSAMPLING > TSMM_K_MAX as usize {
                return cpu();
            }
            symmetric_svd_vectors_gpu(
                a_sym,
                n_svd,
                seed as u64,
                SPECTRAL_LANDMARK_OVERSAMPLING,
                SPECTRAL_LANDMARK_POWER_ITER,
                &device.clone().client(),
            )
        }
        #[cfg(mlx_available)]
        SpectralBackend::Mlx => symmetric_svd_vectors_mlx(
            a_sym,
            n_svd,
            seed as u64,
            SPECTRAL_LANDMARK_OVERSAMPLING,
            SPECTRAL_LANDMARK_POWER_ITER,
        ),
    }
}

/// Cluster the spectral embedding into landmarks.
///
/// ### Params
///
/// * `embedding` - Row-major spectral embedding, length `n * dim`
/// * `dim` - Embedding dimensionality
/// * `n` - Number of samples
/// * `n_landmarks` - Number of clusters
/// * `seed` - Seed for the initialisation
/// * `verbosity` - Verbosity of the k-means
/// * `backend` - Where to run
///
/// ### Returns
///
/// Landmark per sample, length `n`
fn spectral_assignments<T>(
    embedding: &[T],
    dim: usize,
    n: usize,
    n_landmarks: usize,
    seed: usize,
    verbosity: Verbosity,
    backend: &SpectralBackend,
) -> Result<Vec<usize>, ManifoldsError>
where
    T: ManifoldsFloat,
{
    // the device paths run in f32 whatever `T` is
    #[cfg(any(feature = "gpu", mlx_available))]
    let to_f32 = || -> Vec<f32> { embedding.par_iter().map(|x| x.to_f32().unwrap()).collect() };

    let assignments: Vec<u32> = match backend {
        SpectralBackend::Cpu => {
            let centroids = train_centroids(
                embedding,
                dim,
                n,
                n_landmarks,
                &Dist::SquaredEuclidean,
                Some(KMeansTrainingParams::new(
                    SPECTRAL_LANDMARK_KMEANS_ITERS,
                    None,
                    None,
                )),
                seed,
                verbosity,
            )?;
            let centroid_norms = vec![T::one(); n_landmarks]; // Euclidean, unused
            let data_norms = vec![T::one(); n];
            assign_all_parallel(
                embedding,
                &data_norms,
                dim,
                n,
                &centroids,
                &centroid_norms,
                n_landmarks,
                &Dist::SquaredEuclidean,
            )
        }
        #[cfg(feature = "gpu")]
        SpectralBackend::Wgpu(device) => {
            let (_, labels) = k_means_gpu_flat::<f32>(
                &to_f32(),
                dim,
                n,
                n_landmarks,
                &Dist::SquaredEuclidean,
                Some(KMeansGpuParams::new(
                    SPECTRAL_LANDMARK_KMEANS_ITERS,
                    None,
                    true,
                    false,
                )),
                seed,
                &device.clone().client(),
                verbosity,
            )?;
            labels
        }
        #[cfg(mlx_available)]
        SpectralBackend::Mlx => {
            let data = to_f32();
            let (_, labels) = k_means_clusters_mlx(
                (data.as_slice(), n, dim),
                "euclidean",
                n_landmarks,
                Some(KMeansTrainingParams::new(
                    SPECTRAL_LANDMARK_KMEANS_ITERS,
                    None,
                    None,
                )),
                seed,
                verbosity,
            )?;
            labels
        }
    };

    Ok(assignments.into_iter().map(|c| c as usize).collect())
}

/// Enum representing different landmark diffusion methods.
#[derive(Debug, Clone)]
pub enum LandmarkMethod {
    /// Randomly select landmarks.
    Random {
        /// Seed for the randomised landmark selection
        seed: u64,
    },
    /// Use spectral clustering to select landmarks.
    Spectral {
        /// Rank of the SVD the k-means runs on
        n_svd: usize,
    },
    /// Density - leverage node degree
    Density {
        /// Seed for the randomised landmark selection
        seed: u64,
    },
}

impl Default for LandmarkMethod {
    /// Spectral selection on [`LANDMARK_DEFAULT_N_SVD`] singular vectors.
    fn default() -> Self {
        LandmarkMethod::Spectral {
            n_svd: LANDMARK_DEFAULT_N_SVD,
        }
    }
}

/// Type alias for a flattened matrix.
type FlattenData<T> = (Vec<T>, usize, usize);

/// Parse a string into a LandmarkMethod.
///
/// ### Params
///
/// * `s` - String to parse.
/// * `seed` - Optional seed for random landmark selection.
/// * `n_svd` - Optional number of singular values to use for spectral landmark
///   selection.
///
/// ### Returns
///
/// `Option<LandmarkMethod>` (which implements default if need be)
pub fn parse_landmark_method(
    s: &str,
    seed: Option<usize>,
    n_svd: Option<usize>,
) -> Option<LandmarkMethod> {
    let seed = seed.unwrap_or(42);
    let n_svd = n_svd.unwrap_or(LANDMARK_DEFAULT_N_SVD);

    match s.to_lowercase().as_str() {
        "random" => Some(LandmarkMethod::Random { seed: seed as u64 }),
        "spectral" => Some(LandmarkMethod::Spectral { n_svd }),
        "density" => Some(LandmarkMethod::Density { seed: seed as u64 }),
        _ => None,
    }
}

/// Flatten a matrix to a vector
///
/// ### Params
///
/// * `mat` - Matrix reference to flatten
///
/// ### Returns
///
/// The flatten vector
fn matrix_to_flat<T>(mat: MatRef<T>) -> FlattenData<T>
where
    T: Float,
{
    let n = mat.nrows();
    let dim = mat.ncols();

    let mut vectors_flat = Vec::with_capacity(n * dim);
    for i in 0..n {
        vectors_flat.extend(mat.row(i).iter().cloned());
    }

    (vectors_flat, n, dim)
}

/// Copy an `f64` matrix into a flat row-major vector of `T`.
///
/// ### Params
///
/// * `mat` - Matrix to copy, stored by column
///
/// ### Returns
///
/// The rows, `nrows * ncols` long.
fn rows_to_flat<T>(mat: MatRef<f64>) -> Vec<T>
where
    T: ManifoldsFloat,
{
    let k = mat.ncols();
    let mut flat = vec![T::zero(); mat.nrows() * k];
    flat.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
        for (c, out) in row.iter_mut().enumerate() {
            *out = T::from_f64(mat[(i, c)]).unwrap();
        }
    });
    flat
}

/// Randomly sample indices from a range.
///
/// ### Params
///
/// * `range` - Range of indices to sample from.
/// * `n` - Number of indices to sample.
/// * `seed` - Seed for the random number generator.
///
/// ### Returns
///
/// A vector of sampled indices.
fn random_sample(range: Range<usize>, n: usize, seed: u64) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    sample(&mut rng, range.len(), n)
        .into_iter()
        .map(|i| range.start + i)
        .collect()
}

/// Assign a data point to the nearest landmark.
///
/// ### Params
///
/// * `data` - Data point to assign.
/// * `landmark_data` - Landmark data.
/// * `n_landmark` - Number of landmarks.
/// * `dim` - Dimensionality of the data.
/// * `norm_data` - Norm of the data point, read by the cosine metric.
/// * `norm_landmark` - Norms of the landmarks, read by the cosine metric.
/// * `dist` - Distance metric. Euclidean or cosine; the entry points reject
///   Manhattan before it gets here.
///
/// ### Returns
///
/// Index of the nearest landmark.
fn assign_to_landmark<T>(
    data: &[T],
    norm_data: &T,
    landmark_data: &[T],
    norm_landmark: &[T],
    dist: &Dist,
    n_landmark: usize,
    dim: usize,
) -> usize
where
    T: ManifoldsFloat,
{
    let mut min_dist = T::infinity();
    let mut min_idx = 0;

    for idx in 0..n_landmark {
        let landmark = &landmark_data[idx * dim..(idx + 1) * dim];
        let dist = match dist {
            Dist::SquaredEuclidean => T::euclidean_simd(landmark, data),
            Dist::Cosine => {
                let dot = T::dot_simd(landmark, data);
                T::one() - (dot / (*norm_data * norm_landmark[idx]))
            }
            Dist::Manhattan => unreachable!(),
        };
        if dist < min_dist {
            min_dist = dist;
            min_idx = idx;
        }
    }

    min_idx
}

/// Sum the kernel rows of the points assigned to each landmark.
///
/// ### Params
///
/// * `kernel` - The kernel, not yet normalised, so a point weighs in by its degree.
/// * `assignments` - The assignments of points to landmarks.
/// * `n_landmarks` - The number of landmarks.
///
/// ### Returns
///
/// The landmark by point matrix, not yet normalised.
fn build_landmarks_to_data<T>(
    kernel: &CompressedSparseData<T>,
    assignments: &[usize],
    n_landmarks: usize,
) -> CompressedSparseData<T>
where
    T: ManifoldsFloat,
{
    let n = assignments.len();
    let mut landmark_pts: Vec<Vec<usize>> = vec![Vec::new(); n_landmarks];
    for (point_idx, &landmark_idx) in assignments.iter().enumerate() {
        landmark_pts[landmark_idx].push(point_idx);
    }

    let rows: Vec<SparseRow<T>> = (0..n_landmarks)
        .into_par_iter()
        .map(|landmark_idx| sparse_row_sum(kernel, &landmark_pts[landmark_idx]))
        .collect();

    sparse_row_to_csr(&rows, n)
}

/// Relabel landmark assignments so that no landmark is left without points.
///
/// K-means can leave a centroid empty. An empty landmark has no transitions, so
/// it would sit in the operator as a zero row.
///
/// ### Params
///
/// * `assignments` - Landmark label of each point, modified in place
/// * `n_landmarks` - Number of labels before compaction
///
/// ### Returns
///
/// The number of landmarks that have at least one point.
fn compact_assignments(assignments: &mut [usize], n_landmarks: usize) -> usize {
    let mut used = vec![false; n_landmarks];
    for &a in assignments.iter() {
        used[a] = true;
    }

    let mut new_label = vec![0; n_landmarks];
    let mut n_used = 0;
    for (label, &is_used) in new_label.iter_mut().zip(&used) {
        if is_used {
            *label = n_used;
            n_used += 1;
        }
    }

    for a in assignments.iter_mut() {
        *a = new_label[*a];
    }
    n_used
}

/// Find the knee point in a curve using the method from PHATE
///
/// This identifies the "elbow" where the curve transitions from steep to flat.
///
/// ### Params
///
/// * `y` - The curve values (e.g., entropy at different t)
///
/// ### Returns
///
/// Index of the knee point, or [`ManifoldsError::TMaxTooSmall`] if the curve
/// has fewer than [`MIN_KNEE_CURVE_LEN`] values.
pub fn find_knee_point<T>(y: &[T]) -> Result<usize, ManifoldsError>
where
    T: Float,
{
    let n = y.len();

    if n < MIN_KNEE_CURVE_LEN {
        return Err(ManifoldsError::TMaxTooSmall {
            t_max: n,
            min: MIN_KNEE_CURVE_LEN,
        });
    }

    // use indices as x values
    let x: Vec<T> = (0..n).map(|i| T::from(i).unwrap()).collect();

    // compute cumulative sums for linear fits
    let mut sigma_x = Vec::with_capacity(n);
    let mut sigma_y = Vec::with_capacity(n);
    let mut sigma_xy = Vec::with_capacity(n);
    let mut sigma_xx = Vec::with_capacity(n);

    let mut sum_x = T::zero();
    let mut sum_y = T::zero();
    let mut sum_xy = T::zero();
    let mut sum_xx = T::zero();

    for i in 0..n {
        sum_x = sum_x + x[i];
        sum_y = sum_y + y[i];
        sum_xy = sum_xy + x[i] * y[i];
        sum_xx = sum_xx + x[i] * x[i];

        sigma_x.push(sum_x);
        sigma_y.push(sum_y);
        sigma_xy.push(sum_xy);
        sigma_xx.push(sum_xx);
    }

    // compute forward fits (left of knee)
    let mut mfwd = Vec::with_capacity(n - 1);
    let mut bfwd = Vec::with_capacity(n - 1);

    for i in 1..n {
        let n_points = T::from(i + 1).unwrap();
        let det = n_points * sigma_xx[i] - sigma_x[i] * sigma_x[i];

        if det.abs() > T::epsilon() {
            let m = (n_points * sigma_xy[i] - sigma_x[i] * sigma_y[i]) / det;
            let b = (sigma_xx[i] * sigma_y[i] - sigma_x[i] * sigma_xy[i]) / det;
            mfwd.push(m);
            bfwd.push(b);
        } else {
            mfwd.push(T::zero());
            bfwd.push(y[0]);
        }
    }

    // compute backward fits (right of knee) by reversing
    let x_rev: Vec<T> = x.iter().rev().copied().collect();
    let y_rev: Vec<T> = y.iter().rev().copied().collect();

    let mut sigma_x_rev = Vec::with_capacity(n);
    let mut sigma_y_rev = Vec::with_capacity(n);
    let mut sigma_xy_rev = Vec::with_capacity(n);
    let mut sigma_xx_rev = Vec::with_capacity(n);

    sum_x = T::zero();
    sum_y = T::zero();
    sum_xy = T::zero();
    sum_xx = T::zero();

    for i in 0..n {
        sum_x = sum_x + x_rev[i];
        sum_y = sum_y + y_rev[i];
        sum_xy = sum_xy + x_rev[i] * y_rev[i];
        sum_xx = sum_xx + x_rev[i] * x_rev[i];

        sigma_x_rev.push(sum_x);
        sigma_y_rev.push(sum_y);
        sigma_xy_rev.push(sum_xy);
        sigma_xx_rev.push(sum_xx);
    }

    let mut mbck = Vec::with_capacity(n - 1);
    let mut bbck = Vec::with_capacity(n - 1);

    for i in 1..n {
        let n_points = T::from(i + 1).unwrap();
        let det = n_points * sigma_xx_rev[i] - sigma_x_rev[i] * sigma_x_rev[i];

        if det.abs() > T::epsilon() {
            let m = (n_points * sigma_xy_rev[i] - sigma_x_rev[i] * sigma_y_rev[i]) / det;
            let b = (sigma_xx_rev[i] * sigma_y_rev[i] - sigma_x_rev[i] * sigma_xy_rev[i]) / det;
            mbck.push(m);
            bbck.push(b);
        } else {
            mbck.push(T::zero());
            bbck.push(y_rev[0]);
        }
    }

    mbck.reverse();
    bbck.reverse();

    // Compute error for each potential breakpoint
    let mut error_curve = vec![T::infinity(); n];

    for breakpt in 1..n - 1 {
        let mut error = T::zero();

        // Error from left fit
        for i in 0..=breakpt {
            let predicted = mfwd[breakpt - 1] * x[i] + bfwd[breakpt - 1];
            error = error + (predicted - y[i]).abs();
        }

        // error from right fit
        for i in breakpt..n {
            let predicted = mbck[breakpt - 1] * x[i] + bbck[breakpt - 1];
            error = error + (predicted - y[i]).abs();
        }

        error_curve[breakpt] = error;
    }

    // find minimum error
    let mut min_idx = 1;
    let mut min_error = error_curve[1];

    for (i, &err) in error_curve.iter().enumerate().skip(1).take(n - 2) {
        if err < min_error {
            min_error = err;
            min_idx = i;
        }
    }

    Ok(min_idx)
}

////////////////////
// PhateLandmarks //
////////////////////

/// Struct representing PhateLandmarks.
#[allow(dead_code)]
pub struct PhateLandmarks<T>
where
    T: Float + ComplexField,
{
    /// Number of landmarks to select.
    n_landmarks: usize,
    /// Method to use for landmark selection.
    method: LandmarkMethod,
    /// Data point → landmark mapping.
    assignments: Vec<usize>,
    /// Small L matrix.
    landmark_op: CompressedSparseData<T>,
    /// P_nm for interpolation.
    transitions: CompressedSparseData<T>,
}

impl<T> PhateLandmarks<T>
where
    T: ManifoldsFloat,
{
    /// Build landmarks from an existing diffusion operator
    ///
    /// ### Params
    ///
    /// * `data` - Original data (N × features)
    /// * `affinity` - Original affinity matrix (N x N) - not yet normalised.
    /// * `diffusion_op` - The P matrix (N × N) already built and normalised.
    /// * `n_landmarks` - Number of landmarks to use
    /// * `method` - Random or Spectral
    /// * `distance` - Distance metric used
    /// * `seed` - Seed for random number generator
    /// * `n_svd` - Number of SVD components to use
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    ///
    /// ### Returns
    ///
    /// PhateLandmarks structure ready for powering
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        data: MatRef<T>,
        affinity: &CompressedSparseData<T>,
        diffusion_op: &CompressedSparseData<T>,
        n_landmarks: usize,
        method: &str,
        distance: &str,
        seed: usize,
        n_svd: Option<usize>,
        verbose: usize,
    ) -> Result<Self, ManifoldsError> {
        Self::build_with_backend(
            data,
            affinity,
            diffusion_op,
            n_landmarks,
            method,
            distance,
            seed,
            n_svd,
            &SpectralBackend::Cpu,
            verbose,
        )
    }

    /// [`PhateLandmarks::build`] with a choice of where the spectral landmark
    /// selection runs. Other landmark methods ignore `backend`.
    ///
    /// ### Params
    ///
    /// * `data` - Original data (N × features)
    /// * `affinity` - Original affinity matrix (N x N) - not yet normalised.
    /// * `diffusion_op` - The P matrix (N × N) already built and normalised.
    /// * `n_landmarks` - Number of landmarks to use
    /// * `method` - Random or Spectral
    /// * `distance` - Distance metric used
    /// * `seed` - Seed for random number generator
    /// * `n_svd` - Number of SVD components to use
    /// * `backend` - Where the SVD and k-means of the spectral selection run
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    ///
    /// ### Returns
    ///
    /// PhateLandmarks structure ready for powering
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_backend(
        data: MatRef<T>,
        affinity: &CompressedSparseData<T>,
        diffusion_op: &CompressedSparseData<T>,
        n_landmarks: usize,
        method: &str,
        distance: &str,
        seed: usize,
        n_svd: Option<usize>,
        backend: &SpectralBackend,
        verbose: usize,
    ) -> Result<Self, ManifoldsError> {
        let verbosity = parse_verbosity_level(verbose);

        let (data, n, dim) = matrix_to_flat(data);
        let landmark_method = parse_landmark_method(method, Some(seed), n_svd).unwrap_or_default();
        let distance = parse_ann_dist(distance).unwrap_or_default();

        let mut assignments: Vec<usize> = match landmark_method {
            LandmarkMethod::Random { seed } => {
                if verbosity.detailed_verbosity() {
                    println!("  Using random selection of landmarks")
                }

                let landmark_indices = random_sample(0..n, n_landmarks, seed);

                let landmark_data: Vec<T> = landmark_indices
                    .iter()
                    .flat_map(|&i| data[i * dim..(i + 1) * dim].iter().copied())
                    .collect();

                // calculate norms if distance is cosine
                let norm_data = match distance {
                    Dist::Cosine => (0..n)
                        .into_par_iter()
                        .map(|i| T::calculate_l2_norm(&data[i * dim..(i + 1) * dim]))
                        .collect::<Vec<_>>(),
                    Dist::SquaredEuclidean => Vec::new(),
                    Dist::Manhattan => {
                        unreachable!("PHATE Landmark does not support Manhattan distance.")
                    }
                };
                let norm_landmark = match distance {
                    Dist::Cosine => (0..n_landmarks)
                        .into_par_iter()
                        .map(|i| T::calculate_l2_norm(&landmark_data[i * dim..(i + 1) * dim]))
                        .collect::<Vec<_>>(),
                    Dist::SquaredEuclidean => Vec::new(),
                    Dist::Manhattan => {
                        unreachable!("PHATE Landmark does not support Manhattan distance.")
                    }
                };

                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let norm = if matches!(distance, Dist::SquaredEuclidean) {
                            T::zero() // unused by assign_to_landmark for Euclidean
                        } else {
                            norm_data[i]
                        };
                        assign_to_landmark(
                            &data[i * dim..(i + 1) * dim],
                            &norm,
                            &landmark_data,
                            &norm_landmark,
                            &distance,
                            n_landmarks,
                            dim,
                        )
                    })
                    .collect()
            }
            #[allow(unused_variables)]
            LandmarkMethod::Spectral { n_svd } => {
                if verbosity.detailed_verbosity() {
                    println!("  Using spectral detection of landmarks")
                }

                let v = spectral_vectors(&symmetric_normalise(affinity), n_svd, seed, backend)?;

                if verbosity.detailed_verbosity() {
                    println!("  Finished calculation of randomised SVD on the affinity matrix.")
                }

                let k = v.ncols();
                let v_rows: Vec<T> = rows_to_flat(v.as_ref());
                let mut embedding_flat: Vec<T> = vec![T::zero(); n * k];

                embedding_flat
                    .par_chunks_mut(k)
                    .enumerate()
                    .for_each(|(i, row)| {
                        for idx in diffusion_op.indptr[i]..diffusion_op.indptr[i + 1] {
                            let j = diffusion_op.indices[idx];
                            let a_val = diffusion_op.data[idx];
                            let v_row = &v_rows[j * k..(j + 1) * k];
                            for (out, &v) in row.iter_mut().zip(v_row) {
                                *out += a_val * v;
                            }
                        }
                    });

                let assignments = spectral_assignments(
                    &embedding_flat,
                    k,
                    n,
                    n_landmarks,
                    seed,
                    parse_verbosity_level(ann_verbose(verbosity)),
                    backend,
                )?;

                if verbosity.detailed_verbosity() {
                    println!("  Landmark assignments done.")
                }

                assignments
            }
            LandmarkMethod::Density { seed } => {
                if verbosity.detailed_verbosity() {
                    println!("  Using degree-weighted (density) selection of landmarks")
                }

                // calculate weights
                let weights: Vec<f64> = (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let start = affinity.indptr[i];
                        let end = affinity.indptr[i + 1];
                        let mut sum = T::zero();
                        for idx in start..end {
                            sum += affinity.data[idx];
                        }

                        // damping prevents massive clusters from hogging all
                        // the landmarks, ensuring rare branches/trajectories
                        // still get sampled. neat trick over just degree-based
                        // sampling
                        sum.to_f64().unwrap_or(0.0).sqrt()
                    })
                    .collect();

                let mut rng = StdRng::seed_from_u64(seed);
                let dist = WeightedIndex::new(&weights).expect("Failed to create weighted index. Check affinity matrix for negative/NaN values.");

                // generate exactly n_landmarks
                let mut landmark_set =
                    FxHashSet::with_capacity_and_hasher(n_landmarks, FxBuildHasher);
                while landmark_set.len() < n_landmarks {
                    landmark_set.insert(dist.sample(&mut rng));
                }

                let landmark_indices: Vec<usize> = landmark_set.into_iter().collect();

                // extract the data
                let landmark_data: Vec<T> = landmark_indices
                    .iter()
                    .flat_map(|&i| data[i * dim..(i + 1) * dim].iter().copied())
                    .collect();

                // norms
                let norm_data = match distance {
                    Dist::Cosine => (0..n)
                        .into_par_iter()
                        .map(|i| T::calculate_l2_norm(&data[i * dim..(i + 1) * dim]))
                        .collect::<Vec<_>>(),
                    Dist::SquaredEuclidean => Vec::new(),
                    Dist::Manhattan => {
                        unreachable!("PHATE Landmark does not support Manhattan distance.")
                    }
                };
                let norm_landmark = match distance {
                    Dist::Cosine => (0..n_landmarks)
                        .into_par_iter()
                        .map(|i| T::calculate_l2_norm(&landmark_data[i * dim..(i + 1) * dim]))
                        .collect::<Vec<_>>(),
                    Dist::SquaredEuclidean => Vec::new(),
                    Dist::Manhattan => {
                        unreachable!("PHATE Landmark does not support Manhattan distance.")
                    }
                };

                // assign
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let norm = if matches!(distance, Dist::SquaredEuclidean) {
                            T::zero() // unused by assign_to_landmark for Euclidean
                        } else {
                            norm_data[i]
                        };
                        assign_to_landmark(
                            &data[i * dim..(i + 1) * dim],
                            &norm,
                            &landmark_data,
                            &norm_landmark,
                            &distance,
                            n_landmarks,
                            dim,
                        )
                    })
                    .collect()
            }
        };

        let n_used = compact_assignments(&mut assignments, n_landmarks);
        if verbosity.detailed_verbosity() && n_used < n_landmarks {
            println!("  Dropped {} empty landmarks", n_landmarks - n_used);
        }

        let mut p_mn = build_landmarks_to_data(affinity, &assignments, n_used);
        let mut p_nm = p_mn.transpose();

        normalise_csr_rows_l1(&mut p_mn)?;
        normalise_csr_rows_l1(&mut p_nm)?;

        let landmark_op = csr_matmul_csr(&p_mn, &p_nm)?;

        Ok(Self {
            n_landmarks: n_used,
            method: landmark_method,
            assignments,
            landmark_op,
            transitions: p_nm,
        })
    }

    /// Power the landmark operator t times
    ///
    /// ### Params
    ///
    /// * `t` - Number of diffusion steps
    ///
    /// ### Returns
    ///
    /// L^t (n_landmarks × n_landmarks matrix)
    pub fn power(&self, t: usize) -> Result<CompressedSparseData<T>, ManifoldsError>
    where
        T: AddAssign,
    {
        matrix_power(&self.landmark_op, t)
    }

    /// Power the landmark operator t times, as a dense matrix
    ///
    /// ### Params
    ///
    /// * `t` - Number of diffusion steps
    ///
    /// ### Returns
    ///
    /// L^t (n_landmarks × n_landmarks matrix)
    pub fn power_dense(&self, t: usize) -> Result<Mat<T>, ManifoldsError> {
        dense_matrix_power(&self.landmark_op, t)
    }

    /// Compute diffusion at optimal time
    ///
    /// ### Params
    ///
    /// * `t_max` - Maximum time to search for knee point
    ///
    /// ### Returns
    ///
    /// P^t at optimal t
    pub fn power_optimal(&self, t_max: usize) -> Result<CompressedSparseData<T>, ManifoldsError>
    where
        T: AddAssign,
    {
        let t_opt = self.find_optimal_t(t_max)?;
        let res = matrix_power(&self.landmark_op, t_opt)?;
        Ok(res)
    }

    /// Interpolate landmark diffusion back to full data space
    ///
    /// Computes P^t ≈ P_nm × L^t
    ///
    /// ### Params
    ///
    /// * `landmark_diffusion` - L^t (n_landmarks × n_landmarks)
    ///
    /// ### Returns
    ///
    /// Full diffusion operator P^t (N × n_landmarks)
    pub fn interpolate(
        &self,
        landmark_diffusion: &CompressedSparseData<T>,
    ) -> Result<CompressedSparseData<T>, ManifoldsError>
    where
        T: AddAssign,
    {
        // P^t ≈ P_nm × L^t
        // (N × n_landmarks) × (n_landmarks × n_landmarks) = (N × n_landmarks)
        csr_matmul_csr(&self.transitions, landmark_diffusion)
    }

    /// Determine optimal diffusion time using Von Neumann entropy
    ///
    /// ### Params
    ///
    /// * `t_max` - Maximum time to search
    ///
    /// ### Returns
    ///
    /// Optimal t value (knee point of entropy curve), or
    /// [`ManifoldsError::TMaxTooSmall`] if `t_max` is too small for the search.
    pub fn find_optimal_t(&self, t_max: usize) -> Result<usize, ManifoldsError> {
        let entropy = von_neumann_entropy_f64(&self.landmark_op, t_max)?;
        find_knee_point(&entropy)
    }

    /// Interpolate embedding from landmark embedding
    ///
    /// ### Params
    ///
    /// * `landmark_embedding` - The embedding calculated on the landmark
    ///   transition matrix
    ///
    /// ### Returns
    ///
    /// Interpolated embedding
    pub fn interpolate_embedding(&self, landmark_embedding: &[Vec<T>]) -> Vec<Vec<T>>
    where
        T: AddAssign,
    {
        let n = self.transitions.shape().0;
        let n_dim = landmark_embedding[0].len();
        let mut embedding = vec![vec![T::zero(); n_dim]; n];

        for i in 0..n {
            let start = self.transitions.indptr[i];
            let end = self.transitions.indptr[i + 1];
            for idx in start..end {
                let l = self.transitions.indices[idx];
                let w = self.transitions.data[idx];
                for d in 0..n_dim {
                    embedding[i][d] += w * landmark_embedding[l][d];
                }
            }
        }

        embedding
    }

    /// Get the numbers of landmarks
    pub fn get_n_landmarks(&self) -> usize {
        self.n_landmarks
    }
}

///////////////////
// Diffusion map //
///////////////////

/// Apply anisotropic (alpha) normalisation to a symmetric kernel:
///
/// `K'_ij = K_ij / (q_i^alpha * q_j^alpha),  q_i = sum_j K_ij.`
///
/// alpha = 0 leaves the kernel unchanged (normalised graph Laplacian).
/// alpha = 0.5 yields a Fokker-Planck-like operator.
/// alpha = 1 yields the Laplace-Beltrami operator, invariant to sampling
/// density -> the standard choice for diffusion maps.
///
/// ### Params
///
/// * `kernel` - The kernel matrix
/// * `alpha` - The alpha parameter
///
/// ### Returns
///
/// The anisotropic normalised kernel
pub fn apply_anisotropic_normalisation<T>(
    kernel: &CompressedSparseData<T>,
    alpha: T,
) -> Result<CompressedSparseData<T>, ManifoldsError>
where
    T: ManifoldsFloat,
{
    if !kernel.cs_type.is_csr() {
        return Err(ManifoldsError::SparseMatrixIsNotCsr);
    }

    if alpha.abs() < T::epsilon() {
        return Ok(kernel.clone());
    }

    let (nrows, _) = kernel.shape();

    let q: Vec<T> = (0..nrows)
        .into_par_iter()
        .map(|i| {
            let start = kernel.indptr[i];
            let end = kernel.indptr[i + 1];
            kernel.data[start..end].iter().copied().sum()
        })
        .collect();

    let q_alpha: Vec<T> = q.par_iter().map(|&qi| qi.powf(alpha)).collect();

    let data: Vec<T> = (0..nrows)
        .into_par_iter()
        .flat_map(|i| {
            let start = kernel.indptr[i];
            let end = kernel.indptr[i + 1];
            (start..end)
                .map(|idx| {
                    let j = kernel.indices[idx];
                    let denom = q_alpha[i] * q_alpha[j];
                    if denom > T::zero() {
                        kernel.data[idx] / denom
                    } else {
                        T::zero()
                    }
                })
                .collect::<Vec<T>>()
        })
        .collect();

    Ok(CompressedSparseData::new_csr(
        &data,
        &kernel.indices,
        &kernel.indptr,
        kernel.shape(),
    ))
}

/// Build the symmetric diffusion operator P_sym = D^{-1/2} K D^{-1/2}.
///
/// P_sym shares eigenvalues with the row-stochastic operator `P = D^{-1} K`;
/// right eigenvectors of P are recovered via `phi[i] = u[i] / sqrt(d_i)`,
/// where u are the symmetric eigenvectors. That division has no answer for a
/// zero-degree point, so such a kernel is refused here.
///
/// ### Params
///
/// * `kernel` - The kernel matrix
///
/// ### Returns
///
/// `(P_sym, sqrt(degrees))`, or [`ManifoldsError::IsolatedPoints`] if any point
/// has zero degree.
pub fn build_symmetric_diffusion_operator<T>(
    kernel: &CompressedSparseData<T>,
) -> Result<(CompressedSparseData<T>, Vec<T>), ManifoldsError>
where
    T: ManifoldsFloat,
{
    if !kernel.cs_type.is_csr() {
        return Err(ManifoldsError::SparseMatrixIsNotCsr);
    }

    let (nrows, _) = kernel.shape();

    let degrees: Vec<T> = (0..nrows)
        .into_par_iter()
        .map(|i| {
            let start = kernel.indptr[i];
            let end = kernel.indptr[i + 1];
            kernel.data[start..end].iter().copied().sum()
        })
        .collect();

    let n_isolated = degrees.iter().filter(|&&d| d <= T::zero()).count();
    if n_isolated > 0 {
        return Err(ManifoldsError::IsolatedPoints {
            n_isolated,
            n_total: nrows,
        });
    }

    let sqrt_d: Vec<T> = degrees.par_iter().map(|&d| d.sqrt()).collect();

    let data: Vec<T> = (0..nrows)
        .into_par_iter()
        .flat_map(|i| {
            let start = kernel.indptr[i];
            let end = kernel.indptr[i + 1];
            let sqrt_di = sqrt_d[i];
            (start..end)
                .map(|idx| {
                    let j = kernel.indices[idx];
                    let denom = sqrt_di * sqrt_d[j];
                    if denom > T::zero() {
                        kernel.data[idx] / denom
                    } else {
                        T::zero()
                    }
                })
                .collect::<Vec<T>>()
        })
        .collect();

    let p_sym =
        CompressedSparseData::new_csr(&data, &kernel.indices, &kernel.indptr, kernel.shape());

    Ok((p_sym, sqrt_d))
}

/// For each data point, compute Gaussian-weighted transitions to its k
/// nearest landmarks and row-normalise. Used as P_nl in Nystroem extension.
///
/// Bandwidth is set to the distance to the k-th nearest landmark, so sparse
/// rows are possible if few landmarks survive the threshold cut.
///
/// ### Params
///
/// * `data` - Flat N×dim data array
/// * `n` - Number of data points
/// * `landmark_data` - Flat n_landmarks×dim landmark coordinates
/// * `n_landmarks` - Number of landmarks
/// * `dim` - Feature dimensionality
/// * `k` - Number of nearest landmarks to connect each point to
/// * `bandwidth_scale` - Multiplicative factor applied to the bandwidth
/// * `thresh` - Minimum affinity weight; entries below this are dropped
/// * `distance` - Distance metric
///
/// ### Returns
///
/// Row-stochastic CSR matrix of shape (N, n_landmarks)
#[allow(clippy::too_many_arguments)]
fn build_data_to_landmark_transitions<T>(
    data: &[T],
    n: usize,
    landmark_data: &[T],
    n_landmarks: usize,
    dim: usize,
    k: usize,
    bandwidth_scale: T,
    thresh: T,
    distance: &Dist,
) -> CompressedSparseData<T>
where
    T: ManifoldsFloat,
{
    let k_used = k.min(n_landmarks).max(1);
    let machine_epsilon = T::epsilon();

    let data_norms: Vec<T> = match distance {
        Dist::Cosine => (0..n)
            .into_par_iter()
            .map(|i| T::calculate_l2_norm(&data[i * dim..(i + 1) * dim]))
            .collect(),
        Dist::SquaredEuclidean => Vec::new(),
        Dist::Manhattan => unreachable!(),
    };
    let landmark_norms: Vec<T> = match distance {
        Dist::Cosine => (0..n_landmarks)
            .into_par_iter()
            .map(|l| T::calculate_l2_norm(&landmark_data[l * dim..(l + 1) * dim]))
            .collect(),
        Dist::SquaredEuclidean => Vec::new(),
        Dist::Manhattan => unreachable!(),
    };

    let bw_factor = match distance {
        Dist::SquaredEuclidean => bandwidth_scale * bandwidth_scale,
        Dist::Cosine => bandwidth_scale,
        Dist::Manhattan => unreachable!(),
    };

    let rows: Vec<(Vec<usize>, Vec<T>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let x = &data[i * dim..(i + 1) * dim];

            let mut dists: Vec<(usize, T)> = (0..n_landmarks)
                .map(|l| {
                    let y = &landmark_data[l * dim..(l + 1) * dim];
                    let d = match distance {
                        Dist::SquaredEuclidean => T::euclidean_simd(x, y),
                        Dist::Cosine => {
                            let dot = T::dot_simd(x, y);
                            let denom = data_norms[i] * landmark_norms[l];
                            if denom > T::zero() {
                                T::one() - (dot / denom)
                            } else {
                                T::zero()
                            }
                        }
                        Dist::Manhattan => unreachable!(),
                    };
                    (l, d)
                })
                .collect();

            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.truncate(k_used);

            let bandwidth_raw = dists.last().map(|(_, d)| *d).unwrap_or(T::zero());
            let bandwidth = (bandwidth_raw * bw_factor).max(machine_epsilon);

            let mut kept: Vec<(usize, T)> = dists
                .into_iter()
                .filter_map(|(l, d)| {
                    let w = match distance {
                        Dist::SquaredEuclidean => (-(d / bandwidth)).exp(),
                        Dist::Cosine => {
                            let scaled = d / bandwidth;
                            (-(scaled * scaled)).exp()
                        }
                        Dist::Manhattan => unreachable!(),
                    };
                    if w >= thresh {
                        Some((l, w))
                    } else {
                        None
                    }
                })
                .collect();

            let sum: T = kept.iter().map(|(_, w)| *w).sum();
            if sum > T::zero() {
                for (_, w) in &mut kept {
                    *w /= sum;
                }
            }

            kept.sort_by_key(|(l, _)| *l);
            let indices: Vec<usize> = kept.iter().map(|(l, _)| *l).collect();
            let values: Vec<T> = kept.iter().map(|(_, w)| *w).collect();
            (indices, values)
        })
        .collect();

    let total_nnz: usize = rows.iter().map(|(i, _)| i.len()).sum();
    let mut out_data = Vec::with_capacity(total_nnz);
    let mut out_indices = Vec::with_capacity(total_nnz);
    let mut out_indptr = Vec::with_capacity(n + 1);
    out_indptr.push(0);

    for (idx, vals) in rows {
        out_data.extend(vals);
        out_indices.extend(idx);
        out_indptr.push(out_data.len());
    }

    CompressedSparseData::new_csr(&out_data, &out_indices, &out_indptr, (n, n_landmarks))
}

/// Landmark diffusion maps operator.
///
/// Builds a symmetric Gaussian kernel on landmark coordinates, applies
/// anisotropic normalisation, constructs P_sym on landmarks, and stores a
/// row-stochastic data-to-landmark transition matrix for Nystroem extension.
///
/// Use `eigendecompose` → `compute_landmark_embedding` → `nystrom_extend`
/// in sequence to obtain a full-data diffusion embedding.
#[allow(dead_code)]
pub struct DiffusionMapsLandmarks<T>
where
    T: Float + ComplexField,
{
    /// Number of landmarks selected
    n_landmarks: usize,
    /// Landmark selection method used
    method: LandmarkMethod,
    /// Indices into the original N data points identifying each landmark
    landmark_indices: Vec<usize>,
    /// Symmetric L×L diffusion operator on landmarks: D^{-1/2} K D^{-1/2}
    p_sym_ll: CompressedSparseData<T>,
    /// Square-root of landmark node degrees; used to recover right eigenvectors
    /// of the row-stochastic operator from the symmetric eigenvectors
    sqrt_degrees_l: Vec<T>,
    /// Row-stochastic N×L transition matrix for Nystroem extension
    p_nl: CompressedSparseData<T>,
}

impl<T> DiffusionMapsLandmarks<T>
where
    T: ManifoldsFloat,
{
    /// Build the landmark diffusion maps operator.
    ///
    /// Selects landmarks, constructs a symmetric Gaussian kernel on them,
    /// applies anisotropic normalisation, and builds the Nystroem transition
    /// matrix. The `affinity` matrix is only used for `"spectral"` and
    /// `"density"` landmark selection; `"random"` ignores it.
    ///
    /// ### Params
    ///
    /// * `data` - Original N×features data matrix
    /// * `affinity` - Full N×N Gaussian kernel in CSR format
    /// * `n_landmarks` - Number of landmarks to select
    /// * `method` - Landmark selection method: `"random"`, `"spectral"`, or
    ///   `"density"`
    /// * `distance` - Distance metric: `"euclidean"` or `"cosine"`
    /// * `alpha_norm` - Anisotropic normalisation exponent (0 = none,
    ///   1 = Laplace-Beltrami)
    /// * `k` - Number of nearest neighbours for graph construction and Nystroem
    ///   transitions
    /// * `bandwidth_scale` - Multiplicative factor applied to the kernel
    ///   bandwidth
    /// * `thresh` - Minimum affinity; entries below this are dropped for
    ///   sparsity
    /// * `graph_symmetry` - Symmetrisation method: `"add"`, `"multiply"`,
    ///   `"mnn"` or `"none"`
    /// * `seed` - RNG seed
    /// * `n_svd` - Number of SVD components for spectral landmark selection
    /// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for
    ///   detailed verbosity.
    ///
    /// ### Returns
    ///
    /// Initialised `DiffusionMapsLandmarks`
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        data: MatRef<T>,
        affinity: Option<&CompressedSparseData<T>>,
        n_landmarks: usize,
        method: &str,
        distance: &str,
        alpha_norm: T,
        k: usize,
        bandwidth_scale: T,
        thresh: T,
        graph_symmetry: &str,
        seed: usize,
        n_svd: Option<usize>,
        verbose: usize,
    ) -> Result<Self, ManifoldsError>
    where
        HnswIndex<T>: HnswState<T>,
        NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
    {
        let verbosity = parse_verbosity_level(verbose);

        let start = Instant::now();

        let (data_flat, n, dim) = matrix_to_flat(data);
        let landmark_method = parse_landmark_method(method, Some(seed), n_svd).unwrap_or_default();
        let dist_enum = parse_ann_dist(distance).unwrap_or_default();

        // landmark selection
        let landmark_indices: Vec<usize> = match landmark_method {
            LandmarkMethod::Random { seed } => {
                if verbosity.detailed_verbosity() {
                    println!("  Using random landmark selection");
                }
                random_sample(0..n, n_landmarks, seed)
            }
            LandmarkMethod::Density { seed } => {
                if verbosity.detailed_verbosity() {
                    println!("  Using density landmark selection");
                }
                let affinity = affinity.expect("density landmarks require affinity matrix");
                let weights: Vec<f64> = (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let start = affinity.indptr[i];
                        let end = affinity.indptr[i + 1];
                        let mut sum = T::zero();
                        for idx in start..end {
                            sum += affinity.data[idx];
                        }
                        sum.to_f64().unwrap_or(0.0).sqrt()
                    })
                    .collect();
                let mut rng = StdRng::seed_from_u64(seed);
                let weighted = WeightedIndex::new(&weights)
                    .expect("Failed to create weighted index for density landmarks");
                let mut set = FxHashSet::with_capacity_and_hasher(n_landmarks, FxBuildHasher);
                while set.len() < n_landmarks {
                    set.insert(weighted.sample(&mut rng));
                }
                set.into_iter().collect()
            }
            LandmarkMethod::Spectral { n_svd } => {
                if verbosity.detailed_verbosity() {
                    println!("  Using spectral landmark selection");
                }
                let affinity = affinity.expect("spectral landmarks require affinity matrix");
                let svd = sparse_randomised_svd(
                    &symmetric_normalise(affinity),
                    n_svd,
                    seed as u64,
                    None,
                    Some(SPECTRAL_LANDMARK_POWER_ITER),
                )?;
                // project with P, not the kernel: a point's position must not
                // scale with its degree, or k-means splits on degree
                let projector = build_diffusion_operator(affinity);
                let v = &svd.v;
                let k_proj = v.ncols();
                let mut embedding_flat: Vec<T> = vec![T::zero(); n * k_proj];

                for i in 0..n {
                    for idx in projector.indptr[i]..projector.indptr[i + 1] {
                        let j = projector.indices[idx];
                        let a_val = projector.data[idx];
                        for col in 0..k_proj {
                            embedding_flat[i * k_proj + col] +=
                                a_val * T::from_f64(v[(j, col)]).unwrap();
                        }
                    }
                }

                let k_means_params = KMeansTrainingParams::new(30, None, None);

                let centroids = train_centroids(
                    &embedding_flat,
                    k_proj,
                    n,
                    n_landmarks,
                    &Dist::SquaredEuclidean,
                    Some(k_means_params),
                    seed,
                    parse_verbosity_level(ann_verbose(verbosity)),
                )?;
                let centroid_norms = vec![T::one(); n_landmarks];
                let data_norms = vec![T::one(); n];
                let assignments = assign_all_parallel(
                    &embedding_flat,
                    &data_norms,
                    k_proj,
                    n,
                    &centroids,
                    &centroid_norms,
                    n_landmarks,
                    &Dist::SquaredEuclidean,
                );

                // for each cluster, pick the point closest to its centroid
                let mut indices: Vec<usize> = Vec::with_capacity(n_landmarks);
                for cluster_idx in 0..n_landmarks {
                    let centroid = &centroids[cluster_idx * k_proj..(cluster_idx + 1) * k_proj];
                    let mut best: Option<(usize, T)> = None;
                    for (point_idx, &c) in assignments.iter().enumerate() {
                        if c as usize == cluster_idx {
                            let point =
                                &embedding_flat[point_idx * k_proj..(point_idx + 1) * k_proj];
                            let d = T::euclidean_simd(centroid, point);
                            if best.is_none_or(|(_, bd)| d < bd) {
                                best = Some((point_idx, d));
                            }
                        }
                    }
                    if let Some((idx, _)) = best {
                        if !indices.contains(&idx) {
                            indices.push(idx);
                        }
                    }
                }

                // fill remainder (empty clusters or duplicate picks) with
                // random
                if indices.len() < n_landmarks {
                    let mut rng = StdRng::seed_from_u64(seed as u64 + 1);
                    let mut remaining: Vec<usize> =
                        (0..n).filter(|i| !indices.contains(i)).collect();
                    remaining.shuffle(&mut rng);
                    while indices.len() < n_landmarks {
                        if let Some(pick) = remaining.pop() {
                            indices.push(pick);
                        } else {
                            break;
                        }
                    }
                }
                indices
            }
        };

        // get coordinates
        let landmark_data: Vec<T> = landmark_indices
            .iter()
            .flat_map(|&i| data_flat[i * dim..(i + 1) * dim].iter().copied())
            .collect();

        // landmark-landmark gaussian kernel via self-kNN on landmarks
        let landmark_mat =
            faer::Mat::<T>::from_fn(n_landmarks, dim, |i, j| landmark_data[i * dim + j]);
        let k_ll = k.min(n_landmarks.saturating_sub(1)).max(1);
        let nn_params = NearestNeighbourParams {
            dist_metric: distance.to_string(),
            ..Default::default()
        };
        let (ll_indices, ll_dists) = run_ann_search(
            landmark_mat.as_ref(),
            k_ll,
            "kmknn".to_string(),
            &nn_params,
            seed,
            ann_verbose(verbosity),
        )?;
        // with this config it behaves like a Gaussian
        let ll_graph = phate_alpha_decay_affinities(
            &ll_indices,
            &ll_dists,
            k_ll,
            Some(T::from_f64(2.0).unwrap()),
            bandwidth_scale,
            thresh,
            graph_symmetry,
        );
        let kernel_ll = coo_to_csr(&ll_graph);
        let kernel_ll_norm = apply_anisotropic_normalisation(&kernel_ll, alpha_norm)?;
        let (p_sym_ll, sqrt_degrees_l) = build_symmetric_diffusion_operator(&kernel_ll_norm)?;

        // data-to-landmark transitions (for Nystroem)
        let p_nl = build_data_to_landmark_transitions(
            &data_flat,
            n,
            &landmark_data,
            n_landmarks,
            dim,
            k,
            bandwidth_scale,
            thresh,
            &dist_enum,
        );

        if verbosity.detailed_verbosity() {
            println!("  Finished landmark generation in {:.2?}", start.elapsed());
        }

        Ok(Self {
            n_landmarks,
            method: landmark_method,
            landmark_indices,
            p_sym_ll,
            sqrt_degrees_l,
            p_nl,
        })
    }

    /// Compute the top `n_dim + 1` eigenpairs of the symmetric landmark
    /// operator via Lanczos.
    ///
    /// Asks for `n_dim + 1` pairs so the caller can discard the trivial first
    /// eigenvalue (λ₀ ≈ 1) and retain `n_dim` meaningful components.
    ///
    /// ### Params
    ///
    /// * `n_dim` - Number of diffusion components required (excluding the
    ///   trivial one)
    /// * `seed` - RNG seed for the Lanczos starting vector
    ///
    /// ### Returns
    ///
    /// `(eigenvalues, eigenvectors)` both in f64; eigenvectors indexed as
    /// `evecs[point][component]`.
    ///
    /// Fewer than `n_dim + 1` pairs come back when the landmark operator has
    /// too few distinct eigenvalues for the Lanczos recurrence to fill the
    /// requested Krylov space, which degenerate landmark sets (duplicates, a
    /// disconnected landmark graph) do hit. Callers must check the width
    /// before indexing up to `n_dim`.
    pub fn eigendecompose(
        &self,
        n_dim: usize,
        seed: u64,
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>), ManifoldsError> {
        compute_largest_eigenpairs_lanczos(&self.p_sym_ll, n_dim + 1, seed)
    }

    /// Select the optimal diffusion time via the Von Neumann entropy knee on
    /// the landmark operator.
    ///
    /// ### Params
    ///
    /// * `t_max` - Maximum number of diffusion steps to evaluate
    ///
    /// ### Returns
    ///
    /// Optimal `t` (knee point of the entropy curve), or
    /// [`ManifoldsError::TMaxTooSmall`] if `t_max` is too small for the search.
    pub fn find_optimal_t(&self, t_max: usize) -> Result<usize, ManifoldsError> {
        let entropy = landmark_von_neumann_entropy(&self.p_sym_ll, t_max)?;
        find_knee_point(&entropy)
    }

    /// Compute diffusion coordinates on the landmarks: y_k(l) = λ_k^t · φ_k(l).
    ///
    /// Right eigenvectors of the row-stochastic operator are recovered as
    /// φ_k(l) = u_k(l) / sqrt(d_l), where u_k are eigenvectors of P_sym.
    /// The trivial first component is dropped; sign is fixed to the
    /// largest-magnitude element per component.
    ///
    /// ### Params
    ///
    /// * `evals` - Eigenvalues from `eigendecompose` (length n_dim + 1)
    /// * `evecs` - Eigenvectors from `eigendecompose`, indexed as
    ///   `evecs[point][component]`
    /// * `n_dim` - Number of diffusion dimensions to retain
    /// * `t` - Diffusion time (number of steps to raise eigenvalues to)
    ///
    /// ### Returns
    ///
    /// `(landmark_embedding, lambdas)` where `landmark_embedding` is L×n_dim
    /// and `lambdas` holds the n_dim eigenvalues used (needed for Nystroem
    /// extension)
    pub fn compute_landmark_embedding(
        &self,
        evals: &[f64],
        evecs: &[Vec<f64>],
        n_dim: usize,
        t: usize,
    ) -> (Vec<Vec<T>>, Vec<T>) {
        let mut embedding: Vec<Vec<T>> = vec![vec![T::zero(); n_dim]; self.n_landmarks];
        let mut lambdas: Vec<T> = Vec::with_capacity(n_dim);

        for comp_idx in 1..=n_dim {
            let lambda = T::from_f64(evals[comp_idx]).unwrap();
            let lambda_t = lambda.powi(t as i32);
            lambdas.push(lambda);

            let mut max_abs = 0.0f64;
            let mut sign = 1.0f64;
            for l in 0..self.n_landmarks {
                let v = evecs[l][comp_idx];
                if v.abs() > max_abs {
                    max_abs = v.abs();
                    sign = if v >= 0.0 { 1.0 } else { -1.0 };
                }
            }

            for l in 0..self.n_landmarks {
                let u = T::from_f64(evecs[l][comp_idx] * sign).unwrap();
                embedding[l][comp_idx - 1] = lambda_t * u / self.sqrt_degrees_l[l];
            }
        }

        (embedding, lambdas)
    }

    /// Extend the landmark embedding to all N points via Nystroem
    /// approximation.
    ///
    /// Computes y_k(x) = (1/λ_k) · Σ_l P_nl(x, l) · y_k(l). Components
    /// whose eigenvalue is near zero are left as zero rather than divided.
    ///
    /// ### Params
    ///
    /// * `landmark_embedding` - L×n_dim embedding from
    ///   `compute_landmark_embedding`
    /// * `lambdas` - Per-component eigenvalues from
    ///   `compute_landmark_embedding`
    ///
    /// ### Returns
    ///
    /// Full N×n_dim diffusion embedding
    pub fn nystrom_extend(&self, landmark_embedding: &[Vec<T>], lambdas: &[T]) -> Vec<Vec<T>> {
        let n = self.p_nl.shape().0;
        let n_dim = landmark_embedding[0].len();
        let mut out = vec![vec![T::zero(); n_dim]; n];

        for i in 0..n {
            let start = self.p_nl.indptr[i];
            let end = self.p_nl.indptr[i + 1];
            for idx in start..end {
                let l = self.p_nl.indices[idx];
                let w = self.p_nl.data[idx];
                for d in 0..n_dim {
                    out[i][d] += w * landmark_embedding[l][d];
                }
            }
        }

        for i in 0..n {
            for d in 0..n_dim {
                if lambdas[d].abs() > T::epsilon() {
                    out[i][d] /= lambdas[d];
                }
            }
        }

        out
    }

    /// Get the number of landmarks
    ///
    /// ### Returns
    ///
    /// Number of landmarks
    pub fn get_n_landmarks(&self) -> usize {
        self.n_landmarks
    }

    /// Get the landmark indices
    ///
    /// ### Return
    ///
    /// The indices of the landmarks
    pub fn get_landmark_indices(&self) -> &[usize] {
        &self.landmark_indices
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_landmark_labels {
    use super::*;

    #[test]
    fn test_compact_assignments_drops_empty_labels() {
        let mut labels = vec![4, 0, 4, 2];
        let n_used = compact_assignments(&mut labels, 6);
        assert_eq!(n_used, 3);
        assert_eq!(labels, vec![2, 0, 2, 1]);
    }

    #[test]
    fn test_compact_assignments_keeps_full_labels() {
        let mut labels = vec![1, 0, 2];
        assert_eq!(compact_assignments(&mut labels, 3), 3);
        assert_eq!(labels, vec![1, 0, 2]);
    }
}

#[cfg(test)]
mod test_data_diffusion {
    use super::*;

    #[test]
    fn test_diffusion_operator_normalization() {
        // Create a CSR matrix manually (or via conversion)
        // Row 0: [10.0, 30.0] -> Sum = 40.0
        // Row 1: [5.0]        -> Sum = 5.0
        // Row 2: [0.0]        -> Sum = 0.0 (Empty)

        let data = vec![10.0, 30.0, 5.0];
        let indices = vec![1, 2, 0];
        let indptr = vec![0, 2, 3, 3]; // Row 0 (0..2), Row 1 (2..3), Row 2 (3..3)

        let kernel = CompressedSparseData::new_csr(&data, &indices, &indptr, (3, 3));

        let diffusion_op = build_diffusion_operator(&kernel);

        // Check Row 0: 10/40 = 0.25, 30/40 = 0.75
        assert_eq!(diffusion_op.data[0], 0.25);
        assert_eq!(diffusion_op.data[1], 0.75);

        // Check Row 1: 5/5 = 1.0
        assert_eq!(diffusion_op.data[2], 1.0);

        // Check Row Sums
        let row0_sum = diffusion_op.data[0] + diffusion_op.data[1];
        assert!((row0_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_diffusion_operator_zero_degree_safety() {
        // Test ensuring that a row with all zeros (or empty) doesn't cause NaN
        // Row 0: Empty
        // Row 1: [2.0, 2.0]

        let data = vec![2.0, 2.0];
        let indices = vec![0, 2];
        let indptr = vec![0, 0, 2];

        let kernel = CompressedSparseData::new_csr(&data, &indices, &indptr, (2, 3));
        let diffusion_op = build_diffusion_operator(&kernel);

        // Row 0 is empty, indptr should remain [0, 0, ...]
        assert_eq!(diffusion_op.indptr[0], 0);
        assert_eq!(diffusion_op.indptr[1], 0);

        // Row 1 should be normalized
        assert_eq!(diffusion_op.data[0], 0.5);
        assert_eq!(diffusion_op.data[1], 0.5);
    }

    #[test]
    fn test_anisotropic_alpha_zero_is_identity() {
        let data = vec![0.5, 0.3, 0.2, 0.7];
        let indices = vec![0, 1, 0, 1];
        let indptr = vec![0, 2, 4];
        let kernel = CompressedSparseData::new_csr(&data, &indices, &indptr, (2, 2));

        let result = apply_anisotropic_normalisation(&kernel, 0.0).unwrap();
        assert_eq!(result.data, kernel.data);
        assert_eq!(result.indices, kernel.indices);
        assert_eq!(result.indptr, kernel.indptr);
    }

    #[test]
    fn test_anisotropic_alpha_one_values() {
        // K = [[1.0, 2.0], [2.0, 1.0]], symmetric
        // q = [3.0, 3.0]; with alpha = 1: K'_ij = K_ij / (3 * 3) = K_ij / 9
        let data = vec![1.0, 2.0, 2.0, 1.0];
        let indices = vec![0, 1, 0, 1];
        let indptr = vec![0, 2, 4];
        let kernel = CompressedSparseData::new_csr(&data, &indices, &indptr, (2, 2));

        let result = apply_anisotropic_normalisation(&kernel, 1.0).unwrap();
        use approx::assert_relative_eq;
        assert_relative_eq!(result.data[0], 1.0 / 9.0, epsilon = 1e-10);
        assert_relative_eq!(result.data[1], 2.0 / 9.0, epsilon = 1e-10);
        assert_relative_eq!(result.data[2], 2.0 / 9.0, epsilon = 1e-10);
        assert_relative_eq!(result.data[3], 1.0 / 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_anisotropic_preserves_symmetry() {
        let data = vec![1.0, 0.5, 0.5, 1.0, 0.3, 0.3, 1.0];
        let indices = vec![0, 1, 0, 1, 2, 1, 2];
        let indptr = vec![0, 2, 5, 7];
        let kernel = CompressedSparseData::new_csr(&data, &indices, &indptr, (3, 3));

        let result = apply_anisotropic_normalisation(&kernel, 0.5).unwrap();
        let dense = result.to_dense();
        use approx::assert_relative_eq;
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(dense[(i, j)], dense[(j, i)], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_symmetric_operator_is_symmetric() {
        let data = vec![1.0, 0.5, 0.5, 1.0, 0.3, 0.3, 1.0];
        let indices = vec![0, 1, 0, 1, 2, 1, 2];
        let indptr = vec![0, 2, 5, 7];
        let kernel = CompressedSparseData::new_csr(&data, &indices, &indptr, (3, 3));

        let (p_sym, _) = build_symmetric_diffusion_operator(&kernel).unwrap();
        let dense = p_sym.to_dense();
        use approx::assert_relative_eq;
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(dense[(i, j)], dense[(j, i)], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_symmetric_operator_sqrt_degrees() {
        let data = vec![1.0, 0.5, 0.5, 1.0, 0.3, 0.3, 1.0];
        let indices = vec![0, 1, 0, 1, 2, 1, 2];
        let indptr = vec![0, 2, 5, 7];
        let kernel = CompressedSparseData::new_csr(&data, &indices, &indptr, (3, 3));

        let (_, sqrt_d) = build_symmetric_diffusion_operator(&kernel).unwrap();
        use approx::assert_relative_eq;
        assert_relative_eq!(sqrt_d[0], 1.5_f64.sqrt(), epsilon = 1e-10);
        assert_relative_eq!(sqrt_d[1], 1.8_f64.sqrt(), epsilon = 1e-10);
        assert_relative_eq!(sqrt_d[2], 1.3_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_symmetric_operator_largest_eigenvalue_is_one() {
        // For any row-stochastic P = D^{-1} K on a connected graph,
        // P_sym = D^{-1/2} K D^{-1/2} has largest eigenvalue = 1.
        use crate::utils::math::compute_largest_eigenpairs_lanczos;

        let data = vec![1.0, 0.5, 0.5, 1.0, 0.3, 0.3, 1.0];
        let indices = vec![0, 1, 0, 1, 2, 1, 2];
        let indptr = vec![0, 2, 5, 7];
        let kernel = CompressedSparseData::new_csr(&data, &indices, &indptr, (3, 3));

        let (p_sym, _) = build_symmetric_diffusion_operator(&kernel).unwrap();
        let (evals, _) = compute_largest_eigenpairs_lanczos(&p_sym, 1, 42).unwrap();
        use approx::assert_relative_eq;
        assert_relative_eq!(evals[0], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_symmetric_operator_refuses_isolated_point() {
        // the last row is empty, so point 2 has zero degree
        let data = vec![1.0, 0.5, 0.5, 1.0];
        let indices = vec![0, 1, 0, 1];
        let indptr = vec![0, 2, 4, 4];
        let kernel = CompressedSparseData::new_csr(&data, &indices, &indptr, (3, 3));

        assert!(matches!(
            build_symmetric_diffusion_operator(&kernel),
            Err(ManifoldsError::IsolatedPoints {
                n_isolated: 1,
                n_total: 3
            })
        ));
    }

    #[test]
    fn test_find_knee_point_curve_length() {
        for n in 0..MIN_KNEE_CURVE_LEN {
            assert!(matches!(
                find_knee_point(&vec![1.0_f64; n]),
                Err(ManifoldsError::TMaxTooSmall { t_max, min: 3 }) if t_max == n
            ));
        }
        assert_eq!(find_knee_point(&[3.0_f64, 2.0, 1.0]).unwrap(), 1);
    }
}
