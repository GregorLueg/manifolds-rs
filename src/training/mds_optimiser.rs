use ann_search_rs::utils::dist::{Dist, SimdDistance};
use faer::Mat;
use faer_traits::{ComplexField, RealField};
use num_traits::{Float, FromPrimitive};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use thousands::*;

use crate::data::structures::CompressedSparseData;
use crate::utils::math::randomised_svd;
use crate::utils::sparse_ops::csr_row_to_dense;

/////////////
// Globals //
/////////////

pub const DEFAULT_LR: f64 = 0.01;

///////////
// Enums //
///////////

/// Represents the different methods for performing multi-dimensional scaling
/// (MDS).
#[derive(Default, Clone, Debug)]
pub enum MdsMethod {
    /// Works on a dense distance matrix of size n x n.
    #[default]
    SgdDense,
    /// Generates the distance calculations on the fly to avoid materialising
    /// the n x n distance matrix.
    SgdStreaming,
    /// Uses the classic MDS algorithm.
    ClassicMds,
}

/// Parses a string into an MdsMethod.
///
/// ### Params
///
/// * `s` - The string to parse.
///
/// ### Returns
///
/// The Option of MdsMethod
pub fn parse_mds_method(s: &str) -> Option<MdsMethod> {
    match s.to_lowercase().as_str() {
        "sgd_dense" | "dense" => Some(MdsMethod::SgdDense),
        "sgd_streaming" | "streaming" => Some(MdsMethod::SgdStreaming),
        "classic" => Some(MdsMethod::ClassicMds),
        _ => None,
    }
}

////////////
// Params //
////////////

/// Parameters for Mds optimisation
///
/// ### Fields
///
/// * `randomised` - Shall randomised SVD be used for classical MDS
/// * `n_iter` - How many iterations to run the optimisation for. For the SGD
///   variants.
/// * `n_pairs` - The number of pairs to evaluate per given iteration. For the
///   SGD variants.
/// * `lr` - The learning rate.
/// * `n_threads` - Set this to ≥1 if you are happy with Hogwild parallel SGD.
///   Faster, but not determistic anymore.
pub struct MdsOptimParams<T> {
    pub randomised: bool,
    pub n_iter: usize,
    pub pairs_per_iter: usize,
    pub lr: T,
    pub n_threads: usize,
}

impl<T> MdsOptimParams<T>
where
    T: Float + FromPrimitive,
{
    /// Create a new instance
    ///
    /// ### Params
    ///
    /// * `n` - Number of samples in the data. Will be used to estimate the
    ///   number of pairs to evaluate during optimisation. For SGD methods.
    /// * `randomised` - Shall randomised SVD be used to solve the classical
    ///   MDS.
    /// * `n_iter` - Optional number of iterations to test. If not provided,
    ///   it will default to 1000.
    /// * `lr` - Optional learning rate.
    ///
    /// ### Returns
    ///
    /// Initialised `Self`
    pub fn new(
        n: usize,
        randomised: bool,
        n_iter: Option<usize>,
        lr: Option<T>,
        n_threads: Option<usize>,
    ) -> Self {
        let lr = lr.unwrap_or(T::from_f64(DEFAULT_LR).unwrap());
        let n_iter = n_iter.unwrap_or(1000);
        let pairs_per_iter = (n as f64 * (n as f64).ln() * 2.0) as usize;
        let n_threads = n_threads.unwrap_or(1);

        Self {
            randomised,
            n_iter,
            pairs_per_iter,
            lr,
            n_threads,
        }
    }
}

/////////////
// Helpers //
/////////////

/// Wrapper to share a raw mutable pointer across threads.
/// Safe in the Hogwild sense: concurrent writes to distinct indices are benign
/// data races on floating-point values, giving the standard Hogwild convergence
/// behaviour. Writes to the same index from different threads are racy but
/// bounded in harm: the worst case is a stale gradient step, not corruption.
#[derive(Copy, Clone)]
struct HogwildPtr<T>(*mut T);

unsafe impl<T: Send> Send for HogwildPtr<T> {}
unsafe impl<T: Send> Sync for HogwildPtr<T> {}

impl<T> HogwildPtr<T> {
    #[inline(always)]
    pub fn as_ptr(&self) -> *mut T {
        self.0
    }
}

/// Compute standard deviation of embedding
///
/// ### Params
///
/// * `embedding` - Embedding matrix
///
/// ### Returns
///
/// The standard deviation of the embedding
fn compute_std<T>(embedding: &[T]) -> T
where
    T: Float + std::iter::Sum,
{
    if embedding.is_empty() {
        return T::zero();
    }
    let sum: T = embedding.iter().map(|&v| v * v).sum();
    (sum / T::from(embedding.len()).unwrap()).sqrt()
}

/////////////////
// MDS methods //
/////////////////

/// Classic MDS using PCA on centered distance matrix
///
/// Fast but lower quality than SGD-MDS. Good for initialisation.
///
/// ### Params
///
/// * `distances` - N × N distance matrix
/// * `n_components` - Embedding dimensions
///
/// ### Returns
///
/// N × n_components embedding
pub fn classic_mds<T>(
    dist: &[Vec<T>],
    n_components: usize,
    randomised: bool,
    seed: usize,
) -> Vec<Vec<T>>
where
    T: Float + ComplexField + RealField + Send + Sync + std::iter::Sum,
    StandardNormal: Distribution<T>,
{
    let n = dist.len();

    let mut d_sq = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            d_sq[(i, j)] = dist[i][j] * dist[i][j];
        }
    }

    let mean_row: Vec<T> = (0..n)
        .map(|j| {
            let sum: T = (0..n).map(|i| d_sq[(i, j)]).sum();
            sum / T::from(n).unwrap()
        })
        .collect();
    let mean_col: Vec<T> = (0..n)
        .map(|i| {
            let sum: T = (0..n).map(|j| d_sq[(i, j)]).sum();
            sum / T::from(n).unwrap()
        })
        .collect();
    let mean_total: T = mean_row.iter().copied().sum::<T>() / T::from(n).unwrap();

    for i in 0..n {
        for j in 0..n {
            let val = d_sq[(i, j)];
            let centred = val - mean_row[j] - mean_col[i] + mean_total;
            d_sq[(i, j)] = -centred / T::from(2.0).unwrap();
        }
    }

    let mut embedding = vec![vec![T::zero(); n_components]; n];

    if randomised {
        let rsvd = randomised_svd(d_sq.as_ref(), n_components, seed, None, None);
        for i in 0..n {
            for k in 0..n_components {
                let singular_val = rsvd.s[k];
                if singular_val > T::zero() {
                    embedding[i][k] = rsvd.u[(i, k)] * singular_val.sqrt();
                }
            }
        }
    } else {
        let svd = d_sq.svd().unwrap();
        let s = svd.S();
        let u = svd.U();
        for i in 0..n {
            for k in 0..n_components {
                let singular_val = s[k];
                if singular_val > T::zero() {
                    embedding[i][k] = u[(i, k)] * singular_val.sqrt();
                }
            }
        }
    }

    embedding
}

/// SGD-based Metric MDS with Hogwild parallel online updates.
///
/// With params.n_threads = 1, updates are sequential and deterministic.
/// With params.n_threads > 1, pairs within each iteration are processed in
/// parallel with unsynchronised writes to the embedding (Hogwild). Collision
/// probability per pair is ~2 * n_threads / n; negligible for large n.
///
/// ### Params
///
/// * `dist` - N × N distance matrix
/// * `n_dim` - Number of embedding dimensions
/// * `params` - Optimisation parameters (includes n_threads)
/// * `init` - Optional flat initial embedding (n * n_dim, row-major)
/// * `seed` - Random seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Flat N × n_dim embedding (row-major)
///
/// N × n_components embedding
pub fn sgd_mds<T>(
    dist: &[Vec<T>],
    n_dim: usize,
    params: &MdsOptimParams<T>,
    init: Option<Vec<T>>,
    seed: usize,
    verbose: bool,
) -> Vec<Vec<T>>
where
    T: Float + Send + Sync + std::iter::Sum + FromPrimitive,
{
    let n = dist.len();
    assert!(n > 0, "Empty distance matrix");
    assert_eq!(dist[0].len(), n, "Distance matrix must be square");

    let mut rng = StdRng::seed_from_u64(seed as u64);

    let d_max = dist
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .fold(T::zero(), |acc, x| if x > acc { x } else { acc });

    let d_norm: Vec<T> = if d_max > T::zero() {
        dist.iter()
            .flat_map(|row| row.iter().map(|&d| d / d_max))
            .collect()
    } else {
        dist.iter().flat_map(|row| row.iter().copied()).collect()
    };

    let mut y = if let Some(init_y) = init {
        let y_std = compute_std(&init_y);
        if y_std > T::zero() {
            init_y.iter().map(|&v| v / y_std).collect()
        } else {
            init_y
        }
    } else {
        (0..n * n_dim)
            .map(|_| T::from(rng.random::<f64>() * 0.01).unwrap())
            .collect()
    };

    let n_iter = params.n_iter;
    let pairs_per_iter = params.pairs_per_iter;
    let eta_max = params.lr;
    let eta_min = eta_max * T::from(0.01).unwrap();
    let lambda = if n_iter > 1 {
        (eta_max / eta_min).ln() / T::from(n_iter - 1).unwrap()
    } else {
        T::zero()
    };

    if verbose {
        println!(
            "SGD-MDS: n={}, pairs_per_iter={}, n_iter={}, eta_max={:.6}, eta_min={:.6}, n_threads={}",
            n.separate_with_underscores(),
            pairs_per_iter.separate_with_underscores(),
            n_iter.separate_with_underscores(),
            eta_max.to_f64().unwrap(),
            eta_min.to_f64().unwrap(),
            params.n_threads,
        );
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(params.n_threads)
        .build()
        .expect("Failed to build thread pool");

    let mut prev_stress = None;

    for iteration in 0..n_iter {
        let lr_i = eta_max * (-lambda * T::from(iteration).unwrap()).exp();

        let pairs: Vec<(usize, usize)> =
            std::iter::from_fn(|| Some((rng.random_range(0..n), rng.random_range(0..n))))
                .filter(|(i, j)| i != j)
                .take(pairs_per_iter)
                .collect();

        let ptr = HogwildPtr(y.as_mut_ptr());

        // Stress accumulated with a small race risk on the float sum, which is
        // acceptable: it is only used for convergence checking.
        let total_err: T = pool.install(|| {
            pairs
                .par_iter()
                .map(|&(i, j)| {
                    let target_dist = d_norm[i * n + j];

                    // Force the closure to capture the struct via the getter
                    let raw = ptr.as_ptr();

                    let mut current_dist_sq = T::zero();

                    // SAFETY: Hogwild racy reads/writes. By avoiding slices, we prevent
                    // LLVM from making strict aliasing assumptions and caching reads.
                    unsafe {
                        let pi_base = raw.add(i * n_dim);
                        let pj_base = raw.add(j * n_dim);

                        // 1. Compute distance directly from raw pointers
                        for k in 0..n_dim {
                            let a = *pi_base.add(k);
                            let b = *pj_base.add(k);
                            current_dist_sq = current_dist_sq + (a - b) * (a - b);
                        }

                        let current_dist = current_dist_sq.sqrt().max(T::from(1e-10).unwrap());
                        let error = target_dist - current_dist;
                        let weight = T::from(-2.0).unwrap() * error / current_dist;

                        // 2. Online update: read fresh values and write immediately
                        for k in 0..n_dim {
                            let pi = pi_base.add(k);
                            let pj = pj_base.add(k);

                            let diff = *pi - *pj;
                            let grad = diff * weight;

                            *pi = *pi - lr_i * grad;
                            *pj = *pj + lr_i * grad;
                        }

                        error * error
                    }
                })
                .sum()
        });

        let stress = total_err / T::from(pairs.len()).unwrap();

        if verbose && iteration % 100 == 0 {
            println!(
                "Iter {}: stress={:.6}, lr={:.6}",
                iteration.separate_with_underscores(),
                stress.to_f64().unwrap(),
                lr_i.to_f64().unwrap(),
            );
        }

        if let Some(prev) = prev_stress {
            let rel_change = ((stress - prev) / (prev + T::from(1e-10).unwrap())).abs();
            if rel_change < T::from(1e-6).unwrap() && iteration > 50 {
                if verbose {
                    println!(
                        "Converged at iteration {} (rel_change={:.2e})",
                        iteration,
                        rel_change.to_f64().unwrap()
                    );
                }
                break;
            }
        }
        prev_stress = Some(stress);
    }

    if d_max > T::zero() {
        y.iter_mut().for_each(|v| *v = *v * d_max);
    }

    let mut embedding = vec![vec![T::zero(); n_dim]; n];

    for i in 0..n {
        for j in 0..n_dim {
            embedding[i][j] = y[i * n_dim + j];
        }
    }

    embedding
}

/// Streaming SGD-MDS with Hogwild parallel online updates.
///
/// Memory: O(N × d) instead of O(N²).
/// Distance computation is parallelised cleanly (read-only on dense rows).
/// Embedding updates use Hogwild unsynchronised writes; see sgd_mds for notes.
///
/// ### Params
///
/// * `potential` - Diffusion potential (N × n_landmarks) CSR matrix
/// * `n_dim` - Embedding dimensions
/// * `metric` - Distance metric
/// * `params` - Optimisation parameters (includes n_threads)
/// * `init` - Optional flat initial embedding (n * n_dim, row-major)
/// * `seed` - Random seed
/// * `verbose` - Verbosity
///
/// ### Returns
///
/// Flat N × n_dim embedding (row-major)
pub fn sgd_mds_streaming<T>(
    potential: &CompressedSparseData<T>,
    n_dim: usize,
    metric: &Dist,
    params: &MdsOptimParams<T>,
    init: Option<Vec<T>>,
    seed: usize,
    verbose: bool,
) -> Vec<Vec<T>>
where
    T: Float + Send + Sync + SimdDistance + std::iter::Sum + ComplexField + FromPrimitive,
{
    let n = potential.shape().0;
    let n_cols = potential.shape().1;

    let dense: Vec<T> = (0..n)
        .into_par_iter()
        .flat_map(|i| csr_row_to_dense(potential, i))
        .collect();

    let norms: Vec<T> = if matches!(metric, Dist::Cosine) {
        (0..n)
            .into_par_iter()
            .map(|i| T::calculate_norm(&dense[i * n_cols..(i + 1) * n_cols]))
            .collect()
    } else {
        Vec::new()
    };

    let compute_distance = |i: usize, j: usize| -> T {
        if i == j {
            return T::zero();
        }
        let row_i = &dense[i * n_cols..(i + 1) * n_cols];
        let row_j = &dense[j * n_cols..(j + 1) * n_cols];
        match metric {
            Dist::Euclidean => T::euclidean_simd(row_i, row_j).sqrt(),
            Dist::Cosine => {
                let dot = T::dot_simd(row_i, row_j);
                let denom = norms[i] * norms[j];
                if denom > T::zero() {
                    T::one() - (dot / denom)
                } else {
                    T::zero()
                }
            }
        }
    };

    let mut rng = StdRng::seed_from_u64(seed as u64);
    let sample_size = (n as f64).sqrt() as usize * 100;
    let mut d_max = T::zero();
    for _ in 0..sample_size.min(n * n / 2) {
        let i = rng.random_range(0..n);
        let j = rng.random_range(0..n);
        if i != j {
            let d = compute_distance(i, j);
            if d > d_max {
                d_max = d;
            }
        }
    }

    let mut y = if let Some(init_y) = init {
        let y_std = compute_std(&init_y);
        if y_std > T::zero() {
            init_y.iter().map(|&v| v / y_std).collect()
        } else {
            init_y
        }
    } else {
        (0..n * n_dim)
            .map(|_| T::from(rng.random::<f64>() * 0.01).unwrap())
            .collect()
    };

    let n_iter = params.n_iter;
    let pairs_per_iter = params.pairs_per_iter;
    let eta_max = params.lr;
    let eta_min = eta_max * T::from(0.01).unwrap();
    let lambda = if n_iter > 1 {
        (eta_max / eta_min).ln() / T::from(n_iter - 1).unwrap()
    } else {
        T::zero()
    };

    if verbose {
        println!(
            "Streaming SGD-MDS: n={}, pairs_per_iter={}, n_iter={}, n_threads={}",
            n, pairs_per_iter, n_iter, params.n_threads
        );
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(params.n_threads)
        .build()
        .expect("Failed to build thread pool");

    let mut prev_stress = None;

    for iteration in 0..n_iter {
        let lr_i = eta_max * (-lambda * T::from(iteration).unwrap()).exp();

        let pairs: Vec<(usize, usize)> =
            std::iter::from_fn(|| Some((rng.random_range(0..n), rng.random_range(0..n))))
                .filter(|(i, j)| i != j)
                .take(pairs_per_iter)
                .collect();

        // Read-only distance computation: clean parallel, no unsafety needed.
        let target_dists: Vec<T> = pairs
            .par_iter()
            .map(|&(i, j)| {
                let d = compute_distance(i, j);
                if d_max > T::zero() {
                    d / d_max
                } else {
                    d
                }
            })
            .collect();

        let ptr = HogwildPtr(y.as_mut_ptr());

        let total_error: T = pool.install(|| {
            pairs
                .par_iter()
                .zip(target_dists.par_iter())
                .map(|(&(i, j), &target_dist)| {
                    let raw = ptr.as_ptr();
                    let mut current_dist_sq = T::zero();

                    // SAFETY: Hogwild racy writes using raw pointers.
                    unsafe {
                        let pi_base = raw.add(i * n_dim);
                        let pj_base = raw.add(j * n_dim);

                        for k in 0..n_dim {
                            let a = *pi_base.add(k);
                            let b = *pj_base.add(k);
                            current_dist_sq = current_dist_sq + (a - b) * (a - b);
                        }

                        let current_dist = current_dist_sq.sqrt().max(T::from(1e-10).unwrap());
                        let error = target_dist - current_dist;
                        let weight = T::from(-2.0).unwrap() * error / current_dist;

                        for k in 0..n_dim {
                            let pi = pi_base.add(k);
                            let pj = pj_base.add(k);

                            let diff = *pi - *pj;
                            let grad = diff * weight;

                            *pi = *pi - lr_i * grad;
                            *pj = *pj + lr_i * grad;
                        }

                        error * error
                    }
                })
                .sum()
        });

        let stress = total_error / T::from(pairs.len()).unwrap();

        if verbose && iteration % 100 == 0 {
            println!(
                "Iter {}: stress={:.6}, lr={:.6}",
                iteration,
                stress.to_f64().unwrap(),
                lr_i.to_f64().unwrap(),
            );
        }

        if let Some(prev) = prev_stress {
            let rel_change = ((stress - prev) / (prev + T::from(1e-10).unwrap())).abs();
            if rel_change < T::from(1e-6).unwrap() && iteration > 50 {
                if verbose {
                    println!("Converged at iteration {}", iteration);
                }
                break;
            }
        }
        prev_stress = Some(stress);
    }

    if d_max > T::zero() {
        y.iter_mut().for_each(|v| *v = *v * d_max);
    }

    let mut embedding = vec![vec![T::zero(); n_dim]; n];

    for i in 0..n {
        for j in 0..n_dim {
            embedding[i][j] = y[i * n_dim + j];
        }
    }

    embedding
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_mds {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sgd_mds_identity_distances() {
        // Identity-like distances: all equidistant (equilateral triangle)
        let distances = vec![
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];

        let mds_params = MdsOptimParams::new(distances.len(), true, None, None, None);

        let embedding = sgd_mds(&distances, 2, &mds_params, None, 42, false);

        // Check shape
        assert_eq!(embedding.len(), 3);
        assert_eq!(embedding[0].len(), 2);

        // Check pairwise distances approximately preserved
        for i in 0..3 {
            for j in 0..3 {
                let mut dist_sq = 0.0;
                for k in 0..2 {
                    let diff = embedding[i][k] - embedding[j][k];
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();

                if i == j {
                    assert_relative_eq!(dist, 0.0, epsilon = 1e-2);
                } else {
                    assert_relative_eq!(dist, 1.0, epsilon = 0.3);
                }
            }
        }
    }

    #[test]
    fn test_sgd_mds_converges() {
        // Simple 4-point square
        let distances = vec![
            vec![0.0, 1.0, 1.414, 1.0],
            vec![1.0, 0.0, 1.0, 1.414],
            vec![1.414, 1.0, 0.0, 1.0],
            vec![1.0, 1.414, 1.0, 0.0],
        ];

        let mds_params = MdsOptimParams::new(distances.len(), true, None, None, None);

        let embedding = sgd_mds(&distances, 2, &mds_params, None, 42, false);

        // Check all pairwise distances
        for i in 0..4 {
            for j in 0..4 {
                let mut dist_sq = 0.0;
                for k in 0..2 {
                    let diff = embedding[i][k] - embedding[j][k];
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();
                assert_relative_eq!(dist, distances[i][j], epsilon = 0.15);
            }
        }
    }

    #[test]
    fn test_sgd_mds_streaming_basic() {
        // Create a simple 4x3 CSR potential matrix
        // Row structure:
        // [1.0, 0.0, 0.5]
        // [0.0, 1.0, 0.0]
        // [0.5, 0.0, 1.0]
        // [0.0, 0.5, 0.0]
        let data = vec![1.0, 0.5, 1.0, 1.0, 0.5, 0.5];
        let indices = vec![0, 2, 1, 0, 2, 1];
        let indptr = vec![0, 2, 3, 5, 6];
        let shape = (4, 3);

        let potential = CompressedSparseData::new_csr(&data, &indices, &indptr, shape);

        let mds_params = MdsOptimParams::new(potential.nrows(), true, None, None, None);

        let embedding = sgd_mds_streaming(
            &potential,
            2,
            &Dist::Euclidean,
            &mds_params,
            None,
            42,
            false,
        );

        // Basic sanity checks
        assert_eq!(embedding.len(), 4);
        assert_eq!(embedding[0].len(), 2);

        // Check that similar rows end up close together
        let mut dist_01_sq = 0.0;
        let mut dist_02_sq = 0.0;
        for k in 0..2 {
            let diff_01 = embedding[0][k] - embedding[1][k];
            let diff_02 = embedding[0][k] - embedding[2][k];
            dist_01_sq += diff_01 * diff_01;
            dist_02_sq += diff_02 * diff_02;
        }

        // Rows 0 and 2 are more similar (both have [1.0, 0.0, 0.5] pattern)
        // than rows 0 and 1, so they should be closer
        assert!(dist_02_sq < dist_01_sq * 1.5);
    }

    #[test]
    fn test_sgd_mds_streaming_cosine() {
        // Create a simple 3x2 CSR potential matrix for cosine distance
        let data = vec![1.0, 1.0, 0.0, 1.0, 1.0];
        let indices = vec![0, 1, 1, 0, 1];
        let indptr = vec![0, 2, 3, 5];
        let shape = (3, 2);

        let potential = CompressedSparseData::new_csr(&data, &indices, &indptr, shape);

        let mds_params = MdsOptimParams::new(potential.nrows(), true, None, None, None);

        let embedding =
            sgd_mds_streaming(&potential, 2, &Dist::Cosine, &mds_params, None, 42, false);

        // Basic sanity checks
        assert_eq!(embedding.len(), 3);
        assert_eq!(embedding[0].len(), 2);

        // All points should be finite
        for i in 0..3 {
            for k in 0..2 {
                assert!(embedding[i][k].is_finite());
            }
        }
    }

    #[test]
    fn test_classic_mds_identity() {
        // Identity-like distances
        let distances = vec![
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];

        let embedding = classic_mds(&distances, 2, true, 42);

        // Check shape
        assert_eq!(embedding.len(), 3);
        assert_eq!(embedding[0].len(), 2);

        // Check pairwise distances roughly preserved
        for i in 0..3 {
            for j in 0..3 {
                let mut dist_sq = 0.0;
                for k in 0..2 {
                    let diff = embedding[i][k] - embedding[j][k];
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();

                if i == j {
                    assert_relative_eq!(dist, 0.0, epsilon = 1e-6);
                } else {
                    assert_relative_eq!(dist, 1.0, epsilon = 0.3);
                }
            }
        }
    }
}
