use ann_search_rs::utils::dist::{Dist, SimdDistance};
use faer::Mat;
use faer_traits::{ComplexField, RealField};
use num_traits::Float;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use thousands::*;

use crate::data::structures::CompressedSparseData;
use crate::utils::sparse_ops::csr_row_to_dense;

/////////////
// Globals //
/////////////

pub const DEFAULT_LR: f64 = 0.01;

/////////////
// Helpers //
/////////////

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

/// Auto-tune SGD-MDS parameters based on dataset size
///
/// Generate sensible defaults for iterations and pairs per iteration.
///
/// ### Params
///
/// * `n` - Number of samples in the dataset
///
/// ### Returns
///
/// A tuple of (iterations, pairs per iteration)
fn auto_tune_params(n: usize) -> (usize, usize) {
    if n < 1000 {
        let n_iter = 300;
        let all_pairs = n * (n - 1) / 2;
        let pairs_per_iter = (n * n / 10).max(all_pairs);
        (n_iter, pairs_per_iter)
    } else if n < 5000 {
        let n_iter = 500;
        let pairs_per_iter = (n as f64 * (n as f64).ln() * 2.0) as usize;
        (n_iter, pairs_per_iter)
    } else {
        let n_iter = 800;
        let pairs_per_iter = (n as f64 * (n as f64).ln() * 2.0) as usize;
        (n_iter, pairs_per_iter)
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
fn compute_std<T>(embedding: &[Vec<T>]) -> T
where
    T: Float + std::iter::Sum,
{
    let n = embedding.len();
    if n == 0 {
        return T::zero();
    }

    let mut sum = T::zero();
    let mut count = 0;

    for row in embedding {
        for &val in row {
            sum = sum + val * val;
            count += 1;
        }
    }

    if count > 0 {
        (sum / T::from(count).unwrap()).sqrt()
    } else {
        T::zero()
    }
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
pub fn classic_mds<T>(dist: &[Vec<T>], n_components: usize) -> Vec<Vec<T>>
where
    T: Float + ComplexField + RealField + Send + Sync + std::iter::Sum,
{
    let n = dist.len();

    // square distances
    let mut d_sq = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            d_sq[(i, j)] = dist[i][j] * dist[i][j];
        }
    }

    // double centre: H = I - 1/n * 11^T
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

    // SVD
    let svd = d_sq.svd().unwrap();
    let s = svd.S();
    let u = svd.U();

    // Y = U * sqrt(S)
    let mut embedding = vec![vec![T::zero(); n_components]; n];
    for i in 0..n {
        for k in 0..n_components {
            let singular_val = s[k];
            if singular_val > T::zero() {
                embedding[i][k] = u[(i, k)] * singular_val.sqrt();
            }
        }
    }

    embedding
}

/// SGD-based Metric MDS with random pair sampling
///
/// Much faster than SMACOF (5-10x) while maintaining high quality.
/// Uses exponential learning rate decay and random pair sampling.
///
/// ### Params
///
/// * `dist` - N × N distance matrix
/// * `n_dim` - Number of embedding dimensions (typically 2)
/// * `n_iter` - Number of iterations (auto-tuned if None)
/// * `lr` - Base learning rate (default: 0.001)
/// * `init` - Initial embedding (if None, uses random)
/// * `seed` - Random seed
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// N × n_components embedding
pub fn sgd_mds<T>(
    dist: &[Vec<T>],
    n_dim: usize,
    n_iter: Option<usize>,
    lr: Option<T>,
    init: Option<Vec<Vec<T>>>,
    seed: usize,
    verbose: bool,
) -> Vec<Vec<T>>
where
    T: Float + Send + Sync + std::iter::Sum,
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

    // normalise the distances for stability
    let d_norm: Vec<Vec<T>> = if d_max > T::zero() {
        dist.iter()
            .map(|row| row.iter().map(|&d| d / d_max).collect())
            .collect()
    } else {
        dist.to_vec()
    };

    // initialise embedding
    let mut y = if let Some(init_y) = init {
        // normalise init to match distance scale
        let y_std = compute_std(&init_y);
        if y_std > T::zero() {
            init_y
                .iter()
                .map(|row| row.iter().map(|&v| v / y_std).collect())
                .collect()
        } else {
            init_y
        }
    } else {
        // random init
        (0..n)
            .map(|_| {
                (0..n_dim)
                    .map(|_| T::from(rng.random::<f64>() * 0.01).unwrap())
                    .collect()
            })
            .collect()
    };

    // auto-tune parameters if not provided
    let (n_iter, pairs_per_iter) = if let Some(iters) = n_iter {
        let (_, pairs) = auto_tune_params(n);
        (iters, pairs)
    } else {
        auto_tune_params(n)
    };

    let lr = lr.unwrap_or_else(|| T::from(DEFAULT_LR).unwrap());

    // lr schedule (exponential decay)
    let total_pairs = n * (n - 1) / 2;
    let sampling_ratio = pairs_per_iter as f64 / total_pairs as f64;
    let batch_scale = (1.0 / sampling_ratio).sqrt();

    let eta_max = lr * T::from(batch_scale).unwrap();
    let eta_min = eta_max * T::from(0.01).unwrap();
    let lambda = if n_iter > 1 {
        ((eta_max / eta_min).ln()) / T::from(n_iter - 1).unwrap()
    } else {
        T::zero()
    };

    if verbose {
        println!(
            "SGD-MDS: n={}, pairs_per_iter={}, n_iter={}, eta_max={:.6}, eta_min={:.6}",
            n.separate_with_underscores(),
            pairs_per_iter.separate_with_underscores(),
            n_iter.separate_with_underscores(),
            eta_max.to_f64().unwrap(),
            eta_min.to_f64().unwrap()
        );
    }

    let mut prev_stress = None;

    for iteration in 0..n_iter {
        let lr_i = eta_max * (-lambda * T::from(iteration).unwrap()).exp();

        let mut i_samples = Vec::with_capacity(pairs_per_iter);
        let mut j_samples = Vec::with_capacity(pairs_per_iter);
        let mut attempts = 0;
        while i_samples.len() < pairs_per_iter && attempts < pairs_per_iter * 10 {
            let i = rng.random_range(0..n);
            let j = rng.random_range(0..n);
            if i != j {
                i_samples.push(i);
                j_samples.push(j);
            }
            attempts += 1;
        }

        if i_samples.is_empty() {
            continue;
        }

        let mut gradients = vec![vec![T::zero(); n_dim]; n];
        let mut total_err = T::zero();

        for (&i, &j) in i_samples.iter().zip(&j_samples) {
            let target_dist = d_norm[i][j];

            // current distance
            let mut current_dist_sq = T::zero();
            for k in 0..n_dim {
                let diff = y[i][k] - y[j][k];
                current_dist_sq = current_dist_sq + diff * diff;
            }
            let current_dist = current_dist_sq.sqrt().max(T::from(1e-10).unwrap());

            // error
            let error = target_dist - current_dist;
            total_err = total_err + error * error;

            // gradient: -2 * error * (y_i - y_j) / current_dist
            let weight = T::from(-2.0).unwrap() * error / current_dist;

            for k in 0..n_dim {
                let diff = y[i][k] - y[j][k];
                let grad = diff * weight;
                gradients[i][k] = gradients[i][k] + grad;
                gradients[j][k] = gradients[j][k] - grad;
            }
        }

        // update positions
        for i in 0..n {
            for k in 0..n_dim {
                y[i][k] = y[i][k] - lr_i * gradients[i][k];
            }
        }

        let stress = total_err / T::from(i_samples.len()).unwrap();

        if verbose && iteration % 100 == 0 {
            println!(
                "Iter {}: stress={:.6}, lr={:.6}",
                iteration.separate_with_underscores(),
                stress.to_f64().unwrap(),
                lr.to_f64().unwrap()
            );
        }

        // check convergence / early termination
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

    // Rescale back to original distance scale
    if d_max > T::zero() {
        for i in 0..n {
            for k in 0..n_dim {
                y[i][k] = y[i][k] * d_max;
            }
        }
    }

    y
}

/// Streaming SGD-MDS that computes distances on-the-fly
///
/// Memory: O(N × d) instead of O(N²)
/// Time: Similar to dense (random sampling means cache misses anyway)
///
/// ### Params
///
/// * `potential` - Diffusion potential (N × n_landmarks) CSR matrix
/// * `n_dim` - Embedding dimensions
/// * `metric` - Distance metric
/// * `n_iter` - Number of SGD iterations
/// * `lr` - Base learning rate
/// * `init` - Initial embedding (optional)
/// * `seed` - Random seed
/// * `verbose` - Verbosity
///
/// ### Returns
///
/// N × n_components embedding
pub fn sgd_mds_streaming<T>(
    potential: &CompressedSparseData<T>,
    n_dim: usize,
    metric: &Dist,
    n_iter: Option<usize>,
    lr: Option<T>,
    init: Option<Vec<Vec<T>>>,
    seed: usize,
    verbose: bool,
) -> Vec<Vec<T>>
where
    T: Float + Send + Sync + SimdDistance + std::iter::Sum + ComplexField,
{
    let n = potential.shape().0;

    // pre-extract all rows as dense (O(N × d) memory)
    let dense_rows: Vec<Vec<T>> = (0..n)
        .into_par_iter()
        .map(|i| csr_row_to_dense(potential, i))
        .collect();

    // pre-compute norms for cosine
    let norms: Vec<T> = if matches!(metric, Dist::Cosine) {
        dense_rows
            .par_iter()
            .map(|row| T::calculate_norm(row))
            .collect()
    } else {
        Vec::new()
    };

    // distance computation closure
    let compute_distance = |i: usize, j: usize| -> T {
        if i == j {
            return T::zero();
        }

        match metric {
            Dist::Euclidean => {
                let squared = T::euclidean_simd(&dense_rows[i], &dense_rows[j]);
                squared.sqrt()
            }
            Dist::Cosine => {
                let dot = T::dot_simd(&dense_rows[i], &dense_rows[j]);
                let denom = norms[i] * norms[j];
                if denom > T::zero() {
                    T::one() - (dot / denom)
                } else {
                    T::zero()
                }
            }
        }
    };

    // find max distance for normalisation (sample subset to avoid O(N²))
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

    // initialise embedding
    let mut y = if let Some(init_y) = init {
        let y_std = compute_std(&init_y);
        if y_std > T::zero() {
            init_y
                .iter()
                .map(|row| row.iter().map(|&v| v / y_std).collect())
                .collect()
        } else {
            init_y
        }
    } else {
        (0..n)
            .map(|_| {
                (0..n_dim)
                    .map(|_| T::from(rng.random::<f64>() * 0.01).unwrap())
                    .collect()
            })
            .collect()
    };

    // auto-tune parameters
    let (n_iter, pairs_per_iter) = if let Some(iters) = n_iter {
        let pairs = (n as f64 * (n as f64).ln()) as usize;
        (iters, pairs)
    } else {
        auto_tune_params(n)
    };

    let lr = lr.unwrap_or_else(|| T::from(DEFAULT_LR).unwrap());

    // lr schedule
    let total_pairs = n * (n - 1) / 2;
    let sampling_ratio = pairs_per_iter as f64 / total_pairs as f64;
    let batch_scale = (1.0 / sampling_ratio).sqrt();

    let eta_max = lr * T::from(batch_scale).unwrap();
    let eta_min = eta_max * T::from(0.01).unwrap();
    let lambda = if n_iter > 1 {
        ((eta_max / eta_min).ln()) / T::from(n_iter - 1).unwrap()
    } else {
        T::zero()
    };

    if verbose {
        println!(
            "Streaming SGD-MDS: n={}, pairs_per_iter={}, n_iter={}",
            n, pairs_per_iter, n_iter
        );
    }

    let mut prev_stress = None;

    // SGD loop
    for iteration in 0..n_iter {
        let lr_i = eta_max * (-lambda * T::from(iteration).unwrap()).exp();

        let mut gradients = vec![vec![T::zero(); n_dim]; n];

        // generate pairs upfront
        let pairs: Vec<(usize, usize)> = (0..(pairs_per_iter * 2))
            .map(|_| {
                let i = rng.random_range(0..n);
                let j = rng.random_range(0..n);
                (i, j)
            })
            .filter(|(i, j)| i != j)
            .take(pairs_per_iter)
            .collect();

        // compute target distances in parallel
        let target_dists: Vec<T> = pairs
            .par_iter()
            .map(|&(i, j)| {
                let dist = compute_distance(i, j);
                if d_max > T::zero() {
                    dist / d_max
                } else {
                    dist
                }
            })
            .collect();

        // apply gradients sequentially
        let mut total_error = T::zero();
        for (&(i, j), &target_dist) in pairs.iter().zip(target_dists.iter()) {
            // current embedding distance
            let mut current_dist_sq = T::zero();
            for k in 0..n_dim {
                let diff = y[i][k] - y[j][k];
                current_dist_sq = current_dist_sq + diff * diff;
            }
            let current_dist = current_dist_sq.sqrt().max(T::from(1e-10).unwrap());

            let error = target_dist - current_dist;
            total_error = total_error + error * error;

            let weight = T::from(-2.0).unwrap() * error / current_dist;

            for k in 0..n_dim {
                let diff = y[i][k] - y[j][k];
                let grad = diff * weight;
                gradients[i][k] = gradients[i][k] + grad;
                gradients[j][k] = gradients[j][k] - grad;
            }
        }

        // update
        for i in 0..n {
            for k in 0..n_dim {
                y[i][k] = y[i][k] - lr_i * gradients[i][k];
            }
        }

        let stress = total_error / T::from(pairs.len()).unwrap();

        if verbose && iteration % 100 == 0 {
            println!(
                "Iter {}: stress={:.6}, lr={:.6}",
                iteration,
                stress.to_f64().unwrap(),
                lr.to_f64().unwrap()
            );
        }

        // convergence check
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

    // Rescale
    if d_max > T::zero() {
        for i in 0..n {
            for k in 0..n_dim {
                y[i][k] = y[i][k] * d_max;
            }
        }
    }

    y
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

        let embedding = sgd_mds(&distances, 2, Some(1000), Some(0.01), None, 42, false);

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

        let embedding = sgd_mds(&distances, 2, Some(500), None, None, 42, false);

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

        let embedding = sgd_mds_streaming(
            &potential,
            2,
            &Dist::Euclidean,
            Some(500),
            None,
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

        let embedding = sgd_mds_streaming(
            &potential,
            2,
            &Dist::Cosine,
            Some(500),
            None,
            None,
            42,
            false,
        );

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

        let embedding = classic_mds(&distances, 2);

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
