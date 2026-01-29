use ann_search_rs::utils::dist::{parse_ann_dist, Dist, SimdDistance};
use faer::MatRef;
use num_traits::Float;
use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::SeedableRng;
use rayon::prelude::*;
use std::ops::Range;

use crate::data::structures::*;
use crate::utils::sparse_ops::*;

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
/// Row-normalized CSR matrix
pub fn build_diffusion_operator<T>(matrix: &CompressedSparseData<T>) -> CompressedSparseData<T>
where
    T: Float + Send + Sync + Default,
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
                sum = sum + matrix.data[idx];
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

/////////////
// Helpers //
/////////////

/// Enum representing different landmark diffusion methods.
pub enum LandmarkMethod {
    /// Randomly select landmarks.
    Random { seed: u64 },
    /// Use spectral clustering to select landmarks.
    Spectral { n_svd: usize },
}

impl Default for LandmarkMethod {
    /// Default to random landmark selection with a fixed seed.
    fn default() -> Self {
        LandmarkMethod::Random { seed: 42 }
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
    let n_svd = n_svd.unwrap_or(10);

    match s.to_lowercase().as_str() {
        "random" => Some(LandmarkMethod::Random { seed: seed as u64 }),
        "spectral" => Some(LandmarkMethod::Spectral { n_svd }),
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
    T: Float + SimdDistance + Send + Sync,
{
    let mut min_dist = T::infinity();
    let mut min_idx = 0;

    for idx in 0..n_landmark {
        let landmark = &landmark_data[idx * dim..(idx + 1) * dim];
        let dist = match dist {
            Dist::Euclidean => T::euclidean_simd(landmark, data),
            Dist::Cosine => {
                let dot = T::dot_simd(landmark, data);
                T::one() - (dot / (*norm_data * norm_landmark[idx]))
            }
        };
        if dist < min_dist {
            min_dist = dist;
            min_idx = idx;
        }
    }

    min_idx
}

/// Build the row-stochastic Diffusion Operator P from the landmark data.
///
/// ### Params
///
/// * `diffusion_op`: The diffusion operator.
/// * `assignments`: The assignments of points to landmarks.
/// * `n_landmarks`: The number of landmarks.
/// * `n`: The number of points.
///
/// ### Returns
///
/// The row-stochastic Diffusion Operator P.
fn build_landmarks_to_data<T>(
    diffusion_op: &CompressedSparseData<T>,
    assignments: &[usize],
    n_landmarks: usize,
) -> CompressedSparseData<T>
where
    T: Float + Send + Sync,
{
    let n = assignments.len();
    let mut landmark_pts: Vec<Vec<usize>> = vec![Vec::new(); n_landmarks];
    for (point_idx, &landmark_idx) in assignments.iter().enumerate() {
        landmark_pts[landmark_idx].push(point_idx);
    }

    let rows: Vec<SparseRow<T>> = (0..n_landmarks)
        .into_par_iter()
        .map(|landmark_idx| sparse_row_sum(diffusion_op, &landmark_pts[landmark_idx]))
        .collect();

    sparse_row_to_csr(&rows, n)
}

////////////////////
// PhateLandmarks //
////////////////////

/// Struct representing PhateLandmarks.
///
/// ### Fields
///
/// * `n_landmarks`: Number of landmarks to select.
/// * `method`: Method to use for landmark selection.
/// * `assignments`: Data point → landmark mapping.
/// * `landmark_op`: Small L matrix.
/// * `transitions`: P_nm for interpolation.
pub struct PhateLandmarks<T>
where
    T: Float,
{
    n_landmarks: usize,
    method: LandmarkMethod,
    assignments: Vec<usize>,
    landmark_op: CompressedSparseData<T>,
    transitions: CompressedSparseData<T>,
}

impl<T> PhateLandmarks<T>
where
    T: Float + Send + Sync + Default + SimdDistance + std::iter::Sum<T>,
{
    /// Build landmarks from an existing diffusion operator
    ///
    /// ### Params
    ///
    /// * `data` - Original data (N × features)
    /// * `diffusion_op` - The P matrix (N × N) already built and normalized
    /// * `n_landmarks` - Number of landmarks to use
    /// * `method` - Random or Spectral
    /// * `distance` - Distance metric used
    /// * `seed` - Seed for random number generator
    /// * `n_svd` - Number of SVD components to use
    ///
    /// ### Returns
    ///
    /// PhateLandmarks structure ready for powering
    pub fn build(
        data: MatRef<T>,
        diffusion_op: &CompressedSparseData<T>,
        n_landmarks: usize,
        method: &str,
        distance: &str,
        seed: Option<usize>,
        n_svd: Option<usize>,
    ) {
        let (data, n, dim) = matrix_to_flat(data);
        let landmark_method = parse_landmark_method(method, seed, n_svd).unwrap_or_default();
        let distance = parse_ann_dist(distance).unwrap_or_default();

        let assignments: Vec<usize> = match landmark_method {
            LandmarkMethod::Random { seed } => {
                let landmark_indices = random_sample(0..n, n_landmarks, seed);

                let landmark_data: Vec<T> = landmark_indices
                    .iter()
                    .flat_map(|&i| data[i * dim..(i + 1) + dim].iter().copied())
                    .collect();

                // calculate norms if distance is cosine
                let norm_data = match distance {
                    Dist::Cosine => (0..n)
                        .into_par_iter()
                        .map(|i| T::calculate_norm(&data[i * dim..(i + 1) * dim]))
                        .collect::<Vec<_>>(),
                    Dist::Euclidean => Vec::new(),
                };
                let norm_landmark = match distance {
                    Dist::Cosine => (0..n_landmarks)
                        .into_par_iter()
                        .map(|i| T::calculate_norm(&landmark_data[i * dim..(i + 1) * dim]))
                        .collect::<Vec<_>>(),
                    Dist::Euclidean => Vec::new(),
                };

                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        assign_to_landmark(
                            &data[i * dim..(i + 1) * dim],
                            &norm_data[i],
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
                unimplemented!("Spectral clustering not yet implemented")
            }
        };

        let mut p_mn = build_landmarks_to_data(diffusion_op, &assignments, n_landmarks);
        let mut p_nm = p_mn.transform();

        normalise_csr_rows_l1(&mut p_mn);
        normalise_csr_rows_l1(&mut p_nm);
    }
}

///////////
// Tests //
///////////

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
}
