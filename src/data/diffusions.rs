use ann_search_rs::utils::dist::{parse_ann_dist, Dist, SimdDistance};
use faer::MatRef;
use faer_traits::{ComplexField, RealField};
use num_traits::Float;
use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::SeedableRng;
use rayon::prelude::*;
use std::ops::{AddAssign, Range};

use crate::data::structures::*;
use crate::utils::math::*;
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
    T: Float + Send + Sync + Default + ComplexField,
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
    T: Float + Send + Sync + ComplexField,
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
/// Index of the knee point
pub fn find_knee_point<T>(y: &[T]) -> usize
where
    T: Float,
{
    let n = y.len();

    if n < 3 {
        panic!("Cannot find knee point on vector of length < 3");
    }

    // Use indices as x values
    let x: Vec<T> = (0..n).map(|i| T::from(i).unwrap()).collect();

    // Compute cumulative sums for linear fits
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

    // Compute forward fits (left of knee)
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

    // Compute backward fits (right of knee) by reversing
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

        // Error from right fit
        for i in breakpt..n {
            let predicted = mbck[breakpt - 1] * x[i] + bbck[breakpt - 1];
            error = error + (predicted - y[i]).abs();
        }

        error_curve[breakpt] = error;
    }

    // Find minimum error
    let mut min_idx = 1;
    let mut min_error = error_curve[1];

    for (i, &err) in error_curve.iter().enumerate().skip(1).take(n - 2) {
        if err < min_error {
            min_error = err;
            min_idx = i;
        }
    }

    min_idx
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
    T: Float + ComplexField,
{
    n_landmarks: usize,
    method: LandmarkMethod,
    assignments: Vec<usize>,
    landmark_op: CompressedSparseData<T>,
    transitions: CompressedSparseData<T>,
}

impl<T> PhateLandmarks<T>
where
    T: Float
        + Send
        + Sync
        + Default
        + SimdDistance
        + std::iter::Sum<T>
        + AddAssign
        + ComplexField
        + RealField,
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
    ) -> Self {
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
        let mut p_nm = p_mn.transpose();

        normalise_csr_rows_l1(&mut p_mn);
        normalise_csr_rows_l1(&mut p_nm);

        let landmark_op = csr_matmul_csr(&p_mn, &p_nm);

        Self {
            n_landmarks,
            method: landmark_method,
            assignments,
            landmark_op,
            transitions: p_nm,
        }
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
    pub fn power(&self, t: usize) -> CompressedSparseData<T>
    where
        T: AddAssign,
    {
        if t == 0 {
            // Return identity matrix
            unimplemented!("Return identity for t = 0");
        } else if t == 1 {
            return self.landmark_op.clone();
        }

        let mut result = self.landmark_op.clone();
        for _ in 1..t {
            result = csr_matmul_csr(&result, &self.landmark_op);
        }
        result
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
    pub fn power_optimal(&self, t_max: usize) -> CompressedSparseData<T>
    where
        T: AddAssign,
    {
        let t_opt = self.find_optimal_t(t_max);
        self.power(t_opt)
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
    ) -> CompressedSparseData<T>
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
    /// Optimal t value (knee point of entropy curve)
    pub fn find_optimal_t(&self, t_max: usize) -> usize {
        let dense = self.landmark_op.to_dense();
        let entropy = von_neumann_entropy(dense, t_max);
        find_knee_point(&entropy)
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
