//! Helper functions to calculate potentials for PHATE

use ann_search_rs::utils::dist::{Dist, SimdDistance};
use faer_traits::ComplexField;
use num_traits::Float;
use rayon::prelude::*;
use std::ops::AddAssign;

use crate::data::structures::CompressedSparseData;
use crate::utils::sparse_ops::*;

////////////////////////////
// Potential calculations //
////////////////////////////

/// Apply log potential transformation: -log(P^t + epsilon)
///
/// Used for gamma = 1 in PHATE. The epsilon prevents log(0).
///
/// ### Params
///
/// * `matrix` - CSR matrix P^t
/// * `epsilon` - Small value to add before log (typically 1e-7)
///
/// ### Returns
///
/// Transformed CSR matrix with same sparsity pattern
pub fn apply_log_potential<T>(
    matrix: &CompressedSparseData<T>,
    epsilon: T,
) -> CompressedSparseData<T>
where
    T: Float + Send + Sync + ComplexField,
{
    assert!(matrix.cs_type.is_csr(), "Matrix must be CSR format");

    let data: Vec<T> = matrix
        .data
        .par_iter()
        .map(|&val| {
            let clamped = val.min(T::one()).max(T::zero());
            -(clamped + epsilon).ln()
        })
        .collect();

    CompressedSparseData::new_csr(&data, &matrix.indices, &matrix.indptr, matrix.shape())
}

/// Apply power potential transformation: (P^t)^c / c
///
/// Used for gamma != 1 and gamma != -1 in PHATE.
/// When gamma = -1, just return the matrix unchanged (identity transform).
///
/// ### Params
///
/// * `matrix` - CSR matrix P^t
/// * `gamma` - PHATE gamma parameter
///
/// ### Returns
///
/// Transformed CSR matrix with same sparsity pattern
pub fn apply_power_potential<T>(
    matrix: &CompressedSparseData<T>,
    gamma: T,
) -> CompressedSparseData<T>
where
    T: Float + Send + Sync + ComplexField,
{
    assert!(matrix.cs_type.is_csr(), "Matrix must be CSR format");

    // gamma = -1 is identity transformation
    if (gamma + T::one()).abs() < T::from(1e-10).unwrap() {
        return matrix.clone();
    }

    let c = (T::one() - gamma) / T::from(2.0).unwrap();

    let data: Vec<T> = matrix.data.par_iter().map(|&val| val.powf(c) / c).collect();

    CompressedSparseData::new_csr(&data, &matrix.indices, &matrix.indptr, matrix.shape())
}

/// Calculate diffusion potential at time t with gamma transformation
///
/// Convenience function that powers the diffusion operator and applies
/// the appropriate potential transformation based on gamma.
///
/// ### Params
///
/// * `diffusion_op` - Row-stochastic diffusion operator P
/// * `t` - Diffusion time
/// * `gamma` - Potential transformation parameter
///   - gamma = 1: log potential (PHATE default)
///   - gamma = -1: identity (no transformation)
///   - other: power potential
///
/// ### Returns
///
/// Diffusion potential matrix (CSR format)
pub fn calculate_potential<T>(
    diffusion_op: &CompressedSparseData<T>,
    t: usize,
    gamma: T,
) -> CompressedSparseData<T>
where
    T: Float + Send + Sync + AddAssign + ComplexField,
{
    // power the operator
    let diffused = matrix_power(diffusion_op, t);

    // apply transformation
    if (gamma - T::one()).abs() < T::from(1e-10).unwrap() {
        // log transformation
        apply_log_potential(&diffused, T::from(1e-7).unwrap())
    } else if (gamma + T::one()).abs() < T::from(1e-10).unwrap() {
        // identity transformation
        diffused
    } else {
        // power transformation
        apply_power_potential(&diffused, gamma)
    }
}

///////////////////////
// Dist calculations //
///////////////////////

/// Extract a CSR row as a dense vector
///
/// ### Params
///
/// * `matrix` - CSR matrix
/// * `row_idx` - Row index to extract
///
/// ### Returns
///
/// Dense vector of length ncols (zeros for non-stored elements)
fn csr_row_to_dense<T>(matrix: &CompressedSparseData<T>, row_idx: usize) -> Vec<T>
where
    T: Float + Copy + ComplexField,
{
    let ncols = matrix.shape().1;
    let mut dense = vec![T::zero(); ncols];

    let start = matrix.indptr[row_idx];
    let end = matrix.indptr[row_idx + 1];

    for idx in start..end {
        let col = matrix.indices[idx];
        dense[col] = matrix.data[idx];
    }

    dense
}

/// Compute pairwise distances between rows of a CSR matrix
///
/// Converts sparse rows to dense and uses SIMD-accelerated distance
/// computations from `ann-search-rs`.
///
/// ### Params
///
/// * `potential` - Diffusion potential matrix (N × N or N × n_landmarks)
/// * `metric` - Distance metric ("euclidean" or "cosine")
///
/// ### Returns
///
/// Dense N × N distance matrix (as Vec<Vec<T>>)
pub fn compute_potential_distances<T>(
    potential: &CompressedSparseData<T>,
    metric: &Dist,
) -> Vec<Vec<T>>
where
    T: Float + Send + Sync + SimdDistance + ComplexField,
{
    assert!(potential.cs_type.is_csr(), "Matrix must be CSR format");

    let n = potential.shape().0;

    let dense_rows: Vec<Vec<T>> = (0..n)
        .into_par_iter()
        .map(|i| csr_row_to_dense(potential, i))
        .collect();

    let norms: Vec<T> = if matches!(metric, Dist::Cosine) {
        dense_rows
            .par_iter()
            .map(|row| T::calculate_l2_norm(row))
            .collect()
    } else {
        Vec::new()
    };

    // enumerate all upper-triangle pairs and compute in parallel
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .collect();

    let computed: Vec<T> = pairs
        .par_iter()
        .map(|&(i, j)| match metric {
            Dist::Euclidean => T::euclidean_simd(&dense_rows[i], &dense_rows[j]).sqrt(),
            Dist::Cosine => {
                let dot = T::dot_simd(&dense_rows[i], &dense_rows[j]);
                let denom = norms[i] * norms[j];
                if denom > T::zero() {
                    T::one() - (dot / denom)
                } else {
                    T::zero()
                }
            }
        })
        .collect();

    // fill both triangles sequentially
    let mut dist = vec![vec![T::zero(); n]; n];
    for ((i, j), d) in pairs.iter().zip(computed.iter()) {
        dist[*i][*j] = *d;
        dist[*j][*i] = *d;
    }

    dist
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_potential_transforms {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_matrix() -> CompressedSparseData<f64> {
        // Simple matrix with known values
        let data = vec![0.5, 0.3, 0.2, 0.7, 0.3, 0.8, 0.2];
        let indices = vec![0, 1, 2, 0, 2, 1, 2];
        let indptr = vec![0, 3, 5, 7];
        CompressedSparseData::new_csr(&data, &indices, &indptr, (3, 3))
    }

    #[test]
    fn test_log_potential_values() {
        let mat = create_test_matrix();
        let result = apply_log_potential(&mat, 1e-7);

        // Check structure preserved
        assert_eq!(result.indices, mat.indices);
        assert_eq!(result.indptr, mat.indptr);
        assert_eq!(result.data.len(), mat.data.len());

        // Check values: -log(val + epsilon)
        let epsilon = 1e-7;
        assert_relative_eq!(result.data[0], -(0.5 + epsilon).ln(), epsilon = 1e-10);
        assert_relative_eq!(result.data[1], -(0.3 + epsilon).ln(), epsilon = 1e-10);
        assert_relative_eq!(result.data[2], -(0.2 + epsilon).ln(), epsilon = 1e-10);

        // All values should be positive (since input is < 1)
        for &val in &result.data {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_log_potential_epsilon_prevents_inf() {
        // Create matrix with very small values
        let data = vec![1e-10, 1e-20, 0.5];
        let indices = vec![0, 1, 2];
        let indptr = vec![0, 2, 3];
        let mat = CompressedSparseData::new_csr(&data, &indices, &indptr, (2, 3));

        let result = apply_log_potential(&mat, 1e-7);

        // Should all be finite
        for &val in &result.data {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_power_potential_gamma_negative_one() {
        let mat = create_test_matrix();
        let result = apply_power_potential(&mat, -1.0);

        // gamma = -1 should be identity
        assert_eq!(result.data, mat.data);
        assert_eq!(result.indices, mat.indices);
        assert_eq!(result.indptr, mat.indptr);
    }

    #[test]
    fn test_power_potential_values() {
        let mat = create_test_matrix();
        let gamma = 0.5;
        let result = apply_power_potential(&mat, gamma);

        // c = (1 - gamma) / 2 = (1 - 0.5) / 2 = 0.25
        let c = (1.0 - gamma) / 2.0;

        // Check structure preserved
        assert_eq!(result.indices, mat.indices);
        assert_eq!(result.indptr, mat.indptr);

        // Check values: val^c / c
        assert_relative_eq!(result.data[0], 0.5_f64.powf(c) / c, epsilon = 1e-10);
        assert_relative_eq!(result.data[1], 0.3_f64.powf(c) / c, epsilon = 1e-10);
        assert_relative_eq!(result.data[2], 0.2_f64.powf(c) / c, epsilon = 1e-10);
    }

    #[test]
    fn test_power_potential_gamma_zero() {
        let mat = create_test_matrix();
        let gamma = 0.0;
        let result = apply_power_potential(&mat, gamma);

        // c = (1 - 0) / 2 = 0.5 (square root)
        let c = 0.5;

        // First value should be sqrt(0.5) / 0.5 = sqrt(0.5) * 2
        assert_relative_eq!(result.data[0], 0.5_f64.powf(c) / c, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_potential_log() {
        // Create simple diffusion operator
        let data = vec![0.5, 0.5, 0.4, 0.6, 1.0];
        let indices = vec![0, 1, 0, 1, 2];
        let indptr = vec![0, 2, 4, 5];
        let diff_op = CompressedSparseData::new_csr(&data, &indices, &indptr, (3, 3));

        let result = calculate_potential(&diff_op, 2, 1.0);

        // Should be log potential
        // Values can be slightly negative when diffusion probability ≈ 1
        assert!(!result.data.is_empty());
        for &val in &result.data {
            assert!(val.is_finite());
        }

        // Most values should be positive (small probabilities → large potential)
        let positive_count = result.data.iter().filter(|&&v| v > 0.0).count();
        assert!(positive_count > 0, "Expected some positive potentials");
    }

    #[test]
    fn test_calculate_potential_identity() {
        let data = vec![0.5, 0.5, 0.4, 0.6, 1.0];
        let indices = vec![0, 1, 0, 1, 2];
        let indptr = vec![0, 2, 4, 5];
        let diff_op = CompressedSparseData::new_csr(&data, &indices, &indptr, (3, 3));

        let result = calculate_potential(&diff_op, 2, -1.0);

        // Should be P^2 with no transformation
        let p2 = matrix_power(&diff_op, 2);
        assert_eq!(result.data.len(), p2.data.len());
        for (res_val, p2_val) in result.data.iter().zip(&p2.data) {
            assert_relative_eq!(res_val, p2_val, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_calculate_potential_power() {
        let data = vec![0.5, 0.5, 0.4, 0.6, 1.0];
        let indices = vec![0, 1, 0, 1, 2];
        let indptr = vec![0, 2, 4, 5];
        let diff_op = CompressedSparseData::new_csr(&data, &indices, &indptr, (3, 3));

        let result = calculate_potential(&diff_op, 2, 0.5);

        // Should be power potential
        assert!(!result.data.is_empty());
        for &val in &result.data {
            assert!(val.is_finite());
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_potential_preserves_sparsity() {
        let mat = create_test_matrix();
        let nnz = mat.data.len();

        // All transformations should preserve sparsity
        let log_result = apply_log_potential(&mat, 1e-7);
        assert_eq!(log_result.data.len(), nnz);

        let power_result = apply_power_potential(&mat, 0.5);
        assert_eq!(power_result.data.len(), nnz);
    }
}

#[cfg(test)]
mod test_distance_computation {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_csr_row_to_dense() {
        // Row 0: [0.5, 0.0, 0.3]
        // Row 1: [0.0, 0.7, 0.0]
        let data = vec![0.5, 0.3, 0.7];
        let indices = vec![0, 2, 1];
        let indptr = vec![0, 2, 3];
        let mat = CompressedSparseData::new_csr(&data, &indices, &indptr, (2, 3));

        let row0 = csr_row_to_dense(&mat, 0);
        assert_eq!(row0, vec![0.5, 0.0, 0.3]);

        let row1 = csr_row_to_dense(&mat, 1);
        assert_eq!(row1, vec![0.0, 0.7, 0.0]);
    }

    #[test]
    fn test_euclidean_distances() {
        // Create identity-like matrix
        // Row 0: [1, 0, 0]
        // Row 1: [0, 1, 0]
        // Row 2: [0, 0, 1]
        let data = vec![1.0, 1.0, 1.0];
        let indices = vec![0, 1, 2];
        let indptr = vec![0, 1, 2, 3];
        let mat = CompressedSparseData::new_csr(&data, &indices, &indptr, (3, 3));

        let distances = compute_potential_distances(&mat, &Dist::Euclidean);

        // Check diagonal is zero
        for i in 0..3 {
            assert_relative_eq!(distances[i][i], 0.0, epsilon = 1e-10);
        }

        // Row 0: [1, 0, 0], Row 1: [0, 1, 0]
        // Distance = sqrt((1-0)² + (0-1)² + (0-0)²) = sqrt(2)
        assert_relative_eq!(distances[0][1], 2.0_f64.sqrt(), epsilon = 1e-10);
        assert_relative_eq!(distances[1][0], 2.0_f64.sqrt(), epsilon = 1e-10);

        // Row 0 and Row 2 should also be sqrt(2)
        assert_relative_eq!(distances[0][2], 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_distances() {
        // Row 0: [1, 0, 0]
        // Row 1: [0, 1, 0]
        // Row 2: [1, 1, 0] (normalized: [1/√2, 1/√2, 0])
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let indices = vec![0, 1, 0, 1];
        let indptr = vec![0, 1, 2, 4];
        let mat = CompressedSparseData::new_csr(&data, &indices, &indptr, (3, 3));

        let distances = compute_potential_distances(&mat, &Dist::Cosine);

        // Row 0 and Row 1 are orthogonal → cosine = 0, distance = 1
        assert_relative_eq!(distances[0][1], 1.0, epsilon = 1e-10);

        // Row 0 and Row 2: dot = 1, norms = 1 and √2
        // cosine = 1/√2, distance = 1 - 1/√2
        let expected = 1.0 - (1.0 / 2.0_f64.sqrt());
        assert_relative_eq!(distances[0][2], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_symmetry() {
        let data = vec![0.5, 0.3, 0.7, 0.2, 0.4, 0.6];
        let indices = vec![0, 1, 1, 2, 0, 2];
        let indptr = vec![0, 2, 4, 6];
        let mat = CompressedSparseData::new_csr(&data, &indices, &indptr, (3, 3));

        let distances = compute_potential_distances(&mat, &Dist::Euclidean);

        // Distance matrix should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(distances[i][j], distances[j][i], epsilon = 1e-10);
            }
        }
    }
}
