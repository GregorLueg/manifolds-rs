//! Various sparse operations on the internal `CompressedSparseData` structure.
//! These are designed to be highly efficient and use unsafe under the hood.

use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::data::structures::*;
use crate::utils::traits::*;

////////////////////
// Row extraction //
////////////////////

/// Extracts a row from a compressed sparse matrix and returns it as a dense
/// vector.
///
/// ### Params
///
/// * `matrix` - The compressed sparse matrix to extract the row from
/// * `row` - The index of the row to extract
///
/// ### Returns
///
/// A dense vector containing the values of the specified row
pub fn csr_row_to_dense<T>(matrix: &CompressedSparseData<T>, row: usize) -> Vec<T>
where
    T: ManifoldsFloat,
{
    let ncols = matrix.shape.1;
    let mut dense = vec![T::zero(); ncols];

    let start = matrix.indptr[row];
    let end = matrix.indptr[row + 1];

    for idx in start..end {
        let col = matrix.indices[idx];
        dense[col] = matrix.data[idx];
    }

    dense
}

/////////////////////
// Sparse row sums //
/////////////////////

/// Sums the columns of specific rows
///
/// ### Params
///
/// * `mat` - The compressed sparse matrix to sum over
/// * `row_indices` - The indices of the rows to sum
///
/// ### Returns
///
/// A sparse row containing the summed columns for the specificied rows
pub fn sparse_row_sum<T>(mat: &CompressedSparseData<T>, row_indices: &[usize]) -> SparseRow<T>
where
    T: ManifoldsFloat,
{
    let mut col_sums: FxHashMap<usize, T> =
        FxHashMap::with_capacity_and_hasher(mat.ncols(), FxBuildHasher);

    for &row_idx in row_indices {
        let start = mat.indptr[row_idx];
        let end = mat.indptr[row_idx + 1];

        for col_idx in start..end {
            let col = mat.indices[col_idx];
            let val = mat.data[col_idx];
            *col_sums.entry(col).or_insert(T::zero()) += val;
        }
    }

    let mut cols_vals: Vec<(usize, T)> = col_sums.into_iter().collect();
    cols_vals.sort_by_key(|(col, _)| *col);

    SparseRow {
        indices: cols_vals.iter().map(|(c, _)| *c).collect(),
        data: cols_vals.iter().map(|(_, v)| *v).collect(),
    }
}

////////////////////////
// CSR multiplication //
////////////////////////

/// Sparse accumulator for efficient sparse matrix multiplication
///
/// ### Fields
///
/// * `values` - Vector storing accumulated values for each index
/// * `indices` - Vector of active (non-zero) indices
/// * `flags` - Boolean flags indicating which indices are active
struct SparseAccumulator<T>
where
    T: ManifoldsFloat,
{
    values: Vec<T>,
    indices: Vec<usize>,
    flags: Vec<bool>,
}

impl<T> SparseAccumulator<T>
where
    T: ManifoldsFloat,
{
    /// Create a new sparse accumulator
    ///
    /// ### Params
    ///
    /// * `size` - Maximum number of indices to accumulate
    fn new(size: usize) -> Self {
        Self {
            values: vec![T::zero(); size],
            indices: Vec::with_capacity(size / 10),
            flags: vec![false; size],
        }
    }

    /// Add a value to the accumulator at the given index
    ///
    /// ### Params
    ///
    /// * `idx` - Index to accumulate at
    /// * `val` - Value to add
    ///
    /// ### Safety
    ///
    /// `idx` must be less than the size specified during construction
    #[inline]
    unsafe fn add_acc(&mut self, idx: usize, val: T) {
        if !*self.flags.get_unchecked(idx) {
            *self.flags.get_unchecked_mut(idx) = true;
            self.indices.push(idx);
            *self.values.get_unchecked_mut(idx) = val;
        } else {
            *self.values.get_unchecked_mut(idx) += val;
        }
    }

    /// Extract accumulated values as sorted index-value pairs and reset the accumulator
    ///
    /// ### Returns
    ///
    /// Vector of (index, value) pairs sorted by index
    #[inline]
    fn extract_sorted(&mut self) -> Vec<(usize, T)> {
        self.indices.sort_unstable();
        let result: Vec<(usize, T)> = unsafe {
            self.indices
                .iter()
                .map(|&i| (i, *self.values.get_unchecked(i)))
                .collect()
        };
        // Reset for next use
        unsafe {
            for &idx in &self.indices {
                *self.flags.get_unchecked_mut(idx) = false;
                *self.values.get_unchecked_mut(idx) = T::zero();
            }
        }
        self.indices.clear();
        result
    }
}

/// Multiply two CSR matrices using sparse accumulators and parallel processing
///
/// Adapted for landmark PHATE operations. Ported over the original bixverse
/// Rust code.
///
/// ### Params
///
/// * `a` - Left CSR matrix
/// * `b` - Right CSR matrix
///
/// ### Returns
///
/// Product matrix in CSR format
pub fn csr_matmul_csr<T>(
    a: &CompressedSparseData<T>,
    b: &CompressedSparseData<T>,
) -> CompressedSparseData<T>
where
    T: ManifoldsFloat,
{
    assert!(a.cs_type.is_csr() && b.cs_type.is_csr());
    assert_eq!(a.shape.1, b.shape.0, "Dimension mismatch");

    let nrows = a.shape.0;
    let ncols = b.shape.1;

    let row_results: Vec<Vec<(usize, T)>> = (0..nrows)
        .into_par_iter()
        .map_init(
            || SparseAccumulator::new(ncols),
            |acc, i| {
                unsafe {
                    let a_indptr = a.indptr.as_ptr();
                    let a_indices = a.indices.as_ptr();
                    let a_data = a.data.as_ptr();
                    let b_indptr = b.indptr.as_ptr();
                    let b_indices = b.indices.as_ptr();
                    let b_data = b.data.as_ptr();

                    let a_start = *a_indptr.add(i);
                    let a_end = *a_indptr.add(i + 1);

                    for a_idx in a_start..a_end {
                        let k = *a_indices.add(a_idx);
                        let a_val = *a_data.add(a_idx);

                        let b_start = *b_indptr.add(k);
                        let b_end = *b_indptr.add(k + 1);

                        for b_idx in b_start..b_end {
                            let j = *b_indices.add(b_idx);
                            let b_val = *b_data.add(b_idx);
                            acc.add_acc(j, a_val * b_val);
                        }
                    }
                }

                acc.extract_sorted()
            },
        )
        .collect();

    // direct CSR construction
    let total_nnz: usize = row_results.iter().map(|r| r.len()).sum();
    let mut data = Vec::with_capacity(total_nnz);
    let mut indices = Vec::with_capacity(total_nnz);
    let mut indptr = Vec::with_capacity(nrows + 1);
    indptr.push(0);

    for row in row_results {
        for (col, val) in row {
            data.push(val);
            indices.push(col);
        }
        indptr.push(data.len());
    }

    CompressedSparseData::new_csr(&data, &indices, &indptr, (nrows, ncols))
}

/////////////
// L1 norm //
/////////////

/// Normalises the rows of a CSR matrix to a sum of 1 (L1 norm)
///
/// ### Params
///
/// * `csr` - Mutable reference to the CSR matrix (modified in-place)
pub fn normalise_csr_rows_l1<T>(csr: &mut CompressedSparseData<T>)
where
    T: ManifoldsFloat,
{
    assert!(csr.cs_type.is_csr(), "Matrix must be in CSR format");
    let nrows = csr.shape.0;

    for i in 0..nrows {
        let start = csr.indptr[i];
        let end = csr.indptr[i + 1];
        let row_data_slice = &mut csr.data[start..end];
        let row_sum: T = row_data_slice.iter().copied().sum();

        if row_sum > T::zero() {
            // multiplications are faster than divisions
            let inv_sum = T::one() / row_sum;
            for val in row_data_slice.iter_mut() {
                *val *= inv_sum;
            }
        }
    }
}

//////////////////////////////
// Powering sparse matrices //
/////////////////////////////

/// Raise a CSR matrix to an integer power (naive repeated multiplication)
///
/// Computes P^t by multiplying P by itself t times.
/// Simple but inefficient for large t. Use for small t or when memory is tight.
///
/// ### Params
///
/// * `matrix` - CSR matrix to power
/// * `t` - Exponent (must be > 0)
///
/// ### Returns
///
/// P^t in CSR format
pub fn matrix_power_naive<T>(matrix: &CompressedSparseData<T>, t: usize) -> CompressedSparseData<T>
where
    T: ManifoldsFloat,
{
    assert!(matrix.cs_type.is_csr(), "Matrix must be CSR format");
    assert!(t > 0, "Power must be positive");

    if t == 1 {
        return matrix.clone();
    }

    assert!(matrix.shape.0 == matrix.shape.1, "Matrix must be square");

    let mut result = matrix.clone();
    for _ in 1..t {
        result = csr_matmul_csr(&result, matrix);
    }
    result
}

/// Raise a CSR matrix to an integer power (binary exponentiation)
///
/// Computes P^t using exponentiation by squaring.
/// Much faster than naive: O(log t) multiplications instead of O(t).
///
/// Examples:
/// - P^8 = ((P^2)^2)^2 (3 multiplications vs 7)
/// - P^15 = P × ((P^2)^2)^2 × P (5 multiplications vs 14)
///
/// ### Params
///
/// * `matrix` - CSR matrix to power
/// * `t` - Exponent (must be > 0)
///
/// ### Returns
///
/// P^t in CSR format
pub fn matrix_power<T>(matrix: &CompressedSparseData<T>, t: usize) -> CompressedSparseData<T>
where
    T: ManifoldsFloat,
{
    assert!(matrix.cs_type.is_csr(), "Matrix must be CSR format");
    assert!(t > 0, "Power must be positive");

    if t == 1 {
        return matrix.clone();
    }

    assert!(matrix.shape.0 == matrix.shape.1, "Matrix must be square");

    // binary exponentiation
    let mut base = matrix.clone();
    let mut result = None;
    let mut exp = t;

    while exp > 0 {
        if exp & 1 == 1 {
            // odd exponent - multiply result by base
            result = Some(match result {
                None => base.clone(),
                Some(r) => csr_matmul_csr(&r, &base),
            });
        }
        exp >>= 1;
        if exp > 0 {
            // square the base for next iteration
            base = csr_matmul_csr(&base, &base);
        }
    }

    result.unwrap()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_matrix_power {
    use super::*;
    use approx::assert_relative_eq;

    fn create_simple_stochastic_matrix() -> CompressedSparseData<f64> {
        // Simple 3x3 row-stochastic matrix
        // Row 0: [0.5, 0.5, 0.0]
        // Row 1: [0.3, 0.4, 0.3]
        // Row 2: [0.0, 0.6, 0.4]
        let data = vec![0.5, 0.5, 0.3, 0.4, 0.3, 0.6, 0.4];
        let indices = vec![0, 1, 0, 1, 2, 1, 2];
        let indptr = vec![0, 2, 5, 7];

        CompressedSparseData::new_csr(&data, &indices, &indptr, (3, 3))
    }

    #[test]
    fn test_matrix_power_t1() {
        let mat = create_simple_stochastic_matrix();
        let result = matrix_power(&mat, 1);

        // Should be identical
        assert_eq!(result.data, mat.data);
        assert_eq!(result.indices, mat.indices);
        assert_eq!(result.indptr, mat.indptr);
    }

    #[test]
    fn test_matrix_power_t2_manual() {
        let mat = create_simple_stochastic_matrix();
        let result = matrix_power(&mat, 2);

        // Manually compute P^2 for row 0:
        // [0.5, 0.5, 0.0] × [[0.5, 0.5, 0.0], [0.3, 0.4, 0.3], [0.0, 0.6, 0.4]]
        // = [0.5*0.5 + 0.5*0.3, 0.5*0.5 + 0.5*0.4, 0.5*0.0 + 0.5*0.3]
        // = [0.4, 0.45, 0.15]

        // Extract row 0 from result
        let row0_start = result.indptr[0];
        let row0_end = result.indptr[1];
        let row0_indices: Vec<usize> = result.indices[row0_start..row0_end].to_vec();
        let row0_data: Vec<f64> = result.data[row0_start..row0_end].to_vec();

        // Should have 3 non-zero entries in row 0
        assert_eq!(row0_indices, vec![0, 1, 2]);
        assert_relative_eq!(row0_data[0], 0.4, epsilon = 1e-10);
        assert_relative_eq!(row0_data[1], 0.45, epsilon = 1e-10);
        assert_relative_eq!(row0_data[2], 0.15, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_power_preserves_row_stochastic() {
        let mat = create_simple_stochastic_matrix();

        for t in [2, 5, 10, 20] {
            let result = matrix_power(&mat, t);

            // Check each row sums to 1.0
            for i in 0..result.shape.0 {
                let start = result.indptr[i];
                let end = result.indptr[i + 1];
                let row_sum: f64 = result.data[start..end].iter().sum();
                assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_matrix_power_vs_naive() {
        let mat = create_simple_stochastic_matrix();

        for t in [2, 3, 5, 7, 15] {
            let result_binary = matrix_power(&mat, t);
            let result_naive = matrix_power_naive(&mat, t);

            // Both should produce identical results
            assert_eq!(result_binary.data.len(), result_naive.data.len());

            for (binary_val, naive_val) in result_binary.data.iter().zip(&result_naive.data) {
                assert_relative_eq!(binary_val, naive_val, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_matrix_power_large_t() {
        let mat = create_simple_stochastic_matrix();
        let result = matrix_power(&mat, 100);

        // Should still be valid and row-stochastic
        assert!(!result.data.is_empty());

        for i in 0..result.shape.0 {
            let start = result.indptr[i];
            let end = result.indptr[i + 1];
            let row_sum: f64 = result.data[start..end].iter().sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_matrix_power_identity() {
        // Create 3x3 identity matrix
        let data = vec![1.0, 1.0, 1.0];
        let indices = vec![0, 1, 2];
        let indptr = vec![0, 1, 2, 3];
        let identity = CompressedSparseData::new_csr(&data, &indices, &indptr, (3, 3));

        // I^t = I for any t
        for t in [1, 2, 5, 100] {
            let result = matrix_power(&identity, t);
            assert_eq!(result.data, vec![1.0, 1.0, 1.0]);
            assert_eq!(result.indices, vec![0, 1, 2]);
        }
    }

    #[test]
    fn test_matrix_power_convergence() {
        // For a strongly connected stochastic matrix,
        // P^t should converge to a rank-1 matrix as t → ∞
        let mat = create_simple_stochastic_matrix();

        let p100 = matrix_power(&mat, 100);
        let p200 = matrix_power(&mat, 200);

        // Rows should become increasingly similar (convergence)
        // Just check that we don't diverge
        for i in 0..p100.shape.0 {
            let start = p100.indptr[i];
            let end = p100.indptr[i + 1];
            let sum100: f64 = p100.data[start..end].iter().sum();
            assert_relative_eq!(sum100, 1.0, epsilon = 1e-8);
        }

        for i in 0..p200.shape.0 {
            let start = p200.indptr[i];
            let end = p200.indptr[i + 1];
            let sum200: f64 = p200.data[start..end].iter().sum();
            assert_relative_eq!(sum200, 1.0, epsilon = 1e-8);
        }
    }

    #[test]
    #[should_panic(expected = "Power must be positive")]
    fn test_matrix_power_zero_t() {
        let mat = create_simple_stochastic_matrix();
        matrix_power(&mat, 0);
    }

    #[test]
    #[should_panic(expected = "Matrix must be square")]
    fn test_matrix_power_non_square() {
        // Create a 2x3 matrix
        let data = vec![1.0, 2.0, 3.0];
        let indices = vec![0, 1, 2];
        let indptr = vec![0, 2, 3];
        let mat = CompressedSparseData::new_csr(&data, &indices, &indptr, (2, 3));

        matrix_power(&mat, 2);
    }
}
