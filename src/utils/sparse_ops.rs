//! Various sparse operations on the internal `CompressedSparseData` structure.
//! These are designed to be highly efficient and use unsafe under the hood.

use num_traits::Float;
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::ops::{Add, AddAssign, Mul};

use crate::data::structures::*;

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
    T: Float + Sync + Add + PartialEq + Mul,
{
    let mut col_sums: FxHashMap<usize, T> =
        FxHashMap::with_capacity_and_hasher(mat.ncols(), FxBuildHasher);

    for &row_idx in row_indices {
        let start = mat.indptr[row_idx];
        let end = mat.indptr[row_idx + 1];

        for col_idx in start..end {
            let col = mat.indices[col_idx];
            let val = mat.data[col_idx];
            *col_sums.entry(col).or_insert(T::zero()) = *col_sums.get(&col).unwrap() + val;
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
    T: Float + AddAssign,
{
    values: Vec<T>,
    indices: Vec<usize>,
    flags: Vec<bool>,
}

impl<T> SparseAccumulator<T>
where
    T: Float + AddAssign,
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
    unsafe fn add(&mut self, idx: usize, val: T) {
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
    T: Sync + Send + Float + AddAssign,
{
    assert!(a.cs_type.is_csr() && b.cs_type.is_csr());
    assert_eq!(a.shape.1, b.shape.0, "Dimension mismatch");

    let nrows = a.shape.0;
    let ncols = b.shape.1;

    let row_results: Vec<Vec<(usize, T)>> = (0..nrows)
        .into_par_iter()
        .map(|i| {
            let mut acc = SparseAccumulator::new(ncols);

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
                        acc.add(j, a_val * b_val);
                    }
                }
            }

            acc.extract_sorted()
        })
        .collect();

    // Direct CSR construction
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
    T: Float + Send + Sync + Default + Copy + std::iter::Sum<T>,
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
                *val = *val * inv_sum;
            }
        }
    }
}
