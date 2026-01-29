use num_traits::Float;
use rayon::prelude::*;
use std::ops::{Add, Mul};

/////////////////////
// Data structures //
/////////////////////

/////////
// COO //
/////////

/// Coordinate list
///
/// Represents the graph in COO (Coordinate) format - tensor-friendly
///
/// ### Fields
///
/// * `row_indices` - Row index
/// * `col_indices` - Column index
/// * `values` - The value stored here
/// * `n_samples` - The number of vertices in the graph
#[derive(Clone)]
pub struct CoordinateList<T> {
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<T>,
    pub n_samples: usize,
}

impl<T> CoordinateList<T>
where
    T: Float,
{
    /// Generate an edge list from the COO
    ///
    /// ### Returns
    ///
    /// A vector of tuples representing the edges and their weights
    pub fn to_edge_list(&self) -> Vec<(usize, usize, T)> {
        self.row_indices
            .iter()
            .zip(&self.col_indices)
            .zip(&self.values)
            .map(|((&r, &c), &v)| (r, c, v))
            .collect()
    }

    /// Returns the size of the graph
    ///
    /// ### Returns
    ///
    /// The number of edges in the graph
    pub fn get_size(&self) -> usize {
        self.row_indices.len()
    }
}

/////////////
// CSR/CSC //
/////////////

/// Type to describe the CompressedSparseFormat
#[derive(Debug, Clone)]
pub enum CompressedSparseFormat {
    /// CSC-formatted data
    Csc,
    /// CSR-formatted data
    Csr,
}

impl CompressedSparseFormat {
    /// Returns boolean if it's CSC
    pub fn is_csc(&self) -> bool {
        matches!(self, CompressedSparseFormat::Csc)
    }
    /// Returns boolean if it's CSR
    pub fn is_csr(&self) -> bool {
        matches!(self, CompressedSparseFormat::Csr)
    }
}

/// Structure to store compressed sparse data of either type
///
/// Ported over from the bixverse code; removed the second data structure here.
///
/// ### Fields
///
/// * `data` - The values
/// * `indices` - The indices of the values
/// * `indptr` - The index pointers
/// * `cs_type` - Is the data stored in `Csr` or `Csc`.
/// * `shape` - The shape of the underlying matrix
#[derive(Debug, Clone)]
pub struct CompressedSparseData<T>
where
    T: Clone + Float,
{
    pub data: Vec<T>,
    pub indices: Vec<usize>,
    pub indptr: Vec<usize>,
    pub cs_type: CompressedSparseFormat,
    pub shape: (usize, usize),
}

impl<T> CompressedSparseData<T>
where
    T: Clone + Sync + Add + PartialEq + Mul + Float,
{
    /// Generate a nes CSC version of the matrix
    ///
    /// ### Params
    ///
    /// * `data` - The underlying data
    /// * `indices` - The index positions (in this case row indices)
    /// * `indptr` - The index pointer (in this case the column index pointers)
    /// * `data2` - An optional second layer
    #[allow(dead_code)]
    pub fn new_csc(data: &[T], indices: &[usize], indptr: &[usize], shape: (usize, usize)) -> Self {
        Self {
            data: data.to_vec(),
            indices: indices.to_vec(),
            indptr: indptr.to_vec(),
            cs_type: CompressedSparseFormat::Csc,
            shape,
        }
    }

    /// Generate a nes CSR version of the matrix
    ///
    /// ### Params
    ///
    /// * `data` - The underlying data
    /// * `indices` - The index positions (in this case row indices)
    /// * `indptr` - The index pointer (in this case the column index pointers)
    /// * `data2` - An optional second layer
    pub fn new_csr(data: &[T], indices: &[usize], indptr: &[usize], shape: (usize, usize)) -> Self {
        Self {
            data: data.to_vec(),
            indices: indices.to_vec(),
            indptr: indptr.to_vec(), // Fixed: was using indices instead of indptr
            cs_type: CompressedSparseFormat::Csr,
            shape,
        }
    }

    /// Transform from CSC to CSR or vice versa
    ///
    /// ### Returns
    ///
    /// The transformed/transposed version
    pub fn transform(&self) -> Self {
        match self.cs_type {
            CompressedSparseFormat::Csc => csc_to_csr(self),
            CompressedSparseFormat::Csr => csr_to_csc(self),
        }
    }

    /// Returns the shape of the matrix
    ///
    /// ### Returns
    ///
    /// A tuple of `(nrow, ncol)`
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Returns the NNZ
    ///
    /// ### Returns
    ///
    /// The number of NNZ
    pub fn get_nnz(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of rows
    ///
    /// ### Returns
    ///
    /// The number of rows
    pub fn nrows(&self) -> usize {
        self.shape.0
    }

    /// Returns the number of columns
    ///
    /// ### Returns
    ///
    /// The number of columns
    pub fn ncols(&self) -> usize {
        self.shape.1
    }
}

/// Transforms a CompressedSparseData that is CSC to CSR
///
/// ### Params
///
/// * `sparse_data` - The CompressedSparseData you want to transform
///
/// ### Returns
///
///
pub fn csc_to_csr<T>(sparse_data: &CompressedSparseData<T>) -> CompressedSparseData<T>
where
    T: Clone + Sync + Add + PartialEq + Mul + Float,
{
    // early return if already in the desired format
    if sparse_data.cs_type.is_csr() {
        return sparse_data.clone();
    }

    let (nrow, _) = sparse_data.shape();
    let nnz = sparse_data.get_nnz();
    let mut row_ptr = vec![0; nrow + 1];

    for &r in &sparse_data.indices {
        row_ptr[r + 1] += 1;
    }

    for i in 0..nrow {
        row_ptr[i + 1] += row_ptr[i];
    }

    let mut csr_data = vec![T::zero(); nnz];
    let mut csr_col_ind = vec![0; nnz];
    let mut next = row_ptr[..nrow].to_vec();

    for col in 0..(sparse_data.indptr.len() - 1) {
        for idx in sparse_data.indptr[col]..sparse_data.indptr[col + 1] {
            let row = sparse_data.indices[idx];
            let pos = next[row];

            csr_data[pos] = sparse_data.data[idx];
            csr_col_ind[pos] = col;

            next[row] += 1;
        }
    }

    CompressedSparseData {
        data: csr_data,
        indices: csr_col_ind,
        indptr: row_ptr,
        cs_type: CompressedSparseFormat::Csr,
        shape: sparse_data.shape(),
    }
}

/// Transform CSR stored data into CSC stored data
///
/// This version does a full memory copy of the data.
///
/// ### Params
///
/// * `sparse_data` - The CompressedSparseData you want to transform. Needs
///   to be in CSR format.
///
/// ### Returns
///
/// The data in CSC format, i.e., `CompressedSparseData`
pub fn csr_to_csc<T>(sparse_data: &CompressedSparseData<T>) -> CompressedSparseData<T>
where
    T: Clone + Sync + Add + PartialEq + Mul + Float,
{
    // early return if already in the desired format
    if sparse_data.cs_type.is_csc() {
        return sparse_data.clone();
    }

    let nnz = sparse_data.get_nnz();
    let (_, ncol) = sparse_data.shape();
    let mut col_ptr = vec![0; ncol + 1];

    // Count occurrences per column
    for &c in &sparse_data.indices {
        col_ptr[c + 1] += 1;
    }

    // Cumulative sum to get column pointers
    for i in 0..ncol {
        col_ptr[i + 1] += col_ptr[i];
    }

    let mut csc_data = vec![T::zero(); nnz];
    let mut csc_row_ind = vec![0; nnz];
    let mut next = col_ptr[..ncol].to_vec();

    // Iterate through rows and place data in CSC format
    for row in 0..(sparse_data.indptr.len() - 1) {
        for idx in sparse_data.indptr[row]..sparse_data.indptr[row + 1] {
            let col = sparse_data.indices[idx];
            let pos = next[col];

            csc_data[pos] = sparse_data.data[idx];
            csc_row_ind[pos] = row;

            next[col] += 1;
        }
    }

    CompressedSparseData {
        data: csc_data,
        indices: csc_row_ind,
        indptr: col_ptr,
        cs_type: CompressedSparseFormat::Csc,
        shape: sparse_data.shape(),
    }
}

////////////////
// SparseRows //
////////////////

/// SparseRow represents a row in a sparse matrix.
///
/// ### Fields
///
/// * `data` - Vector of non-zero values in the row
/// * `indices` - Vector of column indices corresponding to non-zero values
pub struct SparseRow<T> {
    pub indices: Vec<usize>,
    pub data: Vec<T>,
}

////////////////
// Conversion //
////////////////

/// Convert COO CoordinateList to CSR CompressedSparseData
///
/// Converts coordinate format to Compressed Sparse Row format. This is required
/// for efficient row-based operations like normalization and matrix multiplication.
///
/// Uses parallel sorting for performance.
///
/// ### Params
///
/// * `graph` - Input graph in COO format
///
/// ### Returns
///
/// Matrix in CSR format with shape (n_samples, n_samples)
pub fn coo_to_csr<T>(graph: &CoordinateList<T>) -> CompressedSparseData<T>
where
    T: Float + Send + Sync + Default,
{
    let n = graph.n_samples;
    let nnz = graph.values.len();

    let mut triplets: Vec<(usize, usize, T)> = (0..nnz)
        .into_par_iter()
        .map(|i| (graph.row_indices[i], graph.col_indices[i], graph.values[i]))
        .collect();

    triplets.par_sort_unstable_by(|(r1, c1, _), (r2, c2, _)| r1.cmp(r2).then(c1.cmp(c2)));

    let mut data = Vec::with_capacity(nnz);
    let mut indices = Vec::with_capacity(nnz);

    for (_, c, v) in triplets.iter() {
        data.push(*v);
        indices.push(*c);
    }

    let mut indptr = vec![0; n + 1];

    for (r, _, _) in triplets.iter() {
        indptr[r + 1] += 1;
    }

    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }

    CompressedSparseData::new_csr(&data, &indices, &indptr, (n, n))
}

/// Transform a slice of rows into a CSR matrix
///
/// ### Params
///
/// * `rows` - Slice of sparse rows to convert
/// * `ncols` - Number of columns in the matrix
///
/// ### Returns
///
/// A CompressedSparseData structure in CSR format
pub fn sparse_row_to_csr<T>(rows: &[SparseRow<T>], ncols: usize) -> CompressedSparseData<T>
where
    T: Clone + Float + Send + Sync,
{
    let nrows = rows.len();
    let nnz: usize = rows.iter().map(|row| row.data.len()).sum();

    let mut data = Vec::with_capacity(nnz);
    let mut indices = Vec::with_capacity(nnz);
    let mut indptr = Vec::with_capacity(nrows + 1);

    indptr.push(0);
    for row in rows {
        data.extend_from_slice(&row.data);
        indices.extend_from_slice(&row.indices);
        indptr.push(data.len());
    }

    CompressedSparseData::new_csr(&data, &indices, &indptr, (nrows, ncols))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_data_struct {
    use super::*;

    #[test]
    fn test_sparse_graph_to_edge_list() {
        let graph = CoordinateList {
            row_indices: vec![0, 0, 1, 2],
            col_indices: vec![1, 2, 2, 0],
            values: vec![1.0, 2.0, 3.0, 4.0],
            n_samples: 3,
        };

        let edges = graph.to_edge_list();
        assert_eq!(edges.len(), 4);
        assert_eq!(edges[0], (0, 1, 1.0));
        assert_eq!(edges[1], (0, 2, 2.0));
        assert_eq!(edges[2], (1, 2, 3.0));
        assert_eq!(edges[3], (2, 0, 4.0));
    }

    #[test]
    fn test_compressed_sparse_format() {
        let csc = CompressedSparseFormat::Csc;
        let csr = CompressedSparseFormat::Csr;

        assert!(csc.is_csc());
        assert!(!csc.is_csr());
        assert!(csr.is_csr());
        assert!(!csr.is_csc());
    }

    #[test]
    fn test_csr_to_csc_conversion() {
        // Create a simple 3x3 CSR matrix:
        // [1.0  0   2.0]
        // [0    3.0 0  ]
        // [4.0  0   5.0]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let indices = vec![0, 2, 1, 0, 2]; // column indices
        let indptr = vec![0, 2, 3, 5]; // row pointers

        let csr = CompressedSparseData::new_csr(&data, &indices, &indptr, (3, 3));
        let csc = csr.transform();

        assert!(csc.cs_type.is_csc());
        assert_eq!(csc.shape(), (3, 3));
        assert_eq!(csc.get_nnz(), 5);

        // Check CSC structure
        // Column 0: rows 0, 2 -> values 1.0, 4.0
        // Column 1: row 1 -> value 3.0
        // Column 2: rows 0, 2 -> values 2.0, 5.0
        assert_eq!(csc.indptr, vec![0, 2, 3, 5]);
    }

    #[test]
    fn test_csc_to_csr_conversion() {
        // Create a simple 3x3 CSC matrix
        let data = vec![1.0, 4.0, 3.0, 2.0, 5.0];
        let indices = vec![0, 2, 1, 0, 2]; // row indices
        let indptr = vec![0, 2, 3, 5]; // column pointers

        let csc = CompressedSparseData::new_csc(&data, &indices, &indptr, (3, 3));
        let csr = csc.transform();

        assert!(csr.cs_type.is_csr());
        assert_eq!(csr.shape(), (3, 3));
        assert_eq!(csr.get_nnz(), 5);
    }

    #[test]
    fn test_transform_roundtrip() {
        let data = vec![1.0, 2.0, 3.0];
        let indices = vec![0, 1, 2];
        let indptr = vec![0, 1, 2, 3];

        let csr = CompressedSparseData::new_csr(&data, &indices, &indptr, (3, 3));
        let csc = csr.transform();
        let csr_again = csc.transform();

        assert!(csr_again.cs_type.is_csr());
        assert_eq!(csr_again.get_nnz(), csr.get_nnz());
        assert_eq!(csr_again.shape(), csr.shape());
    }

    #[test]
    fn test_empty_sparse_matrix() {
        let data: Vec<f64> = vec![];
        let indices: Vec<usize> = vec![];
        let indptr = vec![0, 0, 0];

        let csr = CompressedSparseData::new_csr(&data, &indices, &indptr, (2, 2));
        assert_eq!(csr.get_nnz(), 0);

        let csc = csr.transform();
        assert_eq!(csc.get_nnz(), 0);
    }

    #[test]
    fn test_coo_to_csr_sorting_and_structure() {
        // Construct a COO graph with unsorted entries and mixed row orders
        // (0,1)=1.0, (1,2)=3.0, (0,2)=2.0
        let graph = CoordinateList {
            row_indices: vec![0, 1, 0],
            col_indices: vec![1, 2, 2],
            values: vec![1.0, 3.0, 2.0],
            n_samples: 3,
        };

        let csr = coo_to_csr(&graph);

        // 1. Verify Structure
        assert!(csr.cs_type.is_csr());
        assert_eq!(csr.shape(), (3, 3));
        assert_eq!(csr.get_nnz(), 3);

        // 2. Verify Sorting (Row 0 should come before Row 1, and (0,1) before (0,2))
        // Expected order: (0,1,1.0), (0,2,2.0), (1,2,3.0)
        // Data: [1.0, 2.0, 3.0]
        // Indices: [1, 2, 2]
        // Indptr: [0, 2, 3, 3] (Row 0 has 2 items, Row 1 has 1 item, Row 2 empty)

        assert_eq!(csr.data, vec![1.0, 2.0, 3.0]);
        assert_eq!(csr.indices, vec![1, 2, 2]);
        assert_eq!(csr.indptr, vec![0, 2, 3, 3]);
    }

    #[test]
    fn test_coo_to_csr_empty_rows_and_gaps() {
        // Graph with 4 vertices, but only edges on row 0 and row 3
        let graph = CoordinateList {
            row_indices: vec![0, 3],
            col_indices: vec![1, 2],
            values: vec![10.0, 20.0],
            n_samples: 4,
        };

        let csr = coo_to_csr(&graph);

        // Indptr should reflect empty rows 1 and 2
        // Row 0: len 1 -> start 0, end 1
        // Row 1: len 0 -> start 1, end 1
        // Row 2: len 0 -> start 1, end 1
        // Row 3: len 1 -> start 1, end 2
        // Indptr: [0, 1, 1, 1, 2]

        assert_eq!(csr.indptr, vec![0, 1, 1, 1, 2]);
        assert_eq!(csr.data, vec![10.0, 20.0]);
        assert_eq!(csr.indices, vec![1, 2]);
    }
}
