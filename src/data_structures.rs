use num_traits::Float;

/// Sparse graph in COO (Coordinate) format - tensor-friendly
///
/// ### Fields
///
/// * `row_indices` - Row index
/// * `col_indices` - Column index
/// * `values` - The value stored here
/// * `n_vertices` - The number of vertices in the graph
#[derive(Clone)]
pub struct SparseGraph<T> {
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<T>,
    pub n_vertices: usize,
}

impl<T> SparseGraph<T>
where
    T: Float,
{
    /// Generate an edge list from the COO
    pub fn to_edge_list(&self) -> Vec<(usize, usize, T)> {
        self.row_indices
            .iter()
            .zip(&self.col_indices)
            .zip(&self.values)
            .map(|((&r, &c), &v)| (r, c, v))
            .collect()
    }
}
