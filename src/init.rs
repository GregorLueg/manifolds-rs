use num_traits::{Float, FromPrimitive};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::data_struct::*;
use crate::utils_math::*;

/// Convert COO graph to normalised Laplacian in CSR format
///
/// Computes L = I - D^(-1/2) * A * D^(-1/2) where D is the degree matrix
///
/// ### Params
///
/// * `graph` - Symmetric weighted graph in COO format
///
/// ### Returns
///
/// Normalised Laplacian as CSR matrix
fn graph_to_normalised_laplacian<T>(graph: &SparseGraph<T>) -> CompressedSparseData<f64>
where
    T: Float,
{
    let n = graph.n_vertices;

    // Build adjacency list and compute degrees in one pass
    let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
    let mut degrees = vec![0.0; n];

    for ((&i, &j), &w) in graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
    {
        let w_f64 = w.to_f64().unwrap();
        adj[i].push((j, w_f64));
        degrees[i] += w_f64;
    }

    // Compute D^(-1/2), handling isolated vertices
    let d_inv_sqrt: Vec<f64> = degrees
        .iter()
        .map(|&d| if d > 1e-8 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();

    // Build normalised Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0];

    for i in 0..n {
        // Collect and sort this row's entries
        let mut row_entries = vec![(i, 1.0)]; // diagonal

        for &(j, w) in &adj[i] {
            if i != j {
                let normalised_weight = -d_inv_sqrt[i] * w * d_inv_sqrt[j];
                row_entries.push((j, normalised_weight));
            }
        }

        row_entries.sort_unstable_by_key(|&(idx, _)| idx);

        for (idx, val) in row_entries {
            indices.push(idx);
            data.push(val);
        }

        indptr.push(data.len());
    }

    CompressedSparseData::new_csr(&data, &indices, &indptr, (n, n))
}

/// Spectral embedding initialisation via graph Laplacian eigenvectors
///
/// Computes the first `n_comp` eigenvectors of the normalised graph
/// Laplacian as initial embedding coordinates using Lanczos iteration.
///
/// ### Params
///
/// * `graph` - Symmetric weighted graph in COO format
/// * `n_comp` - Dimensionality of the embedding (typically 2 or 3)
/// * `seed` - Random seed for reproducibility
///
/// ### Returns
///
/// Initial embedding coordinates as `Vec<Vec<T>>` where outer vector is samples
/// and inner vector is coordinates
///
/// ### Notes
///
/// Uses the smallest non-trivial eigenvectors (skipping the constant
/// eigenvector). The result is centred and scaled to [-10, 10] range.
pub fn spectral_layout<T>(graph: &SparseGraph<T>, n_comp: usize, seed: u64) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive + Send + Sync,
{
    let n = graph.n_vertices;

    // convert to normalised Laplacian
    let laplacian = graph_to_normalised_laplacian(graph);

    // compute smallest eigenvectors (skip first which is constant)
    let n_eigs = (n_comp + 1).min(n);
    let (_, evecs) = compute_largest_eigenpairs_lanczos(&laplacian, n_eigs, seed);

    let mut embedding = vec![vec![T::zero(); n_comp]; n];

    // take eigenvectors 1 to n_components (skip the trivial one at index 0)
    for comp_idx in 0..n_comp {
        let evec_idx = comp_idx + 1;
        if evec_idx < evecs[0].len() {
            for i in 0..n {
                embedding[i][comp_idx] = T::from_f32(evecs[i][evec_idx]).unwrap();
            }
        } else {
            // fallback to random if not enough eigenvectors
            let mut rng = StdRng::seed_from_u64(seed + comp_idx as u64);
            for i in 0..n {
                embedding[i][comp_idx] = T::from_f64(rng.random_range(-10.0..10.0)).unwrap();
            }
        }
    }

    // centre and scale each component
    for comp in 0..n_comp {
        let mean: T = embedding
            .iter()
            .map(|v| v[comp])
            .fold(T::zero(), |acc, x| acc + x)
            / T::from_usize(n).unwrap();

        for i in 0..n {
            embedding[i][comp] = embedding[i][comp] - mean;
        }

        let max_val = embedding
            .iter()
            .map(|v| v[comp].abs())
            .fold(T::zero(), |acc, x| if x > acc { x } else { acc });

        if max_val > T::from_f64(1e-8).unwrap() {
            let scale = T::from_f64(10.0).unwrap() / max_val;
            for i in 0..n {
                embedding[i][comp] = embedding[i][comp] * scale;
            }
        }
    }

    embedding
}

/// Random initialisation fallback
///
/// Provides random Gaussian initialisation when spectral embedding isn't
/// suitable (e.g., disconnected graphs, very large datasets).
///
/// ### Params
///
/// * `n_samples` - Number of samples to initialise
/// * `n_comp` - Dimensionality of the embedding
/// * `seed` - Random seed
///
/// ### Returns
///
/// Random embedding coordinates scaled to [-10, 10] range
pub fn random_layout<T>(n_samples: usize, n_comp: usize, seed: u64) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut embedding = vec![vec![T::zero(); n_comp]; n_samples];

    for i in 0..n_samples {
        for j in 0..n_comp {
            embedding[i][j] = T::from_f64(rng.random_range(-10.0..10.0)).unwrap();
        }
    }

    embedding
}

#[cfg(test)]
mod test_init {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_graph_to_normalised_laplacian_simple() {
        // Simple graph: 0 <-> 1 with equal weights
        let graph = SparseGraph {
            row_indices: vec![0, 1],
            col_indices: vec![1, 0],
            values: vec![1.0, 1.0],
            n_vertices: 2,
        };

        let laplacian = graph_to_normalised_laplacian(&graph);

        assert_eq!(laplacian.shape(), (2, 2));
        assert!(laplacian.cs_type.is_csr());

        // Check structure: should have diagonal entries (1.0) and off-diagonal (-1.0)
        assert_eq!(laplacian.get_nnz(), 4); // 2 diagonal + 2 off-diagonal
    }

    #[test]
    fn test_graph_to_normalised_laplacian_isolated_vertex() {
        // Graph with isolated vertex
        let graph = SparseGraph {
            row_indices: vec![0],
            col_indices: vec![1],
            values: vec![1.0],
            n_vertices: 3,
        };

        let laplacian = graph_to_normalised_laplacian(&graph);

        assert_eq!(laplacian.shape(), (3, 3));
        // Should handle isolated vertex (vertex 2) gracefully
    }

    #[test]
    fn test_spectral_layout_basic() {
        // Create a simple connected graph
        let graph = SparseGraph {
            row_indices: vec![0, 0, 1, 1, 2, 2],
            col_indices: vec![1, 2, 0, 2, 0, 1],
            values: vec![1.0, 0.5, 1.0, 1.0, 0.5, 1.0],
            n_vertices: 3,
        };

        let embedding = spectral_layout(&graph, 2, 42);

        assert_eq!(embedding.len(), 3); // 3 vertices
        assert_eq!(embedding[0].len(), 2); // 2 dimensions

        // Check that values are scaled to [-10, 10] range
        for point in &embedding {
            for &coord in point {
                assert!((-10.0..=10.0).contains(&coord));
            }
        }

        // Check that embedding is centred (mean ≈ 0)
        for dim in 0..2 {
            let mean: f64 = embedding.iter().map(|p| p[dim]).sum::<f64>() / 3.0;
            assert_relative_eq!(mean, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_spectral_layout_reproducibility() {
        let graph = SparseGraph {
            row_indices: vec![0, 1, 2],
            col_indices: vec![1, 2, 0],
            values: vec![1.0, 1.0, 1.0],
            n_vertices: 3,
        };

        let embd1 = spectral_layout(&graph, 2, 42);
        let embd2 = spectral_layout(&graph, 2, 42);

        assert_eq!(embd1, embd2);
    }

    #[test]
    fn test_spectral_layout_higher_dimensions() {
        let graph = SparseGraph {
            row_indices: vec![0, 1, 2, 3],
            col_indices: vec![1, 2, 3, 0],
            values: vec![1.0; 4],
            n_vertices: 4,
        };

        let embedding = spectral_layout(&graph, 3, 42);

        assert_eq!(embedding.len(), 4);
        assert_eq!(embedding[0].len(), 3);
    }

    #[test]
    fn test_random_layout_basic() {
        let embedding = random_layout::<f64>(10, 2, 42);

        assert_eq!(embedding.len(), 10);
        assert_eq!(embedding[0].len(), 2);

        // Check range [-10, 10]
        for point in &embedding {
            for &coord in point {
                assert!((-10.0..=10.0).contains(&coord));
            }
        }
    }

    #[test]
    fn test_random_layout_reproducibility() {
        let embd1 = random_layout::<f64>(10, 2, 42);
        let embd2 = random_layout::<f64>(10, 2, 42);

        assert_eq!(embd1, embd2);
    }

    #[test]
    fn test_random_layout_different_seeds() {
        let embd1 = random_layout::<f64>(10, 2, 42);
        let embd2 = random_layout::<f64>(10, 2, 999);

        assert_ne!(embd1, embd2);
    }

    #[test]
    fn test_random_layout_dimensions() {
        let embedding = random_layout::<f32>(5, 3, 42);

        assert_eq!(embedding.len(), 5);
        assert_eq!(embedding[0].len(), 3);
    }

    #[test]
    fn test_spectral_layout_single_vertex() {
        let graph: SparseGraph<f32> = SparseGraph {
            row_indices: vec![],
            col_indices: vec![],
            values: vec![],
            n_vertices: 1,
        };

        let embedding = spectral_layout(&graph, 2, 42);

        assert_eq!(embedding.len(), 1);
        assert_eq!(embedding[0].len(), 2);
    }
}
