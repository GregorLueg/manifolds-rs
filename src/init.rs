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
