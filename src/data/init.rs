use faer::{
    traits::{ComplexField, RealField},
    MatRef,
};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::{
    rngs::StdRng,
    {Rng, SeedableRng},
};
use rand_distr::{Distribution, StandardNormal};
use std::collections::VecDeque;
use std::iter::Sum;

use crate::data::structures::*;
use crate::utils::math::*;

// Different initial ranges for the embedddings
// Defaults based on what works for UMAP

pub const SPECTRAL_RANGE: f64 = 10.0;
pub const RANDOM_RANGE: f64 = 10.0;
pub const PCA_RANGE: f64 = 1.0;

/////////////
// Helpers //
/////////////

/// Different initialisation methods for the UMAP
#[derive(Clone, Debug)]
pub enum EmbdInit<T> {
    /// Spectral initialisation
    SpectralInit { range: Option<T> },
    /// Random initialisation
    RandomInit { range: Option<T> },
    /// PCA initialisation
    PcaInit { range: Option<T>, randomised: bool },
}

/// Parse the respective initialisation
///
/// ### Params
///
/// * `s` - String that defines the initialisation method
/// * `randomised` - Shall randomised SVD be used when initialising via PCA
///
/// ### Returns
///
/// The Option of a EmbdInit
pub fn parse_initilisation<T>(s: &str, randomised: bool, range: Option<T>) -> Option<EmbdInit<T>>
where
    T: Float,
{
    match s.to_lowercase().as_str() {
        "spectral" => Some(EmbdInit::SpectralInit { range }),
        "pca" => Some(EmbdInit::PcaInit { randomised, range }),
        "random" => Some(EmbdInit::RandomInit { range }),
        _ => None,
    }
}

//////////////
// Spectral //
//////////////

/// Convert COO graph to negative normalised adjacency in CSR format
///
/// Computes `M = - D^(-1/2) * A * D^(-1/2)`
/// This allows the Lanczos solver (which finds largest magnitude/smallest
/// algebraic) to converge to the cluster-structure eigenvectors (near -1) much
/// faster than finding Laplacian eigenvectors near 0.
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

    // Build matrix: M = - D^(-1/2) * A * D^(-1/2)
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0];

    for i in 0..n {
        let mut row_entries = vec![];

        for &(j, w) in &adj[i] {
            if i != j {
                // Keep the negative sign
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

/// Find connected components in a sparse graph using BFS
///
/// ### Params
///
/// * `graph` - Sparse graph in COO format
///
/// ### Returns
///
/// Vector of components, where each component is a vector of vertex indices
fn find_connected_components<T>(graph: &SparseGraph<T>) -> Vec<Vec<usize>>
where
    T: Float,
{
    let n = graph.n_vertices;

    // Build adjacency list
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for (&i, &j) in graph.row_indices.iter().zip(&graph.col_indices) {
        adj[i].push(j);
    }

    let mut visited = vec![false; n];
    let mut components = Vec::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }

        let mut component = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited[start] = true;

        while let Some(node) = queue.pop_front() {
            component.push(node);
            for &neighbor in &adj[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        components.push(component);
    }

    components
}

/// Initialise embedding for graphs with multiple connected components
///
/// Places component centroids on a hypersphere and performs spectral
/// embedding within each sufficiently large component. Small components
/// are randomly placed around their centroids.
///
/// ### Params
///
/// * `graph` - Full graph in COO format
/// * `components` - Vector of component vertex indices
/// * `n_comp` - Number of embedding dimensions
/// * `seed` - Random seed
/// * `range` - Scaling range for embedding
///
/// ### Returns
///
/// Initial embedding coordinates
fn multi_component_init<T>(
    graph: &SparseGraph<T>,
    components: &[Vec<usize>],
    n_comp: usize,
    seed: u64,
    range: T,
) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive + Send + Sync + Sum,
{
    let n = graph.n_vertices;
    let n_components = components.len();
    let mut embedding = vec![vec![T::zero(); n_comp]; n];
    let mut rng = StdRng::seed_from_u64(seed);

    // Place component centroids on a circle (or hypersphere for n_comp > 2)
    let centroid_radius = range * T::from_f64(0.6).unwrap();
    let component_spread = range * T::from_f64(0.3).unwrap();

    for (comp_idx, component) in components.iter().enumerate() {
        // Compute centroid position for this component
        let angle = T::from_f64(2.0 * std::f64::consts::PI * comp_idx as f64 / n_components as f64)
            .unwrap();

        let mut centroid = vec![T::zero(); n_comp];
        if n_comp >= 2 {
            centroid[0] = centroid_radius * T::from_f64(angle.to_f64().unwrap().cos()).unwrap();
            centroid[1] = centroid_radius * T::from_f64(angle.to_f64().unwrap().sin()).unwrap();
        }
        // For n_comp > 2, add small random offsets to other dimensions
        for d in 2..n_comp {
            centroid[d] = T::from_f64(rng.random_range(-0.1..0.1)).unwrap() * centroid_radius;
        }

        // If component is large enough, do spectral embedding within it
        if component.len() > n_comp + 1 {
            // Build subgraph for this component
            let subgraph = extract_subgraph(graph, component);
            let sub_embedding = single_component_spectral(
                &subgraph,
                n_comp,
                seed + comp_idx as u64,
                component_spread,
            );

            // Place sub-embedding at centroid
            for (local_idx, &global_idx) in component.iter().enumerate() {
                for d in 0..n_comp {
                    embedding[global_idx][d] = centroid[d] + sub_embedding[local_idx][d];
                }
            }
        } else {
            // Too small for spectral - random placement around centroid
            for &global_idx in component {
                for d in 0..n_comp {
                    let noise = T::from_f64(rng.sample::<f64, _>(StandardNormal)).unwrap()
                        * component_spread
                        * T::from_f64(0.1).unwrap();
                    embedding[global_idx][d] = centroid[d] + noise;
                }
            }
        }
    }

    embedding
}

/// Extract subgraph for a connected component
///
/// Creates a new graph containing only the vertices in the specified component
/// with local index mapping.
///
/// ### Params
///
/// * `graph` - Full graph in COO format
/// * `component` - Vertex indices in this component
///
/// ### Returns
///
/// Subgraph with locally indexed vertices
fn extract_subgraph<T>(graph: &SparseGraph<T>, component: &[usize]) -> SparseGraph<T>
where
    T: Float,
{
    let n = graph.n_vertices;

    // Use Vec for O(1) lookup - faster than HashMap for dense component indices
    let mut global_to_local = vec![None; n];
    for (local, &global) in component.iter().enumerate() {
        global_to_local[global] = Some(local);
    }

    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for ((&i, &j), &v) in graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
    {
        if let (Some(local_i), Some(local_j)) = (global_to_local[i], global_to_local[j]) {
            row_indices.push(local_i);
            col_indices.push(local_j);
            values.push(v);
        }
    }

    SparseGraph {
        row_indices,
        col_indices,
        values,
        n_vertices: component.len(),
    }
}

/// Perform spectral embedding for a single connected component
///
/// Computes eigenvectors of the normalised Laplacian and uses them as
/// embedding coordinates. Falls back to random initialisation for trivially
/// small graphs.
///
/// ### Params
///
/// * `graph` - Connected graph in COO format
/// * `n_comp` - Number of embedding dimensions
/// * `seed` - Random seed
/// * `range` - Scaling range for embedding
///
/// ### Returns
///
/// Initial embedding coordinates
fn single_component_spectral<T>(
    graph: &SparseGraph<T>,
    n_comp: usize,
    seed: u64,
    range: T,
) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive + Send + Sync + Sum,
{
    let n = graph.n_vertices;

    // Handle trivially small graphs
    if n <= n_comp + 1 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut embedding = vec![vec![T::zero(); n_comp]; n];
        for i in 0..n {
            for j in 0..n_comp {
                embedding[i][j] = T::from_f64(rng.random_range(-1.0..1.0)).unwrap() * range;
            }
        }
        return finalise_spectral_embedding(embedding, n_comp, range, seed);
    }

    let laplacian = graph_to_normalised_laplacian(graph);

    let n_eigs = (n_comp + 1).min(n);
    let (_, evecs) = compute_smallest_eigenpairs_lanczos(&laplacian, n_eigs, seed);

    let mut embedding = vec![vec![T::zero(); n_comp]; n];

    // Use eigenvectors 1 through n_comp (skip trivial eigenvector 0)
    for comp_idx in 0..n_comp {
        let evec_idx = comp_idx + 1;
        if evec_idx < evecs[0].len() {
            for i in 0..n {
                embedding[i][comp_idx] = T::from_f32(evecs[i][evec_idx]).unwrap();
            }
        } else {
            let mut rng = StdRng::seed_from_u64(seed + comp_idx as u64);
            for i in 0..n {
                embedding[i][comp_idx] = T::from_f64(rng.random_range(-1.0..1.0)).unwrap();
            }
        }
    }

    finalise_spectral_embedding(embedding, n_comp, range, seed)
}

/// Finalise spectral embedding by centring, scaling and adding noise
///
/// Post-processes raw eigenvector coordinates by centring each dimension,
/// scaling to the specified range, and adding small Gaussian noise for
/// numerical stability.
///
/// ### Params
///
/// * `embedding` - Raw embedding coordinates
/// * `n_comp` - Number of dimensions
/// * `range` - Target range for scaling
/// * `seed` - Random seed for noise
///
/// ### Returns
///
/// Finalised embedding coordinates
fn finalise_spectral_embedding<T>(
    mut embedding: Vec<Vec<T>>,
    n_comp: usize,
    range: T,
    seed: u64,
) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive + Sum,
{
    let n = embedding.len();
    let n_t = T::from_usize(n).unwrap();

    // Centre each component
    for comp in 0..n_comp {
        let mean: T = embedding.iter().map(|v| v[comp]).sum::<T>() / n_t;

        for i in 0..n {
            embedding[i][comp] = embedding[i][comp] - mean;
        }
    }

    // Scale so max absolute value is range
    let max_abs: T = embedding
        .iter()
        .flat_map(|v| v.iter())
        .fold(T::zero(), |acc, &x| {
            let abs_x = x.abs();
            if abs_x > acc {
                abs_x
            } else {
                acc
            }
        });

    if max_abs > T::from_f64(1e-8).unwrap() {
        let scale = range / max_abs;
        for row in &mut embedding {
            for val in row {
                *val = *val * scale;
            }
        }
    }

    // Add small noise for numerical stability
    let mut rng = StdRng::seed_from_u64(seed + 9999);
    let noise_std = T::from_f64(1e-4).unwrap();

    for row in &mut embedding {
        for val in row {
            let noise = T::from_f64(rng.sample::<f64, _>(StandardNormal)).unwrap() * noise_std;
            *val = *val + noise;
        }
    }

    embedding
}

/// Compute spectral layout initialisation for graph
///
/// Uses spectral decomposition of the normalised Laplacian to initialise
/// embedding coordinates. Handles disconnected graphs by placing components
/// separately and performing spectral embedding within each component.
///
/// ### Params
///
/// * `graph` - Symmetric weighted graph in COO format
/// * `n_comp` - Number of embedding dimensions
/// * `seed` - Random seed for reproducibility
/// * `range` - Optional scaling range (defaults to SPECTRAL_RANGE)
///
/// ### Returns
///
/// Initial embedding coordinates for each vertex
pub fn spectral_layout<T>(
    graph: &SparseGraph<T>,
    n_comp: usize,
    seed: u64,
    range: Option<T>,
) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive + Send + Sync + Sum,
{
    let range = range.unwrap_or(T::from_f64(SPECTRAL_RANGE).unwrap());

    // Find connected components
    let components = find_connected_components(graph);

    if components.len() > 1 {
        // Multiple components - place each at different location
        return multi_component_init(graph, &components, n_comp, seed, range);
    }

    // Single connected component - standard spectral embedding
    single_component_spectral(graph, n_comp, seed, range)
}

////////////
// Random //
////////////

/// Random initialisation fallback
///
/// Provides random uniform initialisation matching Python UMAP behaviour.
/// Uses uniform distribution in [-10, 10] range, NOT Gaussian near origin.
///
/// ### Params
///
/// * `n_samples` - Number of samples to initialise
/// * `n_comp` - Dimensionality of the embedding
/// * `seed` - Random seed
///
/// ### Returns
///
/// Random embedding coordinates uniformly distributed in [-10, 10] range
pub fn random_layout<T>(n_samples: usize, n_comp: usize, seed: u64, range: Option<T>) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive + ToPrimitive,
{
    let range = range
        .unwrap_or(T::from_f64(RANDOM_RANGE).unwrap())
        .to_f64()
        .unwrap();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut embedding = vec![vec![T::zero(); n_comp]; n_samples];

    // Use uniform distribution [-10, 10] like Python UMAP, not Gaussian
    for i in 0..n_samples {
        for j in 0..n_comp {
            embedding[i][j] = T::from_f64(rng.random_range(-range..range)).unwrap();
        }
    }

    embedding
}

/////////
// PCA //
/////////

/// PCA-based embedding initialisation
///
/// Scales PCA scores to have reasonable spread like other init methods.
/// Now scales to have standard deviation of ~0.0001 then multiplies by 10,
/// giving coordinates roughly in [-0.003, 0.003] initially.
///
/// ### Params
///
/// * `data` - Input data matrix (samples × features)
/// * `n_comp` - Number of principal components
/// * `randomised` - Whether to use randomised SVD
/// * `seed` - Random seed
///
/// ### Returns
///
/// PCA-based embedding coordinates
pub fn pca_layout<T>(
    data: MatRef<T>,
    n_comp: usize,
    randomised: bool,
    range: Option<T>,
    seed: u64,
) -> Vec<Vec<T>>
where
    T: Float + Send + Sync + Sum + ComplexField + RealField + ToPrimitive + FromPrimitive,
    StandardNormal: Distribution<T>,
{
    let target_std = range.unwrap_or(T::from_f64(PCA_RANGE).unwrap());
    let (n_samples, n_features) = (data.nrows(), data.ncols());

    // Centre the data
    let mut centred = data.to_owned();
    for j in 0..n_features {
        let mean = (0..n_samples).map(|i| data[(i, j)]).sum::<T>() / T::from(n_samples).unwrap();
        for i in 0..n_samples {
            centred[(i, j)] = centred[(i, j)] - mean;
        }
    }

    // Compute SVD
    let svd_result = if randomised {
        randomised_svd(centred.as_ref(), n_comp, seed as usize, None, None)
    } else {
        let svd = centred.thin_svd().unwrap();
        RandomSvdResults {
            u: svd.U().cloned(),
            v: svd.V().cloned(),
            s: svd.S().column_vector().iter().copied().collect(),
        }
    };

    // Project onto first n_comp components: PC scores = U * S
    let u_truncated = svd_result.u.get(.., ..n_comp);
    let s_diagonal = faer::Mat::from_fn(n_comp, n_comp, |i, j| {
        if i == j {
            svd_result.s[i]
        } else {
            T::zero()
        }
    });
    let pca_scores = u_truncated * s_diagonal;

    // Convert to Vec<Vec<T>> and scale to small std like uwot
    let mut embedding = vec![vec![T::zero(); n_comp]; n_samples];

    for comp in 0..n_comp {
        // Extract component values
        let col: Vec<T> = (0..n_samples).map(|i| pca_scores[(i, comp)]).collect();

        // Compute mean and standard deviation
        let mean = col.iter().copied().sum::<T>() / T::from(n_samples).unwrap();
        let variance =
            col.iter().map(|&x| (x - mean) * (x - mean)).sum::<T>() / T::from(n_samples).unwrap();
        let current_std = variance.sqrt();

        // Scale to target std_dev
        let scale_factor = if current_std > T::from_f64(1e-8).unwrap() {
            target_std / current_std
        } else {
            T::one()
        };

        // Apply scaling and centre
        for i in 0..n_samples {
            embedding[i][comp] = (pca_scores[(i, comp)] - mean) * scale_factor;
        }
    }

    embedding
}

//////////
// Main //
//////////

/// Initialise embedding coordinates using specified method
///
/// ### Params
///
/// * `init_method` - Initialization strategy to use
/// * `n_comp` - Target embedding dimensionality (typically 2-3)
/// * `seed` - Random seed for reproducibility
/// * `graph` - Fuzzy simplicial set graph (required for spectral init)
/// * `data` - Original input data (required for PCA init)
///
/// ### Returns
///
/// Initial embedding coordinates as `Vec<Vec<T>>` where outer vector is samples
/// and inner vector is components
///
/// ### Notes
///
/// * **Spectral**: Uses graph Laplacian eigenvectors, scaled to [-10, 10]
/// * **PCA**: Projects onto principal components, scaled to `std_dev = 1e-4`
/// * **Random**: Gaussian random values in [-10, 10] range
pub fn initialise_embedding<T>(
    init_method: &EmbdInit<T>,
    n_comp: usize,
    seed: u64,
    graph: &SparseGraph<T>,
    data: MatRef<T>,
) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive + Send + Sync + Sum + ComplexField + RealField + ToPrimitive,
    StandardNormal: Distribution<T>,
{
    match init_method {
        EmbdInit::SpectralInit { range } => spectral_layout(graph, n_comp, seed, *range),
        EmbdInit::RandomInit { range } => {
            let n_samples = data.nrows();
            random_layout(n_samples, n_comp, seed, *range)
        }
        EmbdInit::PcaInit { randomised, range } => {
            pca_layout(data, n_comp, *randomised, *range, seed)
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_init {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_parse_initialisation() {
        // spectral
        assert!(matches!(
            parse_initilisation::<f32>("spectral", false, None),
            Some(EmbdInit::SpectralInit { range: None })
        ));
        assert!(matches!(
            parse_initilisation::<f32>("SPECTRAL", false, None),
            Some(EmbdInit::SpectralInit { range: None })
        ));
        assert!(matches!(
            parse_initilisation::<f32>("spectral", false, Some(0.01)),
            Some(EmbdInit::SpectralInit { range: Some(0.01) })
        ));
        // random
        assert!(matches!(
            parse_initilisation::<f32>("random", false, None),
            Some(EmbdInit::RandomInit { range: None })
        ));
        assert!(matches!(
            parse_initilisation::<f32>("random", false, Some(0.01)),
            Some(EmbdInit::RandomInit { range: Some(0.01) })
        ));
        //
        assert!(matches!(
            parse_initilisation::<f32>("pca", false, None),
            Some(EmbdInit::PcaInit {
                randomised: false,
                range: None
            })
        ));
        assert!(matches!(
            parse_initilisation::<f32>("pca", true, None),
            Some(EmbdInit::PcaInit {
                randomised: true,
                range: None
            })
        ));
        assert!(matches!(
            parse_initilisation::<f32>("pca", false, Some(0.01)),
            Some(EmbdInit::PcaInit {
                randomised: false,
                range: Some(0.01)
            })
        ));
        // error
        assert!(parse_initilisation::<f32>("invalid", false, None).is_none());
    }

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

        // Only off-diagonal entries (no diagonal in negative normalised adjacency)
        assert_eq!(laplacian.get_nnz(), 2);
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

        let embedding = spectral_layout(&graph, 2, 42, None);

        assert_eq!(embedding.len(), 3); // 3 vertices
        assert_eq!(embedding[0].len(), 2); // 2 dimensions

        // Check that values are approximately in [-10, 10] range (allowing for noise)
        for point in &embedding {
            for &coord in point {
                assert!((-10.01..=10.01).contains(&coord));
            }
        }

        // Check that embedding is centred (mean ≈ 0, allowing for noise)
        for dim in 0..2 {
            let mean: f64 = embedding.iter().map(|p| p[dim]).sum::<f64>() / 3.0;
            assert_relative_eq!(mean, 0.0, epsilon = 0.01);
        }
    }

    #[test]
    fn test_spectral_layout_range_bound() {
        // Create a simple connected graph
        let graph = SparseGraph {
            row_indices: vec![0, 0, 1, 1, 2, 2],
            col_indices: vec![1, 2, 0, 2, 0, 1],
            values: vec![1.0, 0.5, 1.0, 1.0, 0.5, 1.0],
            n_vertices: 3,
        };

        let embedding = spectral_layout(&graph, 2, 42, Some(1.0));

        assert_eq!(embedding.len(), 3); // 3 vertices
        assert_eq!(embedding[0].len(), 2); // 2 dimensions

        // Check that values are approximately in [-10, 10] range (allowing for noise)
        for point in &embedding {
            for &coord in point {
                assert!((-1.01..=1.01).contains(&coord));
            }
        }

        // Check that embedding is centred (mean ≈ 0, allowing for noise)
        for dim in 0..2 {
            let mean: f64 = embedding.iter().map(|p| p[dim]).sum::<f64>() / 3.0;
            assert_relative_eq!(mean, 0.0, epsilon = 0.01);
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

        let embd1 = spectral_layout(&graph, 2, 42, None);
        let embd2 = spectral_layout(&graph, 2, 42, None);

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

        let embedding = spectral_layout(&graph, 3, 42, None);

        assert_eq!(embedding.len(), 4);
        assert_eq!(embedding[0].len(), 3);
    }

    #[test]
    fn test_random_layout_basic() {
        let embedding = random_layout::<f64>(10, 2, 42, None);

        assert_eq!(embedding.len(), 10);
        assert_eq!(embedding[0].len(), 2);

        // Check range [-10, 10]
        for point in &embedding {
            for &coord in point {
                assert!((-10.01..=10.01).contains(&coord));
            }
        }
    }

    #[test]
    fn test_random_layout_basic_rangebound() {
        let embedding = random_layout::<f64>(10, 2, 42, Some(1.0));

        assert_eq!(embedding.len(), 10);
        assert_eq!(embedding[0].len(), 2);

        // Check range [-10, 10]
        for point in &embedding {
            for &coord in point {
                assert!((-1.01..=1.01).contains(&coord));
            }
        }
    }

    #[test]
    fn test_random_layout_reproducibility() {
        let embd1 = random_layout::<f64>(10, 2, 42, None);
        let embd2 = random_layout::<f64>(10, 2, 42, None);

        assert_eq!(embd1, embd2);
    }

    #[test]
    fn test_random_layout_different_seeds() {
        let embd1 = random_layout::<f64>(10, 2, 42, None);
        let embd2 = random_layout::<f64>(10, 2, 999, None);

        assert_ne!(embd1, embd2);
    }

    #[test]
    fn test_random_layout_dimensions() {
        let embedding = random_layout::<f32>(5, 3, 42, None);

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

        let embedding = spectral_layout(&graph, 2, 42, None);

        assert_eq!(embedding.len(), 1);
        assert_eq!(embedding[0].len(), 2);
    }

    #[test]
    fn test_pca_layout_basic() {
        // Data with variance in multiple dimensions
        let data = faer::mat![
            [1.0, 2.0, 1.0],
            [2.0, 3.0, 2.0],
            [3.0, 4.0, 1.5],
            [4.0, 5.0, 2.5],
            [5.0, 6.0, 2.0],
        ];
        let embedding = pca_layout(data.as_ref(), 2, false, None, 42);
        assert_eq!(embedding.len(), 5);
        assert_eq!(embedding[0].len(), 2);

        // Check mean is approximately zero
        for dim in 0..2 {
            let mean: f64 = embedding.iter().map(|p| p[dim]).sum::<f64>() / 5.0;
            assert_relative_eq!(mean, 0.0, epsilon = 1e-6);
        }

        // Check at least first component has std approximately PCA_RANGE
        let mean_0: f64 = embedding.iter().map(|p| p[0]).sum::<f64>() / 5.0;
        let variance_0: f64 = embedding
            .iter()
            .map(|p| (p[0] - mean_0).powi(2))
            .sum::<f64>()
            / 5.0;
        let std_0 = variance_0.sqrt();
        assert_relative_eq!(std_0, PCA_RANGE, epsilon = 1e-4);
    }

    #[test]
    fn test_pca_layout_custom_range() {
        let data = faer::mat![
            [1.0, 2.0, 1.0],
            [2.0, 3.0, 2.0],
            [3.0, 4.0, 1.5],
            [4.0, 5.0, 2.5],
            [5.0, 6.0, 2.0],
        ];
        let custom_range = 1.0;
        let embedding = pca_layout(data.as_ref(), 2, false, Some(custom_range), 42);
        assert_eq!(embedding.len(), 5);
        assert_eq!(embedding[0].len(), 2);

        let mean_0: f64 = embedding.iter().map(|p| p[0]).sum::<f64>() / 5.0;
        let variance_0: f64 = embedding
            .iter()
            .map(|p| (p[0] - mean_0).powi(2))
            .sum::<f64>()
            / 5.0;
        let std_0 = variance_0.sqrt();
        assert_relative_eq!(std_0, custom_range, epsilon = 1e-4);
    }

    #[test]
    fn test_pca_layout_reproducibility() {
        let data = faer::mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let embd1 = pca_layout(data.as_ref(), 2, false, None, 42);
        let embd2 = pca_layout(data.as_ref(), 2, false, None, 42);

        assert_eq!(embd1, embd2);
    }

    #[test]
    fn test_pca_layout_randomised() {
        let data = faer::mat![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ];

        let embd_standard = pca_layout(data.as_ref(), 2, false, None, 42);
        let embd_randomised = pca_layout(data.as_ref(), 2, true, None, 42);

        assert_eq!(embd_standard.len(), 3);
        assert_eq!(embd_randomised.len(), 3);
    }

    #[test]
    fn test_initialise_embedding_spectral() {
        let graph = SparseGraph {
            row_indices: vec![0, 1],
            col_indices: vec![1, 0],
            values: vec![1.0, 1.0],
            n_vertices: 2,
        };
        let data = faer::mat![[1.0, 2.0], [3.0, 4.0],];

        let embedding = initialise_embedding(
            &EmbdInit::SpectralInit { range: None },
            2,
            42,
            &graph,
            data.as_ref(),
        );

        assert_eq!(embedding.len(), 2);
        assert_eq!(embedding[0].len(), 2);
    }

    #[test]
    fn test_initialise_embedding_random() {
        let graph = SparseGraph {
            row_indices: vec![],
            col_indices: vec![],
            values: vec![],
            n_vertices: 3,
        };
        let data = faer::mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let embedding = initialise_embedding(
            &EmbdInit::RandomInit { range: None },
            2,
            42,
            &graph,
            data.as_ref(),
        );

        assert_eq!(embedding.len(), 3);
        assert_eq!(embedding[0].len(), 2);
    }

    #[test]
    fn test_initialise_embedding_pca() {
        let graph = SparseGraph {
            row_indices: vec![],
            col_indices: vec![],
            values: vec![],
            n_vertices: 3,
        };
        let data = faer::mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];

        let embedding = initialise_embedding(
            &EmbdInit::PcaInit {
                randomised: false,
                range: None,
            },
            2,
            42,
            &graph,
            data.as_ref(),
        );

        assert_eq!(embedding.len(), 3);
        assert_eq!(embedding[0].len(), 2);
    }
}
