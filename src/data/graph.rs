use num_traits::Float;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::data::structures::*;

/// Smooth kNN distances via binary search to find sigma for each point
///
/// For each point, finds the bandwidth (sigma) such that the sum of
/// similarities to its k nearest neighbours approximates log(k). Uses binary
/// search for efficiency.
///
/// ### Params
///
/// * `dist` - kNN distance matrix where each row contains distances to k
///   nearest neighbours
/// * `k` - Number of nearest neighbours (used to compute target = ln(k))
/// * `local_connectivity` - Number of nearest neighbours to assume are at
///   distance zero (typically 1.0). Allows for local manifold structure.
/// * `bandwidth` - Convergence tolerance for binary search (typically 1e-5)
/// * `n_iter` - Maximum number of binary search iterations (typically 64)
///
/// ### Returns
///
/// * `sigmas` - Smoothing bandwidth for each point
/// * `rhos` - Distance to the `local_connectivity`-th nearest neighbour for
///   each point
pub fn smooth_knn_dist<T>(
    dist: &[Vec<T>],
    k: usize,
    local_connectivity: T,
    bandwidth: T,
    n_iter: usize,
) -> (Vec<T>, Vec<T>)
where
    T: Float + Send + Sync,
{
    dist.par_iter()
        .map(|dists| {
            let target = (k as f64).ln();

            let rho = if local_connectivity > T::zero() {
                let idx = (local_connectivity - T::one()) 
                    .max(T::zero())
                    .floor()
                    .to_usize()
                    .unwrap()
                    .min(dists.len() - 1);

                let fraction = (local_connectivity - T::one()).max(T::zero())
                    - (local_connectivity - T::one()).max(T::zero()).floor();

                if fraction > T::zero() && idx + 1 < dists.len() {
                    dists[idx] * (T::one() - fraction) + dists[idx + 1] * fraction
                } else {
                    dists[idx]
                }
            } else {
                T::zero()
            };

            // Binary search for sigma (rest unchanged)
            let mut lo = T::zero();
            let mut hi = T::max_value();
            let mut mid = T::one();

            for _ in 0..n_iter {
                let mut val = T::zero();
                for &d in dists.iter() {
                    let adjusted = (d - rho).max(T::zero());
                    val = val + (-(adjusted / mid)).exp();
                }

                if (val.to_f64().unwrap() - target).abs() < bandwidth.to_f64().unwrap() {
                    break;
                }

                if val.to_f64().unwrap() > target {
                    hi = mid;
                    mid = (lo + hi) / (T::one() + T::one());
                } else {
                    lo = mid;
                    if hi == T::max_value() {
                        mid = mid * (T::one() + T::one());
                    } else {
                        mid = (lo + hi) / (T::one() + T::one());
                    }
                }
            }

            (mid, rho)
        })
        .unzip()
}

/// Convert kNN graph to sparse COO (Coordinate) format with membership
/// strengths
///
/// Computes fuzzy simplicial set membership strengths based on distances,
/// local connectivity (rho), and smoothed bandwidths (sigma).
///
/// ### Params
///
/// * `knn_indices` - Indices of k nearest neighbours for each point
/// * `knn_dists` - Distances to k nearest neighbours for each point
/// * `sigmas` - Smoothing bandwidth for each point (from `smooth_knn_dist`)
/// * `rhos` - Local connectivity distance for each point (from
///   `smooth_knn_dist`)
///
/// ### Returns
///
/// Sparse graph in COO format where weights represent membership strengths
/// computed as exp(-(max(0, dist - rho) / sigma))
pub fn knn_to_coo<T>(
    knn_indices: &[Vec<usize>],
    knn_dists: &[Vec<T>],
    sigmas: &[T],
    rhos: &[T],
) -> SparseGraph<T>
where
    T: Float + Send + Sync,
{
    let n = knn_indices.len();
    let capacity: usize = knn_indices.iter().map(|v| v.len()).sum();

    let mut row_indices = Vec::with_capacity(capacity);
    let mut col_indices = Vec::with_capacity(capacity);
    let mut values = Vec::with_capacity(capacity);

    for (i, (neighbours, dists)) in knn_indices.iter().zip(knn_dists.iter()).enumerate() {
        let sigma = sigmas[i];
        let rho = rhos[i];

        for (&j, &dist) in neighbours.iter().zip(dists.iter()) {
            if i == j {
                continue;
            }

            let adjusted = (dist - rho).max(T::zero());
            let weight = if sigma > T::zero() {
                (-(adjusted / sigma)).exp()
            } else if adjusted > T::zero() {
                T::zero()
            } else {
                T::one()
            };

            row_indices.push(i);
            col_indices.push(j);
            values.push(weight);
        }
    }

    SparseGraph {
        row_indices,
        col_indices,
        values,
        n_vertices: n,
    }
}

/// Symmetrise graph using probabilistic t-conorm (fuzzy set union)
///
/// Creates symmetric graph by combining directed edges using fuzzy union:
/// w_sym = w_ij + w_ji - w_ij * w_ji, weighted by `mix_weight`.
///
/// ### Params
///
/// * `graph` - Input directed graph in COO format
/// * `mix_weight` - Balance between fuzzy union (0.5) and directed graph (1.0).
///   Controls how much to weight the union operation.
///
/// ### Returns
///
/// Symmetrised graph in COO format
///
/// ### Notes
///
/// * `mix_weight = 1.0`: Full fuzzy union (standard UMAP, symmetric)
/// * `mix_weight = 0.5`: Weighted average of union and directed)
/// * `mix_weight = 0.0`: Use only outgoing edges (directed)
pub fn symmetrise_graph<T>(graph: SparseGraph<T>, mix_weight: T) -> SparseGraph<T>
where
    T: Float + Send + Sync,
{
    let n = graph.n_vertices;

    // Build adjacency maps for fast lookups
    let mut forward: Vec<FxHashMap<usize, T>> = vec![FxHashMap::default(); n];
    let mut backward: Vec<FxHashMap<usize, T>> = vec![FxHashMap::default(); n];

    for ((&i, &j), &w) in graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
    {
        forward[i].insert(j, w);
        backward[j].insert(i, w);
    }

    // Parallel symmetrisation
    let edges: Vec<Vec<(usize, T)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut combined = FxHashMap::default();

            // Process all unique neighbours
            for &j in forward[i].keys().chain(backward[i].keys()) {
                let w_ij = forward[i].get(&j).copied().unwrap_or(T::zero());
                let w_ji = backward[i].get(&j).copied().unwrap_or(T::zero());

                // Fuzzy union: a + b - a*b, weighted by mix_weight
                let union = w_ij + w_ji - w_ij * w_ji;
                let w_sym = mix_weight * union + (T::one() - mix_weight) * w_ij;

                if w_sym > T::zero() {
                    combined.insert(j, w_sym);
                }
            }

            let mut result: Vec<(usize, T)> = combined.into_iter().collect();
            result.sort_unstable_by_key(|&(idx, _)| idx);
            result
        })
        .collect();

    // Flatten back to COO
    let capacity: usize = edges.iter().map(|v| v.len()).sum();
    let mut row_indices = Vec::with_capacity(capacity);
    let mut col_indices = Vec::with_capacity(capacity);
    let mut values = Vec::with_capacity(capacity);

    for (i, neighbours) in edges.into_iter().enumerate() {
        for (j, w) in neighbours {
            row_indices.push(i);
            col_indices.push(j);
            values.push(w);
        }
    }

    SparseGraph {
        row_indices,
        col_indices,
        values,
        n_vertices: n,
    }
}

/// Convert COO sparse graph to adjacency list representation
///
/// More efficient for SGD optimisation where we need to iterate over neighbours
/// of each vertex.
///
/// ### Params
///
/// * `graph` - Sparse graph in COO format
///
/// ### Returns
///
/// Adjacency list where `result[i]` contains `(neighbour_index, edge_weight)`
/// pairs for vertex `i`
pub fn coo_to_adjacency_list<T>(graph: &SparseGraph<T>) -> Vec<Vec<(usize, T)>>
where
    T: Float + Copy,
{
    let mut adj = vec![Vec::new(); graph.n_vertices];

    for ((&i, &j), &w) in graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
    {
        adj[i].push((j, w));
    }

    adj
}

/// Filter out edges that are too weak to be sampled during optimization
///
/// Removes edges where weight < max_weight / n_epochs, matching uwot's
/// preprocessing step. These weak edges would never be sampled during
/// optimization and can cause fragmentation.
///
/// ### Params
///
/// * `graph` - Input graph in COO format
/// * `n_epochs` - Optimization parameters (uses n_epochs for threshold)
///
/// ### Returns
///
/// Filtered graph with weak edges removed
pub fn filter_weak_edges<T>(graph: SparseGraph<T>, n_epochs: usize) -> SparseGraph<T>
where
    T: Float + Send + Sync,
{
    let max_weight = graph
        .values
        .iter()
        .copied()
        .fold(T::zero(), |acc, w| if w > acc { w } else { acc });

    let threshold = max_weight / T::from(n_epochs).unwrap();

    let mut filtered_rows = Vec::new();
    let mut filtered_cols = Vec::new();
    let mut filtered_vals = Vec::new();

    for ((&i, &j), &w) in graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
    {
        if w >= threshold {
            filtered_rows.push(i);
            filtered_cols.push(j);
            filtered_vals.push(w);
        }
    }

    SparseGraph {
        row_indices: filtered_rows,
        col_indices: filtered_cols,
        values: filtered_vals,
        n_vertices: graph.n_vertices,
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_data_gen {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_smooth_knn_dist_basic() {
        // Simple test with 3 points, k=2
        let dist = vec![vec![1.0, 2.0], vec![1.5, 3.0], vec![0.5, 1.5]];

        let (sigmas, rhos) = smooth_knn_dist(&dist, 2, 1.0, 1e-5, 64);

        assert_eq!(sigmas.len(), 3);
        assert_eq!(rhos.len(), 3);

        // Rhos should be approximately the distance to the first neighbour
        assert_relative_eq!(rhos[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(rhos[1], 1.5, epsilon = 1e-4);
        assert_relative_eq!(rhos[2], 0.5, epsilon = 1e-4);

        // Sigmas should be positive
        for sigma in sigmas.iter() {
            assert!(*sigma > 0.0);
        }
    }

    #[test]
    fn test_smooth_knn_dist_zero_local_connectivity() {
        let dist = vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]];

        let (sigmas, rhos) = smooth_knn_dist(&dist, 2, 0.0, 1e-5, 64);

        // With zero local connectivity, rhos should all be zero
        assert!(rhos.iter().all(|&r| r == 0.0));
        assert_eq!(sigmas.len(), 2);
    }

    #[test]
    fn test_knn_to_coo_basic() {
        let knn_indices = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let knn_dists = vec![vec![1.0, 2.0], vec![1.0, 1.5], vec![2.0, 1.5]];
        let sigmas = vec![1.0, 1.0, 1.0];
        let rhos = vec![0.0, 0.0, 0.0];

        let graph = knn_to_coo(&knn_indices, &knn_dists, &sigmas, &rhos);

        assert_eq!(graph.n_vertices, 3);
        assert_eq!(graph.row_indices.len(), 6); // 3 points × 2 neighbours
        assert_eq!(graph.col_indices.len(), 6);
        assert_eq!(graph.values.len(), 6);

        // All weights should be between 0 and 1
        for &w in &graph.values {
            assert!((0.0..=1.0).contains(&w));
        }
    }

    #[test]
    fn test_knn_to_coo_self_loop_excluded() {
        // Include self in neighbours
        let knn_indices = vec![vec![0, 1], vec![1, 0]];
        let knn_dists = vec![vec![0.0, 1.0], vec![0.0, 1.0]];
        let sigmas = vec![1.0, 1.0];
        let rhos = vec![0.0, 0.0];

        let graph = knn_to_coo(&knn_indices, &knn_dists, &sigmas, &rhos);

        // Self-loops should be excluded
        assert_eq!(graph.values.len(), 2); // Only 2 edges, not 4
        assert!(graph
            .row_indices
            .iter()
            .zip(&graph.col_indices)
            .all(|(&i, &j)| i != j));
    }

    #[test]
    fn test_symmetrise_graph_full_union() {
        let graph = SparseGraph {
            row_indices: vec![0, 1],
            col_indices: vec![1, 0],
            values: vec![0.8, 0.6],
            n_vertices: 2,
        };

        let sym_graph = symmetrise_graph(graph, 0.5);

        assert_eq!(sym_graph.n_vertices, 2);

        // With mix_weight = 0.5:
        // union = 0.8 + 0.6 - 0.8*0.6 = 0.92
        // w_sym = 0.5 * union + 0.5 * w_ij
        // For 0->1: 0.5 * 0.92 + 0.5 * 0.8 = 0.86
        // For 1->0: 0.5 * 0.92 + 0.5 * 0.6 = 0.76

        let edges = sym_graph.to_edge_list();
        assert_eq!(edges.len(), 2);

        let edge_01 = edges.iter().find(|&&(i, j, _)| i == 0 && j == 1).unwrap();
        let edge_10 = edges.iter().find(|&&(i, j, _)| i == 1 && j == 0).unwrap();

        assert_relative_eq!(edge_01.2, 0.86, epsilon = 1e-6);
        assert_relative_eq!(edge_10.2, 0.76, epsilon = 1e-6);
    }

    #[test]
    fn test_symmetrise_graph_directed() {
        let graph = SparseGraph {
            row_indices: vec![0, 1],
            col_indices: vec![1, 0],
            values: vec![0.8, 0.6],
            n_vertices: 2,
        };

        // With mix_weight = 1.0, we get full fuzzy union
        // union = 0.8 + 0.6 - 0.8*0.6 = 0.92
        // w_sym = 1.0 * union + 0.0 * w_ij = 0.92 for both edges
        let sym_graph = symmetrise_graph(graph.clone(), 1.0);

        let edges = sym_graph.to_edge_list();
        assert_eq!(edges.len(), 2);

        let union = 0.8 + 0.6 - 0.8 * 0.6;

        for (_, _, w) in edges {
            assert_relative_eq!(w, union, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_coo_to_adjacency_list() {
        let graph = SparseGraph {
            row_indices: vec![0, 0, 1, 2],
            col_indices: vec![1, 2, 2, 0],
            values: vec![0.5, 0.3, 0.8, 0.9],
            n_vertices: 3,
        };

        let adj = coo_to_adjacency_list(&graph);

        assert_eq!(adj.len(), 3);
        assert_eq!(adj[0].len(), 2); // vertex 0 has 2 neighbours
        assert_eq!(adj[1].len(), 1); // vertex 1 has 1 neighbour
        assert_eq!(adj[2].len(), 1); // vertex 2 has 1 neighbour

        assert!(adj[0].contains(&(1, 0.5)));
        assert!(adj[0].contains(&(2, 0.3)));
        assert!(adj[1].contains(&(2, 0.8)));
        assert!(adj[2].contains(&(0, 0.9)));
    }

    #[test]
    fn test_coo_to_adjacency_list_empty() {
        let graph: SparseGraph<f64> = SparseGraph {
            row_indices: vec![],
            col_indices: vec![],
            values: vec![],
            n_vertices: 3,
        };

        let adj = coo_to_adjacency_list(&graph);

        assert_eq!(adj.len(), 3);
        assert!(adj[0].is_empty());
        assert!(adj[1].is_empty());
        assert!(adj[2].is_empty());
    }
}
