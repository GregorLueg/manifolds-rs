use num_traits::Float;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::data_structures::*;

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

            // rho: distance to nearest neighbour (considering local_connectivity)
            let rho = if local_connectivity > T::zero() {
                let idx = local_connectivity
                    .floor()
                    .to_usize()
                    .unwrap()
                    .min(dists.len() - 1);
                let fraction = local_connectivity - local_connectivity.floor();
                if fraction > T::zero() && idx + 1 < dists.len() {
                    dists[idx] * (T::one() - fraction) + dists[idx + 1] * fraction
                } else {
                    dists[idx]
                }
            } else {
                T::zero()
            };

            // Binary search for sigma
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
/// - `mix_weight = 1.0`: Use only outgoing edges (directed)
/// - `mix_weight = 0.5`: Full fuzzy union (standard UMAP, symmetric)
/// - `mix_weight = 0.0`: Use only incoming edges (transpose)
pub fn symmetrize_graph<T>(graph: SparseGraph<T>, mix_weight: T) -> SparseGraph<T>
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
