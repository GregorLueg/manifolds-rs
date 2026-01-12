use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use thousands::*;

use crate::data::structures::*;

//////////
// UMAP //
//////////

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
///
/// ### Notes
///
/// Used for UMAP
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
pub fn filter_weak_edges<T>(graph: SparseGraph<T>, n_epochs: usize, verbose: bool) -> SparseGraph<T>
where
    T: Float + Send + Sync,
{
    let max_weight = graph
        .values
        .iter()
        .copied()
        .fold(T::zero(), |acc, w| if w > acc { w } else { acc });

    let original_edge_no = graph.col_indices.len();

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

    let filtered_edge_no = filtered_cols.len();

    if verbose {
        println!(
            " Filtered out {} weak edges.",
            (original_edge_no - filtered_edge_no).separate_with_underscores(),
        );
    }

    SparseGraph {
        row_indices: filtered_rows,
        col_indices: filtered_cols,
        values: filtered_vals,
        n_vertices: graph.n_vertices,
    }
}

//////////
// tSNE //
//////////

/// Compute Gaussian affinities from k-nearest neighbours using perplexity-based
/// calibration
///
/// For each point i, computes conditional probabilities p_{j|i} using a
/// Gaussian kernel with bandwidth calibrated via binary search to achieve a
/// target perplexity. The result is a sparse graph where edge (i,j) has weight
/// p_{j|i}.
///
/// ### Params
///
/// * `knn_indices` - For each point, indices of its k nearest neighbours
/// * `knn_dists` - For each point, distances to its k nearest neighbours
///   (same order as indices!)
/// * `perplexity` - Target perplexity (effective number of neighbours). Typical
///   values: 5-50
/// * `tol` - Convergence tolerance for entropy (typical: 1e-5)
/// * `max_iter` - Maximum iterations for binary search (typical: 50-200)
///
/// ### Returns
///
/// A `SparseGraph` containing the asymmetric conditional probabilities p_{j|i}
///
/// ### Notes
///
/// Used for tSNE
pub fn gaussian_knn_affinities<T>(
    knn_indices: &[Vec<usize>],
    knn_dists: &[Vec<T>],
    perplexity: T,
    tol: T,
    max_iter: usize,
) -> SparseGraph<T>
where
    T: Float + Send + Sync + FromPrimitive + ToPrimitive,
{
    let n = knn_indices.len();
    let target_entropy = perplexity.log2();
    let machine_epsilon = T::epsilon();

    let results: Vec<Vec<T>> = knn_indices
        .par_iter()
        .zip(knn_dists.par_iter())
        .map(|(_, dists)| {
            // binary search for precision (beta = 1 / (2*sigma^2))
            let mut beta = T::one();
            let mut min_beta = T::neg_infinity();
            let mut max_beta = T::infinity();
            let mut current_probs = vec![T::zero(); dists.len()];

            for _ in 0..max_iter {
                // compute P_i with current beta: p_{j|i} = exp(-beta * d_{ij}^2)
                let mut sum_p = T::zero();
                for (j, &d) in dists.iter().enumerate() {
                    if d < T::epsilon() {
                        continue;
                    }
                    let p = (-beta * d).exp();
                    current_probs[j] = p;
                    sum_p = sum_p + p;
                }

                // check for numerical stability
                if sum_p.abs() < machine_epsilon {
                    sum_p = machine_epsilon;
                }

                // normalise to get probabilities and compute entropy H
                let mut entropy = T::zero();
                for p in current_probs.iter_mut() {
                    *p = *p / sum_p;
                    if *p > machine_epsilon {
                        entropy = entropy - *p * p.log2();
                    }
                }

                // check convergence
                let entropy_diff = entropy - target_entropy;
                if entropy_diff.abs() < tol {
                    break;
                }

                // adjust beta
                if entropy_diff > T::zero() {
                    // entropy too high → distribution too flat → increase beta (narrow curve)
                    min_beta = beta;
                    if max_beta.is_infinite() {
                        beta = beta * (T::one() + T::one());
                    } else {
                        beta = (beta + max_beta) / (T::one() + T::one());
                    }
                } else {
                    // entropy too low → distribution too peaked → decrease beta (widen curve)
                    max_beta = beta;
                    if min_beta.is_infinite() {
                        beta = beta / (T::one() + T::one());
                    } else {
                        beta = (beta + min_beta) / (T::one() + T::one());
                    }
                }
            }

            current_probs
        })
        .collect();

    // build sparse graph
    let capacity: usize = results.iter().map(|p| p.len()).sum();
    let mut row_indices = Vec::with_capacity(capacity);
    let mut col_indices = Vec::with_capacity(capacity);
    let mut values = Vec::with_capacity(capacity);

    for (i, probs) in results.into_iter().enumerate() {
        for (&j, p) in knn_indices[i].iter().zip(probs) {
            if p > machine_epsilon && j != i {
                row_indices.push(i);
                col_indices.push(j);
                values.push(p);
            }
        }
    }

    SparseGraph {
        row_indices,
        col_indices,
        values,
        n_vertices: n,
    }
}

/// Symmetrise graph for t-SNE: P_sym = (P + P^T) / 2N
///
/// Converts conditional probabilities P(j|i) to symmetric joint probabilities
/// P_ij. This ensures P_ij = P_ji and Σ_ij P_ij = 1.
///
/// ### Params
///
/// * `graph` - Directed sparse graph containing conditional probabilities P(j|i)
///
/// ### Returns
///
/// Symmetric `SparseGraph` where:
/// - Each edge (i,j) has weight P_ij = (P(j|i) + P(i|j)) / 2N
/// - P_ij = P_ji (symmetric)
/// - All weights sum to 1.0
///
/// ### Algorithm
///
/// 1. Build adjacency map for fast lookup of P(j|i)
/// 2. Collect all unique unordered pairs {i,j} from input edges
/// 3. For each pair: compute P_ij = (P(j|i) + P(i|j)) / 2N
/// 4. Add both directions (i,j) and (j,i) to output with weight P_ij
pub fn symmetrise_affinities_tsne<T>(graph: SparseGraph<T>) -> SparseGraph<T>
where
    T: Float + Send + Sync + FromPrimitive,
{
    let n = graph.n_vertices;
    let n_float = T::from_usize(n).unwrap();
    let two = T::from_f64(2.0).unwrap();
    let normalization = two * n_float;

    // Build adjacency map for O(1) lookup
    let mut adj: Vec<FxHashMap<usize, T>> = vec![FxHashMap::default(); n];
    for ((&i, &j), &w) in graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
    {
        adj[i].insert(j, w);
    }

    // Collect all unique unordered pairs {i,j} from edges
    let mut pairs_set: FxHashSet<(usize, usize)> = FxHashSet::default();
    for ((&i, &j), _) in graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
    {
        let pair = if i < j {
            (i, j)
        } else if i > j {
            (j, i)
        } else {
            (i, i)
        };
        pairs_set.insert(pair);
    }

    let pairs: Vec<(usize, usize)> = pairs_set.into_iter().collect();

    // Process each unique pair in parallel
    let edges: Vec<Vec<(usize, usize, T)>> = pairs
        .par_iter()
        .map(|&(i, j)| {
            // Get both directions (may not exist)
            let w_ij = adj[i].get(&j).copied().unwrap_or_else(T::zero);
            let w_ji = adj[j].get(&i).copied().unwrap_or_else(T::zero);
            let p_sym = (w_ij + w_ji) / normalization;

            let mut local_edges = Vec::new();
            if p_sym > T::epsilon() {
                local_edges.push((i, j, p_sym));
                if i != j {
                    local_edges.push((j, i, p_sym));
                }
            }
            local_edges
        })
        .collect();

    // Flatten into final arrays
    let total_capacity: usize = edges.iter().map(|v| v.len()).sum();
    let mut rows = Vec::with_capacity(total_capacity);
    let mut cols = Vec::with_capacity(total_capacity);
    let mut vals = Vec::with_capacity(total_capacity);

    for edge_vec in edges {
        for (i, j, w) in edge_vec {
            rows.push(i);
            cols.push(j);
            vals.push(w);
        }
    }

    SparseGraph {
        row_indices: rows,
        col_indices: cols,
        values: vals,
        n_vertices: n,
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
