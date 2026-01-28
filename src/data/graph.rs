use std::ops::AddAssign;

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
/// * `distances_squared` - If true, distances are already squared (e.g.,
///   squared Euclidean). If false, distances will be squared before computing
///   the kernel.
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
    distances_squared: bool,
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
                // compute P_i with current beta: p_{j|i} = exp(-beta * d²)
                let mut sum_p = T::zero();
                for (j, &d) in dists.iter().enumerate() {
                    if d < T::epsilon() {
                        continue;
                    }
                    let d_sq = if distances_squared { d } else { d * d };
                    let p = (-beta * d_sq).exp();
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
// PHATE //
///////////

///////////
// Enums //
///////////

#[derive(Default)]
pub enum PhateGraphSymmetrisation {
    /// Additive symmetrisation - used in PHATE
    #[default]
    Additive,
    /// Multiplicative symmetrisation
    Multiplicative,
    /// Min-max symmetrisation
    Mnn,
    /// No symmetrisation
    None,
}

/// Parse a string into a PhateGraphSymmetrisation enum
///
/// ### Params
///
/// * `s` - String to parse
///
/// ### Returns
///
/// `Some(PhateGraphSymmetrisation)` pending on parsing.
pub fn parse_phate_symmetrisation(s: &str) -> Option<PhateGraphSymmetrisation> {
    match s.to_lowercase().as_str() {
        "additive" | "add" => Some(PhateGraphSymmetrisation::Additive),
        "multiplicative" | "mult" | "multiply" => Some(PhateGraphSymmetrisation::Multiplicative),
        "mnn" => Some(PhateGraphSymmetrisation::Mnn),
        "none" => Some(PhateGraphSymmetrisation::None),
        _ => None,
    }
}

/////////////
// Helpers //
/////////////

/// Additive symmetrisation
///
/// K = (K + K^T) / 2
///
/// ### Params
///
/// * `graph` - Reference to the graph to symmetrise
fn symmetrise_additive<T>(graph: &mut SparseGraph<T>)
where
    T: Float + Sync + Send + AddAssign,
{
    let two = T::one() + T::one();

    // do this in parallel
    // maybe slower for smaller graphs, but for sure much faster for larger ones
    let edge_map: FxHashMap<(usize, usize), T> = graph
        .row_indices
        .par_iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
        .fold(
            FxHashMap::default, // Changed here
            |mut local_map, ((&i, &j), &v)| {
                *local_map.entry((i, j)).or_insert(T::zero()) += v;
                *local_map.entry((j, i)).or_insert(T::zero()) += v;
                local_map
            },
        )
        .reduce(
            FxHashMap::default, // And here
            |mut map1, map2| {
                for (key, val) in map2 {
                    *map1.entry(key).or_insert(T::zero()) += val;
                }
                map1
            },
        );

    // Rebuild graph with (K + K^T) / 2
    graph.row_indices.clear();
    graph.col_indices.clear();
    graph.values.clear();

    graph.row_indices.reserve(edge_map.len());
    graph.col_indices.reserve(edge_map.len());
    graph.values.reserve(edge_map.len());

    for ((i, j), v) in edge_map {
        graph.row_indices.push(i);
        graph.col_indices.push(j);
        graph.values.push(v / two);
    }
}

/// Multiplicative symmetrisation
///
/// K = K ⊙ K^T (element-wise product)
///
/// ### Params
///
/// * `graph` - Reference to the graph to symmetrise
fn symmetrise_multiplicative<T>(graph: &mut SparseGraph<T>)
where
    T: Float + Send + Sync,
{
    // forward map
    let forward_map: FxHashMap<(usize, usize), T> = graph
        .row_indices
        .par_iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
        .fold(FxHashMap::default, |mut map, ((&i, &j), &v)| {
            map.insert((i, j), v);
            map
        })
        .reduce(FxHashMap::default, |mut map1, map2| {
            map1.extend(map2);
            map1
        });

    // backward map
    let backward_map: FxHashMap<(usize, usize), T> = graph
        .row_indices
        .par_iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
        .fold(FxHashMap::default, |mut map, ((&i, &j), &v)| {
            map.insert((j, i), v);
            map
        })
        .reduce(FxHashMap::default, |mut map1, map2| {
            map1.extend(map2);
            map1
        });

    // compute element-wise product (parallel)
    let products: Vec<(usize, usize, T)> = forward_map
        .par_iter()
        .filter_map(|(&(i, j), &v_ij)| backward_map.get(&(i, j)).map(|&v_ji| (i, j, v_ij * v_ji)))
        .collect();

    // rebuild graph
    graph.row_indices.clear();
    graph.col_indices.clear();
    graph.values.clear();

    graph.row_indices.reserve(products.len());
    graph.col_indices.reserve(products.len());
    graph.values.reserve(products.len());

    for (i, j, v) in products {
        graph.row_indices.push(i);
        graph.col_indices.push(j);
        graph.values.push(v);
    }
}

/// MNN symmetrisation
///
/// K = θ * min(K, K^T) + (1-θ) * max(K, K^T)
///
/// ### Params
///
/// * `graph` - Reference to the graph to symmetrise
fn symmetrise_mnn<T>(graph: &mut SparseGraph<T>, theta: T)
where
    T: Float + Send + Sync,
{
    let edge_map: FxHashMap<(usize, usize), (Option<T>, Option<T>)> = graph
        .row_indices
        .par_iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
        .fold(FxHashMap::default, |mut map, ((&i, &j), &v)| {
            map.entry((i, j)).or_insert((None, None)).0 = Some(v);
            map.entry((j, i)).or_insert((None, None)).1 = Some(v);
            map
        })
        .reduce(FxHashMap::default, |mut map1, map2| {
            for (key, (v_ij, v_ji)) in map2 {
                let entry = map1.entry(key).or_insert((None, None));
                if v_ij.is_some() {
                    entry.0 = v_ij;
                }
                if v_ji.is_some() {
                    entry.1 = v_ji;
                }
            }
            map1
        });

    let one_minus_theta = T::one() - theta;

    // compute weighted min-max (parallel)
    let results: Vec<(usize, usize, T)> = edge_map
        .par_iter()
        .filter_map(|(&(i, j), &(v_ij, v_ji))| {
            let v_ij = v_ij.unwrap_or(T::zero());
            let v_ji = v_ji.unwrap_or(T::zero());
            let min_val = v_ij.min(v_ji);
            let max_val = v_ij.max(v_ji);
            let combined = theta * min_val + one_minus_theta * max_val;

            if combined > T::epsilon() {
                Some((i, j, combined))
            } else {
                None
            }
        })
        .collect();

    // Rebuild graph
    graph.row_indices.clear();
    graph.col_indices.clear();
    graph.values.clear();

    graph.row_indices.reserve(results.len());
    graph.col_indices.reserve(results.len());
    graph.values.reserve(results.len());

    for (i, j, v) in results {
        graph.row_indices.push(i);
        graph.col_indices.push(j);
        graph.values.push(v);
    }
}

/// Binary connectivity
///
/// Used for decay = None case
///
/// ### Params
///
/// * `knn_indices`: The indices of the k-nearest neighbors for each vertex.
/// * `knn`: The number of nearest neighbors to consider.
/// * `symmetrise`: The symmetrisation method to use.
///
/// ### Returns
///
/// The SparseGraph representing the binary connectivity.
fn binary_knn_connectivity<T>(
    knn_indices: &[Vec<usize>],
    knn: usize,
    symmetrise: PhateGraphSymmetrisation,
) -> SparseGraph<T>
where
    T: Float + Send + Sync + AddAssign,
{
    let n = knn_indices.len();
    let k_actual = knn.min(knn_indices[0].len());

    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for (i, indices) in knn_indices.iter().enumerate() {
        for &j in indices.iter().take(k_actual) {
            if j != i {
                row_indices.push(i);
                col_indices.push(j);
                values.push(T::one());
            }
        }
    }

    let mut graph = SparseGraph {
        row_indices,
        col_indices,
        values,
        n_vertices: n,
    };

    if matches!(symmetrise, PhateGraphSymmetrisation::None) {
        symmetrise_additive(&mut graph);
    }

    graph
}

////////////////////////
// Alpha decay kernel //
////////////////////////

/// Compute alpha-decay affinities from k-nearest neighbours for PHATE
///
/// For each point i, computes affinities using an adaptive Gaussian kernel:
///
/// `K(i,j) = exp(-(d(i,j) / σ_i)^α)`
///
/// where σ_i is the distance to the kth nearest neighbour.
///
/// ### Params
///
/// * `knn_indices` - kNN indices (including self)
/// * `knn_dists` - kNN distances (including self)
/// * `knn` - Which neighbour to use for bandwidth (e.g., 5 means use 5th
///   nearest neighbour distance)
/// * `decay` - Decay exponent alpha (typical: 40). If None, returns binary
///   connectivity
/// * `bandwidth_scale` - Multiplicative factor for bandwidth (default: 1.0)
/// * `thresh` - Threshold below which affinities are set to 0 (default: 1e-4,
///   for sparsity)
/// * `distances_squared` - If true, distances are already squared (squared
///   Euclidean). If false, use as-is (cosine, etc.)
/// * `symmetrise` - Symmetrization method: "add" for (K+K^T)/2, "multiply" for
///   K*K^T, "none" for asymmetric.
///
/// ### Returns
///
/// A `SparseGraph` containing the (optionally symmetrized) affinities
#[allow(clippy::too_many_arguments)]
pub fn phate_alpha_decay_affinities<T>(
    knn_indices: &[Vec<usize>],
    knn_dists: &[Vec<T>],
    knn: usize,
    decay: Option<T>,
    bandwidth_scale: T,
    thresh: T,
    symmetrise: &str,
    distances_squared: bool,
) -> SparseGraph<T>
where
    T: Float + Send + Sync + FromPrimitive + ToPrimitive + AddAssign,
{
    let n = knn_indices.len();
    let machine_epsilon = T::epsilon();

    let symmetrise = parse_phate_symmetrisation(symmetrise).unwrap_or_default();

    // handle binary connectivity case (decay = None)
    if decay.is_none() {
        // return binary_knn_connectivity(knn_indices, knn, symmetrise);
    }

    let decay_val = decay.unwrap();

    // parallel computations of affinities
    let results: Vec<(Vec<usize>, Vec<T>)> = knn_indices
        .par_iter()
        .zip(knn_dists.par_iter())
        .enumerate()
        .map(|(i, (indices, dists))| {
            // bandwidth: dist to kth nearest neighbour
            // note: indices[0] is self, so indices[knn-1] is the kth neighbour
            // (excluding self)
            let bandwidth_dist = if knn > 0 && knn <= dists.len() {
                // this is needed as the ANN liibraries return squared distances
                // for speed
                if distances_squared {
                    dists[knn - 1].sqrt() // convert squared distance to distance
                } else {
                    dists[knn - 1] // already a distance
                }
            } else {
                // Fallback: use last neighbour
                if distances_squared {
                    dists[dists.len() - 1].sqrt()
                } else {
                    dists[dists.len() - 1]
                }
            };

            let bandwidth = bandwidth_dist * bandwidth_scale;

            // handle edge case of zero bandwidth
            let bandwidth = bandwidth.max(machine_epsilon);

            // pre-allocate
            let mut neighbor_indices = Vec::with_capacity(indices.len());
            let mut neighbor_values = Vec::with_capacity(indices.len());

            // compute affinities for each neighbour
            for (&j, &dist_val) in indices.iter().zip(dists.iter()) {
                // skip self-loops
                if j == i {
                    continue;
                }

                // handle zero distances
                if dist_val < machine_epsilon {
                    neighbor_indices.push(j);
                    neighbor_values.push(T::one());
                    continue;
                }

                // convert to actual distance if needed
                let d = if distances_squared {
                    dist_val.sqrt() // convert squared distance to distance
                } else {
                    dist_val // already a distance
                };

                // compute affinity: exp(-(d / σ)^α)
                let scaled = d / bandwidth;
                let powered = scaled.powf(decay_val);
                let affinity = (-powered).exp();

                // apply threshold for sparsity
                if affinity >= thresh {
                    neighbor_indices.push(j);
                    neighbor_values.push(affinity);
                }
            }

            (neighbor_indices, neighbor_values)
        })
        .collect();

    // Build asymmetric sparse graph
    let capacity: usize = results.iter().map(|(idx, _)| idx.len()).sum();
    let mut row_indices = Vec::with_capacity(capacity);
    let mut col_indices = Vec::with_capacity(capacity);
    let mut values = Vec::with_capacity(capacity);

    for (i, (indices, vals)) in results.into_iter().enumerate() {
        for (&j, v) in indices.iter().zip(vals) {
            row_indices.push(i);
            col_indices.push(j);
            values.push(v);
        }
    }

    let mut graph = SparseGraph {
        row_indices,
        col_indices,
        values,
        n_vertices: n,
    };

    match symmetrise {
        PhateGraphSymmetrisation::Additive => symmetrise_additive(&mut graph),
        PhateGraphSymmetrisation::Multiplicative => symmetrise_multiplicative(&mut graph),
        PhateGraphSymmetrisation::Mnn => symmetrise_mnn(&mut graph, T::one()),
        PhateGraphSymmetrisation::None => {}
    };

    graph
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_data_gen {
    use super::*;
    use approx::assert_relative_eq;

    ////////////////
    // Umap stuff //
    ////////////////

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

    ////////////////
    // tSNE stuff //
    ////////////////

    /// Helper: build adjacency map from sparse graph for easier testing
    fn graph_to_adj<T: Float + Copy>(graph: &SparseGraph<T>) -> Vec<Vec<(usize, T)>> {
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

    /// Helper: compute entropy of a probability distribution
    fn entropy(probs: &[f64]) -> f64 {
        probs
            .iter()
            .filter(|&&p| p > 1e-12)
            .map(|&p| -p * p.log2())
            .sum()
    }

    #[test]
    fn test_row_probabilities_sum_to_one() {
        // 5 points, each has 4 neighbours (excluding self)
        let knn_indices = vec![
            vec![1, 2, 3, 4],
            vec![0, 2, 3, 4],
            vec![0, 1, 3, 4],
            vec![0, 1, 2, 4],
            vec![0, 1, 2, 3],
        ];
        // Squared Euclidean distances
        let knn_dists = vec![
            vec![1.0, 4.0, 9.0, 16.0],
            vec![1.0, 1.0, 4.0, 9.0],
            vec![4.0, 1.0, 1.0, 4.0],
            vec![9.0, 4.0, 1.0, 1.0],
            vec![16.0, 9.0, 4.0, 1.0],
        ];

        let perplexity = 2.0;
        let graph = gaussian_knn_affinities(&knn_indices, &knn_dists, perplexity, 1e-5, 200, true);
        let adj = graph_to_adj(&graph);

        for (i, neighbours) in adj.iter().enumerate() {
            let sum: f64 = neighbours.iter().map(|(_, w)| *w).sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-4, max_relative = 1e-4);
            println!("Row {}: sum = {:.6}", i, sum);
        }
    }

    #[test]
    fn test_entropy_matches_target_perplexity() {
        // Create data where we can verify entropy
        let knn_indices = vec![
            vec![1, 2, 3, 4, 5, 6, 7],
            vec![0, 2, 3, 4, 5, 6, 7],
            vec![0, 1, 3, 4, 5, 6, 7],
            vec![0, 1, 2, 4, 5, 6, 7],
            vec![0, 1, 2, 3, 5, 6, 7],
            vec![0, 1, 2, 3, 4, 6, 7],
            vec![0, 1, 2, 3, 4, 5, 7],
            vec![0, 1, 2, 3, 4, 5, 6],
        ];
        // Squared distances with some variation
        let knn_dists: Vec<Vec<f64>> = (0..8)
            .map(|i| {
                (0..7)
                    .map(|j| ((j + 1) as f64) * (1.0 + 0.1 * (i as f64)))
                    .collect()
            })
            .collect();

        let perplexity = 3.0;
        let target_entropy = perplexity.log2();

        let graph = gaussian_knn_affinities(&knn_indices, &knn_dists, perplexity, 1e-5, 200, true);
        let adj = graph_to_adj(&graph);

        for (i, neighbours) in adj.iter().enumerate() {
            let probs: Vec<f64> = neighbours.iter().map(|(_, w)| *w).collect();
            let h = entropy(&probs);
            println!(
                "Row {}: entropy = {:.4}, target = {:.4}, diff = {:.6}",
                i,
                h,
                target_entropy,
                (h - target_entropy).abs()
            );
            assert_relative_eq!(h, target_entropy, epsilon = 1e-3, max_relative = 1e-3);
        }
    }

    #[test]
    fn test_squared_vs_unsquared_equivalence() {
        // Same underlying distances, but one is squared, one is not
        let knn_indices = vec![
            vec![1, 2, 3, 4],
            vec![0, 2, 3, 4],
            vec![0, 1, 3, 4],
            vec![0, 1, 2, 4],
            vec![0, 1, 2, 3],
        ];

        // Unsquared Euclidean distances
        let unsquared: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.0, 1.0, 2.0, 3.0],
            vec![2.0, 1.0, 1.0, 2.0],
            vec![3.0, 2.0, 1.0, 1.0],
            vec![4.0, 3.0, 2.0, 1.0],
        ];

        // Squared Euclidean distances
        let squared: Vec<Vec<f64>> = unsquared
            .iter()
            .map(|row| row.iter().map(|d| d * d).collect())
            .collect();

        let perplexity = 2.0;

        let graph_unsq =
            gaussian_knn_affinities(&knn_indices, &unsquared, perplexity, 1e-5, 200, false);
        let graph_sq = gaussian_knn_affinities(&knn_indices, &squared, perplexity, 1e-5, 200, true);

        let adj_unsq = graph_to_adj(&graph_unsq);
        let adj_sq = graph_to_adj(&graph_sq);

        // Results should be identical
        for i in 0..5 {
            assert_eq!(adj_unsq[i].len(), adj_sq[i].len());
            for (a, b) in adj_unsq[i].iter().zip(adj_sq[i].iter()) {
                assert_eq!(a.0, b.0); // same neighbour index
                assert_relative_eq!(a.1, b.1, epsilon = 1e-10);
            }
        }
        println!("Squared vs unsquared: results match!");
    }

    #[test]
    fn test_self_loops_excluded() {
        // kNN includes self (index i appears in knn_indices[i])
        let knn_indices = vec![
            vec![0, 1, 2, 3], // includes self
            vec![1, 0, 2, 3], // includes self
            vec![2, 0, 1, 3], // includes self
            vec![3, 0, 1, 2], // includes self
        ];
        let knn_dists = vec![
            vec![0.0, 1.0, 4.0, 9.0], // distance to self is 0
            vec![0.0, 1.0, 1.0, 4.0],
            vec![0.0, 4.0, 1.0, 1.0],
            vec![0.0, 9.0, 4.0, 1.0],
        ];

        let graph = gaussian_knn_affinities(&knn_indices, &knn_dists, 2.0, 1e-5, 200, true);

        // Check no self-loops in output
        for (&i, &j) in graph.row_indices.iter().zip(&graph.col_indices) {
            assert_ne!(i, j, "Self-loop found: {} -> {}", i, j);
        }
        println!("No self-loops in output graph.");
    }

    #[test]
    fn test_closer_neighbours_have_higher_probability() {
        let knn_indices = vec![vec![1, 2, 3, 4]];
        // Strictly increasing squared distances
        let knn_dists = vec![vec![1.0, 4.0, 9.0, 16.0]];

        let graph = gaussian_knn_affinities(&knn_indices, &knn_dists, 2.0, 1e-5, 200, true);
        let adj = graph_to_adj(&graph);

        let probs: Vec<(usize, f64)> = adj[0].clone();
        println!("Probabilities: {:?}", probs);

        // Closer neighbours should have higher probability
        // neighbour 1 (d²=1) > neighbour 2 (d²=4) > neighbour 3 (d²=9) > neighbour 4 (d²=16)
        let p1 = probs.iter().find(|(j, _)| *j == 1).unwrap().1;
        let p2 = probs.iter().find(|(j, _)| *j == 2).unwrap().1;
        let p3 = probs.iter().find(|(j, _)| *j == 3).unwrap().1;
        let p4 = probs.iter().find(|(j, _)| *j == 4).unwrap().1;

        assert!(p1 > p2, "p1={} should be > p2={}", p1, p2);
        assert!(p2 > p3, "p2={} should be > p3={}", p2, p3);
        assert!(p3 > p4, "p3={} should be > p4={}", p3, p4);
    }

    #[test]
    fn test_uniform_distances_give_uniform_probs() {
        // All neighbours at same distance → should get uniform distribution
        // Use perplexity = 4 so target entropy matches uniform entropy over 4 items
        let knn_indices = vec![vec![1, 2, 3, 4]];
        let knn_dists = vec![vec![4.0, 4.0, 4.0, 4.0]]; // all same squared distance

        let perplexity = 4.0; // Changed from 2.0
        let graph = gaussian_knn_affinities(&knn_indices, &knn_dists, perplexity, 1e-5, 200, true);
        let adj = graph_to_adj(&graph);

        let probs: Vec<f64> = adj[0].iter().map(|(_, p)| *p).collect();
        let expected = 0.25; // uniform over 4 neighbours

        for (i, &p) in probs.iter().enumerate() {
            assert_relative_eq!(p, expected, epsilon = 1e-4);
            println!("Neighbour {}: p = {:.6}", i, p);
        }
    }

    #[test]
    fn test_perplexity_affects_distribution_spread() {
        let knn_indices = vec![vec![1, 2, 3, 4, 5, 6, 7]];
        let knn_dists = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]; // unsquared

        // Low perplexity → more concentrated distribution
        let graph_low = gaussian_knn_affinities(&knn_indices, &knn_dists, 1.5, 1e-5, 200, false);
        let adj_low = graph_to_adj(&graph_low);
        let probs_low: Vec<f64> = adj_low[0].iter().map(|(_, p)| *p).collect();
        let entropy_low = entropy(&probs_low);

        // High perplexity → more spread distribution
        let graph_high = gaussian_knn_affinities(&knn_indices, &knn_dists, 4.0, 1e-5, 200, false);
        let adj_high = graph_to_adj(&graph_high);
        let probs_high: Vec<f64> = adj_high[0].iter().map(|(_, p)| *p).collect();
        let entropy_high = entropy(&probs_high);

        println!("Low perplexity (1.5): entropy = {:.4}", entropy_low);
        println!("High perplexity (4.0): entropy = {:.4}", entropy_high);

        assert!(
            entropy_high > entropy_low,
            "Higher perplexity should give higher entropy"
        );
    }

    ///////////
    // PHATE //
    ///////////

    #[test]
    fn test_phate_basic_affinity_computation() {
        // 4 points, each has 3 neighbours (including self at position 0)
        let knn_indices = vec![
            vec![0, 1, 2, 3],
            vec![1, 0, 2, 3],
            vec![2, 0, 1, 3],
            vec![3, 0, 1, 2],
        ];
        // Squared Euclidean distances (self at position 0 with distance 0)
        let knn_dists = vec![
            vec![0.0, 1.0, 4.0, 9.0],
            vec![0.0, 1.0, 4.0, 9.0],
            vec![0.0, 4.0, 1.0, 9.0],
            vec![0.0, 9.0, 4.0, 1.0],
        ];

        let graph = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            2,          // knn: use 2nd neighbor (indices[1]) for bandwidth
            Some(40.0), // decay
            1.0,        // bandwidth_scale
            1e-4,       // thresh
            "none",     // no symmetrization
            true,
        );

        assert_eq!(graph.n_vertices, 4);
        assert!(graph.row_indices.is_empty());
        assert_eq!(graph.row_indices.len(), graph.col_indices.len());
        assert_eq!(graph.row_indices.len(), graph.values.len());

        // All affinities should be between 0 and 1
        for &v in &graph.values {
            assert!(v >= 0.0 && v <= 1.0, "Affinity {} out of range", v);
        }
        println!("Basic test passed: {} edges created", graph.values.len());
    }
}
