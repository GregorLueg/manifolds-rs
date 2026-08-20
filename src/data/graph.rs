//! Module containing helper functions to generate the graphs used by UMAP,
//! tSNE and PHATE.

use rayon::prelude::*;

use thousands::*;

use crate::data::structures::*;
use crate::prelude::*;

//////////
// UMAP //
//////////

////////////
// Consts //
////////////

/// Edges per chunk when compacting a COO graph in parallel.
const FILTER_CHUNK: usize = 16_384;

/////////////////////
// UmapGraphParams //
/////////////////////

/// UMAP algorithm parameters
///
/// Controls the fuzzy simplicial set construction and graph symmetrisation.
#[derive(Clone, Debug)]
pub struct UmapGraphParams<T> {
    /// Convergence tolerance for smooth kNN distance binary search (typically
    /// 1e-5). Controls how precisely sigma values are computed.
    pub bandwidth: T,
    /// Number of nearest neighbours assumed to be at distance zero (typically
    /// 1.0). Allows for local manifold structure by treating the nearest
    /// neighbour(s) as having maximal membership strength.
    pub local_connectivity: T,
    /// Balance between fuzzy union and directed graph during symmetrisation
    /// (typically 1.0).
    pub mix_weight: T,
}

impl<T> Default for UmapGraphParams<T>
where
    T: ManifoldsFloat,
{
    /// Returns sensible defaults for UMAP
    ///
    /// ### Returns
    ///
    /// * `bandwidth = 1e-5` - Tight convergence for sigma computation
    /// * `local_connectivity = 1.0` - Treat nearest neighbour as connected
    /// * `mix_weight = 1.0` - Standard symmetric fuzzy union
    fn default() -> Self {
        Self {
            local_connectivity: T::from(1.0).unwrap(),
            bandwidth: T::from(1e-5).unwrap(),
            mix_weight: T::from(1.0).unwrap(),
        }
    }
}

/////////////
// Helpers //
/////////////

/// Widen a float to `f64`.
///
/// [`ManifoldsFloat`] only admits `f32` and `f64`, so the cast cannot fail.
/// Kept as a helper because the sigma binary search compares in `f64` regardless
/// of `T` to avoid the tolerance being swamped by `f32` rounding.
///
/// ### Params
///
/// * `value` - Value to widen
///
/// ### Returns
///
/// The same value as an `f64`.
#[inline(always)]
fn as_f64<T>(value: T) -> f64
where
    T: ManifoldsFloat,
{
    value
        .to_f64()
        .expect("ManifoldsFloat is f32 or f64, both of which widen to f64")
}

/// Interpolated distance to the `local_connectivity`-th nearest neighbour.
///
/// `local_connectivity` is a real number, so a fractional part interpolates
/// linearly between the two bracketing neighbours. Expects `dists` sorted
/// ascending, which is what `run_ann_search` returns.
///
/// ### Params
///
/// * `dists` - Distances to the nearest neighbours, sorted ascending
/// * `local_connectivity` - Number of neighbours assumed to sit at distance zero
///
/// ### Returns
///
/// The value of `rho` for this point, or zero when `local_connectivity` is not
/// positive or the row is empty.
#[inline]
fn local_connectivity_distance<T>(dists: &[T], local_connectivity: T) -> T
where
    T: ManifoldsFloat,
{
    if local_connectivity <= T::zero() || dists.is_empty() {
        return T::zero();
    }

    let offset = (local_connectivity - T::one()).max(T::zero());
    let whole = offset.floor();
    let fraction = offset - whole;
    let idx = whole.to_usize().unwrap_or(usize::MAX).min(dists.len() - 1);

    if fraction > T::zero() && idx + 1 < dists.len() {
        dists[idx] * (T::one() - fraction) + dists[idx + 1] * fraction
    } else {
        dists[idx]
    }
}

/// Half-open bounds of one [`FILTER_CHUNK`]-sized chunk, clipped to `len`.
///
/// ### Params
///
/// * `chunk` - Zero-based chunk index
/// * `len` - Total number of elements being chunked
///
/// ### Returns
///
/// `(lo, hi)` such that the chunk covers `lo..hi`.
#[inline(always)]
fn chunk_bounds(chunk: usize, len: usize) -> (usize, usize) {
    let lo = chunk * FILTER_CHUNK;
    (lo, (lo + FILTER_CHUNK).min(len))
}

/// Hand out one disjoint mutable slice per CSR row.
///
/// Rayon has no variable-width counterpart to `par_chunks_mut`, so the row views
/// are carved out sequentially with `split_at_mut` and consumed in parallel
/// afterwards. The carving loop is `n` pointer bumps and never touches the data.
///
/// ### Params
///
/// * `buf` - Flat buffer holding every row back to back
/// * `indptr` - Row offsets, `n + 1` entries, the last equal to `buf.len()`
///
/// ### Returns
///
/// One mutable slice per row, in row order.
fn row_slices_mut<'a, U>(buf: &'a mut [U], indptr: &[usize]) -> Vec<&'a mut [U]> {
    let mut slices = Vec::with_capacity(indptr.len().saturating_sub(1));
    let mut rest = buf;

    for window in indptr.windows(2) {
        let (head, tail) = rest.split_at_mut(window[1] - window[0]);
        slices.push(head);
        rest = tail;
    }

    slices
}

/// Convert a COO graph to CSR with every row sorted by column index.
///
/// Row grouping is a counting sort rather than a comparison sort, so the cost is
/// `O(nnz)` plus one `k log k` sort per row. Rows are sorted in parallel; the
/// scatter itself is sequential because the per-row cursors alias.
///
/// ### Params
///
/// * `graph` - Input graph in COO format, in any order
///
/// ### Returns
///
/// `(indptr, entries)` where `entries[indptr[i]..indptr[i + 1]]` holds row `i`
/// as `(column, weight)` pairs sorted ascending by column.
fn coo_to_sorted_csr<T>(graph: &CoordinateList<T>) -> (Vec<usize>, Vec<(usize, T)>)
where
    T: ManifoldsFloat,
{
    let n = graph.n_samples;
    let nnz = graph.values.len();

    let mut indptr = vec![0usize; n + 1];
    for &i in &graph.row_indices {
        indptr[i + 1] += 1;
    }
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }

    let mut entries = vec![(0usize, T::zero()); nnz];
    let mut cursor = indptr[..n].to_vec();

    for ((&i, &j), &w) in graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
    {
        entries[cursor[i]] = (j, w);
        cursor[i] += 1;
    }

    row_slices_mut(&mut entries, &indptr)
        .into_par_iter()
        .for_each(|row| row.sort_unstable_by_key(|&(col, _)| col));

    (indptr, entries)
}

/// Transpose a square CSR matrix via counting sort.
///
/// Because the source rows are visited in ascending order, each output row comes
/// out sorted ascending by index without an extra sort. `O(nnz)` with one
/// scattered write per entry.
///
/// ### Params
///
/// * `indptr` - Row offsets of the input, `n + 1` entries
/// * `entries` - Input entries as `(column, weight)` pairs
/// * `n` - Side length of the square matrix
///
/// ### Returns
///
/// `(indptr, entries)` of the transpose, rows sorted ascending by index. Row `i`
/// holds the incoming edges of vertex `i` as `(source, weight)` pairs.
fn transpose_csr<T>(
    indptr: &[usize],
    entries: &[(usize, T)],
    n: usize,
) -> (Vec<usize>, Vec<(usize, T)>)
where
    T: ManifoldsFloat,
{
    let mut t_indptr = vec![0usize; n + 1];
    for &(col, _) in entries {
        t_indptr[col + 1] += 1;
    }
    for i in 0..n {
        t_indptr[i + 1] += t_indptr[i];
    }

    let mut t_entries = vec![(0usize, T::zero()); entries.len()];
    let mut cursor = t_indptr[..n].to_vec();

    for row in 0..n {
        for &(col, w) in &entries[indptr[row]..indptr[row + 1]] {
            t_entries[cursor[col]] = (row, w);
            cursor[col] += 1;
        }
    }

    (t_indptr, t_entries)
}

/// Count the distinct indices in the union of two index-sorted rows.
///
/// Duplicate indices within a row collapse, matching the merge below, so the
/// result is an exact upper bound on the merged row length.
///
/// ### Params
///
/// * `a` - First row, sorted ascending by index
/// * `b` - Second row, sorted ascending by index
///
/// ### Returns
///
/// Number of distinct indices appearing in either row.
fn union_len<T>(a: &[(usize, T)], b: &[(usize, T)]) -> usize {
    let (mut ia, mut ib, mut count) = (0usize, 0usize, 0usize);

    while ia < a.len() || ib < b.len() {
        let col = match (a.get(ia), b.get(ib)) {
            (Some(&(ca, _)), Some(&(cb, _))) => ca.min(cb),
            (Some(&(ca, _)), None) => ca,
            (None, Some(&(cb, _))) => cb,
            (None, None) => break,
        };

        while ia < a.len() && a[ia].0 == col {
            ia += 1;
        }
        while ib < b.len() && b[ib].0 == col {
            ib += 1;
        }
        count += 1;
    }

    count
}

/// Pointwise merge of one vertex's outgoing and incoming edges.
///
/// Walks two index-sorted rows in lockstep and writes `combine(w_ij, w_ji)` for
/// every column appearing in either, keeping only results strictly above
/// `min_weight`. A column missing from one side contributes zero to that
/// argument. Duplicate columns within an input collapse to the last occurrence,
/// matching the map-insert semantics this replaced.
///
/// ### Params
///
/// * `out_row` - Outgoing edges as `(target, weight)`, sorted by target
/// * `in_row` - Incoming edges as `(source, weight)`, sorted by source
/// * `min_weight` - Results at or below this are dropped
/// * `combine` - Symmetrisation kernel applied to `(w_ij, w_ji)`
/// * `dst` - Destination, must hold at least `union_len(out_row, in_row)` entries
///
/// ### Returns
///
/// Number of entries written to `dst`, which is at most its length.
fn merge_row<T, F>(
    out_row: &[(usize, T)],
    in_row: &[(usize, T)],
    min_weight: T,
    combine: &F,
    dst: &mut [(usize, T)],
) -> usize
where
    T: ManifoldsFloat,
    F: Fn(T, T) -> T + Sync,
{
    let (mut ia, mut ib, mut written) = (0usize, 0usize, 0usize);

    while ia < out_row.len() || ib < in_row.len() {
        let col = match (out_row.get(ia), in_row.get(ib)) {
            (Some(&(ca, _)), Some(&(cb, _))) => ca.min(cb),
            (Some(&(ca, _)), None) => ca,
            (None, Some(&(cb, _))) => cb,
            (None, None) => break,
        };

        let mut w_ij = T::zero();
        while ia < out_row.len() && out_row[ia].0 == col {
            w_ij = out_row[ia].1;
            ia += 1;
        }

        let mut w_ji = T::zero();
        while ib < in_row.len() && in_row[ib].0 == col {
            w_ji = in_row[ib].1;
            ib += 1;
        }

        let w_sym = combine(w_ij, w_ji);

        if w_sym > min_weight {
            dst[written] = (col, w_sym);
            written += 1;
        }
    }

    written
}

/// Symmetrise a COO graph with an arbitrary pointwise kernel.
///
/// Shared by UMAP, tSNE and all three PHATE variants; only `combine` and
/// `min_weight` differ between them. The graph is grouped into CSR by counting
/// sort, transposed by counting sort, and the two are merged row by row with a
/// two-pointer walk. Everything past the two counting sorts is parallel, and the
/// output is allocated once from exact per-row survivor counts.
///
/// Both the input rows and the output are sorted ascending by column, so the
/// result is reproducible run to run and downstream CSR conversion can skip its
/// sort.
///
/// ### Params
///
/// * `graph` - Input graph in COO format, in any order
/// * `min_weight` - Combined weights at or below this are dropped
/// * `combine` - Applied to `(w_ij, w_ji)`; a missing direction passes zero
///
/// ### Returns
///
/// Symmetrised graph in COO format, grouped by row and sorted by column within
/// each row.
fn symmetrise_csr<T, F>(graph: &CoordinateList<T>, min_weight: T, combine: F) -> CoordinateList<T>
where
    T: ManifoldsFloat,
    F: Fn(T, T) -> T + Sync,
{
    let n = graph.n_samples;

    if n == 0 || graph.values.is_empty() {
        return CoordinateList {
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            n_samples: n,
        };
    }

    let (out_indptr, out_entries) = coo_to_sorted_csr(graph);
    let (in_indptr, in_entries) = transpose_csr(&out_indptr, &out_entries, n);

    let mut merge_indptr = vec![0usize; n + 1];
    merge_indptr[1..]
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, slot)| {
            *slot = union_len(
                &out_entries[out_indptr[i]..out_indptr[i + 1]],
                &in_entries[in_indptr[i]..in_indptr[i + 1]],
            );
        });
    for i in 0..n {
        merge_indptr[i + 1] += merge_indptr[i];
    }

    let mut merged = vec![(0usize, T::zero()); merge_indptr[n]];
    let mut lengths = vec![0usize; n];

    row_slices_mut(&mut merged, &merge_indptr)
        .into_par_iter()
        .zip(lengths.par_iter_mut())
        .enumerate()
        .for_each(|(i, (dst, len))| {
            *len = merge_row(
                &out_entries[out_indptr[i]..out_indptr[i + 1]],
                &in_entries[in_indptr[i]..in_indptr[i + 1]],
                min_weight,
                &combine,
                dst,
            );
        });

    // Exact offsets now that the surviving edge count per row is known
    let mut coo_indptr = vec![0usize; n + 1];
    for i in 0..n {
        coo_indptr[i + 1] = coo_indptr[i] + lengths[i];
    }

    let total = coo_indptr[n];
    let mut row_indices = vec![0usize; total];
    let mut col_indices = vec![0usize; total];
    let mut values = vec![T::zero(); total];

    row_slices_mut(&mut row_indices, &coo_indptr)
        .into_par_iter()
        .zip(row_slices_mut(&mut col_indices, &coo_indptr))
        .zip(row_slices_mut(&mut values, &coo_indptr))
        .enumerate()
        .for_each(|(i, ((rows, cols), vals))| {
            let src = &merged[merge_indptr[i]..merge_indptr[i] + lengths[i]];
            for (idx, &(col, w)) in src.iter().enumerate() {
                rows[idx] = i;
                cols[idx] = col;
                vals[idx] = w;
            }
        });

    CoordinateList {
        row_indices,
        col_indices,
        values,
        n_samples: n,
    }
}

///////////////
// Front end //
///////////////

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
    T: ManifoldsFloat,
{
    let n = dist.len();
    let target = (k as f64).ln();
    let tol = as_f64(bandwidth);
    let two = T::one() + T::one();
    let widest_row = dist.iter().map(|row| row.len()).max().unwrap_or(0);

    let mut sigmas = vec![T::zero(); n];
    let mut rhos = vec![T::zero(); n];

    sigmas
        .par_iter_mut()
        .zip(rhos.par_iter_mut())
        .zip(dist.par_iter())
        .for_each_init(
            || Vec::<T>::with_capacity(widest_row),
            |adjusted, ((sigma, rho_out), dists)| {
                let rho = local_connectivity_distance(dists, local_connectivity);

                // `max(d - rho, 0)` does not depend on the bandwidth, so it is
                // computed once here rather than inside all `n_iter` passes.
                adjusted.clear();
                adjusted.extend(dists.iter().map(|&d| (d - rho).max(T::zero())));

                let mut lo = T::zero();
                let mut hi = T::max_value();
                let mut mid = T::one();

                for _ in 0..n_iter {
                    // Reciprocal hoisted out of the row so the inner loop
                    // multiplies rather than divides per element.
                    let inv_mid = T::one() / mid;
                    let val = as_f64(adjusted.iter().map(|&a| (-a * inv_mid).exp()).sum::<T>());

                    if (val - target).abs() < tol {
                        break;
                    }

                    if val > target {
                        hi = mid;
                        mid = (lo + hi) / two;
                    } else {
                        lo = mid;
                        if hi == T::max_value() {
                            mid *= two;
                        } else {
                            mid = (lo + hi) / two;
                        }
                    }
                }

                *sigma = mid;
                *rho_out = rho;
            },
        );

    (sigmas, rhos)
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
) -> CoordinateList<T>
where
    T: ManifoldsFloat,
{
    let n = knn_indices.len();

    // Exact per-row edge counts, so the output is allocated once and every row
    // can be written into its own disjoint slice in parallel.
    let mut offsets = vec![0usize; n + 1];
    offsets[1..]
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, slot)| {
            *slot = knn_indices[i]
                .iter()
                .zip(knn_dists[i].iter())
                .filter(|(&j, _)| j != i)
                .count();
        });

    for i in 0..n {
        offsets[i + 1] += offsets[i];
    }

    let total = offsets[n];
    let mut row_indices = vec![0usize; total];
    let mut col_indices = vec![0usize; total];
    let mut values = vec![T::zero(); total];

    row_slices_mut(&mut row_indices, &offsets)
        .into_par_iter()
        .zip(row_slices_mut(&mut col_indices, &offsets))
        .zip(row_slices_mut(&mut values, &offsets))
        .enumerate()
        .for_each(|(i, ((rows, cols), vals))| {
            let sigma = sigmas[i];
            let rho = rhos[i];

            // Reciprocal hoisted out of the row so the inner loop multiplies
            // rather than divides per neighbour.
            let smooth = sigma > T::zero();
            let inv_sigma = if smooth { T::one() / sigma } else { T::zero() };

            let mut written = 0usize;
            for (&j, &dist) in knn_indices[i].iter().zip(knn_dists[i].iter()) {
                if i == j {
                    continue;
                }

                let adjusted = (dist - rho).max(T::zero());
                let weight = if smooth {
                    (-adjusted * inv_sigma).exp()
                } else if adjusted > T::zero() {
                    // Degenerate bandwidth: full membership only at rho itself.
                    T::zero()
                } else {
                    T::one()
                };

                rows[written] = i;
                cols[written] = j;
                vals[written] = weight;
                written += 1;
            }
        });

    CoordinateList {
        row_indices,
        col_indices,
        values,
        n_samples: n,
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
pub fn symmetrise_graph<T>(graph: CoordinateList<T>, mix_weight: T) -> CoordinateList<T>
where
    T: ManifoldsFloat,
{
    let directed_weight = T::one() - mix_weight;

    symmetrise_csr(&graph, T::zero(), |w_ij, w_ji| {
        let union = w_ij + w_ji - w_ij * w_ji;
        mix_weight * union + directed_weight * w_ij
    })
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
pub fn coo_to_adjacency_list<T>(graph: &CoordinateList<T>) -> Vec<Vec<(usize, T)>>
where
    T: ManifoldsFloat,
{
    let n = graph.n_samples;

    let mut degrees = vec![0usize; n];
    for &i in &graph.row_indices {
        degrees[i] += 1;
    }

    if graph
        .row_indices
        .par_windows(2)
        .all(|pair| pair[0] <= pair[1])
    {
        let mut indptr = vec![0usize; n + 1];
        for i in 0..n {
            indptr[i + 1] = indptr[i] + degrees[i];
        }

        return (0..n)
            .into_par_iter()
            .map(|i| {
                let (lo, hi) = (indptr[i], indptr[i + 1]);
                graph.col_indices[lo..hi]
                    .iter()
                    .copied()
                    .zip(graph.values[lo..hi].iter().copied())
                    .collect()
            })
            .collect();
    }

    let mut adj: Vec<Vec<(usize, T)>> = degrees
        .par_iter()
        .map(|&degree| Vec::with_capacity(degree))
        .collect();

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
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// Filtered graph with weak edges removed
pub fn filter_weak_edges<T>(
    graph: CoordinateList<T>,
    n_epochs: usize,
    verbose: usize,
) -> CoordinateList<T>
where
    T: ManifoldsFloat,
{
    let verbosity = parse_verbosity_level(verbose);

    let max_weight =
        graph
            .values
            .par_iter()
            .copied()
            .reduce(T::zero, |acc, w| if w > acc { w } else { acc });

    let original_edge_no = graph.col_indices.len();

    let threshold = max_weight / T::from(n_epochs).unwrap();

    let n_chunks = original_edge_no.div_ceil(FILTER_CHUNK);
    let mut offsets = vec![0usize; n_chunks + 1];

    offsets[1..]
        .par_iter_mut()
        .enumerate()
        .for_each(|(chunk, slot)| {
            let (lo, hi) = chunk_bounds(chunk, original_edge_no);
            *slot = graph.values[lo..hi]
                .iter()
                .filter(|&&w| w >= threshold)
                .count();
        });

    for chunk in 0..n_chunks {
        offsets[chunk + 1] += offsets[chunk];
    }

    let filtered_edge_no = offsets[n_chunks];
    let mut filtered_rows = vec![0usize; filtered_edge_no];
    let mut filtered_cols = vec![0usize; filtered_edge_no];
    let mut filtered_vals = vec![T::zero(); filtered_edge_no];

    row_slices_mut(&mut filtered_rows, &offsets)
        .into_par_iter()
        .zip(row_slices_mut(&mut filtered_cols, &offsets))
        .zip(row_slices_mut(&mut filtered_vals, &offsets))
        .enumerate()
        .for_each(|(chunk, ((rows, cols), vals))| {
            let (lo, hi) = chunk_bounds(chunk, original_edge_no);
            let mut written = 0usize;

            for idx in lo..hi {
                let w = graph.values[idx];
                if w >= threshold {
                    rows[written] = graph.row_indices[idx];
                    cols[written] = graph.col_indices[idx];
                    vals[written] = w;
                    written += 1;
                }
            }
        });

    if verbosity.detailed_verbosity() {
        println!(
            " Filtered out {} weak edges.",
            (original_edge_no - filtered_edge_no).separate_with_underscores(),
        );
    }

    CoordinateList {
        row_indices: filtered_rows,
        col_indices: filtered_cols,
        values: filtered_vals,
        n_samples: graph.n_samples,
    }
}

//////////
// tSNE //
//////////

////////////
// Consts //
////////////

/// `log2(e)`, converting the natural-log form of the Shannon entropy into bits.
///
/// The perplexity target is stated in bits (`log2(perplexity)`), but the fused
/// entropy falls out of the Gaussian kernel in nats, so the conversion happens
/// once per binary-search iteration rather than once per neighbour.
const LOG2_E: f64 = std::f64::consts::LOG2_E;

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
/// A `CoordinateList` containing the asymmetric conditional probabilities p_{j|i}
///
/// ### Errors
///
/// Returns an error if `perplexity >= k` for any point, since the maximum
/// achievable entropy with k neighbours is log2(k) and the binary search
/// cannot converge to a target entropy of log2(perplexity).
///
/// ### Notes
///
/// Used for tSNE. Quality degrades when perplexity approaches k, since the
/// binary search still converges but the bandwidth becomes wide enough that
/// tail neighbours dominate the affinities. Callers should size k at roughly
/// `3 * perplexity` or larger.
pub fn gaussian_knn_affinities<T>(
    knn_indices: &[Vec<usize>],
    knn_dists: &[Vec<T>],
    perplexity: T,
    tol: T,
    max_iter: usize,
    distances_squared: bool,
) -> Result<CoordinateList<T>, ManifoldsError>
where
    T: ManifoldsFloat,
{
    let n = knn_indices.len();

    // perplexity vs kNN size validation
    let min_k = knn_indices.iter().map(|idx| idx.len()).min().unwrap_or(0);

    let perp_f64 = perplexity.to_f64().unwrap_or(f64::NAN);

    // hard error: perplexity must be strictly less than k
    if perp_f64 >= min_k as f64 {
        return Err(ManifoldsError::PerplexityTooLarge {
            perplexity: perp_f64,
            k: min_k,
        });
    }

    // The search compares in `f64` regardless of `T`, matching `smooth_knn_dist`,
    // so an `f32` tolerance is not swamped by rounding in the entropy.
    let target_entropy = perp_f64.log2();
    let tol_f64 = as_f64(tol);
    let machine_epsilon = T::epsilon();
    let log2_e = T::from_f64(LOG2_E).unwrap();
    let two = T::one() + T::one();
    let widest_row = knn_dists.iter().map(|row| row.len()).max().unwrap_or(0);

    // Upper-bound layout: one slot per input neighbour. Survivors are written to
    // the front of each row, so the exact offsets are known after the search.
    let mut in_offsets = vec![0usize; n + 1];
    for i in 0..n {
        in_offsets[i + 1] = in_offsets[i] + knn_dists[i].len();
    }

    let mut col_flat = vec![0usize; in_offsets[n]];
    let mut val_flat = vec![T::zero(); in_offsets[n]];
    let mut kept = vec![0usize; n];

    row_slices_mut(&mut col_flat, &in_offsets)
        .into_par_iter()
        .zip(row_slices_mut(&mut val_flat, &in_offsets))
        .zip(kept.par_iter_mut())
        .enumerate()
        .for_each_init(
            || {
                (
                    Vec::<T>::with_capacity(widest_row),
                    Vec::<usize>::with_capacity(widest_row),
                    Vec::<T>::with_capacity(widest_row),
                )
            },
            |(d_sq, origin, probs), (i, ((cols, vals), keep))| {
                // Squaring and the zero-distance skip do not depend on beta, so
                // the row is compacted once rather than inside all `max_iter`
                // passes. The search loop then runs over a dense slice with no
                // branch in it. `origin` maps each survivor back to its slot in
                // the kNN row.
                d_sq.clear();
                origin.clear();
                for (m, &d) in knn_dists[i].iter().enumerate() {
                    if d < machine_epsilon {
                        continue;
                    }
                    d_sq.push(if distances_squared { d } else { d * d });
                    origin.push(m);
                }

                probs.clear();
                probs.resize(d_sq.len(), T::zero());

                // binary search for precision (beta = 1 / (2*sigma^2))
                let mut beta = T::one();
                let mut min_beta = T::neg_infinity();
                let mut max_beta = T::infinity();
                let mut sum_p = machine_epsilon;

                for _ in 0..max_iter {
                    // Unnormalised kernel and its first moment in one scan; the
                    // two accumulators are independent, so they pipeline.
                    sum_p = T::zero();
                    let mut sum_dp = T::zero();
                    for (p, &d) in probs.iter_mut().zip(d_sq.iter()) {
                        let e = (-beta * d).exp();
                        *p = e;
                        sum_p += e;
                        sum_dp += d * e;
                    }

                    // check for numerical stability
                    if sum_p.abs() < machine_epsilon {
                        sum_p = machine_epsilon;
                    }

                    // H = log2(S) - (1/S) * sum(p * log2 p) with
                    // log2 p = -beta * d² * log2(e), so the whole entropy costs
                    // one log rather than one per neighbour, and the row never
                    // has to be normalised inside the search.
                    let entropy = log2_e * (sum_p.ln() + beta * sum_dp / sum_p);
                    let entropy_diff = as_f64(entropy) - target_entropy;

                    if entropy_diff.abs() < tol_f64 {
                        break;
                    }

                    // adjust beta
                    if entropy_diff > 0.0 {
                        // entropy too high → distribution too flat → increase beta (narrow curve)
                        min_beta = beta;
                        if max_beta.is_infinite() {
                            beta *= two;
                        } else {
                            beta = (beta + max_beta) / two;
                        }
                    } else {
                        // entropy too low → distribution too peaked → decrease beta (widen curve)
                        max_beta = beta;
                        if min_beta.is_infinite() {
                            beta /= two;
                        } else {
                            beta = (beta + min_beta) / two;
                        }
                    }
                }

                // Normalise once, against the sum of the last evaluated beta.
                let inv_sum_p = T::one() / sum_p;
                let mut written = 0usize;

                for (&p, &m) in probs.iter().zip(origin.iter()) {
                    let prob = p * inv_sum_p;
                    let j = knn_indices[i][m];

                    if prob > machine_epsilon && j != i {
                        cols[written] = j;
                        vals[written] = prob;
                        written += 1;
                    }
                }

                *keep = written;
            },
        );

    // Exact offsets now that the surviving edge count per row is known
    let mut out_offsets = vec![0usize; n + 1];
    for i in 0..n {
        out_offsets[i + 1] = out_offsets[i] + kept[i];
    }

    let total = out_offsets[n];
    let mut row_indices = vec![0usize; total];
    let mut col_indices = vec![0usize; total];
    let mut values = vec![T::zero(); total];

    row_slices_mut(&mut row_indices, &out_offsets)
        .into_par_iter()
        .zip(row_slices_mut(&mut col_indices, &out_offsets))
        .zip(row_slices_mut(&mut values, &out_offsets))
        .enumerate()
        .for_each(|(i, ((rows, cols), vals))| {
            let lo = in_offsets[i];
            rows.fill(i);
            cols.copy_from_slice(&col_flat[lo..lo + kept[i]]);
            vals.copy_from_slice(&val_flat[lo..lo + kept[i]]);
        });

    Ok(CoordinateList {
        row_indices,
        col_indices,
        values,
        n_samples: n,
    })
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
/// Symmetric `CoordinateList` where:
/// - Each edge (i,j) has weight P_ij = (P(j|i) + P(i|j)) / 2N
/// - P_ij = P_ji (symmetric)
/// - All weights sum to 1.0
/// - Edges are grouped by row and sorted by column within each row
///
/// ### Notes
///
/// Edges whose symmetrised weight lands at or below `T::epsilon()` are dropped.
/// The weights scale as `1 / 2N`, so in `f32` this threshold bites hard once `N`
/// runs into the tens of thousands.
///
/// ### Algorithm
///
/// Delegates to the shared CSR symmetrisation with the arithmetic-mean kernel.
pub fn symmetrise_affinities_tsne<T>(graph: CoordinateList<T>) -> CoordinateList<T>
where
    T: ManifoldsFloat,
{
    let n_float = T::from_usize(graph.n_samples).unwrap();
    let two = T::from_f64(2.0).unwrap();

    // Hoisted so the merge kernel multiplies rather than divides per edge.
    let inv_normalisation = T::one() / (two * n_float);

    symmetrise_csr(&graph, T::epsilon(), |w_ij, w_ji| {
        (w_ij + w_ji) * inv_normalisation
    })
}

///////////
// PHATE //
///////////

///////////
// Enums //
///////////

/// Which symmetrisation to use for the PHATE graph.
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
///
/// ### Notes
///
/// Duplicate `(i, j)` entries in the input collapse to the last occurrence
/// rather than accumulating. kNN rows carry unique columns, so this only differs
/// from a summing merge on malformed input.
fn symmetrise_additive<T>(graph: &mut CoordinateList<T>)
where
    T: ManifoldsFloat,
{
    let half = T::one() / (T::one() + T::one());

    *graph = symmetrise_csr(graph, T::zero(), |w_ij, w_ji| (w_ij + w_ji) * half);
}

/// Multiplicative symmetrisation
///
/// K = K ⊙ K^T (element-wise product)
///
/// ### Params
///
/// * `graph` - Reference to the graph to symmetrise
///
/// ### Notes
///
/// A one-sided edge multiplies against zero and is dropped, so the result is the
/// intersection of `K` and `K^T`.
fn symmetrise_multiplicative<T>(graph: &mut CoordinateList<T>)
where
    T: ManifoldsFloat,
{
    *graph = symmetrise_csr(graph, T::zero(), |w_ij, w_ji| w_ij * w_ji);
}

/// MNN symmetrisation
///
/// K = θ * min(K, K^T) + (1-θ) * max(K, K^T)
///
/// ### Params
///
/// * `graph` - Reference to the graph to symmetrise
fn symmetrise_mnn<T>(graph: &mut CoordinateList<T>, theta: T)
where
    T: ManifoldsFloat,
{
    let one_minus_theta = T::one() - theta;

    *graph = symmetrise_csr(graph, T::epsilon(), |w_ij, w_ji| {
        theta * w_ij.min(w_ji) + one_minus_theta * w_ij.max(w_ji)
    });
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
/// The CoordinateList representing the binary connectivity.
fn binary_knn_connectivity<T>(
    knn_indices: &[Vec<usize>],
    knn: usize,
    symmetrise: PhateGraphSymmetrisation,
) -> CoordinateList<T>
where
    T: ManifoldsFloat,
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

    let mut graph = CoordinateList {
        row_indices,
        col_indices,
        values,
        n_samples: n,
    };

    if !matches!(symmetrise, PhateGraphSymmetrisation::None) {
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
/// * `symmetrise` - symmetrisation method: "add" for (K+K^T)/2, "multiply" for
///   K*K^T, "none" for asymmetric.
///
/// ### Returns
///
/// A `CoordinateList` containing the (optionally symmetrized) affinities
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
) -> CoordinateList<T>
where
    T: ManifoldsFloat,
{
    let n = knn_indices.len();
    let machine_epsilon = T::epsilon();

    let symmetrise = parse_phate_symmetrisation(symmetrise).unwrap_or_default();

    // handle binary connectivity case (decay = None)
    if decay.is_none() {
        return binary_knn_connectivity(knn_indices, knn, symmetrise);
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
                // fallback: use last neighbour
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

    let mut graph = CoordinateList {
        row_indices,
        col_indices,
        values,
        n_samples: n,
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
    use num_traits::Float;

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

        assert_eq!(graph.n_samples, 3);
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
        let graph = CoordinateList {
            row_indices: vec![0, 1],
            col_indices: vec![1, 0],
            values: vec![0.8, 0.6],
            n_samples: 2,
        };

        let sym_graph = symmetrise_graph(graph, 0.5);

        assert_eq!(sym_graph.n_samples, 2);

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
        let graph = CoordinateList {
            row_indices: vec![0, 1],
            col_indices: vec![1, 0],
            values: vec![0.8, 0.6],
            n_samples: 2,
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
        let graph = CoordinateList {
            row_indices: vec![0, 0, 1, 2],
            col_indices: vec![1, 2, 2, 0],
            values: vec![0.5, 0.3, 0.8, 0.9],
            n_samples: 3,
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
        let graph: CoordinateList<f64> = CoordinateList {
            row_indices: vec![],
            col_indices: vec![],
            values: vec![],
            n_samples: 3,
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
    fn graph_to_adj<T: Float + Copy>(graph: &CoordinateList<T>) -> Vec<Vec<(usize, T)>> {
        let mut adj = vec![Vec::new(); graph.n_samples];
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
        let graph =
            gaussian_knn_affinities(&knn_indices, &knn_dists, perplexity, 1e-5, 200, true).unwrap();
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

        let graph =
            gaussian_knn_affinities(&knn_indices, &knn_dists, perplexity, 1e-5, 200, true).unwrap();
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
            gaussian_knn_affinities(&knn_indices, &unsquared, perplexity, 1e-5, 200, false)
                .unwrap();
        let graph_sq =
            gaussian_knn_affinities(&knn_indices, &squared, perplexity, 1e-5, 200, true).unwrap();

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

        let graph =
            gaussian_knn_affinities(&knn_indices, &knn_dists, 2.0, 1e-5, 200, true).unwrap();

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

        let graph =
            gaussian_knn_affinities(&knn_indices, &knn_dists, 2.0, 1e-5, 200, true).unwrap();
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

        let perplexity = 3.99999999; // to not throw an error
        let graph =
            gaussian_knn_affinities(&knn_indices, &knn_dists, perplexity, 1e-5, 200, true).unwrap();
        let adj = graph_to_adj(&graph);

        let probs: Vec<f64> = adj[0].iter().map(|(_, p)| *p).collect();
        let expected = 0.25; // uniform over 4 neighbours

        for (i, &p) in probs.iter().enumerate() {
            assert_relative_eq!(p, expected, epsilon = 1e-1); // generous threshold
            println!("Neighbour {}: p = {:.6}", i, p);
        }
    }

    #[test]
    fn test_perplexity_affects_distribution_spread() {
        let knn_indices = vec![vec![1, 2, 3, 4, 5, 6, 7]];
        let knn_dists = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]; // unsquared

        // Low perplexity → more concentrated distribution
        let graph_low =
            gaussian_knn_affinities(&knn_indices, &knn_dists, 1.5, 1e-5, 200, false).unwrap();
        let adj_low = graph_to_adj(&graph_low);
        let probs_low: Vec<f64> = adj_low[0].iter().map(|(_, p)| *p).collect();
        let entropy_low = entropy(&probs_low);

        // High perplexity → more spread distribution
        let graph_high =
            gaussian_knn_affinities(&knn_indices, &knn_dists, 4.0, 1e-5, 200, false).unwrap();
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
            "none",     // no symmetrisation
            true,
        );

        assert_eq!(graph.n_samples, 4);
        assert!(!graph.row_indices.is_empty());
        assert_eq!(graph.row_indices.len(), graph.col_indices.len());
        assert_eq!(graph.row_indices.len(), graph.values.len());

        // All affinities should be between 0 and 1
        for &v in &graph.values {
            assert!((0.0..=1.0).contains(&v), "Affinity {} out of range", v);
        }
        println!("Basic test passed: {} edges created", graph.values.len());
    }

    #[test]
    fn test_phate_self_loops_excluded() {
        // Self is at position 0 in the knn arrays
        let knn_indices = vec![vec![0, 1, 2], vec![1, 0, 2], vec![2, 0, 1]];
        let knn_dists = vec![
            vec![0.0, 1.0, 4.0],
            vec![0.0, 1.0, 4.0],
            vec![0.0, 4.0, 1.0],
        ];

        let graph = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            2,
            Some(40.0),
            1.0,
            1e-4,
            "none",
            true,
        );

        // Check no self-loops
        for (&i, &j) in graph.row_indices.iter().zip(&graph.col_indices) {
            assert_ne!(i, j, "Self-loop found: {} -> {}", i, j);
        }
        println!("No self-loops in PHATE graph");
    }

    #[test]
    fn test_phate_closer_neighbours_higher_affinity() {
        // Single point with 4 neighbors at increasing distances
        let knn_indices = vec![vec![0, 1, 2, 3, 4]];
        // Squared distances: 0 (self), 1, 4, 9, 16
        let knn_dists = vec![vec![0.0, 1.0, 4.0, 9.0, 16.0]];

        let graph = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            2,
            Some(2.0),
            1.0,
            1e-10,
            "none",
            true,
        );

        let adj = graph_to_adj(&graph);
        let affinities: Vec<(usize, f64)> = adj[0].clone();

        // Get affinities for each neighbor
        let a1 = affinities.iter().find(|(j, _)| *j == 1).unwrap().1;
        let a2 = affinities.iter().find(|(j, _)| *j == 2).unwrap().1;
        let a3 = affinities.iter().find(|(j, _)| *j == 3).unwrap().1;
        let a4 = affinities.iter().find(|(j, _)| *j == 4).unwrap().1;

        println!(
            "Affinities: a1={:.6}, a2={:.6}, a3={:.6}, a4={:.6}",
            a1, a2, a3, a4
        );

        // closer neighbors should have higher affinities
        assert!(a1 > a2, "a1={} should be > a2={}", a1, a2);
        assert!(a2 > a3, "a2={} should be > a3={}", a2, a3);
        assert!(a3 > a4, "a3={} should be > a4={}", a3, a4);
    }

    #[test]
    fn test_phate_bandwidth_from_kth_neighbor() {
        // Test that bandwidth is correctly computed from the kth neighbor
        let knn_indices = vec![vec![0, 1, 2, 3]];
        let knn_dists = vec![vec![0.0, 1.0, 4.0, 16.0]]; // squared distances

        // Use k=2: bandwidth should be sqrt(1.0) = 1.0
        let graph_k2 = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            2,
            Some(2.0),
            1.0,
            1e-10,
            "none",
            true,
        );

        // Use k=3: bandwidth should be sqrt(4.0) = 2.0
        let graph_k3 = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            3,
            Some(2.0),
            1.0,
            1e-10,
            "none",
            true,
        );

        let adj_k2 = graph_to_adj(&graph_k2);
        let adj_k3 = graph_to_adj(&graph_k3);

        // Get affinities for near and far neighbors
        let a1_k2 = adj_k2[0].iter().find(|(j, _)| *j == 1).unwrap().1; // near
        let a2_k2 = adj_k2[0].iter().find(|(j, _)| *j == 2).unwrap().1; // far

        let a1_k3 = adj_k3[0].iter().find(|(j, _)| *j == 1).unwrap().1; // near
        let a2_k3 = adj_k3[0].iter().find(|(j, _)| *j == 2).unwrap().1; // far

        println!(
            "k=2: a1={:.6}, a2={:.6}, ratio={:.6}",
            a1_k2,
            a2_k2,
            a1_k2 / a2_k2
        );
        println!(
            "k=3: a1={:.6}, a2={:.6}, ratio={:.6}",
            a1_k3,
            a2_k3,
            a1_k3 / a2_k3
        );

        // Smaller bandwidth (k=2) should give LARGER ratio between near and far neighbors
        // (more peaked distribution)
        let ratio_k2 = a1_k2 / a2_k2;
        let ratio_k3 = a1_k3 / a2_k3;

        assert!(
        ratio_k2 > ratio_k3,
        "Smaller bandwidth should give higher ratio (more peaked): ratio_k2={:.6} vs ratio_k3={:.6}",
        ratio_k2, ratio_k3
    );

        // Also verify: larger bandwidth gives higher absolute affinity to nearest neighbor
        assert!(
            a1_k3 > a1_k2,
            "Larger bandwidth should give higher absolute affinity to nearest neighbor"
        );
    }

    #[test]
    fn test_phate_bandwidth_scale_effect() {
        let knn_indices = vec![vec![0, 1, 2, 3]];
        let knn_dists = vec![vec![0.0, 1.0, 4.0, 9.0]];

        // Test with bandwidth_scale = 1.0
        let graph_scale1 = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            2,
            Some(2.0),
            1.0,
            1e-10,
            "none",
            true,
        );

        // Test with bandwidth_scale = 2.0 (wider kernel)
        let graph_scale2 = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            2,
            Some(2.0),
            2.0,
            1e-10,
            "none",
            true,
        );

        let adj1 = graph_to_adj(&graph_scale1);
        let adj2 = graph_to_adj(&graph_scale2);

        // For a distant neighbor, wider bandwidth should give higher affinity
        let a3_scale1 = adj1[0].iter().find(|(j, _)| *j == 3).unwrap().1;
        let a3_scale2 = adj2[0].iter().find(|(j, _)| *j == 3).unwrap().1;

        println!(
            "scale=1.0: a3={:.6}, scale=2.0: a3={:.6}",
            a3_scale1, a3_scale2
        );
        assert!(
            a3_scale2 > a3_scale1,
            "Larger bandwidth scale should give higher affinity to distant neighbors"
        );
    }

    #[test]
    fn test_phate_decay_parameter_effect() {
        let knn_indices = vec![vec![0, 1, 2, 3, 4]];
        let knn_dists = vec![vec![0.0, 1.0, 4.0, 9.0, 16.0]];

        // Low decay (α=10): gentler falloff
        let graph_low = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            2,
            Some(1.0),
            1.0,
            1e-10,
            "none",
            true,
        );

        // High decay (α=80): sharper falloff
        let graph_high = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            2,
            Some(2.0),
            1.0,
            1e-10,
            "none",
            true,
        );

        let adj_low = graph_to_adj(&graph_low);
        let adj_high = graph_to_adj(&graph_high);

        // For distant neighbor, low decay should give higher affinity
        if let Some(&(_, a4_low)) = adj_low[0].iter().find(|(j, _)| *j == 4) {
            if let Some(&(_, a4_high)) = adj_high[0].iter().find(|(j, _)| *j == 4) {
                println!("decay=10: a4={:.6}, decay=80: a4={:.6}", a4_low, a4_high);
                assert!(
                    a4_low > a4_high,
                    "Lower decay should give higher affinity to distant neighbors"
                );
            }
        }
    }

    #[test]
    fn test_phate_thresholding() {
        let knn_indices = vec![vec![0, 1, 2, 3, 4, 5]];
        // Create distances that will result in some very small affinities
        let knn_dists = vec![vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0]];

        // Strict threshold - should exclude far neighbors
        let graph_strict = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            2,
            Some(2.0),
            1.0,
            1e-5, // higher threshold
            "none",
            true,
        );

        // Lenient threshold - should include more neighbors
        let graph_lenient = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            2,
            Some(2.0),
            1.0,
            1e-10, // lower threshold
            "none",
            true,
        );

        println!(
            "Strict threshold edges: {}, Lenient threshold edges: {}",
            graph_strict.values.len(),
            graph_lenient.values.len()
        );

        assert!(
            graph_lenient.values.len() >= graph_strict.values.len(),
            "Lenient threshold should produce at least as many edges"
        );
    }

    #[test]
    fn test_phate_binary_connectivity() {
        let knn_indices = vec![
            vec![0, 1, 2, 3],
            vec![1, 0, 2, 3],
            vec![2, 0, 1, 3],
            vec![3, 0, 1, 2],
        ];
        let knn_dists = vec![
            vec![0.0, 1.0, 4.0, 9.0],
            vec![0.0, 1.0, 4.0, 9.0],
            vec![0.0, 4.0, 1.0, 9.0],
            vec![0.0, 9.0, 4.0, 1.0],
        ];

        // decay = None should give binary connectivity
        let graph = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            2,
            None, // binary mode
            1.0,
            1e-4,
            "none",
            true,
        );

        // All edges should have weight 1.0
        for &v in &graph.values {
            assert_relative_eq!(v, 1.0, epsilon = 1e-10);
        }
        println!(
            "Binary connectivity: all {} edges have weight 1.0",
            graph.values.len()
        );
    }

    #[test]
    fn test_phate_additive_symmetrisation() {
        // Create an asymmetric scenario
        let knn_indices = vec![
            vec![0, 1, 2],
            vec![1, 2, 0], // Note: 1 is closer to 2 than to 0
            vec![2, 1, 0],
        ];
        let knn_dists = vec![
            vec![0.0, 1.0, 4.0],
            vec![0.0, 1.0, 9.0],
            vec![0.0, 1.0, 4.0],
        ];

        let graph = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            2,
            Some(2.0),
            1.0,
            1e-4,
            "add", // additive symmetrisation
            true,
        );

        // Check symmetry: K[i,j] should equal K[j,i]
        let edges: std::collections::HashMap<(usize, usize), f64> = graph
            .row_indices
            .iter()
            .zip(&graph.col_indices)
            .zip(&graph.values)
            .map(|((&i, &j), &v)| ((i, j), v))
            .collect();

        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            if let (Some(&v_ij), Some(&v_ji)) = (edges.get(&(i, j)), edges.get(&(j, i))) {
                assert_relative_eq!(v_ij, v_ji, epsilon = 1e-6, max_relative = 1e-6);
                println!(
                    "Symmetric edge ({},{}): v_ij={:.6}, v_ji={:.6}",
                    i, j, v_ij, v_ji
                );
            }
        }
    }

    #[test]
    fn test_phate_multiplicative_symmetrisation() {
        let knn_indices = vec![vec![0, 1, 2], vec![1, 0, 2], vec![2, 0, 1]];
        let knn_dists = vec![
            vec![0.0, 1.0, 4.0],
            vec![0.0, 1.0, 4.0],
            vec![0.0, 4.0, 1.0],
        ];

        // Get asymmetric version first
        let graph_asym = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            2,
            Some(2.0),
            1.0,
            1e-10,
            "none",
            true,
        );

        // Get multiplicative symmetrized version
        let graph_sym = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            2,
            Some(2.0),
            1.0,
            1e-10,
            "multiply",
            true,
        );

        // Multiplicative should have fewer edges (only mutual neighbors)
        assert!(
            graph_sym.values.len() <= graph_asym.values.len(),
            "Multiplicative symmetrisation should have <= edges"
        );

        // Check symmetry
        let edges: std::collections::HashMap<(usize, usize), f64> = graph_sym
            .row_indices
            .iter()
            .zip(&graph_sym.col_indices)
            .zip(&graph_sym.values)
            .map(|((&i, &j), &v)| ((i, j), v))
            .collect();

        for (&(i, j), &v_ij) in &edges {
            if let Some(&v_ji) = edges.get(&(j, i)) {
                assert_relative_eq!(v_ij, v_ji, epsilon = 1e-6);
            }
        }
        println!(
            "Multiplicative symmetrisation: {} edges",
            graph_sym.values.len()
        );
    }

    #[test]
    fn test_phate_symmetrisation_comparison() {
        let knn_indices = vec![
            vec![0, 1, 2, 3],
            vec![1, 0, 2, 3],
            vec![2, 1, 0, 3],
            vec![3, 0, 1, 2],
        ];
        let knn_dists = vec![
            vec![0.0, 1.0, 4.0, 9.0],
            vec![0.0, 2.0, 3.0, 8.0],
            vec![0.0, 1.0, 5.0, 7.0],
            vec![0.0, 6.0, 7.0, 8.0],
        ];

        let graph_none = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            2,
            Some(2.0),
            1.0,
            1e-10,
            "none",
            true,
        );
        let graph_add = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            2,
            Some(2.0),
            1.0,
            1e-10,
            "add",
            true,
        );
        let graph_mult = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            2,
            Some(2.0),
            1.0,
            1e-10,
            "multiply",
            true,
        );

        println!("Asymmetric: {} edges", graph_none.values.len());
        println!("Additive: {} edges", graph_add.values.len());
        println!("Multiplicative: {} edges", graph_mult.values.len());

        // Additive should preserve most edges
        assert!(graph_add.values.len() >= graph_mult.values.len());

        // All methods should produce reasonable number of edges
        assert!(!graph_add.values.is_empty());
        assert!(!graph_mult.values.is_empty());
    }

    #[test]
    fn test_phate_affinity_formula_verification() {
        // Manual verification of the affinity formula
        let knn_indices = vec![vec![0, 1]];
        let knn_dists = vec![vec![0.0, 4.0]]; // squared distance = 4, actual distance = 2

        let decay = 2.0;
        let bandwidth_scale = 1.0;

        let graph = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            2,
            Some(decay),
            bandwidth_scale,
            1e-10,
            "none",
            true,
        );

        // Manual calculation:
        // bandwidth = sqrt(4.0) * 1.0 = 2.0
        // distance = sqrt(4.0) = 2.0
        // scaled = 2.0 / 2.0 = 1.0
        // powered = 1.0^40 = 1.0
        // affinity = exp(-1.0) ≈ 0.3678794411714423

        let expected = (-1.0_f64).exp();
        let actual = graph.values[0];

        println!("Expected affinity: {:.10}", expected);
        println!("Actual affinity: {:.10}", actual);

        assert_relative_eq!(actual, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_phate_zero_distance_handling() {
        // Test that zero distances (duplicates) are handled correctly
        let knn_indices = vec![vec![0, 1, 2]];
        let knn_dists = vec![vec![0.0, 0.0, 1.0]]; // duplicate at distance 0

        let graph = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            2,
            Some(40.0),
            1.0,
            1e-4,
            "none",
            true,
        );

        // Find affinity for the zero-distance neighbor
        let adj = graph_to_adj(&graph);
        if let Some(&(j, v)) = adj[0].iter().find(|(j, _)| *j == 1) {
            println!("Zero-distance neighbor {} has affinity {:.6}", j, v);
            assert_relative_eq!(v, 1.0, epsilon = 1e-6,);
        }
    }

    #[test]
    fn test_phate_large_dataset_edge_count() {
        // Test with a larger dataset to ensure edge count is reasonable
        let n = 100;
        let k = 10;

        let knn_indices: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                let mut indices: Vec<usize> = (0..k).map(|j| (i + j) % n).collect();
                indices[0] = i; // self at position 0
                indices
            })
            .collect();

        let knn_dists: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                let mut dists: Vec<f64> = (0..k).map(|j| (j as f64).powi(2)).collect();
                dists[0] = 0.0; // self distance
                dists
            })
            .collect();

        let graph = phate_alpha_decay_affinities(
            &knn_indices,
            &knn_dists,
            5,
            Some(2.0),
            1.0,
            1e-10,
            "add",
            true,
        );

        println!(
            "Large dataset: {} vertices, {} edges",
            n,
            graph.values.len()
        );

        // Should have roughly n * (k-1) edges (minus self-loops and thresholding)
        assert!(!graph.values.is_empty());
        assert!(graph.values.len() <= n * (k - 1) * 2); // *2 for symmetrisation
    }
}
