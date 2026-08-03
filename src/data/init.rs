//! Module containing functions to initialise embeddings. This ranges from
//! random initialisation, PCA-based initialisation and spectral initialisation.

use faer::MatRef;
use rand::{
    rngs::StdRng,
    {Rng, SeedableRng},
};
use rand_distr::{Distribution, StandardNormal};
use std::collections::VecDeque;

use crate::data::structures::*;
use crate::prelude::*;
use crate::utils::math::*;

/// Range for spectral-based initialisation (based on uwot UMAP)
pub const SPECTRAL_RANGE: f64 = 10.0;

/// Range for randomly generated initialisation (based on uwot UMAP)
pub const RANDOM_RANGE: f64 = 10.0;

/// Range for PCA-based initialisation (based on uwot UMAP)
pub const PCA_RANGE: f64 = 1.0;

/// Component count above which the spectral meta-layout is abandoned.
///
/// `component_spectral_meta` builds a *dense* affinity graph over component
/// centroids, costing `O(n_components^2)` edges plus an
/// `O(n_components^2 * n_features)` distance loop before the Lanczos solve. At a
/// few hundred components that is free; at a few thousand it dwarfs the rest of
/// the initialisation.
///
/// The fallback above this threshold is deliberately NOT
/// `component_simplex_meta`: that gives a non-zero centroid to only the first
/// `2 * n_comp` components and leaves the rest at the origin, which would pile
/// thousands of components on top of each other.
const SPECTRAL_META_MAX_COMPONENTS: usize = 2_000;

/////////////
// Helpers //
/////////////

/// Different initialisation methods for the UMAP
#[derive(Clone, Debug)]
pub enum EmbdInit<T> {
    /// Spectral initialisation
    SpectralInit {
        /// Optional range parameters that overrides defaults
        range: Option<T>,
    },
    /// Random initialisation
    RandomInit {
        /// Optional range parameters that overrides defaults
        range: Option<T>,
    },
    /// PCA initialisation
    PcaInit {
        /// Optional range parameters that overrides defaults
        range: Option<T>,
        /// Shall randomised SVD be used
        randomised: bool,
    },
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
    T: ManifoldsFloat,
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

/////////////
// Helpers //
/////////////

/// Build a CSR row index over a COO graph via counting sort
///
/// Entries keep their COO order within a row. The BFS that consumes this walks
/// rows in that order, so the ordering is part of its reproducibility contract.
///
/// ### Params
///
/// * `graph` - Graph in COO format
///
/// ### Returns
///
/// `(indptr, indices)` describing the out-neighbours of each vertex.
fn coo_to_csr_adjacency<T>(graph: &CoordinateList<T>) -> (Vec<usize>, Vec<usize>)
where
    T: ManifoldsFloat,
{
    let n = graph.n_samples;

    let mut indptr = vec![0usize; n + 1];
    for &i in &graph.row_indices {
        indptr[i + 1] += 1;
    }
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }

    let mut indices = vec![0usize; graph.row_indices.len()];
    let mut cursor = indptr.clone();

    for (&i, &j) in graph.row_indices.iter().zip(&graph.col_indices) {
        indices[cursor[i]] = j;
        cursor[i] += 1;
    }

    (indptr, indices)
}

/// Build a column-sorted CSR from a COO graph via a two-pass radix sort
///
/// Sorts by column first, then stably by row, so each row ends up with its
/// column indices ascending, in `O(nnz + n)` and without a per-row allocation.
///
/// The column ordering is load-bearing: it fixes the summation order of the
/// sparse matrix-vector product downstream. Subgraphs cut for disconnected
/// components carry BFS-permuted local indices, so their columns genuinely do
/// arrive out of order.
///
/// ### Params
///
/// * `graph` - Graph in COO format
///
/// ### Returns
///
/// `(indptr, indices, values)` in CSR order with values cast to `f64`.
fn coo_to_csr_sorted<T>(graph: &CoordinateList<T>) -> (Vec<usize>, Vec<usize>, Vec<f64>)
where
    T: ManifoldsFloat,
{
    let n = graph.n_samples;
    let nnz = graph.row_indices.len();

    // pass one: order the edges by column
    let mut counts = vec![0usize; n + 1];
    for &j in &graph.col_indices {
        counts[j + 1] += 1;
    }
    for j in 0..n {
        counts[j + 1] += counts[j];
    }
    let mut by_col = vec![0usize; nnz];
    for (e, &j) in graph.col_indices.iter().enumerate() {
        by_col[counts[j]] = e;
        counts[j] += 1;
    }

    // pass two: stable scatter into rows, preserving the column order above
    let mut indptr = vec![0usize; n + 1];
    for &i in &graph.row_indices {
        indptr[i + 1] += 1;
    }
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }

    let mut indices = vec![0usize; nnz];
    let mut values = vec![0.0f64; nnz];
    let mut cursor = indptr.clone();

    for &e in &by_col {
        let i = graph.row_indices[e];
        let p = cursor[i];
        indices[p] = graph.col_indices[e];
        values[p] = graph.values[e].to_f64().unwrap();
        cursor[i] += 1;
    }

    (indptr, indices, values)
}

/// Convert COO graph to negative normalised adjacency in CSR format
///
/// Computes `M = - D^(-1/2) * A * D^(-1/2)`
/// This allows the Lanczos solver (which finds largest magnitude/smallest
/// algebraic) to converge to the cluster-structure eigenvectors (near -1) much
/// faster than finding Laplacian eigenvectors near 0.
///
/// Degrees include self-loops; the matrix itself does not.
///
/// ### Params
///
/// * `graph` - Symmetric weighted graph in COO format
///
/// ### Returns
///
/// Negative normalised adjacency matrix as CSR
fn graph_to_normalised_laplacian<T>(graph: &CoordinateList<T>) -> CompressedSparseData<f64>
where
    T: ManifoldsFloat,
{
    let n = graph.n_samples;
    let (indptr, indices, values) = coo_to_csr_sorted(graph);

    // accumulated in COO order, not CSR order: the two differ once a subgraph's
    // BFS-local indexing permutes the columns, and the sum order sets the bits
    let mut degrees = vec![0.0f64; n];
    for (&i, &w) in graph.row_indices.iter().zip(&graph.values) {
        degrees[i] += w.to_f64().unwrap();
    }

    // Compute D^(-1/2), handling isolated vertices
    let d_inv_sqrt: Vec<f64> = degrees
        .iter()
        .map(|&d| if d > 1e-8 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();

    // Build matrix: M = - D^(-1/2) * A * D^(-1/2)
    let mut out_data = Vec::with_capacity(values.len());
    let mut out_indices = Vec::with_capacity(indices.len());
    let mut out_indptr = Vec::with_capacity(n + 1);
    out_indptr.push(0);

    for i in 0..n {
        for idx in indptr[i]..indptr[i + 1] {
            let j = indices[idx];
            if i != j {
                // Keep the negative sign
                out_indices.push(j);
                out_data.push(-d_inv_sqrt[i] * values[idx] * d_inv_sqrt[j]);
            }
        }
        out_indptr.push(out_data.len());
    }

    // constructed directly rather than via `new_csr`, which copies all three
    CompressedSparseData {
        data: out_data,
        indices: out_indices,
        indptr: out_indptr,
        cs_type: CompressedSparseFormat::Csr,
        shape: (n, n),
    }
}

/// Compute raw spectral embedding for a single connected component
///
/// Computes eigenvectors of the negative normalised adjacency matrix and
/// returns them as embedding coordinates. Falls back to random initialisation
/// for graphs too small for spectral decomposition.
///
/// ### Params
///
/// * `graph` - Connected graph in COO format
/// * `n_comp` - Number of embedding dimensions
/// * `seed` - Random seed
///
/// ### Returns
///
/// Raw embedding coordinates, one vector per vertex
fn single_component_spectral_raw<T>(
    graph: &CoordinateList<T>,
    n_comp: usize,
    seed: u64,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
{
    let n = graph.n_samples;
    let mut embedding = vec![vec![T::zero(); n_comp]; n];

    if n <= n_comp + 1 {
        let mut rng = StdRng::seed_from_u64(seed);
        for row in &mut embedding {
            for v in row.iter_mut() {
                *v = T::from_f64(rng.random_range(-1.0..1.0)).unwrap();
            }
        }
        return Ok(embedding);
    }

    let laplacian = graph_to_normalised_laplacian(graph);
    let n_eigs = (n_comp + 1).min(n);
    let (_, evecs) = compute_smallest_eigenpairs_lanczos(&laplacian, n_eigs, seed)?;

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

    Ok(embedding)
}

/// Connected-component decomposition of a graph
///
/// Carries the inverse maps alongside the membership lists so per-component
/// subgraphs can be cut in a single pass over the edges.
struct Components {
    /// Vertex indices belonging to each component
    members: Vec<Vec<usize>>,
    /// Component label of each vertex
    labels: Vec<usize>,
    /// Position of each vertex within its own component
    local: Vec<usize>,
}

/// Find connected components in a sparse graph using BFS
///
/// The traversal order (ascending start vertex, FIFO queue, CSR row order) fixes
/// both the component ordering and the vertex ordering within each component,
/// which in turn fixes the local indices every subgraph is cut with.
///
/// Note this follows edges in the row-to-column direction only. On a symmetric
/// graph that gives true connected components; on a directed graph, such as the
/// raw kNN graph PaCMAP passes in, it gives forward reachability.
///
/// ### Params
///
/// * `graph` - Sparse graph in COO format
///
/// ### Returns
///
/// The component decomposition, including per-vertex label and local index.
fn find_connected_components<T>(graph: &CoordinateList<T>) -> Components
where
    T: ManifoldsFloat,
{
    let n = graph.n_samples;
    let (indptr, indices) = coo_to_csr_adjacency(graph);

    let mut labels = vec![usize::MAX; n];
    let mut local = vec![0usize; n];
    let mut members = Vec::new();
    let mut queue = VecDeque::new();

    for start in 0..n {
        if labels[start] != usize::MAX {
            continue;
        }

        let label = members.len();
        let mut component = Vec::new();
        queue.clear();
        queue.push_back(start);
        labels[start] = label;

        while let Some(node) = queue.pop_front() {
            local[node] = component.len();
            component.push(node);
            for &neighbour in &indices[indptr[node]..indptr[node + 1]] {
                if labels[neighbour] == usize::MAX {
                    labels[neighbour] = label;
                    queue.push_back(neighbour);
                }
            }
        }

        members.push(component);
    }

    Components {
        members,
        labels,
        local,
    }
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
/// * `comps` - Component decomposition of `graph`
/// * `n_comp` - Number of embedding dimensions
/// * `seed` - Random seed
/// * `range` - Scaling range for embedding
/// * `data` - Optional `MatRef` to the data.
///
/// ### Returns
///
/// Initial embedding coordinates
fn multi_component_init<T>(
    graph: &CoordinateList<T>,
    comps: &Components,
    n_comp: usize,
    seed: u64,
    range: T,
    data: Option<MatRef<T>>,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
{
    let n = graph.n_samples;
    let mut embedding = vec![vec![T::zero(); n_comp]; n];
    let mut rng = StdRng::seed_from_u64(seed);

    // Position component centroids relative to each other, unit max-abs.
    // Final scaling to `range` happens at the end.
    let meta = component_meta_layout(&comps.members, n_comp, data, seed)?;

    // Cutting every subgraph costs a full copy of the edge list, so skip it
    // when no component is large enough to take the spectral branch below.
    // That is the common shape once the graph shatters into many small pieces.
    let needs_subgraphs = comps.members.iter().any(|c| c.len() >= 2 * n_comp);
    let subgraphs = if needs_subgraphs {
        extract_all_subgraphs(graph, comps)
    } else {
        Vec::new()
    };

    for (label, component) in comps.members.iter().enumerate() {
        let centroid = &meta[label];
        let data_range = meta_data_range(&meta, label);

        if component.len() < 2 * n_comp {
            // Too small for spectral: uniform random within data_range of centroid
            for &global in component {
                for d in 0..n_comp {
                    let u = T::from_f64(rng.random_range(-1.0..1.0)).unwrap();
                    embedding[global][d] = centroid[d] + u * data_range;
                }
            }
            continue;
        }

        // Spectral embedding within the component, expanded to data_range
        let sub = single_component_spectral_raw(&subgraphs[label], n_comp, seed + label as u64)?;

        let max_abs = sub
            .iter()
            .flat_map(|v| v.iter())
            .fold(T::zero(), |acc, &x| {
                let a = x.abs();
                if a > acc {
                    a
                } else {
                    acc
                }
            });
        let expansion = if max_abs > T::from_f64(1e-8).unwrap() {
            data_range / max_abs
        } else {
            T::one()
        };

        for (local, &global) in component.iter().enumerate() {
            for d in 0..n_comp {
                embedding[global][d] = centroid[d] + sub[local][d] * expansion;
            }
        }
    }

    // Centre and scale to `range`, matching the single-component path's contract
    let n_t = T::from_usize(n).unwrap();
    for d in 0..n_comp {
        let mean = embedding.iter().map(|v| v[d]).sum::<T>() / n_t;
        for row in &mut embedding {
            row[d] -= mean;
        }
    }
    let max_abs = embedding
        .iter()
        .flat_map(|v| v.iter())
        .fold(T::zero(), |acc, &x| {
            let a = x.abs();
            if a > acc {
                a
            } else {
                acc
            }
        });
    if max_abs > T::from_f64(1e-8).unwrap() {
        let s = range / max_abs;
        for row in &mut embedding {
            for v in row {
                *v *= s;
            }
        }
    }

    Ok(embedding)
}

/// Compute centroid layout for connected components
///
/// Positions component centroids in embedding space. Uses spectral embedding
/// of the inter-component affinity graph when the number of components exceeds
/// `2 * n_comp`, otherwise falls back to simplex placement. Normalises the
/// result so the maximum absolute value is 1.
///
/// ### Params
///
/// * `components` - Vector of component vertex indices
/// * `n_comp` - Number of embedding dimensions
/// * `data` - Optional feature matrix used for spectral meta-layout
/// * `seed` - Random seed
///
/// ### Returns
///
/// Centroid coordinates, one vector per component, normalised to unit max-abs
fn component_meta_layout<T>(
    components: &[Vec<usize>],
    n_comp: usize,
    data: Option<MatRef<T>>,
    seed: u64,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
{
    let n_components = components.len();

    let mut meta = if n_components <= 2 * n_comp {
        component_simplex_meta(n_components, n_comp)
    } else {
        match data {
            Some(_) if n_components > SPECTRAL_META_MAX_COMPONENTS => {
                component_scatter_meta(n_components, n_comp, seed)
            }
            Some(d) => component_spectral_meta(components, n_comp, d, seed)?,
            None => component_simplex_meta(n_components, n_comp),
        }
    };

    let mut max_abs = meta
        .iter()
        .flat_map(|v| v.iter())
        .fold(T::zero(), |acc, &x| {
            let a = x.abs();
            if a > acc {
                a
            } else {
                acc
            }
        });

    // spectral meta can still degenerate; fall back rather than collapse.
    if max_abs <= T::from_f64(1e-8).unwrap() {
        meta = component_simplex_meta(n_components, n_comp);
        max_abs = meta
            .iter()
            .flat_map(|v| v.iter())
            .fold(T::zero(), |acc, &x| {
                let a = x.abs();
                if a > acc {
                    a
                } else {
                    acc
                }
            });
    }

    if max_abs > T::from_f64(1e-8).unwrap() {
        for row in &mut meta {
            for v in row {
                *v /= max_abs;
            }
        }
    }
    Ok(meta)
}

/// Place component centroids on a simplex
///
/// Assigns centroids by stacking rows of the identity matrix and their
/// negations, taking the first `n_components` rows. Used when the number of
/// components is at most `2 * n_comp`.
///
/// ### Params
///
/// * `n_components` - Number of connected components
/// * `n_comp` - Number of embedding dimensions
///
/// ### Returns
///
/// Centroid coordinates, one vector per component
fn component_simplex_meta<T>(n_components: usize, n_comp: usize) -> Vec<Vec<T>>
where
    T: ManifoldsFloat,
{
    let k = n_components.div_ceil(2);
    let mut meta = vec![vec![T::zero(); n_comp]; n_components];
    for label in 0..n_components {
        let (idx, sign) = if label < k {
            (label, T::one())
        } else {
            (label - k, -T::one())
        };
        if idx < n_comp {
            meta[label][idx] = sign;
        }
    }
    meta
}

/// Scatter component centroids pseudo-randomly over the unit cube
///
/// Fallback for when there are too many components to afford the dense
/// affinity graph `component_spectral_meta` needs. It carries no information
/// about how components relate, but it is `O(n_components * n_comp)` and, unlike
/// the simplex placement, gives every component its own position instead of
/// stacking all but the first `2 * n_comp` at the origin.
///
/// ### Params
///
/// * `n_components` - Number of connected components
/// * `n_comp` - Number of embedding dimensions
/// * `seed` - Random seed
///
/// ### Returns
///
/// Centroid coordinates, one vector per component
fn component_scatter_meta<T>(n_components: usize, n_comp: usize, seed: u64) -> Vec<Vec<T>>
where
    T: ManifoldsFloat,
{
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_components)
        .map(|_| {
            (0..n_comp)
                .map(|_| T::from_f64(rng.random_range(-1.0..1.0)).unwrap())
                .collect()
        })
        .collect()
}

/// Place component centroids via spectral embedding of their affinity graph
///
/// Computes per-component centroids in feature space, builds a fully connected
/// affinity graph between them using `exp(-d^2)`, and embeds that graph
/// spectrally. Used when the number of components exceeds `2 * n_comp`.
///
/// ### Params
///
/// * `components` - Vector of component vertex indices
/// * `n_comp` - Number of embedding dimensions
/// * `data` - Feature matrix used to compute centroids
/// * `seed` - Random seed
///
/// ### Returns
///
/// Centroid coordinates, one vector per component
fn component_spectral_meta<T>(
    components: &[Vec<usize>],
    n_comp: usize,
    data: MatRef<T>,
    seed: u64,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
{
    let n_components = components.len();
    let n_features = data.ncols();

    let mut centroids = vec![vec![T::zero(); n_features]; n_components];
    for (l, comp) in components.iter().enumerate() {
        let inv = T::one() / T::from_usize(comp.len()).unwrap();
        for &i in comp {
            for f in 0..n_features {
                centroids[l][f] += data[(i, f)];
            }
        }
        for f in 0..n_features {
            centroids[l][f] *= inv;
        }
    }

    // squared distances between centroids, with the max for normalisation.
    // without normalising, exp(-d^2) on separated clusters underflows the
    // affinity to exactly zero and the whole meta-layout collapses
    // completely...
    let mut sq_dists = vec![vec![T::zero(); n_components]; n_components];
    let mut max_d2 = T::zero();
    for a in 0..n_components {
        for b in (a + 1)..n_components {
            let mut d2 = T::zero();
            for f in 0..n_features {
                let diff = centroids[a][f] - centroids[b][f];
                d2 += diff * diff;
            }
            sq_dists[a][b] = d2;
            sq_dists[b][a] = d2;
            if d2 > max_d2 {
                max_d2 = d2;
            }
        }
    }
    let scale = if max_d2 > T::from_f64(1e-12).unwrap() {
        T::one() / max_d2
    } else {
        T::one()
    };

    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    for a in 0..n_components {
        for b in 0..n_components {
            if a == b {
                continue;
            }
            let aff = T::from_f64((-(sq_dists[a][b] * scale).to_f64().unwrap()).exp()).unwrap();
            row_indices.push(a);
            col_indices.push(b);
            values.push(aff);
        }
    }

    let affinity = CoordinateList {
        row_indices,
        col_indices,
        values,
        n_samples: n_components,
    };
    single_component_spectral_raw(&affinity, n_comp, seed)
}

/// Compute the spread radius for a component centroid
///
/// Returns half the maximum distance from the given centroid to any other
/// centroid. Used to scale per-component embeddings so they do not overlap.
///
/// ### Params
///
/// * `meta` - Centroid coordinates, one vector per component
/// * `label` - Index of the component whose radius is computed
///
/// ### Returns
///
/// Half the maximum distance to any other centroid
fn meta_data_range<T>(meta: &[Vec<T>], label: usize) -> T
where
    T: ManifoldsFloat,
{
    let mut max_d = T::zero();
    for other in 0..meta.len() {
        let mut d2 = T::zero();
        for d in 0..meta[label].len() {
            let diff = meta[label][d] - meta[other][d];
            d2 += diff * diff;
        }
        let dist = d2.sqrt();
        if dist > max_d {
            max_d = dist;
        }
    }
    max_d / T::from_f64(2.0).unwrap()
}

/// Cut every component's subgraph in a single pass over the edges
///
/// Bucketing by component label costs `O(nnz + n_components)` in total, rather
/// than one full edge scan per component. The scatter is stable, so each
/// subgraph keeps its edges in the original COO order.
///
/// Edges whose endpoints carry different labels are dropped. That can only
/// happen on a directed graph, where BFS reachability is asymmetric.
///
/// Note this holds every subgraph in memory at once, so peak usage is roughly a
/// second copy of the edge list.
///
/// ### Params
///
/// * `graph` - Full graph in COO format
/// * `comps` - Component decomposition of `graph`
///
/// ### Returns
///
/// One locally indexed subgraph per component, in component order.
fn extract_all_subgraphs<T>(graph: &CoordinateList<T>, comps: &Components) -> Vec<CoordinateList<T>>
where
    T: ManifoldsFloat,
{
    let n_components = comps.members.len();

    let mut offsets = vec![0usize; n_components + 1];
    for (&i, &j) in graph.row_indices.iter().zip(&graph.col_indices) {
        if comps.labels[i] == comps.labels[j] {
            offsets[comps.labels[i] + 1] += 1;
        }
    }
    for c in 0..n_components {
        offsets[c + 1] += offsets[c];
    }

    let total = offsets[n_components];
    let mut rows = vec![0usize; total];
    let mut cols = vec![0usize; total];
    let mut vals = vec![T::zero(); total];
    let mut cursor = offsets.clone();

    for ((&i, &j), &w) in graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
    {
        let label = comps.labels[i];
        if label != comps.labels[j] {
            continue;
        }
        let p = cursor[label];
        rows[p] = comps.local[i];
        cols[p] = comps.local[j];
        vals[p] = w;
        cursor[label] += 1;
    }

    (0..n_components)
        .map(|c| {
            let (lo, hi) = (offsets[c], offsets[c + 1]);
            CoordinateList {
                row_indices: rows[lo..hi].to_vec(),
                col_indices: cols[lo..hi].to_vec(),
                values: vals[lo..hi].to_vec(),
                n_samples: comps.members[c].len(),
            }
        })
        .collect()
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
    graph: &CoordinateList<T>,
    n_comp: usize,
    seed: u64,
    range: T,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
{
    let raw = single_component_spectral_raw(graph, n_comp, seed)?;
    Ok(finalise_spectral_embedding(raw, n_comp, range, seed))
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
    T: ManifoldsFloat,
{
    let n = embedding.len();
    let n_t = T::from_usize(n).unwrap();

    // Centre each component
    for comp in 0..n_comp {
        let mean: T = embedding.iter().map(|v| v[comp]).sum::<T>() / n_t;

        for i in 0..n {
            embedding[i][comp] -= mean;
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
                *val *= scale;
            }
        }
    }

    // Add small noise for numerical stability
    let mut rng = StdRng::seed_from_u64(seed + 9999);
    let noise_std = T::from_f64(1e-4).unwrap();

    for row in &mut embedding {
        for val in row {
            let noise = T::from_f64(rng.sample::<f64, _>(StandardNormal)).unwrap() * noise_std;
            *val += noise;
        }
    }

    embedding
}

//////////////////////////
// Main spectral layout //
//////////////////////////

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
    graph: &CoordinateList<T>,
    n_comp: usize,
    seed: u64,
    range: Option<T>,
    data: Option<MatRef<T>>,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
{
    let range = range.unwrap_or(T::from_f64(SPECTRAL_RANGE).unwrap());
    let comps = find_connected_components(graph);

    if comps.members.len() > 1 {
        return multi_component_init(graph, &comps, n_comp, seed, range, data);
    }

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
    T: ManifoldsFloat,
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
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
    StandardNormal: Distribution<T>,
{
    let target_std = range.unwrap_or(T::from_f64(PCA_RANGE).unwrap());
    let (n_samples, n_features) = (data.nrows(), data.ncols());

    // Centre the data
    let mut centred = data.to_owned();
    for j in 0..n_features {
        let mean = (0..n_samples).map(|i| data[(i, j)]).sum::<T>() / T::from(n_samples).unwrap();
        for i in 0..n_samples {
            centred[(i, j)] -= mean;
        }
    }

    // Compute SVD
    let svd_result = if randomised {
        randomised_svd(centred.as_ref(), n_comp, seed as usize, None, None)?
    } else {
        let svd = centred
            .thin_svd()
            .map_err(|_| ManifoldsError::FaerSvdError)?;
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

    Ok(embedding)
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
    graph: &CoordinateList<T>,
    data: MatRef<T>,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
    StandardNormal: Distribution<T>,
{
    match init_method {
        EmbdInit::SpectralInit { range } => {
            spectral_layout(graph, n_comp, seed, *range, Some(data))
        }
        EmbdInit::RandomInit { range } => {
            let n_samples = data.nrows();
            Ok(random_layout(n_samples, n_comp, seed, *range))
        }
        EmbdInit::PcaInit { randomised, range } => {
            Ok(pca_layout(data, n_comp, *randomised, *range, seed)?)
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
        let graph = CoordinateList {
            row_indices: vec![0, 1],
            col_indices: vec![1, 0],
            values: vec![1.0, 1.0],
            n_samples: 2,
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
        let graph = CoordinateList {
            row_indices: vec![0],
            col_indices: vec![1],
            values: vec![1.0],
            n_samples: 3,
        };

        let laplacian = graph_to_normalised_laplacian(&graph);

        assert_eq!(laplacian.shape(), (3, 3));
        // Should handle isolated vertex (vertex 2) gracefully
    }

    #[test]
    fn test_spectral_layout_basic() {
        // Create a simple connected graph
        let graph = CoordinateList {
            row_indices: vec![0, 0, 1, 1, 2, 2],
            col_indices: vec![1, 2, 0, 2, 0, 1],
            values: vec![1.0, 0.5, 1.0, 1.0, 0.5, 1.0],
            n_samples: 3,
        };

        let embedding = spectral_layout(&graph, 2, 42, None, None).unwrap();

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
        let graph = CoordinateList {
            row_indices: vec![0, 0, 1, 1, 2, 2],
            col_indices: vec![1, 2, 0, 2, 0, 1],
            values: vec![1.0, 0.5, 1.0, 1.0, 0.5, 1.0],
            n_samples: 3,
        };

        let embedding = spectral_layout(&graph, 2, 42, Some(1.0), None).unwrap();

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

    // NOTE: n = 3 with n_comp = 2 trips the `n <= n_comp + 1` guard in
    // `single_component_spectral_raw`, so this covers the random fallback, NOT
    // Lanczos. Real Lanczos determinism lives in `tests/spectral_bench.rs`.
    #[test]
    fn test_spectral_layout_reproducibility() {
        let graph = CoordinateList {
            row_indices: vec![0, 1, 2],
            col_indices: vec![1, 2, 0],
            values: vec![1.0, 1.0, 1.0],
            n_samples: 3,
        };

        let embd1 = spectral_layout(&graph, 2, 42, None, None).unwrap();
        let embd2 = spectral_layout(&graph, 2, 42, None, None).unwrap();

        assert_eq!(embd1, embd2);
    }

    #[test]
    fn test_spectral_layout_higher_dimensions() {
        let graph = CoordinateList {
            row_indices: vec![0, 1, 2, 3],
            col_indices: vec![1, 2, 3, 0],
            values: vec![1.0; 4],
            n_samples: 4,
        };

        let embedding = spectral_layout(&graph, 3, 42, None, None).unwrap();

        assert_eq!(embedding.len(), 4);
        assert_eq!(embedding[0].len(), 3);
    }

    #[test]
    fn test_spectral_layout_disconnected() {
        // Two disjoint edges: {0,1} and {2,3}
        let graph = CoordinateList {
            row_indices: vec![0, 1, 2, 3],
            col_indices: vec![1, 0, 3, 2],
            values: vec![1.0, 1.0, 1.0, 1.0],
            n_samples: 4,
        };

        // No data: forces the simplex meta-layout path
        let embedding = spectral_layout::<f64>(&graph, 2, 42, None, None).unwrap();

        assert_eq!(embedding.len(), 4);
        assert_eq!(embedding[0].len(), 2);

        // Centred and within range
        for dim in 0..2 {
            let mean: f64 = embedding.iter().map(|p| p[dim]).sum::<f64>() / 4.0;
            assert_relative_eq!(mean, 0.0, epsilon = 0.01);
        }
        for point in &embedding {
            for &coord in point {
                assert!((-10.01..=10.01).contains(&coord));
            }
        }
    }

    // NOTE: both components have 2 < 2 * n_comp vertices, so this covers the
    // random placement branch of `multi_component_init`, NOT Lanczos. Real
    // multi-component determinism lives in `tests/spectral_bench.rs`.
    #[test]
    fn test_spectral_layout_disconnected_reproducibility() {
        let graph = CoordinateList {
            row_indices: vec![0, 1, 2, 3],
            col_indices: vec![1, 0, 3, 2],
            values: vec![1.0, 1.0, 1.0, 1.0],
            n_samples: 4,
        };

        let embd1 = spectral_layout::<f64>(&graph, 2, 42, None, None).unwrap();
        let embd2 = spectral_layout::<f64>(&graph, 2, 42, None, None).unwrap();

        assert_eq!(embd1, embd2);
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
        let graph: CoordinateList<f32> = CoordinateList {
            row_indices: vec![],
            col_indices: vec![],
            values: vec![],
            n_samples: 1,
        };

        let embedding = spectral_layout(&graph, 2, 42, None, None).unwrap();

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
        let embedding = pca_layout(data.as_ref(), 2, false, None, 42).unwrap();
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
        let embedding = pca_layout(data.as_ref(), 2, false, Some(custom_range), 42).unwrap();
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

        let embd1 = pca_layout(data.as_ref(), 2, false, None, 42).unwrap();
        let embd2 = pca_layout(data.as_ref(), 2, false, None, 42).unwrap();

        assert_eq!(embd1, embd2);
    }

    #[test]
    fn test_pca_layout_randomised() {
        let data = faer::mat![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ];

        let embd_standard = pca_layout(data.as_ref(), 2, false, None, 42).unwrap();
        let embd_randomised = pca_layout(data.as_ref(), 2, true, None, 42).unwrap();

        assert_eq!(embd_standard.len(), 3);
        assert_eq!(embd_randomised.len(), 3);
    }

    #[test]
    fn test_initialise_embedding_spectral() {
        let graph = CoordinateList {
            row_indices: vec![0, 1],
            col_indices: vec![1, 0],
            values: vec![1.0, 1.0],
            n_samples: 2,
        };
        let data = faer::mat![[1.0, 2.0], [3.0, 4.0],];

        let embedding = initialise_embedding(
            &EmbdInit::SpectralInit { range: None },
            2,
            42,
            &graph,
            data.as_ref(),
        )
        .unwrap();

        assert_eq!(embedding.len(), 2);
        assert_eq!(embedding[0].len(), 2);
    }

    #[test]
    fn test_initialise_embedding_spectral_disconnected() {
        // Two disjoint edges, exercised through the full dispatch so the data
        // matrix is forwarded into the multi-component path
        let graph = CoordinateList {
            row_indices: vec![0, 1, 2, 3],
            col_indices: vec![1, 0, 3, 2],
            values: vec![1.0, 1.0, 1.0, 1.0],
            n_samples: 4,
        };
        let data = faer::mat![[1.0, 2.0], [1.5, 2.5], [8.0, 9.0], [8.5, 9.5],];

        let embedding = initialise_embedding(
            &EmbdInit::SpectralInit { range: None },
            2,
            42,
            &graph,
            data.as_ref(),
        )
        .unwrap();

        assert_eq!(embedding.len(), 4);
        assert_eq!(embedding[0].len(), 2);
    }

    #[test]
    fn test_initialise_embedding_random() {
        let graph = CoordinateList {
            row_indices: vec![],
            col_indices: vec![],
            values: vec![],
            n_samples: 3,
        };
        let data = faer::mat![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let embedding = initialise_embedding(
            &EmbdInit::RandomInit { range: None },
            2,
            42,
            &graph,
            data.as_ref(),
        )
        .unwrap();

        assert_eq!(embedding.len(), 3);
        assert_eq!(embedding[0].len(), 2);
    }

    #[test]
    fn test_initialise_embedding_pca() {
        let graph = CoordinateList {
            row_indices: vec![],
            col_indices: vec![],
            values: vec![],
            n_samples: 3,
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
        )
        .unwrap();

        assert_eq!(embedding.len(), 3);
        assert_eq!(embedding[0].len(), 2);
    }

    #[test]
    fn test_spectral_layout_disconnected_within_component_spectral() {
        // Two disjoint 5-cycles: {0,1,2,3,4} and {5,6,7,8,9}. Each component is
        // large enough (5 >= 2 * n_comp) to take the within-component spectral
        // branch rather than random placement.
        let graph = CoordinateList {
            row_indices: vec![
                0, 1, 1, 2, 2, 3, 3, 4, 4, 0, // first cycle
                5, 6, 6, 7, 7, 8, 8, 9, 9, 5, // second cycle
            ],
            col_indices: vec![
                1, 0, 2, 1, 3, 2, 4, 3, 0, 4, // first cycle
                6, 5, 7, 6, 8, 7, 9, 8, 5, 9, // second cycle
            ],
            values: vec![1.0; 20],
            n_samples: 10,
        };

        let embedding = spectral_layout::<f64>(&graph, 2, 42, None, None).unwrap();

        assert_eq!(embedding.len(), 10);
        assert_eq!(embedding[0].len(), 2);

        // Centred and within range
        for dim in 0..2 {
            let mean: f64 = embedding.iter().map(|p| p[dim]).sum::<f64>() / 10.0;
            assert_relative_eq!(mean, 0.0, epsilon = 0.01);
        }
        for point in &embedding {
            for &coord in point {
                assert!((-10.01..=10.01).contains(&coord));
            }
        }

        // The two components should occupy separable regions: the spread of
        // component-1 centroids vs component-2 centroids should be non-trivial
        let c1: Vec<f64> = (0..5).map(|i| embedding[i][0]).collect();
        let c2: Vec<f64> = (5..10).map(|i| embedding[i][0]).collect();
        let mean1: f64 = c1.iter().sum::<f64>() / 5.0;
        let mean2: f64 = c2.iter().sum::<f64>() / 5.0;
        assert!((mean1 - mean2).abs() > 1e-3);
    }
}
