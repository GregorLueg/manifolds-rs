//! Recursive label-propagation initialisation, a port of umap-learn 0.6's
//! `init="recursive"`.
//!
//! The fuzzy graph is coarsened by label propagation from high-degree hubs
//! into about a quarter as many parts, recursively, until it has at most
//! `RECURSIVE_BASE_THRESHOLD` nodes. That level is placed from a sampled-PCA
//! anchor of the data. Going back up, each coarse level is laid out with a
//! short run of the 0.6 Adam, expanded onto the finer level and rotated onto
//! the anchor by Procrustes. Coarse graphs are small, so they are held in
//! `f64`; only the finest level keeps the caller's float type.

use faer::Mat;
use rand::{rngs::SmallRng, seq::index::sample, Rng, SeedableRng};
use rayon::prelude::*;

use crate::data::structures::CoordinateList;
use crate::prelude::*;
use crate::training::umap_csr::UmapCsrGraph;
use crate::training::umap_optimiser_v06::{optimise_layout_v06, V06Config};
use crate::utils::math::randomised_svd;

///////////////
// Constants //
///////////////

/// Graphs with at most this many nodes are placed from the anchor rather than
/// coarsened further.
const RECURSIVE_BASE_THRESHOLD: usize = 1024;

/// Target reduction in node count per level.
const RECURSIVE_COARSENING_RATIO: usize = 4;

/// Epochs of the coarse layout at the finest level. Each coarser level runs
/// `sqrt(2)` times as many, since its graph is smaller and cheaper.
const RECURSIVE_BASE_EPOCHS: usize = 50;

/// Exponent applied to `a` and `b` once at the top and once per level down
/// (the reference's `strong_to_one` schedule). Coarse levels get a flatter
/// curve, so parts spread out rather than clump.
const RECURSIVE_CURVE_EXPONENT: f64 = 0.25;

/// Repulsion strength of the coarse layouts.
const RECURSIVE_GAMMA: f64 = 4.0;

/// Negative samples per positive sample in the coarse layouts.
const RECURSIVE_NEG_SAMPLE_RATE: usize = 1;

/// Initial learning rate of the coarse layouts.
const RECURSIVE_LR: f64 = 0.5;

/// Hard-negative window of a coarse layout, as a fraction of its node count.
const RECURSIVE_NEG_RANGE_FRACTION: f64 = 0.5;

/// Upper bound on label-propagation sweeps per level. Propagation usually
/// settles well before.
const LABEL_PROP_MAX_ITER: usize = 100;

/// Convergence of label propagation is checked every this many sweeps.
const LABEL_PROP_CHECK_EVERY: usize = 5;

/// Below the top level, a node joins the label holding the largest share of
/// its edge weight, if that share exceeds this. Transition weights are
/// normalised per part, so the top level's fixed threshold of 1 would never
/// be met there.
const LABEL_PROP_SHARE: f64 = 0.1;

/// Vote a label needs at the top level, where weights are fuzzy memberships.
const LABEL_PROP_TOP_THRESHOLD: f64 = 1.0;

/// Steps of the breadth-first search that hands a label to a node label
/// propagation left unlabelled.
const OUTLIER_SEARCH_STEPS: usize = 100;

/// Rows of the data the anchor PCA is computed on.
const ANCHOR_SAMPLE_SIZE: usize = 16_384;

/// Oversampling of the anchor's randomised SVD.
const ANCHOR_OVERSAMPLING: usize = 8;

/// Power iterations of the anchor's randomised SVD. The anchor only orients
/// the layout, so one is enough.
const ANCHOR_POWER_ITER: usize = 1;

/// Per-axis range the anchor is scaled to.
const ANCHOR_RANGE: f64 = 10.0;

/// Quantiles whose span sets the base-case spread.
const BASE_SPREAD_QUANTILES: (f64, f64) = (0.05, 0.95);

/// Default output range: the largest absolute coordinate.
pub const RECURSIVE_RANGE: f64 = 10.0;

////////////
// Graphs //
////////////

/// Weighted graph in CSR form.
struct Csr<W> {
    /// Row pointers, `n + 1` long
    indptr: Vec<usize>,
    /// Column of each entry
    indices: Vec<u32>,
    /// Weight of each entry
    data: Vec<W>,
}

impl<W> Csr<W>
where
    W: ManifoldsFloat,
{
    /// Number of nodes.
    ///
    /// ### Returns
    ///
    /// Row count.
    fn n(&self) -> usize {
        self.indptr.len() - 1
    }

    /// Entries of row `i`.
    ///
    /// ### Params
    ///
    /// * `i` - Row
    ///
    /// ### Returns
    ///
    /// Iterator of `(column, weight)`.
    fn row(&self, i: usize) -> impl Iterator<Item = (usize, W)> + '_ {
        let (lo, hi) = (self.indptr[i], self.indptr[i + 1]);
        self.indices[lo..hi]
            .iter()
            .map(|&j| j as usize)
            .zip(self.data[lo..hi].iter().copied())
    }

    /// Build from a COO graph with a counting sort on rows.
    ///
    /// ### Params
    ///
    /// * `graph` - Symmetric graph in COO form
    ///
    /// ### Returns
    ///
    /// The graph in CSR form, entry order within a row as in the COO.
    fn from_coo(graph: &CoordinateList<W>) -> Result<Self, ManifoldsError> {
        let n = graph.n_samples;
        if n > u32::MAX as usize {
            return Err(ManifoldsError::TooManyVertices { n });
        }
        let mut indptr = vec![0usize; n + 1];
        for &r in &graph.row_indices {
            indptr[r + 1] += 1;
        }
        for i in 0..n {
            indptr[i + 1] += indptr[i];
        }
        let nnz = indptr[n];
        let mut indices = vec![0u32; nnz];
        let mut data = vec![W::zero(); nnz];
        let mut cursor = indptr[..n].to_vec();
        for ((&r, &c), &w) in graph
            .row_indices
            .iter()
            .zip(&graph.col_indices)
            .zip(&graph.values)
        {
            indices[cursor[r]] = c as u32;
            data[cursor[r]] = w;
            cursor[r] += 1;
        }
        Ok(Self {
            indptr,
            indices,
            data,
        })
    }

    /// Adjacency list with self-loops dropped, the input format of
    /// [`UmapCsrGraph::from_graph`].
    ///
    /// ### Returns
    ///
    /// `adj[i] = [(j, w), ...]`.
    fn to_adjacency(&self) -> Vec<Vec<(usize, W)>> {
        (0..self.n())
            .map(|i| self.row(i).filter(|&(j, _)| j != i).collect())
            .collect()
    }
}

/// Assemble a CSR graph from per-row entry lists.
///
/// ### Params
///
/// * `rows` - `(column, weight)` entries per row
///
/// ### Returns
///
/// The graph.
fn csr_from_rows(rows: Vec<Vec<(u32, f64)>>) -> Csr<f64> {
    let mut indptr = Vec::with_capacity(rows.len() + 1);
    indptr.push(0);
    let nnz = rows.iter().map(Vec::len).sum();
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    for row in rows {
        for (t, w) in row {
            indices.push(t);
            data.push(w);
        }
        indptr.push(indices.len());
    }
    Csr {
        indptr,
        indices,
        data,
    }
}

/// Summed coarse graph `R^T W R`: total fine weight between parts, internal
/// weight on the diagonal. Parallel over parts, each with a dense scratch
/// row and a list of the parts it touched.
///
/// ### Params
///
/// * `graph` - Fine graph (or the finer level's summed graph)
/// * `labels` - Part of each fine node
/// * `members` - Fine nodes per part
///
/// ### Returns
///
/// The summed coarse graph, columns sorted, zero entries dropped.
fn summed_coarse<W>(graph: &Csr<W>, labels: &[u32], members: &[Vec<u32>]) -> Csr<f64>
where
    W: ManifoldsFloat,
{
    let n_parts = members.len();
    let rows = members
        .par_iter()
        .map_init(
            || {
                (
                    vec![0.0f64; n_parts],
                    vec![false; n_parts],
                    Vec::<u32>::new(),
                )
            },
            |(acc, seen, touched), part| {
                for &i in part {
                    for (j, w) in graph.row(i as usize) {
                        let t = labels[j];
                        if !seen[t as usize] {
                            seen[t as usize] = true;
                            touched.push(t);
                        }
                        acc[t as usize] += w.to_f64().unwrap();
                    }
                }
                touched.sort_unstable();
                let row = touched
                    .iter()
                    .filter(|&&t| acc[t as usize] != 0.0)
                    .map(|&t| (t, acc[t as usize]))
                    .collect();
                for &t in touched.iter() {
                    acc[t as usize] = 0.0;
                    seen[t as usize] = false;
                }
                touched.clear();
                row
            },
        )
        .collect();
    csr_from_rows(rows)
}

/// The graph a coarse level is laid out on, built the way UMAP builds its own
/// graph.
///
/// Each part's directed membership to another is the share of its external
/// weight going there, `f_PQ = W_PQ / sum_Q' W_PQ'`, and the two directions
/// are combined by the fuzzy union `f_PQ + f_QP - f_PQ f_QP`. Part sizes
/// cancel, so no single large part sets the scale.
///
/// ### Params
///
/// * `summed` - Symmetric summed coarse graph
///
/// ### Returns
///
/// Symmetric layout graph without self-loops, weights in `[0, 1]`.
fn transition_layout_edges(summed: &Csr<f64>) -> Csr<f64> {
    let n = summed.n();
    let out: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|p| summed.row(p).filter(|&(q, _)| q != p).map(|(_, w)| w).sum())
        .collect();
    let inv = |p: usize| if out[p] > 0.0 { 1.0 / out[p] } else { 1.0 };

    let rows: Vec<Vec<(u32, f64)>> = (0..n)
        .into_par_iter()
        .map(|p| {
            summed
                .row(p)
                .filter(|&(q, w)| q != p && w != 0.0)
                .map(|(q, w)| {
                    let (fpq, fqp) = (w * inv(p), w * inv(q));
                    (q as u32, (fpq + fqp - fpq * fqp).clamp(0.0, 1.0))
                })
                .collect()
        })
        .collect();
    csr_from_rows(rows)
}

///////////////////////
// Label propagation //
///////////////////////

/// Seed labels on the `n_parts` highest-degree nodes.
///
/// ### Params
///
/// * `graph` - Graph of this level
/// * `n_parts` - Number of seeds
///
/// ### Returns
///
/// Labels, `-1` for unseeded nodes.
fn seed_hub_labels<W>(graph: &Csr<W>, n_parts: usize) -> Vec<i64>
where
    W: ManifoldsFloat,
{
    let n = graph.n();
    let degrees: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| graph.row(i).map(|(_, w)| w.to_f64().unwrap()).sum())
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.par_sort_unstable_by(|&i, &j| degrees[i].total_cmp(&degrees[j]).then(i.cmp(&j)));
    let mut labels = vec![-1i64; n];
    // as the reference: the lowest-degree hub gets label 0
    for (l, &i) in order[n - n_parts..].iter().enumerate() {
        labels[i] = l as i64;
    }
    labels
}

/// One sweep of label propagation over the unlabelled nodes.
///
/// A node takes the label whose neighbours hold the largest summed weight,
/// provided it is at least the threshold: `LABEL_PROP_TOP_THRESHOLD` at the
/// top level, `LABEL_PROP_SHARE` of the node's total edge weight below. A
/// later label equal to the running maximum wins the tie, as in the
/// reference, whose current label is always unset here.
///
/// ### Params
///
/// * `graph` - Graph of this level
/// * `labels` - Current labels, `-1` for unlabelled
/// * `share_rule` - Use the share rule rather than the fixed threshold
///
/// ### Returns
///
/// The labels after the sweep.
fn label_prop_sweep<W>(graph: &Csr<W>, labels: &[i64], share_rule: bool) -> Vec<i64>
where
    W: ManifoldsFloat,
{
    (0..graph.n())
        .into_par_iter()
        .map_init(Vec::<(i64, f64)>::new, |votes, i| {
            if labels[i] >= 0 {
                return labels[i];
            }
            votes.clear();
            let mut total = 0.0;
            for (j, w) in graph.row(i) {
                let w = w.to_f64().unwrap();
                total += w;
                let l = labels[j];
                match votes.iter_mut().find(|(vl, _)| *vl == l) {
                    Some(v) => v.1 += w,
                    None => votes.push((l, w)),
                }
            }
            let mut best = if share_rule {
                LABEL_PROP_SHARE * total
            } else {
                LABEL_PROP_TOP_THRESHOLD
            };
            let mut result = -1;
            for &(l, v) in votes.iter() {
                if l < 0 {
                    continue;
                }
                if v >= best {
                    best = v;
                    result = l;
                }
            }
            result
        })
        .collect()
}

/// Give every node left unlabelled the label of the first labelled node a
/// short depth-first search from it reaches, or a random label.
///
/// Sequential, as the reference: a node labelled here can label later ones.
///
/// ### Params
///
/// * `graph` - Graph of this level
/// * `labels` - Labels, modified in place
/// * `rng` - Random generator for the fallback
fn label_outliers<W>(graph: &Csr<W>, labels: &mut [i64], rng: &mut SmallRng)
where
    W: ManifoldsFloat,
{
    let n_labels = (labels.iter().copied().max().unwrap_or(-1) + 1).max(1);
    let mut stack = Vec::new();
    for i in 0..graph.n() {
        if labels[i] >= 0 {
            continue;
        }
        stack.clear();
        stack.push(i);
        let mut steps = 0;
        'search: while steps < OUTLIER_SEARCH_STEPS {
            let Some(node) = stack.pop() else { break };
            steps += 1;
            for (j, _) in graph.row(node) {
                if labels[j] >= 0 {
                    labels[i] = labels[j];
                    break 'search;
                }
                stack.push(j);
            }
        }
        if labels[i] < 0 {
            labels[i] = rng.random_range(0..n_labels);
        }
    }
}

/// Map labels onto `0..n_parts`, keeping their order.
///
/// ### Params
///
/// * `labels` - Non-negative labels
///
/// ### Returns
///
/// `(labels, n_parts)`.
fn remap_labels(labels: &[i64]) -> (Vec<u32>, usize) {
    let max = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut map = vec![u32::MAX; max + 1];
    for &l in labels {
        map[l as usize] = 0;
    }
    let mut next = 0u32;
    for m in map.iter_mut() {
        if *m == 0 {
            *m = next;
            next += 1;
        }
    }
    (
        labels.iter().map(|&l| map[l as usize]).collect(),
        next as usize,
    )
}

/// Partition a level by label propagation from hubs.
///
/// ### Params
///
/// * `graph` - Graph of this level
/// * `n_parts` - Number of hub seeds
/// * `share_rule` - Use the share rule (all levels below the top)
/// * `rng` - Random generator for the outlier fallback
///
/// ### Returns
///
/// `(labels, n_parts)` with contiguous labels.
fn partition<W>(
    graph: &Csr<W>,
    n_parts: usize,
    share_rule: bool,
    rng: &mut SmallRng,
) -> (Vec<u32>, usize)
where
    W: ManifoldsFloat,
{
    let mut labels = seed_hub_labels(graph, n_parts);
    let count_unlabelled = |l: &[i64]| l.par_iter().filter(|&&x| x < 0).count();
    let mut prev = count_unlabelled(&labels);
    for it in 0..LABEL_PROP_MAX_ITER {
        labels = label_prop_sweep(graph, &labels, share_rule);
        if it % LABEL_PROP_CHECK_EVERY == 0 {
            let now = count_unlabelled(&labels);
            if now == 0 || now == prev {
                break;
            }
            prev = now;
        }
    }
    label_outliers(graph, &mut labels, rng);
    remap_labels(&labels)
}

////////////
// Layout //
////////////

/// Anchor coordinates of the sampled nodes of one level.
struct Anchor {
    /// Flat row-major coordinates of the sampled nodes, in node order
    coords: Vec<f64>,
    /// Which nodes are sampled
    mask: Vec<bool>,
}

/// Linear-interpolation quantile of a column, as numpy's default.
///
/// ### Params
///
/// * `values` - The values; sorted in place
/// * `q` - Quantile in `[0, 1]`
///
/// ### Returns
///
/// The quantile, `0` for an empty slice.
fn quantile(values: &mut [f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(|a, b| a.total_cmp(b));
    let pos = q * (values.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(values.len() - 1);
    values[lo] + (pos - lo as f64) * (values[hi] - values[lo])
}

/// Sampled-PCA anchor that orients every level.
///
/// PCA of up to [`ANCHOR_SAMPLE_SIZE`] rows, each axis scaled to
/// `[0, ANCHOR_RANGE]`. Axes PCA cannot provide (fewer features or samples
/// than `n_dim`) are filled uniformly in the same range.
///
/// ### Params
///
/// * `data` - Input data, samples x features
/// * `n_dim` - Embedding dimension
/// * `seed` - Seed for the sample and the SVD
///
/// ### Returns
///
/// The anchor.
fn anchor_reference<T>(
    data: faer::MatRef<T>,
    n_dim: usize,
    seed: u64,
) -> Result<Anchor, ManifoldsError>
where
    T: ManifoldsFloat,
{
    let (n, n_feat) = (data.nrows(), data.ncols());
    let mut rng = SmallRng::seed_from_u64(seed);
    let rows: Vec<usize> = if ANCHOR_SAMPLE_SIZE < n {
        let mut s = sample(&mut rng, n, ANCHOR_SAMPLE_SIZE).into_vec();
        s.sort_unstable();
        s
    } else {
        (0..n).collect()
    };
    let mut mask = vec![false; n];
    rows.iter().for_each(|&r| mask[r] = true);

    let m = rows.len();
    let mut centred = Mat::<f64>::from_fn(m, n_feat, |i, j| data[(rows[i], j)].to_f64().unwrap());
    for j in 0..n_feat {
        let mean = centred.col(j).iter().sum::<f64>() / m as f64;
        centred.col_mut(j).iter_mut().for_each(|x| *x -= mean);
    }

    let rank = n_dim.min(n_feat).min(m);
    let svd = randomised_svd(
        centred.as_ref(),
        rank,
        seed as usize,
        Some(ANCHOR_OVERSAMPLING),
        Some(ANCHOR_POWER_ITER),
    )?;

    let mut coords = vec![0.0f64; m * n_dim];
    for c in 0..n_dim {
        if c < rank {
            let col: Vec<f64> = (0..m).map(|i| svd.u[(i, c)] * svd.s[c]).collect();
            let min = col.iter().copied().fold(f64::INFINITY, f64::min);
            let max = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let span = if max - min == 0.0 { 1.0 } else { max - min };
            for (i, v) in col.iter().enumerate() {
                coords[i * n_dim + c] = (v - min) / span * ANCHOR_RANGE;
            }
        } else {
            for i in 0..m {
                coords[i * n_dim + c] = rng.random_range(0.0..ANCHOR_RANGE);
            }
        }
    }
    Ok(Anchor { coords, mask })
}

/// Coordinates for every node of a level from the sampled ones: an unsampled
/// node takes the edge-weighted mean of its sampled neighbours, or the mean
/// of all sampled nodes.
///
/// ### Params
///
/// * `graph` - Graph of this level
/// * `anchor` - Anchor of this level
/// * `n_dim` - Embedding dimension
///
/// ### Returns
///
/// Flat coordinates `[n * n_dim]`.
fn fill_unsampled<W>(graph: &Csr<W>, anchor: &Anchor, n_dim: usize) -> Vec<f64>
where
    W: ManifoldsFloat,
{
    let n = graph.n();
    let mut out = vec![0.0f64; n * n_dim];
    let mut k = 0;
    for i in 0..n {
        if anchor.mask[i] {
            out[i * n_dim..(i + 1) * n_dim]
                .copy_from_slice(&anchor.coords[k * n_dim..(k + 1) * n_dim]);
            k += 1;
        }
    }
    let n_sampled = k.max(1) as f64;
    let mut fallback = vec![0.0; n_dim];
    for p in anchor.coords.chunks_exact(n_dim) {
        for d in 0..n_dim {
            fallback[d] += p[d] / n_sampled;
        }
    }
    for i in 0..n {
        if anchor.mask[i] {
            continue;
        }
        let mut acc = vec![0.0; n_dim];
        let mut total = 0.0;
        for (j, w) in graph.row(i) {
            if anchor.mask[j] {
                let w = w.to_f64().unwrap();
                total += w;
                for d in 0..n_dim {
                    acc[d] += w * out[j * n_dim + d];
                }
            }
        }
        for d in 0..n_dim {
            out[i * n_dim + d] = if total > 0.0 {
                acc[d] / total
            } else {
                fallback[d]
            };
        }
    }
    out
}

/// Layout of a graph small enough to place from its anchor.
///
/// Centred, each axis given a 5th-95th percentile span of
/// `log10(n) * 3 * log2(depth + 1)`.
///
/// ### Params
///
/// * `graph` - Graph of this level
/// * `anchor` - Anchor of this level
/// * `n_dim` - Embedding dimension
/// * `depth` - Recursion depth, `1` at the finest level
///
/// ### Returns
///
/// Flat coordinates `[n * n_dim]`.
fn base_case_layout<W>(graph: &Csr<W>, anchor: &Anchor, n_dim: usize, depth: usize) -> Vec<f64>
where
    W: ManifoldsFloat,
{
    let mut out = if anchor.mask.iter().all(|&m| m) {
        anchor.coords.clone()
    } else {
        fill_unsampled(graph, anchor, n_dim)
    };
    let n = graph.n();
    let scale = (n as f64).log10() * 3.0 * ((depth + 1) as f64).log2();
    let mut col = vec![0.0; n];
    for d in 0..n_dim {
        let mean = (0..n).map(|i| out[i * n_dim + d]).sum::<f64>() / n as f64;
        for i in 0..n {
            out[i * n_dim + d] -= mean;
            col[i] = out[i * n_dim + d];
        }
        let lo = quantile(&mut col, BASE_SPREAD_QUANTILES.0);
        let hi = quantile(&mut col, BASE_SPREAD_QUANTILES.1);
        let spread = if hi - lo == 0.0 { 1.0 } else { hi - lo };
        for i in 0..n {
            out[i * n_dim + d] *= scale / spread;
        }
    }
    out
}

/// Expand a coarse layout onto the finer level: each node goes halfway
/// between its own part and the edge-weighted mean of its neighbours' parts.
///
/// ### Params
///
/// * `graph` - Graph of the finer level
/// * `labels` - Part of each fine node
/// * `coarse` - Flat coarse layout `[n_parts * n_dim]`
/// * `n_dim` - Embedding dimension
///
/// ### Returns
///
/// Flat fine layout `[n * n_dim]`.
fn expand_layout<W>(graph: &Csr<W>, labels: &[u32], coarse: &[f64], n_dim: usize) -> Vec<f64>
where
    W: ManifoldsFloat,
{
    let mut out = vec![0.0f64; graph.n() * n_dim];
    out.par_chunks_exact_mut(n_dim)
        .enumerate()
        .for_each(|(i, p)| {
            let mut total = 0.0;
            for (j, w) in graph.row(i) {
                let w = w.to_f64().unwrap();
                total += w;
                let l = labels[j] as usize;
                for d in 0..n_dim {
                    p[d] += w * coarse[l * n_dim + d];
                }
            }
            let own = labels[i] as usize;
            for d in 0..n_dim {
                if total > 0.0 {
                    p[d] /= total;
                }
                p[d] = 0.5 * (p[d] + coarse[own * n_dim + d]);
            }
        });
    out
}

/// Rotation that best maps `moving` onto `target` after centring and RMS
/// scaling, without reflection.
///
/// ### Params
///
/// * `target` - Flat target points `[m * n_dim]`
/// * `moving` - Flat points to rotate `[m * n_dim]`
/// * `n_dim` - Dimension
///
/// ### Returns
///
/// Row-major `n_dim x n_dim` rotation `R`, applied as `moving @ R`; the
/// identity if either set has collapsed.
fn procrustes_rotation(
    target: &[f64],
    moving: &[f64],
    n_dim: usize,
) -> Result<Vec<f64>, ManifoldsError> {
    let m = target.len() / n_dim;
    let identity = (0..n_dim * n_dim)
        .map(|k| if k / n_dim == k % n_dim { 1.0 } else { 0.0 })
        .collect();
    let centred = |x: &[f64]| -> (Vec<f64>, f64) {
        let mut mean = vec![0.0; n_dim];
        for p in x.chunks_exact(n_dim) {
            for d in 0..n_dim {
                mean[d] += p[d] / m as f64;
            }
        }
        let c: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(k, v)| v - mean[k % n_dim])
            .collect();
        let rms = (c.iter().map(|v| v * v).sum::<f64>() / c.len() as f64).sqrt();
        (c, rms)
    };
    let (t, t_rms) = centred(target);
    let (mv, m_rms) = centred(moving);
    if t_rms == 0.0 || m_rms == 0.0 {
        return Ok(identity);
    }
    // covariance = moving^T target, both RMS-scaled
    let cov = Mat::<f64>::from_fn(n_dim, n_dim, |r, c| {
        (0..m)
            .map(|i| mv[i * n_dim + r] * t[i * n_dim + c])
            .sum::<f64>()
            / (m_rms * t_rms)
    });
    let svd = cov.svd().map_err(|_| ManifoldsError::FaerSvdError)?;
    let (u, v) = (svd.U(), svd.V());
    let uvt = u * v.transpose();
    let mut u = u.to_owned();
    if uvt.determinant() < 0.0 {
        u.col_mut(n_dim - 1).iter_mut().for_each(|x| *x = -*x);
    }
    let rot = &u * v.transpose();
    Ok((0..n_dim * n_dim)
        .map(|k| rot[(k / n_dim, k % n_dim)])
        .collect())
}

/// Settings that stay fixed down the recursion.
struct Level {
    /// Embedding dimension
    n_dim: usize,
    /// Seed for this run
    seed: u64,
}

/// Lay out one level, recursing through coarser ones.
///
/// ### Params
///
/// * `graph` - Graph of this level
/// * `summed` - Summed weights from the finer level, `None` at the top
/// * `anchor` - Anchor of this level
/// * `a` - Curve parameter `a` of this level's layout
/// * `b` - Curve parameter `b` of this level's layout
/// * `n_parts` - Hub seeds for this level, `None` at the top
/// * `n_epochs` - Epochs of this level's coarse layout
/// * `depth` - Recursion depth, `1` at the top
/// * `ctx` - Fixed settings
///
/// ### Returns
///
/// Flat layout of this level `[n * n_dim]`.
#[allow(clippy::too_many_arguments)]
fn layout_level<W>(
    graph: &Csr<W>,
    summed: Option<&Csr<f64>>,
    anchor: &Anchor,
    a: f64,
    b: f64,
    n_parts: Option<usize>,
    n_epochs: usize,
    depth: usize,
    ctx: &Level,
) -> Result<Vec<f64>, ManifoldsError>
where
    W: ManifoldsFloat,
{
    let n = graph.n();
    let n_dim = ctx.n_dim;
    if n <= RECURSIVE_BASE_THRESHOLD {
        return Ok(base_case_layout(graph, anchor, n_dim, depth));
    }

    let n_parts = n_parts
        .unwrap_or_else(|| RECURSIVE_BASE_THRESHOLD.max(n / RECURSIVE_COARSENING_RATIO))
        .min(n / 2)
        .max(2);
    let level_seed = ctx.seed.wrapping_add(depth as u64);
    let mut rng = SmallRng::seed_from_u64(level_seed);

    let (labels, n_coarse) = partition(graph, n_parts, summed.is_some(), &mut rng);
    let mut members = vec![Vec::new(); n_coarse];
    for (i, &l) in labels.iter().enumerate() {
        members[l as usize].push(i as u32);
    }

    // the reference also builds a fuzzy-union coarse graph, but with the
    // default transition weights nothing reads its weights
    let next_summed = match summed {
        Some(s) => summed_coarse(s, &labels, &members),
        None => summed_coarse(graph, &labels, &members),
    };
    let layout_edges = transition_layout_edges(&next_summed);

    // anchor of the coarse level: mean of each part's sampled members
    let mut sums = vec![0.0f64; n_coarse * n_dim];
    let mut counts = vec![0usize; n_coarse];
    let mut k = 0;
    for i in 0..n {
        if anchor.mask[i] {
            let l = labels[i] as usize;
            counts[l] += 1;
            for d in 0..n_dim {
                sums[l * n_dim + d] += anchor.coords[k * n_dim + d];
            }
            k += 1;
        }
    }
    let coarse_mask: Vec<bool> = counts.iter().map(|&c| c > 0).collect();
    let mut coarse_coords = Vec::with_capacity(n_coarse * n_dim);
    for l in 0..n_coarse {
        if counts[l] > 0 {
            coarse_coords.extend(
                sums[l * n_dim..(l + 1) * n_dim]
                    .iter()
                    .map(|s| s / counts[l] as f64),
            );
        }
    }
    let coarse_anchor = Anchor {
        coords: coarse_coords,
        mask: coarse_mask,
    };

    let next_a = a.powf(RECURSIVE_CURVE_EXPONENT);
    let next_b = b.powf(RECURSIVE_CURVE_EXPONENT);
    let mut coarse_layout = layout_level(
        &layout_edges,
        Some(&next_summed),
        &coarse_anchor,
        next_a,
        next_b,
        Some(n_parts / RECURSIVE_COARSENING_RATIO),
        (n_epochs as f64 * std::f64::consts::SQRT_2) as usize,
        depth + 1,
        ctx,
    )?;
    let good_init = n_parts / RECURSIVE_COARSENING_RATIO > RECURSIVE_BASE_THRESHOLD;

    let adjacency = layout_edges.to_adjacency();
    // a coarse graph of isolated parts has nothing to optimise
    if adjacency.iter().any(|r| !r.is_empty()) {
        let csr = UmapCsrGraph::from_graph(&adjacency)?;
        let range =
            ((RECURSIVE_NEG_RANGE_FRACTION * n_coarse as f64).round() as usize).clamp(1, n_coarse);
        let cfg = V06Config {
            a,
            b,
            gamma: RECURSIVE_GAMMA,
            lr: RECURSIVE_LR,
            n_epochs,
            neg_sample_rate: RECURSIVE_NEG_SAMPLE_RATE,
            neg_selection_range: range,
            good_init,
        };
        optimise_layout_v06(&mut coarse_layout, n_dim, &csr, &cfg, level_seed, 0)?;
    }

    let mut result = expand_layout(graph, &labels, &coarse_layout, n_dim);
    let mut mean = vec![0.0; n_dim];
    for p in result.chunks_exact(n_dim) {
        for d in 0..n_dim {
            mean[d] += p[d] / n as f64;
        }
    }
    result
        .iter_mut()
        .enumerate()
        .for_each(|(k, v)| *v -= mean[k % n_dim]);

    let sampled: Vec<f64> = result
        .chunks_exact(n_dim)
        .zip(&anchor.mask)
        .filter(|(_, &m)| m)
        .flat_map(|(p, _)| p.iter().copied())
        .collect();
    let rot = procrustes_rotation(&anchor.coords, &sampled, n_dim)?;
    result.par_chunks_exact_mut(n_dim).for_each(|p| {
        let old = p.to_vec();
        for c in 0..n_dim {
            p[c] = (0..n_dim).map(|r| old[r] * rot[r * n_dim + c]).sum();
        }
    });
    Ok(result)
}

/// Recursive label-propagation initialisation of a UMAP embedding.
///
/// ### Params
///
/// * `graph` - Symmetric fuzzy graph in COO form
/// * `data` - Input data, samples x features; only feeds the PCA anchor
/// * `n_dim` - Embedding dimension
/// * `a` - UMAP curve parameter `a`
/// * `b` - UMAP curve parameter `b`
/// * `range` - Largest absolute coordinate of the output, [`RECURSIVE_RANGE`]
///   by default. One factor for all axes, so the aspect ratio survives.
/// * `seed` - Random seed
///
/// ### Returns
///
/// Initial coordinates `[n][n_dim]`.
///
/// ### References
///
/// McInnes, umap-learn 0.6, `umap/label_prop.py` (`recursive_init`).
pub fn recursive_layout<T>(
    graph: &CoordinateList<T>,
    data: faer::MatRef<T>,
    n_dim: usize,
    a: T,
    b: T,
    range: Option<T>,
    seed: u64,
) -> Result<Vec<Vec<T>>, ManifoldsError>
where
    T: ManifoldsFloat,
{
    let n = graph.n_samples;
    if n == 0 {
        return Err(ManifoldsError::NoData);
    }
    let fine = Csr::from_coo(graph)?;
    let anchor = anchor_reference(data, n_dim, seed)?;
    let ctx = Level { n_dim, seed };
    let a0 = a.to_f64().unwrap().powf(RECURSIVE_CURVE_EXPONENT);
    let b0 = b.to_f64().unwrap().powf(RECURSIVE_CURVE_EXPONENT);
    let flat = layout_level(
        &fine,
        None,
        &anchor,
        a0,
        b0,
        None,
        RECURSIVE_BASE_EPOCHS,
        1,
        &ctx,
    )?;

    let max_abs = flat.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let target = range
        .map(|r| r.to_f64().unwrap())
        .unwrap_or(RECURSIVE_RANGE);
    let mult = if max_abs > 0.0 { target / max_abs } else { 1.0 };
    Ok(flat
        .chunks_exact(n_dim)
        .map(|p| p.iter().map(|v| T::from_f64(v * mult).unwrap()).collect())
        .collect())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Symmetric CSR from undirected `(i, j, w)` edges.
    fn csr(n: usize, edges: &[(usize, usize, f64)]) -> Csr<f64> {
        let mut rows = vec![Vec::new(); n];
        for &(i, j, w) in edges {
            rows[i].push((j as u32, w));
            if i != j {
                rows[j].push((i as u32, w));
            }
        }
        rows.iter_mut()
            .for_each(|r| r.sort_unstable_by_key(|e| e.0));
        csr_from_rows(rows)
    }

    /// Two rings of `n_per` nodes in COO, each node linked to its `k` ring
    /// neighbours on either side.
    fn two_rings_coo(n_per: usize, k: usize) -> CoordinateList<f64> {
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for ring in 0..2 {
            for i in 0..n_per {
                for o in 1..=k {
                    let a = ring * n_per + i;
                    let b = ring * n_per + (i + o) % n_per;
                    r.extend([a, b]);
                    c.extend([b, a]);
                    v.extend([0.8, 0.8]);
                }
            }
        }
        CoordinateList {
            row_indices: r,
            col_indices: c,
            values: v,
            n_samples: 2 * n_per,
        }
    }

    /// Data whose first coordinate tells the rings apart.
    fn ring_data(n_per: usize) -> Mat<f64> {
        let mut rng = SmallRng::seed_from_u64(4);
        Mat::from_fn(2 * n_per, 5, |i, j| {
            let shift = if j == 0 && i >= n_per { 20.0 } else { 0.0 };
            shift + rng.random_range(-1.0..1.0)
        })
    }

    #[test]
    fn test_remap_labels_contiguous_and_ordered() {
        let (labels, n) = remap_labels(&[7, 2, 7, 9, 2]);
        assert_eq!(n, 3);
        assert_eq!(labels, vec![1, 0, 1, 2, 0]);
    }

    #[test]
    fn test_partition_labels_every_node() {
        let coo = two_rings_coo(1500, 5);
        let g = Csr::from_coo(&coo).unwrap();
        let mut rng = SmallRng::seed_from_u64(0);
        let (labels, n_parts) = partition(&g, 750, false, &mut rng);
        assert_eq!(labels.len(), 3000);
        assert!((2..=3000).contains(&n_parts));
        assert!(labels.iter().all(|&l| (l as usize) < n_parts));
    }

    #[test]
    fn test_summed_coarse_hand_case() {
        // path 0-1-2-3, parts {0,1} and {2,3}
        let g = csr(4, &[(0, 1, 0.5), (1, 2, 0.25), (2, 3, 1.0)]);
        let labels = vec![0, 0, 1, 1];
        let members = vec![vec![0, 1], vec![2, 3]];
        let s = summed_coarse(&g, &labels, &members);
        let row0: Vec<_> = s.row(0).collect();
        let row1: Vec<_> = s.row(1).collect();
        assert_eq!(row0, vec![(0, 1.0), (1, 0.25)]);
        assert_eq!(row1, vec![(0, 0.25), (1, 2.0)]);
    }

    #[test]
    fn test_transition_layout_edges_hand_case() {
        // part 0 sends 1 to part 1 and 3 to part 2; parts 1 and 2 only link to 0
        let s = csr(3, &[(0, 0, 5.0), (0, 1, 1.0), (0, 2, 3.0)]);
        let e = transition_layout_edges(&s);
        let row0: Vec<_> = e.row(0).collect();
        assert_eq!(row0.len(), 2);
        // f_01 = 1/4, f_10 = 1 -> 1/4 + 1 - 1/4 = 1
        assert_relative_eq!(row0[0].1, 1.0, epsilon = 1e-12);
        assert_relative_eq!(row0[1].1, 1.0, epsilon = 1e-12);
        assert!(e.row(1).all(|(q, _)| q != 1));
    }

    #[test]
    fn test_procrustes_recovers_rotation() {
        let target = vec![0.0, 0.0, 2.0, 0.0, 0.0, 1.0, -1.0, 3.0];
        let theta: f64 = 0.7;
        let (c, s) = (theta.cos(), theta.sin());
        // moving = target rotated by theta, so R must rotate it back
        let moving: Vec<f64> = target
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|p| [c * p[0] - s * p[1], s * p[0] + c * p[1]])
            .collect();
        let rot = procrustes_rotation(&target, &moving, 2).unwrap();
        let back: Vec<f64> = moving
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|p| [p[0] * rot[0] + p[1] * rot[2], p[0] * rot[1] + p[1] * rot[3]])
            .collect();
        let mean = |x: &[f64], d: usize| x.iter().skip(d).step_by(2).sum::<f64>() / 4.0;
        for k in 0..8 {
            let (tc, bc) = (
                target[k] - mean(&target, k % 2),
                back[k] - mean(&back, k % 2),
            );
            assert_relative_eq!(tc, bc, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_procrustes_collapsed_is_identity() {
        let rot = procrustes_rotation(&[1.0, 1.0, 1.0, 1.0], &[0.0, 1.0, 2.0, 3.0], 2).unwrap();
        assert_eq!(rot, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_expand_layout_two_parts() {
        // 0-1 inside part 0, 1-2 across, 2-3 inside part 1
        let g = csr(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]);
        let coarse = vec![0.0, 0.0, 4.0, 0.0];
        let out = expand_layout(&g, &[0, 0, 1, 1], &coarse, 2);
        // node 1: neighbours in parts 0 and 1 -> mean x = 2, own 0 -> 1
        assert_relative_eq!(out[2], 1.0);
        // node 0: only part 0 -> 0
        assert_relative_eq!(out[0], 0.0);
        assert_relative_eq!(out[6], 4.0);
    }

    #[test]
    fn test_recursive_layout_shape_range_reproducible() {
        let coo = two_rings_coo(1500, 5);
        let data = ring_data(1500);
        let (a, b) = (1.577, 0.895);
        let x = recursive_layout(&coo, data.as_ref(), 2, a, b, None, 3).unwrap();
        let y = recursive_layout(&coo, data.as_ref(), 2, a, b, None, 3).unwrap();
        assert_eq!(x.len(), 3000);
        assert!(x
            .iter()
            .all(|p| p.len() == 2 && p.iter().all(|v| v.is_finite())));
        let max_abs = x.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
        assert_relative_eq!(max_abs, RECURSIVE_RANGE, epsilon = 1e-9);
        assert_eq!(x, y);
    }

    #[test]
    fn test_recursive_layout_base_case_small_graph() {
        let coo = two_rings_coo(200, 5);
        let data = ring_data(200);
        let x = recursive_layout(&coo, data.as_ref(), 2, 1.577, 0.895, Some(1.0), 0).unwrap();
        let max_abs = x.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
        assert_relative_eq!(max_abs, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_recursive_layout_pads_missing_axes() {
        // one feature, three output dimensions
        let coo = two_rings_coo(100, 3);
        let data = Mat::from_fn(200, 1, |i, _| i as f64);
        let x = recursive_layout(&coo, data.as_ref(), 3, 1.577, 0.895, None, 1).unwrap();
        assert!(x
            .iter()
            .all(|p| p.len() == 3 && p.iter().all(|v| v.is_finite())));
    }
}
