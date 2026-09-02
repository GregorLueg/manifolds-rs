//! Shared machinery for the density-preserving embeddings densMAP and den-SNE.
//!
//! Both methods add `-lambda * Corr(log R_o, log R_e)` to their usual loss,
//! where `R_o` is a graph-weighted local radius in the original space and
//! `R_e` the matching kernel-weighted radius in the embedding. Maximising that
//! correlation makes a tight cluster stay tight and a diffuse one stay diffuse,
//! which plain UMAP and tSNE do not guarantee.
//!
//! Everything here is optimiser-agnostic: recovering the high-dimensional edge
//! distances, the (constant) original radii, and the per-epoch correlation
//! statistics. The per-edge gradient lives with each optimiser, since it needs
//! that optimiser's kernel.
//!
//! Two deliberate deviations from the reference implementations, both
//! documented at their use sites: we use the sample (ddof 1) form for every
//! variance and covariance (umap-learn mixes a population variance with a
//! sample covariance; the difference is `O(1/n)`), and we keep [`DENS_EPS`]
//! inside the log for den-SNE too, which the reference sets to zero.
//!
//! ### References
//!
//! Narayan, Berger & Cho, Nature Biotechnology, 2021.

use rayon::prelude::*;

use crate::data::structures::CoordinateList;
use crate::errors::ManifoldsError;
use crate::utils::traits::*;

/////////////
// Globals //
/////////////

/// Additive shift inside the log of both local radii. Keeps `log(0)` finite for
/// points whose neighbours have collapsed onto them. Matches umap-learn; the
/// den-SNE reference uses `0` and relies on the variance shift alone, which is
/// less forgiving.
pub const DENS_EPS: f64 = 1e-8;

/// Default density weight for densMAP.
pub const DENSMAP_LAMBDA: f64 = 2.0;

/// Default density weight for den-SNE. Twenty times smaller than densMAP's
/// because den-SNE's gradient has neither a log epsilon nor gradient clipping
/// to absorb a collapsed radius.
pub const DENSNE_LAMBDA: f64 = 0.1;

/// Default fraction of the run, at the end, over which the density term is
/// active. The preceding epochs are plain UMAP / tSNE.
pub const DENS_FRAC: f64 = 0.3;

/// Default additive shift on the variance of the embedding log-radii. Stops the
/// correlation from exploding while the embedding is still unstructured and its
/// radii nearly constant.
pub const DENS_VAR_SHIFT: f64 = 0.1;

/// Chunk size for the correlation reductions. Fixed rather than derived from
/// the thread count so the summation order, and therefore the embedding, is
/// identical across runs and machines. Large enough that the per-chunk `Vec` of
/// partials stays negligible.
const DENS_REDUCE_CHUNK: usize = 8192;

////////////
// Params //
////////////

/// Tunable knobs for the density-preserving term shared by densMAP and den-SNE.
#[derive(Debug, Clone, Copy)]
pub struct DensParams<T> {
    /// Weight of the density term. `0` disables it entirely, recovering plain
    /// UMAP / tSNE. See [`DENSMAP_LAMBDA`] and [`DENSNE_LAMBDA`].
    pub lambda: T,
    /// Fraction of the total epochs, at the end of the run, over which the
    /// density term is active. See [`DENS_FRAC`].
    pub frac: T,
    /// Additive shift on the variance of the embedding log-radii. See
    /// [`DENS_VAR_SHIFT`].
    pub var_shift: T,
}

impl<T> DensParams<T>
where
    T: ManifoldsFloat,
{
    /// Build density parameters, falling back to the documented defaults.
    ///
    /// ### Params
    ///
    /// * `lambda` - Density weight. Defaults to [`DENSMAP_LAMBDA`]; pass
    ///   [`DENSNE_LAMBDA`] explicitly for den-SNE
    /// * `frac` - Fraction of the final epochs with the term active. Defaults
    ///   to [`DENS_FRAC`]
    /// * `var_shift` - Variance shift. Defaults to [`DENS_VAR_SHIFT`]
    ///
    /// ### Returns
    ///
    /// The resolved parameter set.
    pub fn new(lambda: Option<T>, frac: Option<T>, var_shift: Option<T>) -> Self {
        Self {
            lambda: lambda.unwrap_or_else(|| T::from_f64(DENSMAP_LAMBDA).unwrap()),
            frac: frac.unwrap_or_else(|| T::from_f64(DENS_FRAC).unwrap()),
            var_shift: var_shift.unwrap_or_else(|| T::from_f64(DENS_VAR_SHIFT).unwrap()),
        }
    }

    /// densMAP defaults (`lambda = 2.0`).
    ///
    /// ### Returns
    ///
    /// Parameters with the densMAP density weight.
    pub fn densmap_default() -> Self {
        Self::new(None, None, None)
    }

    /// den-SNE defaults (`lambda = 0.1`).
    ///
    /// ### Returns
    ///
    /// Parameters with the den-SNE density weight.
    pub fn densne_default() -> Self {
        Self::new(Some(T::from_f64(DENSNE_LAMBDA).unwrap()), None, None)
    }
}

impl<T> Default for DensParams<T>
where
    T: ManifoldsFloat,
{
    fn default() -> Self {
        Self::densmap_default()
    }
}

///////////
// State //
///////////

/// Everything the optimiser needs that is constant across the whole run.
#[derive(Debug, Clone)]
pub struct DensState<T> {
    /// The density knobs.
    pub params: DensParams<T>,
    /// Z-scored log local radii in the original space, `[n]`. Node-indexed, in
    /// the same order as the embedding.
    pub r: Vec<T>,
    /// Per-node graph weight sum `sum_j mu_ij`, `[n]`. Row sums of the graph.
    pub mu_sum: Vec<T>,
    /// Total graph weight over every COO entry, i.e. twice the undirected
    /// total. Used by densMAP to undo the bias from visiting edges at a rate
    /// proportional to their weight; unused by den-SNE, which sweeps every edge
    /// every epoch.
    ///
    /// This matches umap-learn's `dens_mu_tot`, which is `sum(mu_sum) / 2` over
    /// a `mu_sum` that bumps *both* endpoints of every COO entry and is
    /// therefore twice our row sum. Keeping the same value keeps `lambda`
    /// comparable with the reference.
    pub mu_tot: T,
}

impl<T> DensState<T>
where
    T: ManifoldsFloat,
{
    /// Build the constant density state from the graph and the kNN results.
    ///
    /// ### Params
    ///
    /// * `params` - Density knobs
    /// * `graph` - The symmetrised graph the optimiser will run on. Both
    ///   directions of every edge are expected to be present
    /// * `knn_indices` - kNN indices per point, self excluded
    /// * `knn_dists` - kNN distances per point, aligned with `knn_indices`,
    ///   as true distances. `run_ann_search` roots the Euclidean backends'
    ///   squared output, so every metric arrives on the same footing
    ///
    /// ### Returns
    ///
    /// The density state, or [`ManifoldsError::DegenerateLocalRadii`] if every
    /// point has the same local radius and the correlation is undefined.
    pub fn new(
        params: DensParams<T>,
        graph: &CoordinateList<T>,
        knn_indices: &[Vec<usize>],
        knn_dists: &[Vec<T>],
    ) -> Result<Self, ManifoldsError> {
        let edge_d_sq = edge_distances_from_knn(graph, knn_indices, knn_dists);
        let (r, mu_sum) = original_log_radii(graph, &edge_d_sq)?;

        let mu_tot = mu_sum
            .iter()
            .fold(0.0f64, |acc, &m| acc + m.to_f64().unwrap());

        Ok(Self {
            params,
            r,
            mu_sum,
            mu_tot: T::from_f64(mu_tot).unwrap(),
        })
    }

    /// Reinterpret the state in a different float type.
    ///
    /// The GPU path runs in `f32` only, so the whole state has to be narrowed
    /// before it is uploaded. The radii are already z-scored and `O(1)`, so the
    /// narrowing is benign; the correlation statistics derived from them are
    /// still accumulated in `f64` on the host every epoch.
    ///
    /// ### Returns
    ///
    /// The same state with every field cast to `U`.
    pub fn cast<U>(&self) -> DensState<U>
    where
        U: ManifoldsFloat,
    {
        let convert = |x: T| U::from_f64(x.to_f64().unwrap_or(0.0)).unwrap();

        DensState {
            params: DensParams {
                lambda: convert(self.params.lambda),
                frac: convert(self.params.frac),
                var_shift: convert(self.params.var_shift),
            },
            r: self.r.iter().map(|&x| convert(x)).collect(),
            mu_sum: self.mu_sum.iter().map(|&x| convert(x)).collect(),
            mu_tot: convert(self.mu_tot),
        }
    }

    /// Whether the density term contributes to the gradient this epoch.
    ///
    /// Active for the final `frac` fraction of the run, and never when
    /// `lambda <= 0`.
    ///
    /// ### Params
    ///
    /// * `epoch` - Current epoch, 0-based
    /// * `n_epochs` - Total number of epochs
    ///
    /// ### Returns
    ///
    /// `true` if the density gradient should be applied.
    pub fn is_active(&self, epoch: usize, n_epochs: usize) -> bool {
        if self.params.lambda <= T::zero() || n_epochs == 0 {
            return false;
        }
        let progress = T::from_usize(epoch + 1).unwrap() / T::from_usize(n_epochs).unwrap();
        progress > T::one() - self.params.frac
    }
}

/////////////
// Helpers //
/////////////

/// Distance from `i` to `j` in `i`'s kNN row, if `j` is one of `i`'s
/// neighbours.
///
/// Linear scan over the row. `k` is small (tens), and the rows are contiguous
/// `Vec`s, so this beats building a hash map per node.
///
/// ### Params
///
/// * `knn_indices` - kNN indices per point
/// * `knn_dists` - kNN distances per point
/// * `i` - Row to search
/// * `j` - Neighbour to find
///
/// ### Returns
///
/// The stored distance, or `None` if `j` is not in `i`'s row.
#[inline]
fn lookup_knn_dist<T>(
    knn_indices: &[Vec<usize>],
    knn_dists: &[Vec<T>],
    i: usize,
    j: usize,
) -> Option<T>
where
    T: ManifoldsFloat,
{
    knn_indices
        .get(i)?
        .iter()
        .position(|&cand| cand == j)
        .and_then(|pos| knn_dists.get(i)?.get(pos).copied())
}

/// Recover the squared high-dimensional distance for every graph edge.
///
/// The graph builders consume the kNN distances and do not keep them, so they
/// are recovered here rather than plumbed through. Symmetrisation can create an
/// edge `(i, j)` where `j` is not in `i`'s kNN row, so the reverse row is
/// checked as a fallback; every graph edge originates from a kNN edge in at
/// least one direction, so this always resolves.
///
/// ### Params
///
/// * `graph` - Graph in COO form
/// * `knn_indices` - kNN indices per point, self excluded
/// * `knn_dists` - kNN distances per point, aligned with `knn_indices`
///
/// ### Returns
///
/// Squared distances, one per COO entry, aligned with `graph.values`.
pub fn edge_distances_from_knn<T>(
    graph: &CoordinateList<T>,
    knn_indices: &[Vec<usize>],
    knn_dists: &[Vec<T>],
) -> Vec<T>
where
    T: ManifoldsFloat,
{
    graph
        .row_indices
        .par_iter()
        .zip(graph.col_indices.par_iter())
        .map(|(&i, &j)| {
            let raw = lookup_knn_dist(knn_indices, knn_dists, i, j)
                .or_else(|| lookup_knn_dist(knn_indices, knn_dists, j, i))
                .unwrap_or_else(T::zero);

            raw * raw
        })
        .collect()
}

/// Per-node log local radius in the original space, z-scored.
///
/// `Ro[i] = log(eps + sum_j mu_ij d_ij^2 / sum_j mu_ij)`, accumulated row-wise
/// over the COO. The graph is symmetric, so a row sweep already sees every
/// neighbour of every node; bumping only the row endpoint keeps `mu_sum` the
/// natural row sum, which the per-epoch embedding radii must then match.
///
/// The accumulation is sequential. The COO is not sorted by row, so a parallel
/// scatter would need either atomics or a per-thread `[n]` buffer, and this
/// runs exactly once per embedding.
///
/// ### Params
///
/// * `graph` - Graph in COO form, both directions present
/// * `edge_d_sq` - Squared high-dimensional distance per COO entry, from
///   [`edge_distances_from_knn`]
///
/// ### Returns
///
/// `(r, mu_sum)` where `r` is the z-scored log radius per node and `mu_sum` the
/// per-node graph weight sum. [`ManifoldsError::DegenerateLocalRadii`] if the
/// radii have no spread, which makes the correlation undefined.
pub fn original_log_radii<T>(
    graph: &CoordinateList<T>,
    edge_d_sq: &[T],
) -> Result<(Vec<T>, Vec<T>), ManifoldsError>
where
    T: ManifoldsFloat,
{
    let n = graph.n_samples;
    if n == 0 {
        return Err(ManifoldsError::NoData);
    }

    let mut ro = vec![0.0f64; n];
    let mut mu_sum = vec![0.0f64; n];

    for ((&i, &mu), &d_sq) in graph
        .row_indices
        .iter()
        .zip(graph.values.iter())
        .zip(edge_d_sq.iter())
    {
        let mu = mu.to_f64().unwrap_or(0.0);
        ro[i] += mu * d_sq.to_f64().unwrap_or(0.0);
        mu_sum[i] += mu;
    }

    // an isolated node (every edge pruned by filter_weak_edges) would give 0/0;
    // the clamp turns that into log(DENS_EPS), a constant that drops out of the
    // correlation instead of poisoning every other point with a NaN.
    for (radius, &weight) in ro.iter_mut().zip(mu_sum.iter()) {
        let denom = weight.max(f64::EPSILON);
        *radius = (DENS_EPS + *radius / denom).ln();
    }

    standardise(&mut ro)?;

    let r = ro.iter().map(|&x| T::from_f64(x).unwrap()).collect();
    let mu_sum = mu_sum.iter().map(|&x| T::from_f64(x).unwrap()).collect();

    Ok((r, mu_sum))
}

/// Centre and scale in place to zero mean and unit sample standard deviation.
///
/// ### Params
///
/// * `values` - Values to standardise, modified in place
///
/// ### Returns
///
/// `Ok(())`, or [`ManifoldsError::DegenerateLocalRadii`] if the spread is zero
/// or not finite.
fn standardise(values: &mut [f64]) -> Result<(), ManifoldsError> {
    let n = values.len();
    if n < 2 {
        return Err(ManifoldsError::DegenerateLocalRadii);
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    let var = values
        .iter()
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / (n - 1) as f64;
    let sd = var.sqrt();

    // a NaN here means an inf crept into a radius upstream; treat it the same
    // as no spread rather than propagating it into every gradient
    if sd.is_nan() || sd <= f64::EPSILON {
        return Err(ManifoldsError::DegenerateLocalRadii);
    }

    for x in values.iter_mut() {
        *x = (*x - mean) / sd;
    }

    Ok(())
}

/// Per-epoch statistics of the embedding log-radii against the original ones.
///
/// Accumulated in `f64` regardless of `T`: this is a sum over `n` points of
/// log-radii with a subtracted mean, and the GPU path runs in `f32`.
///
/// Parallelised over fixed-size chunks whose partials are then summed in index
/// order, rather than via `par_iter().sum()`. Rayon's adaptive splitting varies
/// with thread scheduling, and float addition is not associative, so the
/// straightforward version makes the whole embedding non-reproducible for a
/// given seed. Same reasoning as the serial mean in `recentre_embedding`.
///
/// ### Params
///
/// * `re` - Embedding log local radii, `[n]`
/// * `r` - Z-scored original log local radii, `[n]`, from
///   [`original_log_radii`]
/// * `var_shift` - Additive shift applied to the variance before the square
///   root, see [`DENS_VAR_SHIFT`]
///
/// ### Returns
///
/// `(re_mean, re_std, cov)`, where `re_std = sqrt(var(re) + var_shift)` and
/// `cov` is the sample covariance of `re` with `r`. Since `r` is z-scored,
/// `cov / re_std` is the Pearson correlation.
pub fn correlation_stats<T>(re: &[T], r: &[T], var_shift: T) -> (T, T, T)
where
    T: ManifoldsFloat,
{
    let n = re.len();
    let denom = (n.saturating_sub(1)).max(1) as f64;

    let sums: Vec<f64> = re
        .par_chunks(DENS_REDUCE_CHUNK)
        .map(|chunk| {
            chunk
                .iter()
                .fold(0.0f64, |acc, x| acc + x.to_f64().unwrap_or(0.0))
        })
        .collect();
    let mean = sums.iter().sum::<f64>() / n.max(1) as f64;

    let partials: Vec<(f64, f64)> = re
        .par_chunks(DENS_REDUCE_CHUNK)
        .zip(r.par_chunks(DENS_REDUCE_CHUNK))
        .map(|(re_chunk, r_chunk)| {
            re_chunk
                .iter()
                .zip(r_chunk.iter())
                .fold((0.0f64, 0.0f64), |(ss, cross), (&e, &ri)| {
                    let d = e.to_f64().unwrap_or(0.0) - mean;
                    (ss + d * d, cross + d * ri.to_f64().unwrap_or(0.0))
                })
        })
        .collect();

    let (ss, cross) = partials
        .iter()
        .fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

    let var = ss / denom;
    let cov = cross / denom;
    let std = (var + var_shift.to_f64().unwrap_or(0.0)).sqrt();

    (
        T::from_f64(mean).unwrap(),
        T::from_f64(std).unwrap(),
        T::from_f64(cov).unwrap(),
    )
}

/// Per-node log local radius in the embedding, from pre-accumulated sums.
///
/// `re[i] = log(eps + weighted_sq[i] / kernel_sum[i])`. Both inputs are
/// accumulated by the optimiser inside its own edge sweep, since the kernel
/// differs between densMAP and den-SNE.
///
/// ### Params
///
/// * `weighted_sq` - `sum_j phi_ij * ||y_i - y_j||^2` per node, `[n]`
/// * `kernel_sum` - `sum_j phi_ij` per node, `[n]`
/// * `re` - Output buffer, `[n]`, overwritten
///
/// ### Returns
///
/// Nothing; `re` is overwritten.
pub fn embedding_log_radii<T>(weighted_sq: &[T], kernel_sum: &[T], re: &mut [T])
where
    T: ManifoldsFloat,
{
    let eps = T::from_f64(DENS_EPS).unwrap();
    let floor = T::epsilon();

    re.par_iter_mut()
        .zip(weighted_sq.par_iter())
        .zip(kernel_sum.par_iter())
        .for_each(|((out, &num), &den)| {
            *out = (eps + num / den.max(floor)).ln();
        });
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Two-node graph with both directions present.
    fn tiny_graph() -> CoordinateList<f64> {
        CoordinateList {
            row_indices: vec![0, 1],
            col_indices: vec![1, 0],
            values: vec![1.0, 1.0],
            n_samples: 2,
        }
    }

    #[test]
    fn test_edge_distances_squares_the_true_distance() {
        let graph = tiny_graph();
        let knn_indices = vec![vec![1], vec![0]];
        let knn_dists = vec![vec![9.0], vec![9.0]];

        let d_sq = edge_distances_from_knn(&graph, &knn_indices, &knn_dists);

        // kNN distances are true distances now, so squaring happens here and
        // only here. There is no longer a mode where the input is passed
        // through untouched.
        assert_relative_eq!(d_sq[0], 81.0);
        assert_relative_eq!(d_sq[1], 81.0);
    }

    #[test]
    fn test_edge_distances_falls_back_to_reverse_row() {
        // edge (0, 1) exists in the graph, but 1 is not in 0's kNN row.
        // symmetrisation creates exactly this case.
        let graph = tiny_graph();
        let knn_indices = vec![vec![7], vec![0]];
        let knn_dists = vec![vec![1.0], vec![2.0]];

        let d_sq = edge_distances_from_knn(&graph, &knn_indices, &knn_dists);

        assert_relative_eq!(d_sq[0], 4.0);
        assert_relative_eq!(d_sq[1], 4.0);
    }

    #[test]
    fn test_original_log_radii_known_value() {
        // node 0 has neighbours 1 and 2 with weights 1 and 3, d_sq 4 and 8.
        // node 1 and 2 each see only node 0.
        let graph = CoordinateList {
            row_indices: vec![0, 0, 1, 2],
            col_indices: vec![1, 2, 0, 0],
            values: vec![1.0, 3.0, 1.0, 3.0],
            n_samples: 3,
        };
        let edge_d_sq = vec![4.0, 8.0, 4.0, 8.0];

        let (r, mu_sum) = original_log_radii(&graph, &edge_d_sq).unwrap();

        assert_relative_eq!(mu_sum[0], 4.0);
        assert_relative_eq!(mu_sum[1], 1.0);
        assert_relative_eq!(mu_sum[2], 3.0);

        // raw radii before standardising: node 0 = (1*4 + 3*8)/4 = 7
        let raw = [7.0f64, 4.0, 8.0].map(|x| (DENS_EPS + x).ln());
        let mean = raw.iter().sum::<f64>() / 3.0;
        let var = raw.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 2.0;
        let sd = var.sqrt();

        for (got, want) in r.iter().zip(raw.iter()) {
            assert_relative_eq!(*got, (want - mean) / sd, epsilon = 1e-12);
        }

        // z-scored: mean 0, sample sd 1
        assert_relative_eq!(r.iter().sum::<f64>(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_original_log_radii_constant_is_degenerate() {
        let graph = tiny_graph();
        let edge_d_sq = vec![4.0, 4.0];

        let err = original_log_radii(&graph, &edge_d_sq);

        assert!(matches!(err, Err(ManifoldsError::DegenerateLocalRadii)));
    }

    #[test]
    fn test_original_log_radii_isolated_node_does_not_nan() {
        // node 2 has no edges at all -> mu_sum 2 is zero
        let graph: CoordinateList<f64> = CoordinateList {
            row_indices: vec![0, 1],
            col_indices: vec![1, 0],
            values: vec![1.0, 2.0],
            n_samples: 3,
        };
        let edge_d_sq = vec![4.0, 9.0];

        let (r, mu_sum) = original_log_radii(&graph, &edge_d_sq).unwrap();

        assert_relative_eq!(mu_sum[2], 0.0);
        assert!(r.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_correlation_stats_perfect_correlation() {
        // re is an affine function of r, so cov/std is 1 once var_shift is 0
        let r = vec![-1.0, 0.0, 1.0];
        let re: Vec<f64> = r.iter().map(|x| 3.0 * x + 5.0).collect();

        let (mean, std, cov) = correlation_stats(&re, &r, 0.0);

        assert_relative_eq!(mean, 5.0, epsilon = 1e-12);
        assert_relative_eq!(std, 3.0, epsilon = 1e-12);
        assert_relative_eq!(cov, 3.0, epsilon = 1e-12);
        assert_relative_eq!(cov / std, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_correlation_stats_var_shift_widens_std() {
        let r = vec![-1.0, 0.0, 1.0];
        let re = vec![-1.0, 0.0, 1.0];

        let (_, std_plain, _) = correlation_stats(&re, &r, 0.0);
        let (_, std_shift, _) = correlation_stats(&re, &r, 0.1);

        assert_relative_eq!(std_plain, 1.0, epsilon = 1e-12);
        // shift is inside the root
        assert_relative_eq!(std_shift, 1.1f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn test_embedding_log_radii_clamps_zero_denominator() {
        let weighted_sq = vec![6.0, 0.0];
        let kernel_sum = vec![3.0, 0.0];
        let mut re = vec![0.0; 2];

        embedding_log_radii(&weighted_sq, &kernel_sum, &mut re);

        assert_relative_eq!(re[0], (DENS_EPS + 2.0f64).ln(), epsilon = 1e-12);
        assert!(re[1].is_finite());
        assert_relative_eq!(re[1], DENS_EPS.ln(), epsilon = 1e-12);
    }

    #[test]
    fn test_is_active_covers_final_fraction_only() {
        let params = DensParams::<f64>::new(Some(2.0), Some(0.3), None);
        let state = DensState {
            params,
            r: vec![0.0; 2],
            mu_sum: vec![1.0; 2],
            mu_tot: 1.0,
        };

        // 100 epochs, frac 0.3 -> active from epoch index 70 onwards
        assert!(!state.is_active(69, 100));
        assert!(state.is_active(70, 100));
        assert!(state.is_active(99, 100));
    }

    #[test]
    fn test_is_active_off_when_lambda_zero() {
        let params = DensParams::<f64>::new(Some(0.0), Some(1.0), None);
        let state = DensState {
            params,
            r: vec![0.0; 2],
            mu_sum: vec![1.0; 2],
            mu_tot: 1.0,
        };

        assert!(!state.is_active(99, 100));
    }
}
