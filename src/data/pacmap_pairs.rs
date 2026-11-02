//! Pair construction for PaCMAP and dummy graph generation for PaCMAP.
//!
//! Constructs the three pair types used in PaCMAP optimisation:
//! - Near pairs: k nearest neighbours
//! - Mid-near pairs: second closest of six random points
//! - Further pairs: random distant points

use ann_search_rs::utils::dist::Dist;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;

use crate::data::structures::CoordinateList;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Random points drawn per mid-near pair, of which the second closest is kept.
/// Set by the PaCMAP paper.
const MN_CANDIDATES: usize = 6;

/////////////
// Helpers //
/////////////

/// The three pair types used in PaCMAP optimisation.
///
/// All pairs are stored as flat `(i, j)` index tuples with no weights —
/// unlike UMAP, tSNE or PHATE there is no graph symmetrisation or edge
/// weighting.
pub struct PacmapPairs {
    /// Near pairs from kNN (i, neighbour_of_i)
    pub near: Vec<(usize, usize)>,
    /// Mid-near pairs: second closest of six random points
    pub mid_near: Vec<(usize, usize)>,
    /// Further pairs sampled randomly
    pub further: Vec<(usize, usize)>,
}

/// Distance between two rows, used to rank mid-near candidates.
///
/// ### Params
///
/// * `a` - First row.
/// * `b` - Second row.
/// * `metric` - Distance metric. Squared Euclidean is fine here, since only the
///   ordering matters.
///
/// ### Returns
///
/// The distance between the rows.
fn candidate_dist<T>(a: &[T], b: &[T], metric: &Dist) -> T
where
    T: ManifoldsFloat,
{
    match metric {
        Dist::SquaredEuclidean => T::euclidean_simd(a, b),
        Dist::Cosine => {
            let denom = T::calculate_l2_norm(a) * T::calculate_l2_norm(b);
            if denom > T::zero() {
                T::one() - T::dot_simd(a, b) / denom
            } else {
                T::one()
            }
        }
        Dist::Manhattan => a
            .iter()
            .zip(b)
            .fold(T::zero(), |acc, (&x, &y)| acc + (x - y).abs()),
    }
}

/// Sample mid-near pairs for a single point.
///
/// For each pair, draws `MN_CANDIDATES` random points and keeps the second
/// closest. That lands well outside the kNN list, which is the point: these
/// pairs carry the global structure, so they cannot be neighbours. Candidates
/// exclude `i` and the points already picked for `i`, so the result may be
/// shorter than `n_mid_near` for very small `n`.
///
/// ### Params
///
/// * `i` - Index of the source point.
/// * `data` - Row-major data, `n * dim` long.
/// * `dim` - Number of features.
/// * `n_mid_near` - Number of mid-near pairs to return.
/// * `metric` - Distance metric used to rank the candidates.
/// * `rng` - RNG for drawing the candidates.
///
/// ### Returns
///
/// Up to `n_mid_near` pairs `(i, j)`.
///
/// ### References
///
/// Wang et al., JMLR, 2021
fn sample_mid_near_pairs<T>(
    i: usize,
    data: &[T],
    dim: usize,
    n_mid_near: usize,
    metric: &Dist,
    rng: &mut SmallRng,
) -> Vec<(usize, usize)>
where
    T: ManifoldsFloat,
{
    let n = data.len() / dim;
    let row = |idx: usize| &data[idx * dim..(idx + 1) * dim];
    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(n_mid_near);

    for _ in 0..n_mid_near {
        let mut candidates = [0usize; MN_CANDIDATES];
        let mut found = 0;
        let mut attempts = 0;
        while found < MN_CANDIDATES && attempts < MN_CANDIDATES * 10 {
            attempts += 1;
            let j = rng.random_range(0..n);
            if j != i && !candidates[..found].contains(&j) && !pairs.iter().any(|p| p.1 == j) {
                candidates[found] = j;
                found += 1;
            }
        }
        if found == 0 {
            break;
        }

        let mut ranked: Vec<(T, usize)> = candidates[..found]
            .iter()
            .map(|&j| (candidate_dist(row(i), row(j), metric), j))
            .collect();
        ranked.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        pairs.push((i, ranked[1.min(found - 1)].1));
    }

    pairs
}

/// Sample further (random) pairs for a single point.
///
/// Draws `n_further` indices uniformly at random from `[0, n)`, excluding
/// `i` itself. Makes up to `n_further * 10` attempts before giving up, so
/// the returned vec may be shorter than `n_further` for very small `n`.
///
/// ### Params
///
/// * `i` - Index of the source point; excluded from sampling.
/// * `n` - Total number of points in the dataset.
/// * `n_further` - Number of further pairs to return.
/// * `rng` - RNG for random index generation.
///
/// ### Returns
///
/// Up to `n_further` pairs `(i, j)` where `j` is drawn uniformly from
/// `[0, n) \ {i}`.
fn sample_further_pairs(
    i: usize,
    n: usize,
    n_further: usize,
    rng: &mut SmallRng,
) -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(n_further);
    let mut attempts = 0;
    while pairs.len() < n_further && attempts < n_further * 10 {
        let j = rng.random_range(0..n);
        if j != i {
            pairs.push((i, j));
        }
        attempts += 1;
    }
    pairs
}

//////////
// Main //
//////////

/// Construct all three PaCMAP pair types.
///
/// ### Params
///
/// * `data` - Row-major data, `n_samples * dim` long.
/// * `dim` - Number of features.
/// * `knn_indices` - kNN indices excluding self, shape `[n_samples][k]`, with
///   `k >= n_near`.
/// * `n_near` - Number of near pairs per point. Uses the first `n_near`
///   entries from each row of `knn_indices`. Defaults to 10 in the paper.
/// * `n_mid_near` - Number of mid-near pairs per point. Defaults to 5 in
///   the paper (MN_ratio=0.5 * n_neighbors=10).
/// * `n_further` - Number of further (random) pairs per point. Defaults to
///   20 in the paper (FP_ratio=2.0 * n_neighbors=10).
/// * `metric` - Distance metric used to rank the mid-near candidates.
/// * `seed` - Seed for reproducibility.
///
/// ### Returns
///
/// `PacmapPairs` with all three pair lists populated.
#[allow(clippy::too_many_arguments)]
pub fn construct_pacmap_pairs<T>(
    data: &[T],
    dim: usize,
    knn_indices: &[Vec<usize>],
    n_near: usize,
    n_mid_near: usize,
    n_further: usize,
    metric: &Dist,
    seed: u64,
) -> PacmapPairs
where
    T: ManifoldsFloat,
{
    let n = knn_indices.len();

    let near: Vec<(usize, usize)> = knn_indices
        .par_iter()
        .enumerate()
        .flat_map(|(i, neighbours)| {
            neighbours
                .iter()
                .take(n_near)
                .map(move |&j| (i, j))
                .collect::<Vec<_>>()
        })
        .collect();

    let mid_near: Vec<(usize, usize)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed + i as u64);
            sample_mid_near_pairs(i, data, dim, n_mid_near, metric, &mut rng)
        })
        .collect();

    let further: Vec<(usize, usize)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed + n as u64 + i as u64);
            sample_further_pairs(i, n, n_further, &mut rng)
        })
        .collect();

    PacmapPairs {
        near,
        mid_near,
        further,
    }
}

/// Construct a coordinate-list (COO) sparse graph from kNN indices with
/// uniform unit weights.
///
/// ### Params
///
/// * `knn_indices` - kNN indices excluding self, shape `[n_samples][k]`.
///
/// ### Returns
///
/// `CoordinateList<T>` with uniform weights of `1.0` for all edges.
pub fn knn_to_coo_unweighted<T>(knn_indices: &[Vec<usize>]) -> CoordinateList<T>
where
    T: ManifoldsFloat,
{
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for (i, neighbours) in knn_indices.iter().enumerate() {
        for &j in neighbours {
            row_indices.push(i);
            col_indices.push(j);
            values.push(T::one());
        }
    }

    CoordinateList {
        row_indices,
        col_indices,
        values,
        n_samples: knn_indices.len(),
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_pacmap_pairs {
    use super::*;

    fn pairs_for(knn: &[Vec<usize>], near: usize, mn: usize, fp: usize, seed: u64) -> PacmapPairs {
        let data: Vec<f64> = (0..knn.len() * 2).map(|x| (x * x % 17) as f64).collect();
        construct_pacmap_pairs(&data, 2, knn, near, mn, fp, &Dist::SquaredEuclidean, seed)
    }

    fn dummy_knn(n: usize, k: usize) -> Vec<Vec<usize>> {
        (0..n)
            .map(|i| (0..n).filter(|&j| j != i).cycle().take(k).collect())
            .collect()
    }

    #[test]
    fn test_near_pairs_count() {
        let knn = dummy_knn(10, 50);
        let pairs = pairs_for(&knn, 10, 5, 20, 42);
        assert_eq!(pairs.near.len(), 10 * 10);
    }

    #[test]
    fn test_mid_near_pairs_count() {
        let knn = dummy_knn(10, 50);
        let pairs = pairs_for(&knn, 10, 5, 20, 42);
        assert_eq!(pairs.mid_near.len(), 10 * 5);
    }

    #[test]
    fn test_further_pairs_count() {
        let knn = dummy_knn(10, 50);
        let pairs = pairs_for(&knn, 10, 5, 20, 42);
        assert_eq!(pairs.further.len(), 10 * 20);
    }

    #[test]
    fn test_no_self_pairs_near() {
        let knn = dummy_knn(20, 50);
        let pairs = pairs_for(&knn, 10, 5, 20, 42);
        for (i, j) in &pairs.near {
            assert_ne!(i, j, "self-pair found in near pairs at index {}", i);
        }
    }

    #[test]
    fn test_no_self_pairs_mid_near() {
        let knn = dummy_knn(20, 50);
        let pairs = pairs_for(&knn, 10, 5, 20, 42);
        for (i, j) in &pairs.mid_near {
            assert_ne!(i, j, "self-pair found in mid-near pairs at index {}", i);
        }
    }

    #[test]
    fn test_no_self_pairs_further() {
        let knn = dummy_knn(20, 50);
        let pairs = pairs_for(&knn, 10, 5, 20, 42);
        for (i, j) in &pairs.further {
            assert_ne!(i, j, "self-pair found in further pairs at index {}", i);
        }
    }

    #[test]
    fn test_near_pairs_source_indices_valid() {
        let n = 15;
        let knn = dummy_knn(n, 50);
        let pairs = pairs_for(&knn, 10, 5, 20, 42);
        for &(i, j) in &pairs.near {
            assert!(i < n);
            assert!(j < n);
        }
    }

    #[test]
    fn test_further_pairs_indices_in_bounds() {
        let n = 15;
        let knn = dummy_knn(n, 50);
        let pairs = pairs_for(&knn, 10, 5, 20, 42);
        for &(i, j) in &pairs.further {
            assert!(i < n);
            assert!(j < n);
        }
    }

    #[test]
    fn test_near_pairs_drawn_from_first_n_near() {
        let n = 10;
        let k = 50;
        let n_near = 10;
        let knn = dummy_knn(n, k);
        let pairs = pairs_for(&knn, n_near, 5, 20, 42);
        for &(i, j) in &pairs.near {
            assert!(
                knn[i][..n_near].contains(&j),
                "near pair ({}, {}) not from first {} neighbours",
                i,
                j,
                n_near
            );
        }
    }

    #[test]
    fn test_mid_near_distinct_within_a_point() {
        let knn = dummy_knn(50, 10);
        let pairs = pairs_for(&knn, 10, 5, 20, 42);
        for chunk in pairs.mid_near.chunks_exact(5) {
            let mut js: Vec<usize> = chunk.iter().map(|p| p.1).collect();
            js.sort_unstable();
            js.dedup();
            assert_eq!(js.len(), 5);
        }
    }

    #[test]
    fn test_mid_near_is_never_the_closest_candidate() {
        // Row 1 sits next to row 0 and everything else is far away, so row 1 is
        // the closest candidate whenever it is drawn, and must never be kept.
        let n = 20;
        let mut data = vec![0.0f64; n * 2];
        data[2] = 0.001;
        for i in 2..n {
            data[i * 2] = 100.0 + i as f64;
        }
        let knn = dummy_knn(n, 10);
        let pairs = construct_pacmap_pairs(&data, 2, &knn, 10, 5, 20, &Dist::SquaredEuclidean, 42);
        assert!(pairs.mid_near.iter().filter(|p| p.0 == 0).all(|p| p.1 != 1));
    }

    #[test]
    fn test_reproducibility() {
        let knn = dummy_knn(20, 50);
        let pairs_a = pairs_for(&knn, 10, 5, 20, 99);
        let pairs_b = pairs_for(&knn, 10, 5, 20, 99);
        assert_eq!(pairs_a.near, pairs_b.near);
        assert_eq!(pairs_a.mid_near, pairs_b.mid_near);
        assert_eq!(pairs_a.further, pairs_b.further);
    }

    #[test]
    fn test_different_seeds_differ() {
        let knn = dummy_knn(20, 50);
        let pairs_a = pairs_for(&knn, 10, 5, 20, 1);
        let pairs_b = pairs_for(&knn, 10, 5, 20, 2);
        assert_ne!(pairs_a.further, pairs_b.further);
    }

    #[test]
    fn test_empty_knn_does_not_panic() {
        let knn: Vec<Vec<usize>> = vec![vec![]; 5];
        let pairs = pairs_for(&knn, 10, 5, 20, 42);
        assert!(pairs.near.is_empty());
    }
}
