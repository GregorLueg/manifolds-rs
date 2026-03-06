//! Pair construction for PaCMAP.
//!
//! Constructs the three pair types used in PaCMAP optimisation:
//! - Near pairs: k nearest neighbours
//! - Mid-near pairs: sampled from a wider neighbourhood (candidates ~4-50)
//! - Further pairs: random distant points

use num_traits::Float;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;

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
    /// Mid-near pairs sampled from a wider neighbourhood
    pub mid_near: Vec<(usize, usize)>,
    /// Further pairs sampled randomly
    pub further: Vec<(usize, usize)>,
}

/// Sample mid-near pairs for a single point.
///
/// Slices the candidate window `knn_indices[candidate_start..candidate_end]`
/// and takes the first `n_mid_near` entries. Function here assumes that the
/// kNN indices are sorted. Should the kNN list is shorter than
/// `candidate_start`, falls back to sampling randomly from whatever neighbours
/// are available.
///
/// ### Params
///
/// * `i` - Index of the source point.
/// * `neighbours` - kNN indices for point `i`, sorted by ascending distance,
///   excluding self.
/// * `n_mid_near` - Number of mid-near pairs to return.
/// * `candidate_start` - Start of the candidate window into `neighbours`.
///   Skips the nearest neighbours; paper/official default is 4.
/// * `candidate_end` - End of the candidate window into `neighbours`.
///   Paper default is 50; requires kNN to have been run with k >= this value.
/// * `rng` - RNG used only in the fallback path.
///
/// ### Returns
///
/// Up to `n_mid_near` pairs `(i, j)` drawn from the candidate window.
fn sample_mid_near_pairs(
    i: usize,
    neighbours: &[usize],
    n_mid_near: usize,
    candidate_start: usize,
    candidate_end: usize,
    rng: &mut SmallRng,
) -> Vec<(usize, usize)> {
    let available_end = candidate_end.min(neighbours.len());

    if candidate_start >= available_end {
        // not enough neighbours — fall back to sampling from whatever is
        // available
        let fallback_end = neighbours.len().min(candidate_end);
        if fallback_end == 0 {
            return vec![];
        }
        return (0..n_mid_near.min(fallback_end))
            .map(|_| {
                let idx = rng.random_range(0..fallback_end);
                (i, neighbours[idx])
            })
            .collect();
    }

    let candidates = &neighbours[candidate_start..available_end];

    // official paper samples 6 candidates and picks the 2 closest.
    candidates
        .iter()
        .take(n_mid_near)
        .map(|&j| (i, j))
        .collect()
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

/// Construct all three PaCMAP pair types from a precomputed kNN graph.
///
/// ### Params
///
/// * `knn_indices` - kNN indices excluding self, shape [n_samples][k].
///   Should be computed with k large enough to cover mid-near candidates
///   (recommend k >= 50).
/// * `n_mid_near` - Number of mid-near pairs per point. Defaults to 2 in
///   the paper.
/// * `n_further` - Number of further (random) pairs per point. Defaults to
///   2 in the paper.
/// * `mn_candidate_start` - Start index into kNN list for mid-near sampling.
///   Defaults to 4 (skip the 4 nearest).
/// * `mn_candidate_end` - End index into kNN list for mid-near sampling.
///   Defaults to 50.
/// * `n` - Total number of points (for random further pair sampling).
/// * `seed` - Seed for reproducibility.
///
/// ### Returns
///
/// `PacmapPairs` with all three pair lists populated.
pub fn construct_pacmap_pairs<T>(
    knn_indices: &[Vec<usize>],
    n_mid_near: usize,
    n_further: usize,
    mn_candidate_start: usize,
    mn_candidate_end: usize,
    seed: u64,
) -> PacmapPairs
where
    T: Float + Send + Sync,
{
    let n = knn_indices.len();

    // near pairs: all k neighbours for each point
    let near: Vec<(usize, usize)> = knn_indices
        .par_iter()
        .enumerate()
        .flat_map(|(i, neighbours)| neighbours.iter().map(move |&j| (i, j)).collect::<Vec<_>>())
        .collect();

    // mid-near pairs: sample n_mid_near from candidates mn_candidate_start to
    // mn_candidate_end in the knn list.
    let mid_near: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed + i as u64);
            sample_mid_near_pairs(
                i,
                &knn_indices[i],
                n_mid_near,
                mn_candidate_start,
                mn_candidate_end,
                &mut rng,
            )
        })
        .collect();

    // further pairs: uniformly random points, not neighbours
    let further: Vec<(usize, usize)> = (0..n)
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

///////////
// Tests //
///////////

#[cfg(test)]
mod test_pacmap_pairs {
    use super::*;

    fn dummy_knn(n: usize, k: usize) -> Vec<Vec<usize>> {
        (0..n)
            .map(|i| (0..n).filter(|&j| j != i).cycle().take(k).collect())
            .collect()
    }

    #[test]
    fn test_near_pairs_count() {
        let knn = dummy_knn(10, 50);
        let pairs = construct_pacmap_pairs::<f64>(&knn, 2, 2, 4, 50, 42);
        assert_eq!(pairs.near.len(), 10 * 50);
    }

    #[test]
    fn test_mid_near_pairs_count() {
        let knn = dummy_knn(10, 50);
        let pairs = construct_pacmap_pairs::<f64>(&knn, 2, 2, 4, 50, 42);
        assert_eq!(pairs.mid_near.len(), 10 * 2);
    }

    #[test]
    fn test_further_pairs_count() {
        let knn = dummy_knn(10, 50);
        let pairs = construct_pacmap_pairs::<f64>(&knn, 2, 2, 4, 50, 42);
        assert_eq!(pairs.further.len(), 10 * 2);
    }

    #[test]
    fn test_no_self_pairs_near() {
        let knn = dummy_knn(20, 50);
        let pairs = construct_pacmap_pairs::<f64>(&knn, 2, 2, 4, 50, 42);
        for (i, j) in &pairs.near {
            assert_ne!(i, j, "self-pair found in near pairs at index {}", i);
        }
    }

    #[test]
    fn test_no_self_pairs_mid_near() {
        let knn = dummy_knn(20, 50);
        let pairs = construct_pacmap_pairs::<f64>(&knn, 2, 2, 4, 50, 42);
        for (i, j) in &pairs.mid_near {
            assert_ne!(i, j, "self-pair found in mid-near pairs at index {}", i);
        }
    }

    #[test]
    fn test_no_self_pairs_further() {
        let knn = dummy_knn(20, 50);
        let pairs = construct_pacmap_pairs::<f64>(&knn, 2, 2, 4, 50, 42);
        for (i, j) in &pairs.further {
            assert_ne!(i, j, "self-pair found in further pairs at index {}", i);
        }
    }

    #[test]
    fn test_near_pairs_source_indices_valid() {
        let n = 15;
        let knn = dummy_knn(n, 50);
        let pairs = construct_pacmap_pairs::<f64>(&knn, 2, 2, 4, 50, 42);
        for &(i, j) in &pairs.near {
            assert!(i < n);
            assert!(j < n);
        }
    }

    #[test]
    fn test_further_pairs_indices_in_bounds() {
        let n = 15;
        let knn = dummy_knn(n, 50);
        let pairs = construct_pacmap_pairs::<f64>(&knn, 2, 2, 4, 50, 42);
        for &(i, j) in &pairs.further {
            assert!(i < n);
            assert!(j < n);
        }
    }

    #[test]
    fn test_mid_near_drawn_from_candidate_window() {
        let n = 10;
        let k = 50;
        let candidate_start = 4;
        let candidate_end = 50;
        let knn = dummy_knn(n, k);
        let pairs = construct_pacmap_pairs::<f64>(&knn, 2, 2, candidate_start, candidate_end, 42);

        for &(i, j) in &pairs.mid_near {
            let window = &knn[i][candidate_start..candidate_end.min(knn[i].len())];
            assert!(
                window.contains(&j),
                "mid-near pair ({}, {}) not from candidate window",
                i,
                j
            );
        }
    }

    #[test]
    fn test_reproducibility() {
        let knn = dummy_knn(20, 50);
        let pairs_a = construct_pacmap_pairs::<f64>(&knn, 2, 2, 4, 50, 99);
        let pairs_b = construct_pacmap_pairs::<f64>(&knn, 2, 2, 4, 50, 99);
        assert_eq!(pairs_a.near, pairs_b.near);
        assert_eq!(pairs_a.mid_near, pairs_b.mid_near);
        assert_eq!(pairs_a.further, pairs_b.further);
    }

    #[test]
    fn test_different_seeds_differ() {
        let knn = dummy_knn(20, 50);
        let pairs_a = construct_pacmap_pairs::<f64>(&knn, 2, 2, 4, 50, 1);
        let pairs_b = construct_pacmap_pairs::<f64>(&knn, 2, 2, 4, 50, 2);
        // further pairs are random so must differ; near pairs are deterministic
        assert_ne!(pairs_a.further, pairs_b.further);
    }

    #[test]
    fn test_fallback_when_k_smaller_than_candidate_start() {
        // k=3 is smaller than candidate_start=4 — should not panic, returns
        // something reasonable via the fallback path
        let knn: Vec<Vec<usize>> = (0..10)
            .map(|i| (0..3).map(|o| (i + o + 1) % 10).collect())
            .collect();
        let pairs = construct_pacmap_pairs::<f64>(&knn, 2, 2, 4, 50, 42);
        for &(i, j) in &pairs.mid_near {
            assert_ne!(i, j);
            assert!(j < 10);
        }
    }

    #[test]
    fn test_empty_knn_does_not_panic() {
        let knn: Vec<Vec<usize>> = vec![vec![]; 5];
        let pairs = construct_pacmap_pairs::<f64>(&knn, 2, 2, 4, 50, 42);
        assert!(pairs.near.is_empty());
        assert!(pairs.mid_near.is_empty());
    }
}
