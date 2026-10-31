//! The contract `run_ann_search` offers its callers.
//!
//! Two things worth pinning down. Distances are true distances in every
//! metric, and the NNDescent backend hands back the graph it built, full rows
//! and no self edge.

mod commons;

use std::collections::HashSet;

use manifolds_rs::prelude::*;
use manifolds_rs::{tsne, TsneParams};

/////////////
// Helpers //
/////////////

/// Brute-force true Euclidean distances to the `k` nearest neighbours.
///
/// ### Params
///
/// * `data` - Samples by features.
/// * `k` - Neighbours per point, self excluded.
///
/// ### Returns
///
/// One sorted row per point.
fn brute_force_euclidean(data: &faer::Mat<f64>, k: usize) -> Vec<Vec<f64>> {
    let n = data.nrows();
    (0..n)
        .map(|i| {
            let mut d: Vec<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    (0..data.ncols())
                        .map(|c| (data[(i, c)] - data[(j, c)]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .collect();
            d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            d.truncate(k);
            d
        })
        .collect()
}

/// Recall of `found` against `truth`, as a fraction of all requested slots.
fn recall(found: &[Vec<usize>], truth: &[Vec<usize>], k: usize) -> f64 {
    let hits: usize = found
        .iter()
        .zip(truth)
        .map(|(a, b)| {
            let s: HashSet<_> = b.iter().collect();
            a.iter().filter(|j| s.contains(j)).count()
        })
        .sum();
    hits as f64 / (found.len() * k) as f64
}

///////////////
// Distances //
///////////////

#[test]
fn knn_returns_true_euclidean_not_squared() {
    let (data, _) = commons::create_diagnostic_data(40, 8, 42);
    let k = 5;

    let (_, dist) = run_ann_search(
        data.as_ref(),
        k,
        "exhaustive".to_string(),
        &NearestNeighbourParams::default(),
        42,
        0,
    )
    .unwrap();

    let expected = brute_force_euclidean(&data, k);
    for (got, want) in dist.iter().zip(&expected) {
        for (g, w) in got.iter().zip(want) {
            approx::assert_relative_eq!(g, w, epsilon = 1e-9);
        }
    }
}

#[test]
fn euclidean_and_l2_are_the_same_metric() {
    // `parse_ann_dist` maps both onto `SquaredEuclidean`, so the searches are
    // identical. Downstream used to decide whether to square again by comparing
    // the metric name against `"euclidean"`, which made `"l2"` a different
    // algorithm with the same neighbours.
    assert!(metric_returns_squared("euclidean"));
    assert!(metric_returns_squared("l2"));
    assert!(!metric_returns_squared("cosine"));
    assert!(!metric_returns_squared("manhattan"));

    let (data, _) = commons::create_diagnostic_data(40, 8, 42);

    let search = |metric: &str| {
        let params = NearestNeighbourParams {
            dist_metric: metric.to_string(),
            ..NearestNeighbourParams::<f64>::default()
        };
        run_ann_search(data.as_ref(), 5, "exhaustive".to_string(), &params, 42, 0).unwrap()
    };

    let (i_euclidean, d_euclidean) = search("euclidean");
    let (i_l2, d_l2) = search("l2");
    assert_eq!(i_euclidean, i_l2);
    assert_eq!(d_euclidean, d_l2);

    // and the same all the way through an embedding
    let embed = |metric: &str| {
        let params = TsneParams {
            perplexity: 5.0,
            nn_params: NearestNeighbourParams {
                dist_metric: metric.to_string(),
                ..NearestNeighbourParams::default()
            },
            optim_params: TsneOptimParams {
                n_epochs: 50,
                ..TsneOptimParams::default()
            },
            ..TsneParams::<f64>::default()
        };
        tsne(data.as_ref(), None, &params, "barnes_hut", 42, 0).unwrap()
    };
    assert_eq!(embed("euclidean"), embed("l2"));
}

#[test]
fn cosine_distances_are_left_alone() {
    // Cosine never came back squared, so nothing should have changed for it.
    // Bounded in [0, 2], which a squared value would not be for these inputs.
    let (data, _) = commons::create_diagnostic_data(40, 8, 42);
    let params = NearestNeighbourParams {
        dist_metric: "cosine".to_string(),
        ..NearestNeighbourParams::<f64>::default()
    };

    let (_, dist) =
        run_ann_search(data.as_ref(), 5, "exhaustive".to_string(), &params, 42, 0).unwrap();

    assert!(dist
        .iter()
        .flatten()
        .all(|&d| (0.0..=2.0).contains(&d) && d.is_finite()));
}

///////////////////
// NNDescent kNN //
///////////////////

#[test]
fn nndescent_graph_matches_exhaustive() {
    let (data, _) = commons::create_diagnostic_data(60, 10, 42);
    let k = 10;
    let params = NearestNeighbourParams::<f64>::default();

    let truth = run_ann_search(data.as_ref(), k, "exhaustive".to_string(), &params, 42, 0)
        .unwrap()
        .0;
    let (idx, dist) =
        run_ann_search(data.as_ref(), k, "nndescent".to_string(), &params, 42, 0).unwrap();

    assert_eq!(idx.len(), data.nrows());
    assert!(idx.iter().all(|r| r.len() == k), "rows must be full");
    assert!(
        idx.iter().enumerate().all(|(i, r)| !r.contains(&i)),
        "self must not appear"
    );
    assert!(
        dist.iter().all(|r| r.windows(2).all(|w| w[0] <= w[1])),
        "rows must stay sorted"
    );
    let r = recall(&idx, &truth, k);
    assert!(r > 0.9, "recall {r} is too low");
}

#[test]
fn nndescent_is_reproducible() {
    // The CPU graph is; the GPU one is not.
    let (data, _) = commons::create_diagnostic_data(50, 10, 42);
    let params = NearestNeighbourParams::<f64>::default();

    let run =
        || run_ann_search(data.as_ref(), 10, "nndescent".to_string(), &params, 42, 0).unwrap();

    assert_eq!(run(), run());
}

#[test]
fn nndescent_fills_large_k() {
    // tSNE asks for `3 * perplexity` neighbours, well past the crate's default
    // degree of 30, so the graph has to be built at the requested `k`.
    let (data, _) = commons::create_diagnostic_data(60, 10, 42);
    let k = 50;

    let (idx, _) = run_ann_search(
        data.as_ref(),
        k,
        "nndescent".to_string(),
        &NearestNeighbourParams::<f64>::default(),
        42,
        0,
    )
    .unwrap();

    assert!(idx.iter().all(|r| r.len() == k), "rows came back short");
}
