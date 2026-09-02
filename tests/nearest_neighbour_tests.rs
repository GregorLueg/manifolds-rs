//! The contract `run_ann_search` offers its callers.
//!
//! Two things worth pinning down, because both used to be wrong. Distances are
//! true distances in every metric, and the NNDescent backend can hand back the
//! graph it built rather than searching it.

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
fn extraction_and_query_agree_on_the_neighbours() {
    let (data, _) = commons::create_diagnostic_data(60, 10, 42);
    let k = 10;

    let truth = run_ann_search(
        data.as_ref(),
        k,
        "exhaustive".to_string(),
        &NearestNeighbourParams::default(),
        42,
        0,
    )
    .unwrap()
    .0;

    let search = |extract: bool| {
        let params = NearestNeighbourParams {
            extract_knn: extract,
            ..NearestNeighbourParams::<f64>::default()
        };
        run_ann_search(data.as_ref(), k, "nndescent".to_string(), &params, 42, 0).unwrap()
    };

    for extract in [false, true] {
        let (idx, dist) = search(extract);

        assert_eq!(idx.len(), data.nrows());
        assert!(
            idx.iter().all(|r| r.len() == k),
            "extract={extract}: rows must be full, extraction drops unfilled slots"
        );
        assert!(
            idx.iter().enumerate().all(|(i, r)| !r.contains(&i)),
            "extract={extract}: self must not appear; the two paths remove it differently"
        );
        assert!(
            dist.iter().all(|r| r.windows(2).all(|w| w[0] <= w[1])),
            "extract={extract}: rows must stay sorted"
        );
        assert!(
            recall(&idx, &truth, k) > 0.9,
            "extract={extract}: recall {} is too low",
            recall(&idx, &truth, k)
        );
    }
}

#[test]
fn extraction_is_reproducible() {
    // The CPU path is; the GPU one is not, which is why `extract_knn` defaults
    // to true here and false there.
    let (data, _) = commons::create_diagnostic_data(50, 10, 42);
    let params = NearestNeighbourParams {
        extract_knn: true,
        ..NearestNeighbourParams::<f64>::default()
    };

    let run =
        || run_ann_search(data.as_ref(), 10, "nndescent".to_string(), &params, 42, 0).unwrap();

    assert_eq!(run(), run());
}

#[test]
fn extraction_widens_the_graph_to_cover_k() {
    // The build degree defaults to 30 and extraction cannot return more than
    // the graph holds, so a larger `k` has to widen it rather than come back
    // short.
    let (data, _) = commons::create_diagnostic_data(60, 10, 42);
    let k = 50;
    let params = NearestNeighbourParams {
        extract_knn: true,
        ..NearestNeighbourParams::<f64>::default()
    };

    let (idx, _) =
        run_ann_search(data.as_ref(), k, "nndescent".to_string(), &params, 42, 0).unwrap();

    assert!(
        idx.iter().all(|r| r.len() == k),
        "rows came back short, so the build degree did not cover k"
    );
}

#[test]
fn extraction_is_the_default() {
    assert!(NearestNeighbourParams::<f64>::default().extract_knn);
}
