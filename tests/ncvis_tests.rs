mod commons;
use commons::*;

use manifolds_rs::prelude::*;
use manifolds_rs::*;

/// Smallest distance between cluster centroids over the largest mean
/// within-cluster distance to the centroid.
fn separation_ratio(embd: &[Vec<f64>], labels: &[usize]) -> f64 {
    let n_clusters = labels.iter().max().unwrap() + 1;
    let mut centres = vec![(0.0, 0.0, 0usize); n_clusters];
    for (i, &l) in labels.iter().enumerate() {
        centres[l].0 += embd[0][i];
        centres[l].1 += embd[1][i];
        centres[l].2 += 1;
    }
    let centres: Vec<(f64, f64)> = centres
        .iter()
        .map(|&(x, y, c)| (x / c as f64, y / c as f64))
        .collect();

    let mut spread = vec![0.0; n_clusters];
    let mut counts = vec![0usize; n_clusters];
    for (i, &l) in labels.iter().enumerate() {
        spread[l] += (embd[0][i] - centres[l].0).hypot(embd[1][i] - centres[l].1);
        counts[l] += 1;
    }
    let max_spread = spread
        .iter()
        .zip(&counts)
        .map(|(s, &c)| s / c as f64)
        .fold(0.0, f64::max);

    let mut min_between = f64::MAX;
    for a in 0..n_clusters {
        for b in (a + 1)..n_clusters {
            let d = (centres[a].0 - centres[b].0).hypot(centres[a].1 - centres[b].1);
            min_between = min_between.min(d);
        }
    }
    min_between / max_spread
}

/// Run on a one-thread Rayon pool, where NCVis is deterministic.
fn single_thread<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(f)
}

#[test]
fn ncvis_integration_01_cluster_separation() {
    let (data, labels) = create_diagnostic_data(200, 20, 42);
    let params = NcvisParams::<f64>::default();
    let embd = ncvis(data.as_ref(), None, &params, 42, 0).unwrap();
    assert_eq!(embd.len(), 2);
    assert_eq!(embd[0].len(), 1_000);
    assert!(embd.iter().flatten().all(|v| v.is_finite()));
    let ratio = separation_ratio(&embd, &labels);
    assert!(ratio > 2.0, "clusters not separated, ratio = {ratio:.3}");
}

#[test]
fn ncvis_integration_02_reproducible_on_one_thread() {
    let (data, _) = create_diagnostic_data(100, 10, 7);
    let params = NcvisParams::<f64>::default();
    let a = single_thread(|| ncvis(data.as_ref(), None, &params, 3, 0).unwrap());
    let b = single_thread(|| ncvis(data.as_ref(), None, &params, 3, 0).unwrap());
    assert_eq!(a, b);
}

#[test]
fn ncvis_integration_03_different_seeds_differ() {
    let (data, _) = create_diagnostic_data(100, 10, 7);
    let params = NcvisParams::<f64> {
        initialisation: "random".to_string(),
        ..NcvisParams::default()
    };
    let a = single_thread(|| ncvis(data.as_ref(), None, &params, 1, 0).unwrap());
    let b = single_thread(|| ncvis(data.as_ref(), None, &params, 2, 0).unwrap());
    assert_ne!(a, b);
}

#[test]
fn ncvis_integration_04_precomputed_knn_matches_internal() {
    let (data, _) = create_diagnostic_data(100, 10, 11);
    let params = NcvisParams::<f64>::default();
    let knn = run_ann_search(
        data.as_ref(),
        params.k,
        params.ann_type.clone(),
        &params.nn_params,
        5,
        0,
    )
    .unwrap();
    let internal = single_thread(|| ncvis(data.as_ref(), None, &params, 5, 0).unwrap());
    let pre = single_thread(|| ncvis(data.as_ref(), Some(knn), &params, 5, 0).unwrap());
    assert_eq!(internal, pre);
}

#[test]
fn ncvis_integration_05_f32_and_three_dims() {
    let (data, _) = create_diagnostic_data(50, 10, 3);
    let data = mat_to_f32(data);
    let params = NcvisParams::<f32> {
        n_dim: 3,
        ..NcvisParams::default()
    };
    let embd = ncvis(data.as_ref(), None, &params, 42, 0).unwrap();
    assert_eq!(embd.len(), 3);
    assert_eq!(embd[0].len(), 250);
    assert!(embd.iter().flatten().all(|v| v.is_finite()));
}

#[test]
fn ncvis_integration_06_cauchy_kernel() {
    let (data, labels) = create_diagnostic_data(100, 10, 13);
    let params = NcvisParams::<f64> {
        optim_params: NcvisOptimParams::new(50, 10.0, 1.0, 1.0, 2.0, 1.0),
        ..NcvisParams::default()
    };
    let embd = ncvis(data.as_ref(), None, &params, 42, 0).unwrap();
    assert!(embd.iter().flatten().all(|v| v.is_finite()));
    assert!(separation_ratio(&embd, &labels) > 2.0);
}

#[test]
fn ncvis_integration_07_invalid_params() {
    let (data, _) = create_diagnostic_data(20, 5, 1);
    let check = |params: NcvisParams<f64>| {
        assert!(matches!(
            ncvis(data.as_ref(), None, &params, 0, 0),
            Err(ManifoldsError::NcvisInvalidParam { .. })
        ));
    };
    let base = NcvisParams::<f64>::default;
    check(NcvisParams {
        initialisation: "spectral".to_string(),
        ..base()
    });
    check(NcvisParams {
        init_scale: 0.0,
        ..base()
    });
    let mut p = base();
    p.optim_params.n_negative = 0.5;
    check(p);
    let mut p = base();
    p.optim_params.n_epochs = 0;
    check(p);

    let tiny = faer::Mat::<f64>::from_fn(2, 3, |i, j| (i + j) as f64);
    assert!(matches!(
        ncvis(tiny.as_ref(), None, &base(), 0, 0),
        Err(ManifoldsError::NcvisInvalidParam { .. })
    ));
}
