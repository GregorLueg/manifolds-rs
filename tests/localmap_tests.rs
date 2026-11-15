mod commons;
use commons::*;

use manifolds_rs::prelude::*;
use manifolds_rs::training::pacmap_optimiser::optimise_localmap_parallel;
use manifolds_rs::*;

/////////////
// Helpers //
/////////////

fn max_abs_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| x.iter().zip(y).map(|(&p, &q)| (p - q).abs()))
        .fold(0.0_f64, f64::max)
}

/// Mean distance to the nearest point of another cluster, over mean distance to
/// the own cluster's centroid.
fn gap_ratio(embd: &[Vec<f64>], labels: &[usize]) -> f64 {
    let n = labels.len();
    let n_clusters = labels.iter().max().unwrap() + 1;
    let mut centres = vec![(0.0, 0.0, 0usize); n_clusters];
    for i in 0..n {
        let c = &mut centres[labels[i]];
        c.0 += embd[0][i];
        c.1 += embd[1][i];
        c.2 += 1;
    }
    let intra: f64 = (0..n)
        .map(|i| {
            let c = centres[labels[i]];
            let (cx, cy) = (c.0 / c.2 as f64, c.1 / c.2 as f64);
            ((embd[0][i] - cx).powi(2) + (embd[1][i] - cy).powi(2)).sqrt()
        })
        .sum::<f64>()
        / n as f64;
    let inter: f64 = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| labels[j] != labels[i])
                .map(|j| {
                    ((embd[0][i] - embd[0][j]).powi(2) + (embd[1][i] - embd[1][j]).powi(2)).sqrt()
                })
                .fold(f64::INFINITY, f64::min)
        })
        .sum::<f64>()
        / n as f64;
    inter / intra
}

///////////
// Tests //
///////////

#[test]
fn localmap_integration_01_shape_and_finite() {
    let (data, _) = create_diagnostic_data(50, 10, 42);
    let embd = localmap(data.as_ref(), None, &LocalMapParams::default(), 42, 0).unwrap();

    assert_eq!(embd.len(), 2);
    assert_eq!(embd[0].len(), data.nrows());
    assert!(embd.iter().flatten().all(|x| x.is_finite()));
}

#[test]
fn localmap_integration_02_reproducible() {
    let (data, _) = create_diagnostic_data(50, 10, 42);
    let params = LocalMapParams::default();
    let a = localmap(data.as_ref(), None, &params, 42, 0).unwrap();
    let b = localmap(data.as_ref(), None, &params, 42, 0).unwrap();
    assert_eq!(a, b);
}

#[test]
fn localmap_integration_03_thread_count_independent() {
    // the PCA initialisation upstream already varies in the last bits with the
    // thread count, so fix the pairs and the start and test the optimiser
    let n = 300;
    let pairs = PacmapPairs {
        near: (0..n)
            .flat_map(|i| (1..=5).map(move |o| (i, (i + o) % n)))
            .collect(),
        mid_near: (0..n).map(|i| (i, (i * 7 + 3) % n)).collect(),
        further: (0..n)
            .flat_map(|i| (1..=4).map(move |o| (i, (i + o * 61) % n)))
            .collect(),
    };
    let start: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            vec![
                (i as f64 * 0.37).sin() * 0.01,
                (i as f64 * 0.11).cos() * 0.01,
            ]
        })
        .collect();
    let params = PacmapOptimParams::default();
    let run = |threads: usize| {
        let mut embd = start.clone();
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| optimise_localmap_parallel(&mut embd, &pairs, &params, 10.0, 42, 0))
            .unwrap();
        embd
    };
    assert_eq!(run(1), run(4));
}

#[test]
fn localmap_integration_04_sequential_parallel_agree() {
    let (data, _) = create_diagnostic_data(50, 10, 42);
    let seq = LocalMapParams {
        optimiser: "adam".to_string(),
        ..LocalMapParams::default()
    };
    let par = LocalMapParams::default();
    let a = localmap(data.as_ref(), None, &seq, 42, 0).unwrap();
    let b = localmap(data.as_ref(), None, &par, 42, 0).unwrap();
    let diff = max_abs_diff(&a, &b);
    assert!(diff < 1e-2, "sequential and parallel diverged: {diff}");
}

#[test]
fn localmap_integration_05_differs_from_pacmap() {
    let (data, _) = create_diagnostic_data(50, 10, 42);
    let a = localmap(data.as_ref(), None, &LocalMapParams::default(), 42, 0).unwrap();
    let b = pacmap(data.as_ref(), None, &PacmapParams::default(), 42, 0).unwrap();
    assert!(max_abs_diff(&a, &b) > 0.01);
}

#[test]
fn localmap_integration_06_phases_one_two_match_pacmap() {
    // stopping before phase 3 leaves nothing LocalMAP-specific
    let (data, _) = create_diagnostic_data(50, 10, 42);
    let optim = PacmapOptimParams::new(Some(200), None, None, None, None, None, None);
    let lm = LocalMapParams {
        optim_params: optim.clone(),
        ..LocalMapParams::default()
    };
    let pm = PacmapParams {
        optim_params: optim,
        ..PacmapParams::default()
    };
    let a = localmap(data.as_ref(), None, &lm, 42, 0).unwrap();
    let b = pacmap(data.as_ref(), None, &pm, 42, 0).unwrap();
    assert_eq!(a, b);
}

#[test]
fn localmap_integration_07_clusters_separate() {
    let (data, labels) = create_diagnostic_data(50, 10, 42);
    let embd = localmap(data.as_ref(), None, &LocalMapParams::default(), 42, 0).unwrap();
    let ratio = gap_ratio(&embd, &labels);
    assert!(ratio > 1.0, "clusters touch: gap ratio {ratio:.2}");
}

#[test]
fn localmap_integration_08_rejects_bad_max_distance() {
    let (data, _) = create_diagnostic_data(50, 10, 42);
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let params = LocalMapParams {
            max_distance: bad,
            ..LocalMapParams::default()
        };
        assert!(matches!(
            localmap(data.as_ref(), None, &params, 42, 0),
            Err(ManifoldsError::LocalMapInvalidMaxDistance { .. })
        ));
    }
}
