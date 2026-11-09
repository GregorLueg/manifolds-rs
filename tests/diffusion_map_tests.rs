#![allow(clippy::needless_range_loop)]

mod commons;
use commons::*;

use manifolds_rs::data::graph::phate_alpha_decay_affinities;
use manifolds_rs::data::structures::{coo_to_csr, CompressedSparseData};
use manifolds_rs::prelude::*;
use manifolds_rs::utils::diffusions::parse_phate_time;
use manifolds_rs::utils::diffusions::{
    apply_anisotropic_normalisation, build_symmetric_diffusion_operator, DiffusionMapsLandmarks,
};
use manifolds_rs::utils::math::compute_largest_eigenpairs_lanczos;
use manifolds_rs::*;

fn row_sums(op: &CompressedSparseData<f64>) -> Vec<f64> {
    let (nrows, _) = op.shape();
    (0..nrows)
        .map(|i| {
            let start = op.indptr[i];
            let end = op.indptr[i + 1];
            op.data[start..end].iter().sum()
        })
        .collect()
}

fn separation_ratio(embedding: &[Vec<f64>], labels: &[usize], n_clusters: usize) -> f64 {
    let mut sums = vec![(0.0f64, 0.0f64, 0usize); n_clusters];
    for (i, &label) in labels.iter().enumerate() {
        sums[label].0 += embedding[0][i];
        sums[label].1 += embedding[1][i];
        sums[label].2 += 1;
    }
    let centroids: Vec<(f64, f64)> = sums
        .iter()
        .map(|&(sx, sy, c)| (sx / c as f64, sy / c as f64))
        .collect();

    let mut min_inter = f64::INFINITY;
    for i in 0..n_clusters {
        for j in (i + 1)..n_clusters {
            let d = ((centroids[i].0 - centroids[j].0).powi(2)
                + (centroids[i].1 - centroids[j].1).powi(2))
            .sqrt();
            min_inter = min_inter.min(d);
        }
    }

    let mut avg_intra = 0.0;
    for (label, &(cx, cy)) in centroids.iter().enumerate() {
        let pts: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == label)
            .map(|(i, _)| i)
            .collect();
        let intra: f64 = pts
            .iter()
            .map(|&i| ((embedding[0][i] - cx).powi(2) + (embedding[1][i] - cy).powi(2)).sqrt())
            .sum::<f64>()
            / pts.len() as f64;
        avg_intra += intra;
    }
    avg_intra /= n_clusters as f64;

    min_inter / avg_intra
}

/// Test 1: kNN sanity
#[test]
fn dm_integration_01_knn_correctness() {
    let (data, labels) = create_diagnostic_data(100, 10, 42);
    let k = 10;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "kmknn".to_string(), &nn_params, 42, 0).unwrap();

    println!("\n=== DM DIAGNOSTIC 1: kNN Search ===");

    let mut intra = 0.0;
    for (i, ns) in knn_indices.iter().enumerate() {
        let same = ns.iter().filter(|&&j| labels[j] == labels[i]).count();
        intra += same as f64 / ns.len() as f64;
    }
    intra /= knn_indices.len() as f64;
    println!("Intra-cluster ratio: {:.1}%", intra * 100.0);
    assert!(intra > 0.8);

    let min_d = knn_dist
        .iter()
        .flatten()
        .copied()
        .fold(f64::INFINITY, f64::min);
    assert!(min_d >= 0.0);
}

/// Test 2: Gaussian kernel (alpha-decay with decay=2) is symmetric
#[test]
fn dm_integration_02_gaussian_affinities() {
    let (data, _) = create_diagnostic_data(50, 10, 42);
    let k = 10;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "kmknn".to_string(), &nn_params, 42, 0).unwrap();

    println!("\n=== DM DIAGNOSTIC 2: Gaussian Affinities ===");

    let graph =
        phate_alpha_decay_affinities(&knn_indices, &knn_dist, k, Some(2.0), 1.0, 1e-4, "add");

    for (&i, &j) in graph.row_indices.iter().zip(&graph.col_indices) {
        assert_ne!(i, j);
    }
    for &v in &graph.values {
        assert!(v > 0.0 && v <= 1.0 + 1e-10);
    }

    let edges: std::collections::HashMap<(usize, usize), f64> = graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
        .map(|((&i, &j), &v)| ((i, j), v))
        .collect();
    let mut max_asym = 0.0f64;
    for (&(i, j), &v_ij) in &edges {
        if let Some(&v_ji) = edges.get(&(j, i)) {
            max_asym = max_asym.max((v_ij - v_ji).abs());
        }
    }
    assert!(max_asym < 1e-10);
    println!("Gaussian kernel symmetric, {} edges", graph.values.len());
}

/// Test 3: alpha_norm = 0 is identity
#[test]
fn dm_integration_03_alpha_norm_zero() {
    let (data, _) = create_diagnostic_data(40, 10, 42);
    let k = 10;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "kmknn".to_string(), &nn_params, 42, 0).unwrap();

    let graph =
        phate_alpha_decay_affinities(&knn_indices, &knn_dist, k, Some(2.0), 1.0, 1e-4, "add");
    let kernel = coo_to_csr(&graph);
    let result = apply_anisotropic_normalisation(&kernel, 0.0).unwrap();

    println!("\n=== DM DIAGNOSTIC 3: Alpha-Norm Zero is Identity ===");
    assert_eq!(result.data, kernel.data);
    assert_eq!(result.indices, kernel.indices);
    assert_eq!(result.indptr, kernel.indptr);
}

/// Test 4: alpha_norm = 1 preserves symmetry
#[test]
fn dm_integration_04_alpha_norm_one_symmetry() {
    let (data, _) = create_diagnostic_data(40, 10, 42);
    let k = 10;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "kmknn".to_string(), &nn_params, 42, 0).unwrap();

    let graph =
        phate_alpha_decay_affinities(&knn_indices, &knn_dist, k, Some(2.0), 1.0, 1e-4, "add");
    let kernel = coo_to_csr(&graph);
    let result = apply_anisotropic_normalisation(&kernel, 1.0).unwrap();

    println!("\n=== DM DIAGNOSTIC 4: Alpha-Norm One Preserves Symmetry ===");

    for &v in &result.data {
        assert!(v.is_finite());
        assert!(v >= 0.0);
    }

    let dense = result.to_dense();
    let n = dense.nrows();
    let mut max_asym = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            max_asym = max_asym.max((dense[(i, j)] - dense[(j, i)]).abs());
        }
    }
    assert!(max_asym < 1e-10);
}

/// Test 5: Symmetric diffusion operator is symmetric
#[test]
fn dm_integration_05_symmetric_operator() {
    let (data, _) = create_diagnostic_data(40, 10, 42);
    let k = 10;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "kmknn".to_string(), &nn_params, 42, 0).unwrap();

    let graph =
        phate_alpha_decay_affinities(&knn_indices, &knn_dist, k, Some(2.0), 1.0, 1e-4, "add");
    let kernel = coo_to_csr(&graph);
    let kernel_norm = apply_anisotropic_normalisation(&kernel, 1.0).unwrap();
    let (p_sym, sqrt_d) = build_symmetric_diffusion_operator(&kernel_norm).unwrap();

    println!("\n=== DM DIAGNOSTIC 5: Symmetric Operator ===");

    for &v in &p_sym.data {
        assert!(v.is_finite());
    }
    for &s in &sqrt_d {
        assert!(s > 0.0);
    }

    let dense = p_sym.to_dense();
    let n = dense.nrows();
    let mut max_asym = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            max_asym = max_asym.max((dense[(i, j)] - dense[(j, i)]).abs());
        }
    }
    assert!(max_asym < 1e-10);

    let sums = row_sums(&p_sym);
    for s in sums {
        assert!(s.is_finite());
    }
}

/// Test 6: Largest eigenvalue of P_sym is ~1
#[test]
fn dm_integration_06_trivial_eigenvalue() {
    let (data, _) = create_diagnostic_data(40, 10, 42);
    let k = 10;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "kmknn".to_string(), &nn_params, 42, 0).unwrap();

    let graph =
        phate_alpha_decay_affinities(&knn_indices, &knn_dist, k, Some(2.0), 1.0, 1e-4, "add");
    let kernel = coo_to_csr(&graph);
    let kernel_norm = apply_anisotropic_normalisation(&kernel, 1.0).unwrap();
    let (p_sym, _) = build_symmetric_diffusion_operator(&kernel_norm).unwrap();

    let (evals, _) = compute_largest_eigenpairs_lanczos(&p_sym, 3, 42).unwrap();

    println!("\n=== DM DIAGNOSTIC 6: Trivial Eigenvalue ===");
    println!("Top 3 eigenvalues: {:?}", evals);

    let mut sorted = evals.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert!((sorted[0] - 1.0).abs() < 1e-3);
    assert!(sorted[1] < sorted[0] + 1e-6);
    assert!(sorted[2] < sorted[1] + 1e-6);
}

/// Test 7: Full DM produces well-separated clusters
#[test]
fn dm_integration_07_full_dm_quality() {
    let (data, labels) = create_diagnostic_data(40, 10, 123);
    let n_clusters = 5;

    println!("\n=== DM DIAGNOSTIC 7: Full DM Quality ===");

    let params = DiffusionMapsParams {
        k: 10,
        ..DiffusionMapsParams::default()
    };

    let embedding = diffusion_maps(data.as_ref(), None, params, 42, 0).unwrap();

    let all: Vec<f64> = embedding.iter().flat_map(|d| d.iter().copied()).collect();
    assert_eq!(all.iter().filter(|v| v.is_nan()).count(), 0);
    assert_eq!(all.iter().filter(|v| v.is_infinite()).count(), 0);

    let range = all.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - all.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(range > 1e-6);
    assert!(range < 1e6);

    let sep = separation_ratio(&embedding, &labels, n_clusters);
    println!("Cluster separation ratio: {:.2}", sep);
    assert!(sep > 1.0);
}

/// Test 8: Full DM reproducibility
#[test]
fn dm_integration_08_reproducibility() {
    let (data, _) = create_diagnostic_data(40, 10, 42);

    let params = DiffusionMapsParams {
        k: 10,
        t: parse_phate_time(None, 50),
        ..DiffusionMapsParams::default()
    };

    let e1 = diffusion_maps(data.as_ref(), None, params.clone(), 42, 0).unwrap();
    let e2 = diffusion_maps(data.as_ref(), None, params, 42, 0).unwrap();

    let mut max_diff = 0.0f64;
    for i in 0..e1[0].len() {
        for d in 0..2 {
            max_diff = max_diff.max((e1[d][i] - e2[d][i]).abs());
        }
    }
    println!("\n=== DM DIAGNOSTIC 8: Reproducibility ===");
    println!("Max diff: {:.2e}", max_diff);
    assert!(max_diff < 1e-5);
}

/// Test 9: Different seeds
#[test]
fn dm_integration_09_different_seeds() {
    let (data, _) = create_diagnostic_data(40, 10, 42);

    let params = DiffusionMapsParams {
        k: 10,
        t: parse_phate_time(None, 50),
        ..DiffusionMapsParams::default()
    };

    let e1 = diffusion_maps(data.as_ref(), None, params.clone(), 42, 0).unwrap();
    let e2 = diffusion_maps(data.as_ref(), None, params, 123, 0).unwrap();

    let mut max_diff = 0.0f64;
    for i in 0..e1[0].len() {
        for d in 0..2 {
            max_diff = max_diff.max((e1[d][i] - e2[d][i]).abs());
        }
    }
    println!("\n=== DM DIAGNOSTIC 9: Different Seeds ===");
    println!("Max diff: {:.4}", max_diff);
    assert!(max_diff.is_finite());
}

/// Test 10: Precomputed kNN produces identical embedding
#[test]
fn dm_integration_10_precomputed_knn() {
    let (data, _) = create_diagnostic_data(40, 10, 42);
    let k = 10;

    let nn_params = NearestNeighbourParams::default();
    let (ki, kd) = run_ann_search(
        data.as_ref(),
        k,
        DiffusionMapsParams::<f64>::default().ann_type,
        &nn_params,
        42,
        0,
    )
    .unwrap();

    let params = DiffusionMapsParams {
        k,
        t: parse_phate_time(None, 50),
        ..DiffusionMapsParams::default()
    };

    let e_pre = diffusion_maps(data.as_ref(), Some((ki, kd)), params.clone(), 42, 0).unwrap();
    let e_int = diffusion_maps(data.as_ref(), None, params, 42, 0).unwrap();

    let mut max_diff = 0.0f64;
    for i in 0..e_pre[0].len() {
        for d in 0..2 {
            max_diff = max_diff.max((e_pre[d][i] - e_int[d][i]).abs());
        }
    }
    println!("\n=== DM DIAGNOSTIC 10: Precomputed kNN ===");
    println!("Max diff: {:.2e}", max_diff);
    assert!(max_diff < 1e-5);
}

/// Test 11: Fixed t vs auto t
#[test]
fn dm_integration_11_fixed_vs_auto_t() {
    let (data, labels) = create_diagnostic_data(100, 10, 123);
    let n_clusters = 5;

    let params_auto = DiffusionMapsParams {
        k: 15,
        t: parse_phate_time(None, 50),
        ..DiffusionMapsParams::default()
    };
    let params_fixed = DiffusionMapsParams {
        k: 15,
        t: parse_phate_time(Some(5), 100),
        ..DiffusionMapsParams::default()
    };

    let e_auto = diffusion_maps(data.as_ref(), None, params_auto, 42, 0).unwrap();
    let e_fixed = diffusion_maps(data.as_ref(), None, params_fixed, 42, 0).unwrap();

    println!("\n=== DM DIAGNOSTIC 11: Fixed vs Auto t ===");
    for (embd, label) in [(&e_auto, "auto"), (&e_fixed, "fixed")] {
        let sep = separation_ratio(embd, &labels, n_clusters);
        println!("{} t: separation = {:.2}", label, sep);
        assert!(sep > 1.0);
    }
}

/// Test 12: alpha_norm changes the embedding
#[test]
fn dm_integration_12_alpha_norm_effect() {
    let (data, _) = create_diagnostic_data(40, 10, 42);

    let params_a0 = DiffusionMapsParams {
        k: 10,
        alpha_norm: 0.0,
        t: parse_phate_time(None, 50),
        ..DiffusionMapsParams::default()
    };
    let params_a1 = DiffusionMapsParams {
        k: 10,
        t: parse_phate_time(None, 50),
        ..DiffusionMapsParams::default()
    };

    let e0 = diffusion_maps(data.as_ref(), None, params_a0, 42, 0).unwrap();
    let e1 = diffusion_maps(data.as_ref(), None, params_a1, 42, 0).unwrap();

    let mut max_diff = 0.0f64;
    for i in 0..e0[0].len() {
        for d in 0..2 {
            max_diff = max_diff.max((e0[d][i] - e1[d][i]).abs());
        }
    }
    println!("\n=== DM DIAGNOSTIC 12: Alpha-Norm Effect ===");
    println!("Max diff a=0 vs a=1: {:.4}", max_diff);
    assert!(max_diff > 1e-3);
}

/// Test 13: Bandwidth scale effect
#[test]
fn dm_integration_13_bandwidth_scale_effect() {
    let (data, _) = create_diagnostic_data(50, 10, 42);
    let k = 10;

    let nn_params = NearestNeighbourParams::default();
    let (ki, kd) =
        run_ann_search(data.as_ref(), k, "kmknn".to_string(), &nn_params, 42, 0).unwrap();

    let g_narrow = phate_alpha_decay_affinities(&ki, &kd, k, Some(2.0), 0.5, 1e-4, "none");
    let g_wide = phate_alpha_decay_affinities(&ki, &kd, k, Some(2.0), 2.0, 1e-4, "none");

    let mean_n = g_narrow.values.iter().sum::<f64>() / g_narrow.values.len() as f64;
    let mean_w = g_wide.values.iter().sum::<f64>() / g_wide.values.len() as f64;

    println!("\n=== DM DIAGNOSTIC 13: Bandwidth Scale ===");
    println!("Mean affinity narrow={:.4}, wide={:.4}", mean_n, mean_w);
    assert!(mean_w > mean_n);
}

/// Test 14: Landmark DM (random selection)
#[test]
fn dm_integration_14_landmark_quality_random() {
    let (data, labels) = create_diagnostic_data(100, 10, 123);
    let n_clusters = 5;

    println!("\n=== DM DIAGNOSTIC 14: Landmark DM (random) ===");

    let params = DiffusionMapsParams {
        k: 10,
        n_landmarks: Some(30),
        landmark_method: "random".to_string(),
        ..DiffusionMapsParams::default()
    };

    let embd = diffusion_maps(data.as_ref(), None, params, 42, 0).unwrap();
    let all: Vec<f64> = embd.iter().flat_map(|d| d.iter().copied()).collect();

    assert_eq!(all.iter().filter(|v| v.is_nan()).count(), 0);
    assert_eq!(all.iter().filter(|v| v.is_infinite()).count(), 0);

    let range = all.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - all.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(range > 1e-6);

    let sep = separation_ratio(&embd, &labels, n_clusters);
    println!("Separation ratio: {:.2}", sep);
    assert!(sep > 1.0);
}

/// Test 15: Landmark DM (spectral selection)
#[test]
fn dm_integration_15_landmark_quality_spectral() {
    let (data, labels) = create_diagnostic_data(60, 10, 123);
    let n_clusters = 5;

    println!("\n=== DM DIAGNOSTIC 15: Landmark DM (spectral) ===");

    let params = DiffusionMapsParams {
        k: 10,
        n_landmarks: Some(30),
        n_svd: Some(10),
        ..DiffusionMapsParams::default()
    };

    let embd = diffusion_maps(data.as_ref(), None, params, 42, 0).unwrap();
    let all: Vec<f64> = embd.iter().flat_map(|d| d.iter().copied()).collect();
    assert_eq!(all.iter().filter(|v| v.is_nan()).count(), 0);

    let sep = separation_ratio(&embd, &labels, n_clusters);
    println!("Separation ratio: {:.2}", sep);
    assert!(sep > 1.0);
}

/// Test 16: Landmark DM (density selection)
#[test]
fn dm_integration_16_landmark_quality_density() {
    let (data, labels) = create_diagnostic_data(60, 10, 123);
    let n_clusters = 5;

    println!("\n=== DM DIAGNOSTIC 16: Landmark DM (density) ===");

    let params = DiffusionMapsParams {
        k: 10,
        n_landmarks: Some(30),
        landmark_method: "density".to_string(),
        ..DiffusionMapsParams::default()
    };

    let embd = diffusion_maps(data.as_ref(), None, params, 42, 0).unwrap();
    let all: Vec<f64> = embd.iter().flat_map(|d| d.iter().copied()).collect();
    assert_eq!(all.iter().filter(|v| v.is_nan()).count(), 0);

    let sep = separation_ratio(&embd, &labels, n_clusters);
    println!("Separation ratio: {:.2}", sep);
    assert!(sep > 0.9);
}

/// Test 17: n_landmarks >= n falls back to full DM
#[test]
fn dm_integration_17_landmark_fallback_to_full() {
    let (data, _) = create_diagnostic_data(30, 10, 42);

    let params_full = DiffusionMapsParams {
        k: 10,
        t: parse_phate_time(None, 50),
        ..DiffusionMapsParams::default()
    };
    let params_fallback = DiffusionMapsParams {
        k: 10,
        t: parse_phate_time(None, 50),
        n_landmarks: Some(500),
        ..DiffusionMapsParams::default()
    };

    let e_full = diffusion_maps(data.as_ref(), None, params_full, 42, 0).unwrap();
    let e_fb = diffusion_maps(data.as_ref(), None, params_fallback, 42, 0).unwrap();

    let mut max_diff = 0.0f64;
    for i in 0..e_full[0].len() {
        for d in 0..2 {
            max_diff = max_diff.max((e_full[d][i] - e_fb[d][i]).abs());
        }
    }
    println!("\n=== DM DIAGNOSTIC 17: Landmark Fallback ===");
    println!("Max diff: {:.2e}", max_diff);
    assert!(max_diff < 1e-6);
}

/// Test 18: Landmark DM reproducibility
#[test]
fn dm_integration_18_landmark_reproducibility() {
    let (data, _) = create_diagnostic_data(50, 10, 42);

    let params = DiffusionMapsParams {
        k: 10,
        t: parse_phate_time(None, 50),
        n_landmarks: Some(25),
        landmark_method: "random".to_string(),
        ..DiffusionMapsParams::default()
    };

    let e1 = diffusion_maps(data.as_ref(), None, params.clone(), 42, 0).unwrap();
    let e2 = diffusion_maps(data.as_ref(), None, params, 42, 0).unwrap();

    let mut max_diff = 0.0f64;
    for i in 0..e1[0].len() {
        for d in 0..2 {
            max_diff = max_diff.max((e1[d][i] - e2[d][i]).abs());
        }
    }
    println!("\n=== DM DIAGNOSTIC 18: Landmark Reproducibility ===");
    println!("Max diff: {:.2e}", max_diff);
    assert!(max_diff < 1e-4);
}

/// Test 19: Landmark coverage
#[test]
fn dm_integration_19_landmark_coverage() {
    let (data, _) = create_diagnostic_data(50, 10, 42);
    let k = 10;

    let nn_params = NearestNeighbourParams::default();
    let (ki, kd) =
        run_ann_search(data.as_ref(), k, "kmknn".to_string(), &nn_params, 42, 0).unwrap();

    let graph = phate_alpha_decay_affinities(&ki, &kd, k, Some(2.0), 1.0, 1e-4, "add");
    let kernel = coo_to_csr(&graph);

    let landmarks = DiffusionMapsLandmarks::build(
        data.as_ref(),
        Some(&kernel),
        20,
        "random",
        "euclidean",
        1.0,
        k,
        1.0,
        1e-4,
        "add",
        42,
        None,
        0,
    )
    .unwrap();

    assert_eq!(landmarks.get_n_landmarks(), 20);
    assert_eq!(landmarks.get_landmark_indices().len(), 20);

    println!("\n=== DM DIAGNOSTIC 19: Landmark Coverage ===");
    println!("{} landmarks selected", landmarks.get_n_landmarks());
}

/// Clustered data plus one far outlier, whose neighbours never point back at it
/// and all sit at about the same distance.
fn data_with_outlier() -> faer::Mat<f64> {
    let (data, _) = create_diagnostic_data(20, 10, 42);
    let n = data.nrows();
    faer::Mat::from_fn(
        n + 1,
        data.ncols(),
        |i, j| {
            if i == n {
                1e3
            } else {
                data[(i, j)]
            }
        },
    )
}

/// Parameters for the zero-degree tests.
fn outlier_params(symmetry: &str, thresh: f64) -> DiffusionMapsParams<f64> {
    DiffusionMapsParams {
        k: 5,
        ann_type: "exhaustive".to_string(),
        graph_symmetry: symmetry.to_string(),
        thresh,
        ..DiffusionMapsParams::default()
    }
}

/// Test 20: mutual-edge symmetrisation isolates the outlier and is refused
#[test]
fn dm_integration_20_mutual_symmetry_isolated_point_errors() {
    let data = data_with_outlier();
    for symmetry in ["mnn", "multiply"] {
        let res = diffusion_maps(data.as_ref(), None, outlier_params(symmetry, 1e-4), 42, 0);
        assert!(
            matches!(
                res,
                Err(ManifoldsError::IsolatedPoints {
                    n_isolated: 1..,
                    n_total: 101
                })
            ),
            "{symmetry}: {res:?}"
        );
    }
}

/// Test 21: additive symmetry keeps the outlier connected, so the run stays
/// valid and finite
#[test]
fn dm_integration_21_additive_symmetry_keeps_outlier() {
    let data = data_with_outlier();
    let embd = diffusion_maps(data.as_ref(), None, outlier_params("add", 1e-4), 42, 0).unwrap();
    assert!(embd.iter().flatten().all(|v| v.is_finite()));
}

/// Test 22: a threshold above the affinity of the outlier's neighbours isolates
/// it under additive symmetry too
#[test]
fn dm_integration_22_high_thresh_isolated_point_errors() {
    let data = data_with_outlier();
    let res = diffusion_maps(data.as_ref(), None, outlier_params("add", 0.6), 42, 0);
    assert!(
        matches!(res, Err(ManifoldsError::IsolatedPoints { .. })),
        "{res:?}"
    );
}

/// Test 23: the landmark operator refuses an isolated landmark as well. All
/// points are landmarks bar one, and the seed keeps the outlier among them.
#[test]
fn dm_integration_23_landmark_isolated_point_errors() {
    let data = data_with_outlier();
    let params = DiffusionMapsParams {
        n_landmarks: Some(data.nrows() - 1),
        landmark_method: "random".to_string(),
        ..outlier_params("mnn", 1e-4)
    };
    let res = diffusion_maps(data.as_ref(), None, params, 42, 0);
    assert!(
        matches!(res, Err(ManifoldsError::IsolatedPoints { .. })),
        "{res:?}"
    );
}

/// Test 24: `t_max` below the knee search minimum errors on both operators,
/// the minimum itself runs
#[test]
fn dm_integration_24_small_t_max() {
    let (data, _) = create_diagnostic_data(20, 10, 42);
    for n_landmarks in [None, Some(30)] {
        let params = |t_max| DiffusionMapsParams {
            k: 10,
            t: PhateTime::Auto { t_max },
            n_landmarks,
            landmark_method: "random".to_string(),
            ..DiffusionMapsParams::default()
        };
        for t_max in 0..3 {
            let res = diffusion_maps::<f64>(data.as_ref(), None, params(t_max), 42, 0);
            assert!(
                matches!(res, Err(ManifoldsError::TMaxTooSmall { t_max: t, min: 3 }) if t == t_max),
                "t_max = {t_max}, n_landmarks = {n_landmarks:?}: {res:?}"
            );
        }
        diffusion_maps::<f64>(data.as_ref(), None, params(3), 42, 0).unwrap();
    }
}

/// Test 25: PHATE shares the knee search and returns the same error
#[test]
fn dm_integration_25_phate_small_t_max() {
    let (data, _) = create_diagnostic_data(20, 10, 42);
    let params = PhateParams::<f64> {
        diffusion_params: PhateDiffusionParams {
            t: PhateTime::Auto { t_max: 2 },
            ..PhateDiffusionParams::default()
        },
        ..PhateParams::default()
    };
    let res = phate(data.as_ref(), None, params, 42, 0);
    assert!(
        matches!(res, Err(ManifoldsError::TMaxTooSmall { t_max: 2, min: 3 })),
        "{res:?}"
    );
}
