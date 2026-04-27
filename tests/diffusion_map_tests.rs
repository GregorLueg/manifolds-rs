#![allow(clippy::needless_range_loop)]

mod commons;
use commons::*;

use manifolds_rs::data::graph::phate_alpha_decay_affinities;
use manifolds_rs::data::structures::{coo_to_csr, CompressedSparseData};
use manifolds_rs::prelude::*;
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
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42, false);

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
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42, false);

    println!("\n=== DM DIAGNOSTIC 2: Gaussian Affinities ===");

    let graph = phate_alpha_decay_affinities(
        &knn_indices,
        &knn_dist,
        k,
        Some(2.0),
        1.0,
        1e-4,
        "add",
        true,
    );

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
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42, false);

    let graph = phate_alpha_decay_affinities(
        &knn_indices,
        &knn_dist,
        k,
        Some(2.0),
        1.0,
        1e-4,
        "add",
        true,
    );
    let kernel = coo_to_csr(&graph);
    let result = apply_anisotropic_normalisation(&kernel, 0.0);

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
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42, false);

    let graph = phate_alpha_decay_affinities(
        &knn_indices,
        &knn_dist,
        k,
        Some(2.0),
        1.0,
        1e-4,
        "add",
        true,
    );
    let kernel = coo_to_csr(&graph);
    let result = apply_anisotropic_normalisation(&kernel, 1.0);

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
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42, false);

    let graph = phate_alpha_decay_affinities(
        &knn_indices,
        &knn_dist,
        k,
        Some(2.0),
        1.0,
        1e-4,
        "add",
        true,
    );
    let kernel = coo_to_csr(&graph);
    let kernel_norm = apply_anisotropic_normalisation(&kernel, 1.0);
    let (p_sym, sqrt_d) = build_symmetric_diffusion_operator(&kernel_norm);

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
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42, false);

    let graph = phate_alpha_decay_affinities(
        &knn_indices,
        &knn_dist,
        k,
        Some(2.0),
        1.0,
        1e-4,
        "add",
        true,
    );
    let kernel = coo_to_csr(&graph);
    let kernel_norm = apply_anisotropic_normalisation(&kernel, 1.0);
    let (p_sym, _) = build_symmetric_diffusion_operator(&kernel_norm);

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

    let params = DiffusionMapsParams::new(
        Some(2),
        Some(10),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );

    let embedding = diffusion_maps(data.as_ref(), None, params, 42, true).unwrap();

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

    let params = DiffusionMapsParams::new(
        Some(2),
        Some(10),
        None,
        None,
        None,
        None,
        None,
        Some(50),
        None,
        None,
        None,
        None,
    );

    let e1 = diffusion_maps(data.as_ref(), None, params.clone(), 42, false).unwrap();
    let e2 = diffusion_maps(data.as_ref(), None, params, 42, false).unwrap();

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

    let params = DiffusionMapsParams::new(
        Some(2),
        Some(10),
        None,
        None,
        None,
        None,
        None,
        Some(50),
        None,
        None,
        None,
        None,
    );

    let e1 = diffusion_maps(data.as_ref(), None, params.clone(), 42, false).unwrap();
    let e2 = diffusion_maps(data.as_ref(), None, params, 123, false).unwrap();

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
    let (ki, kd) = run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42, false);

    let params = DiffusionMapsParams::new(
        Some(2),
        Some(k),
        None,
        None,
        None,
        None,
        None,
        Some(50),
        None,
        None,
        None,
        None,
    );

    let e_pre = diffusion_maps(data.as_ref(), Some((ki, kd)), params.clone(), 42, false).unwrap();
    let e_int = diffusion_maps(data.as_ref(), None, params, 42, false).unwrap();

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

    let params_auto = DiffusionMapsParams::new(
        Some(2),
        Some(15),
        None,
        None,
        None,
        None,
        None,
        Some(50),
        None,
        None,
        None,
        None,
    );
    let params_fixed = DiffusionMapsParams::new(
        Some(2),
        Some(15),
        None,
        None,
        None,
        None,
        None,
        None,
        Some(5),
        None,
        None,
        None,
    );

    let e_auto = diffusion_maps(data.as_ref(), None, params_auto, 42, false).unwrap();
    let e_fixed = diffusion_maps(data.as_ref(), None, params_fixed, 42, false).unwrap();

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

    let params_a0 = DiffusionMapsParams::new(
        Some(2),
        Some(10),
        None,
        None,
        None,
        None,
        Some(0.0),
        Some(50),
        None,
        None,
        None,
        None,
    );
    let params_a1 = DiffusionMapsParams::new(
        Some(2),
        Some(10),
        None,
        None,
        None,
        None,
        Some(1.0),
        Some(50),
        None,
        None,
        None,
        None,
    );

    let e0 = diffusion_maps(data.as_ref(), None, params_a0, 42, false).unwrap();
    let e1 = diffusion_maps(data.as_ref(), None, params_a1, 42, false).unwrap();

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
    let (ki, kd) = run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42, false);

    let g_narrow = phate_alpha_decay_affinities(&ki, &kd, k, Some(2.0), 0.5, 1e-4, "none", true);
    let g_wide = phate_alpha_decay_affinities(&ki, &kd, k, Some(2.0), 2.0, 1e-4, "none", true);

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

    let params = DiffusionMapsParams::new(
        Some(2),
        Some(10),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(30),
        Some("random".to_string()),
        None,
    );

    let embd = diffusion_maps(data.as_ref(), None, params, 42, true).unwrap();
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

    let params = DiffusionMapsParams::new(
        Some(2),
        Some(10),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(30),
        Some("spectral".to_string()),
        Some(10),
    );

    let embd = diffusion_maps(data.as_ref(), None, params, 42, false).unwrap();
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

    let params = DiffusionMapsParams::new(
        Some(2),
        Some(10),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(30),
        Some("density".to_string()),
        None,
    );

    let embd = diffusion_maps(data.as_ref(), None, params, 42, false).unwrap();
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

    let params_full = DiffusionMapsParams::new(
        Some(2),
        Some(10),
        None,
        None,
        None,
        None,
        None,
        Some(50),
        None,
        None,
        None,
        None,
    );
    let params_fallback = DiffusionMapsParams::new(
        Some(2),
        Some(10),
        None,
        None,
        None,
        None,
        None,
        Some(50),
        None,
        Some(500),
        None,
        None,
    );

    let e_full = diffusion_maps(data.as_ref(), None, params_full, 42, false).unwrap();
    let e_fb = diffusion_maps(data.as_ref(), None, params_fallback, 42, false).unwrap();

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

    let params = DiffusionMapsParams::new(
        Some(2),
        Some(10),
        None,
        None,
        None,
        None,
        None,
        Some(50),
        None,
        Some(25),
        Some("random".to_string()),
        None,
    );

    let e1 = diffusion_maps(data.as_ref(), None, params.clone(), 42, false).unwrap();
    let e2 = diffusion_maps(data.as_ref(), None, params, 42, false).unwrap();

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
    let (ki, kd) = run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42, false);

    let graph = phate_alpha_decay_affinities(&ki, &kd, k, Some(2.0), 1.0, 1e-4, "add", true);
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
        false,
    )
    .unwrap();

    assert_eq!(landmarks.get_n_landmarks(), 20);
    assert_eq!(landmarks.get_landmark_indices().len(), 20);

    println!("\n=== DM DIAGNOSTIC 19: Landmark Coverage ===");
    println!("{} landmarks selected", landmarks.get_n_landmarks());
}
