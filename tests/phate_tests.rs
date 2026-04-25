#![allow(clippy::needless_range_loop)]

mod commons;
use commons::*;

use manifolds_rs::data::graph::phate_alpha_decay_affinities;
use manifolds_rs::data::structures::{coo_to_csr, CompressedSparseData};
use manifolds_rs::prelude::*;
use manifolds_rs::utils::diffusions::build_diffusion_operator;
use manifolds_rs::utils::potentials::calculate_potential;
use manifolds_rs::utils::sparse_ops::matrix_power;
use manifolds_rs::*;
use rustc_hash::FxHashMap;

/// Helper: build edge map from COO graph for lookups
fn edges_from_coo(graph: &CoordinateList<f64>) -> FxHashMap<(usize, usize), f64> {
    graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
        .map(|((&i, &j), &v)| ((i, j), v))
        .collect()
}

/// Helper: compute row sums from a CSR matrix
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

/// Helper: centroid separation ratio for embeddings
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

/// Test 1: kNN search finds correct neighbours - PHATE
#[test]
fn phate_integration_01_knn_correctness() {
    let (data, labels) = create_diagnostic_data(100, 10, 42);
    let k = 10;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42, false);

    println!("\n=== PHATE DIAGNOSTIC 1: kNN Search ===");
    println!("Points per cluster: 100, k = {}", k);

    let mut self_in_neighbours = 0;
    for (i, neighbours) in knn_indices.iter().enumerate() {
        if neighbours.contains(&i) {
            self_in_neighbours += 1;
        }
    }
    if self_in_neighbours > 0 {
        println!(
            "WARNING: {} points have themselves in neighbours",
            self_in_neighbours
        );
    } else {
        println!("No self-loops in kNN");
    }

    let mut intra_ratio = 0.0;
    for (i, neighbours) in knn_indices.iter().enumerate() {
        let same = neighbours
            .iter()
            .filter(|&&j| labels[j] == labels[i])
            .count();
        intra_ratio += same as f64 / neighbours.len() as f64;
    }
    intra_ratio /= knn_indices.len() as f64;

    println!("Intra-cluster neighbour ratio: {:.1}%", intra_ratio * 100.0);
    assert!(
        intra_ratio > 0.8,
        "kNN should find mostly same-cluster neighbours, got {:.1}%",
        intra_ratio * 100.0
    );

    let all_dists: Vec<f64> = knn_dist.iter().flatten().copied().collect();
    let min_d = all_dists.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(min_d >= 0.0, "Distances must be non-negative");

    println!("Min distance: {:.6}", min_d);
}

/// Test 2: Alpha decay affinities - values, symmetry, no self-loops
#[test]
fn phate_integration_02_alpha_decay_affinities() {
    let (data, _labels) = create_diagnostic_data(50, 10, 42);
    let k = 10;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "kmknn".to_string(), &nn_params, 42, false);

    println!("\n=== PHATE DIAGNOSTIC 2: Alpha Decay Affinities ===");

    let graph = phate_alpha_decay_affinities(
        &knn_indices,
        &knn_dist,
        k,
        Some(40.0),
        1.0,
        1e-4,
        "add",
        true,
    );

    println!("Graph has {} edges", graph.values.len());

    // No self-loops
    for (&i, &j) in graph.row_indices.iter().zip(&graph.col_indices) {
        assert_ne!(i, j, "Self-loop found at {}", i);
    }
    println!("No self-loops");

    // All weights in (0, 1]
    for &v in &graph.values {
        assert!(v > 0.0 && v <= 1.0 + 1e-10, "Affinity out of range: {}", v);
    }
    println!("All affinities in (0, 1]");

    // Symmetric (additive symmetrisation)
    let edges = edges_from_coo(&graph);
    let mut max_asymmetry = 0.0f64;
    for (&(i, j), &v_ij) in &edges {
        if let Some(&v_ji) = edges.get(&(j, i)) {
            max_asymmetry = max_asymmetry.max((v_ij - v_ji).abs());
        }
    }
    println!("Max asymmetry: {:.2e}", max_asymmetry);
    assert!(
        max_asymmetry < 1e-10,
        "Graph should be symmetric, max asymmetry = {:.2e}",
        max_asymmetry
    );
    println!("Graph is symmetric");

    // Decay should produce values in (0, 1): farther points have lower affinity
    let min_w = graph.values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_w = graph
        .values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    println!("Affinity range: [{:.6}, {:.6}]", min_w, max_w);
    assert!(
        min_w > 0.0,
        "All affinities should be positive after thresholding"
    );
    assert!(max_w <= 1.0 + 1e-10, "Max affinity should not exceed 1.0");
}

/// Test 3: Binary kernel (decay = None) produces 0/1 affinities
#[test]
fn phate_integration_03_binary_kernel() {
    let (data, _labels) = create_diagnostic_data(50, 10, 42);
    let k = 10;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42, false);

    println!("\n=== PHATE DIAGNOSTIC 3: Binary Kernel ===");

    let graph = phate_alpha_decay_affinities(
        &knn_indices,
        &knn_dist,
        k,
        None, // binary kernel
        1.0,
        1e-10,
        "add",
        true,
    );

    // With binary kernel and very low threshold, all non-zero affinities
    // should be 1.0 before symmetrisation. After additive symmetrisation
    // (average), mutual edges stay at 1.0 and one-directional edges become 0.5.
    let distinct_values: std::collections::HashSet<u64> = graph
        .values
        .iter()
        .map(|&v| (v * 1e10).round() as u64)
        .collect();

    println!(
        "Binary graph has {} edges, {} distinct values",
        graph.values.len(),
        distinct_values.len()
    );

    // Should only have values 0.5 and 1.0 (after additive sym)
    for &v in &graph.values {
        assert!(
            (v - 0.5).abs() < 1e-6 || (v - 1.0).abs() < 1e-6,
            "Binary kernel should only produce 0.5 or 1.0 after symmetrisation, got {}",
            v
        );
    }
    println!("Binary kernel produces only 0.5 / 1.0 values after symmetrisation");
}

/// Test 4: Diffusion operator is row-stochastic
#[test]
fn phate_integration_04_diffusion_operator() {
    let (data, _labels) = create_diagnostic_data(50, 10, 42);
    let k = 10;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42, false);

    println!("\n=== PHATE DIAGNOSTIC 4: Diffusion Operator ===");

    let graph = phate_alpha_decay_affinities(
        &knn_indices,
        &knn_dist,
        k,
        Some(40.0),
        1.0,
        1e-4,
        "add",
        true,
    );

    let kernel_csr = coo_to_csr(&graph);
    let diffusion_op = build_diffusion_operator(&kernel_csr);

    let sums = row_sums(&diffusion_op);

    let min_sum = sums.iter().copied().fold(f64::INFINITY, f64::min);
    let max_sum = sums.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    println!("Row sums: min={:.8}, max={:.8}", min_sum, max_sum);

    assert!(
        (min_sum - 1.0).abs() < 1e-6,
        "Min row sum should be 1.0, got {}",
        min_sum
    );
    assert!(
        (max_sum - 1.0).abs() < 1e-6,
        "Max row sum should be 1.0, got {}",
        max_sum
    );
    println!("All rows sum to 1.0");

    // All values non-negative
    for &v in &diffusion_op.data {
        assert!(v >= 0.0, "Diffusion operator has negative value: {}", v);
    }
    println!("All values non-negative");
}

/// Test 5: Matrix power preserves row-stochastic property
#[test]
fn phate_integration_05_matrix_power() {
    let (data, _labels) = create_diagnostic_data(50, 10, 42);
    let k = 10;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42, false);

    println!("\n=== PHATE DIAGNOSTIC 5: Matrix Power ===");

    let graph = phate_alpha_decay_affinities(
        &knn_indices,
        &knn_dist,
        k,
        Some(40.0),
        1.0,
        1e-4,
        "add",
        true,
    );
    let kernel_csr = coo_to_csr(&graph);
    let diffusion_op = build_diffusion_operator(&kernel_csr);

    for t in [1, 2, 5, 10] {
        let powered = matrix_power(&diffusion_op, t);
        let sums = row_sums(&powered);
        let min_sum = sums.iter().copied().fold(f64::INFINITY, f64::min);
        let max_sum = sums.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        println!("P^{}: row sums min={:.8}, max={:.8}", t, min_sum, max_sum);
        assert!(
            (min_sum - 1.0).abs() < 1e-4,
            "P^{} min row sum = {}, expected 1.0",
            t,
            min_sum
        );
        assert!(
            (max_sum - 1.0).abs() < 1e-4,
            "P^{} max row sum = {}, expected 1.0",
            t,
            max_sum
        );
    }
    println!("Row-stochastic property preserved across t = 1, 2, 5, 10");
}

/// Test 6: Potential calculations produce finite, well-formed output
#[test]
fn phate_integration_06_potential_calculation() {
    let (data, _labels) = create_diagnostic_data(50, 10, 42);
    let k = 10;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42, false);

    println!("\n=== PHATE DIAGNOSTIC 6: Potential Calculation ===");

    let graph = phate_alpha_decay_affinities(
        &knn_indices,
        &knn_dist,
        k,
        Some(40.0),
        1.0,
        1e-4,
        "add",
        true,
    );
    let kernel_csr = coo_to_csr(&graph);
    let diffusion_op = build_diffusion_operator(&kernel_csr);

    for (gamma, label) in [(1.0, "log"), (-1.0, "identity"), (0.5, "power")] {
        let potential = calculate_potential(&diffusion_op, 5, gamma);

        let nan_count = potential.data.iter().filter(|v| v.is_nan()).count();
        let inf_count = potential.data.iter().filter(|v| v.is_infinite()).count();

        println!(
            "gamma={} ({}): {} nnz, {} NaN, {} Inf",
            gamma,
            label,
            potential.data.len(),
            nan_count,
            inf_count
        );

        assert_eq!(nan_count, 0, "gamma={} produced NaN values", gamma);
        assert_eq!(inf_count, 0, "gamma={} produced Inf values", gamma);
        assert!(!potential.data.is_empty(), "Potential should not be empty");
    }
}

/// Test 7: Full PHATE produces well-separated clusters
#[test]
fn phate_integration_07_full_phate_quality() {
    let (data, labels) = create_diagnostic_data(40, 10, 123);
    let n_clusters = 5;

    println!("\n=== PHATE DIAGNOSTIC 7: Full PHATE Quality ===");

    let params = PhateParams::new(
        Some(2),
        // knn
        Some(10),
        None,
        // diffusion
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
        None,
        None,
    );

    let embedding = phate(data.as_ref(), None, params, 42, true).unwrap();

    let all_coords: Vec<f64> = embedding.iter().flat_map(|d| d.iter().copied()).collect();
    let nan_count = all_coords.iter().filter(|v| v.is_nan()).count();
    let inf_count = all_coords.iter().filter(|v| v.is_infinite()).count();

    println!("NaN: {}, Inf: {}", nan_count, inf_count);
    assert_eq!(nan_count, 0, "Embedding has NaN values");
    assert_eq!(inf_count, 0, "Embedding has Inf values");

    let min_c = all_coords.iter().copied().fold(f64::INFINITY, f64::min);
    let max_c = all_coords.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max_c - min_c;

    println!(
        "Coordinate range: [{:.2}, {:.2}] (span: {:.2})",
        min_c, max_c, range
    );
    assert!(range < 10000.0, "Embedding exploded, range = {}", range);
    assert!(range > 0.01, "Embedding collapsed, range = {}", range);

    let sep = separation_ratio(&embedding, &labels, n_clusters);
    println!("Cluster separation ratio: {:.2}", sep);
    assert!(
        sep > 1.0,
        "Clusters should be separated, got ratio {:.2}",
        sep
    );
    println!("Clusters are well separated");
}

/// Test 8: Landmark PHATE produces reasonable output
#[test]
fn phate_integration_08_landmark_phate_quality() {
    let (data, labels) = create_diagnostic_data(60, 10, 123);
    let n_clusters = 5;

    println!("\n=== PHATE DIAGNOSTIC 8: Landmark PHATE Quality ===");

    let params = PhateParams::new(
        Some(2),
        // knn
        Some(10),
        None,
        // diffusion
        None,
        None,
        None,
        None,
        None,
        Some(20),
        None,
        None,
        None,
        None,
        None,
        None,
    );

    let embedding = phate(data.as_ref(), None, params, 42, true).unwrap();

    let all_coords: Vec<f64> = embedding.iter().flat_map(|d| d.iter().copied()).collect();
    let nan_count = all_coords.iter().filter(|v| v.is_nan()).count();
    let inf_count = all_coords.iter().filter(|v| v.is_infinite()).count();

    println!("NaN: {}, Inf: {}", nan_count, inf_count);
    assert_eq!(nan_count, 0, "Landmark embedding has NaN values");
    assert_eq!(inf_count, 0, "Landmark embedding has Inf values");

    let min_c = all_coords.iter().copied().fold(f64::INFINITY, f64::min);
    let max_c = all_coords.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max_c - min_c;

    println!("Coordinate range span: {:.2}", range);
    assert!(range > 0.01, "Landmark embedding collapsed");

    let sep = separation_ratio(&embedding, &labels, n_clusters);
    println!("Cluster separation ratio: {:.2}", sep);
    assert!(
        sep > 1.0,
        "Landmark PHATE clusters should be separated, got {:.2}",
        sep
    );
    println!("Landmark PHATE produces valid embedding");
}

/// Test 9: PHATE is reproducible with the same seed
#[test]
fn phate_integration_09_reproducibility() {
    let (data, _) = create_diagnostic_data(40, 10, 42);

    println!("\n=== PHATE DIAGNOSTIC 9: Reproducibility ===");

    let params = PhateParams::new(
        Some(2),  // n_dim
        Some(10), // k
        None,     // ann_type
        None,     // decay
        None,     // bandwidth_scale
        None,     // graph_symmetry
        Some(50), // t_max
        None,     // gamma
        None,     // n_landmarks
        None,     // landmark_method
        None,     // n_svd
        None,     // t_custom
        None,     // mds_method
        None,     // mds_iter
        None,     // randomised
    );

    let embd1 = phate(data.as_ref(), None, params.clone(), 42, false).unwrap();
    let embd2 = phate(data.as_ref(), None, params, 42, false).unwrap();

    let mut max_diff = 0.0f64;
    for i in 0..embd1[0].len() {
        for dim in 0..2 {
            max_diff = max_diff.max((embd1[dim][i] - embd2[dim][i]).abs());
        }
    }

    println!("Max coordinate difference: {:.2e}", max_diff);
    assert!(
        max_diff < 1e-6,
        "PHATE should be reproducible with same seed, got diff = {}",
        max_diff
    );
    println!("PHATE is reproducible");
}

/// Test 10: Different seeds produce different results
#[test]
fn phate_integration_10_different_seeds() {
    let (data, _) = create_diagnostic_data(40, 10, 42);

    println!("\n=== PHATE DIAGNOSTIC 10: Different Seeds ===");

    let params = PhateParams::new(
        Some(2),  // n_dim
        Some(10), // k
        None,     // ann_type
        None,     // decay
        None,     // bandwidth_scale
        None,     // graph_symmetry
        Some(50), // t_max
        None,     // gamma
        None,     // n_landmarks
        None,     // landmark_method
        None,     // n_svd
        None,     // t_custom
        None,     // mds_method
        None,     // mds_iter
        None,     // randomised
    );

    let embd1 = phate(data.as_ref(), None, params.clone(), 42, false).unwrap();
    let embd2 = phate(data.as_ref(), None, params, 123, false).unwrap();

    let mut max_diff = 0.0f64;
    for i in 0..embd1[0].len() {
        for dim in 0..2 {
            max_diff = max_diff.max((embd1[dim][i] - embd2[dim][i]).abs());
        }
    }

    println!("Max diff with different seeds: {:.4}", max_diff);
    assert!(
        max_diff > 0.01,
        "Different seeds should produce different embeddings"
    );
    println!("Different seeds produce different results");
}

/// Test 11: Precomputed kNN produces identical results
#[test]
fn phate_integration_11_precomputed_knn() {
    let (data, _labels) = create_diagnostic_data(40, 10, 42);
    let k = 10;

    println!("\n=== PHATE DIAGNOSTIC 11: Precomputed kNN ===");

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42, false);

    let params = PhateParams::new(
        Some(2),  // n_dim
        Some(k),  // k
        None,     // ann_type
        None,     // decay
        None,     // bandwidth_scale
        None,     // graph_symmetry
        Some(50), // t_max
        None,     // gamma
        None,     // n_landmarks
        None,     // landmark_method
        None,     // n_svd
        None,     // t_custom
        None,     // mds_method
        None,     // mds_iter
        None,     // randomised
    );

    let embd_precomputed = phate(
        data.as_ref(),
        Some((knn_indices, knn_dist)),
        params.clone(),
        42,
        false,
    )
    .unwrap();
    let embd_internal = phate(data.as_ref(), None, params, 42, false).unwrap();

    let mut max_diff = 0.0f64;
    for i in 0..embd_precomputed[0].len() {
        for dim in 0..2 {
            max_diff = max_diff.max((embd_precomputed[dim][i] - embd_internal[dim][i]).abs());
        }
    }

    println!("Max diff precomputed vs internal: {:.2e}", max_diff);
    assert!(
        max_diff < 1e-6,
        "Precomputed kNN should produce identical results, max diff = {}",
        max_diff
    );
    println!("Precomputed kNN produces identical results");
}

/// Test 12: Fixed t vs auto t both produce valid embeddings
#[test]
fn phate_integration_12_fixed_vs_auto_t() {
    let (data, labels) = create_diagnostic_data(40, 10, 42);
    let n_clusters = 5;

    println!("\n=== PHATE DIAGNOSTIC 12: Fixed t vs Auto t ===");

    let params_auto = PhateParams::new(
        Some(2),  // n_dim
        Some(10), // k
        None,     // ann_type
        None,     // decay
        None,     // bandwidth_scale
        None,     // graph_symmetry
        Some(50), // t_max
        None,     // gamma
        None,     // n_landmarks
        None,     // landmark_method
        None,     // n_svd
        None,     // t_custom — None means Auto
        None,     // mds_method
        None,     // mds_iter
        None,     // randomised
    );

    let params_fixed = PhateParams::new(
        Some(2),  // n_dim
        Some(10), // k
        None,     // ann_type
        None,     // decay
        None,     // bandwidth_scale
        None,     // graph_symmetry
        None,     // t_max
        None,     // gamma
        None,     // n_landmarks
        None,     // landmark_method
        None,     // n_svd
        Some(10), // t_custom — Some(10) means Fixed(10)
        None,     // mds_method
        None,     // mds_iter
        None,     // randomised
    );

    let embd_auto = phate(data.as_ref(), None, params_auto, 42, false).unwrap();
    let embd_fixed = phate(data.as_ref(), None, params_fixed, 42, false).unwrap();

    for (embd, label) in [(&embd_auto, "auto"), (&embd_fixed, "fixed")] {
        let all_coords: Vec<f64> = embd.iter().flat_map(|d| d.iter().copied()).collect();
        let nan_count = all_coords.iter().filter(|v| v.is_nan()).count();
        assert_eq!(nan_count, 0, "{} t produced NaN", label);

        let range = all_coords.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - all_coords.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(range > 0.01, "{} t produced collapsed embedding", label);

        let sep = separation_ratio(embd, &labels, n_clusters);
        println!("{} t: separation ratio = {:.2}", label, sep);
        assert!(
            sep > 1.0,
            "{} t should separate clusters, got {:.2}",
            label,
            sep
        );
    }
}

/// Test 13: Bandwidth scale affects affinity spread
#[test]
fn phate_integration_13_bandwidth_scale_effect() {
    let (data, _labels) = create_diagnostic_data(50, 10, 42);
    let k = 10;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42, false);

    println!("\n=== PHATE DIAGNOSTIC 13: Bandwidth Scale Effect ===");

    let graph_narrow = phate_alpha_decay_affinities(
        &knn_indices,
        &knn_dist,
        k,
        Some(40.0),
        0.5,
        1e-4,
        "none",
        true,
    );
    let graph_wide = phate_alpha_decay_affinities(
        &knn_indices,
        &knn_dist,
        k,
        Some(40.0),
        2.0,
        1e-4,
        "none",
        true,
    );

    let mean_narrow = graph_narrow.values.iter().sum::<f64>() / graph_narrow.values.len() as f64;
    let mean_wide = graph_wide.values.iter().sum::<f64>() / graph_wide.values.len() as f64;

    println!("Mean affinity (scale=0.5): {:.6}", mean_narrow);
    println!("Mean affinity (scale=2.0): {:.6}", mean_wide);

    // Wider bandwidth → distances scaled down → affinities closer to 1
    assert!(
        mean_wide > mean_narrow,
        "Wider bandwidth should produce higher mean affinity: narrow={:.4}, wide={:.4}",
        mean_narrow,
        mean_wide
    );
    println!("Wider bandwidth produces higher affinities as expected");
}
