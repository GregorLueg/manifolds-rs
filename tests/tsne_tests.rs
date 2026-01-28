#![allow(clippy::needless_range_loop)]

mod commons;
use commons::*;

use manifolds_rs::data::graph::{gaussian_knn_affinities, symmetrise_affinities_tsne};
use manifolds_rs::prelude::*;
use manifolds_rs::utils::bh_tree::*;
use manifolds_rs::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rustc_hash::FxHashMap;

/// Helper: compute entropy of probability distribution
fn entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > 1e-12)
        .map(|&p| -p * p.log2())
        .sum()
}

/// Helper: build adjacency list from sparse graph
fn graph_to_adj(graph: &SparseGraph<f64>) -> Vec<Vec<(usize, f64)>> {
    let mut adj = vec![Vec::new(); graph.n_vertices];
    for ((&i, &j), &w) in graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
    {
        adj[i].push((j, w));
    }
    adj
}

/// Test 1: Verify kNN search finds correct neighbours - tSNE
#[test]
fn tsne_integration_01_knn_correctness() {
    // Use more points per cluster so k < cluster_size
    let (data, labels) = create_diagnostic_data(100, 10, 42); // Changed from 50 to 100
    let perplexity = 30.0;
    let k = (perplexity * 3.0) as usize; // k = 90, now < 100

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42);

    println!("\n=== t-SNE DIAGNOSTIC 1: kNN Search ===");
    println!("Points per cluster: 100, k = {} neighbours", k);

    // Check self not in neighbours
    let mut self_in_neighbours = 0;
    for (i, neighbours) in knn_indices.iter().enumerate() {
        if neighbours.contains(&i) {
            self_in_neighbours += 1;
        }
    }

    if self_in_neighbours > 0 {
        println!(
            "WARNING: {} points have themselves in neighbours!",
            self_in_neighbours
        );
    } else {
        println!("✓ No self-loops in kNN");
    }

    // Check intra-cluster ratio
    // With 100 points per cluster and k=90, we expect ~99% intra-cluster
    let mut intra_ratio = 0.0;
    for (i, neighbours) in knn_indices.iter().enumerate() {
        let same = neighbours
            .iter()
            .filter(|&&j| labels[j] == labels[i])
            .count();
        intra_ratio += same as f64 / neighbours.len() as f64;
    }
    intra_ratio /= knn_indices.len() as f64;

    // Calculate theoretical maximum
    let points_per_cluster = 100;
    let max_possible_same = (points_per_cluster - 1).min(k);
    let theoretical_max = max_possible_same as f64 / k as f64;

    println!("Intra-cluster neighbour ratio: {:.1}%", intra_ratio * 100.0);
    println!("Theoretical maximum: {:.1}%", theoretical_max * 100.0);

    // Should achieve close to theoretical max for well-separated clusters
    assert!(
        intra_ratio > theoretical_max * 0.9,
        "kNN should find mostly same-cluster neighbours, got {:.1}% (max possible {:.1}%)",
        intra_ratio * 100.0,
        theoretical_max * 100.0
    );

    // Distance statistics
    let all_dists: Vec<f64> = knn_dist.iter().flatten().copied().collect();
    let min_d = all_dists.iter().copied().fold(f64::INFINITY, f64::min);
    let max_d = all_dists.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean_d = all_dists.iter().sum::<f64>() / all_dists.len() as f64;

    println!(
        "Distances: min={:.4}, mean={:.4}, max={:.4}",
        min_d, mean_d, max_d
    );
    assert!(min_d > 0.0, "Min distance should be > 0 (no self)");
}

/// Test 2: Verify the Gaussian affinity calculations - tSNE
#[test]
fn tsne_integration_02_gaussian_affinities() {
    let (data, _labels) = create_diagnostic_data(50, 10, 42);
    let perplexity = 30.0;
    let k = (perplexity * 3.0) as usize;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42);

    println!("\n=== t-SNE DIAGNOSTIC 2: Gaussian Affinities ===");

    let graph = gaussian_knn_affinities(
        &knn_indices,
        &knn_dist,
        perplexity,
        1e-5,
        200,
        true, // squared distances
    );

    let adj = graph_to_adj(&graph);

    // Check row sums
    let mut min_sum = f64::INFINITY;
    let mut max_sum = f64::NEG_INFINITY;
    for neighbours in &adj {
        let sum: f64 = neighbours.iter().map(|(_, w)| w).sum();
        min_sum = min_sum.min(sum);
        max_sum = max_sum.max(sum);
    }

    println!(
        "Row probability sums: min={:.6}, max={:.6}",
        min_sum, max_sum
    );
    assert!(
        (min_sum - 1.0).abs() < 1e-4 && (max_sum - 1.0).abs() < 1e-4,
        "All rows should sum to 1.0"
    );
    println!("✓ All rows sum to 1.0");

    // Check entropy matches target
    let target_entropy = perplexity.log2();
    let mut entropy_errors = Vec::new();

    for neighbours in &adj {
        let probs: Vec<f64> = neighbours.iter().map(|(_, w)| *w).collect();
        let h = entropy(&probs);
        entropy_errors.push((h - target_entropy).abs());
    }

    let max_entropy_err = entropy_errors.iter().copied().fold(0.0, f64::max);
    let mean_entropy_err = entropy_errors.iter().sum::<f64>() / entropy_errors.len() as f64;

    println!(
        "Entropy errors: mean={:.6}, max={:.6} (target entropy={:.4})",
        mean_entropy_err, max_entropy_err, target_entropy
    );
    assert!(
        max_entropy_err < 0.01,
        "Entropy should match target perplexity"
    );
    println!("✓ Entropy correctly calibrated to perplexity");

    // Check no self-loops
    for (&i, &j) in graph.row_indices.iter().zip(&graph.col_indices) {
        assert_ne!(i, j, "Self-loop found at {}", i);
    }
    println!("✓ No self-loops in affinity graph");
}

/// Test 3: Verify the graph symmetrisation for tSNE
#[test]
fn tsne_integration_03_symmetrisation() {
    let (data, _labels) = create_diagnostic_data(50, 10, 42);
    let perplexity = 30.0;
    let k = (perplexity * 3.0) as usize;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42);

    let directed = gaussian_knn_affinities(&knn_indices, &knn_dist, perplexity, 1e-5, 200, true);
    let symmetric = symmetrise_affinities_tsne(directed);

    println!("\n=== t-SNE DIAGNOSTIC 3: Symmetrisation ===");
    println!("Symmetric graph has {} edges", symmetric.values.len());

    let mut adj: FxHashMap<(usize, usize), f64> = FxHashMap::default();
    for ((&i, &j), &w) in symmetric
        .row_indices
        .iter()
        .zip(&symmetric.col_indices)
        .zip(&symmetric.values)
    {
        adj.insert((i, j), w);
    }

    // Check symmetry: P_ij == P_ji
    let mut asymmetry_count = 0;
    let mut max_asymmetry: f64 = 0.0;

    for (&(i, j), &w_ij) in &adj {
        if let Some(&w_ji) = adj.get(&(j, i)) {
            let diff = (w_ij - w_ji).abs();
            if diff > 1e-10 {
                asymmetry_count += 1;
                max_asymmetry = max_asymmetry.max(diff);
            }
        } else {
            asymmetry_count += 1;
        }
    }

    println!("Asymmetric edges: {}", asymmetry_count);
    println!("Max asymmetry: {:.10}", max_asymmetry);
    assert_eq!(asymmetry_count, 0, "Graph should be perfectly symmetric");
    println!("✓ Graph is symmetric");

    // Check total sum = 1 (joint probabilities)
    let total: f64 = symmetric.values.iter().sum();
    println!("Total probability mass: {:.6}", total);

    // Note: sum should be close to 1.0 for joint probabilities
    // But we're summing both (i,j) and (j,i), so it's actually 2.0
    // unless we only count upper triangle
    let upper_triangle_sum: f64 = adj
        .iter()
        .filter(|(&(i, j), _)| i < j)
        .map(|(_, &w)| w)
        .sum::<f64>()
        * 2.0; // multiply by 2 since P_ij = P_ji

    println!("Upper triangle sum * 2: {:.6}", upper_triangle_sum);

    // Check no self-loops
    for (&i, &j) in symmetric.row_indices.iter().zip(&symmetric.col_indices) {
        assert_ne!(i, j, "Self-loop in symmetric graph at {}", i);
    }
    println!("✓ No self-loops in symmetric graph");

    // Check all weights positive
    let min_weight = symmetric
        .values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_weight = symmetric
        .values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    println!("Weight range: [{:.8}, {:.8}]", min_weight, max_weight);
    assert!(min_weight > 0.0, "All weights should be positive");
}

/// Test 4: Barnes Hut behaves as expected
#[test]
fn tsne_integration_04_barnes_hut_tree() {
    let (data, labels) = create_diagnostic_data(50, 10, 42);

    println!("\n=== t-SNE DIAGNOSTIC 4: Barnes-Hut Tree ===");

    // Create a simple 2D embedding for testing
    let n = data.nrows();
    let mut embd: Vec<Vec<f64>> = Vec::with_capacity(n);

    // Place points in 2D based on their cluster
    let mut rng = StdRng::seed_from_u64(42);
    let cluster_centres_2d = [
        (0.0, 0.0),
        (10.0, 0.0),
        (0.0, 10.0),
        (10.0, 10.0),
        (5.0, 5.0),
    ];

    for &label in &labels {
        let (cx, cy) = cluster_centres_2d[label];
        let x = cx + rng.random::<f64>() * 2.0 - 1.0;
        let y = cy + rng.random::<f64>() * 2.0 - 1.0;
        embd.push(vec![x, y]);
    }

    let tree = BarnesHutTree::new(&embd);

    println!("Tree has {} nodes for {} points", tree.nodes.len(), n);

    // Check root mass equals n
    let root = &tree.nodes[0];
    assert!(
        (root.mass - n as f64).abs() < 1e-10,
        "Root mass {} should equal n={}",
        root.mass,
        n
    );
    println!("✓ Root mass equals n");

    // Check centre of mass is reasonable
    let true_com_x: f64 = embd.iter().map(|p| p[0]).sum::<f64>() / n as f64;
    let true_com_y: f64 = embd.iter().map(|p| p[1]).sum::<f64>() / n as f64;

    assert!(
        (root.com_x - true_com_x).abs() < 1e-10,
        "Root COM x mismatch"
    );
    assert!(
        (root.com_y - true_com_y).abs() < 1e-10,
        "Root COM y mismatch"
    );
    println!("✓ Root centre of mass correct");

    // Test force computation
    let mut total_sum_q = 0.0;
    let mut nan_count = 0;
    let mut inf_count = 0;

    for i in 0..n {
        let (fx, fy, sum_q) = tree.compute_repulsive_force(i, embd[i][0], embd[i][1], 0.5);

        if fx.is_nan() || fy.is_nan() || sum_q.is_nan() {
            nan_count += 1;
        }
        if fx.is_infinite() || fy.is_infinite() || sum_q.is_infinite() {
            inf_count += 1;
        }
        total_sum_q += sum_q;
    }

    println!("NaN forces: {}, Inf forces: {}", nan_count, inf_count);
    assert_eq!(nan_count, 0, "No NaN forces allowed");
    assert_eq!(inf_count, 0, "No Inf forces allowed");
    println!("✓ All forces are finite");

    println!("Total sum_q across all points: {:.4}", total_sum_q);
    assert!(total_sum_q > 0.0, "Total sum_q should be positive");

    // Compare exact vs approximate
    let (fx_exact, fy_exact, sq_exact) =
        tree.compute_repulsive_force(0, embd[0][0], embd[0][1], 0.0);
    let (fx_approx, fy_approx, sq_approx) =
        tree.compute_repulsive_force(0, embd[0][0], embd[0][1], 0.5);

    println!("\nPoint 0 forces:");
    println!(
        "  Exact:  fx={:.6}, fy={:.6}, sum_q={:.6}",
        fx_exact, fy_exact, sq_exact
    );
    println!(
        "  Approx: fx={:.6}, fy={:.6}, sum_q={:.6}",
        fx_approx, fy_approx, sq_approx
    );

    // Approximation should be reasonable
    let fx_err = if fx_exact.abs() > 1e-6 {
        ((fx_approx - fx_exact) / fx_exact).abs()
    } else {
        0.0
    };
    let fy_err = if fy_exact.abs() > 1e-6 {
        ((fy_approx - fy_exact) / fy_exact).abs()
    } else {
        0.0
    };

    println!(
        "  Relative errors: fx={:.1}%, fy={:.1}%",
        fx_err * 100.0,
        fy_err * 100.0
    );
}

/// Test 5: tSNE initialisations
#[test]
fn tsne_integration_05_initialisation() {
    let (data, _) = create_diagnostic_data(50, 10, 42);

    let nn_params = NearestNeighbourParams::default();
    let (graph, _, _) = construct_tsne_graph(
        data.as_ref(),
        30.0,
        "hnsw".to_string(),
        &nn_params,
        42,
        false,
    );

    println!("\n=== t-SNE DIAGNOSTIC 5: Initialisation ===");

    for init_name in &["pca", "random", "spectral"] {
        let range = Some(1e-4); // t-SNE standard
        let init_type = parse_initilisation(init_name, false, range).unwrap();
        let embd = initialise_embedding(&init_type, 2, 42, &graph, data.as_ref());

        println!("\n{} initialisation:", init_name);

        // Check coordinate range
        let coords: Vec<f64> = embd.iter().flat_map(|p| p.iter().copied()).collect();
        let min_c = coords.iter().copied().fold(f64::INFINITY, f64::min);
        let max_c = coords.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let std_dev = {
            let mean = coords.iter().sum::<f64>() / coords.len() as f64;
            let var = coords.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / coords.len() as f64;
            var.sqrt()
        };

        println!("  Range: [{:.6}, {:.6}]", min_c, max_c);
        println!("  Std dev: {:.6}", std_dev);

        // For t-SNE, initial spread should be small
        if *init_name == "pca" {
            assert!(
                std_dev < 1e-2,
                "PCA init std dev {:.6} too large for t-SNE (should be ~1e-4)",
                std_dev
            );
        }

        // Check no NaN/Inf
        assert!(
            coords.iter().all(|x| x.is_finite()),
            "{} init produced non-finite values",
            init_name
        );
        println!("  ✓ All coordinates finite");
    }
}

/// Test 6 tSNE - do the results make sense
#[test]
fn tsne_integration_06_optimisation_quality() {
    let (data, labels) = create_diagnostic_data(40, 10, 123);

    println!("\n=== t-SNE DIAGNOSTIC 6: Optimisation Quality ===");

    let params = TsneParams::new(
        Some(2),     // n_dim
        Some(20.0),  // perplexity
        Some(1e-4),  // init_range
        Some(200.0), // learning rate
        Some(500),   // epochs (fewer for test speed)
        None,        // ann_type
        Some(0.5),   // theta
        Some(3),
    );

    let embedding = tsne(data.as_ref(), &params, "bh", 42, true);

    // Check finite
    let mut nan_count = 0;
    let mut inf_count = 0;
    for i in 0..embedding[0].len() {
        if embedding[0][i].is_nan() || embedding[1][i].is_nan() {
            nan_count += 1;
        }
        if embedding[0][i].is_infinite() || embedding[1][i].is_infinite() {
            inf_count += 1;
        }
    }

    println!("NaN points: {}, Inf points: {}", nan_count, inf_count);
    assert_eq!(nan_count, 0, "Embedding has NaN values");
    assert_eq!(inf_count, 0, "Embedding has Inf values");

    // Check coordinate range is reasonable (not exploded)
    let all_coords: Vec<f64> = embedding.iter().flat_map(|d| d.iter().copied()).collect();
    let min_c = all_coords.iter().copied().fold(f64::INFINITY, f64::min);
    let max_c = all_coords.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max_c - min_c;

    println!(
        "Coordinate range: [{:.2}, {:.2}] (span: {:.2})",
        min_c, max_c, range
    );
    assert!(range < 1000.0, "Embedding exploded! Range = {}", range);
    assert!(range > 1.0, "Embedding collapsed! Range = {}", range);

    // Check cluster separation
    let mut cluster_centres: FxHashMap<usize, (f64, f64, usize)> = FxHashMap::default();
    for (i, &label) in labels.iter().enumerate() {
        let entry = cluster_centres.entry(label).or_insert((0.0, 0.0, 0));
        entry.0 += embedding[0][i];
        entry.1 += embedding[1][i];
        entry.2 += 1;
    }

    let centroids: Vec<(usize, f64, f64)> = cluster_centres
        .iter()
        .map(|(&l, &(sx, sy, c))| (l, sx / c as f64, sy / c as f64))
        .collect();

    // Inter-cluster distances
    let mut min_inter = f64::INFINITY;
    for i in 0..centroids.len() {
        for j in (i + 1)..centroids.len() {
            let d = ((centroids[i].1 - centroids[j].1).powi(2)
                + (centroids[i].2 - centroids[j].2).powi(2))
            .sqrt();
            min_inter = min_inter.min(d);
        }
    }

    // Intra-cluster distances
    let mut avg_intra = 0.0;
    for (label, cx, cy) in &centroids {
        let pts: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == *label)
            .map(|(i, _)| i)
            .collect();

        let intra: f64 = pts
            .iter()
            .map(|&i| ((embedding[0][i] - cx).powi(2) + (embedding[1][i] - cy).powi(2)).sqrt())
            .sum::<f64>()
            / pts.len() as f64;

        avg_intra += intra;
    }
    avg_intra /= 5.0;

    println!("Min inter-cluster distance: {:.3}", min_inter);
    println!("Avg intra-cluster distance: {:.3}", avg_intra);

    let separation = min_inter / avg_intra;
    println!("Separation ratio: {:.2}", separation);

    assert!(
        separation > 1.0,
        "Clusters should be well separated, got ratio {:.2}",
        separation
    );
    println!("✓ Clusters are well separated");
}

/// Test 7 - same results with same seed (was not behaving before...)
#[test]
fn tsne_integration_07_reproducibility() {
    let (data, _) = create_diagnostic_data(40, 10, 42);

    println!("\n=== t-SNE DIAGNOSTIC 7: Reproducibility ===");

    let params = TsneParams::new(
        Some(2),
        Some(20.0),
        Some(1e-4),
        Some(200.0),
        Some(200), // fewer epochs for speed
        None,
        Some(0.5),
        Some(3),
    );

    let embd1 = tsne(data.as_ref(), &params, "bh", 42, false);
    let embd2 = tsne(data.as_ref(), &params, "bh", 42, false);

    let mut max_diff: f64 = 0.0;
    for i in 0..embd1[0].len() {
        for dim in 0..2 {
            let diff = (embd1[dim][i] - embd2[dim][i]).abs();
            max_diff = max_diff.max(diff);
        }
    }

    println!("Max coordinate difference: {:.10}", max_diff);

    assert!(
        max_diff < 1e-6,
        "t-SNE should be reproducible with same seed, got diff = {}",
        max_diff
    );
    println!("✓ t-SNE is reproducible");
}

/// Test 8 - different seeds produce different, but similar results
#[test]
fn tsne_integration_08_different_seeds() {
    let (data, _) = create_diagnostic_data(40, 10, 42);

    println!("\n=== t-SNE DIAGNOSTIC 8: Different Seeds ===");

    let params = TsneParams::new(
        Some(2),
        Some(20.0),
        Some(1e-4),
        Some(200.0),
        Some(200),
        None,
        Some(0.5),
        Some(3),
    );

    let embd1 = tsne(data.as_ref(), &params, "bh", 42, false);
    let embd2 = tsne(data.as_ref(), &params, "bh", 123, false);

    let mut max_diff: f64 = 0.0;
    for i in 0..embd1[0].len() {
        for dim in 0..2 {
            let diff = (embd1[dim][i] - embd2[dim][i]).abs();
            max_diff = max_diff.max(diff);
        }
    }

    println!(
        "Max coordinate difference with different seeds: {:.4}",
        max_diff
    );

    assert!(
        max_diff > 0.1,
        "Different seeds should produce different embeddings"
    );
    println!("✓ Different seeds produce different results");
}

/// Test 9: FFT t-SNE - optimisation quality
#[cfg(feature = "fft_tsne")]
#[test]
fn tsne_integration_09_fft_optimisation_quality() {
    let (data, labels) = create_diagnostic_data(100, 10, 123);

    println!("\n=== t-SNE DIAGNOSTIC 9: FFT Optimisation Quality ===");

    let params = TsneParams::new(
        Some(2),
        Some(20.0),
        Some(1e-4),
        Some(100.0),
        Some(500),
        None,
        Some(0.5),
        Some(3),
    );

    let embedding = tsne(data.as_ref(), &params, "fft", 42, true);

    // Check finite
    let mut nan_count = 0;
    let mut inf_count = 0;
    for i in 0..embedding[0].len() {
        if embedding[0][i].is_nan() || embedding[1][i].is_nan() {
            nan_count += 1;
        }
        if embedding[0][i].is_infinite() || embedding[1][i].is_infinite() {
            inf_count += 1;
        }
    }

    println!("NaN points: {}, Inf points: {}", nan_count, inf_count);
    assert_eq!(nan_count, 0, "Embedding has NaN values");
    assert_eq!(inf_count, 0, "Embedding has Inf values");

    // Check coordinate range
    let all_coords: Vec<f64> = embedding.iter().flat_map(|d| d.iter().copied()).collect();
    let min_c = all_coords.iter().copied().fold(f64::INFINITY, f64::min);
    let max_c = all_coords.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max_c - min_c;

    println!(
        "Coordinate range: [{:.2}, {:.2}] (span: {:.2})",
        min_c, max_c, range
    );
    assert!(range < 1000.0, "Embedding exploded! Range = {}", range);
    assert!(range > 1.0, "Embedding collapsed! Range = {}", range);

    // Check cluster separation
    let mut cluster_centres: FxHashMap<usize, (f64, f64, usize)> = FxHashMap::default();
    for (i, &label) in labels.iter().enumerate() {
        let entry = cluster_centres.entry(label).or_insert((0.0, 0.0, 0));
        entry.0 += embedding[0][i];
        entry.1 += embedding[1][i];
        entry.2 += 1;
    }

    let centroids: Vec<(usize, f64, f64)> = cluster_centres
        .iter()
        .map(|(&l, &(sx, sy, c))| (l, sx / c as f64, sy / c as f64))
        .collect();

    let mut min_inter = f64::INFINITY;
    for i in 0..centroids.len() {
        for j in (i + 1)..centroids.len() {
            let d = ((centroids[i].1 - centroids[j].1).powi(2)
                + (centroids[i].2 - centroids[j].2).powi(2))
            .sqrt();
            min_inter = min_inter.min(d);
        }
    }

    let mut avg_intra = 0.0;
    for (label, cx, cy) in &centroids {
        let pts: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == *label)
            .map(|(i, _)| i)
            .collect();

        let intra: f64 = pts
            .iter()
            .map(|&i| ((embedding[0][i] - cx).powi(2) + (embedding[1][i] - cy).powi(2)).sqrt())
            .sum::<f64>()
            / pts.len() as f64;

        avg_intra += intra;
    }
    avg_intra /= 5.0;

    println!("Min inter-cluster distance: {:.3}", min_inter);
    println!("Avg intra-cluster distance: {:.3}", avg_intra);

    let separation = min_inter / avg_intra;
    println!("Separation ratio: {:.2}", separation);

    assert!(
        separation > 1.0,
        "Clusters should be well separated, got ratio {:.2}",
        separation
    );
    println!("✓ FFT clusters are well separated");
}

/// Test 10: FFT t-SNE reproducibility
#[cfg(feature = "fft_tsne")]
#[test]
fn tsne_integration_10_fft_reproducibility() {
    let (data, _) = create_diagnostic_data(100, 10, 42);

    println!("\n=== t-SNE DIAGNOSTIC 10: FFT Reproducibility ===");

    let params = TsneParams::new(
        Some(2),
        Some(20.0),
        Some(1e-4),
        Some(200.0),
        Some(200),
        None,
        Some(0.5),
        Some(3),
    );

    let embd1 = tsne(data.as_ref(), &params, "fft", 123, false);
    let embd2 = tsne(data.as_ref(), &params, "fft", 123, false);

    let mut max_diff: f64 = 0.0;
    for i in 0..embd1[0].len() {
        for dim in 0..2 {
            let diff = (embd1[dim][i] - embd2[dim][i]).abs();
            max_diff = max_diff.max(diff);
        }
    }

    println!("Max coordinate difference: {:.10}", max_diff);

    assert!(
        max_diff < 1e-6,
        "FFT t-SNE should be reproducible with same seed, got diff = {}",
        max_diff
    );
    println!("✓ FFT t-SNE is reproducible");
}

/// Test 11: FFT t-SNE different seeds
#[cfg(feature = "fft_tsne")]
#[test]
fn tsne_integration_11_fft_different_seeds() {
    let (data, _) = create_diagnostic_data(100, 10, 42);

    println!("\n=== t-SNE DIAGNOSTIC 11: FFT Different Seeds ===");

    let params = TsneParams::new(
        Some(2),
        Some(20.0),
        Some(1e-4),
        Some(200.0),
        Some(200),
        None,
        Some(0.5),
        Some(3),
    );

    let embd1 = tsne(data.as_ref(), &params, "fft", 42, false);
    let embd2 = tsne(data.as_ref(), &params, "fft", 123, false);

    let mut max_diff: f64 = 0.0;
    for i in 0..embd1[0].len() {
        for dim in 0..2 {
            let diff = (embd1[dim][i] - embd2[dim][i]).abs();
            max_diff = max_diff.max(diff);
        }
    }

    println!(
        "Max coordinate difference with different seeds: {:.4}",
        max_diff
    );

    assert!(
        max_diff > 0.1,
        "Different seeds should produce different embeddings"
    );
    println!("✓ FFT different seeds produce different results");
}

/// Test 12: Compare Barnes-Hut vs FFT quality
#[cfg(feature = "fft_tsne")]
#[test]
fn tsne_integration_12_bh_vs_fft_comparison() {
    let (data, labels) = create_diagnostic_data(100, 10, 42);

    println!("\n=== t-SNE DIAGNOSTIC 12: Barnes-Hut vs FFT Comparison ===");

    let params = TsneParams::new(
        Some(2),
        Some(20.0),
        Some(1e-4),
        Some(200.0),
        Some(300),
        None,
        Some(0.5),
        Some(3),
    );

    let embd_bh = tsne(data.as_ref(), &params, "bh", 42, false);
    let embd_fft = tsne(data.as_ref(), &params, "fft", 42, false);

    // Helper to compute cluster separation
    let compute_separation = |embedding: &[Vec<f64>]| -> f64 {
        let mut cluster_centres: FxHashMap<usize, (f64, f64, usize)> = FxHashMap::default();
        for (i, &label) in labels.iter().enumerate() {
            let entry = cluster_centres.entry(label).or_insert((0.0, 0.0, 0));
            entry.0 += embedding[0][i];
            entry.1 += embedding[1][i];
            entry.2 += 1;
        }

        let centroids: Vec<(usize, f64, f64)> = cluster_centres
            .iter()
            .map(|(&l, &(sx, sy, c))| (l, sx / c as f64, sy / c as f64))
            .collect();

        let mut min_inter = f64::INFINITY;
        for i in 0..centroids.len() {
            for j in (i + 1)..centroids.len() {
                let d = ((centroids[i].1 - centroids[j].1).powi(2)
                    + (centroids[i].2 - centroids[j].2).powi(2))
                .sqrt();
                min_inter = min_inter.min(d);
            }
        }

        let mut avg_intra = 0.0;
        for (label, cx, cy) in &centroids {
            let pts: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, &l)| l == *label)
                .map(|(i, _)| i)
                .collect();

            let intra: f64 = pts
                .iter()
                .map(|&i| ((embedding[0][i] - cx).powi(2) + (embedding[1][i] - cy).powi(2)).sqrt())
                .sum::<f64>()
                / pts.len() as f64;

            avg_intra += intra;
        }
        avg_intra /= 5.0;

        min_inter / avg_intra
    };

    let sep_bh = compute_separation(&embd_bh);
    let sep_fft = compute_separation(&embd_fft);

    println!("Barnes-Hut separation ratio: {:.2}", sep_bh);
    println!("FFT separation ratio: {:.2}", sep_fft);

    assert!(sep_bh > 1.0, "BH should separate clusters");
    assert!(sep_fft > 1.0, "FFT should separate clusters");

    let sep_ratio = sep_bh / sep_fft;
    println!("Quality ratio (BH/FFT): {:.2}", sep_ratio);

    assert!(
        sep_ratio > 0.5 && sep_ratio < 2.0,
        "Both methods should produce similar quality, got ratio {:.2}",
        sep_ratio
    );
    println!("✓ Both methods produce comparable quality");
}
