#![allow(clippy::needless_range_loop)]

mod commons;
use commons::*;

use manifolds_rs::data::graph::smooth_knn_dist;
use manifolds_rs::prelude::*;
use manifolds_rs::*;
use rustc_hash::FxHashMap;

/// Test 1: Verify kNN search finds correct neighbours
#[test]
fn umap_integration_01_knn_correctness() {
    let (data, labels) = create_diagnostic_data(50, 10, 42);
    let k = 15;

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42);

    println!("\n=== DIAGNOSTIC 1: kNN Search Correctness ===");
    println!(
        "Data shape: {} samples, {} features",
        data.nrows(),
        data.ncols()
    );
    println!("Requested k = {} neighbours", k);
    println!("Returned {} neighbours per point", knn_indices[0].len());

    // Check that kNN doesn't include self
    let mut self_in_neighbours = 0;
    for (i, neighbours) in knn_indices.iter().enumerate() {
        if neighbours.contains(&i) {
            self_in_neighbours += 1;
            if self_in_neighbours == 1 {
                println!(
                    "WARNING: Point {} has itself in neighbours: {:?}",
                    i, neighbours
                );
            }
        }
    }

    if self_in_neighbours > 0 {
        println!(
            "ERROR: {} points have themselves in their neighbours!",
            self_in_neighbours
        );
    } else {
        println!("✓ No point has itself in neighbours (correct)");
    }

    // check that neighbours are mostly from same cluster
    let mut intra_cluster_ratio = 0.0;
    for (i, neighbours) in knn_indices.iter().enumerate() {
        let my_label = labels[i];
        let same_cluster = neighbours
            .iter()
            .filter(|&&j| labels[j] == my_label)
            .count();
        intra_cluster_ratio += same_cluster as f64 / neighbours.len() as f64;
    }
    intra_cluster_ratio /= knn_indices.len() as f64;

    println!(
        "Average intra-cluster neighbor ratio: {:.2}%",
        intra_cluster_ratio * 100.0
    );

    assert!(
        intra_cluster_ratio > 0.8,
        "kNN should find mostly same-cluster neighbours, got {:.2}",
        intra_cluster_ratio
    );

    // check distance statistics
    let all_dists: Vec<f64> = knn_dist.iter().flatten().copied().collect();
    let min_dist = all_dists.iter().copied().fold(f64::INFINITY, f64::min);
    let max_dist = all_dists.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean_dist = all_dists.iter().sum::<f64>() / all_dists.len() as f64;

    println!(
        "Distance statistics: min = {:.3}, mean = {:.3}, max = {:.3}",
        min_dist, mean_dist, max_dist
    );

    assert!(min_dist > 0.0, "Minimum distance should be > 0 (no self)");
}

/// Test 2: Verify smooth_knn_dist produces reasonable sigma/rho
#[test]
fn umap_integration_02_smooth_knn_dist() {
    let (data, _labels) = create_diagnostic_data(50, 10, 42);
    let k = 15;

    let nn_params = NearestNeighbourParams::default();
    let (_, knn_dist) = run_ann_search(data.as_ref(), k, "hnsw".to_string(), &nn_params, 42);

    println!("\n=== DIAGNOSTIC 2: smooth_knn_dist Values ===");

    let umap_params = UmapGraphParams::default();
    let (sigma, rho) = smooth_knn_dist(
        &knn_dist,
        knn_dist[0].len(),
        umap_params.local_connectivity,
        umap_params.bandwidth,
        64,
    );

    println!("Sigma statistics:");
    let min_sigma = sigma.iter().copied().fold(f64::INFINITY, f64::min);
    let max_sigma = sigma.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean_sigma = sigma.iter().sum::<f64>() / sigma.len() as f64;
    println!(
        "  min = {:.6}, mean = {:.6}, max = {:.6}",
        min_sigma, mean_sigma, max_sigma
    );

    println!("Rho statistics:");
    let min_rho = rho.iter().copied().fold(f64::INFINITY, f64::min);
    let max_rho = rho.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean_rho = rho.iter().sum::<f64>() / rho.len() as f64;
    let zero_rho = rho.iter().filter(|&&r| r == 0.0).count();
    println!(
        "  min = {:.6}, mean = {:.6}, max = {:.6}",
        min_rho, mean_rho, max_rho
    );
    println!("  Points with rho=0: {} / {}", zero_rho, rho.len());

    // Critical checks
    assert!(min_sigma > 0.0, "All sigma values should be > 0");
    assert!(
        mean_sigma > 0.01,
        "Mean sigma seems too small: {}",
        mean_sigma
    );

    if zero_rho > 0 {
        println!(
            "WARNING: {} points have rho = 0 (first neighbor at distance 0!)",
            zero_rho
        );
        println!("This suggests self is still in the neighbor list!");
    }

    assert_eq!(
        zero_rho, 0,
        "No point should have rho=0 (would mean self is in neighbours)"
    );

    // Check that rho values are reasonable (should be smallest distance to actual neighbor)
    for i in 0..knn_dist.len() {
        let expected_rho = knn_dist[i][0]; // First neighbor distance
        let actual_rho = rho[i];

        assert!(
            (expected_rho - actual_rho).abs() < 1e-6,
            "Point {}: rho = {:.6} but first neighbor is at distance {:.6}",
            i,
            actual_rho,
            expected_rho
        );
    }
    println!("✓ All rho values correctly match first neighbor distance");
}

/// Test 3: Verify graph construction creates strong intra-cluster edges
#[test]
fn umap_integration_03_graph_connectivity() {
    let (data, labels) = create_diagnostic_data(50, 10, 42);
    let k = 15;

    let nn_params = NearestNeighbourParams::default();
    let umap_params = UmapGraphParams::default();

    let (graph, _, _) = construct_umap_graph(
        data.as_ref(),
        k,
        "hnsw".to_string(),
        &umap_params,
        &nn_params,
        500,
        42,
        false,
    );

    println!("\n=== DIAGNOSTIC 3: Graph Connectivity ===");
    println!("Graph has {} edges", graph.values.len());

    // Build adjacency list
    let n = graph.n_samples;
    let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
    for ((&i, &j), &w) in graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
    {
        adj[i].push((j, w));
    }

    // Check connectivity within each cluster
    for cluster_id in 0..5 {
        let cluster_points: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == cluster_id)
            .map(|(i, _)| i)
            .collect();

        println!(
            "\nCluster {} ({} points):",
            cluster_id,
            cluster_points.len()
        );

        // BFS to check if cluster is connected
        let mut visited = vec![false; n];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(cluster_points[0]);
        visited[cluster_points[0]] = true;
        let mut reachable = 1;

        while let Some(node) = queue.pop_front() {
            for &(neighbor, _) in &adj[node] {
                if !visited[neighbor] && cluster_points.contains(&neighbor) {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                    reachable += 1;
                }
            }
        }

        println!(
            "  Reachable within cluster: {} / {}",
            reachable,
            cluster_points.len()
        );

        // Compute average edge weights within cluster
        let mut intra_weights = Vec::new();
        let mut inter_weights = Vec::new();

        for &i in &cluster_points {
            for &(j, w) in &adj[i] {
                if cluster_points.contains(&j) {
                    intra_weights.push(w);
                } else {
                    inter_weights.push(w);
                }
            }
        }

        if !intra_weights.is_empty() {
            let avg_intra = intra_weights.iter().sum::<f64>() / intra_weights.len() as f64;
            let min_intra = intra_weights.iter().copied().fold(f64::INFINITY, f64::min);
            let max_intra = intra_weights
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            println!(
                "  Intra-cluster edges: min = {:.6}, avg = {:.6}, max = {:.6}",
                min_intra, avg_intra, max_intra
            );
        }

        if !inter_weights.is_empty() {
            let avg_inter = inter_weights.iter().sum::<f64>() / inter_weights.len() as f64;
            let min_inter = inter_weights.iter().copied().fold(f64::INFINITY, f64::min);
            let max_inter = inter_weights
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            println!(
                "  Inter-cluster edges: min = {:.6}, avg = {:.6}, max = {:.6}",
                min_inter, avg_inter, max_inter
            );
        }

        assert_eq!(
            reachable,
            cluster_points.len(),
            "Cluster {} is fragmented! Only {} / {} points reachable",
            cluster_id,
            reachable,
            cluster_points.len()
        );
    }
}

/// Test 4: Verify initialisation doesn't pre-split clusters
#[test]
fn umap_integration_04_initialisation() {
    let (data, labels) = create_diagnostic_data(50, 10, 42);

    let umap_params = UmapGraphParams::default();
    let nn_params = NearestNeighbourParams::default();

    let (graph, _, _) = construct_umap_graph(
        data.as_ref(),
        15,
        "hnsw".to_string(),
        &umap_params,
        &nn_params,
        500,
        42,
        false,
    );

    println!("\n=== DIAGNOSTIC 4: Initialisation Quality ===");

    // Test each initialisation method including PCA
    for init_name in &["spectral", "random", "pca"] {
        let init_type = parse_initilisation(init_name, false, None).unwrap();

        println!("What is this: {:?}", init_type);

        let embedding = initialise_embedding(&init_type, 2, 42, &graph, data.as_ref());

        println!("\n{} initialisation:", init_name);

        // Check coordinate range
        let coords: Vec<f64> = embedding.iter().flat_map(|p| p.iter().copied()).collect();
        let min_coord = coords.iter().copied().fold(f64::INFINITY, f64::min);
        let max_coord = coords.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = max_coord - min_coord;
        println!(
            "  Coordinate range: [{:.3}, {:.3}] (span: {:.3})",
            min_coord, max_coord, range
        );

        // Check if clusters are separated
        let mut cluster_centres: FxHashMap<usize, (f64, f64, usize)> = FxHashMap::default();

        for (i, &label) in labels.iter().enumerate() {
            let entry = cluster_centres.entry(label).or_insert((0.0, 0.0, 0));
            entry.0 += embedding[i][0];
            entry.1 += embedding[i][1];
            entry.2 += 1;
        }

        // Compute centroids
        let mut centroids: Vec<(usize, f64, f64)> = Vec::new();
        for (label, (sum_x, sum_y, count)) in cluster_centres {
            centroids.push((label, sum_x / count as f64, sum_y / count as f64));
        }

        // Check pairwise distances between centroids
        let mut min_centroid_dist = f64::INFINITY;
        let mut max_centroid_dist = f64::NEG_INFINITY;
        for i in 0..centroids.len() {
            for j in (i + 1)..centroids.len() {
                let dist = ((centroids[i].1 - centroids[j].1).powi(2)
                    + (centroids[i].2 - centroids[j].2).powi(2))
                .sqrt();
                min_centroid_dist = min_centroid_dist.min(dist);
                max_centroid_dist = max_centroid_dist.max(dist);
            }
        }

        println!(
            "  Inter-cluster centroid distances: min = {:.3}, max = {:.3}",
            min_centroid_dist, max_centroid_dist
        );

        // Check spread within each cluster
        let mut avg_intra_dist = 0.0;
        for (label, cx, cy) in &centroids {
            let points: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, &l)| l == *label)
                .map(|(i, _)| i)
                .collect();

            let intra_dist: f64 = points
                .iter()
                .map(|&i| ((embedding[i][0] - cx).powi(2) + (embedding[i][1] - cy).powi(2)).sqrt())
                .sum::<f64>()
                / points.len() as f64;

            avg_intra_dist += intra_dist;
        }
        avg_intra_dist /= 5.0;
        println!("  Average intra-cluster distance: {:.3}", avg_intra_dist);

        // Critical assertions
        assert!(
            range > 1.0,
            "{} initialization has insufficient spread: range = {:.3} (need > 1.0)",
            init_name,
            range
        );

        // For spectral and random, we expect decent separation
        if init_name != &"pca" {
            assert!(
                min_centroid_dist > 0.1 || max_centroid_dist > 2.0,
                "{} initialisation has poor initial separation: min = {:.3}, max = {:.3}",
                init_name,
                min_centroid_dist,
                max_centroid_dist
            );
        }
    }
}

/// Test 5: Check optimisation with different optimisers
#[test]
fn umap_integration_05_optimisation_quality() {
    let (data, labels) = create_diagnostic_data(50, 10, 123);

    println!("\n=== DIAGNOSTIC 5: Optimisation Quality ===");

    // Test all init+optimizer combinations
    let configs = vec![
        ("spectral", "adam"),
        ("spectral", "sgd"),
        ("spectral", "adam_parallel"),
        ("pca", "adam"),
        ("pca", "sgd"),
        ("pca", "adam_parallel"),
        ("random", "adam"),
        ("random", "sgd"),
        ("random", "adam_parallel"),
    ];

    for (init, opt) in configs {
        println!("\n--- Testing: init = {}, optimiser = {} ---", init, opt);

        let params = UmapParams::new(
            Some(2),
            Some(15),
            Some(opt.to_string()),
            None,
            Some(init.to_string()),
            None,
            None,
            None,
            None,
            None,
        );

        let embedding = umap(data.as_ref(), &params, 42, false);

        // Check that coordinates are finite
        let mut has_nan = false;
        let mut has_inf = false;
        for i in 0..embedding[0].len() {
            if embedding[0][i].is_nan() || embedding[1][i].is_nan() {
                has_nan = true;
            }
            if embedding[0][i].is_infinite() || embedding[1][i].is_infinite() {
                has_inf = true;
            }
        }

        assert!(!has_nan, "Embedding contains NaN values!");
        assert!(!has_inf, "Embedding contains infinite values!");

        // Check cluster separation
        let mut cluster_centres: FxHashMap<usize, (f64, f64, usize)> = FxHashMap::default();

        for (i, &label) in labels.iter().enumerate() {
            let entry = cluster_centres.entry(label).or_insert((0.0, 0.0, 0));
            entry.0 += embedding[0][i];
            entry.1 += embedding[1][i];
            entry.2 += 1;
        }

        let mut centroids: Vec<(usize, f64, f64)> = Vec::new();
        for (label, (sum_x, sum_y, count)) in cluster_centres {
            centroids.push((label, sum_x / count as f64, sum_y / count as f64));
        }

        // Check if any cluster is fragmented
        for cluster_id in 0..5 {
            let points: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, &l)| l == cluster_id)
                .map(|(i, _)| i)
                .collect();

            // Check connectivity in 2D space (threshold = 3.0)
            let threshold = 3.0;

            let mut visited = vec![false; points.len()];
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(0);
            visited[0] = true;
            let mut reachable = 1;

            while let Some(idx) = queue.pop_front() {
                let pi = points[idx];

                for (other_idx, &other_i) in points.iter().enumerate() {
                    if !visited[other_idx] {
                        let dist = ((embedding[0][pi] - embedding[0][other_i]).powi(2)
                            + (embedding[1][pi] - embedding[1][other_i]).powi(2))
                        .sqrt();

                        if dist < threshold {
                            visited[other_idx] = true;
                            queue.push_back(other_idx);
                            reachable += 1;
                        }
                    }
                }
            }

            let connectivity_ratio = reachable as f64 / points.len() as f64;
            println!(
                "  Cluster {}: {}/{} points connected ({:.1}%)",
                cluster_id,
                reachable,
                points.len(),
                connectivity_ratio * 100.0
            );

            assert!(
                connectivity_ratio > 0.85,
                "Cluster {} is fragmented with init={}, opt={}! Only {:.1}% connected",
                cluster_id,
                init,
                opt,
                connectivity_ratio * 100.0
            );
        }

        // Check minimum inter-cluster distance
        let mut min_inter_dist = f64::INFINITY;
        let mut avg_inter_dist = 0.0;
        let mut count = 0;

        for i in 0..centroids.len() {
            for j in (i + 1)..centroids.len() {
                let dist = ((centroids[i].1 - centroids[j].1).powi(2)
                    + (centroids[i].2 - centroids[j].2).powi(2))
                .sqrt();
                min_inter_dist = min_inter_dist.min(dist);
                avg_inter_dist += dist;
                count += 1;
            }
        }
        avg_inter_dist /= count as f64;

        println!(
            "  Inter-cluster distances: min = {:.3}, avg = {:.3}",
            min_inter_dist, avg_inter_dist
        );

        assert!(
            min_inter_dist > 0.5,
            "Clusters too close with init = {}, opt = {}: min dist = {:.3}",
            init,
            opt,
            min_inter_dist
        );

        // Check average intra-cluster compactness
        let mut avg_intra_dist = 0.0;
        for (cluster_id, cx, cy) in &centroids {
            let points: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, &l)| l == *cluster_id)
                .map(|(i, _)| i)
                .collect();

            let intra_dist: f64 = points
                .iter()
                .map(|&i| ((embedding[0][i] - cx).powi(2) + (embedding[1][i] - cy).powi(2)).sqrt())
                .sum::<f64>()
                / points.len() as f64;

            avg_intra_dist += intra_dist;
        }
        avg_intra_dist /= 5.0;

        println!("  Average intra-cluster distance: {:.3}", avg_intra_dist);

        // Quality metric: inter-cluster distance should be >> intra-cluster distance
        let separation_ratio = min_inter_dist / avg_intra_dist;
        println!("  Separation ratio (inter/intra): {:.2}", separation_ratio);

        assert!(
            separation_ratio > 0.3,
            "Poor separation with init = {}, opt = {}: ratio = {:.2}",
            init,
            opt,
            separation_ratio
        );
    }

    println!("\n✓ All init+optimiser combinations produced valid embeddings!");
}

/// Test 6: Compare optimisation consistency across runs
#[test]
fn umap_integration_06_reproducibility() {
    let (data, _) = create_diagnostic_data(50, 10, 42);

    println!("\n=== DIAGNOSTIC 6: Reproducibility ===");

    // Run UMAP twice with same seed
    let params = UmapParams::new(
        Some(2),
        Some(15),
        Some("adam_parallel".to_string()),
        None,
        Some("spectral".to_string()),
        None,
        None,
        None,
        None,
        None,
    );

    let embedding1 = umap(data.as_ref(), &params, 42, false);
    let embedding2 = umap(data.as_ref(), &params, 42, false);

    // Check if embeddings are identical
    let mut max_diff: f64 = 0.0;
    for i in 0..embedding1[0].len() {
        for dim in 0..2 {
            let diff = (embedding1[dim][i] - embedding2[dim][i]).abs();
            max_diff = max_diff.max(diff);
        }
    }

    println!(
        "Maximum coordinate difference between runs: {:.10}",
        max_diff
    );

    assert!(
        max_diff < 1e-6,
        "UMAP should be reproducible with same seed, but max diff = {}",
        max_diff
    );

    println!("✓ UMAP is reproducible with same seed");
}
