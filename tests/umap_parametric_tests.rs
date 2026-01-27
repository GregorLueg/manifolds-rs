#![cfg(feature = "parametric")]
#![allow(clippy::needless_range_loop)]

mod commons;
use commons::*;

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use faer::Mat;
use rustc_hash::FxHashMap;

use manifolds_rs::prelude::*;
use manifolds_rs::*;

// Define the TestBackend type
type TestBackend = Autodiff<NdArray<f64>>;

fn fast_test_params_custom(
    n_dim: Option<usize>,
    n_neighbours: Option<usize>,
    min_dist: Option<f64>,
    spread: Option<f64>,
    hidden_layers: Vec<usize>,
    corr_weight: Option<f64>,
) -> ParametricUmapParams<f64> {
    let n_dim = n_dim.unwrap_or(2);
    let n_neighbours = n_neighbours.unwrap_or(15);
    let min_dist = min_dist.unwrap_or(0.1);
    let corr_weight = corr_weight.unwrap_or(0.0);
    let spread = spread.unwrap_or(1.0);

    let fit_params = TrainParametricParams::from_min_dist_spread(
        min_dist,
        spread,
        corr_weight,
        None,
        Some(10),
        Some(50),
        None,
    );

    ParametricUmapParams::new(
        Some(n_dim),
        Some(n_neighbours),
        Some("annoy".into()),
        Some(hidden_layers),
        None,
        None,
        Some(fit_params),
    )
}

fn fast_test_params() -> ParametricUmapParams<f64> {
    fast_test_params_custom(Some(2), Some(15), Some(0.1), Some(1.0), vec![32], Some(0.0))
}

#[test]
fn parametric_01_comprehensive_quality() {
    let (data, labels) = create_diagnostic_data(20, 10, 42);
    let device = NdArrayDevice::Cpu;

    println!("\n=== PARAMETRIC TEST 1: Comprehensive Quality ===");
    println!("Data: {} samples, {} features", data.nrows(), data.ncols());
    println!("Training: 10 epochs, 32 hidden units, batch size 50");

    let params = fast_test_params();
    let embedding = parametric_umap::<f64, TestBackend>(data.as_ref(), &params, &device, 42, false);

    // Basic shape checks
    assert_eq!(embedding.len(), 2, "Should have 2 dimensions");
    assert_eq!(embedding[0].len(), 100, "Should have 100 samples");
    println!(
        "✓ Embedding shape: {} dimensions × {} samples",
        embedding.len(),
        embedding[0].len()
    );

    // Check for finite values
    let mut has_nan = false;
    let mut has_inf = false;
    for dim in 0..2 {
        for i in 0..embedding[dim].len() {
            if embedding[dim][i].is_nan() {
                has_nan = true;
            }
            if embedding[dim][i].is_infinite() {
                has_inf = true;
            }
        }
    }
    assert!(!has_nan, "Embedding contains NaN values");
    assert!(!has_inf, "Embedding contains infinite values");
    println!("✓ All coordinates are finite");

    // Coordinate statistics
    for dim in 0..2 {
        let min = embedding[dim].iter().copied().fold(f64::INFINITY, f64::min);
        let max = embedding[dim]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean = embedding[dim].iter().sum::<f64>() / embedding[dim].len() as f64;
        println!(
            "  Dim {}: min = {:.3}, mean = {:.3}, max = {:.3}",
            dim, min, mean, max
        );
    }

    // Compute cluster centroids
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

    // Check inter-cluster separation
    println!("\nCluster analysis:");
    let mut min_inter_dist = f64::INFINITY;
    let mut max_inter_dist = f64::NEG_INFINITY;
    let mut avg_inter_dist = 0.0;
    let mut count = 0;

    for i in 0..centroids.len() {
        for j in (i + 1)..centroids.len() {
            let dist = ((centroids[i].1 - centroids[j].1).powi(2)
                + (centroids[i].2 - centroids[j].2).powi(2))
            .sqrt();
            min_inter_dist = min_inter_dist.min(dist);
            max_inter_dist = max_inter_dist.max(dist);
            avg_inter_dist += dist;
            count += 1;
        }
    }
    avg_inter_dist /= count as f64;

    println!(
        "  Inter-cluster distances: min = {:.3}, avg = {:.3}, max = {:.3}",
        min_inter_dist, avg_inter_dist, max_inter_dist
    );

    assert!(
        min_inter_dist > 0.5,
        "Clusters too close: min distance = {:.3}",
        min_inter_dist
    );

    // Check cluster connectivity
    for cluster_id in 0..5 {
        let points: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == cluster_id)
            .map(|(i, _)| i)
            .collect();

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

        let connectivity = reachable as f64 / points.len() as f64;
        println!(
            "  Cluster {}: {}/{} connected ({:.1}%)",
            cluster_id,
            reachable,
            points.len(),
            connectivity * 100.0
        );

        assert!(
            connectivity > 0.85,
            "Cluster {} fragmented: only {:.1}% connected",
            cluster_id,
            connectivity * 100.0
        );
    }

    // Compute intra-cluster compactness
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

    let separation_ratio = min_inter_dist / avg_intra_dist;
    println!("  Average intra-cluster distance: {:.3}", avg_intra_dist);
    println!("  Separation ratio (inter/intra): {:.2}", separation_ratio);

    assert!(
        separation_ratio > 0.3,
        "Poor separation: ratio = {:.2}",
        separation_ratio
    );

    println!("✓ All quality checks passed");
}

#[test]
fn parametric_02_different_dimensions() {
    let (data, _) = create_diagnostic_data(15, 10, 42);
    let device = NdArrayDevice::Cpu;

    println!("\n=== PARAMETRIC TEST 2: Different Output Dimensions ===");

    for n_dim in [2, 3, 5] {
        println!("\nTesting {} dimensions...", n_dim);

        let params = fast_test_params_custom(Some(n_dim), None, None, None, vec![32], None);
        let embedding =
            parametric_umap::<f64, TestBackend>(data.as_ref(), &params, &device, 42, false);

        assert_eq!(embedding.len(), n_dim, "Should have {} dimensions", n_dim);

        for dim in 0..n_dim {
            assert_eq!(
                embedding[dim].len(),
                75,
                "Dimension {} should have 75 samples",
                dim
            );

            let has_non_finite = embedding[dim].iter().any(|&x| !x.is_finite());
            assert!(!has_non_finite, "Dimension {} has non-finite values", dim);
        }

        println!("  ✓ {} dimensions: all finite, correct shape", n_dim);
    }
}

#[test]
fn parametric_03_different_architectures() {
    let (data, _) = create_diagnostic_data(15, 10, 42);
    let device = NdArrayDevice::Cpu;

    println!("\n=== PARAMETRIC TEST 3: Different Network Architectures ===");

    let layer_configs = vec![vec![32], vec![64, 32], vec![128, 64, 32]];

    for hidden_layers in layer_configs {
        println!("\nTesting architecture: {:?}", hidden_layers);

        let params = fast_test_params_custom(None, None, None, None, hidden_layers.clone(), None);

        let embedding =
            parametric_umap::<f64, TestBackend>(data.as_ref(), &params, &device, 42, false);

        assert_eq!(embedding.len(), 2);
        assert_eq!(embedding[0].len(), 75);

        let has_non_finite = embedding[0]
            .iter()
            .chain(&embedding[1])
            .any(|&x| !x.is_finite());
        assert!(
            !has_non_finite,
            "Architecture {:?} produced non-finite values",
            hidden_layers
        );

        println!("  ✓ Architecture {:?}: all finite", hidden_layers);
    }
}

#[test]
fn parametric_04_correlation_loss() {
    let (data, _) = create_diagnostic_data(15, 10, 42);
    let device = NdArrayDevice::Cpu;

    println!("\n=== PARAMETRIC TEST 4: Correlation Loss ===");

    let params = fast_test_params_custom(None, None, None, None, vec![32], Some(0.5));
    let embedding = parametric_umap::<f64, TestBackend>(data.as_ref(), &params, &device, 42, false);

    assert_eq!(embedding.len(), 2);
    assert_eq!(embedding[0].len(), 75);

    let has_non_finite = embedding[0]
        .iter()
        .chain(&embedding[1])
        .any(|&x| !x.is_finite());
    assert!(
        !has_non_finite,
        "Correlation loss produced non-finite values"
    );

    println!("  ✓ Correlation loss (λ=0.5): all finite");
}

#[test]
fn parametric_05_min_dist_spread() {
    let (data, _) = create_diagnostic_data(15, 10, 42);
    let device = NdArrayDevice::Cpu;

    println!("\n=== PARAMETRIC TEST 5: min_dist and spread ===");

    let configs = vec![(0.1, 1.0), (0.5, 1.0), (0.1, 2.0)];

    for (min_dist, spread) in configs {
        println!("\nTesting min_dist={}, spread={}...", min_dist, spread);

        let params =
            fast_test_params_custom(None, None, Some(min_dist), Some(spread), vec![32], None);

        let embedding =
            parametric_umap::<f64, TestBackend>(data.as_ref(), &params, &device, 42, false);

        let has_non_finite = embedding[0]
            .iter()
            .chain(&embedding[1])
            .any(|&x| !x.is_finite());
        assert!(
            !has_non_finite,
            "min_dist={}, spread={} produced non-finite values",
            min_dist, spread
        );

        println!("  ✓ min_dist={}, spread={}: all finite", min_dist, spread);
    }
}

#[test]
fn parametric_06_small_dataset() {
    let data = Mat::from_fn(10, 5, |i, j| (i as f64 + j as f64) * 0.1);
    let device = NdArrayDevice::Cpu;

    println!("\n=== PARAMETRIC TEST 6: Small Dataset ===");
    println!("Data: {} samples, {} features", data.nrows(), data.ncols());

    let params = fast_test_params_custom(None, Some(5), None, None, vec![32], None);

    let embedding = parametric_umap::<f64, TestBackend>(data.as_ref(), &params, &device, 42, false);

    assert_eq!(embedding.len(), 2);
    assert_eq!(embedding[0].len(), 10);

    let has_non_finite = embedding[0]
        .iter()
        .chain(&embedding[1])
        .any(|&x| !x.is_finite());
    assert!(!has_non_finite, "Small dataset produced non-finite values");

    println!("  ✓ Small dataset (10 samples): all finite");
}
