#![cfg(feature = "gpu")]
#![allow(clippy::needless_range_loop)]

mod commons;
use commons::*;

use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use manifolds_rs::prelude::*;
use manifolds_rs::*;
use rustc_hash::FxHashMap;

/// Test 1: GPU kNN search returns sensible neighbours (no self, intra-cluster)
#[test]
fn umap_gpu_integration_01_knn_correctness() {
    let (data, labels) = create_diagnostic_data(50, 10, 42);
    let data = mat_to_f32(data);

    let k = 15;

    let device = WgpuDevice::default();
    let nn_params = NearestNeighbourParamsGpu::default();
    let (knn_indices, knn_dist) = run_ann_search_gpu::<f32, WgpuRuntime>(
        data.as_ref(),
        k,
        "exhaustive_gpu".to_string(),
        &nn_params,
        device,
        42,
        false,
    );

    println!("\n=== GPU UMAP DIAGNOSTIC 1: kNN Search Correctness ===");
    println!("Returned {} neighbours per point", knn_indices[0].len());

    let mut self_in_neighbours = 0;
    for (i, neighbours) in knn_indices.iter().enumerate() {
        if neighbours.contains(&i) {
            self_in_neighbours += 1;
        }
    }
    assert_eq!(self_in_neighbours, 0, "Self should not be in neighbours");

    let mut intra_ratio = 0.0;
    for (i, neighbours) in knn_indices.iter().enumerate() {
        let same = neighbours
            .iter()
            .filter(|&&j| labels[j] == labels[i])
            .count();
        intra_ratio += same as f64 / neighbours.len() as f64;
    }
    intra_ratio /= knn_indices.len() as f64;
    println!("Intra-cluster ratio: {:.2}%", intra_ratio * 100.0);
    assert!(intra_ratio > 0.8);

    let all_dists: Vec<f32> = knn_dist.iter().flatten().copied().collect();
    let min_d = all_dists.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(min_d > 0.0, "Min distance should be > 0 (no self)");
}

/// Test 2: GPU ANN dispatch works for all three backends
#[test]
fn umap_gpu_integration_02_ann_dispatch() {
    let (data, _labels) = create_diagnostic_data(50, 10, 42);
    let data = mat_to_f32(data);

    let k = 15;
    let nn_params = NearestNeighbourParamsGpu::default();

    for ann_type in ["exhaustive_gpu", "ivf_gpu", "nndescent_gpu"] {
        let device = WgpuDevice::default();
        let (indices, dist) = run_ann_search_gpu::<f32, WgpuRuntime>(
            data.as_ref(),
            k,
            ann_type.to_string(),
            &nn_params,
            device,
            42,
            false,
        );
        assert_eq!(indices.len(), data.nrows());
        assert_eq!(indices[0].len(), k);
        assert_eq!(dist.len(), data.nrows());
        println!("{} dispatch ok", ann_type);
    }
}

/// Test 3: GPU graph construction produces connected intra-cluster edges
#[test]
fn umap_gpu_integration_03_graph_connectivity() {
    let (data, labels) = create_diagnostic_data(50, 10, 42);
    let data = mat_to_f32(data);

    let k = 15;

    let device = WgpuDevice::default();
    let nn_params = NearestNeighbourParamsGpu::default();
    let umap_params = UmapGraphParams::default();

    let (graph, _, _) = construct_umap_graph_gpu::<f32, WgpuRuntime>(
        data.as_ref(),
        None,
        k,
        "exhaustive_gpu".to_string(),
        &umap_params,
        &nn_params,
        500,
        device,
        42,
        false,
    );

    println!("\n=== GPU UMAP DIAGNOSTIC 3: Graph Connectivity ===");
    println!("Graph has {} edges", graph.values.len());

    let n = graph.n_samples;
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for (&i, &j) in graph.row_indices.iter().zip(&graph.col_indices) {
        adj[i].push(j);
    }

    for cluster_id in 0..5 {
        let cluster_points: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == cluster_id)
            .map(|(i, _)| i)
            .collect();

        let mut visited = vec![false; n];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(cluster_points[0]);
        visited[cluster_points[0]] = true;
        let mut reachable = 1;

        while let Some(node) = queue.pop_front() {
            for &neighbour in &adj[node] {
                if !visited[neighbour] && cluster_points.contains(&neighbour) {
                    visited[neighbour] = true;
                    queue.push_back(neighbour);
                    reachable += 1;
                }
            }
        }

        assert_eq!(
            reachable,
            cluster_points.len(),
            "Cluster {} fragmented: {}/{} reachable",
            cluster_id,
            reachable,
            cluster_points.len()
        );
    }
}

/// Test 4: Full GPU UMAP pipeline produces valid separated embedding
#[test]
fn umap_gpu_integration_04_optimisation_quality() {
    let (data, labels) = create_diagnostic_data(50, 10, 123);
    let data = mat_to_f32(data);

    println!("\n=== GPU UMAP DIAGNOSTIC 4: Optimisation Quality ===");

    let device = WgpuDevice::default();
    let params = UmapParamsGpu::new(
        Some(2),
        Some(15),
        Some("adam_parallel".to_string()),
        Some("exhaustive_gpu".to_string()),
        Some("spectral".to_string()),
        None,
        None,
        None,
        None,
        None,
    );

    let embedding =
        umap_gpu::<f32, WgpuRuntime>(data.as_ref(), None, &params, device, 42, false).unwrap();

    for i in 0..embedding[0].len() {
        assert!(
            embedding[0][i].is_finite() && embedding[1][i].is_finite(),
            "Non-finite coords at point {}",
            i
        );
    }

    let mut cluster_centres: FxHashMap<usize, (f32, f32, usize)> = FxHashMap::default();
    for (i, &label) in labels.iter().enumerate() {
        let e = cluster_centres.entry(label).or_insert((0.0, 0.0, 0));
        e.0 += embedding[0][i];
        e.1 += embedding[1][i];
        e.2 += 1;
    }

    let centroids: Vec<(usize, f32, f32)> = cluster_centres
        .iter()
        .map(|(&l, &(sx, sy, c))| (l, sx / c as f32, sy / c as f32))
        .collect();

    let mut min_inter = f32::INFINITY;
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
        let intra: f32 = pts
            .iter()
            .map(|&i| ((embedding[0][i] - cx).powi(2) + (embedding[1][i] - cy).powi(2)).sqrt())
            .sum::<f32>()
            / pts.len() as f32;
        avg_intra += intra;
    }
    avg_intra /= 5.0;

    println!(
        "Min inter-cluster: {:.3}, avg intra: {:.3}",
        min_inter, avg_intra
    );
    assert!(min_inter > 0.5);
    assert!(min_inter / avg_intra > 0.3);
}

/// Test 5: GPU UMAP is reproducible with same seed
#[test]
fn umap_gpu_integration_05_reproducibility() {
    let (data, _) = create_diagnostic_data(50, 10, 42);
    let data = mat_to_f32(data);

    let params = UmapParamsGpu::new(
        Some(2),
        Some(15),
        Some("adam_parallel".to_string()),
        Some("exhaustive_gpu".to_string()),
        Some("spectral".to_string()),
        None,
        None,
        None,
        None,
        None,
    );

    let device1 = WgpuDevice::default();
    let embd1 =
        umap_gpu::<f32, WgpuRuntime>(data.as_ref(), None, &params, device1, 42, false).unwrap();
    let device2 = WgpuDevice::default();
    let embd2 =
        umap_gpu::<f32, WgpuRuntime>(data.as_ref(), None, &params, device2, 42, false).unwrap();

    let mut max_diff: f32 = 0.0;
    for i in 0..embd1[0].len() {
        for dim in 0..2 {
            max_diff = max_diff.max((embd1[dim][i] - embd2[dim][i]).abs());
        }
    }
    println!("Max coordinate difference: {:.10}", max_diff);
    assert!(max_diff < 1e-6, "Not reproducible: max diff = {}", max_diff);
}

/// Test 6: Precomputed kNN produces identical embedding to internal kNN
#[test]
fn umap_gpu_integration_06_precomputed_knn() {
    let (data, _) = create_diagnostic_data(50, 10, 42);
    let data = mat_to_f32(data);

    let k = 15;

    let device = WgpuDevice::default();
    let nn_params = NearestNeighbourParamsGpu::default();
    let (knn_indices, knn_dist) = run_ann_search_gpu::<f32, WgpuRuntime>(
        data.as_ref(),
        k,
        "exhaustive_gpu".to_string(),
        &nn_params,
        device,
        42,
        false,
    );

    let params = UmapParamsGpu::new(
        Some(2),
        Some(k),
        Some("adam_parallel".to_string()),
        Some("exhaustive_gpu".to_string()),
        Some("spectral".to_string()),
        None,
        None,
        None,
        None,
        None,
    );

    let device_pre = WgpuDevice::default();
    let embd_pre = umap_gpu::<f32, WgpuRuntime>(
        data.as_ref(),
        Some((knn_indices.clone(), knn_dist.clone())),
        &params,
        device_pre,
        42,
        false,
    )
    .unwrap();

    let device_int = WgpuDevice::default();
    let embd_int =
        umap_gpu::<f32, WgpuRuntime>(data.as_ref(), None, &params, device_int, 42, false).unwrap();

    let mut max_diff: f32 = 0.0;
    for i in 0..embd_pre[0].len() {
        for dim in 0..2 {
            max_diff = max_diff.max((embd_pre[dim][i] - embd_int[dim][i]).abs());
        }
    }
    println!("Max diff precomputed vs internal: {:.10}", max_diff);
    assert!(max_diff < 1e-6);
}
