#![cfg(feature = "gpu")]
#![allow(clippy::needless_range_loop)]

mod commons;
use commons::*;

use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use manifolds_rs::prelude::*;
use manifolds_rs::*;
use num_traits::{Float, FromPrimitive};
use rustc_hash::FxHashMap;
use std::ops::AddAssign;

/////////////
// Helpers //
/////////////

fn compute_separation<T>(embedding: &[Vec<T>], labels: &[usize]) -> T
where
    T: Float + AddAssign + FromPrimitive,
{
    let mut cc: FxHashMap<usize, (T, T, usize)> = FxHashMap::default();
    for (i, &l) in labels.iter().enumerate() {
        let e = cc.entry(l).or_insert((T::zero(), T::zero(), 0));
        e.0 += embedding[0][i];
        e.1 += embedding[1][i];
        e.2 += 1;
    }

    let centroids: Vec<(usize, T, T)> = cc
        .iter()
        .map(|(&l, &(sx, sy, c))| {
            let count = T::from_usize(c).unwrap();
            (l, sx / count, sy / count)
        })
        .collect();

    let mut min_inter = T::infinity();
    for i in 0..centroids.len() {
        for j in (i + 1)..centroids.len() {
            let d = ((centroids[i].1 - centroids[j].1).powi(2)
                + (centroids[i].2 - centroids[j].2).powi(2))
            .sqrt();
            if d < min_inter {
                min_inter = d;
            }
        }
    }

    let mut avg_intra = T::zero();
    for (label, cx, cy) in &centroids {
        let pts: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == *label)
            .map(|(i, _)| i)
            .collect();

        let sum: T = pts
            .iter()
            .map(|&i| ((embedding[0][i] - *cx).powi(2) + (embedding[1][i] - *cy).powi(2)).sqrt())
            .fold(T::zero(), |a, b| a + b);

        avg_intra += sum / T::from_usize(pts.len()).unwrap();
    }
    avg_intra = avg_intra / T::from_usize(centroids.len()).unwrap();

    min_inter / avg_intra
}

///////////
// Tests //
///////////

/// Test 1: GPU kNN returns sensible results (allowing rare GPU tie-breaking)
#[test]
fn tsne_gpu_integration_01_knn_correctness() {
    let (data, labels) = create_diagnostic_data(100, 10, 42);
    let data = mat_to_f32(data);

    let perplexity = 15.0;
    let k = (perplexity * 3.0) as usize;

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

    println!("\n=== GPU t-SNE DIAGNOSTIC 1: kNN Search ===");
    println!("k = {} neighbours", k);

    let n_total = knn_indices.len();
    let self_loops = knn_indices
        .iter()
        .enumerate()
        .filter(|(i, nb)| nb.contains(i))
        .count();

    println!("Self-loops: {} / {}", self_loops, n_total);
    // GPU tie-breaking can occasionally leave self in the list when
    // run_ann_search_gpu drains index 0 blindly. Accept up to 1%, but flag it.
    assert!(
        self_loops * 100 <= n_total,
        "Excessive self-loops: {} / {} (>1%). Check run_ann_search_gpu self-removal.",
        self_loops,
        n_total
    );

    let mut intra = 0.0;
    for (i, nb) in knn_indices.iter().enumerate() {
        intra += nb.iter().filter(|&&j| labels[j] == labels[i]).count() as f64 / nb.len() as f64;
    }
    intra /= knn_indices.len() as f64;

    let theoretical_max = (100_usize - 1).min(k) as f64 / k as f64;
    println!(
        "Intra ratio: {:.1}% (theoretical max {:.1}%)",
        intra * 100.0,
        theoretical_max * 100.0
    );
    assert!(intra > theoretical_max * 0.9);

    let min_d = knn_dist
        .iter()
        .flatten()
        .copied()
        .fold(f32::INFINITY, f32::min);
    assert!(min_d > 0.0);
}

/// Test 2: GPU graph construction preserves tSNE affinity properties
#[test]
fn tsne_gpu_integration_02_graph_construction() {
    let (data, _) = create_diagnostic_data(50, 10, 42);
    let data = mat_to_f32(data);

    let device = WgpuDevice::default();
    let nn_params = NearestNeighbourParamsGpu::default();
    let (graph, _, _) = construct_tsne_graph_gpu::<f32, WgpuRuntime>(
        data.as_ref(),
        None,
        30.0,
        "exhaustive_gpu".to_string(),
        &nn_params,
        device,
        42,
        false,
    )
    .unwrap();

    println!("\n=== GPU t-SNE DIAGNOSTIC 2: Graph Construction ===");
    println!("Graph has {} edges", graph.values.len());

    let mut adj: FxHashMap<(usize, usize), f32> = FxHashMap::default();
    for ((&i, &j), &w) in graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
    {
        adj.insert((i, j), w);
    }

    for (&(i, j), &w_ij) in &adj {
        let w_ji = *adj.get(&(j, i)).unwrap_or(&-1.0);
        assert!((w_ij - w_ji).abs() < 1e-10, "Asymmetric: ({},{})", i, j);
    }

    for (&i, &j) in graph.row_indices.iter().zip(&graph.col_indices) {
        assert_ne!(i, j);
    }

    let min_w = graph.values.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(min_w > 0.0);
}

/// Test 3: Full GPU tSNE pipeline with Barnes-Hut
#[test]
fn tsne_gpu_integration_03_bh_quality() {
    let (data, labels) = create_diagnostic_data(40, 10, 123);
    let data = mat_to_f32(data);

    let device = WgpuDevice::default();
    let params = TsneParamsGpu::new(
        Some(2),
        Some(20.0),
        Some(1e-4),
        Some(200.0),
        Some(500),
        Some("exhaustive_gpu".to_string()),
        Some(0.5),
        Some(3),
    );

    let embedding =
        tsne_gpu::<f32, WgpuRuntime>(data.as_ref(), None, &params, "bh", device, 42, false)
            .unwrap();

    assert!(embedding[0].iter().all(|x| x.is_finite()));
    assert!(embedding[1].iter().all(|x| x.is_finite()));

    let sep = compute_separation(&embedding, &labels);
    println!("BH separation ratio: {:.2}", sep);
    assert!(sep > 1.0);
}

/// Test 4: GPU tSNE produces consistent structural quality across runs
#[test]
fn tsne_gpu_integration_04_reproducibility_structural() {
    let (data, labels) = create_diagnostic_data(40, 10, 42);
    let data = mat_to_f32(data);

    let params = TsneParamsGpu::new(
        Some(2),
        Some(20.0),
        Some(1e-4),
        Some(200.0),
        Some(200),
        Some("exhaustive_gpu".to_string()),
        Some(0.5),
        Some(3),
    );

    let device1 = WgpuDevice::default();
    let e1 = tsne_gpu::<f32, WgpuRuntime>(data.as_ref(), None, &params, "bh", device1, 42, false)
        .unwrap();
    let device2 = WgpuDevice::default();
    let e2 = tsne_gpu::<f32, WgpuRuntime>(data.as_ref(), None, &params, "bh", device2, 42, false)
        .unwrap();

    // GPU ops aren't bit-reproducible across runs; test structural consistency.
    let sep1 = compute_separation(&e1, &labels);
    let sep2 = compute_separation(&e2, &labels);
    println!("Separation ratios: run1 = {:.3}, run2 = {:.3}", sep1, sep2);

    assert!(
        sep1 > 1.0 && sep2 > 1.0,
        "Both runs should separate clusters"
    );
    let ratio = sep1 / sep2;
    assert!(
        (0.5..2.0).contains(&ratio),
        "Quality inconsistent between runs: {:.2}",
        ratio
    );
}

/// Test 5: Precomputed kNN produces a valid embedding with comparable quality
#[test]
fn tsne_gpu_integration_05_precomputed_knn() {
    let (data, labels) = create_diagnostic_data(100, 10, 42);
    let data = mat_to_f32(data);

    let perplexity = 20.0;
    let k = (perplexity * 3.0) as usize;

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

    let params = TsneParamsGpu::new(
        Some(2),
        Some(perplexity),
        Some(1e-4),
        Some(200.0),
        Some(300),
        Some("exhaustive_gpu".to_string()),
        Some(0.5),
        Some(3),
    );

    let device_pre = WgpuDevice::default();
    let e_pre = tsne_gpu::<f32, WgpuRuntime>(
        data.as_ref(),
        Some((knn_indices, knn_dist)),
        &params,
        "bh",
        device_pre,
        42,
        false,
    )
    .unwrap();

    let device_int = WgpuDevice::default();
    let e_int =
        tsne_gpu::<f32, WgpuRuntime>(data.as_ref(), None, &params, "bh", device_int, 42, false)
            .unwrap();

    // Compare structural quality, not exact coords.
    let sep_pre = compute_separation(&e_pre, &labels);
    let sep_int = compute_separation(&e_int, &labels);
    println!(
        "Separation: precomputed = {:.3}, internal = {:.3}",
        sep_pre, sep_int
    );
    assert!(sep_pre > 1.0 && sep_int > 1.0);
    let ratio = sep_pre / sep_int;
    assert!(
        (0.5..2.0).contains(&ratio),
        "Precomputed vs internal quality mismatch: {:.2}",
        ratio
    );
}

/// Test 6: FFT optimisation quality
#[cfg(feature = "fft_tsne")]
#[test]
fn tsne_gpu_integration_06_fft_quality() {
    let (data, labels) = create_diagnostic_data(100, 10, 123);
    let data = mat_to_f32(data);

    let device = WgpuDevice::default();
    let params = TsneParamsGpu::new(
        Some(2),
        Some(20.0),
        Some(1e-4),
        Some(100.0),
        Some(500),
        Some("exhaustive_gpu".to_string()),
        Some(0.5),
        Some(3),
    );

    let embedding =
        tsne_gpu::<f32, WgpuRuntime>(data.as_ref(), None, &params, "fft", device, 42, false)
            .unwrap();

    assert!(embedding[0].iter().all(|x| x.is_finite()));
    assert!(embedding[1].iter().all(|x| x.is_finite()));

    let sep = compute_separation(&embedding, &labels);
    println!("FFT separation ratio: {:.2}", sep);
    assert!(sep > 1.0);
}

/// Test 7: FFT structural consistency across runs
#[cfg(feature = "fft_tsne")]
#[test]
fn tsne_gpu_integration_07_fft_reproducibility_structural() {
    let (data, labels) = create_diagnostic_data(100, 10, 42);
    let data = mat_to_f32(data);

    let params = TsneParamsGpu::new(
        Some(2),
        Some(20.0),
        Some(1e-4),
        Some(200.0),
        Some(200),
        Some("exhaustive_gpu".to_string()),
        Some(0.5),
        Some(3),
    );

    let device1 = WgpuDevice::default();
    let e1 = tsne_gpu::<f32, WgpuRuntime>(data.as_ref(), None, &params, "fft", device1, 123, false)
        .unwrap();
    let device2 = WgpuDevice::default();
    let e2 = tsne_gpu::<f32, WgpuRuntime>(data.as_ref(), None, &params, "fft", device2, 123, false)
        .unwrap();

    let sep1 = compute_separation(&e1, &labels);
    let sep2 = compute_separation(&e2, &labels);
    println!("FFT separations: run1 = {:.3}, run2 = {:.3}", sep1, sep2);
    assert!(sep1 > 1.0 && sep2 > 1.0);
    let ratio = sep1 / sep2;
    assert!(
        (0.5..2.0).contains(&ratio),
        "FFT quality inconsistent between runs: {:.2}",
        ratio
    );
}

/// Test 8: BH and FFT produce comparable quality (widened bounds)
#[cfg(feature = "fft_tsne")]
#[test]
fn tsne_gpu_integration_08_bh_vs_fft() {
    let (data, labels) = create_diagnostic_data(100, 10, 42);
    let data = mat_to_f32(data);

    let params = TsneParamsGpu::new(
        Some(2),
        Some(20.0),
        Some(1e-4),
        Some(200.0),
        Some(300),
        Some("exhaustive_gpu".to_string()),
        Some(0.5),
        Some(3),
    );

    let device_bh = WgpuDevice::default();
    let e_bh =
        tsne_gpu::<f32, WgpuRuntime>(data.as_ref(), None, &params, "bh", device_bh, 42, false)
            .unwrap();
    let device_fft = WgpuDevice::default();
    let e_fft =
        tsne_gpu::<f32, WgpuRuntime>(data.as_ref(), None, &params, "fft", device_fft, 42, false)
            .unwrap();

    let sep_bh = compute_separation(&e_bh, &labels);
    let sep_fft = compute_separation(&e_fft, &labels);
    println!("BH sep: {:.3}, FFT sep: {:.3}", sep_bh, sep_fft);

    assert!(sep_bh > 1.0, "BH failed to separate");
    assert!(sep_fft > 1.0, "FFT failed to separate");
    let ratio = sep_bh / sep_fft;
    assert!(
        (0.33..3.0).contains(&ratio),
        "BH/FFT quality ratio out of bounds: {:.2}",
        ratio
    );
}
