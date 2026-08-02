#![cfg(feature = "gpu")]
#![allow(clippy::needless_range_loop)]

mod commons;
use commons::*;

use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use faer::Mat;
use manifolds_rs::prelude::*;
use manifolds_rs::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

/////////////
// Helpers //
/////////////

/// Clusters with deliberately different spreads, as `f32` for the GPU path.
fn variable_density_clusters_f32(
    n_per_cluster: usize,
    n_dim: usize,
    stds: &[f64],
    seed: u64,
) -> (Mat<f32>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_clusters = stds.len();
    let n_total = n_per_cluster * n_clusters;

    let mut data_vec = Vec::with_capacity(n_total * n_dim);
    let mut labels = Vec::with_capacity(n_total);

    for (cluster_id, &std) in stds.iter().enumerate() {
        let mut centre = vec![0.0; n_dim];
        centre[cluster_id % n_dim] = 40.0 * (cluster_id + 1) as f64;

        for _ in 0..n_per_cluster {
            for dim in 0..n_dim {
                let noise: f64 = rng.random::<f64>() * 2.0 - 1.0;
                data_vec.push((centre[dim] + noise * std) as f32);
            }
            labels.push(cluster_id);
        }
    }

    let data = Mat::from_fn(n_total, n_dim, |i, j| data_vec[i * n_dim + j]);
    (data, labels)
}

/// Mean distance of each cluster's points from its own centroid in the
/// embedding.
fn cluster_radii(embd: &[Vec<f32>], labels: &[usize], n_clusters: usize) -> Vec<f64> {
    let n = labels.len();

    let mut cx = vec![0.0f64; n_clusters];
    let mut cy = vec![0.0f64; n_clusters];
    let mut counts = vec![0.0f64; n_clusters];

    for i in 0..n {
        cx[labels[i]] += embd[0][i] as f64;
        cy[labels[i]] += embd[1][i] as f64;
        counts[labels[i]] += 1.0;
    }
    for c in 0..n_clusters {
        cx[c] /= counts[c];
        cy[c] /= counts[c];
    }

    let mut radii = vec![0.0f64; n_clusters];
    for i in 0..n {
        let c = labels[i];
        let dx = embd[0][i] as f64 - cx[c];
        let dy = embd[1][i] as f64 - cy[c];
        radii[c] += (dx * dx + dy * dy).sqrt();
    }
    for c in 0..n_clusters {
        radii[c] /= counts[c];
    }

    radii
}

/// Mean distance of each cluster's points from its centroid in the input space.
fn original_cluster_radii(data: &Mat<f32>, labels: &[usize], n_clusters: usize) -> Vec<f64> {
    let n = data.nrows();
    let dim = data.ncols();

    let mut centroids = vec![vec![0.0f64; dim]; n_clusters];
    let mut counts = vec![0.0f64; n_clusters];

    for i in 0..n {
        for d in 0..dim {
            centroids[labels[i]][d] += data[(i, d)] as f64;
        }
        counts[labels[i]] += 1.0;
    }
    for c in 0..n_clusters {
        for d in 0..dim {
            centroids[c][d] /= counts[c];
        }
    }

    let mut radii = vec![0.0f64; n_clusters];
    for i in 0..n {
        let c = labels[i];
        let mut acc = 0.0;
        for d in 0..dim {
            let diff = data[(i, d)] as f64 - centroids[c][d];
            acc += diff * diff;
        }
        radii[c] += acc.sqrt();
    }
    for c in 0..n_clusters {
        radii[c] /= counts[c];
    }

    radii
}

/// Pearson correlation between two equal-length vectors.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let da = x - mean_a;
        let db = y - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    cov / (var_a.sqrt() * var_b.sqrt())
}

///////////
// Tests //
///////////

/// Test 1: The GPU density kernels leave the output finite
#[test]
fn densmap_gpu_integration_01_output_is_finite() {
    let (data, _) = variable_density_clusters_f32(50, 10, &[0.1, 2.0, 5.0], 7);

    println!("\n=== densMAP GPU DIAGNOSTIC 1: Numerical Health ===");

    // lambda well above the default, to stress the density term
    let mut params = DensmapParamsGpu::<f32>::new_default_2d(None, None, Some(20.0));
    params.umap_params.ann_type = "exhaustive_gpu".to_string();

    let embd =
        densmap_gpu::<f32, WgpuRuntime>(data.as_ref(), None, &params, WgpuDevice::default(), 42, 0)
            .unwrap();

    let bad = embd.iter().flatten().filter(|x| !x.is_finite()).count();

    println!("Non-finite coordinates: {}", bad);
    assert_eq!(
        bad, 0,
        "densMAP GPU produced {} non-finite coordinates",
        bad
    );

    println!("✓ GPU densMAP output is finite even at lambda = 20");
}

/// Test 2: lambda = 0 recovers plain GPU UMAP exactly
#[test]
fn densmap_gpu_integration_02_zero_lambda_matches_umap() {
    let (data, _) = create_diagnostic_data(40, 10, 42);
    let data = mat_to_f32(data);

    println!("\n=== densMAP GPU DIAGNOSTIC 2: lambda = 0 Is Inert ===");

    let umap_params = UmapParamsGpu {
        ann_type: "exhaustive_gpu".to_string(),
        ..UmapParamsGpu::<f32>::default()
    };
    let params = DensmapParamsGpu::new(
        umap_params.clone(),
        DensParams::new(Some(0.0f32), None, None),
    );

    let plain = umap_gpu::<f32, WgpuRuntime>(
        data.as_ref(),
        None,
        &umap_params,
        WgpuDevice::default(),
        42,
        0,
    )
    .unwrap();
    let dens =
        densmap_gpu::<f32, WgpuRuntime>(data.as_ref(), None, &params, WgpuDevice::default(), 42, 0)
            .unwrap();

    let mut max_diff: f32 = 0.0;
    for i in 0..plain[0].len() {
        for dim in 0..2 {
            max_diff = max_diff.max((plain[dim][i] - dens[dim][i]).abs());
        }
    }

    println!("Max coordinate difference: {:.10}", max_diff);

    assert!(
        max_diff < 1e-4,
        "lambda = 0 must reproduce plain GPU UMAP, got diff = {}",
        max_diff
    );

    println!("✓ The GPU density branch is inert when lambda = 0");
}

/// Test 3: GPU densMAP preserves relative cluster density better than GPU UMAP
#[test]
fn densmap_gpu_integration_03_preserves_relative_density() {
    let (data, labels) = variable_density_clusters_f32(60, 10, &[0.2, 1.0, 4.0], 42);

    println!("\n=== densMAP GPU DIAGNOSTIC 3: Density Preservation ===");

    let umap_params = UmapParamsGpu {
        ann_type: "exhaustive_gpu".to_string(),
        ..UmapParamsGpu::<f32>::default()
    };
    let dens_params = DensmapParamsGpu::new(
        umap_params.clone(),
        DensParams::new(Some(2.0f32), None, None),
    );

    let plain = umap_gpu::<f32, WgpuRuntime>(
        data.as_ref(),
        None,
        &umap_params,
        WgpuDevice::default(),
        42,
        0,
    )
    .unwrap();
    let dens = densmap_gpu::<f32, WgpuRuntime>(
        data.as_ref(),
        None,
        &dens_params,
        WgpuDevice::default(),
        42,
        0,
    )
    .unwrap();

    let orig = original_cluster_radii(&data, &labels, 3);
    let r_plain = cluster_radii(&plain, &labels, 3);
    let r_dens = cluster_radii(&dens, &labels, 3);

    println!("Original radii: {:?}", orig);
    println!("UMAP radii:     {:?}", r_plain);
    println!("densMAP radii:  {:?}", r_dens);

    let log = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| x.ln()).collect() };
    let corr_plain = pearson(&log(&orig), &log(&r_plain));
    let corr_dens = pearson(&log(&orig), &log(&r_dens));

    println!("log-radius correlation, UMAP:    {:.4}", corr_plain);
    println!("log-radius correlation, densMAP: {:.4}", corr_dens);

    assert!(
        corr_dens > corr_plain,
        "GPU densMAP must track the original densities better than GPU UMAP ({:.4} vs {:.4})",
        corr_dens,
        corr_plain
    );

    println!("✓ GPU densMAP improves the density correlation");
}

/// Test 4: GPU densMAP is reproducible with the same seed
#[test]
fn densmap_gpu_integration_04_reproducibility() {
    let (data, _) = create_diagnostic_data(40, 10, 42);
    let data = mat_to_f32(data);

    println!("\n=== densMAP GPU DIAGNOSTIC 4: Reproducibility ===");

    let mut params = DensmapParamsGpu::<f32>::default();
    params.umap_params.ann_type = "exhaustive_gpu".to_string();

    let embd1 =
        densmap_gpu::<f32, WgpuRuntime>(data.as_ref(), None, &params, WgpuDevice::default(), 42, 0)
            .unwrap();
    let embd2 =
        densmap_gpu::<f32, WgpuRuntime>(data.as_ref(), None, &params, WgpuDevice::default(), 42, 0)
            .unwrap();

    let mut max_diff: f32 = 0.0;
    for i in 0..embd1[0].len() {
        for dim in 0..2 {
            max_diff = max_diff.max((embd1[dim][i] - embd2[dim][i]).abs());
        }
    }

    println!("Max coordinate difference: {:.10}", max_diff);

    assert!(
        max_diff < 1e-4,
        "GPU densMAP should be reproducible with the same seed, got diff = {}",
        max_diff
    );

    println!("✓ GPU densMAP is reproducible");
}

/// Test 5: The GPU and CPU density paths agree on the same data
#[test]
fn densmap_gpu_integration_05_agrees_with_cpu() {
    let (data, labels) = variable_density_clusters_f32(60, 10, &[0.2, 1.0, 4.0], 42);

    println!("\n=== densMAP GPU DIAGNOSTIC 5: GPU vs CPU ===");

    let mut gpu_params = DensmapParamsGpu::<f32>::default();
    gpu_params.umap_params.ann_type = "exhaustive_gpu".to_string();

    let gpu = densmap_gpu::<f32, WgpuRuntime>(
        data.as_ref(),
        None,
        &gpu_params,
        WgpuDevice::default(),
        42,
        0,
    )
    .unwrap();

    let cpu_params = DensmapParams::<f32>::default();
    let cpu = densmap(data.as_ref(), None, &cpu_params, 42, 0).unwrap();

    let orig = original_cluster_radii(&data, &labels, 3);
    let log = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| x.ln()).collect() };

    let corr_gpu = pearson(&log(&orig), &log(&cluster_radii(&gpu, &labels, 3)));
    let corr_cpu = pearson(&log(&orig), &log(&cluster_radii(&cpu, &labels, 3)));

    println!("log-radius correlation, GPU: {:.4}", corr_gpu);
    println!("log-radius correlation, CPU: {:.4}", corr_cpu);

    // the two optimisers differ in their negative sampling, so the coordinates
    // never match; what must agree is that both actually preserve density
    assert!(
        corr_gpu > 0.9,
        "GPU densMAP density correlation too low: {:.4}",
        corr_gpu
    );
    assert!(
        (corr_gpu - corr_cpu).abs() < 0.15,
        "GPU and CPU densMAP should reach similar density correlations ({:.4} vs {:.4})",
        corr_gpu,
        corr_cpu
    );

    println!("✓ GPU and CPU densMAP agree on density preservation");
}
