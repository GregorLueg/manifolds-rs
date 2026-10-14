#![cfg(feature = "gpu")]
#![allow(clippy::needless_range_loop)]

mod commons;
use commons::*;

use cubecl::wgpu::WgpuDevice;
use manifolds_rs::prelude::*;
use manifolds_rs::*;

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

    let embd = densmap_gpu::<f32>(
        data.as_ref(),
        None,
        &params,
        WgpuDevice::default().clone().into(),
        42,
        0,
    )
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

    let plain = umap_gpu::<f32>(
        data.as_ref(),
        None,
        &umap_params,
        WgpuDevice::default().clone().into(),
        42,
        0,
    )
    .unwrap();
    let dens = densmap_gpu::<f32>(
        data.as_ref(),
        None,
        &params,
        WgpuDevice::default().clone().into(),
        42,
        0,
    )
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

    let plain = umap_gpu::<f32>(
        data.as_ref(),
        None,
        &umap_params,
        WgpuDevice::default().clone().into(),
        42,
        0,
    )
    .unwrap();
    let dens = densmap_gpu::<f32>(
        data.as_ref(),
        None,
        &dens_params,
        WgpuDevice::default().clone().into(),
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

    let embd1 = densmap_gpu::<f32>(
        data.as_ref(),
        None,
        &params,
        WgpuDevice::default().clone().into(),
        42,
        0,
    )
    .unwrap();
    let embd2 = densmap_gpu::<f32>(
        data.as_ref(),
        None,
        &params,
        WgpuDevice::default().clone().into(),
        42,
        0,
    )
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

    let gpu = densmap_gpu::<f32>(
        data.as_ref(),
        None,
        &gpu_params,
        WgpuDevice::default().clone().into(),
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
