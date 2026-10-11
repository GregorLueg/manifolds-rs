#![cfg(mlx_available)]
#![allow(clippy::needless_range_loop)]

mod commons;
use commons::*;

use manifolds_rs::prelude::*;
use manifolds_rs::*;
use rustc_hash::FxHashMap;

/////////////
// Helpers //
/////////////

/// UMAP parameters with the MLX optimiser and an exhaustive kNN.
fn umap_params_mlx() -> UmapParams<f32> {
    UmapParams {
        optimiser: "adam_mlx".to_string(),
        ann_type: "exhaustive".to_string(),
        ..UmapParams::default()
    }
}

/// densMAP parameters with the MLX optimiser and an exhaustive kNN.
fn densmap_params_mlx(lambda: f32) -> DensmapParams<f32> {
    DensmapParams::new(umap_params_mlx(), DensParams::new(Some(lambda), None, None))
}

/// Largest absolute coordinate difference between two embeddings.
fn max_abs_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

///////////
// Tests //
///////////

/// Test 1: MLX UMAP separates the five diagnostic clusters
#[test]
fn umap_mlx_integration_01_optimisation_quality() {
    let (data, labels) = create_diagnostic_data(50, 10, 123);
    let data = mat_to_f32(data);

    let embedding = umap(data.as_ref(), None, &umap_params_mlx(), 42, 0).unwrap();
    assert!(embedding.iter().flatten().all(|x| x.is_finite()));

    let mut centres: FxHashMap<usize, (f32, f32, usize)> = FxHashMap::default();
    for (i, &label) in labels.iter().enumerate() {
        let e = centres.entry(label).or_insert((0.0, 0.0, 0));
        e.0 += embedding[0][i];
        e.1 += embedding[1][i];
        e.2 += 1;
    }
    let centroids: Vec<(usize, f32, f32)> = centres
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
        let pts: Vec<usize> = (0..labels.len()).filter(|&i| labels[i] == *label).collect();
        avg_intra += pts
            .iter()
            .map(|&i| ((embedding[0][i] - cx).powi(2) + (embedding[1][i] - cy).powi(2)).sqrt())
            .sum::<f32>()
            / pts.len() as f32;
    }
    avg_intra /= centroids.len() as f32;

    println!("Min inter-cluster: {min_inter:.3}, avg intra: {avg_intra:.3}");
    assert!(min_inter > 0.5);
    assert!(min_inter / avg_intra > 0.3);
}

/// Test 2: MLX UMAP is reproducible with the same seed
#[test]
fn umap_mlx_integration_02_reproducibility() {
    let (data, _) = create_diagnostic_data(40, 10, 42);
    let data = mat_to_f32(data);

    let a = umap(data.as_ref(), None, &umap_params_mlx(), 42, 0).unwrap();
    let b = umap(data.as_ref(), None, &umap_params_mlx(), 42, 0).unwrap();
    assert_eq!(a, b, "same seed, different MLX embedding");
}

/// Test 3: densMAP with lambda = 0 reproduces plain MLX UMAP
#[test]
fn umap_mlx_integration_03_zero_lambda_matches_umap() {
    let (data, _) = create_diagnostic_data(40, 10, 42);
    let data = mat_to_f32(data);

    let plain = umap(data.as_ref(), None, &umap_params_mlx(), 42, 0).unwrap();
    let dens = densmap(data.as_ref(), None, &densmap_params_mlx(0.0), 42, 0).unwrap();
    let diff = max_abs_diff(&plain, &dens);
    assert!(
        diff < 1e-4,
        "lambda = 0 must reproduce MLX UMAP, diff = {diff}"
    );
}

/// Test 4: MLX densMAP tracks the original cluster densities better than MLX
/// UMAP, and as well as the CPU densMAP
#[test]
fn umap_mlx_integration_04_preserves_relative_density() {
    let (data, labels) = variable_density_clusters_f32(60, 10, &[0.2, 1.0, 4.0], 42);

    let plain = umap(data.as_ref(), None, &umap_params_mlx(), 42, 0).unwrap();
    let dens = densmap(data.as_ref(), None, &densmap_params_mlx(2.0), 42, 0).unwrap();
    let cpu = densmap(data.as_ref(), None, &DensmapParams::<f32>::default(), 42, 0).unwrap();
    assert!(dens.iter().flatten().all(|x| x.is_finite()));

    let log = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| x.ln()).collect() };
    let orig = log(&original_cluster_radii(&data, &labels, 3));
    let corr_plain = pearson(&orig, &log(&cluster_radii(&plain, &labels, 3)));
    let corr_dens = pearson(&orig, &log(&cluster_radii(&dens, &labels, 3)));
    let corr_cpu = pearson(&orig, &log(&cluster_radii(&cpu, &labels, 3)));

    println!("log-radius correlation: UMAP {corr_plain:.4}, densMAP MLX {corr_dens:.4}, densMAP CPU {corr_cpu:.4}");
    assert!(corr_dens > corr_plain);
    assert!(corr_dens > 0.9);
    assert!((corr_dens - corr_cpu).abs() < 0.15);
}

/// Test 5: MLX densMAP is reproducible with the same seed
#[test]
fn umap_mlx_integration_05_densmap_reproducibility() {
    let (data, _) = create_diagnostic_data(40, 10, 42);
    let data = mat_to_f32(data);

    let a = densmap(data.as_ref(), None, &densmap_params_mlx(2.0), 42, 0).unwrap();
    let b = densmap(data.as_ref(), None, &densmap_params_mlx(2.0), 42, 0).unwrap();
    assert_eq!(a, b, "same seed, different MLX densMAP embedding");
}
