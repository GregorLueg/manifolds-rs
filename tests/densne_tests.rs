#![allow(clippy::needless_range_loop)]

mod commons;
use commons::*;

use faer::Mat;
use manifolds_rs::prelude::*;
use manifolds_rs::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

/////////////
// Helpers //
/////////////

/// Clusters with deliberately different spreads.
///
/// `create_diagnostic_data` uses one noise scale for every cluster, so it
/// cannot distinguish a density-preserving embedding from a plain one. Here
/// each cluster gets its own standard deviation, and the centres are pushed
/// far enough apart that the kNN graph stays within a cluster.
fn variable_density_clusters(
    n_per_cluster: usize,
    n_dim: usize,
    stds: &[f64],
    seed: u64,
) -> (Mat<f64>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_clusters = stds.len();
    let n_total = n_per_cluster * n_clusters;

    let mut data_vec = Vec::with_capacity(n_total * n_dim);
    let mut labels = Vec::with_capacity(n_total);

    for (cluster_id, &std) in stds.iter().enumerate() {
        // centres on the axes, spaced well beyond the widest cluster
        let mut centre = vec![0.0; n_dim];
        centre[cluster_id % n_dim] = 40.0 * (cluster_id + 1) as f64;

        for _ in 0..n_per_cluster {
            for dim in 0..n_dim {
                let noise: f64 = rng.random::<f64>() * 2.0 - 1.0;
                data_vec.push(centre[dim] + noise * std);
            }
            labels.push(cluster_id);
        }
    }

    let data = Mat::from_fn(n_total, n_dim, |i, j| data_vec[i * n_dim + j]);
    (data, labels)
}

/// Mean distance of each cluster's points from its own centroid, per cluster.
fn cluster_radii(embd: &[Vec<f64>], labels: &[usize], n_clusters: usize) -> Vec<f64> {
    let n = labels.len();

    let mut cx = vec![0.0; n_clusters];
    let mut cy = vec![0.0; n_clusters];
    let mut counts = vec![0.0; n_clusters];

    for i in 0..n {
        cx[labels[i]] += embd[0][i];
        cy[labels[i]] += embd[1][i];
        counts[labels[i]] += 1.0;
    }
    for c in 0..n_clusters {
        cx[c] /= counts[c];
        cy[c] /= counts[c];
    }

    let mut radii = vec![0.0; n_clusters];
    for i in 0..n {
        let c = labels[i];
        let dx = embd[0][i] - cx[c];
        let dy = embd[1][i] - cy[c];
        radii[c] += (dx * dx + dy * dy).sqrt();
    }
    for c in 0..n_clusters {
        radii[c] /= counts[c];
    }

    radii
}

/// Mean distance of each cluster's points from its centroid in the input space.
fn original_cluster_radii(data: &Mat<f64>, labels: &[usize], n_clusters: usize) -> Vec<f64> {
    let n = data.nrows();
    let dim = data.ncols();

    let mut centroids = vec![vec![0.0; dim]; n_clusters];
    let mut counts = vec![0.0; n_clusters];

    for i in 0..n {
        for d in 0..dim {
            centroids[labels[i]][d] += data[(i, d)];
        }
        counts[labels[i]] += 1.0;
    }
    for c in 0..n_clusters {
        for d in 0..dim {
            centroids[c][d] /= counts[c];
        }
    }

    let mut radii = vec![0.0; n_clusters];
    for i in 0..n {
        let c = labels[i];
        let mut acc = 0.0;
        for d in 0..dim {
            let diff = data[(i, d)] - centroids[c][d];
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

/// Test 1: The original local radii are finite and actually vary
#[test]
fn densne_integration_01_original_radii_are_informative() {
    let (data, labels) = variable_density_clusters(60, 10, &[0.2, 1.0, 4.0], 42);

    println!("\n=== den-SNE DIAGNOSTIC 1: Original Local Radii ===");

    let nn_params = NearestNeighbourParams::default();
    let (knn_indices, knn_dist) =
        run_ann_search(data.as_ref(), 45, "kmknn".to_string(), &nn_params, 42, 0).unwrap();

    let (graph, _, _) = construct_tsne_graph(
        data.as_ref(),
        Some((knn_indices.clone(), knn_dist.clone())),
        15.0,
        "kmknn".to_string(),
        &nn_params,
        42,
        0,
    )
    .unwrap();

    let state = DensState::new(
        DensParams::densne_default(),
        &graph,
        &knn_indices,
        &knn_dist,
    )
    .unwrap();

    assert!(
        state.r.iter().all(|x| x.is_finite()),
        "z-scored original radii must all be finite"
    );

    // per-cluster mean of the z-scored radius should track the cluster spread
    let mut per_cluster = [0.0; 3];
    let mut counts = [0.0; 3];
    for (i, &l) in labels.iter().enumerate() {
        per_cluster[l] += state.r[i];
        counts[l] += 1.0;
    }
    for c in 0..3 {
        per_cluster[c] /= counts[c];
        println!(
            "Cluster {} (std {}): mean R = {:.3}",
            c,
            [0.2, 1.0, 4.0][c],
            per_cluster[c]
        );
    }

    assert!(
        per_cluster[0] < per_cluster[1] && per_cluster[1] < per_cluster[2],
        "tighter clusters must get smaller original radii, got {:?}",
        per_cluster
    );

    println!("✓ Original local radii order the clusters by spread");
}

/// Test 2: lambda = 0 recovers plain t-SNE exactly
#[test]
fn densne_integration_02_zero_lambda_matches_tsne() {
    let (data, _) = create_diagnostic_data(40, 10, 42);

    println!("\n=== den-SNE DIAGNOSTIC 2: lambda = 0 Is Inert ===");

    let mut tsne_params = TsneParams {
        perplexity: 20.0,
        ..Default::default()
    };
    tsne_params.init_range = Some(1e-4);
    tsne_params.optim_params.lr = Some(200.0);
    tsne_params.optim_params.n_epochs = 200;

    let params = DensneParams::new(tsne_params.clone(), DensParams::new(Some(0.0), None, None));

    let plain = tsne(data.as_ref(), None, &tsne_params, "bh", 42, 0).unwrap();
    let dens = densne(data.as_ref(), None, &params, "bh", 42, 0).unwrap();

    let mut max_diff: f64 = 0.0;
    for i in 0..plain[0].len() {
        for dim in 0..2 {
            max_diff = max_diff.max((plain[dim][i] - dens[dim][i]).abs());
        }
    }

    println!("Max coordinate difference: {:.12}", max_diff);

    assert!(
        max_diff < 1e-6,
        "lambda = 0 must reproduce plain t-SNE, got diff = {}",
        max_diff
    );

    println!("✓ The density branch is inert when lambda = 0");
}

/// Test 3: den-SNE preserves relative cluster density better than t-SNE
#[test]
fn densne_integration_03_preserves_relative_density() {
    let (data, labels) = variable_density_clusters(60, 10, &[0.2, 1.0, 4.0], 42);

    println!("\n=== den-SNE DIAGNOSTIC 3: Density Preservation ===");

    let mut tsne_params = TsneParams {
        perplexity: 15.0,
        ..Default::default()
    };
    tsne_params.init_range = Some(1e-4);
    tsne_params.optim_params.lr = Some(100.0);
    tsne_params.optim_params.n_epochs = 750;

    let dens_params =
        DensneParams::new(tsne_params.clone(), DensParams::new(Some(2.0), None, None));

    let plain = tsne(data.as_ref(), None, &tsne_params, "bh", 42, 0).unwrap();
    let dens = densne(data.as_ref(), None, &dens_params, "bh", 42, 0).unwrap();

    let orig = original_cluster_radii(&data, &labels, 3);
    let r_plain = cluster_radii(&plain, &labels, 3);
    let r_dens = cluster_radii(&dens, &labels, 3);

    println!("Original radii:  {:?}", orig);
    println!("t-SNE radii:     {:?}", r_plain);
    println!("den-SNE radii:   {:?}", r_dens);

    // compare on the log scale, which is what the objective actually optimises
    let log = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| x.ln()).collect() };
    let corr_plain = pearson(&log(&orig), &log(&r_plain));
    let corr_dens = pearson(&log(&orig), &log(&r_dens));

    println!("log-radius correlation, t-SNE:   {:.4}", corr_plain);
    println!("log-radius correlation, den-SNE: {:.4}", corr_dens);

    assert!(
        corr_dens > corr_plain,
        "den-SNE must track the original densities better than t-SNE ({:.4} vs {:.4})",
        corr_dens,
        corr_plain
    );

    println!("✓ den-SNE improves the density correlation");
}

/// Test 4: den-SNE is reproducible with the same seed
#[test]
fn densne_integration_04_reproducibility() {
    let (data, _) = create_diagnostic_data(40, 10, 42);

    println!("\n=== den-SNE DIAGNOSTIC 4: Reproducibility ===");

    let mut params = DensneParams::<f64>::default();
    params.tsne_params.perplexity = 20.0;
    params.tsne_params.init_range = Some(1e-4);
    params.tsne_params.optim_params.lr = Some(200.0);
    params.tsne_params.optim_params.n_epochs = 300;

    let embd1 = densne(data.as_ref(), None, &params, "bh", 42, 0).unwrap();
    let embd2 = densne(data.as_ref(), None, &params, "bh", 42, 0).unwrap();

    let mut max_diff: f64 = 0.0;
    for i in 0..embd1[0].len() {
        for dim in 0..2 {
            max_diff = max_diff.max((embd1[dim][i] - embd2[dim][i]).abs());
        }
    }

    println!("Max coordinate difference: {:.10}", max_diff);

    assert!(
        max_diff < 1e-6,
        "den-SNE should be reproducible with the same seed, got diff = {}",
        max_diff
    );

    println!("✓ den-SNE is reproducible");
}

/// Test 5: Different seeds give different embeddings
#[test]
fn densne_integration_05_different_seeds_diverge() {
    let (data, _) = create_diagnostic_data(40, 10, 42);

    println!("\n=== den-SNE DIAGNOSTIC 5: Seed Sensitivity ===");

    let mut params = DensneParams::<f64>::default();
    params.tsne_params.perplexity = 20.0;
    params.tsne_params.initialisation = "random".to_string();
    params.tsne_params.optim_params.lr = Some(200.0);
    params.tsne_params.optim_params.n_epochs = 300;

    let embd1 = densne(data.as_ref(), None, &params, "bh", 42, 0).unwrap();
    let embd2 = densne(data.as_ref(), None, &params, "bh", 7, 0).unwrap();

    let mut max_diff: f64 = 0.0;
    for i in 0..embd1[0].len() {
        for dim in 0..2 {
            max_diff = max_diff.max((embd1[dim][i] - embd2[dim][i]).abs());
        }
    }

    println!("Max coordinate difference: {:.6}", max_diff);

    assert!(
        max_diff > 0.1,
        "different seeds should give different embeddings, got diff = {}",
        max_diff
    );

    println!("✓ den-SNE responds to the seed");
}

/// Test 6: Output is free of NaN and Inf
#[test]
fn densne_integration_06_output_is_finite() {
    let (data, _) = variable_density_clusters(50, 10, &[0.1, 2.0, 5.0], 7);

    println!("\n=== den-SNE DIAGNOSTIC 6: Numerical Health ===");

    // lambda well above the default, to stress the density term
    let mut params = DensneParams::<f64>::new_default_2d(Some(15.0), Some(10.0));
    params.tsne_params.optim_params.n_epochs = 500;

    let embd = densne(data.as_ref(), None, &params, "bh", 42, 0).unwrap();

    let bad = embd.iter().flatten().filter(|x| !x.is_finite()).count();

    println!("Non-finite coordinates: {}", bad);
    assert_eq!(bad, 0, "den-SNE produced {} non-finite coordinates", bad);

    println!("✓ den-SNE output is finite even at lambda = 10");
}

/// Test 7: Precomputed kNN matches the internal search
#[test]
fn densne_integration_07_precomputed_knn() {
    let (data, _) = create_diagnostic_data(40, 10, 42);

    println!("\n=== den-SNE DIAGNOSTIC 7: Precomputed kNN ===");

    let mut params = DensneParams::<f64>::default();
    params.tsne_params.perplexity = 20.0;
    params.tsne_params.init_range = Some(1e-4);
    params.tsne_params.optim_params.lr = Some(200.0);
    params.tsne_params.optim_params.n_epochs = 200;

    let k = (3.0 * 20.0_f64) as usize;
    let nn_params = NearestNeighbourParams::default();
    let knn = run_ann_search(data.as_ref(), k, "kmknn".to_string(), &nn_params, 42, 0).unwrap();

    let internal = densne(data.as_ref(), None, &params, "bh", 42, 0).unwrap();
    let external = densne(data.as_ref(), Some(knn), &params, "bh", 42, 0).unwrap();

    let mut max_diff: f64 = 0.0;
    for i in 0..internal[0].len() {
        for dim in 0..2 {
            max_diff = max_diff.max((internal[dim][i] - external[dim][i]).abs());
        }
    }

    println!("Max coordinate difference: {:.10}", max_diff);

    assert!(
        max_diff < 1e-6,
        "precomputed kNN should match the internal search, got diff = {}",
        max_diff
    );

    println!("✓ Precomputed kNN gives an identical embedding");
}

/// Test 8: den-SNE preserves relative density on the FFT path too
#[cfg(feature = "fft_tsne")]
#[test]
fn densne_integration_08_fft_preserves_relative_density() {
    let (data, labels) = variable_density_clusters(60, 10, &[0.2, 1.0, 4.0], 42);

    println!("\n=== den-SNE DIAGNOSTIC 8: Density Preservation (FFT) ===");

    let mut tsne_params = TsneParams {
        perplexity: 15.0,
        ..Default::default()
    };
    tsne_params.init_range = Some(1e-4);
    tsne_params.optim_params.lr = Some(100.0);
    tsne_params.optim_params.n_epochs = 750;

    let dens_params =
        DensneParams::new(tsne_params.clone(), DensParams::new(Some(2.0), None, None));

    let plain = tsne(data.as_ref(), None, &tsne_params, "fft", 42, 0).unwrap();
    let dens = densne(data.as_ref(), None, &dens_params, "fft", 42, 0).unwrap();

    let orig = original_cluster_radii(&data, &labels, 3);
    let r_plain = cluster_radii(&plain, &labels, 3);
    let r_dens = cluster_radii(&dens, &labels, 3);

    println!("Original radii:  {:?}", orig);
    println!("t-SNE radii:     {:?}", r_plain);
    println!("den-SNE radii:   {:?}", r_dens);

    let log = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| x.ln()).collect() };
    let corr_plain = pearson(&log(&orig), &log(&r_plain));
    let corr_dens = pearson(&log(&orig), &log(&r_dens));

    println!("log-radius correlation, t-SNE:   {:.4}", corr_plain);
    println!("log-radius correlation, den-SNE: {:.4}", corr_dens);

    assert!(
        corr_dens > corr_plain,
        "den-SNE (FFT) must track the original densities better than t-SNE ({:.4} vs {:.4})",
        corr_dens,
        corr_plain
    );

    println!("✓ den-SNE improves the density correlation on the FFT path");
}

/// Test 9: lambda = 0 recovers plain FFT t-SNE exactly
#[cfg(feature = "fft_tsne")]
#[test]
fn densne_integration_09_fft_zero_lambda_matches_tsne() {
    let (data, _) = create_diagnostic_data(40, 10, 42);

    println!("\n=== den-SNE DIAGNOSTIC 9: lambda = 0 Is Inert (FFT) ===");

    let mut tsne_params = TsneParams {
        perplexity: 20.0,
        ..Default::default()
    };
    tsne_params.init_range = Some(1e-4);
    tsne_params.optim_params.lr = Some(200.0);
    tsne_params.optim_params.n_epochs = 200;

    let params = DensneParams::new(tsne_params.clone(), DensParams::new(Some(0.0), None, None));

    let plain = tsne(data.as_ref(), None, &tsne_params, "fft", 42, 0).unwrap();
    let dens = densne(data.as_ref(), None, &params, "fft", 42, 0).unwrap();

    let mut max_diff: f64 = 0.0;
    for i in 0..plain[0].len() {
        for dim in 0..2 {
            max_diff = max_diff.max((plain[dim][i] - dens[dim][i]).abs());
        }
    }

    println!("Max coordinate difference: {:.12}", max_diff);

    assert!(
        max_diff < 1e-6,
        "lambda = 0 must reproduce plain FFT t-SNE, got diff = {}",
        max_diff
    );

    println!("✓ The FFT density branch is inert when lambda = 0");
}
