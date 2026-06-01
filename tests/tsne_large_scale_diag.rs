#![cfg(feature = "large_scale_diagnostics")]
#![allow(clippy::needless_range_loop)]

//! Scaling diagnostics for the t-SNE optimisers.
//!
//! Generates `k` well-separated Gaussian blobs in high-dimensional space with
//! KNOWN labels, runs the full t-SNE pipeline (graph -> init -> optimise) at
//! increasing N, and measures whether the cluster structure survives.
//!
//! The headline metric is the separation ratio:
//!
//!   ratio = mean(inter-cluster pairwise distance) / mean(intra-cluster distance)
//!
//! Interpretation:
//! - ratio >> 1 and stable across N  -> healthy embedding, clusters separated
//! - ratio -> 1 with small radius     -> collapse (everything on the centroid)
//! - ratio -> 1 with large radius     -> uniform blob (repulsion ran away)
//!
//! Both the ratio and the embedding radius are printed so the two failure
//! modes are distinguishable. Run with:
//!
//!   cargo test --features large_scale_diagnostics tsne_scaling -- --nocapture

use faer::Mat;
use rand::prelude::*;
use rand_distr::Normal;

use manifolds_rs::prelude::*;
use manifolds_rs::*;

/// Generate `k` Gaussian blobs in `dim`-dimensional space.
///
/// Cluster centres are placed on a scaled hypercube so they are well
/// separated relative to the within-cluster spread. Returns the data matrix
/// (n_total x dim) and the per-row cluster label.
fn make_blobs(
    n_per_cluster: usize,
    k: usize,
    dim: usize,
    centre_spread: f64,
    cluster_std: f64,
    seed: u64,
) -> (Mat<f32>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::<f64>::new(0.0, cluster_std).unwrap();

    // distinct centre per cluster: random sign pattern scaled out
    let centres: Vec<Vec<f64>> = (0..k)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    if rng.random::<bool>() {
                        centre_spread
                    } else {
                        -centre_spread
                    }
                })
                .collect()
        })
        .collect();

    let n_total = n_per_cluster * k;
    let mut data = Mat::<f32>::zeros(n_total, dim);
    let mut labels = vec![0usize; n_total];

    let mut row = 0;
    for c in 0..k {
        for _ in 0..n_per_cluster {
            for d in 0..dim {
                let v = centres[c][d] + normal.sample(&mut rng);
                data[(row, d)] = v as f32;
            }
            labels[row] = c;
            row += 1;
        }
    }

    (data, labels)
}

/// Compute the separation ratio and embedding radius from a labelled
/// embedding. Distances are sampled (not all-pairs) to keep the metric O(samples)
/// rather than O(N^2).
fn separation_metrics(
    embd: &[Vec<f32>],
    labels: &[usize],
    n_samples: usize,
    seed: u64,
) -> (f64, f64, f64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = embd.len();

    let mut intra_sum = 0.0f64;
    let mut intra_cnt = 0usize;
    let mut inter_sum = 0.0f64;
    let mut inter_cnt = 0usize;

    for _ in 0..n_samples {
        let a = rng.random_range(0..n);
        let b = rng.random_range(0..n);
        if a == b {
            continue;
        }
        let dx = (embd[a][0] - embd[b][0]) as f64;
        let dy = (embd[a][1] - embd[b][1]) as f64;
        let dist = (dx * dx + dy * dy).sqrt();

        if labels[a] == labels[b] {
            intra_sum += dist;
            intra_cnt += 1;
        } else {
            inter_sum += dist;
            inter_cnt += 1;
        }
    }

    let intra = if intra_cnt > 0 {
        intra_sum / intra_cnt as f64
    } else {
        f64::NAN
    };
    let inter = if inter_cnt > 0 {
        inter_sum / inter_cnt as f64
    } else {
        f64::NAN
    };

    // embedding radius: max abs coordinate
    let radius = embd
        .iter()
        .flat_map(|p| p.iter())
        .fold(0.0f64, |acc, &v| acc.max((v as f64).abs()));

    let ratio = inter / intra.max(1e-12);
    (ratio, radius, intra)
}

/// Run the full t-SNE pipeline for one configuration and report metrics.
fn run_one(
    n_per_cluster: usize,
    k: usize,
    dim: usize,
    approx: &str,
    lr: Option<f32>,
    init_range: Option<f32>,
    seed: usize,
) {
    let (data, labels) = make_blobs(n_per_cluster, k, dim, 6.0, 1.0, seed as u64);
    let n = data.nrows();

    let optim_params = TsneOptimParams {
        lr,
        ..TsneOptimParams::default()
    };

    let params = TsneParams {
        n_dim: 2,
        perplexity: 30.0,
        ann_type: "kmknn".to_string(),
        initialisation: "pca".to_string(),
        init_range,
        nn_params: NearestNeighbourParams::default(),
        optim_params,
        randomised_init: true,
    };

    let embd_flat = tsne(data.as_ref(), None, &params, approx, seed, 0).unwrap();

    // tsne returns [n_dim][n_samples]; transpose to per-point for the metric
    let embd: Vec<Vec<f32>> = (0..n)
        .map(|i| vec![embd_flat[0][i], embd_flat[1][i]])
        .collect();

    let (ratio, radius, intra) = separation_metrics(&embd, &labels, 200_000, seed as u64);

    let lr_str = match lr {
        Some(v) => format!("{v:.0}"),
        None => "auto".to_string(),
    };
    let ir_str = match init_range {
        Some(v) => format!("{v:.0e}"),
        None => "default".to_string(),
    };

    println!(
        "N={n:>7} k={k} {approx:<3} lr={lr_str:<6} init={ir_str:<8} | sep_ratio={ratio:6.2} radius={radius:8.2} intra={intra:7.3}"
    );

    // a healthy embedding keeps clusters clearly separated
    assert!(
        ratio > 2.0,
        "N={n} {approx}: clusters not separated (sep_ratio={ratio:.2}); \
         radius={radius:.2} indicates {} ",
        if radius < 1.0 {
            "collapse to a point"
        } else {
            "a structureless blob"
        }
    );
}

/// Sweep N at fixed settings. This is the core diagnostic: if the optimiser is
/// scale-stable, sep_ratio stays well above 1 across all N.
#[test]
fn tsne_scaling_fft() {
    println!("\n=== FFT t-SNE scaling sweep (5 blobs, 50-dim) ===");
    let k = 5;
    let dim = 50;
    for &n_per in &[20_000usize, 50_000, 100_000] {
        run_one(n_per, k, dim, "fft", Some(200.0), Some(1e-4), 42);
    }
}

#[test]
fn tsne_scaling_bh() {
    println!("\n=== BH t-SNE scaling sweep (5 blobs, 50-dim) ===");
    let k = 5;
    let dim = 50;
    for &n_per in &[20_000usize, 50_000, 100_000] {
        run_one(n_per, k, dim, "bh", Some(200.0), Some(1e-4), 42);
    }
}

/// Grid over lr and init_range at a single mid-size N. This is the knob-finder:
/// it prints the metric for every combination so the working region is visible
/// rather than guessed.
#[test]
fn tsne_knob_grid() {
    println!("\n=== FFT t-SNE knob grid (5 blobs, 50-dim, N=250000) ===");
    let k = 5;
    let dim = 50;
    let n_per = 50_000; // N = 250k

    for &init_range in &[1e-4f32, 1e-2, 1.0] {
        for &lr in &[200.0f32, 1000.0, 5000.0] {
            // don't fail the grid on a bad combo; we want the whole map
            let res = std::panic::catch_unwind(|| {
                run_one(n_per, k, dim, "fft", Some(lr), Some(init_range), 42);
            });
            if res.is_err() {
                println!("  (combination failed the separation assertion)");
            }
        }
    }
}
