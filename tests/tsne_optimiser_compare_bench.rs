#![cfg(all(
    feature = "gpu",
    feature = "fft_tsne",
    mlx_available,
    feature = "large_scale_diagnostics"
))]

//! Timing harness: three-kernel FFT t-SNE on CPU (rustfft + rayon), wgpu
//! (cubecl) and MLX, on one shared affinity graph and one shared random
//! initialisation.
//!
//! Only the optimiser call is timed. Each backend gets `REPEATS` runs; the
//! first pays shader compilation and buffer-pool first touch.
//!
//! ```sh
//! cargo test --release --features gpu,fft_tsne,mlx,large_scale_diagnostics \
//!   --test tsne_optimiser_compare_bench -- --nocapture --test-threads 1
//! ```

mod commons;

use std::time::{Duration, Instant};

use cubecl::wgpu::WgpuDevice;
use rand::{rngs::StdRng, Rng, SeedableRng};

use manifolds_rs::construct_tsne_graph_gpu;
use manifolds_rs::mlx::tsne_optimiser_mlx::optimise_fft3k_tsne_mlx;
use manifolds_rs::prelude::*;
use manifolds_rs::training::tsne_optimiser::optimise_fft3k_tsne;
use manifolds_rs::training::tsne_optimiser_gpu::optimise_fft3k_tsne_gpu;

use commons::{create_diagnostic_data, mat_to_f32};

////////////
// Consts //
////////////

/// Feature dimensionality of the synthetic input data.
const N_FEATURES: usize = 32;

/// Perplexity of the affinity graph.
const PERPLEXITY: f32 = 30.0;

/// Epochs per timed run. Matches `TsneOptimParams::default`.
const N_EPOCHS: usize = 1000;

/// Timed repeats per backend.
const REPEATS: usize = 3;

/////////////
// Helpers //
/////////////

/// Uniform random 2D embedding in `[-1e-2, 1e-2]`, the usual t-SNE start
/// scale.
///
/// ### Params
///
/// * `n` - Number of points
/// * `seed` - Seed for the generator
///
/// ### Returns
///
/// Initial embedding as `[n][2]`.
fn random_embedding(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..2).map(|_| rng.random::<f32>() * 2e-2 - 1e-2).collect())
        .collect()
}

/// Ratio of the smallest centroid distance to the mean within-cluster spread.
/// A sanity check that every backend converged to a comparable layout.
///
/// ### Params
///
/// * `embd` - Embedding `[n][2]`
/// * `labels` - Cluster label per point, `0..5`
///
/// ### Returns
///
/// `min_inter / mean_intra`.
fn separation(embd: &[Vec<f32>], labels: &[usize]) -> f64 {
    let k = 5;
    let mut c = vec![[0.0f64; 3]; k];
    for (p, &l) in embd.iter().zip(labels) {
        c[l][0] += p[0] as f64;
        c[l][1] += p[1] as f64;
        c[l][2] += 1.0;
    }
    let cent: Vec<(f64, f64)> = c.iter().map(|v| (v[0] / v[2], v[1] / v[2])).collect();
    let mut intra = vec![0.0f64; k];
    for (p, &l) in embd.iter().zip(labels) {
        intra[l] += ((p[0] as f64 - cent[l].0).powi(2) + (p[1] as f64 - cent[l].1).powi(2)).sqrt();
    }
    let mean_intra = intra.iter().zip(&c).map(|(s, v)| s / v[2]).sum::<f64>() / k as f64;
    let mut min_inter = f64::INFINITY;
    for i in 0..k {
        for j in (i + 1)..k {
            let d = ((cent[i].0 - cent[j].0).powi(2) + (cent[i].1 - cent[j].1).powi(2)).sqrt();
            min_inter = min_inter.min(d);
        }
    }
    min_inter / mean_intra
}

/// Time one backend `REPEATS` times and print best / worst / separation.
///
/// ### Params
///
/// * `name` - Backend label
/// * `init` - Shared initial embedding
/// * `labels` - Cluster labels, for the separation check
/// * `run` - Runs the optimiser in place
///
/// ### Returns
///
/// The best time.
fn time_backend(
    name: &str,
    init: &[Vec<f32>],
    labels: &[usize],
    run: impl Fn(&mut Vec<Vec<f32>>),
) -> Duration {
    let mut times = Vec::with_capacity(REPEATS);
    let mut last = Vec::new();
    for _ in 0..REPEATS {
        let mut embd = init.to_vec();
        let start = Instant::now();
        run(&mut embd);
        times.push(start.elapsed());
        assert!(embd.iter().flatten().all(|x| x.is_finite()), "{name}: NaN");
        last = embd;
    }
    let best = *times.iter().min().unwrap();
    let worst = *times.iter().max().unwrap();
    println!(
        "  {name:<8} best {best:>9.2?} | worst {worst:>9.2?} | {:.3} ms/epoch | separation {:.2}",
        best.as_secs_f64() * 1e3 / N_EPOCHS as f64,
        separation(&last, labels)
    );
    best
}

/// Build the graph once, then time all three optimisers on it.
///
/// ### Params
///
/// * `n_per_cluster` - Points per cluster; five clusters
fn run_compare(n_per_cluster: usize) {
    let device = WgpuDevice::default();
    let (data, labels) = create_diagnostic_data(n_per_cluster, N_FEATURES, 42);
    let data = mat_to_f32(data);
    let n = data.nrows();

    let start = Instant::now();
    let (graph, _, _) = construct_tsne_graph_gpu::<f32>(
        data.as_ref(),
        None,
        PERPLEXITY,
        "nndescent_gpu".to_string(),
        &NearestNeighbourParamsGpu::default(),
        device.clone().into(),
        42,
        0,
    )
    .expect("graph construction failed");
    println!(
        "\n=== n = {n}, d = {N_FEATURES}, perplexity = {PERPLEXITY}, {N_EPOCHS} epochs, {} rayon threads (graph {:.2?}, not timed) ===",
        rayon::current_num_threads(),
        start.elapsed()
    );

    let params = TsneOptimParams::<f32> {
        n_epochs: N_EPOCHS,
        ..Default::default()
    };
    let init = random_embedding(n, 7);

    let cpu = time_backend("cpu", &init, &labels, |e| {
        optimise_fft3k_tsne(e, &params, &graph, None, None, 0).unwrap()
    });
    let wgpu = time_backend("wgpu", &init, &labels, |e| {
        optimise_fft3k_tsne_gpu::<f32>(e, &params, &graph, None, device.clone().into(), 0).unwrap()
    });
    let mlx = time_backend("mlx", &init, &labels, |e| {
        optimise_fft3k_tsne_mlx(e, &params, &graph, None, 0).unwrap()
    });
    println!(
        "  speed-up of mlx: {:.1}x over cpu, {:.1}x over wgpu",
        cpu.as_secs_f64() / mlx.as_secs_f64(),
        wgpu.as_secs_f64() / mlx.as_secs_f64()
    );
}

///////////
// Tests //
///////////

#[test]
fn tsne_optimiser_compare_10k() {
    run_compare(2_000);
}

#[test]
fn tsne_optimiser_compare_100k() {
    run_compare(20_000);
}

#[test]
#[ignore = "takes a while; run with -- --ignored"]
fn tsne_optimiser_compare_500k() {
    run_compare(100_000);
}

#[test]
#[ignore = "takes a while; run with -- --ignored"]
fn tsne_optimiser_compare_1m() {
    run_compare(200_000);
}
