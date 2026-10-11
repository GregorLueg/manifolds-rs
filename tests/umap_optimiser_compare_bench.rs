#![cfg(all(feature = "gpu", mlx_available, feature = "large_scale_diagnostics"))]

//! Timing harness: the UMAP Adam optimiser on CPU (rayon), wgpu (cubecl) and
//! MLX, on one shared kNN graph and one shared random initialisation.
//!
//! Only the optimiser call is timed. Each backend gets `REPEATS` runs; the
//! first pays shader compilation and buffer-pool first touch.
//!
//! ```sh
//! cargo test --release --features gpu,mlx,large_scale_diagnostics \
//!   --test umap_optimiser_compare_bench -- --nocapture --test-threads 1
//! ```

mod commons;

use std::time::{Duration, Instant};

use cubecl::wgpu::WgpuDevice;
use rand::{rngs::StdRng, Rng, SeedableRng};

use manifolds_rs::construct_umap_graph_gpu;
use manifolds_rs::data::graph::coo_to_adjacency_list;
use manifolds_rs::mlx::umap_optimiser_mlx::optimise_embedding_adam_mlx;
use manifolds_rs::prelude::*;
use manifolds_rs::training::umap_optimiser_gpu::optimise_embedding_adam_gpu;
use manifolds_rs::training::umap_optimisers::optimise_embedding_adam_parallel;

use commons::{create_diagnostic_data, mat_to_f32};

////////////
// Consts //
////////////

/// Feature dimensionality of the synthetic input data.
const N_FEATURES: usize = 32;

/// Neighbours per node in the kNN graph.
const K: usize = 15;

/// Epochs per timed run.
const N_EPOCHS: usize = 500;

/// Timed repeats per backend.
const REPEATS: usize = 3;

/////////////
// Helpers //
/////////////

/// Uniform random 2D embedding in `[-10, 10]`.
///
/// ### Params
///
/// * `n` - Number of nodes
/// * `seed` - Seed for the generator
///
/// ### Returns
///
/// Initial embedding as `[n][2]`.
fn random_embedding(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..2).map(|_| rng.random::<f32>() * 20.0 - 10.0).collect())
        .collect()
}

/// Ratio of the smallest centroid distance to the mean within-cluster spread.
/// A sanity check that every backend converged to a comparable layout, not a
/// quality benchmark.
///
/// ### Params
///
/// * `embd` - Embedding `[n][2]`
/// * `labels` - Cluster label per node, `0..5`
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
    let (graph, _, _) = construct_umap_graph_gpu::<f32>(
        data.as_ref(),
        None,
        K,
        "nndescent_gpu".to_string(),
        &UmapGraphParams::default(),
        &NearestNeighbourParamsGpu::default(),
        N_EPOCHS,
        device.clone().into(),
        42,
        0,
    )
    .expect("graph construction failed");
    let graph_adj = coo_to_adjacency_list(&graph);
    println!(
        "\n=== n = {n}, d = {N_FEATURES}, k = {K}, {N_EPOCHS} epochs, {} rayon threads (graph {:.2?}, not timed) ===",
        rayon::current_num_threads(),
        start.elapsed()
    );

    let params = UmapOptimParams::<f32> {
        n_epochs: N_EPOCHS,
        ..Default::default()
    };
    let init = random_embedding(n, 7);

    let cpu = time_backend("cpu", &init, &labels, |e| {
        optimise_embedding_adam_parallel(e, &graph_adj, &params, None, 42, 0).unwrap()
    });
    let wgpu = time_backend("wgpu", &init, &labels, |e| {
        optimise_embedding_adam_gpu::<f32>(
            e,
            &graph_adj,
            &params,
            None,
            device.clone().into(),
            42,
            0,
        )
        .unwrap()
    });
    let mlx = time_backend("mlx", &init, &labels, |e| {
        optimise_embedding_adam_mlx(e, &graph_adj, &params, None, 42, 0).unwrap()
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
fn umap_optimiser_compare_10k() {
    run_compare(2_000);
}

#[test]
fn umap_optimiser_compare_100k() {
    run_compare(20_000);
}

#[test]
#[ignore = "takes a while; run with -- --ignored"]
fn umap_optimiser_compare_1m() {
    run_compare(200_000);
}
