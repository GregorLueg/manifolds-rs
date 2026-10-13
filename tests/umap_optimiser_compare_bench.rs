#![cfg(all(feature = "gpu", mlx_available, feature = "large_scale_diagnostics"))]

//! Timing harness: the kNN search on wgpu and MLX, and the UMAP Adam optimiser
//! on CPU (rayon), wgpu (cubecl) and MLX, on one shared kNN graph and one
//! shared random initialisation.
//!
//! The `knn_compare_*` tests run exhaustive, IVF and NN-Descent on both
//! devices at every input dimension in `KNN_DIMS`, `KNN_REPEATS` times each,
//! and score IVF and NN-Descent recall against the MLX exhaustive result. The
//! `umap_optimiser_compare_*` tests time only the optimiser call, each backend
//! `REPEATS` times. In both the first run pays shader compilation and
//! buffer-pool first touch, so the best run is reported.
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

use commons::{create_clustered_data_f32, create_diagnostic_data, mat_to_f32};

////////////
// Consts //
////////////

/// Feature dimensionality of the synthetic input data for the optimiser
/// tests.
const N_FEATURES: usize = 32;

/// Feature dimensionalities swept by the kNN tests.
const KNN_DIMS: [usize; 3] = [32, 128, 256];

/// Clusters in the kNN test data.
const KNN_CLUSTERS: usize = 25;

/// Neighbours per node in the kNN graph.
const K: usize = 15;

/// Epochs per timed run.
const N_EPOCHS: usize = 500;

/// Timed repeats per optimiser backend.
const REPEATS: usize = 3;

/// Timed repeats per kNN backend. Exhaustive at 1m is minutes per run.
const KNN_REPEATS: usize = 2;

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

/// Time one kNN backend `KNN_REPEATS` times.
///
/// ### Params
///
/// * `name` - Backend label
/// * `run` - Runs the search, returns the neighbour indices
///
/// ### Returns
///
/// The best time and the indices from the last run.
fn time_knn(name: &str, run: impl Fn() -> Vec<Vec<usize>>) -> (Duration, Vec<Vec<usize>>) {
    let mut times = Vec::with_capacity(KNN_REPEATS);
    let mut last = Vec::new();
    for _ in 0..KNN_REPEATS {
        let start = Instant::now();
        last = run();
        times.push(start.elapsed());
    }
    let best = *times.iter().min().unwrap();
    println!(
        "  {name:<16} best {best:>9.2?} | worst {:>9.2?}",
        times.iter().max().unwrap()
    );
    (best, last)
}

/// Fraction of the reference neighbours a search recovered.
///
/// ### Params
///
/// * `found` - Neighbour indices under test
/// * `truth` - Reference neighbour indices
///
/// ### Returns
///
/// Recall in `[0, 1]`.
fn recall(found: &[Vec<usize>], truth: &[Vec<usize>]) -> f64 {
    let hits: usize = found
        .iter()
        .zip(truth)
        .map(|(a, b)| a.iter().filter(|j| b.contains(j)).count())
        .sum();
    hits as f64 / (truth.len() * K) as f64
}

/// Time exhaustive, IVF and NN-Descent kNN on wgpu and MLX.
///
/// ### Params
///
/// * `data` - Input data, samples x features
/// * `device` - wgpu device
fn run_knn_compare(data: faer::MatRef<f32>, device: &WgpuDevice) {
    let nn_cpu = NearestNeighbourParams::<f32>::default();
    let nn_gpu = NearestNeighbourParamsGpu::<f32>::default();
    let wgpu = |ann: &str| {
        run_ann_search_gpu(data, K, ann.into(), &nn_gpu, device.clone().into(), 42, 0)
            .unwrap()
            .0
    };
    let mlx = |ann: &str| {
        run_ann_search(data, K, ann.into(), &nn_cpu, 42, 0)
            .unwrap()
            .0
    };

    println!("  -- kNN, k = {K} --");
    let (ex_wgpu, _) = time_knn("exhaustive wgpu", || wgpu("exhaustive_gpu"));
    let (ex_mlx, truth) = time_knn("exhaustive mlx", || mlx("exhaustive_mlx"));
    let (ivf_wgpu, ivf_wgpu_idx) = time_knn("ivf wgpu", || wgpu("ivf_gpu"));
    let (ivf_mlx, ivf_mlx_idx) = time_knn("ivf mlx", || mlx("ivf_mlx"));
    let (nd_wgpu, nd_wgpu_idx) = time_knn("nndescent wgpu", || wgpu("nndescent_gpu"));
    let (nd_mlx, nd_mlx_idx) = time_knn("nndescent mlx", || mlx("nndescent_mlx"));
    println!(
        "  exhaustive: mlx {:.1}x over wgpu | ivf: mlx {:.1}x over wgpu, recall wgpu {:.4}, mlx {:.4} | nndescent: mlx {:.1}x over wgpu, recall wgpu {:.4}, mlx {:.4}",
        ex_wgpu.as_secs_f64() / ex_mlx.as_secs_f64(),
        ivf_wgpu.as_secs_f64() / ivf_mlx.as_secs_f64(),
        recall(&ivf_wgpu_idx, &truth),
        recall(&ivf_mlx_idx, &truth),
        nd_wgpu.as_secs_f64() / nd_mlx.as_secs_f64(),
        recall(&nd_wgpu_idx, &truth),
        recall(&nd_mlx_idx, &truth)
    );
}

/// Run the kNN comparison at every input dimension in `KNN_DIMS`.
///
/// ### Params
///
/// * `n` - Number of points, split over `KNN_CLUSTERS` clusters
fn run_knn_sweep(n: usize) {
    let device = WgpuDevice::default();
    for d in KNN_DIMS {
        let (data, _) = create_clustered_data_f32(n / KNN_CLUSTERS, KNN_CLUSTERS, d, 42);
        println!(
            "\n=== kNN, n = {}, d = {d}, {KNN_CLUSTERS} clusters ===",
            data.nrows()
        );
        run_knn_compare(data.as_ref(), &device);
    }
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

    println!("\n=== n = {n}, d = {N_FEATURES} ===");

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
        "  -- optimiser, k = {K}, {N_EPOCHS} epochs, {} rayon threads (graph {:.2?}, not timed) --",
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
fn knn_compare_10k() {
    run_knn_sweep(10_000);
}

#[test]
fn knn_compare_100k() {
    run_knn_sweep(100_000);
}

#[test]
#[ignore = "takes a while; run with -- --ignored"]
fn knn_compare_500k() {
    run_knn_sweep(500_000);
}

#[test]
#[ignore = "takes a while; run with -- --ignored"]
fn knn_compare_1m() {
    run_knn_sweep(1_000_000);
}

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
fn umap_optimiser_compare_500k() {
    run_compare(100_000);
}

#[test]
#[ignore = "takes a while; run with -- --ignored"]
fn umap_optimiser_compare_1m() {
    run_compare(200_000);
}
