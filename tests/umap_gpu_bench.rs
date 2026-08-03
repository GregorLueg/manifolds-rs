#![cfg(all(feature = "gpu", feature = "large_scale_diagnostics"))]

//! Timing harness for the GPU UMAP Adam optimiser.
//!
//! Isolates `optimise_embedding_adam_gpu` from the rest of the pipeline: the
//! kNN graph is built once up front and reused across every timed repeat, and
//! the embedding is initialised randomly rather than spectrally, so what gets
//! measured is the epoch loop and nothing else.
//!
//! ```sh
//! cargo test --release --features gpu,large_scale_diagnostics \
//!   --test umap_gpu_bench -- --nocapture
//!
//! # the 1M case is #[ignore]d, it takes a while
//! cargo test --release --features gpu,large_scale_diagnostics \
//!   --test umap_gpu_bench -- --nocapture --ignored
//!
//! # per-kernel attribution
//! CUBECL_DEBUG_OPTION=profile-medium CUBECL_DEBUG_LOG=stdout \
//!   cargo test --release --features gpu,large_scale_diagnostics \
//!   --test umap_gpu_bench -- --nocapture
//! ```
//!
//! Never run two of these concurrently: parallel benchmark runs move the
//! numbers by more than most of the changes being measured.

mod commons;

use std::time::{Duration, Instant};

use ann_search_rs::gpu::grid_2d;
use cubecl::prelude::ComputeClient;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::Runtime;
use rand::{rngs::StdRng, Rng, SeedableRng};

use manifolds_rs::construct_umap_graph_gpu;
use manifolds_rs::data::graph::coo_to_adjacency_list;
use manifolds_rs::prelude::*;
use manifolds_rs::training::umap_optimiser_gpu::{
    optimise_embedding_adam_gpu, resolve_workgroup_size, UmapCsrGraph,
};

use commons::{create_diagnostic_data, mat_to_f32};

////////////
// Consts //
////////////

/// Feature dimensionality of the synthetic input data.
const N_FEATURES: usize = 32;

/// Neighbours per node in the kNN graph. Matches `UmapParamsGpu::default`.
const K: usize = 15;

/// Embedding dimensionality. The crate is designed for 2D embeddings and this
/// harness does not sweep it.
const N_DIM: usize = 2;

/// Epochs per timed run. Matches `UmapOptimParams::default_2d`.
const N_EPOCHS: usize = 500;

/// Timed repeats per configuration. Report best *and* worst: the first run
/// pays shader compilation and buffer-pool first touch, one-off costs that
/// every variant shares, so a single-shot number is a first-call number.
const REPEATS: usize = 3;

/// Half-width of the uniform random embedding initialisation.
const INIT_RANGE: f32 = 10.0;

/////////////
// Helpers //
/////////////

/// Uniform random embedding, `[n][N_DIM]`, in `[-INIT_RANGE, INIT_RANGE]`.
///
/// Spectral initialisation costs more than the thing being measured at these
/// sizes, and the optimiser does not care where it starts.
///
/// ### Params
///
/// * `n` - Number of nodes
/// * `seed` - Seed for the generator
///
/// ### Returns
///
/// Initial embedding as `[n][N_DIM]`.
fn random_embedding(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            (0..N_DIM)
                .map(|_| rng.random::<f32>() * 2.0 * INIT_RANGE - INIT_RANGE)
                .collect()
        })
        .collect()
}

/// Sum of absolute coordinates. Used as a cheap guard against the silent
/// failure mode of `launch_unchecked`: a kernel that busts a device limit does
/// no work, returns zeros and reports no error on any thread the caller can
/// see, which reads as an implausibly fast run rather than a crash.
///
/// ### Params
///
/// * `embd` - Embedding `[n][N_DIM]`
///
/// ### Returns
///
/// `sum_i sum_d |embd[i][d]|` in `f64`.
fn checksum(embd: &[Vec<f32>]) -> f64 {
    embd.iter()
        .flat_map(|p| p.iter())
        .map(|&x| x.abs() as f64)
        .sum()
}

/// Assert the optimiser actually ran and produced finite coordinates.
///
/// ### Params
///
/// * `before` - Embedding before optimisation
/// * `after` - Embedding after optimisation
fn assert_did_work(before: &[Vec<f32>], after: &[Vec<f32>]) {
    assert!(
        after.iter().flat_map(|p| p.iter()).all(|x| x.is_finite()),
        "embedding contains NaN or Inf"
    );
    let (cs_before, cs_after) = (checksum(before), checksum(after));
    assert!(
        cs_after > 0.0 && (cs_after - cs_before).abs() > 1e-3 * cs_before,
        "embedding barely moved (before {cs_before:.3e}, after {cs_after:.3e}): \
         a launch_unchecked kernel almost certainly did no work"
    );
}

/// Print the dispatch geometry every launcher in the optimiser derives from
/// the resolved workgroup size, and assert each grid fits the device limit.
///
/// The edge-schedule launch is the one that goes two-dimensional first, and a
/// busted dispatch limit is not a clean failure: wgpu rejects it on the cubecl
/// server thread, that thread dies, and the error then surfaces from an
/// unrelated later call with nothing pointing at the dispatch.
///
/// ### Params
///
/// * `client` - CubeCL compute client for the target device
/// * `n` - Number of nodes
/// * `n_edges` - Number of unique undirected edges
fn print_geometry(client: &ComputeClient<WgpuRuntime>, n: usize, n_edges: usize) {
    let wg = resolve_workgroup_size(client);
    let (max_x, max_y, _) = client.properties().hardware.max_cube_count;

    let grids = [
        ("grad", (n as u32).div_ceil(wg)),
        ("adam", ((n * N_DIM) as u32).div_ceil(wg)),
        ("schedule", (n_edges as u32).div_ceil(wg)),
    ];
    print!("  geometry (wg = {wg}, max cube count {max_x} x {max_y}):");
    for (name, cubes) in grids {
        let (gx, gy) = grid_2d(cubes);
        assert!(
            gx <= max_x && gy <= max_y,
            "{name} grid {gx} x {gy} exceeds the device limit {max_x} x {max_y}"
        );
        assert!(
            gx as u64 * gy as u64 >= cubes as u64,
            "{name} grid {gx} x {gy} does not cover {cubes} cubes"
        );
        print!(" {name} {cubes} cubes ({gx}, {gy});");
    }
    println!();
}

//////////////
// The runs //
//////////////

/// Build the graph once, then time the optimiser `REPEATS` times.
///
/// ### Params
///
/// * `label` - Human-readable name for the configuration
/// * `n_per_cluster` - Points per cluster; `create_diagnostic_data` makes five
///   clusters, so `n = 5 * n_per_cluster`
fn run_bench(label: &str, n_per_cluster: usize) {
    let device = WgpuDevice::DefaultDevice;

    let start_data = Instant::now();
    let (data, _) = create_diagnostic_data(n_per_cluster, N_FEATURES, 42);
    let data = mat_to_f32(data);
    let n = data.nrows();
    println!("\n=== {label}: n = {n}, d = {N_FEATURES}, k = {K} ===");
    println!("  data generated in {:.2?}", start_data.elapsed());

    let start_graph = Instant::now();
    let (graph, _, _) = construct_umap_graph_gpu::<f32, WgpuRuntime>(
        data.as_ref(),
        None,
        K,
        "nndescent_gpu".to_string(),
        &UmapGraphParams::default(),
        &NearestNeighbourParamsGpu::default(),
        N_EPOCHS,
        device.clone(),
        42,
        0,
    )
    .expect("graph construction failed");
    let graph_adj = coo_to_adjacency_list(&graph);
    println!(
        "  graph built in {:.2?} (setup, not timed)",
        start_graph.elapsed()
    );

    // The CSR build is pure host work that `optimise_embedding_adam_gpu` does
    // internally on every call. Timing it here separates the host share from
    // upload plus epoch loop plus readback.
    let start_csr = Instant::now();
    let csr = UmapCsrGraph::from_graph(&graph_adj).expect("CSR build failed");
    let csr_time = start_csr.elapsed();
    let n_edges = csr.n_edges;
    println!("  host CSR build: {csr_time:.2?} for {n_edges} edges");
    print_geometry(&WgpuRuntime::client(&device), n, n_edges);
    drop(csr);

    let params = UmapOptimParams::<f32> {
        n_epochs: N_EPOCHS,
        ..Default::default()
    };
    let init = random_embedding(n, 7);

    let mut times: Vec<Duration> = Vec::with_capacity(REPEATS);
    for rep in 0..REPEATS {
        let mut embd = init.clone();
        let start = Instant::now();
        optimise_embedding_adam_gpu::<WgpuRuntime, f32>(
            &mut embd,
            &graph_adj,
            &params,
            None,
            device.clone(),
            42,
            0,
        )
        .expect("optimisation failed");
        let elapsed = start.elapsed();
        assert_did_work(&init, &embd);
        println!("  repeat {}: {elapsed:.2?}", rep + 1);
        times.push(elapsed);
    }

    let best = times.iter().min().copied().unwrap_or_default();
    let worst = times.iter().max().copied().unwrap_or_default();
    let per_epoch = best.as_secs_f64() * 1e3 / N_EPOCHS as f64;
    println!(
        "  BEST {best:.2?} | WORST {worst:.2?} | {per_epoch:.3} ms/epoch \
         over {N_EPOCHS} epochs (host CSR {csr_time:.2?} of that)"
    );
}

///////////
// Tests //
///////////

#[test]
fn umap_gpu_bench_100k() {
    run_bench("100k", 20_000);
}

#[test]
#[ignore = "takes several minutes; run with -- --ignored"]
fn umap_gpu_bench_1m() {
    run_bench("1M", 200_000);
}
