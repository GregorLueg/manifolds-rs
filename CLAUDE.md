# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Crate

`manifolds-rs` (lib name `manifolds_rs`, Rust 2021). Dimensionality-reduction algorithms: UMAP, tSNE (Barnes-Hut + optional FFT), PHATE, PaCMAP, Diffusion Maps, and optional parametric UMAP via `burn`. Powers the `manifoldsR` R package.

## Common commands

```sh
# Default build (CPU, no optional features)
cargo build --release

# Full-featured build
cargo build --release --features gpu,parametric,fft_tsne

# CI's exact test invocations
cargo test --release --no-default-features                  # CPU only, all 3 OSes
cargo test --release --features gpu,parametric,fft_tsne     # Full features, Linux/macOS

# Run a single test (test profile is opt-level=2, so tests take a while unbuilt)
cargo test --release --features <needed> <test_name>
cargo test --release --features gpu umap_gpu_integration_01_knn_correctness

# Large-scale diagnostics (dev-only feature, prints scaling metrics)
cargo test --release --features large_scale_diagnostics --test tsne_large_scale_diag

# Lint / format
cargo clippy --release --all-targets --features gpu,parametric,fft_tsne
cargo fmt

# Docs (fails on missing docs — `#![warn(missing_docs)]` in lib.rs)
cargo doc --no-deps --features gpu,parametric,fft_tsne
```

Linux GPU tests need Vulkan (`libvulkan1 mesa-vulkan-drivers libgl1-mesa-dri`) and `WGPU_BACKEND=vulkan`.

## Feature flags

- `parametric` — parametric UMAP via `burn` (+ `serde`, `bincode` for model I/O).
- `fft_tsne` — FFT-accelerated tSNE via `fftw` (system FFTW required).
- `gpu` — GPU kNN + GPU Adam UMAP via `cubecl` (wgpu/CUDA) and `ann-search-rs/gpu`. GPU code paths run in `f32` only (WGSL has no `f64`).
- `large_scale_diagnostics` — dev-only, gates `tests/tsne_large_scale_diag.rs`.

Feature-gated code, tests, and prelude re-exports are conditional on these; when adding a new symbol, mirror the `#[cfg(feature = "…")]` gating in `src/prelude.rs`.

## Architecture

`src/lib.rs` is the ~3.4k-line facade. Every user-facing algorithm has three things there: a `<Algo>Params<T>` struct (with `Default` and `new_default_2d`), an optional `construct_<algo>_graph` helper, and the entry-point free function (`umap`, `tsne`, `phate`, `pacmap`, `diffusion_maps`, `parametric_umap`, `umap_gpu`, `tsne_gpu`). Sub-modules hold the implementation:

- `src/data/` — graph construction and neighbour indices.
  - `nearest_neighbours.rs` / `nearest_neighbours_gpu.rs` — thin wrapper over `ann-search-rs`. Backends are picked by **string keys**: CPU = `"exhaustive"`, `"kmknn"`, `"balltree"`, `"annoy"`, `"nndescent"`, `"hnsw"`, `"ivf"`; GPU = `"exhaustive_gpu"`, `"ivf_gpu"`, `"nndescent_gpu"`.
  - `graph.rs` — fuzzy simplicial set / UMAP graph construction.
  - `init.rs` — spectral (Lanczos), random, PCA embedding initialisation.
  - `pacmap_pairs.rs` — near / mid-near / further pair sampling for PaCMAP.
  - `structures.rs` — `CoordinateList<T>` (COO) and other sparse structures.
  - `synthetic.rs` — swiss roll, clusters, branching trajectories used by tests and README examples.
- `src/training/` — optimisers. `umap_optimisers.rs` (SGD, Adam, parallel Adam), `umap_optimiser_gpu.rs` (GPU Adam), `tsne_optimiser.rs`, `pacmap_optimiser.rs`, `mds_optimiser.rs` (PHATE MDS). Shared Adam constants (`UMAP_BETA1`, `BETA1`, `EPS`, …) live in `training/mod.rs`.
- `src/utils/` — `bh_tree.rs` (Barnes-Hut quadtree), `fft.rs` (FFT-accelerated tSNE, `fft_tsne` only), `diffusions.rs` (PHATE + diffusion maps operators, VNE knee), `potentials.rs` (PHATE potential distance), `sparse_ops.rs`, `math.rs` (Lanczos eigenpairs, VNE), `traits.rs` (`ManifoldsFloat`, `ManifoldsFloatGpu`), `macros.rs`.
- `src/parametric/` (feature `parametric`) — `model.rs` (encoder + `TrainedUmapModel` with serde/bincode I/O), `dataset.rs`, `batch.rs`, `parametric_train.rs`.
- `src/errors.rs` — `ManifoldsError` (thiserror). Prefer returning errors over panics; recent versions have been migrating panics to variants here.
- `src/prelude.rs` — re-exports and the `Verbosity` enum (`parse_verbosity_level(0|1|2)`).

Numeric generics: everything flows through `T: ManifoldsFloat` (CPU) or `T: ManifoldsFloatGpu` (GPU-side). GPU functions are additionally generic over a `cubecl::Runtime` (`WgpuRuntime`, `CudaRuntime`). Linear algebra is `faer`; parallelism is `rayon`. When a fix requires wider precision, the existing convention (see tSNE) is to cast to `f64` at the numerically sensitive step, not to widen the whole pipeline.

Pre-computed kNN is a first-class shortcut: every entry point accepts `Option<(Vec<Vec<usize>>, Vec<Vec<T>>)>` so callers running many embeddings can compute the graph once via `run_ann_search`.

## Tests

Integration tests live in `tests/*.rs` (there are no `#[cfg(test)]` unit tests inside `src/` beyond a couple of macros). Shared helpers in `tests/commons/mod.rs`: `create_diagnostic_data` (clustered f64) and `mat_to_f32` (for GPU tests). GPU/parametric/fft_tsne tests are guarded with `#![cfg(feature = "…")]` at file top, so they compile away when the feature is off.
