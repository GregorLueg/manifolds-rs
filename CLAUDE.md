# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Crate

`manifolds-rs` (lib name `manifolds_rs`, Rust 2021). Dimensionality-reduction algorithms: UMAP, densMAP, tSNE (Barnes-Hut + optional FFT), den-SNE, DREAMS, PHATE, PaCMAP, Diffusion Maps, ForceAtlas2 (Barnes-Hut), and optional parametric UMAP via `burn`. Powers the `manifoldsR` R package and the Python package in `python/`.

## Common commands

```sh
# Default build (CPU, no optional features)
cargo build --release

# Full-featured build
cargo build --release --features gpu,parametric,fft_tsne,ndarray

# CI's exact test invocations
cargo test --release --no-default-features                       # CPU only, all 3 OSes
cargo test --release --features gpu,parametric,fft_tsne,ndarray  # Full features, Linux/macOS

# Run a single test (test profile is opt-level=2, so tests take a while unbuilt)
cargo test --release --features <needed> <test_name>
cargo test --release --features gpu umap_gpu_integration_01_knn_correctness

# Large-scale diagnostics (dev-only feature, prints scaling metrics).
# Do not run these routinely: the CI invocations above are enough to validate a
# change. Run them only when asked, or when a change targets large-scale behaviour.
cargo test --release --features large_scale_diagnostics --test tsne_large_scale_diag
cargo test --release --features large_scale_diagnostics --test spectral_bench -- --nocapture
cargo test --release --features gpu,large_scale_diagnostics --test umap_gpu_bench -- --nocapture

# Lint / format
cargo clippy --release --all-targets --features gpu,parametric,fft_tsne,ndarray
cargo fmt

# Docs (warns on missing docs, `#![warn(missing_docs)]` in lib.rs)
cargo doc --no-deps --features gpu,parametric,fft_tsne,ndarray
```

Linux GPU tests need Vulkan (`libvulkan1 mesa-vulkan-drivers libgl1-mesa-dri`) and `WGPU_BACKEND=vulkan`.

## Feature flags

- `parametric`: parametric UMAP via `burn` (+ `serde`, `bincode` for model I/O).
- `fft_tsne`: FFT-accelerated tSNE via `rustfft` + `realfft` (pure Rust). `n_fft` is padded to the next `2^a` or `3 * 2^a` (`fft_len` in `utils/fft.rs`).
- `gpu`: GPU kNN + GPU Adam UMAP + device-resident three-kernel FFT tSNE + PHATE's spectral landmark selection via `cubecl` (wgpu/CUDA) and `ann-search-rs/gpu`. GPU code paths run in `f32` only (WGSL has no `f64`). `GpuTensor`, `GpuLimits`, `grid_2d`, `checked_cube_count`, `resolve_workgroup_size` and `CubeclFloat` come from `cubecl-utils-rs`; allocations and grid sizing are fallible and every limit is queried off the device, never assumed.
- `mlx`: experimental, Apple Silicon only, `f32` only. kNN, UMAP Adam, three-kernel FFT tSNE and PHATE's spectral landmark selection on MLX via mlx-c. `build.rs` sets `cfg(mlx_available)` only if it finds mlx-c, so MLX code gates on that cfg rather than on the feature, and `MLX_AVAILABLE` reports the result.
- `ndarray`: accept `ndarray` 2-D arrays wherever a faer matrix is accepted.
- `large_scale_diagnostics`: dev-only, gates `tests/tsne_large_scale_diag.rs`, all of `tests/umap_gpu_bench.rs` (which also needs `gpu`), the `tests/*_optimiser_compare_bench.rs` files (which also need `gpu` and MLX) and the timing benchmarks in `tests/spectral_bench.rs`. Put heavy tests here so CI never compiles them. Not part of routine test runs.

Feature-gated code, tests, and prelude re-exports are conditional on these; when adding a new symbol, mirror the `#[cfg(feature = "…")]` gating in `src/prelude.rs`.

## Architecture

`src/lib.rs` is the ~6k-line facade. Every user-facing algorithm has three things there: a `<Algo>Params<T>` struct (with `Default` and `new_default_2d`), an optional `construct_<algo>_graph` helper, and the entry-point free function (`umap`, `densmap`, `tsne`, `densne`, `dreams`, `phate`, `pacmap`, `diffusion_maps`, `forceatlas2`, `forceatlas2_from_graph`, `parametric_umap`, `train_parametric_umap_model`, `umap_gpu`, `densmap_gpu`, `tsne_gpu`, `dreams_gpu`, `phate_gpu`). Sub-modules hold the implementation:

- `src/data/`: graph construction and neighbour indices.
  - `nearest_neighbours.rs` / `nearest_neighbours_gpu.rs`: thin wrapper over `ann-search-rs`. Backends are picked by **string keys**: CPU = `"exhaustive"`, `"kmknn"`, `"balltree"`, `"annoy"`, `"nndescent"`, `"hnsw"`, `"ivf"`; GPU = `"exhaustive_gpu"`, `"ivf_gpu"`, `"nndescent_gpu"`.
  - `graph.rs`: fuzzy simplicial set / UMAP graph construction.
  - `init.rs`: spectral (Lanczos), random, PCA embedding initialisation.
  - `pacmap_pairs.rs`: near / mid-near / further pair sampling for PaCMAP.
  - `structures.rs`: `CoordinateList<T>` (COO) and other sparse structures.
  - `synthetic.rs`: swiss roll, clusters, branching trajectories used by tests and README examples.
- `src/training/`: optimisers. `umap_optimisers.rs` (SGD, Adam, parallel Adam), `umap_csr.rs` (the `u32` CSR graph they run on), `umap_optimiser_gpu.rs` (GPU Adam), `tsne_optimiser.rs`, `tsne_optimiser_gpu.rs` (device-resident three-kernel FFT), `pacmap_optimiser.rs`, `fa2_optimiser.rs` (ForceAtlas2, mass-weighted BH repulsion), `mds_optimiser.rs` (PHATE MDS). Shared Adam constants (`UMAP_BETA1`, `BETA1`, `EPS`, …) live in `training/mod.rs`.
- `src/utils/`: `bh_tree.rs` (Barnes-Hut quadtree), `fft.rs` (FFT-accelerated tSNE, `fft_tsne` only), `fft2d_gpu.rs` (2D FFT on the device), `density.rs` (densMAP, den-SNE), `dreams.rs` (DREAMS regulariser), `input.rs` (`ManifoldsMatrix`), `diffusions.rs` (PHATE + diffusion maps operators, VNE knee), `potentials.rs` (PHATE potential distance), `sparse_ops.rs`, `math.rs` (Lanczos eigenpairs, VNE), `traits.rs` (`ManifoldsFloat`, `ManifoldsFloatGpu`), `macros.rs`.
- `src/utils/sparse_svd_gpu.rs` (feature `gpu`) and `src/mlx/sparse_svd_mlx.rs` (MLX): randomised SVD of a symmetric sparse matrix, right singular vectors only. They draw the CPU's sketch (`sketch_row_major_f32`) so a seed gives the same subspace on every backend. `SpectralBackend` in `diffusions.rs` picks where PHATE's spectral landmark SVD and k-means run; `phate_gpu` selects wgpu, `landmark_method = "spectral_mlx"` selects MLX. MLX stays a string key rather than an entry point, because the Python crate cannot see `cfg(mlx_available)`.
- `src/mlx/` (`cfg(mlx_available)`): `ffi.rs` (mlx-c bindings, Metal kernels), `umap_optimiser_mlx.rs`, `tsne_optimiser_mlx.rs`, `sparse_svd_mlx.rs`.
- `src/parametric/` (feature `parametric`): `model.rs` (encoder + `TrainedUmapModel` with serde/bincode I/O), `dataset.rs`, `batch.rs`, `parametric_train.rs`.
- `src/errors.rs`: `ManifoldsError` (thiserror). Prefer returning errors over panics; recent versions have been migrating panics to variants here.
- `src/prelude.rs`: re-exports, including `Verbosity` and `parse_verbosity_level(0|1|2)` from `ann-search-rs`.

Numeric generics: everything flows through `T: ManifoldsFloat` (CPU) or `T: ManifoldsFloatGpu` (GPU-side). GPU functions take a `cubecl::Device`, which picks the runtime at run time. Linear algebra is `faer`; parallelism is `rayon`. When a fix requires wider precision, the existing convention (see tSNE) is to cast to `f64` at the numerically sensitive step, not to widen the whole pipeline. PCA initialisation and both randomised SVDs always run in `f64`, because the `f32` factorisations break at large sample counts. The reverse holds for the CPU kNN search: `run_ann_search` always runs in `f32` and casts the distances up to `T`, bar `"exhaustive"`, which stays in `T`.

Pre-computed kNN is a first-class shortcut: every entry point bar `forceatlas2_from_graph` and `train_parametric_umap_model` accepts `Option<(Vec<Vec<usize>>, Vec<Vec<T>>)>` so callers running many embeddings can compute the graph once via `run_ann_search`. UMAP, densMAP and ForceAtlas2 cut a wider graph to `k` (`truncate_knn`) and warn on stderr.

## Tests

Integration tests live in `tests/*.rs`, unit tests in `#[cfg(test)]` modules inside `src/`. Shared helpers in `tests/commons/mod.rs`: `create_diagnostic_data` (clustered f64) and `mat_to_f32` (for GPU tests). GPU/parametric/fft_tsne tests are guarded with `#![cfg(feature = "…")]` at file top, so they compile away when the feature is off.
