# wgpu vs MLX on Apple Silicon

Both backends drive the same Metal GPU. wgpu goes through CubeCL and WGSL, MLX
through mlx-c with MLX ops plus hand-written Metal kernels. The question here is
simple: on a Mac, which one should you pick for which stage?

Short answer: MLX for the kNN search, either for UMAP, wgpu for t-SNE.

## Setup

- Apple M1 Max (8 performance + 2 efficiency cores, 32-core GPU), 64 GB,
  macOS 26.6.2, mlx-c 0.7.0 (MLX 0.32.3), 10 rayon threads for the CPU column.
- Data: `create_diagnostic_data` from the test helpers, five Gaussian clusters
  in 32 dimensions, `f32`. n = 100k, 500k and 1m.
- kNN: k = 15, Euclidean, two runs per backend, best reported. Recall is
  against the MLX exhaustive result.
- UMAP: Adam, 500 epochs, on one shared kNN graph and one shared random
  initialisation. Three runs per backend, best reported.
- t-SNE: three-kernel FFT, 1000 epochs, perplexity 30, again on a shared graph
  and initialisation, three runs, best reported. The CPU column is FFTW plus
  rayon (`fft_tsne`).
- Only the stage itself is timed: no graph construction, no host conversion.
  The first run of each backend pays shader compilation, hence best-of.
- Quiet machine: every other session on it held builds for the duration.

Separation (smallest centroid distance over mean within-cluster spread) is
printed by the harness as a sanity check that every backend converged to a
comparable layout. It was within noise across backends at every size, so it is
left out of the tables.

## kNN search

| n | exhaustive wgpu | exhaustive MLX | speed-up | NN-Descent wgpu | NN-Descent MLX | speed-up | recall (both) |
|---|---|---|---|---|---|---|---|
| 100k | 0.82 s | 0.45 s | 1.8x | 0.55 s | 0.42 s | 1.3x | 0.9935 |
| 500k | 14.97 s | 8.62 s | 1.7x | 2.84 s | 2.29 s | 1.2x | 0.983 |
| 1m | 57.10 s | 43.93 s | 1.3x | 6.53 s | 5.26 s | 1.2x | 0.977 |

MLX wins everywhere, at identical recall. The exhaustive lead shrinks at 1m,
which this run does not explain. NN-Descent is what UMAP and t-SNE use by
default on a GPU, and there the gain is a steady 1.2x to 1.3x.

## UMAP optimiser

| n | CPU | wgpu | MLX | MLX vs wgpu |
|---|---|---|---|---|
| 100k | 3.13 s | 0.40 s | 0.39 s | 1.0x |
| 500k | 16.84 s | 1.72 s | 1.66 s | 1.0x |
| 1m | 41.26 s | 3.45 s | 3.40 s | 1.0x |

A dead heat. Both GPU paths are 8x to 12x faster than parallel Adam on the CPU,
and the MLX one is never more than 4% ahead of wgpu.

## t-SNE optimiser

| n | CPU | wgpu | MLX | MLX vs wgpu |
|---|---|---|---|---|
| 100k | 6.59 s | 1.88 s | 1.81 s | 1.0x |
| 500k | 24.47 s | 6.63 s | 9.74 s | 0.7x |
| 1m | 47.55 s | 13.69 s | 25.83 s | 0.5x |

Level at 100k, then MLX falls behind: at 1m it takes twice as long as wgpu and
is only 1.8x faster than the CPU. Which kernel stops scaling has not been
profiled yet. Until it has, use `"fft_3k_gpu"` for t-SNE on anything large.

## Verdict

| Stage | Pick |
|---|---|
| kNN | MLX, 1.2x to 1.8x over wgpu |
| UMAP / densMAP optimiser | either |
| t-SNE / DREAMS optimiser | wgpu, up to 2x faster than MLX at 1m |

The two mix freely: `tsne_gpu` and `dreams_gpu` take `"fft_3k_mlx"`, and any
entry point takes a precomputed kNN graph, so an MLX search can feed the wgpu
optimiser. Skipping MLX costs little: at most 1.8x on the search and nothing on
UMAP.

## Rerunning

```sh
cargo test --release --features gpu,fft_tsne,mlx,large_scale_diagnostics \
  --test umap_optimiser_compare_bench -- --nocapture --test-threads 1 --include-ignored
cargo test --release --features gpu,fft_tsne,mlx,large_scale_diagnostics \
  --test tsne_optimiser_compare_bench -- --nocapture --test-threads 1 --include-ignored
```

The 500k and 1m cases are `#[ignore]`d; `--include-ignored` runs everything,
which took about 20 minutes on the machine above.
