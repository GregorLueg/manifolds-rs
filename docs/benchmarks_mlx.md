# wgpu vs MLX on Apple Silicon

Both backends drive the same Metal GPU. wgpu goes through CubeCL and WGSL, MLX
through mlx-c with MLX ops plus hand-written Metal kernels. The question here is
simple: on a Mac, which one should you pick for which stage?

Short answer: MLX for the kNN search, either for the UMAP and t-SNE optimisers.
And if you have MLX, have a look at `"ivf_mlx"`: it is very fast.

## Setup

- Apple M1 Max (8 performance + 2 efficiency cores, 32-core GPU), 64 GB,
  macOS 26.6.2, mlx-c 0.7.0 (MLX 0.32.3), 10 rayon threads for the CPU column.
- Only the stage itself is timed: no graph construction, no host conversion.
  The first run of each backend pays shader compilation, hence best-of.
- Quiet machine: every other session on it held builds for the duration.

The kNN search and the optimisers run on different data, because they ask
different questions:

- **kNN:** 25 blobs (`create_clustered_data_f32`), centres and noise both
  uniform per axis, so the clusters are equally well separated at 32,
  128 and 256 dimensions. n = 100k, 500k and 1m. k = 15, Euclidean, two runs per
  backend, best reported. Recall is against the MLX exhaustive result.
- **Optimisers:** five clusters in 32 dimensions (`create_diagnostic_data`),
  one shared kNN graph and one shared random initialisation per size. UMAP:
  Adam, 500 epochs. t-SNE: three-kernel FFT, 1000 epochs, perplexity 30, CPU
  column is FFTW plus rayon (`fft_tsne`). Three runs per backend, best reported.

Separation (smallest centroid distance over mean within-cluster spread) is
printed for every optimiser run as a check that the backends converge to
comparable layouts. It was within noise across backends at every size, so it is
left out of the optimiser tables.

## kNN search

Times are wgpu / MLX, speed-up of MLX in brackets.

| n | d | exhaustive | IVF | NN-Descent |
|---|---|---|---|---|
| 100k | 32 | 0.74 / 0.41 s (1.8x) | 0.22 / 0.17 s (1.3x) | 0.45 / 0.31 s (1.5x) |
| 100k | 128 | 1.99 / 0.67 s (3.0x) | 0.66 / 0.33 s (2.0x) | 1.27 / 0.86 s (1.5x) |
| 100k | 256 | 3.66 / 1.14 s (3.2x) | 1.30 / 0.56 s (2.3x) | 2.08 / 1.41 s (1.5x) |
| 500k | 32 | 14.88 / 8.72 s (1.7x) | 2.26 / 1.26 s (1.8x) | 2.51 / 1.88 s (1.3x) |
| 500k | 128 | 45.10 / 15.57 s (2.9x) | 5.35 / 3.03 s (1.8x) | 7.80 / 5.35 s (1.5x) |
| 500k | 256 | 85.40 / 26.72 s (3.2x) | 9.50 / 5.80 s (1.6x) | 13.23 / 8.49 s (1.6x) |
| 1m | 32 | 56.88 / 44.01 s (1.3x) | 7.21 / 3.42 s (2.1x) | 5.54 / 4.27 s (1.3x) |
| 1m | 128 | 180.75 / 78.10 s (2.3x) | 14.88 / 8.89 s (1.7x) | 16.80 / 11.51 s (1.5x) |
| 1m | 256 | 337.03 / 128.41 s (2.6x) | 25.76 / 17.16 s (1.5x) | 27.62 / 17.82 s (1.5x) |

Recall against exact, identical on both backends to within 0.014:

| n | d | IVF | NN-Descent |
|---|---|---|---|
| 100k | 32 / 128 / 256 | 0.9999 / 0.9994 / 0.9995 | 0.998 / 0.978 / 0.953 |
| 500k | 32 / 128 / 256 | 0.996 / 0.968 / 0.961 | 0.995 / 0.878 / 0.76 |
| 1m | 32 / 128 / 256 | 0.993 / 0.934 / 0.914 | 0.991 / 0.789 / 0.62 to 0.63 |

MLX wins every cell. Exhaustive search is a GEMM, MLX's GEMM is very good, and
the lead grows with the dimension: 1.3x to 1.8x at 32 dimensions, 2.6x to 3.2x
at 256. The wgpu exhaustive at 1m and 256 dimensions takes well over five
minutes per run, so don't.

**IVF on MLX is the standout.** It is the fastest search on MLX in every cell
here, and on real data it is fast as well: on a 2.9m-cell single-cell
PCA (Tahoe-100M, 32 PCs) ann-search-rs reaches 0.95 recall with `ivf_mlx` in
4.7 s, against 9.3 to 9.5 s for NN-Descent on either GPU backend (see
`benchmarks_single_cell.md` in ann-search-rs). If you have the MLX build, pass
`"ivf_mlx"` as the ANN type and see. Its recall at the default
`n_probes = sqrt(n_list)` drops with dimension and n (0.91 at 1m and 256
dimensions); raise `n_probes` to buy it back.

NN-Descent stays the default on both GPU backends and IVF is the opt-in. On
wgpu the two are close on the real data. The section below is why NN-Descent's
low recall on the 256-dimensional blobs is not a problem for the embedding.

### Low recall, close neighbours

NN-Descent, and graph-based searches in general (HNSW, Vamana and friends),
refine a graph by checking the neighbours of neighbours. In high-dimensional
blobs like these, distances within a cluster concentrate: a point's true 15
nearest neighbours are barely closer than many of its other cluster-mates.
The local joins converge on a set of neighbours that is almost as close as the
exact one, but not the exact one. Recall counts every swap as a miss, however
small the difference in distance. IVF compares exactly within the lists it
probes, so it only misses neighbours that fall into a list it did not probe.

So recall alone overstates the damage. The distance ratio (mean distance to the
neighbours found over mean distance to the exact neighbours, per point) and the
embedding tell the rest. UMAP Adam on MLX, 500 epochs, 256 dimensions, from
each MLX graph. Purity is the fraction of each point's 15 nearest neighbours in
2D that share its cluster; the second exhaustive run, with another
initialisation and seed, is the noise floor.

| n | graph | recall | distance ratio, mean / median | purity | separation |
|---|---|---|---|---|---|
| 100k | exhaustive | 1.0000 | 1.0000 / 1.0000 | 1.0000 | 13.67 |
| 100k | exhaustive, seed 2 | 1.0000 | 1.0000 / 1.0000 | 1.0000 | 13.74 |
| 100k | IVF | 0.9995 | 1.0000 / 1.0000 | 1.0000 | 13.87 |
| 100k | NN-Descent | 0.9522 | 1.0005 / 1.0000 | 1.0000 | 13.65 |
| 500k | exhaustive | 1.0000 | 1.0000 / 1.0000 | 1.0000 | 14.67 |
| 500k | exhaustive, seed 2 | 1.0000 | 1.0000 / 1.0000 | 1.0000 | 14.23 |
| 500k | IVF | 0.9609 | 1.0004 / 1.0000 | 1.0000 | 14.65 |
| 500k | NN-Descent | 0.7647 | 1.0028 / 1.0023 | 1.0000 | 15.10 |

At 0.76 recall NN-Descent's neighbours are on average 0.3% further away than
the exact ones, and the layout keeps every point with its cluster. Two caveats,
so nobody reads more into this than is there: the blobs are synthetic and well
separated, so purity sits at 1.0 for every graph and cannot tell them apart
beyond cluster membership. And a noise floor of two seeds is thin: NN-Descent's
separation of 15.10 sits just outside the exhaustive 14.23 to 14.67, which is
not a difference either way.

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
| 100k | 6.37 s | 1.84 s | 1.79 s | 1.0x |
| 500k | 24.10 s | 6.67 s | 6.89 s | 1.0x |
| 1m | 48.96 s | 13.71 s | 13.64 s | 1.0x |

Level again.

## Verdict

| Stage | Pick |
|---|---|
| kNN | MLX, 1.3x to 3.2x over wgpu, more at higher dimension; try `"ivf_mlx"` |
| UMAP / densMAP optimiser | either |
| t-SNE / DREAMS optimiser | either |

The two mix freely: `tsne_gpu` and `dreams_gpu` take `"fft_3k_mlx"`, and any
entry point takes a precomputed kNN graph, so an MLX search can feed the wgpu
optimiser. Skipping MLX costs nothing on the optimisers and 1.3x to 3.2x on the
search.

## Rerunning

```sh
# kNN sweep, embedding quality and the UMAP optimiser
cargo test --release --features gpu,fft_tsne,mlx,large_scale_diagnostics \
  --test umap_optimiser_compare_bench -- --nocapture --test-threads 1 --include-ignored
# t-SNE optimiser
cargo test --release --features gpu,fft_tsne,mlx,large_scale_diagnostics \
  --test tsne_optimiser_compare_bench -- --nocapture --test-threads 1 --include-ignored
```

The 500k and 1m cases are `#[ignore]`d; `--include-ignored` runs everything.
Budget about 45 minutes for the kNN sweep alone, 33 of them at 1m, nearly all
in exhaustive search. Run single tests by name (`knn_compare_100k`,
`knn_embedding_quality_500k`, `tsne_optimiser_compare_1m`, ...) to pick.
