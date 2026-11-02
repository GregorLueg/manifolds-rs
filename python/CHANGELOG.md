# News

Changes to the `manifolds-rs` Python package. The Rust crate it wraps,
`manifolds-rs`, has its own changelog at [`../CHANGELOG.md`](../CHANGELOG.md).

## 0.6.0

Requires `manifolds-rs` 0.9.0.

**Breaking changes**

= DiffusionMaps keeps its eigenpairs in float64. float64 runs shift by about
  1e-7 of a component’s standard deviation at 20k samples.
- `float32` runs of PHATE and DiffusionMaps with spectral landmarks, and of
  PHATE(mds="classic"), give different coordinates than `0.5.0` at the same
  seed. The randomised SVDs now run in float64.
- The neighbour search runs in `float32` on every backend except "exhaustive",
  which keeps your dtype. This is faster with recall unchanged.
  `float64` runs give different coordinates than 0.5.0, and knn_graph values have
  float32 precision.
- PaCMAP defaults to learning_rate=1.0 (was 0.01), drops mn_candidate_start and
  mn_candidate_end, and searches n_near neighbours instead of 50. A precomputed
  graph only needs to be that wide. Embeddings differ from 0.5.0.
- PHATE follows the reference kernel: 3k + 2 neighbours, thresholded by thresh,
  unit diagonal, landmark operator built from the kernel. All runs differ from
  0.5.0, and a precomputed graph should be that wide.
- verbose takes 0 (quiet), 1 (timed stages, epochs every 50) or 2 (also
  parameters, landmark selection, index internals). Other values raise
  ValueError.

**Bug fixes**

- PCA initialisation always runs in `float64`. In `float32` the SVD collapsed
  the layout from about 520k samples. float32 runs differ from 0.5.0.
- PaCMAP no longer compresses the layout and PaCMAP mid-near pairs are now
  genuinely mid-near.
- Spectral landmark selection works at scale (seven power iterations on the
  normalised kernel instead of two on the raw one). DiffusionMaps with spectral
  landmarks gets the same change and now projects with P, so its coordinates
  differ too.
- Spectral landmark selection no longer fails on `float32` input on large data
  sets (≥ 500,000 samples)

## 0.5.0

Requires `manifolds-rs` 0.8.0.

**Breaking changes**

- Spectral init (`UMAP`, `DensMAP`, their GPU variants, `ForceAtlas2`),
  `PHATE` and `DiffusionMaps` give different coordinates than 0.4.x at the
  same seed. Same structure, new eigensolver.
- Default `ann` follows `k`: `"nndescent"` for `UMAP`, `DensMAP`, `PHATE`,
  `PaCMAP`, `DiffusionMaps` and `ForceAtlas2`; `"ivf"` for `TSNE`, `DensNE`
  and `DREAMS`; `"ivf_gpu"` for `TSNEGpu` and `DREAMSGpu`. Different
  embeddings at the same seed.
- NN-Descent returns the graph it builds, no search. Dropped from
  `NeighbourParams`: `diversify_prob`, `ef_budget`, `extract_knn`. Dropped
  from `NeighbourParamsGpu`: `k`, `beam_width`, `max_beam_iters`,
  `n_entry_points`, `extract_knn`.
- `verbose=1` also reports the neighbour search, `verbose=2` its internals.

**Performance**

- Spectral init uses about half the memory on connected graphs and a quarter
  on disconnected ones.
- `UMAP`, `TSNE` and `PHATE` use less memory. Same embeddings per seed.

## 0.4.2

Requires `manifolds-rs` 0.7.3.

**Chore**

- Fixes Windows ARM64 cross-builds. No API change.

## 0.4.1

Requires `manifolds-rs` 0.7.2.

**Features**

- Less memory in kNN graph construction.

## 0.4.0

Requires `manifolds-rs` 0.7.0.

**Features**

- `DREAMS` and `DREAMSGpu` (Kury, Kobak & Damrich, TMLR 2026): t-SNE pulled
  towards a global reference layout.
- Opt-in MLX for Apple Silicon, source builds only (`--features mlx`).
- CPU FFT t-SNE (`approx="fft"` and `"fft_3k"`) is back in the wheel.

## 0.3.2

Requires `manifolds-rs` 0.6.2.

**Features**

- Improved k-means.

## 0.3.1

Requires `manifolds-rs` 0.6.1.

**Features**

- Faster GPU kNN searches.

## 0.3.0

Requires `manifolds-rs` 0.6.0.

**Features**

- Quick-and-dirty Barnes-Hut t-SNE: `approx="qd"` on `TSNE`, `DensNE` and
  `TSNEGpu`.
- Faster kNN searches.

## 0.2.1

Requires `manifolds-rs` 0.5.4.

**Features**

- Faster k-means and Accelerate GEMM on macOS.

## 0.2.0

Requires `manifolds-rs` 0.5.3.

**Features**

- Scanpy-style `mf.umap(adata)` and `mf.tsne(adata)`, with the `anndata`
  extra.
- `TSNEGpu(approx="fft_3k_gpu")`: t-SNE fully on the GPU.

**Breaking changes**

- `approx="fft"` is rejected by `TSNE` / `DensNE`.

**Docs**

- New AnnData / scanpy page.

## 0.1.4

Requires `manifolds-rs` 0.5.2

- Take advantage of faster FFT-accelerated tSNE.

## 0.1.3

Requires `manifolds-rs` 0.5.1

- Wired in ForceAtlas2.

## 0.1.2

Requires `manifolds-rs` 0.5.0.

- Take in latest change from `ann-search-rs` version 0.9.0.

## 0.1.1

Requires `manifolds-rs` 0.4.1.

- AVX2 and AVX-512 distance kernels come through from the parent crate, so the
  neighbour search on x86_64 wheels is faster with no build flags.
- Guide expanded: metrics, precision, thread control, reusing a neighbour graph
  and the sharp edges.

## 0.1.0

Requires `manifolds-rs` 0.4.0. First release on PyPI.

- Python bindings under `python/`, built with PyO3 and maturin. scikit-learn
  shaped estimators over every CPU and GPU embedding, the parameter groups, the
  neighbour wrapper and the synthetic generators.
