# News

Changes to the `manifolds-rs` Python package. The Rust crate it wraps,
`manifolds-rs`, has its own changelog at [`../CHANGELOG.md`](../CHANGELOG.md).

## 0.6.1

Requires `manifolds-rs` 0.9.1.

**Fixes**

Different coordinates than 0.6.0 at the same seed with an automatic `t`:
`PHATE` and `DiffusionMaps` on 5000 or more points (full operator or
landmarks), and `float32` runs of either below that.

- The automatic `t` from 5000 points up used only the top 100 singular values,
  which put the knee far too late: `DiffusionMaps` on 10k cells of a real single
  cell PCA picked `t = 44` where the full spectrum gives 12. It now estimates
  the entropy over the whole spectrum at about the same cost.
- Choosing `t` below 5000 points, including on the default 2048 `PHATE`
  landmarks, takes about half the time. Same `t`.
- Spectral landmark selection in `DiffusionMaps` no longer scales with points
  times landmarks. Same landmarks.

## 0.6.0

Requires `manifolds-rs` 0.9.0.

**Breaking changes**

Different coordinates than 0.5.0 at the same seed: `PaCMAP`, `PHATE`,
`UMAP` and `DensMAP` with `optimiser="sgd"`, `DiffusionMaps` with spectral
landmarks, `float32` PCA initialisation and every `float64` run on an
approximate neighbour search.

- The neighbour search runs in `float32` on every backend bar `"exhaustive"`,
  which keeps your dtype. Same recall, faster. `knn_graph` values have `float32`
  precision.
- `PaCMAP` follows the reference.
  - `learning_rate` defaults to `1.0` (was `0.01`), `ann` to `"ivf"` (was
    `"nndescent"`). `mn_candidate_start` and `mn_candidate_end` are gone.
  - Near pairs are the best `n_near` of an `n_near + 50` wide search by locally
    scaled distance. A precomputed graph needs at least `n_near` neighbours and
    should be `n_near + 50` wide.
  - Further pairs exclude the near pairs.
  - PCA initialisation is the PC scores times `range_` (`None` means `0.01`), so
    the ratio between components survives.
- `PHATE` follows the reference.
  - 2048 landmarks by default (was none). `n_landmarks` and `landmark_method`
    moved from `PhateDiffusion` to the `PHATE` and `PHATEGpu` constructors, and
    `n_landmarks=None` selects the full operator.
  - `ann` defaults to `"ivf"` (was `"nndescent"`). `PHATEGpu` keeps
    `"nndescent_gpu"`.
  - The search is `6k + 5` wide, thresholded by `thresh`. A precomputed graph
    should be that wide.
  - The kernel has a unit diagonal, the landmark operator is built from it, and
    an unreached state has a large log potential rather than zero.
  - SGD-MDS runs 300 iterations below 1000 points, 500 below 5000, 800 otherwise.
  - `mds="classic"` runs its randomised SVDs in `float64`.
- `PHATE` and `PHATEGpu` raise `ValueError` on `metric="manhattan"`. They
  panicked.
- `UMAP`, `DensMAP`, `ForceAtlas2` and the GPU variants cut a precomputed graph
  wider than `n_neighbors` to `n_neighbors`, with a warning on stderr. They used
  every neighbour given.
- `DiffusionMaps` with spectral landmarks and `n_svd=None` takes 100 singular
  vectors, was 10.
- `verbose` takes 0 (quiet), 1 (timed stages, epochs every 50) or 2 (also
  parameters, landmark selection, index internals). Anything else raises
  `ValueError`.
- `n_epochs` defaults to `None`: 500, or 200 for the SGD optimisers above
  10,000 samples, as umap-learn does. `n_epochs=500` restores the old default.

**Features**

- `PHATEGpu`: PHATE with the neighbour search and, for spectral landmarks, the
  randomised SVD and k-means on the device, in `float32`.
- `PHATE(landmark_method="spectral_mlx")` in an `mlx` build: the landmark SVD
  and k-means on MLX, in `float32`. Works with `PHATEGpu` too.
- `optimiser="sgd_parallel"`: SGD run lock-free over all threads. Not
  reproducible run to run.

**Performance**

- `optimiser="sgd"` is about 2.4x faster; FFT t-SNE and parallel Adam UMAP are
  faster on large data on Apple Silicon.

**Bug fixes**

- PCA initialisation runs in `float64`. In `float32` it collapsed the layout
  from about 520k samples.
- `PaCMAP` no longer compresses the layout: mid-near pairs are the second
  closest of six random points, not neighbours.
- Spectral landmark selection works at scale and no longer fails on `float32`
  input from 500k samples. `DiffusionMaps` with spectral landmarks also projects
  with `P` now.
- `PHATE` honours `n_svd` and, with landmarks, `mds_iter`, and drops empty
  landmarks instead of keeping them as zero rows.

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
