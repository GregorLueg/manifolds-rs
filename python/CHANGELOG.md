# News

Changes to the `manifolds-rs` Python package. The Rust crate it wraps,
`manifolds-rs`, has its own changelog at [`../CHANGELOG.md`](../CHANGELOG.md).

## 0.3.2

Requires the `manifolds-rs` 0.6.2.

**Features**

- Pull in the changes from `manifolds-rs` with the improved k-means clustering.

## 0.3.1

Requires the `manifolds-rs` 0.6.1.

**Features**

- Pull in the changes from `manifolds-rs` with the faster GPU-accelerated kNN
  searches.

## 0.3.0

Requires the `manifolds-rs` 0.6.0.

**Features**

- `TSNE(approx="qd")`, `DensNE(approx="qd")` and `TSNEGpu(approx="qd")`: the
  quick-and-dirty Barnes-Hut of qdtsne. The tree depth is capped at the new
  `TsneOptim.max_depth` (default 7) and repulsion is computed once per leaf.
  On 20k points: ca. 3x faster than `"barnes_hut"`, silhouette 0.783 against
  0.796; `max_depth=10` gave 2x faster and 0.795.
- Some of the kNN searches became faster.

## 0.2.1

Requires `manifolds-rs` 0.5.4.

**Features**

- Takes in the faster k-means for some of the indices (and PHATE) and Accelerate
  GEMM for MacOS.

## 0.2.0

Requires `manifolds-rs` 0.5.3.

**Features**

- Scanpy-style `mf.umap(adata)` and `mf.tsne(adata)`. They write
  `obsm["X_umap"]` / `obsm["X_tsne"]` and `uns["umap"]` / `uns["tsne"]` as
  `sc.tl.umap` / `sc.tl.tsne` do, so `sc.pl.*` works unchanged. UMAP reuses the
  `sc.pp.neighbors` graph; t-SNE searches on `use_rep` / `n_pcs`.
  `device="gpu"` switches to the GPU estimators, `key_added` and `copy` behave
  as in scanpy. Needs the new `anndata` extra: `manifolds-rs[anndata]`.
- `TSNEGpu(approx="fft_3k_gpu")`: the device-resident three-kernel FFT
  optimiser. The whole t-SNE optimisation runs on the GPU, no FFTW needed.

**Breaking changes**

- `approx="fft"` is no longer accepted by `TSNE` / `DensNE`. The wheel is built
  without FFTW, so it only ever reached a panic in the core; it now fails
  validation with a `ValueError` like any other unknown name.

**Docs**

- New AnnData / scanpy page with a 25k-cell benchmark against scanpy.
- The reason CPU FFT t-SNE is not in the wheel is now the right one: FFTW is
  GPL-2.0-or-later and would be linked statically.

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
