[![CI](https://github.com/GregorLueg/manifolds-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/manifolds-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/manifolds-rs.svg)](https://crates.io/crates/manifolds-rs)
[![PyPI](https://img.shields.io/pypi/v/manifolds-rs.svg)](https://pypi.org/project/manifolds-rs/)
[![docs.rs](https://img.shields.io/docsrs/manifolds-rs)](https://docs.rs/manifolds-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# manifolds-rs

Dimensionality reduction in Rust, built for single cell and computational
biology: UMAP, t-SNE and friends, generic over `f32`/`f64`, parallel via rayon,
with GPU paths via wgpu or, on Apple Silicon, MLX. The kNN search comes from
[ann-search-rs](https://github.com/GregorLueg/ann-search-rs).

It powers the R package [manifoldsR](https://gregorlueg.github.io/manifoldsR/index.html),
and there is a Python package on [PyPI](https://pypi.org/project/manifolds-rs/)
(`manifolds-rs`, source in [`python/`](python/)) with scikit-learn shaped
estimators and scanpy-style `mf.umap(adata)`. Changes live in the
[CHANGELOG](https://github.com/GregorLueg/manifolds-rs/blob/main/CHANGELOG.md).

## What's in it

| Method | Entry points | Feature |
|---|---|---|
| [UMAP](https://arxiv.org/abs/1802.03426) | `umap`, `umap_gpu` | default, `gpu` |
| [densMAP](https://www.nature.com/articles/s41587-020-00801-7) | `densmap`, `densmap_gpu` | default, `gpu` |
| [Parametric UMAP](https://arxiv.org/abs/2009.12981) | `parametric_umap`, `train_parametric_umap_model` | `parametric` |
| t-SNE ([Barnes-Hut](https://arxiv.org/abs/1301.3342), [qd](https://github.com/libscran/qdtsne), [FFT](https://www.nature.com/articles/s41592-018-0308-4)) | `tsne`, `tsne_gpu` | default, `fft_tsne`, `gpu` |
| [den-SNE](https://www.nature.com/articles/s41587-020-00801-7) | `densne` | default |
| DREAMS (Kury, Kobak & Damrich, TMLR 2026) | `dreams`, `dreams_gpu` | default, `gpu` |
| [PHATE](https://pmc.ncbi.nlm.nih.gov/articles/PMC7073148/) | `phate` | default |
| [PaCMAP](https://arxiv.org/abs/2012.04456) | `pacmap` | default |
| [ForceAtlas2](https://doi.org/10.1371/journal.pone.0098679) | `forceatlas2`, `forceatlas2_from_graph` | default |
| [Diffusion maps](https://www.sciencedirect.com/science/article/pii/S1063520306000546) | `diffusion_maps` | default |

Optimisers and approximations are picked by string:

| What | Options |
|---|---|
| UMAP optimiser | `"adam_parallel"` (default), `"adam"`, `"sgd"`, `"adam_gpu"`, `"adam_mlx"` |
| t-SNE repulsion | `"bh"`, `"qd"`, `"fft"`, `"fft_3k"`, `"fft_3k_gpu"`, `"fft_3k_mlx"` |
| kNN, CPU | `"kmknn"`, `"exhaustive"`, `"hnsw"`, `"nndescent"`, `"annoy"`, `"ivf"`, `"balltree"` |
| kNN, wgpu | `"nndescent_gpu"` (default), `"ivf_gpu"`, `"exhaustive_gpu"` |
| kNN, MLX | `"nndescent_mlx"`, `"ivf_mlx"`, `"exhaustive_mlx"` |

Distances: Euclidean and cosine. Initialisations: spectral (Lanczos), PCA,
random. Synthetic datasets (swiss roll, clusters, branching trajectories) ship
with the crate for testing.

## Installation

```toml
[dependencies]
manifolds-rs = { version = "0.7", features = ["gpu"] }
```

Optional features:

- `gpu`: wgpu (Metal, Vulkan, DX12) or CUDA via
  [CubeCL](https://github.com/tracel-ai/cubecl). GPU kNN, GPU Adam for
  UMAP/densMAP and a device-resident three-kernel FFT t-SNE. `f32` only.
- `mlx`: Apple Silicon only, experimental. MLX kNN and optimisers. See below.
- `fft_tsne`: CPU FFT t-SNE via [`fftw`](https://crates.io/crates/fftw). Builds
  FFTW from source and links it statically. FFTW is GPL-2.0-or-later, so a
  binary you distribute with this on is GPL. `"fft_3k_gpu"` and `"fft_3k_mlx"`
  need no FFTW.
- `parametric`: parametric UMAP on [burn](https://burn.dev), models serialise
  via `bincode`.
- `ndarray`: accept `ndarray` 2-D arrays. Off by default, since it makes
  `ndarray` a public dependency.

## Usage

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;

let (data, labels) = generate_clustered_data(1000, 50, 5, 42);

let params = UmapParams::new_default_2d(
    Some(0.5), // min_dist
    Some(1.0), // spread
);

// [2][n_samples]: embedding[0] are the x, embedding[1] the y coordinates
let embedding = umap(data.as_ref(), None, &params, 42, 1)?;
```

Every entry point follows this shape: data, an optional precomputed kNN, a
params struct, a seed and a verbosity level, and returns
`Result<Vec<Vec<T>>, ManifoldsError>`. Inputs can be a faer matrix, an
`ndarray` view (with the feature) or a flat row-major slice with its shape.

Running several embeddings on the same data? Build the kNN graph once with
`run_ann_search` and hand it to each of them as `Some((indices, distances))`.

Worked examples for every method, including GPU and parametric UMAP, are in
[docs/examples.md](https://github.com/GregorLueg/manifolds-rs/blob/main/docs/examples.md).

## GPU

The `gpu` feature runs on CubeCL with the wgpu backend by default, so the same
kernels run on Metal, Vulkan and DX12; pass a `cubecl::Device::Cuda` for CUDA. The
`*_gpu` entry points move the kNN search to the device, and UMAP's
`"adam_gpu"` (default) and t-SNE's `"fft_3k_gpu"` move the optimiser too.
Device limits are queried at runtime, never assumed.

Everything on the device is `f32`: WGSL has no `f64`, and consumer GPUs run it
at a crawl anyway. Results are reproducible in structure, not bit for bit,
since parallel reductions have no fixed order. Need `f64`? Stay on the CPU.

## MLX (experimental)

Apple Silicon only. The `mlx` feature drives Metal through
[mlx-c](https://github.com/ml-explore/mlx-c) instead of wgpu: MLX ops plus
hand-written Metal kernels for the UMAP/densMAP Adam update and the
three-kernel FFT t-SNE, and the MLX kNN indices from ann-search-rs.

On an M1 Max the MLX kNN search is 1.2x to 1.8x faster than wgpu at equal
recall. The optimisers are a different story: UMAP's Adam ties with wgpu, and
the t-SNE optimiser ties at 100k but falls to half of wgpu's speed at 1m. Pick
per stage; the
[benchmarks](https://github.com/GregorLueg/manifolds-rs/blob/main/docs/benchmarks_mlx.md)
have the numbers. `f32` only.

```bash
brew install mlx-c
```

```toml
manifolds-rs = { version = "0.7", features = ["mlx"] }
```

MLX is not a separate set of entry points. The CPU ones take the MLX names:

```rust
let params = UmapParams {
    ann_type: "nndescent_mlx".into(),
    optimiser: "adam_mlx".into(),
    ..UmapParams::<f32>::default()
};
let embedding = umap(data.as_ref(), None, &params, 42, 0)?;
```

`tsne` and `dreams` take `"fft_3k_mlx"`, as do `tsne_gpu` and `dreams_gpu`
(search on wgpu, optimiser on MLX). den-SNE has no MLX path. If mlx-c lives
somewhere other than `/opt/homebrew`, set `MLX_C_PREFIX`. If it is not found,
the build warns and leaves MLX out; `manifolds_rs::MLX_AVAILABLE` tells you
which one you got.

## Benchmarks

- [wgpu vs MLX](https://github.com/GregorLueg/manifolds-rs/blob/main/docs/benchmarks_mlx.md):
  kNN, UMAP and t-SNE optimisers at 100k, 500k and 1m points.
- kNN backends themselves: see the
  [ann-search-rs benchmarks](https://github.com/GregorLueg/ann-search-rs#benchmarks).

Everything is tested against synthetic data and sanity-checked on real single
cell data, which is no substitute for looking at your own embeddings. Use the
latest release; older ones carry known bugs, notably the PaCMAP one fixed in
`0.3.6`.

## Licence

MIT, see [LICENSE](LICENSE).
