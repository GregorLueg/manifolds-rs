[![CI](https://github.com/GregorLueg/manifolds-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/manifolds-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/manifolds-rs.svg)](https://crates.io/crates/manifolds-rs)
[![docs.rs](https://img.shields.io/docsrs/manifolds-rs)](https://docs.rs/manifolds-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# manifolds-rs

High-performance manifold learning and dimensionality reduction algorithms
implemented in Rust. Contains as for now:

- **UMAP**
  - Has different optimisers: SGD (traditional), Adam and a parallelised
  version of Adam for very fast fitting.
  - Optional GPU-accelerated kNN search and GPU Adam optimiser via
  [`cubecl`](https://crates.io/crates/cubecl).
- **densMAP**, the density-preserving variant of UMAP (also on GPU).
- **Parametric UMAP** (optional feature)
- **tSNE**
  - ***Barnes Hut tSNE*** (With a `O(n log n)` complexity).
  - ***Fast Fourier Transform-accelerated Interpolation-based t-SNE (Flt-SNE)***
  (optional feature; with a `O(n)` complexity for large datasets).
  - Optional GPU-accelerated kNN search.
- **den-SNE**, the density-preserving variant of tSNE (Barnes-Hut and FFT).
- **PHATE**
- **PaCMAP**
- **Diffusion Maps**
  - Classical diffusion maps (Coifman & Lafon, 2006) with anisotropic
  (alpha) normalisation for density correction.
  - Optional landmark-based approximation with Nystroem extension for larger
  datasets.

## Description

Rust implementations of various methods to project data onto two dimensions,
i.e, learn low dimensional manifolds from the data. The current crate contains
the big classic [UMAP](https://arxiv.org/abs/1802.03426) and tSNE (with the
[Barnes-Hut implementation](https://arxiv.org/abs/1301.3342) and optionally the
[FFT-acceleration version](https://www.nature.com/articles/s41592-018-0308-4)).
These are typically used methods for visualising high-dimensional biological
data, but not without [controversy](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011288).
Moreover, the `crate` also provides via the Burn DL framework optionally
[parametric UMAP](https://arxiv.org/abs/2009.12981) that can be optionally be
used via the prospective feature flag. Since release `0.1.8`, we also have
[PHATE](https://pmc.ncbi.nlm.nih.gov/articles/PMC7073148/). With `0.1.9`,
[PaCMAP](https://arxiv.org/abs/2012.04456) has been also implemented. More
recently, classical [diffusion maps](https://www.sciencedirect.com/science/article/pii/S1063520306000546)
have been added as well. Since `0.3.11` there are also the density-preserving
variants [densMAP and den-SNE](https://www.nature.com/articles/s41587-020-00801-7),
which stop the embedding from rendering a tight cluster and a diffuse one at
the same size.
Changelog can be found [here](https://github.com/GregorLueg/manifolds-rs/blob/main/CHANGELOG.md))

## Features

- **UMAP algorithm**: Complete implementation of the UMAP dimensionality
reduction algorithm with several optimisations: SGD, Adam and a parallelised
version of ADAM for increased optimisation speed.
- **tSNE algorithm**: Implementation of the Barnes-Hut accelerated version and
the FFT-accelerated version (optional).
- **densMAP and den-SNE**: Density-preserving versions of UMAP and tSNE. An
extra gradient term maximises the correlation between the local radius of a
point in the original space and in the embedding, so cluster size carries
meaning. Works with all four UMAP optimisers (SGD, Adam, parallel Adam, GPU
Adam) and both tSNE optimisers (Barnes-Hut, FFT).
- **Parametric UMAP** (optional feature `parametric`): A neural network encoder
trained on the UMAP objective via [`burn`](https://burn.dev), so new points can
be embedded without refitting. Models serialise to disk via `bincode`.
- **PHATE**: Implementation of Potential of Heat-diffusion for Affinity-based
Trajectory Embedding with different landmark methods.
- **PaCMAP**: Pairwise Controlled Manifold Approximation, preserving local and
global structure through near, mid-near and further pairs with a three-phase
optimisation schedule.
- **Diffusion Maps**: Classical diffusion maps with anisotropic normalisation
(alpha in [0, 1] controlling density correction from the normalised graph
Laplacian to the Laplace-Beltrami operator), Von Neumann entropy-based
diffusion time selection, and an optional landmark variant with Nystroem
extension for larger datasets.
- **GPU acceleration** (optional feature `gpu`): The most expensive part of
UMAP and tSNE for large datasets is the nearest neighbour search. With the
`gpu` feature enabled, kNN search runs on the GPU via
[`cubecl`](https://crates.io/crates/cubecl), with backends for Vulkan, Metal,
DirectX 12 (through wgpu) and CUDA. UMAP additionally has a GPU Adam optimiser
(the `"adam_gpu"` default for `umap_gpu`); for tSNE only the kNN search moves
to the device.
- **Multiple ANN backends** via [`ann-search-rs`](https://crates.io/crates/ann-search-rs):
  - *Exhaustive* (`"exhaustive"`) - If you want precise results and have a
    small data set in which the approximate nearest neighbour index building is
    actually slower.
  - *KmKnn* (`"kmknn"`) - An exact nearest neighbour search algorithm,
    leveraging k-means clustering under the hood for speed. If you need exact
    results.
  - *BallTree* (`"balltree"`) - A small, fast index for smaller data sets with
    lower dimensions.
  - *Annoy (Approximate Nearest Neighbours Oh Yeah)* (`"annoy"`) - Good for
    medium low-dimensionality datasets.
  - *NNDescent (Nearest Neighbour Descent)* (`"nndescent"`) - good for larger
    datasets with higher dimensionality.
  - *HNSW (Hierarchical Navigable Small World)* (`"hnsw"`) - good for (very)
    larger datasets with higher dimensionality.
  - *IVF (inverted file index)* (`"ivf"`) - Another fast and optimised index.
- **GPU ANN backends** (with `gpu` feature):
  - *Exhaustive GPU* (`"exhaustive_gpu"`) - Brute-force kNN on the GPU;
    deterministic and accurate.
  - *IVF GPU* (`"ivf_gpu"`) - Inverted-file index on the GPU.
  - *NNDescent GPU* (`"nndescent_gpu"`) - CAGRA-style graph construction on the
    GPU. The default for both `umap_gpu` and `tsne_gpu`.
- **Distance metrics**:
  - Euclidean (`"euclidean"`)
  - Cosine (`"cosine"`)
  - Maybe more to come over time ... ?
- **Multiple initialisations**:
  - Graph Laplacian eigenvector-based initialisation using Lanczos iteration
    (`"spectral"`)
  - Random initialisation (`"random"`)
  - PCA-based initialisation (`"pca"`)
- **Customisable parameters**: Full control over fuzzy simplicial set
  construction, graph symmetrisation, and optimisation parameters for tSNE,
  UMAP and PHATE.
- **High performance**: Parallel processing with Rayon, efficient sparse matrix
  operations, cache-friendly structures and optimised SGD and Adam optimisers
  for UMAP (for the latter also a parallelised version...) and fast optimisers
  for tSNE and also PHATE.
- **Synthetic datasets**: Some synthetic datasets are available for testing and
  experimentation: Swiss roll, clustered data and a trajectory-like structure.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
manifolds-rs = "*"
```

If you want to enable parametric UMAP, please use:

```toml
[dependencies]
manifolds-rs = { version = "*", features = [ "parametric" ] }
```

If you want to enable the FFT-accelerated version of tSNE, please use:

```toml
[dependencies]
manifolds-rs = { version = "*", features = [ "fft_tsne" ] }
```

The `fft_tsne` feature binds to system FFTW, so you need that installed.

If you want to enable GPU-accelerated kNN search, please use:

```toml
[dependencies]
manifolds-rs = { version = "*", features = [ "gpu" ] }
```

Feature flags can be combined, e.g. `features = [ "gpu", "fft_tsne" ]`.

## Notes

Please use the latest release. Older versions carry known bugs, notably the
PaCMAP one fixed in `0.3.6`. Everything here is tested against synthetic data
and sanity-checked on real single-cell data, but it is not a substitute for
looking at your own embeddings.

Every entry point returns `Result<Vec<Vec<T>>, ManifoldsError>`. The examples
below use `.unwrap()` for brevity; handle the error properly in real code.

Note on GPU support: kNN runs on GPU through `wgpu` (Vulkan, Metal, DX12) or
CUDA via `cubecl`. Computations on the GPU side are performed in `f32`; this
is a limitation of WGSL (the wgpu shader language) which has no `f64`, and
also reflects the fact that `f64` throughput on consumer GPUs is typically
1/32 to 1/64 of `f32`. If you need double precision, stick to the CPU path.
GPU results are not bit-reproducible across runs (parallel reductions do not
have a fixed accumulation order), but structural quality is consistent.

## Usage

### R package

This crate powers [manifoldsR](https://gregorlueg.github.io/manifoldsR/index.html),
an R package leveraging the incredible speed that Rust offers.

### UMAP Example

Below are examples of how to use UMAP. The parameter structs live at the crate
root, the shared helpers (synthetic data, `run_ann_search`, `ManifoldsError`)
in the prelude, so both glob imports are worth having.

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;

// Generate synthetic clustered data
let (data, labels) = generate_clustered_data(
    1000,  // n_samples
    50,    // dimensionality
    5,     // n_clusters
    42,    // seed
);

// Configure UMAP parameters
let params = UmapParams::new_default_2d(
    Some(0.5),   // min_dist
    Some(1.0),   // spread
);

// Run UMAP
let embedding = umap(
    data.as_ref(),
    None,        // precomputed kNN (None = compute internally)
    &params,
    42,          // seed
    1,           // verbose -> light levels of verbosity
)
.unwrap();

// embedding[0] contains x-coordinates
// embedding[1] contains y-coordinates
```

`UmapParams::default()` uses the `"adam_parallel"` optimiser, `k = 15` and
spectral initialisation. For full control over every field (optimiser, ANN
backend, graph parameters, epochs) use `UmapParams::new`.

### densMAP Example

densMAP is UMAP plus a gradient term that keeps local density interpretable.
`lambda` controls how hard it pushes; larger values preserve density more
aggressively at the cost of cluster separation. The default is `2.0`, and the
term is only active over the final 30% of the epochs.

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;

// Generate synthetic clustered data
let (data, labels) = generate_clustered_data(1000, 50, 5, 42);

// Configure densMAP parameters
let params = DensmapParams::new_default_2d(
    Some(0.5),   // min_dist
    Some(1.0),   // spread
    Some(2.0),   // lambda (density weight)
);

// Run densMAP
let embedding = densmap(
    data.as_ref(),
    None,        // precomputed kNN (None = compute internally)
    &params,
    42,          // seed
    1,           // verbose -> light levels of verbosity
)
.unwrap();
```

For finer control, `DensmapParams::new` takes a full `UmapParams` and a
`DensParams` (`lambda`, `frac`, `var_shift`). Setting `lambda = 0.0` recovers
plain UMAP.

### t-SNE Example

Below are examples of how to use t-SNE.

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;

// Generate synthetic clustered data
let (data, labels) = generate_clustered_data(1000, 50, 5, 42);

// Configure t-SNE parameters
let params = TsneParams::new_default_2d(
    Some(30.0),   // perplexity
);

// Run t-SNE (Barnes-Hut)
let embedding = tsne(
    data.as_ref(),
    None,        // precomputed kNN (None = compute internally)
    &params,
    "bh",        // approximation: "barnes_hut" | "bh", or "fft" (fft_tsne feature)
    42,          // seed
    1,           // verbose -> light levels of verbosity
)
.unwrap();
```

### den-SNE Example

Same idea as densMAP, applied to t-SNE. The default `lambda` here is `0.1`,
twenty times smaller than densMAP's, because the Student-t gradients are much
larger.

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;

// Generate synthetic clustered data
let (data, labels) = generate_clustered_data(1000, 50, 5, 42);

// Configure den-SNE parameters
let params = DensneParams::new_default_2d(
    Some(30.0),   // perplexity
    Some(0.1),    // lambda (density weight)
);

// Run den-SNE
let embedding = densne(
    data.as_ref(),
    None,        // precomputed kNN (None = compute internally)
    &params,
    "bh",        // approximation: "barnes_hut" | "bh", or "fft" (fft_tsne feature)
    42,          // seed
    1,           // verbose -> light levels of verbosity
)
.unwrap();
```

Both `densmap` and `densne` return `ManifoldsError::DegenerateLocalRadii` if
every point ends up with the same local radius, since the correlation is then
undefined.

### Using Precomputed k-NN

Every entry point takes a precomputed k-nearest neighbour graph, which pays off
when you run several embeddings over the same data:

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;

let (data, _) = generate_clustered_data(500, 50, 5, 42);

// Compute k-NN once
let nn_params = NearestNeighbourParams::default();
let (knn_indices, knn_dist) = run_ann_search(
    data.as_ref(),
    15,              // k
    "hnsw".to_string(),
    &nn_params,
    42,              // seed
    1,               // verbose -> light levels of verbosity
)
.unwrap();

// Use precomputed k-NN for UMAP
let params = UmapParams::new_default_2d(None, None);
let embedding = umap(
    data.as_ref(),
    Some((knn_indices.clone(), knn_dist.clone())),
    &params,
    42,
    0,              // no verbose
)
.unwrap();
```

The indices and distances must exclude the point itself, which is what
`run_ann_search` gives you. The graph is taken as-is, no `k` check, so make it
match the `k` in the parameter struct. PaCMAP is the exception: it needs
`k >= mn_candidate_end` for mid-near pair sampling.

### GPU-Accelerated UMAP and tSNE (requires `gpu` feature)

GPU kernels run in `f32`, so the data has to be `f32` as well.

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use faer::Mat;

// Generate synthetic clustered data and cast it down to f32
let (data, _) = generate_clustered_data(
    10_000,  // n_samples
    50,      // dimensionality
    5,       // n_clusters
    42,      // seed
);
let data = Mat::<f32>::from_fn(data.nrows(), data.ncols(), |i, j| data[(i, j)] as f32);

// Configure GPU UMAP parameters
// Defaults to the "adam_gpu" optimiser
let params = UmapParamsGpu::new_default_2d(
    Some(0.5),   // min_dist
    Some(1.0),   // spread
);

// Run GPU UMAP. ann_type defaults to "nndescent_gpu"; alternatives are
// "exhaustive_gpu" (deterministic, brute force) and "ivf_gpu".
let device = WgpuDevice::default();
let embedding = umap_gpu::<f32, WgpuRuntime>(
    data.as_ref(),
    None,        // precomputed kNN
    &params,
    device,
    42,          // seed
    1,           // verbose -> light levels of verbosity
)
.unwrap();
```

GPU t-SNE works analogously, though here only the kNN search is
GPU-accelerated:

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use faer::Mat;

let (data, _) = generate_clustered_data(10_000, 50, 5, 42);
let data = Mat::<f32>::from_fn(data.nrows(), data.ncols(), |i, j| data[(i, j)] as f32);

let params = TsneParamsGpu::new_default_2d(
    Some(30.0),   // perplexity
);

let device = WgpuDevice::default();
let embedding = tsne_gpu::<f32, WgpuRuntime>(
    data.as_ref(),
    None,
    &params,
    "bh",        // "bh" or "fft" (fft requires fft_tsne feature)
    device,
    42,
    1,           // verbose -> light levels of verbosity
)
.unwrap();
```

densMAP has a GPU path too, via `densmap_gpu` and `DensmapParamsGpu`:

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use faer::Mat;

let (data, _) = generate_clustered_data(10_000, 50, 5, 42);
let data = Mat::<f32>::from_fn(data.nrows(), data.ncols(), |i, j| data[(i, j)] as f32);

let params = DensmapParamsGpu::new_default_2d(
    Some(0.5),   // min_dist
    Some(1.0),   // spread
    Some(2.0),   // lambda (density weight)
);

let device = WgpuDevice::default();
let embedding = densmap_gpu::<f32, WgpuRuntime>(
    data.as_ref(),
    None,
    &params,
    device,
    42,
    1,           // verbose -> light levels of verbosity
)
.unwrap();
```

With the default `"adam_gpu"` optimiser the density term runs on the device as
well: one kernel computes the embedding radii, another writes the per-node
correlation sensitivities. The radii are read back in between so the
correlation statistics can be accumulated in `f64` on the host, since summing
`n` log-radii in `f32` loses its significant digits. That readback is a sync
point, but it only happens over the final 30% of the epochs. Pick any of the
other optimisers (`"sgd"`, `"adam"`, `"adam_parallel"`) and the density path
falls back to CPU, with only the kNN search on the GPU.

To use CUDA instead of wgpu, swap `WgpuRuntime`/`WgpuDevice` for
`CudaRuntime`/`CudaDevice` from `cubecl::cuda`. On Linux CI, Vulkan via
`mesa-vulkan-drivers` (lavapipe) is the simplest path for headless testing.

### Parametric UMAP Example (requires `parametric` feature)

Parametric UMAP learns a neural network encoder that can transform new data
points:

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;
use burn::backend::flex::{Flex, FlexDevice};
use burn::backend::Autodiff;
use faer::Mat;

type Backend = Autodiff<Flex<f32>>;

// Generate synthetic clustered data (f32, to match the backend)
let (data, labels) = generate_clustered_data(1000, 50, 5, 42);
let data = Mat::<f32>::from_fn(data.nrows(), data.ncols(), |i, j| data[(i, j)] as f32);

// Configure the training parameters
let fit_params = TrainParametricParams::from_min_dist_spread(
    0.1,       // min_dist
    1.0,       // spread
    0.0,       // corr_weight
    None,      // learning rate
    Some(100), // n_epochs
    Some(16),  // batch_size
    None,      // negative sample rate
);

let params = ParametricUmapParams::new(
    2,                                   // n_dim (output dimensions)
    15,                                  // k (number of neighbours)
    "hnsw".into(),                       // ann_type
    vec![128, 64],                       // hidden_layers (encoder architecture)
    NearestNeighbourParams::default(),   // nn_params
    UmapGraphParams::default(),          // umap_graph_params
    fit_params,                          // training parameters
);

// Set up device
let device = FlexDevice;

// Train parametric UMAP
let embedding = parametric_umap::<f32, Backend>(
    data.as_ref(),
    None,        // precomputed kNN (None = compute internally)
    &params,
    &device,
    42,          // seed
    1,           // verbose -> light levels of verbosity
)
.unwrap();
```

`parametric_umap` throws the encoder away once it has the embedding. Want to
keep it, to embed new points later or write it to disk? Use
`train_parametric_umap_model`, which returns `(embedding, TrainedUmapModel)`.

### PHATE Example

PHATE is well-suited for data with continuous structure and branching
trajectories, such as single-cell differentiation data.

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;

// Generate a synthetic branching trajectory
let branches = generate_example_branches(&TrajectoryTopology::DeepBifurcation);
let (data, branch_assignments) = generate_trajectory(
    1000,        // n_samples
    &branches,   // branch topology
    50,          // dimensionality
    0.5,         // noise
    42,          // seed
);

// Configure PHATE parameters
let diffusion_params = PhateDiffusionParams::new(
    Some(40.0),              // decay
    1.0,                     // bandwidth_scale
    1e-4,                    // thresh
    "average".to_string(),   // graph_symmetry
    None,                    // n_landmarks (None = full operator)
    "spectral".to_string(),  // landmark_method
    None,                    // n_svd
    None,                    // t_max (None = default cap)
    None,                    // t_custom (None = auto-select via VNE knee)
    1.0,                     // gamma
);

let params = PhateParams::new(
    2,                                   // n_dim (output dimensions)
    5,                                   // k (number of neighbours)
    "kmknn".to_string(),                 // ann_type
    NearestNeighbourParams::default(),   // ann_params
    diffusion_params,
    "sgd_dense".to_string(),             // mds_method
    None,                                // mds_iter
    true,                                // randomised
);

// Run PHATE
let embedding = phate(
    data.as_ref(),
    None,        // precomputed kNN (None = compute internally)
    params,      // note: consumed by value, not borrowed
    42,          // seed
    1,           // verbose -> light levels of verbosity
)
.unwrap();
```

`PhateParams::new_default_2d(Some(k))` gives the same thing with sensible
defaults if you only want to tune the neighbour count.

### PaCMAP Example

PaCMAP preserves both local and global structure via three pair types (near,
mid-near, and further pairs) and a phased optimisation schedule. **Warning:**
Prior to version `"0.3.6"` there is a nasty bug in the PaCMAP implementation.
Please use `"0.3.6"` and later for correct PaCMAP.

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;

// Generate synthetic clustered data
let (data, labels) = generate_clustered_data(1000, 50, 5, 42);

// Configure PaCMAP parameters
let params = PacmapParams::default();

// Run PaCMAP
let embedding = pacmap(
    data.as_ref(),
    None,    // precomputed kNN (None = compute internally)
    &params,
    42,      // seed
    1,       // verbose -> light levels of verbosity
)
.unwrap();
```

The defaults are 10 near, 5 mid-near and 20 further pairs, PCA initialisation
and the `"adam_parallel"` optimiser. `PacmapParams::new_default_2d(Some(n))`
gives the same with a different near-pair count.

### Diffusion Maps Example

Classical diffusion maps embed the data via the top eigenvectors of a
row-stochastic diffusion operator, optionally with anisotropic normalisation
to correct for non-uniform sampling density. Setting `alpha_norm = 1.0`
recovers the Laplace-Beltrami operator; `0.5` gives the Fokker-Planck
operator; `0.0` gives the normalised graph Laplacian.

```rust
use manifolds_rs::prelude::*;
use manifolds_rs::*;

// Generate synthetic clustered data
let (data, _) = generate_clustered_data(1000, 50, 5, 42);

// Configure diffusion maps parameters
let params = DiffusionMapsParams::new(
    2,                                   // n_dim (output dimensions)
    5,                                   // k (number of neighbours)
    "kmknn".to_string(),                 // ann_type
    NearestNeighbourParams::default(),   // ann_params
    1.0,                                 // bandwidth_scale
    1e-4,                                // thresh
    "add".to_string(),                   // graph_symmetry
    1.0,                                 // alpha_norm (Laplace-Beltrami)
    PhateTime::Auto { t_max: 100 },      // t (Auto = VNE knee up to t_max)
    None,                                // n_landmarks (None = full operator)
    "spectral".to_string(),              // landmark_method
    None,                                // n_svd
);

// Run diffusion maps
let embedding = diffusion_maps(
    data.as_ref(),
    None,        // precomputed kNN (None = compute internally)
    params,      // note: consumed by value, not borrowed
    42,          // seed
    1,           // verbose -> light levels of verbosity
)
.unwrap();

// embedding[0] contains the first non-trivial diffusion component
// embedding[1] contains the second non-trivial diffusion component
```

## Licence

MIT Licence

Copyright (c) 2025 Gregor Alexander Lueg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
