[![CI](https://github.com/GregorLueg/manifolds-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/manifolds-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/manifolds-rs.svg)](https://crates.io/crates/manifolds-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# manifolds-rs

High-performance manifold learning and dimensionality reduction algorithms
implemented in Rust. Contains as for now:

- **UMAP**
  - Has different optimisers: SGD (traditional), Adam and a parallelised
  version of Adam for very fast fitting.
- **Parametric UMAP** (optional feature)
- **tSNE**
  - ***Barnes Hut tSNE*** (With a `O(n log n)` complexity).
  - ***Fast Fourier Transform-accelerated Interpolation-based t-SNE (Flt-SNE)***
  (optional feature; with a `O(n)` complexity for large datasets).
- **PHATE**
- **PaCMAP**

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
[PaCMAP](https://arxiv.org/abs/2012.04456) has been also implemented.
Changelog can be found [here](https://github.com/GregorLueg/manifolds-rs/blob/main/docs/news.md))

## Features

- **UMAP algorithm**: Complete implementation of the UMAP dimensionality
reduction algorithm with several optimisations: SGD, Adam and a parallelised
version of ADAM for increased optimisation speed.
- **tSNE algorithm**: Implementation of the Barnes-Hut accelerated version and
the FFT-accelerated version (optional).
- **PHATE**: Implementation of Potential of Heat-diffusion for Affinity-based
Trajectory Embedding with different landmark methods.
- **Multiple ANN backends** via [`ann-search-rs`](https://crates.io/crates/ann-search-rs):
  - *Exhaustive* - If you want precise results and have a small data set in
    which the approximate nearest neighbour index building is actually slower.
  - *BallTree* - A small, fast index for smaller data sets with lower
    dimensions.
  - *Annoy (Approximate Nearest Neighbours Oh Yeah)* - Good for medium low-
    dimensionality datasets.
  - *NNDescent (Nearest Neighbour Descent)* - good for larger datasets with
    higher dimensionality.
  - *HNSW (Hierarchical Navigable Small World)* - good for (very) larger
    datasets with higher dimensionality.
- **Distance metrics**:
  - Euclidean
  - Cosine
  - Maybe more to come over time ... ?
- **Multiple initialisations**:
  - Graph Laplacian eigenvector-based initialisation using Lanczos iteration
  - Random initialisation
  - PCA-based initialisation
- **Customisable parameters**: Full control over fuzzy simplicial set
  construction, graph symmetrisation, and optimisation parameters for tSNE,
  UMAP and PHATE.
- **High performance**: Parallel processing with Rayon, efficient sparse matrix
  operations, cache-friendly structures and optimised SGD and Adam optimisers
  for UMAP (for the latter also a parallelised version...) and fast optimisers
  for tSNE and also PHATE.
- **Synthetic datasets**: Some synthetic datasets are available for testing and
  experimentation: Swiss role, clustered data and a trajectory-like structure.

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

## Notes

Please use version `0.1.3` and higher. These ones are not extensively tested
against real data.

## Usage

### R package

This crate powers [manifoldsR](https://gregorlueg.github.io/manifoldsR/index.html),
an R package leveraging the incredible speed that Rust offers.

### UMAP Example

Below are examples of how to use UMAP.

```rust
use manifolds_rs::prelude::*;

// Generate synthetic clustered data
let (data, labels) = generate_clustered_data(
    1000,  // n_samples
    50,    // dimensionality
    5,     // n_clusters
    42,    // seed
);

// Configure UMAP parameters
let params = UmapParams::default_2d(
    Some(2),     // n_dim (output dimensions)
    Some(15),    // k (number of neighbours)
    Some(0.1),   // min_dist
    Some(1.0),   // spread
);

// Run UMAP
let embedding = umap(
    data.as_ref(),
    None,        // precomputed kNN (None = compute internally)
    &params,
    42,          // seed
    true,        // verbose
);

// embedding[0] contains x-coordinates
// embedding[1] contains y-coordinates
```

### t-SNE Example

Below are examples of how to use t-SNE.

```rust
use manifolds_rs::prelude::*;

// Generate synthetic clustered data
let (data, labels) = generate_clustered_data(
    1000,  // n_samples
    50,    // dimensionality
    5,     // n_clusters
    42,    // seed
);

// Configure t-SNE parameters
let params = TsneParams::new(
    Some(2),      // n_dim (output dimensions)
    Some(30.0),   // perplexity
    Some(1e-4),   // init_range
    Some(200.0),  // learning_rate
    Some(1000),   // n_epochs
    None,         // ann_type (None = default "hnsw")
    Some(0.5),    // theta (Barnes-Hut angle)
    Some(3),      // n_interp_points (FFT interpolation grid points)
);

// Run t-SNE (Barnes-Hut)
let embedding = tsne(
    data.as_ref(),
    None,        // precomputed kNN (None = compute internally)
    &params,
    "bh",        // approximation type: "bh" or "fft" (requires fft_tsne feature)
    42,          // seed
    true,        // verbose
);

// embedding[0] contains x-coordinates
// embedding[1] contains y-coordinates
```

### Using Precomputed k-NN

Both algorithms support precomputed k-nearest neighbour graphs for efficiency
when running multiple embeddings:

```rust
use manifolds_rs::prelude::*;

let (data, _) = generate_clustered_data(500, 50, 5, 42);

// Compute k-NN once
let nn_params = NearestNeighbourParams::default();
let (knn_indices, knn_dist) = run_ann_search(
    data.as_ref(),
    15,              // k
    "hnsw".to_string(),
    &nn_params,
    42,              // seed
    true             // verbosity
);

// Use precomputed k-NN for UMAP
let params = UmapParams::default_2d(None, Some(15), None, None);
let embedding = umap(
    data.as_ref(),
    Some((knn_indices.clone(), knn_dist.clone())),
    &params,
    42,
    false,
);
```

### Parametric UMAP Example (requires `parametric` feature)

Parametric UMAP learns a neural network encoder that can transform new data
points:

```rust
use manifolds_rs::prelude::*;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;

type Backend = Autodiff<NdArray<f64>>;

// Generate synthetic clustered data
let (data, labels) = generate_clustered_data(
    1000,  // n_samples
    50,    // dimensionality
    5,     // n_clusters
    42,    // seed
);

// Configure parametric UMAP
let fit_params = TrainParametricParams::from_min_dist_spread(
    0.1,       // min_dist
    1.0,       // spread
    0.0,       // correlation_weight
    None,      // negative_sample_rate
    Some(16),  // batch_size
    Some(100), // n_epochs
    None,      // learning_rate
);

let params = ParametricUmapParams::new(
    Some(2),              // n_dim (output dimensions)
    Some(15),             // n_neighbours
    Some("hnsw".into()),  // ann_type
    Some(vec![128, 64]),  // hidden_layers (neural network architecture)
    None,                 // nn_params
    None,                 // umap_graph_params
    Some(fit_params),     // training parameters
);

// Set up device
let device = NdArrayDevice::Cpu;

// Train parametric UMAP
let embedding = parametric_umap::<f64, Backend>(
    data.as_ref(),
    None,        // precomputed kNN (None = compute internally)
    &params,
    &device,
    42,          // seed
    true,        // verbose
);

// embedding[0] contains x-coordinates
// embedding[1] contains y-coordinates
```

### PHATE Example

PHATE is well-suited for data with continuous structure and branching
trajectories, such as single-cell differentiation data.

```rust
use manifolds_rs::prelude::*;

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
let params = PhateParams::new(
    Some(2),     // n_dim (output dimensions)
    Some(5),     // k (number of neighbours)
    None,        // ann_type (None = default "hnsw")
    None,        // decay (None = default 40.0)
    None,        // bandwidth_scale (None = default 1.0)
    None,        // graph_symmetry (None = default "average")
    None,        // t_max (None = auto)
    None,        // gamma (None = default 1.0)
    None,        // n_landmarks (None = full operator)
    None,        // landmark_method (None = default "spectral")
    None,        // n_svd
    None,        // t_custom
    None,        // mds_method (None = default "sgd_dense")
    None,        // mds_iter
    None,        // randomised (None = default true)
);

// Run PHATE
let embedding = phate(
    data.as_ref(),
    None,        // precomputed kNN (None = compute internally)
    params,      // note: consumed by value, not borrowed
    42,          // seed
    true,        // verbose
);

// embedding[0] contains x-coordinates
// embedding[1] contains y-coordinates
```

### PaCMAP Example

PaCMAP preserves both local and global structure via three pair types (near,
mid-near, and further pairs) and a phased optimisation schedule.
```rust
use manifolds_rs::prelude::*;

// Generate synthetic clustered data
let (data, labels) = generate_clustered_data(
    1000,  // n_samples
    50,    // dimensionality
    5,     // n_clusters
    42,    // seed
);

// Configure PaCMAP parameters
let params = PacmapParams::default();

// Run PaCMAP
let embedding = pacmap(
    data.as_ref(),
    None,    // precomputed kNN (None = compute internally)
    &params,
    42,      // seed
    true,    // verbose
);

// embedding[0] contains x-coordinates
// embedding[1] contains y-coordinates
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
