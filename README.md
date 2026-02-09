[![CI](https://github.com/GregorLueg/manifolds-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/manifolds-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/manifolds-rs.svg)](https://crates.io/crates/manifolds-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# manifolds-rs

High-performance manifold learning and dimensionality reduction algorithms
implemented in Rust. Contains for now

- **UMAP**
- **Parametric UMAP** (optional feature)
- **tSNE**
  - ***Barnes Hut tSNE*** (With a `O(n log n)` complexity).
  - ***Fast Fourier Transform-accelerated Interpolation-based t-SNE (Flt-SNE)***
  (optional feature; with a `O(n)` complexity for large datasets).

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
used via the prospective feature flag. The next one to implement is likely
[PHATE](https://pmc.ncbi.nlm.nih.gov/articles/PMC7073148/).

## Features

- **UMAP algorithm**: Complete implementation of the UMAP dimensionality
reduction algorithm with several optimisations: SGD, Adam and a parallelised
version of ADAM for increased optimisation speed.
- **tSNE algorithm**: Implementation of the Barnes-Hut accelerated version and
the FFT-accelerated version (optional).
- **Multiple ANN backends** via [`ann-search-rs`](https://crates.io/crates/ann-search-rs):
  - Annoy (Approximate Nearest Neighbours Oh Yeah) - good for smaller datasets.
  - HNSW (Hierarchical Navigable Small World) - good for larger datasets.
  - NNDescent (Nearest Neighbour Descent) - good for larger datasets.
- **Distance metrics**:
  - Euclidean
  - Cosine
  - Maybe more to come over time ... ?
- **Multiple initialisations**:
  - Graph Laplacian eigenvector-based initialisation using Lanczos iteration
  - Random initialisation
  - PCA-based initialisation with randomised SVD for veeery large data sets
- **Customisable parameters**: Full control over fuzzy simplicial set
construction, graph symmetrisation, and optimisation parameters for tSNE and
UMAP.
- **High performance**: Parallel processing with Rayon, efficient sparse matrix
operations, and optimised SGD and Adam optimisers for UMAP (for the latter also a
parallelised version...) and rapid optimisations for tSNE.
- **Synthetic datasets**: Some synthetic datasets are available for testing and
experimentation: Swiss role, clustered data and a tree-like structure.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
manifold-rs = "0.1.5"
```

If you want to enable parametric UMAP, please use:

```toml
[dependencies]
manifold-rs = { version = "0.1.5", features = [ "parametric" ] }
```

If you want to enable the FFT-accelerated version of tSNE, please use:

```toml
[dependencies]
manifold-rs = { version = "0.1.5", features = [ "fft_tsne" ] }
```

## Notes

Please use version `0.1.3` and higher. These ones are not extensively tested
against real data.

## Usage

### UMAP Example
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
    Some(3),      // n_jobs
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

Both algorithms support precomputed k-nearest neighbour graphs for efficiency when running multiple embeddings:
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
