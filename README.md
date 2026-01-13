[![CI](https://github.com/GregorLueg/manifolds-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/manifolds-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/ann-search-rs.svg)](https://crates.io/crates/manifolds-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# manifolds-rs

High-performance manifold learning and dimensionality reduction algorithms 
implemented in Rust. Contains for now

- **UMAP**
- **Parametric UMAP**
- **Barnes Hut tSNE**

## Description

Rust implementations of various methods to project data onto two dimensions,
especially [UMAP](https://arxiv.org/abs/1802.03426) and 
[tSNE (Barnes-Hut implementation)](https://arxiv.org/abs/1301.3342).
These are typically used methods for visualising high-dimensional biological 
data, but not without [controversy](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011288).
The crate also provides via the Burn DL framework [parametric UMAP](https://arxiv.org/abs/2009.12981)
that can be optionally be used via the prospective feature flag.

## Features

- **UMAP algorithm**: Complete implementation of the UMAP dimensionality 
reduction algorithm
- **tSNE algorithm**: Implementation of the Barnes-Hut accelerated version. 
A potential future avenue might be to implement the FFT-accelerated tSNE, see
[Linderman, et al.](https://www.nature.com/articles/s41592-018-0308-4).
- **Multiple ANN backends** via [`ann-search-rs`](https://crates.io/crates/ann-search-rs): 
  - Annoy (Approximate Nearest Neighbours Oh Yeah)
  - HNSW (Hierarchical Navigable Small World)
  - NNDescent (Nearest Neighbour Descent)
- **Distance metrics**:
  - Euclidean
  - Cosine
- **Multiple initialisations**: 
  - Graph Laplacian eigenvector-based initialisation using Lanczos iteration
  - Random initialisation
  - PCA-based initialisation with randomised SVD for veeery large data sets
- **Customisable parameters**: Full control over fuzzy simplicial set 
construction, graph symmetrisation, and optimisation.
- **High performance**: Parallel processing with Rayon, efficient sparse matrix
operations, and optimised SGD and Adam optimisers for UMAP (for the latter also a 
parallelised version...) and rapid optimisations for tSNE.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
manifold-rs = "0.1.4"  
```

If you want to enable parametric UMAP, please use:

```toml
[dependencies]
manifold-rs = { version = "0.1.4", features = [ "parametric" ] }  
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