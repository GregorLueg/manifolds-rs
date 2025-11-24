# manifolds-rs

High-performance manifold learning and dimensionality reduction algorithms 
implemented in Rust.

## Description

Rust implementation of UMAP (Uniform Manifold Approximation and Projection), a 
manifold learning technique for dimensionality reduction. UMAP is a commonly 
used method for visualising high-dimensional biological data, but not without
[controversy](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011288).
This package implements a fast UMAP version in Rust. In the future other
versions might be implemented (for example parametric UMAP).

## Features

- **UMAP algorithm**: Complete implementation of the UMAP dimensionality 
reduction algorithm
- **Multiple ANN backends**: 
  - Annoy (Approximate Nearest Neighbours Oh Yeah)
  - HNSW (Hierarchical Navigable Small World)
  - NNDescent (Nearest Neighbour Descent)
- **Distance metrics**:
  - Euclidean
  - Cosine
- **Spectral initialisation**: Graph Laplacian eigenvector-based initialisation 
using Lanczos iteration
- **Customisable parameters**: Full control over fuzzy simplicial set 
construction, graph symmetrisation, and optimisation
- **High performance**: Parallel processing with Rayon, efficient sparse matrix
operations, and optimised SGD

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
manifold-rs = "*"  # always get the latest version
ann-search-rs = "*"  # required for ANN functionality
faer = "*"  # required for matrix operations
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