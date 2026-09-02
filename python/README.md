# manifolds-rs

Dimensionality reduction for single-cell and computational biology. The
[Rust crate](https://github.com/GregorLueg/manifolds-rs) does the work; this is
a scikit-learn shaped layer over it.

Seven algorithms on the CPU, three of them with GPU variants where the neighbour
search and the Adam update move to the device. No CUDA runtime to install: the
GPU backend is wgpu, so it runs on Metal, Vulkan or DX12 and ships in the
ordinary wheel.

## Install

```bash
uv pip install manifolds-rs
```

Wheels are built for Linux x86_64 and macOS on both architectures, against
Python 3.10 and up.

## Thirty seconds

```python
import manifolds_rs as mf

X, labels = mf.datasets.clustered(20_000, dim=50, n_clusters=12)

embedding = mf.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(X)
```

Every estimator takes its parameters in the constructor, its data in `fit`, and
hands back an `(n_samples, n_components)` array. `get_params` and `set_params`
are there, so `clone` and `Pipeline` work without scikit-learn being an install
requirement.

## What's here

| Class | What it is for |
| --- | --- |
| `UMAP` | The default choice. Local structure, decent global structure from the spectral init. |
| `DensMAP` | UMAP with relative density preserved, so a dense region stays dense. |
| `TSNE` | Local structure above all else. Barnes-Hut, 2-D only. |
| `DensNE` | t-SNE with the same density correction. |
| `PHATE` | Continuous structure. Trajectories and branch points survive this. |
| `PaCMAP` | Global structure without leaning on a spectral init. Three pair types. |
| `DiffusionMaps` | The spectral embedding PHATE is built on. |
| `UMAPGpu`, `DensMAPGpu`, `TSNEGpu` | The same, with the neighbour search on the device. |

## Parameters

The knobs people actually turn are ordinary constructor arguments. The rest live
in frozen dataclasses, one per group, and anything you leave alone keeps the
crate's default:

```python
mf.UMAP(
    n_neighbors=30,
    ann="hnsw",
    nn_params=mf.NeighbourParams(m=32, ef_search=200),
    optim_params=mf.UmapOptim(gamma=1.5, neg_sample_rate=10),
)
```

Every parameter has exactly one home. If it is a constructor argument it is not
in the group, so there is never a question of which wins.

## Reusing the neighbour graph

Running several embeddings over the same data? Build the graph once. On anything
large the search is most of the runtime.

```python
ind, dist = mf.knn_graph(X, k=15, ann="hnsw")

a = mf.UMAP().fit_transform(X, knn_indices=ind, knn_distances=dist)
b = mf.PaCMAP().fit_transform(X, knn_indices=ind, knn_distances=dist)
```

## No transform

None of these projects new points, so there is no `transform`. That is the
crate's position rather than a gap in the bindings: embedding new data means
refitting on the combined set, which moves the existing coordinates too. Anyone
telling you otherwise is doing something to the new points that is not the same
operation the old ones went through.

## Precision

float32 in, float32 throughout; float64 in, float64 throughout. Anything else is
promoted to float64 rather than narrowed, so precision is never lost by
accident. The GPU estimators are the exception and cast to float32, because WGSL
has no float64 and the alternative is a failure inside a kernel.

## What is not in the wheel

FFT-accelerated t-SNE. It needs FFTW, a system library no manylinux container
carries, so `approx="fft"` raises unless you build the extension yourself with
the `fft_tsne` feature. Barnes-Hut is the default and is what the wheel does.

Parametric UMAP is in the crate but not yet bound.
