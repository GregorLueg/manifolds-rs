# GPU

Three estimators run part of the work on the device: `UMAPGpu`, `DensMAPGpu` and
`TSNEGpu`. The backend is wgpu, so there is no CUDA runtime to install and the
same wheel works on Metal, Vulkan and DX12.

```python
import manifolds_rs as mf

mf.gpu_available()  # True only if this build has GPU support AND an adapter exists
```

That one call answers both questions, which is what a caller choosing between
`UMAP` and `UMAPGpu` actually wants to know.

## What actually moves to the device

Not everything. The neighbour search and the Adam update run on the GPU; the
graph construction and the initialisation stay on the CPU. For t-SNE it depends
on `approx`: with `"barnes_hut"` the repulsion stays on the CPU, with
`"fft_3k_gpu"` the whole optimisation runs on the device.

```python
mf.TSNEGpu(approx="fft_3k_gpu").fit_transform(X)
```

`"fft_3k_gpu"` is FFT-interpolated repulsion with three kernels (`q`, `q^2 dx`,
`q^2 dy`) instead of the four-term expansion, with the embedding resident on the
device between epochs. There is no CPU FFT t-SNE in the wheel (see
[the guide](guide.md#what-is-not-in-the-wheel)), so this is the only FFT path
from Python.

Otherwise the win depends on whether the neighbour search was your bottleneck.
On a few thousand points it will not be, and the transfers will make the GPU
path slower than the CPU one. Past a few hundred thousand it usually is. Measure
on your own data and hardware before switching; on Apple Silicon in particular
the GPU rarely blows the CPU out of the water for these workloads.

## float32 only

WGSL has no float64. `fit` casts a float64 array down rather than letting it
fail somewhere inside a kernel, and that is the only silent narrowing this
package performs anywhere. If that matters for your data, stay on the CPU
estimators, which run in whatever precision you hand them.

## Reproducibility

GPU embeddings are reproducible in structure, not in coordinates. The device
neighbour searches are not always bit-stable at scale, and 500 epochs of Adam
turn a 0.16% difference in the graph into visibly different coordinates. The
clusters, the branches and the separations come back; the exact positions may
not. See [the guide](guide.md#reproducibility) for the measurements.

The GPU Adam update itself is bit-stable given a fixed graph, so handing in a
precomputed graph makes a GPU run reproducible too.

## Different neighbour backends

The device backends are not the CPU ones with a suffix. `"nndescent_gpu"` builds
a CAGRA graph and is the default because it is the one that scales;
`"ivf_gpu"` and `"exhaustive_gpu"` are the other two. Passing a CPU-only name
like `"hnsw"` is an error rather than a silent fallback.

Their knobs live in [`NeighbourParamsGpu`](api/params.md), which is a different
set from the CPU `NeighbourParams`: CAGRA build degrees and beam-search budgets
have no CPU counterpart.

```python
mf.UMAPGpu(
    n_neighbors=15,
    ann="nndescent_gpu",
    nn_params=mf.NeighbourParamsGpu(k_build=64, beam_width=32),
)
```

Under `TSNEGpu` with `nndescent_gpu`, leaving `NeighbourParamsGpu.k` unset
backfills the graph degree to `3 * perplexity`, so it is sized for the query
t-SNE actually makes.

## Mixing CPU and GPU

The neighbour graph is just two arrays, so you can build it wherever it is
cheapest and hand it to whichever estimator you like:

```python
ind, dist = mf.knn_graph(X, k=15, ann="hnsw")  # CPU search
embedding = mf.UMAPGpu().fit_transform(X, knn_indices=ind, knn_distances=dist)
```

That skips the GPU search entirely and leaves only the Adam update on the
device, which is occasionally the right split.

## Headless machines and CI

`gpu_available()` returns `False` on a box with no adapter rather than raising,
and the GPU estimators are still importable there. That is deliberate: a hosted
CI runner can prove the GPU wheel compiles and degrades cleanly without having a
device. It cannot prove the kernels are correct. That needs a runner with a GPU.
