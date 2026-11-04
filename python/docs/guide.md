# Guide

The details that will otherwise bite you.

## Precision

float32 in, float32 out. float64 in, float64 out. Anything else, integers
included, is promoted to float64 rather than narrowed.

The design matrix picks the precision for the pipeline at any size, including
the optimiser. There is no separate knob for it.

Worth knowing on large data: float32 kept up. On real single cell PCA at 1M cells,
UMAP, t-SNE, PaCMAP and ForceAtlas2 scored the same in both precisions, while
float64 took up to 51% longer and 46% more memory. That is one dataset, and nothing
above 1M was tested.

Four exceptions:

- The neighbour search runs in float32 on every backend bar `"exhaustive"`, and
  the distances are cast back up. So `knn_graph` on float64 input returns float64
  with float32 precision in the values. Want float64 ground truth? Use
  `"exhaustive"`.
- PCA initialisation and the randomised SVDs always run in float64, then cast the
  result back. The float32 factorisations break down at about 520k samples.
- The GPU estimators cast to float32 at any size. That is the only silent
  narrowing of your input in the package.
- `approx="fft_3k_mlx"` optimises in float32 on the device, whatever you pass.

## Distances

`knn_graph` returns true distances in whatever metric you asked for. The
Euclidean backends compute squared distances internally, because that does not
change the ordering they sort on, and the square root is taken before the
distances reach you.

So a graph you build yourself and hand to `fit` should hold true distances too.

Metrics available: `"euclidean"`/`"l2"`, `"cosine"` and `"manhattan"`/`"l1"`.
`PHATE` and `PHATEGpu` take only the first two, and `DiffusionMaps` raises
`ValueError` on Manhattan with landmarks.

## Reusing the neighbour graph

Past a few hundred thousand points the neighbour search is most of the runtime.
Build it once:

```python
ind, dist = mf.knn_graph(X, k=60, ann="hnsw")

umap = mf.UMAP().fit_transform(X, knn_indices=ind[:, :15], knn_distances=dist[:, :15])
pacmap = mf.PaCMAP().fit_transform(X, knn_indices=ind, knn_distances=dist)
```

Three things to get right. Build the graph as wide as the widest consumer needs:
`3 * perplexity` for the t-SNE family, `n_near + 50` for `PaCMAP` and `6k + 5` for
`PHATE`, since its kernel keeps anything just past the `k`-th neighbour.

Then cut it down for `UMAP`, `DensMAP` and `ForceAtlas2`, as above. They use every
neighbour they are given and ignore `n_neighbors`, so a 60-wide graph is
`n_neighbors=60`, whatever the constructor says.

And the graph must be complete: a row
an approximate backend could not fill is padded with `-1`, and that is refused
rather than quietly dropped, because a hole in the graph is not something these
algorithms can embed.

Handing in the graph an estimator would have built itself produces a
bit-identical embedding. It is a shortcut, not a different algorithm.

## Choosing a neighbour backend

The default follows `k`. Estimators that ask for a small neighbourhood (UMAP,
densMAP, PHATE, diffusion maps, ForceAtlas2) default to `"nndescent"`. t-SNE,
den-SNE and DREAMS ask for `3 * perplexity` neighbours and PaCMAP for `n_near + 50`,
and those default to `"ivf"`, which holds up better at that size.

`"kmknn"` is exact and holds up well into the hundreds of thousands, which
surprises people who assume exact means slow. `"exhaustive"` is for ground
truth. `"hnsw"`, `"annoy"` and `"balltree"` are there for the cases where they
win.

Each has its own knobs in `NeighbourParams`, and only the ones belonging to your
chosen backend do anything. Setting `ef_search` while running `"annoy"` is
harmless and has no effect.

## Parameter groups

Every parameter has exactly one home. If it is a constructor argument on the
estimator it is not in the group, and vice versa. So:

- `metric` is on the estimator, not `NeighbourParams.dist_metric`.
- `n_epochs` and `learning_rate` are on the estimator, not `UmapOptim`.
- `min_dist` is on the estimator, not `UmapOptim`. It and `spread` feed the
  curve fit that produces `a` and `b`; letting you move `min_dist` without
  refitting them would be the one combination that is always wrong. Pinning `a`
  and `b` directly is still available in `UmapOptim` for anyone who means it.

A field left at `None` in a group is not sent at all and the crate's default
applies. A misspelled field is a `TypeError` naming the field.

## Nulls that mean something

Three parameters take `None` as a real value rather than "use the default":

- `PHATE(decay=None)` selects a binary connectivity kernel instead of the
  alpha-decay one.
- `PHATE(n_landmarks=None)` runs the full operator instead of landmarks.
- `PaCMAP(range_=None)` hands the range to the initialisation's own default.

Everywhere else `None` means "let the crate decide".

## Verbosity

`verbose=0` is quiet. `verbose=1` prints each stage with its timing, plus the
optimiser's epochs every 50. `verbose=2` adds the parameters, the neighbour index
internals and the landmark selection. Anything else raises `ValueError`.
Warnings print at every level, to stderr.

`verbose=1` or `2` writes progress to the *process* stdout, not `sys.stdout`. In
Jupyter that lands in the terminal running the kernel, not in the notebook. This
is the core printing directly and there is nothing this layer can do about it
short of capturing a file descriptor.

## Threads

The core is rayon throughout. By default it uses rayon's global pool, sized from
`RAYON_NUM_THREADS` or the core count.

```python
mf.set_num_threads(4)  # cap it, for a shared machine or a job scheduler
mf.num_threads()
mf.set_num_threads(0)  # back to the global pool
```

The GIL is released for the entire computation, so an embedding running in a
background thread does not block the interpreter.

## Reproducibility

**On the CPU**, same seed, same data, same parameters gives a bit-identical
embedding. The seed fixes the initialisation, the negative sampling and
anything randomised in the neighbour build.

**On the GPU, the coordinates can differ between runs at the same seed.** The
structure does not.

The source is the neighbour search, not the optimiser: the GPU Adam update is
bit-stable given a fixed graph, and the device searches are not always stable
at scale. Two runs agree on about 99.4% of neighbour slots; where they
disagree, the two candidates sit a median 0.4% apart in distance, and recall
against exhaustive ground truth matches to the fourth decimal.

What that means in practice, measured on 20k points in 50 dimensions across 12
clusters, two runs at the same seed:

- k-means on either embedding recovers the *same* partition. Adjusted Rand
  index between the two clusterings: 1.0000.
- Cluster separation matches: silhouette 0.849 versus 0.858.
- Individual points can swap places within a dense cluster, so
  point-for-point neighbour lists are not identical.

On a swiss roll, both runs came back bit-identical and preserved the manifold
equally (0.623 versus 0.628). So it is data-dependent rather than guaranteed
either way.

If you need the coordinates themselves to be reproducible, use the CPU
estimators, or build the graph once with `knn_graph` and hand it in. Do not turn a plot into a figure and expect the pixels back; do rely on the
clusters, the branches and the separations being there.

Across dtypes reproducibility does not hold either way: float32 and float64
runs of the same configuration will differ, as they should.

Across releases it does not hold either: 0.5.0 changed the eigensolver behind
the spectral initialisation, PHATE and diffusion maps, so coordinates differ
from 0.4.x at the same seed. 0.6.0 moved PCA initialisation and the randomised
SVDs to float64, so float32 runs differ from 0.5.0. It also moved the neighbour
search to float32, so float64 runs differ too.

## Errors

- `ValueError` for most things you can fix by changing an argument, including
  everything the neighbour search rejects.
- `ConvergenceError` when a spectral decomposition ran but did not get there,
  which usually means degenerate data: duplicate points, or a disconnected
  landmark set. It subclasses `ManifoldsRsError`, which is also what the rest of
  the core's errors come as, for example `DensNE(approx="fft_3k_mlx")`.
- `TypeError` for a bad parameter name or type.
- `NotFittedError` for reading `embedding_` before `fit`. It subclasses both
  `ValueError` and `AttributeError`, matching scikit-learn's.

## What is not in the wheel

The MLX backends. They link mlx-c dynamically, so they only come with a source
build on Apple Silicon; see [GPU](gpu.md#mlx-on-apple-silicon).

Parametric UMAP exists in the crate but is not yet bound.
