# Guide

The details that will otherwise bite you.

## Precision

float32 in, float32 throughout. float64 in, float64 throughout. Anything else,
integers included, is promoted to float64 rather than narrowed, so precision is
never lost by accident.

The design matrix picks the precision for the whole pipeline, including the
neighbour search and the optimiser. There is no separate knob for it.

The GPU estimators are the exception: they cast to float32 because WGSL has no
float64. That is the only silent narrowing anywhere in the package.

## Distances

`knn_graph` returns true distances in whatever metric you asked for. The
Euclidean backends compute squared distances internally, because that does not
change the ordering they sort on, and the square root is taken before the
distances reach you.

So a graph you build yourself and hand to `fit` should hold true distances too.

Metrics available: `"euclidean"`/`"l2"`, `"cosine"`, `"manhattan"`/`"l1"`.
`"euclidean"` and `"l2"` are the same metric and now produce the same
embedding; before 0.4 they did not, because the squared convention was
detected by comparing the metric name against the string `"euclidean"` and
`"l2"` fell through it.

## Reusing the neighbour graph

Past a few hundred thousand points the neighbour search is most of the runtime.
Build it once:

```python
ind, dist = mf.knn_graph(X, k=50, ann="hnsw")

umap = mf.UMAP(n_neighbors=15).fit_transform(X, knn_indices=ind, knn_distances=dist)
pacmap = mf.PaCMAP().fit_transform(X, knn_indices=ind, knn_distances=dist)
```

Two things to get right. Give the shared graph the widest `k` any consumer
needs: PaCMAP indexes into the list up to `mn_candidate_end`, 50 by default,
which is usually the binding constraint. And the graph must be complete: a row
an approximate backend could not fill is padded with `-1`, and that is refused
rather than quietly dropped, because a hole in the graph is not something these
algorithms can embed.

Handing in the graph an estimator would have built itself produces a
bit-identical embedding. It is a shortcut, not a different algorithm.

## Choosing a neighbour backend

`"kmknn"` is the default and is exact. It holds up well into the hundreds of
thousands, which surprises people who assume exact means slow.

Past that, `"nndescent"` or `"hnsw"`. `"exhaustive"` is for ground truth.
`"annoy"`, `"balltree"` and `"ivf"` are there for the cases where they win.

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

Two parameters take `None` as a real value rather than "use the default":

- `PHATE(decay=None)` selects a binary connectivity kernel instead of the
  alpha-decay one.
- `PaCMAP(range_=None)` means no initialisation range.

Everywhere else `None` means "let the crate decide".

## Verbosity

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
estimators, or turn the graph extraction off with
`NeighbourParamsGpu(extract_knn=False)`, which narrows but does not eliminate
it. Do not turn a plot into a figure and expect the pixels back; do rely on the
clusters, the branches and the separations being there.

Across dtypes reproducibility does not hold either way: float32 and float64
runs of the same configuration will differ, as they should.

## Errors

- `ValueError` for anything you can fix by changing an argument, including
  everything the neighbour search rejects.
- `ConvergenceError` when a spectral decomposition ran but did not get there,
  which usually means degenerate data: duplicate points, or a disconnected
  landmark set. Both subclass `ManifoldsRsError`.
- `TypeError` for a bad parameter name or type.
- `NotFittedError` for reading `embedding_` before `fit`. It subclasses both
  `ValueError` and `AttributeError`, matching scikit-learn's.

## What is not in the wheel

FFT-accelerated t-SNE. It needs FFTW, a system library no manylinux container
carries, so `approx="fft"` needs an extension you built yourself with the
`fft_tsne` feature. Barnes-Hut is the default and is what the wheel does.

Parametric UMAP exists in the crate but is not yet bound.
