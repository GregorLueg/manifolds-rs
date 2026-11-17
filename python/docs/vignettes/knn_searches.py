# %% [markdown]
# # Nearest neighbour searches
#
# *Python port of the R vignette [Nearest neighbour searches in manifoldsR](https://gregorlueg.github.io/manifoldsR/articles/knn_searches.html). Original author: Gregor Lueg.*
#
# kNN graphs sit underneath nearly every method in this package. UMAP, densMAP,
# t-SNE, PaCMAP, LocalMAP, PHATE, diffusion maps and ForceAtlas2 all start from
# one. This vignette walks through the backends, what each is good at, how
# exact the approximate ones are, and how to build a graph once and hand it to
# several estimators instead of recomputing it every time.

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import manifolds_rs as mf

K = 10

# %% [markdown]
# ## Exhaustive search
#
# The simplest one. A flat index with SIMD distance kernels, so it is cache
# friendly and on small data it beats the approximate searches, which have to
# build an index first. `knn_graph` hands back indices and distances as two
# `(n_samples, k)` arrays, self excluded.

# %%
X, labels = mf.datasets.clustered(1_000)

indices, distances = mf.knn_graph(X, k=K, ann="exhaustive")

print(indices.shape, indices.dtype, distances.dtype)
indices[:3]

# %% [markdown]
# Indices first, distances second. That is the opposite of scikit-learn's
# `kneighbors`, and it is the order every estimator takes them in.
#
# An exact alternative is
# [kMkNN](https://ieeexplore.ieee.org/document/6033373) (k-means for k-nearest
# neighbours). It clusters the data with k-means and uses the triangle
# inequality to skip most of the distance computations. Very fast on data with
# strong structure and modest dimensionality, and the default of `knn_graph`.

# %%
indices_kmknn, _ = mf.knn_graph(X, k=K, ann="kmknn")

np.array_equal(indices, indices_kmknn)

# %% [markdown]
# ## Approximate searches
#
# The package is built for large data and pulls its approximate searches from
# [ann-search-rs](https://github.com/GregorLueg/ann-search-rs). Five of them
# are exposed, on top of the two exact ones:
#
# | Method | Class | Use case |
# | --- | --- | --- |
# | `"balltree"` | Tree | Small data, low dimensionality. |
# | `"annoy"` | Tree | A single-cell classic. Scales to a few hundred thousand points at low dimensionality. |
# | `"hnsw"` | Graph | Large data (500k and up), any dimensionality. Slower build, fast queries. The build uses a benign race. |
# | `"nndescent"` | Graph | The all-rounder from 100k points on. Default for UMAP, densMAP, ForceAtlas2 and diffusion maps. |
# | `"ivf"` | Cluster | The index behind FAISS. Large data, any dimensionality. `n_probes` trades recall for speed. Default for the t-SNE family, PaCMAP, LocalMAP and PHATE. |
#
# Every backend bar `"exhaustive"` searches in float32, whatever you hand it,
# and casts the distances back to your dtype. Same recall, less memory traffic.
# The knobs per backend live in `NeighbourParams`; only the fields of the
# backend you picked do anything.
#
# Default wheels add `"rinnd"`, the NN-Descent from
# [rinnd](https://github.com/lmcinnes/rinnd). Its SIMD kernels target x86 (AVX2),
# so try it there; it is not in the timings below. `nnd_n_trees` sets the seeding
# trees for both NN-Descents.

# %%
mf.NeighbourParams()

# %% [markdown]
# ### Speed
#
# Every backend at three sizes, one run each.

# %%
BACKENDS = ["exhaustive", "kmknn", "balltree", "annoy", "hnsw", "nndescent", "ivf"]


def timed(fn) -> float:
    """Wall time of one call, in seconds."""
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def time_backends(X: np.ndarray) -> dict[str, float]:
    """Wall time of every backend on `X`."""
    params = mf.NeighbourParams(n_probes=25)
    return {
        ann: timed(lambda ann=ann: mf.knn_graph(X, k=K, ann=ann, nn_params=params))
        for ann in BACKENDS
    }


speed = {
    f"{n // 1000}k": time_backends(mf.datasets.clustered(n)[0])
    for n in (1_000, 50_000, 100_000)
}

pl.DataFrame(
    {"method": list(speed["1k"]), **{n: list(t.values()) for n, t in speed.items()}}
).with_columns(pl.exclude("method").round(3))

# %% [markdown]
# Times in seconds. On 1k points nothing matters: exhaustive is as fast as
# anything, since every other backend has to build an index first and SIMD does
# the `O(N^2)` in no time at that size. From 50k on NN-Descent and IVF are the
# fastest, and the gap grows with N. kMkNN, which is exact, is close behind at
# 50k and takes a quarter to a sixth of the exhaustive time on this tightly
# clustered data.
#
# Annoy and HNSW are no faster than exhaustive here: a one-off self-query at
# `k = 10` never pays back their index build. They earn their keep at larger N,
# or when you query the same index again and again.

# %% [markdown]
# ### Precision
#
# Approximate means approximate. Two numbers against the exhaustive ground
# truth on 25k points:
#
# - Recall@k: fraction of the true `k` neighbours found, in any order.
# - Mean distance ratio: distance to the found neighbour over the distance to
#   the true one at the same rank. `1.0` is perfect, and it tells you how wrong
#   the misses are.

# %%
X_25k, _ = mf.datasets.clustered(25_000)
true_idx, true_dist = mf.knn_graph(X_25k, k=K, ann="exhaustive")


def precision(ann: str) -> dict[str, float | str]:
    """Recall@k and mean distance ratio of one backend against exhaustive."""
    idx, dist = mf.knn_graph(
        X_25k, k=K, ann=ann, nn_params=mf.NeighbourParams(n_probes=25)
    )
    hits = sum(len(set(a) & set(b)) for a, b in zip(idx, true_idx, strict=True))
    return {
        "method": ann,
        "recall": hits / true_idx.size,
        "dist_ratio": float(np.mean(dist / true_dist)),
    }


pl.DataFrame([precision(ann) for ann in BACKENDS[1:]]).with_columns(
    pl.col("recall", "dist_ratio").round(4)
)

# %% [markdown]
# Look at the distance ratio before you panic about a recall number. The
# synthetic clusters are tight, so lots of points sit at nearly identical
# distances and swapping one for another costs nothing. For manifold learning
# that is exactly the kind of miss you don't care about. For a much deeper
# comparison of the methods, see the
# [ann-search-rs benchmarks](https://github.com/GregorLueg/ann-search-rs).

# %% [markdown]
# ## Build once, embed many times
#
# On anything large the neighbour search is most of the runtime. Running UMAP
# with three different `min_dist` values, plus t-SNE, plus PaCMAP? Build the
# graph once and pass it in. Each estimator takes `knn_indices` and
# `knn_distances` in `fit` / `fit_transform`.
#
# Mind the width. UMAP takes `n_neighbors` neighbours and cuts a wider graph
# down (with a warning on stderr). t-SNE wants `3 * perplexity`. PaCMAP and
# LocalMAP want `n_near + 50`, because the near pairs are the best `n_near` of a
# wider search by locally scaled distance. So build the widest one you need.

# %%
X_traj, labels_traj = mf.datasets.trajectory(10_000)

idx, dist = mf.knn_graph(X_traj, k=90, ann="nndescent")

embeddings = {
    "UMAP": mf.UMAP().fit_transform(
        X_traj, knn_indices=idx[:, :15], knn_distances=dist[:, :15]
    ),
    "t-SNE": mf.TSNE(perplexity=30.0).fit_transform(
        X_traj, knn_indices=idx, knn_distances=dist
    ),
    "PaCMAP": mf.PaCMAP().fit_transform(
        X_traj, knn_indices=idx[:, :60], knn_distances=dist[:, :60]
    ),
}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
for ax, (name, emb) in zip(axes, embeddings.items(), strict=True):
    ax.scatter(emb[:, 0], emb[:, 1], c=labels_traj, s=1, cmap="tab10", alpha=0.6)
    ax.set_title(f"{name} on one shared graph")
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()

# %% [markdown]
# One search, three embeddings. The graph is just two numpy arrays, so you can
# also build it on the GPU and run the embedding on the CPU, or the other way
# round. The [GPU vignette](../gpu/) has that.
