# Quickstart

Everything below runs against the built-in generators, so you can paste it
without finding a dataset first.

## Your first embedding

```python
import manifolds_rs as mf

X, labels = mf.datasets.clustered(20_000, dim=50, n_clusters=12, seed=0)

embedding = mf.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(X)
embedding.shape  # (20000, 2)
```

`fit_transform` is the method you want. `fit` plus `embedding_` says the same
thing in scikit-learn's shape if that suits the surrounding code better.

## Scoring it, rather than squinting at it

A picture of an embedding is not evidence. These generators hand back ground
truth, so use it.

```python
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def preservation(X, embedding, k=15):
    """Fraction of each point's high-dimensional neighbours still nearby."""
    hi = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)
    lo = (
        NearestNeighbors(n_neighbors=k).fit(embedding).kneighbors(return_distance=False)
    )
    return np.mean([len(set(a) & set(b)) for a, b in zip(hi, lo)]) / k
```

Run that over the swiss roll, which is a genuine manifold, and it separates the
algorithms cleanly. 5000 points, defaults throughout:

| | neighbourhood preservation |
| --- | --- |
| `TSNE` | 0.83 |
| `UMAP` | 0.66 |
| `PHATE` | 0.39 |
| `PaCMAP` | 0.17 |

That ordering is the local-versus-global trade-off, measured. t-SNE wins on
local structure because that is the only thing it optimises.

Now run the same metric over 12 Gaussian clusters in 50D and it collapses for
everyone: 0.07 for UMAP, 0.21 for t-SNE, 0.04 for PaCMAP. Meanwhile silhouette
against the known labels is 0.89, 0.84 and 0.50.

That is not four algorithms failing. Inside an isotropic Gaussian blob in 50
dimensions there is no local ordering to preserve: which 15 of your 400
cluster-mates happen to be nearest is close to arbitrary, so no embedding can
keep it. The clusters come apart perfectly regardless.

The lesson is to pick the metric that matches what you are claiming. Quote
neighbourhood preservation on manifold data and cluster separation on cluster
data, and be suspicious of any benchmark that reports one number.

## Comparing algorithms on the same graph

The neighbour search is most of the runtime on anything large, and every
estimator accepts a precomputed graph. Build it once:

```python
ind, dist = mf.knn_graph(X, k=50, ann="hnsw")

results = {}
for name, estimator in [
    ("umap", mf.UMAP(n_neighbors=15)),
    ("pacmap", mf.PaCMAP()),
    ("tsne", mf.TSNE()),
]:
    results[name] = estimator.fit_transform(X, knn_indices=ind, knn_distances=dist)
```

`k=50` because PaCMAP indexes into the kNN list up to `mn_candidate_end`, which
defaults to 50. Give the shared graph the widest `k` any of your estimators
needs.

## Trajectories

This is where the choice of algorithm stops being cosmetic.

```python
X, branch = mf.datasets.trajectory(10_000, topology="bifurcation", dim=50)

phate = mf.PHATE(k=5).fit_transform(X)
tsne = mf.TSNE(perplexity=30).fit_transform(X)
```

PHATE keeps the branch points connected. t-SNE will give you a set of blobs
that happen to correspond to positions along the trajectory, with the
connectivity gone. Neither is wrong; they are answering different questions.

## Density that means something

```python
X, t = mf.datasets.swiss_roll(10_000, density_bias=2.5)

plain = mf.UMAP().fit_transform(X)
dens = mf.DensMAP(lambda_=2.0).fit_transform(X)
```

The roll is deliberately sampled unevenly. In the plain UMAP the dense end and
the sparse end come out looking similar, because UMAP is free to rescale
locally. In the densMAP they do not.

## Turning the obscure knobs

```python
mf.UMAP(
    n_neighbors=30,
    ann="hnsw",
    nn_params=mf.NeighbourParams(m=32, ef_construction=400, ef_search=200),
    graph_params=mf.UmapGraph(local_connectivity=2.0),
    optim_params=mf.UmapOptim(gamma=1.5, neg_sample_rate=10),
)
```

Anything you leave out of a group keeps the crate's default, and a group you do
not pass at all is the same as passing an empty one. A misspelled field is a
`TypeError` naming the field, not a silent no-op.

## Capping threads

```python
mf.set_num_threads(4)  # for a shared machine or a job scheduler
mf.num_threads()  # 4
mf.set_num_threads(0)  # back to rayon's global pool
```
