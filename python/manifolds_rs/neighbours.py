"""Standalone neighbour search.

Every estimator accepts a precomputed graph, and on anything large the search is
most of the runtime. Build one here, hand it to as many embeddings as you like:

    >>> import manifolds_rs as mf
    >>> X, _ = mf.datasets.clustered(20_000, dim=50)
    >>> ind, dist = mf.knn_graph(X, k=15, ann="hnsw")
    >>> a = mf.UMAP().fit_transform(X, knn_indices=ind, knn_distances=dist)
    >>> b = mf.PaCMAP().fit_transform(X, knn_indices=ind, knn_distances=dist)

The distances come back in whatever the metric produces, squared Euclidean
included. That is deliberate and matches what the embeddings expect: taking a
square root before handing the graph on changes the answer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from beartype import beartype

from . import _manifolds as _core
from ._params import ANN_CPU, METRICS, NeighbourParams, merge
from ._validate import check_choice, check_matrix


@beartype
def knn_graph(
    X: Any,
    k: int = 15,
    *,
    metric: str = "euclidean",
    ann: str = "kmknn",
    nn_params: NeighbourParams | None = None,
    seed: int = 42,
    verbose: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a k-nearest-neighbour graph over `X`.

    Args:
        X: Array-like of shape ``(n_samples, n_features)``. float32 and float64
            are used as-is; anything else is promoted to float64.
        k: Neighbours per point, excluding self.
        metric: ``"euclidean"``/``"l2"``, ``"cosine"`` or ``"manhattan"``/``"l1"``.
        ann: ``"exhaustive"``, ``"kmknn"``, ``"balltree"``, ``"annoy"``,
            ``"hnsw"``, ``"ivf"`` or ``"nndescent"``. ``"kmknn"`` is exact and
            holds up well into the hundreds of thousands; past that
            ``"nndescent"`` or ``"hnsw"``.
        nn_params: Backend-specific knobs. See `NeighbourParams`.
        seed: Fixes anything randomised in the build.
        verbose: ``0`` silent, ``1`` normal, ``2`` detailed.

    Returns:
        ``(indices, distances)``, both ``(n_samples, k)``. Indices are int64,
        distances match the dtype of `X`. Indices first, unlike scikit-learn:
        it is the order the estimators take them in.

    Raises:
        ValueError: If `ann` or `metric` is not recognised, or `X` is empty or
            not finite.
    """
    arr = check_matrix(X)
    group = merge(nn_params, dist_metric=check_choice(metric, METRICS, name="metric"))
    return _core.knn_graph(
        arr,
        k,
        ann=check_choice(ann, ANN_CPU, name="ann"),
        nn_params=group,
        seed=seed,
        verbose=verbose,
    )
