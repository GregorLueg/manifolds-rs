"""The standalone neighbour search."""

from __future__ import annotations

import numpy as np
import pytest

import manifolds_rs as mf

BACKENDS = ["exhaustive", "kmknn", "balltree", "annoy", "hnsw", "ivf", "nndescent"]


@pytest.mark.parametrize("ann", BACKENDS)
def test_every_backend_returns_the_requested_shape(ann: str, X: np.ndarray) -> None:
    ind, dist = mf.knn_graph(X, k=10, ann=ann, seed=7)
    assert ind.shape == dist.shape == (X.shape[0], 10)
    assert ind.dtype == np.int64
    assert dist.dtype == X.dtype


@pytest.mark.parametrize("ann", BACKENDS)
def test_no_point_is_its_own_neighbour(ann: str, X: np.ndarray) -> None:
    """The graph excludes self, which is what the embeddings assume."""
    ind, _ = mf.knn_graph(X, k=10, ann=ann, seed=7)
    assert not (ind == np.arange(X.shape[0])[:, None]).any()


def test_distances_come_back_sorted(X: np.ndarray) -> None:
    _, dist = mf.knn_graph(X, k=10, ann="exhaustive", seed=7)
    assert (np.diff(dist, axis=1) >= 0).all()


def test_exhaustive_is_the_ground_truth(X: np.ndarray) -> None:
    """The exact backend has to agree with a brute-force scan done in numpy.

    Distances are squared Euclidean throughout, so the comparison is too: no
    square root is taken anywhere in the pipeline.
    """
    ind, dist = mf.knn_graph(X, k=5, ann="exhaustive", seed=7)
    gram = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(gram, np.inf)
    expected = np.sort(gram, axis=1)[:, :5]
    assert np.allclose(dist, expected)
    assert ind.shape == expected.shape


def test_float32_input_gives_float32_distances(X32: np.ndarray) -> None:
    _, dist = mf.knn_graph(X32, k=10, seed=7)
    assert dist.dtype == np.float32


def test_same_seed_gives_the_same_graph(X: np.ndarray) -> None:
    a = mf.knn_graph(X, k=10, ann="annoy", seed=3)
    b = mf.knn_graph(X, k=10, ann="annoy", seed=3)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_backend_knobs_reach_the_search(X: np.ndarray) -> None:
    """A narrow HNSW beam should not recall as well as a wide one."""
    truth, _ = mf.knn_graph(X, k=10, ann="exhaustive", seed=7)
    narrow, _ = mf.knn_graph(
        X, k=10, ann="hnsw", nn_params=mf.NeighbourParams(m=4, ef_search=10), seed=7
    )
    wide, _ = mf.knn_graph(
        X, k=10, ann="hnsw", nn_params=mf.NeighbourParams(m=32, ef_search=300), seed=7
    )

    def recall(found: np.ndarray) -> float:
        hits = sum(len(set(f) & set(t)) for f, t in zip(found, truth, strict=True))
        return hits / truth.size

    assert recall(wide) >= recall(narrow)


def test_cosine_differs_from_euclidean(X: np.ndarray) -> None:
    euclidean, _ = mf.knn_graph(X, k=10, metric="euclidean", seed=7)
    cosine, _ = mf.knn_graph(X, k=10, metric="cosine", seed=7)
    assert not np.array_equal(euclidean, cosine)


def test_unknown_backend_is_refused(X: np.ndarray) -> None:
    with pytest.raises(ValueError, match="ann"):
        mf.knn_graph(X, k=10, ann="faiss")


def test_empty_input_is_refused() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        mf.knn_graph(np.zeros((0, 5)), k=10)
