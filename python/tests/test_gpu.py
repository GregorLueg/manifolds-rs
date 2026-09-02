"""The GPU estimators.

Skipped wholesale when there is no adapter, which is the case on every hosted CI
runner. What runs there is the compile: a GPU wheel that imports and reports
`gpu_available() is False` on a machine without a device is most of what CI can
prove. Kernel correctness needs a runner with a GPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import manifolds_rs as mf
from manifolds_rs import BaseEmbedding

pytestmark = pytest.mark.skipif(
    not mf.gpu_available(), reason="no GPU adapter on this machine"
)

GPU_ESTIMATORS = [
    ("UMAPGpu", "UMAPGpu"),
    ("DensMAPGpu", "DensMAPGpu"),
    ("TSNEGpu", "TSNEGpu"),
]


def separation(embedding: np.ndarray, labels: np.ndarray) -> float:
    """Mean between-cluster centroid distance over mean within-cluster spread.

    Above 1 means the clusters are further apart than they are wide. Mirrors
    `compute_separation` in the crate's own GPU tests.
    """
    ids = np.unique(labels)
    centroids = np.stack([embedding[labels == c].mean(axis=0) for c in ids])
    within = np.mean(
        [
            np.linalg.norm(embedding[labels == c] - centroids[i], axis=1).mean()
            for i, c in enumerate(ids)
        ]
    )
    between = np.mean(
        [
            np.linalg.norm(centroids[i] - centroids[j])
            for i in range(len(ids))
            for j in range(i + 1, len(ids))
        ]
    )
    return float(between / max(within, 1e-12))


def build(name: str, **kwargs: Any) -> BaseEmbedding:
    """Instantiate a GPU estimator by name with the epochs turned down."""
    cls: type[BaseEmbedding] = getattr(mf, name)
    kwargs.setdefault("n_epochs", 20)
    return cls(**kwargs)


@pytest.mark.parametrize(("name", "_"), GPU_ESTIMATORS)
def test_embedding_has_one_row_per_sample(name: str, _: str, X32: np.ndarray) -> None:
    embedding = build(name).fit_transform(X32)
    assert embedding.shape == (X32.shape[0], 2)
    assert np.isfinite(embedding).all()


@pytest.mark.parametrize(("name", "_"), GPU_ESTIMATORS)
def test_float64_input_is_cast_not_refused(name: str, _: str, X: np.ndarray) -> None:
    """WGSL has no float64, so `fit` narrows rather than failing inside a
    kernel. It is the only silent narrowing this package performs."""
    assert build(name).fit_transform(X).dtype == np.float32


@pytest.mark.parametrize(("name", "_"), GPU_ESTIMATORS)
def test_same_seed_reproduces_the_structure(
    name: str, _: str, clustered: tuple[np.ndarray, np.ndarray]
) -> None:
    """Two GPU runs at one seed agree on the structure, not on the coordinates.

    The device neighbour searches are not always bit-stable at scale, and the
    optimiser turns a fraction of a percent of graph difference into visibly
    different positions. What has to survive is the thing anyone reads off the
    plot: the clusters, separated by the same margin. See the reproducibility
    section of the guide for the measurements behind this.
    """
    X, labels = clustered
    X32 = np.ascontiguousarray(X, dtype=np.float32)

    a = build(name, seed=3).fit_transform(X32)
    b = build(name, seed=3).fit_transform(X32)

    sep_a, sep_b = separation(a, labels), separation(b, labels)
    assert sep_a > 1.0 and sep_b > 1.0, "both runs must separate the clusters"
    assert 0.5 < sep_a / sep_b < 2.0, (
        f"cluster separation moved between runs: {sep_a:.3f} vs {sep_b:.3f}"
    )


@pytest.mark.parametrize(("name", "_"), GPU_ESTIMATORS)
def test_a_fixed_graph_makes_a_run_bit_reproducible(
    name: str, _: str, X32: np.ndarray, knn: tuple[np.ndarray, np.ndarray]
) -> None:
    """The optimiser is the deterministic half.

    Hand in the same graph twice and the coordinates come back identical, which
    is what pins the non-determinism on the search rather than the Adam update.
    """
    ind, dist = knn
    ind = np.ascontiguousarray(ind)
    dist = np.ascontiguousarray(dist, dtype=np.float32)

    a = build(name, seed=3).fit_transform(X32, knn_indices=ind, knn_distances=dist)
    b = build(name, seed=3).fit_transform(X32, knn_indices=ind, knn_distances=dist)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("ann", ["nndescent_gpu", "ivf_gpu", "exhaustive_gpu"])
def test_every_device_backend_runs(ann: str, X32: np.ndarray) -> None:
    embedding = mf.UMAPGpu(n_epochs=20, ann=ann).fit_transform(X32)
    assert np.isfinite(embedding).all()


def test_cpu_backend_names_are_refused(X32: np.ndarray) -> None:
    """`"hnsw"` is a CPU-only backend, and reaching the GPU path with it would
    fall back silently."""
    with pytest.raises(ValueError, match="ann"):
        mf.UMAPGpu(n_epochs=20, ann="hnsw").fit_transform(X32)


def test_precomputed_knn_is_accepted(X32: np.ndarray) -> None:
    ind, dist = mf.knn_graph(X32, k=15, ann="kmknn", seed=7)
    embedding = mf.UMAPGpu(n_epochs=20).fit_transform(
        X32, knn_indices=ind, knn_distances=dist
    )
    assert embedding.shape == (X32.shape[0], 2)


def test_gpu_and_cpu_umap_agree_on_a_shared_graph(X32: np.ndarray) -> None:
    """Given the same neighbour graph the two optimisers should land in the same
    region, even though the arithmetic order differs. Correlating the pairwise
    distances is the shape-level comparison; the coordinates themselves are only
    defined up to rotation and reflection.
    """
    ind, dist = mf.knn_graph(X32, k=15, ann="kmknn", seed=7)
    kwargs = {"knn_indices": ind, "knn_distances": dist}
    cpu = mf.UMAP(n_epochs=200, seed=3).fit_transform(X32, **kwargs)
    gpu = mf.UMAPGpu(n_epochs=200, seed=3).fit_transform(X32, **kwargs)

    sample = np.random.default_rng(0).choice(len(cpu), size=200, replace=False)
    cpu_d = np.linalg.norm(cpu[sample, None] - cpu[None, sample], axis=-1).ravel()
    gpu_d = np.linalg.norm(gpu[sample, None] - gpu[None, sample], axis=-1).ravel()
    assert np.corrcoef(cpu_d, gpu_d)[0, 1] > 0.7
