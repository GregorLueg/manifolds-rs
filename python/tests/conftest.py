"""Shared fixtures.

Everything here is small and cheap on purpose. These tests check the binding
layer: shapes, dtypes, parameter routing, error paths. Whether the embeddings
are any good is the crate's own test suite's problem, and it has the data sizes
to answer it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest

import manifolds_rs as mf
from manifolds_rs import BaseEmbedding

#: Epochs for every estimator in the test suite. Enough to exercise the
#: optimiser loop, nowhere near enough to produce a sensible embedding.
EPOCHS = 20


@pytest.fixture(scope="session")
def clustered() -> tuple[np.ndarray, np.ndarray]:
    """A small clustered dataset with labels, float64."""
    return mf.datasets.clustered(600, dim=12, n_clusters=4, seed=7)


@pytest.fixture(scope="session")
def X(clustered: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """The design matrix on its own."""
    return clustered[0]


@pytest.fixture(scope="session")
def X32(X: np.ndarray) -> np.ndarray:
    """The same data as float32, for the GPU paths and dtype dispatch."""
    return np.ascontiguousarray(X, dtype=np.float32)


@pytest.fixture(scope="session")
def knn(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A precomputed neighbour graph wide enough for every estimator."""
    return mf.knn_graph(X, k=50, ann="kmknn", seed=7)


def cpu_estimators() -> list[tuple[str, type[BaseEmbedding]]]:
    """Every CPU estimator, for the tests that apply to all of them."""
    return [
        ("UMAP", mf.UMAP),
        ("DensMAP", mf.DensMAP),
        ("TSNE", mf.TSNE),
        ("DensNE", mf.DensNE),
        ("PHATE", mf.PHATE),
        ("PaCMAP", mf.PaCMAP),
        ("LocalMAP", mf.LocalMAP),
        ("DiffusionMaps", mf.DiffusionMaps),
        ("ForceAtlas2", mf.ForceAtlas2),
        ("NCVis", mf.NCVis),
    ]


#: Estimators whose optimiser is hogwild: a fixed seed reproduces only on one
#: thread.
HOGWILD: frozenset[str] = frozenset({"NCVis"})


@contextmanager
def reproducible(name: str) -> Iterator[None]:
    """Cap the pool at one thread for a hogwild estimator, a no-op otherwise."""
    if name not in HOGWILD:
        yield
        return
    mf.set_num_threads(1)
    try:
        yield
    finally:
        mf.set_num_threads(0)


def fast(cls: type[BaseEmbedding], **kwargs: Any) -> BaseEmbedding:
    """Build an estimator with the epoch count turned down.

    PHATE and diffusion maps have no epoch parameter, so the keyword is only
    passed to the ones that take it.
    """
    if "n_epochs" in cls._param_names():
        kwargs.setdefault("n_epochs", EPOCHS)
    return cls(**kwargs)
