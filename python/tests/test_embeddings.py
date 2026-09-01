"""Every CPU estimator: shape, dtype, determinism and the kNN shortcut."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import cpu_estimators, fast

import manifolds_rs as mf
from manifolds_rs import BaseEmbedding


@pytest.mark.parametrize(("name", "cls"), cpu_estimators())
def test_embedding_has_one_row_per_sample(
    name: str, cls: type[BaseEmbedding], X: np.ndarray
) -> None:
    embedding = fast(cls).fit_transform(X)
    assert embedding.shape == (X.shape[0], 2)
    assert np.isfinite(embedding).all()


@pytest.mark.parametrize(("name", "cls"), cpu_estimators())
def test_float32_in_float32_out(
    name: str, cls: type[BaseEmbedding], X32: np.ndarray
) -> None:
    """The design matrix picks the precision the whole pipeline runs in."""
    assert fast(cls).fit_transform(X32).dtype == np.float32


@pytest.mark.parametrize(("name", "cls"), cpu_estimators())
def test_float64_in_float64_out(
    name: str, cls: type[BaseEmbedding], X: np.ndarray
) -> None:
    assert fast(cls).fit_transform(X).dtype == np.float64


@pytest.mark.parametrize(("name", "cls"), cpu_estimators())
def test_same_seed_gives_identical_embedding(
    name: str, cls: type[BaseEmbedding], X: np.ndarray
) -> None:
    a = fast(cls, seed=3).fit_transform(X)
    b = fast(cls, seed=3).fit_transform(X)
    assert np.array_equal(a, b)


@pytest.mark.parametrize(("name", "cls"), cpu_estimators())
def test_precomputed_knn_is_accepted(
    name: str,
    cls: type[BaseEmbedding],
    X: np.ndarray,
    knn: tuple[np.ndarray, np.ndarray],
) -> None:
    ind, dist = knn
    embedding = fast(cls).fit_transform(X, knn_indices=ind, knn_distances=dist)
    assert embedding.shape == (X.shape[0], 2)
    assert np.isfinite(embedding).all()


def test_precomputed_knn_matches_an_internal_search(X: np.ndarray) -> None:
    """Handing in the graph the estimator would have built changes nothing.

    The point of the shortcut is that it is a shortcut, not a different
    algorithm. UMAP's own search is `"kmknn"` with `k = n_neighbors`, so the
    same call built externally has to land on the same embedding.
    """
    ind, dist = mf.knn_graph(X, k=15, ann="kmknn", seed=42)
    internal = mf.UMAP(n_neighbors=15, n_epochs=20).fit_transform(X)
    external = mf.UMAP(n_neighbors=15, n_epochs=20).fit_transform(
        X, knn_indices=ind, knn_distances=dist
    )
    assert np.array_equal(internal, external)


def test_n_components_three_gives_three_columns(X: np.ndarray) -> None:
    assert mf.UMAP(n_components=3, n_epochs=20).fit_transform(X).shape[1] == 3


def test_tsne_rejects_more_than_two_dimensions(X: np.ndarray) -> None:
    """The 2-D restriction is the core's, and it errors rather than silently
    truncating."""
    with pytest.raises(ValueError, match="n_dim"):
        mf.TSNE(n_components=3, n_epochs=20).fit_transform(X)


def test_half_a_knn_graph_is_refused(
    X: np.ndarray, knn: tuple[np.ndarray, np.ndarray]
) -> None:
    with pytest.raises(ValueError, match="together"):
        mf.UMAP(n_epochs=20).fit_transform(X, knn_indices=knn[0])


def test_padded_knn_is_refused(X: np.ndarray) -> None:
    """A row an approximate search could not fill cannot build a graph."""
    ind, dist = mf.knn_graph(X, k=15, ann="kmknn", seed=42)
    ind = ind.copy()
    ind[0, -1] = -1
    with pytest.raises(ValueError, match="padding"):
        mf.UMAP(n_epochs=20).fit_transform(X, knn_indices=ind, knn_distances=dist)


def test_mismatched_knn_row_count_is_refused(
    X: np.ndarray, knn: tuple[np.ndarray, np.ndarray]
) -> None:
    ind, dist = knn
    with pytest.raises(ValueError, match="rows"):
        mf.UMAP(n_epochs=20).fit_transform(
            X, knn_indices=ind[:-1], knn_distances=dist[:-1]
        )


def test_lambda_zero_recovers_plain_umap(X: np.ndarray) -> None:
    """densMAP with no density weight is UMAP, and should be bit-identical."""
    plain = mf.UMAP(n_epochs=20, seed=11).fit_transform(X)
    dens = mf.DensMAP(n_epochs=20, seed=11, lambda_=0.0).fit_transform(X)
    assert np.array_equal(plain, dens)


def test_non_finite_input_is_refused(X: np.ndarray) -> None:
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        mf.UMAP(n_epochs=20).fit_transform(bad)


def test_non_contiguous_input_is_accepted(X: np.ndarray) -> None:
    """A sliced view is the common case and must not reach the core as-is."""
    view = X[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    assert mf.UMAP(n_epochs=20).fit_transform(view).shape == (X.shape[0], 2)


def test_integer_input_is_promoted_not_narrowed(X: np.ndarray) -> None:
    counts = np.rint(np.abs(X) * 10).astype(np.int32)
    assert mf.UMAP(n_epochs=20).fit_transform(counts).dtype == np.float64
