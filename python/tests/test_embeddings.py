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
    algorithm. UMAP's own search is `"nndescent"` with `k = n_neighbors`, so
    the same call built externally has to land on the same embedding.
    """
    ind, dist = mf.knn_graph(X, k=15, ann="nndescent", seed=42)
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


@pytest.mark.parametrize("cls", [mf.UMAP, mf.DensMAP])
def test_sgd_parallel_runs(cls: type[BaseEmbedding], X: np.ndarray) -> None:
    """Hogwild SGD is not reproducible, so only shape and finiteness."""
    embedding = fast(cls, optimiser="sgd_parallel").fit_transform(X)
    assert embedding.shape == (X.shape[0], 2)
    assert np.isfinite(embedding).all()


@pytest.mark.parametrize("cls", [mf.TSNE, mf.DensNE, mf.DREAMS])
@pytest.mark.parametrize("approx", ["fft", "fft_3k"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cpu_fft_tsne_runs(
    cls: type[BaseEmbedding], approx: str, dtype: type, X: np.ndarray
) -> None:
    """The wheel carries `fft_tsne`, so both FFT optimisers reach the core."""
    x = np.ascontiguousarray(X, dtype=dtype)
    embedding = fast(cls, approx=approx).fit_transform(x)
    assert embedding.shape == (X.shape[0], 2)
    assert embedding.dtype == dtype
    assert np.isfinite(embedding).all()


@pytest.mark.parametrize("init", ["spectral", "recursive"])
def test_umap_adam_v06_runs_and_reproduces(init: str, X: np.ndarray) -> None:
    """The umap-learn 0.6 Adam reaches the core, with either initialisation."""
    est = mf.UMAP(n_epochs=20, optimiser="adam_v06", init=init)
    first = est.fit_transform(X)
    assert first.shape == (X.shape[0], 2)
    assert np.isfinite(first).all()
    assert np.array_equal(first, est.fit_transform(X))


@pytest.mark.parametrize("cls", [mf.UMAP, mf.DensMAP])
def test_recursive_init_differs_from_spectral(
    cls: type[BaseEmbedding], X: np.ndarray
) -> None:
    """`init="recursive"` is routed to the core, not silently mapped to spectral."""
    recursive = fast(cls, init="recursive").fit_transform(X)
    spectral = fast(cls, init="spectral").fit_transform(X)
    assert np.isfinite(recursive).all()
    assert not np.array_equal(recursive, spectral)


def test_densmap_refuses_adam_v06_before_the_core(X: np.ndarray) -> None:
    with pytest.raises(ValueError, match="optimiser"):
        mf.DensMAP(optimiser="adam_v06").fit_transform(X)


@pytest.mark.parametrize("cls", [mf.PaCMAP, mf.LocalMAP])
def test_init_recursive_is_refused_outside_umap(
    cls: type[BaseEmbedding], X: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="init"):
        fast(cls, init="recursive").fit_transform(X)


def test_localmap_differs_from_pacmap(X: np.ndarray) -> None:
    """LocalMAP only departs from PaCMAP in phase three, which starts after
    epoch 200, so these run the full default schedule."""
    pacmap = mf.PaCMAP().fit_transform(X)
    localmap = mf.LocalMAP().fit_transform(X)
    assert not np.array_equal(pacmap, localmap)


def test_localmap_max_distance_is_routed(X: np.ndarray) -> None:
    near = mf.LocalMAP(max_distance=1.0).fit_transform(X)
    far = mf.LocalMAP(max_distance=50.0).fit_transform(X)
    assert not np.array_equal(near, far)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_localmap_refuses_a_bad_max_distance(bad: float, X: np.ndarray) -> None:
    with pytest.raises(ValueError, match="max_distance"):
        fast(mf.LocalMAP, max_distance=bad).fit_transform(X)


@pytest.mark.parametrize("cls", [mf.TSNE, mf.DensNE, mf.DREAMS])
def test_int_perplexity_matches_float(cls: type[BaseEmbedding], X: np.ndarray) -> None:
    as_int = fast(cls, perplexity=20).fit_transform(X)
    as_float = fast(cls, perplexity=20.0).fit_transform(X)
    assert np.array_equal(as_int, as_float)
