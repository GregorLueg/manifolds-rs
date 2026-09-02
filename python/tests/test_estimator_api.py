"""The scikit-learn shaped surface, and the places it deliberately stops."""

from __future__ import annotations

import dataclasses
import pickle

import numpy as np
import pytest
from conftest import cpu_estimators, fast

import manifolds_rs as mf
from manifolds_rs import BaseEmbedding


@pytest.mark.parametrize(("name", "cls"), cpu_estimators())
def test_get_params_round_trips_through_the_constructor(
    name: str, cls: type[BaseEmbedding]
) -> None:
    original = cls()
    assert type(original)(**original.get_params()).get_params() == original.get_params()


@pytest.mark.parametrize(("name", "cls"), cpu_estimators())
def test_repr_names_every_parameter(name: str, cls: type[BaseEmbedding]) -> None:
    text = repr(cls())
    assert text.startswith(f"{name}(")
    for param in cls._param_names():
        assert f"{param}=" in text


@pytest.mark.parametrize(("name", "cls"), cpu_estimators())
def test_embedding_before_fit_raises(name: str, cls: type[BaseEmbedding]) -> None:
    with pytest.raises(mf.NotFittedError):
        _ = cls().embedding_


@pytest.mark.parametrize(("name", "cls"), cpu_estimators())
def test_transform_says_why_it_cannot(
    name: str, cls: type[BaseEmbedding], X: np.ndarray
) -> None:
    """No out-of-sample projection exists, and the message has to say so rather
    than looking like an oversight."""
    estimator = fast(cls).fit(X)
    with pytest.raises(NotImplementedError, match="Refit"):
        estimator.transform(X)


def test_set_params_changes_the_next_fit(X: np.ndarray) -> None:
    estimator = mf.UMAP(n_epochs=20, seed=5)
    first = estimator.fit_transform(X)
    second = estimator.set_params(n_neighbors=40).fit_transform(X)
    assert not np.array_equal(first, second)


def test_set_params_discards_the_fitted_embedding(X: np.ndarray) -> None:
    estimator = mf.UMAP(n_epochs=20).fit(X)
    estimator.set_params(n_neighbors=30)
    with pytest.raises(mf.NotFittedError):
        _ = estimator.embedding_


def test_set_params_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="n_neighbours"):
        mf.UMAP().set_params(n_neighbours=15)


def test_fit_returns_self_for_chaining(X: np.ndarray) -> None:
    estimator = mf.UMAP(n_epochs=20)
    assert estimator.fit(X) is estimator


def test_embedding_attribute_matches_fit_transform(X: np.ndarray) -> None:
    estimator = mf.UMAP(n_epochs=20, seed=5)
    returned = estimator.fit_transform(X)
    assert np.array_equal(returned, estimator.embedding_)


def test_fitted_shape_attributes_are_recorded(X: np.ndarray) -> None:
    estimator = mf.UMAP(n_epochs=20).fit(X)
    assert (estimator.n_samples_fit_, estimator.n_features_in_) == X.shape


def test_beartype_rejects_a_wrong_argument_type() -> None:
    with pytest.raises(Exception, match="n_neighbors"):
        mf.UMAP(n_neighbors="fifteen")  # ty: ignore[invalid-argument-type]


def test_an_unfitted_estimator_pickles(X: np.ndarray) -> None:
    """Nothing holds a handle, so the plain object protocol is enough."""
    estimator = mf.UMAP(n_neighbors=30, ann="hnsw")
    assert pickle.loads(pickle.dumps(estimator)).get_params() == estimator.get_params()


def test_parameter_groups_survive_a_round_trip() -> None:
    estimator = mf.UMAP(nn_params=mf.NeighbourParams(m=32, ef_search=200))
    revived = pickle.loads(pickle.dumps(estimator))
    assert revived.nn_params == estimator.nn_params


def test_parameter_groups_are_frozen() -> None:
    """Frozen so a group can be shared across estimators without one of them
    mutating it out from under the others."""
    group = mf.NeighbourParams(m=32)
    with pytest.raises(dataclasses.FrozenInstanceError):
        group.m = 64  # ty: ignore[invalid-assignment]


def test_sklearn_clone_works(X: np.ndarray) -> None:
    """`get_params` / `set_params` are duck-typed, so `clone` needs no base
    class of theirs."""
    sklearn_base = pytest.importorskip("sklearn.base")
    estimator = mf.UMAP(n_neighbors=30, n_epochs=20)
    assert sklearn_base.clone(estimator).get_params() == estimator.get_params()


def test_thread_cap_is_readable_and_restorable() -> None:
    original = mf.num_threads()
    mf.set_num_threads(2)
    assert mf.num_threads() == 2
    mf.set_num_threads(0)
    assert mf.num_threads() == original
