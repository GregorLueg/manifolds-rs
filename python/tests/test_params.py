"""Parameter routing: the groups, the allowlists and the unknown-key guard."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import manifolds_rs as mf
from manifolds_rs import _manifolds as core

##########
# Groups #
##########


def test_group_field_reaches_the_core(X: np.ndarray) -> None:
    """A knob set in a group has to change the answer, or it went nowhere."""
    base = mf.UMAP(n_epochs=20, seed=5).fit_transform(X)
    tweaked = mf.UMAP(
        n_epochs=20, seed=5, optim_params=mf.UmapOptim(gamma=5.0)
    ).fit_transform(X)
    assert not np.array_equal(base, tweaked)


def test_unset_group_field_leaves_the_default(X: np.ndarray) -> None:
    """A group holding only `None`s is the same as no group at all."""
    without = mf.UMAP(n_epochs=20, seed=5).fit_transform(X)
    empty = mf.UMAP(n_epochs=20, seed=5, optim_params=mf.UmapOptim()).fit_transform(X)
    assert np.array_equal(without, empty)


def test_constructor_argument_reaches_its_group(X: np.ndarray) -> None:
    """`n_epochs` lives in `optim_params` on the Rust side, not at the top."""
    short = mf.UMAP(n_epochs=10, seed=5).fit_transform(X)
    long = mf.UMAP(n_epochs=200, seed=5).fit_transform(X)
    assert not np.array_equal(short, long)


def test_min_dist_and_spread_fit_the_curve(X: np.ndarray) -> None:
    """Both feed the a/b fit, so either alone must change the embedding."""
    base = mf.UMAP(n_epochs=20, seed=5).fit_transform(X)
    tight = mf.UMAP(n_epochs=20, seed=5, min_dist=0.01).fit_transform(X)
    wide = mf.UMAP(n_epochs=20, seed=5, spread=3.0).fit_transform(X)
    assert not np.array_equal(base, tight)
    assert not np.array_equal(base, wide)


def test_pinned_curve_parameters_override_the_fit(X: np.ndarray) -> None:
    """`a` and `b` are applied after the fit, so pinning them wins."""
    fitted = mf.UMAP(n_epochs=20, seed=5, min_dist=0.01).fit_transform(X)
    pinned = mf.UMAP(
        n_epochs=20, seed=5, min_dist=0.01, optim_params=mf.UmapOptim(a=1.0, b=1.0)
    ).fit_transform(X)
    assert not np.array_equal(fitted, pinned)


def test_metric_reaches_the_neighbour_search(X: np.ndarray) -> None:
    euclidean = mf.UMAP(n_epochs=20, seed=5, metric="euclidean").fit_transform(X)
    cosine = mf.UMAP(n_epochs=20, seed=5, metric="cosine").fit_transform(X)
    assert not np.array_equal(euclidean, cosine)


##############
# Allowlists #
##############


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ann": "faiss"}, "ann"),
        ({"init": "tsne"}, "init"),
        ({"metric": "jaccard"}, "metric"),
        ({"optimiser": "lbfgs"}, "optimiser"),
    ],
)
def test_unknown_string_is_refused(
    kwargs: dict[str, Any], match: str, X: np.ndarray
) -> None:
    """The core would fall back to a default and print to stdout, which from a
    notebook is invisible. So the allowlist bites first."""
    with pytest.raises(ValueError, match=match):
        mf.UMAP(n_epochs=20, **kwargs).fit_transform(X)


def test_string_case_does_not_matter(X: np.ndarray) -> None:
    """Every parser in the core lowercases first, so this layer does too."""
    lower = mf.UMAP(n_epochs=20, seed=5, ann="kmknn").fit_transform(X)
    upper = mf.UMAP(n_epochs=20, seed=5, ann="KmKnn").fit_transform(X)
    assert np.array_equal(lower, upper)


def test_unknown_topology_is_refused() -> None:
    with pytest.raises(ValueError, match="topology"):
        mf.datasets.trajectory(100, topology="spiral")


###################
# Unknown keys    #
###################


def test_unknown_top_level_key_is_named(X: np.ndarray) -> None:
    with pytest.raises(TypeError, match="min_dst"):
        core.umap(X, {"min_dst": 0.1})


def test_unknown_group_key_is_named(X: np.ndarray) -> None:
    with pytest.raises(TypeError, match="ef_serch"):
        core.umap(X, {"nn_params": {"ef_serch": 100}})


def test_non_finite_parameter_is_refused(X: np.ndarray) -> None:
    """A NaN knob costs a whole optimisation run before anyone notices."""
    with pytest.raises(TypeError, match="finite"):
        core.umap(X, {"min_dist": float("nan")})


def test_wrong_parameter_type_is_named(X: np.ndarray) -> None:
    with pytest.raises(TypeError, match="k"):
        core.umap(X, {"k": "fifteen"})


##########################
# Meaningful null values #
##########################


def test_phate_decay_none_selects_the_binary_kernel(X: np.ndarray) -> None:
    """`None` here is a value, not an absence: it swaps the alpha-decay kernel
    for a binary connectivity one."""
    decayed = mf.PHATE(seed=5, decay=40.0).fit_transform(X)
    binary = mf.PHATE(seed=5, decay=None).fit_transform(X)
    assert not np.array_equal(decayed, binary)


def test_pacmap_range_none_differs_from_the_default(X: np.ndarray) -> None:
    """Same shape of question for PaCMAP's initialisation range."""
    ranged = mf.PaCMAP(n_epochs=20, seed=5, range_=0.01).fit_transform(X)
    unranged = mf.PaCMAP(n_epochs=20, seed=5, range_=None).fit_transform(X)
    assert not np.array_equal(ranged, unranged)
