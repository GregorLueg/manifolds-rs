"""The synthetic generators, and that the structure they claim is really there."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import manifolds_rs as mf


def test_swiss_roll_is_three_dimensional() -> None:
    X, t = mf.datasets.swiss_roll(500, seed=1)
    assert X.shape == (500, 3)
    assert t.shape == (500,)
    assert X.dtype == np.float64


def test_density_bias_concentrates_the_sampling() -> None:
    """The bias is an exponent on the uniform sample, so `1.0` is the neutral
    value. A higher exponent drives each draw towards zero, and `t` with it, so
    the points pile up at the inner end of the roll."""
    _, uniform = mf.datasets.swiss_roll(2000, density_bias=1.0, seed=1)
    _, biased = mf.datasets.swiss_roll(2000, density_bias=2.5, seed=1)
    assert biased.mean() < uniform.mean()


def test_zero_density_bias_is_degenerate() -> None:
    """`0.0` raises every draw to the zeroth power, which collapses the roll to
    a single `t`. The crate's doc comment calls this uniform sampling; it is
    not, and the binding defaults to the neutral `1.0` instead."""
    _, t = mf.datasets.swiss_roll(500, density_bias=0.0, seed=1)
    assert np.allclose(t, t[0])


def test_clustered_labels_cover_every_cluster() -> None:
    X, labels = mf.datasets.clustered(600, dim=10, n_clusters=5, seed=1)
    assert X.shape == (600, 10)
    assert set(np.unique(labels)) == set(range(5))


def test_clusters_are_further_apart_than_they_are_wide() -> None:
    """If they were not, no embedding could be expected to separate them and the
    fixture would be testing nothing."""
    X, labels = mf.datasets.clustered(600, dim=10, n_clusters=4, seed=1)
    centres = np.stack([X[labels == c].mean(axis=0) for c in range(4)])
    spread = np.mean([X[labels == c].std(axis=0).mean() for c in range(4)])
    gaps = [
        np.linalg.norm(centres[i] - centres[j])
        for i in range(4)
        for j in range(i + 1, 4)
    ]
    assert min(gaps) > spread


@pytest.mark.parametrize("topology", ["bifurcation", "linear", "combination"])
def test_every_topology_generates(topology: str) -> None:
    X, branch = mf.datasets.trajectory(400, topology=topology, dim=20, seed=1)
    assert X.shape[1] == 20
    assert len(np.unique(branch)) >= 1


def test_trajectory_refuses_too_few_dimensions() -> None:
    with pytest.raises(ValueError, match="branch count"):
        mf.datasets.trajectory(400, topology="bifurcation", dim=2)


def test_hierarchical_labels_nest() -> None:
    """Each subcluster must sit inside exactly one supergroup, or the two-level
    claim is not true of the data."""
    _, supergroup, subcluster = mf.datasets.hierarchical(
        600, dim=10, n_supergroups=3, n_subclusters=4, seed=1
    )
    for sub in np.unique(subcluster):
        assert len(np.unique(supergroup[subcluster == sub])) == 1


@pytest.mark.parametrize(
    "generator",
    [
        lambda: mf.datasets.swiss_roll(300, seed=2),
        lambda: mf.datasets.clustered(300, dim=8, seed=2),
        lambda: mf.datasets.trajectory(300, dim=10, seed=2),
        lambda: mf.datasets.hierarchical(300, dim=8, seed=2),
    ],
)
def test_same_seed_gives_the_same_data(
    generator: Callable[[], tuple[np.ndarray, ...]],
) -> None:
    first = generator()
    second = generator()
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a, b)
