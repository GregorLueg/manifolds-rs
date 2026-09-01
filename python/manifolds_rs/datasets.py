"""Synthetic data with structure worth recovering.

Gaussian noise tells you nothing about whether an embedding worked. Everything
here comes with ground truth: the swiss roll has a known unrolling, the
trajectory has known lineages, the hierarchical clusters have a known two-level
structure. Score against that rather than looking at the picture and nodding.

float64 throughout. Cast to float32 yourself if you are feeding a GPU
estimator, though those cast for you anyway.
"""

from __future__ import annotations

import numpy as np
from beartype import beartype

from . import _manifolds as _core
from ._params import TOPOLOGIES
from ._validate import check_choice


@beartype
def swiss_roll(
    n_samples: int = 5000,
    *,
    noise: float = 0.05,
    density_bias: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """A 2-D manifold rolled up in 3-D.

    The standard unrolling benchmark, and still a good one: an embedding that
    tears the roll or folds it back on itself is doing something wrong that no
    cluster dataset would have shown you.

    Args:
        n_samples: Number of points.
        noise: Standard deviation of the noise added to the surface.
        density_bias: Exponent on the uniform sample along the roll. ``1.0``
            is neutral and samples uniformly; higher values pile points up at
            the inner end, which is what makes an embedding's density handling
            visible, and ``2.5`` roughly matches the accumulation you see in
            real trajectory data. ``0.0`` is degenerate, not uniform: it puts
            every point at the same ``t``.
        seed: Fixes the sampling.

    Returns:
        ``(X, t)`` where `X` is ``(n_samples, 3)`` and `t` is the position along
        the roll, which is what a correct unrolling recovers as one axis.
    """
    return _core.swiss_roll(
        n_samples, noise=noise, density_bias=density_bias, seed=seed
    )


@beartype
def clustered(
    n_samples: int = 5000, *, dim: int = 50, n_clusters: int = 10, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian clusters of varying size and spread.

    Args:
        n_samples: Number of points.
        dim: Ambient dimensionality.
        n_clusters: Number of clusters.
        seed: Fixes the centres and the sampling.

    Returns:
        ``(X, labels)`` with `X` of shape ``(n_samples, dim)``.
    """
    return _core.clustered(n_samples, dim=dim, n_clusters=n_clusters, seed=seed)


@beartype
def trajectory(
    n_samples: int = 5000,
    *,
    topology: str = "bifurcation",
    dim: int = 50,
    noise: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """A branching differentiation trajectory.

    This is the one that separates the algorithms. PHATE and PaCMAP tend to keep
    the branch points; t-SNE tends to shatter the backbone into blobs.

    Args:
        n_samples: Number of points, split evenly across branches.
        topology: ``"bifurcation"`` for a cascading tree, ``"linear"`` for one
            continuous lineage, ``"combination"`` for a backbone with branches
            leaving it mid-way.
        dim: Ambient dimensionality. Must be at least the branch count.
        noise: Base noise standard deviation, which grows along pseudotime.
        seed: Fixes the sampling.

    Returns:
        ``(X, branch)`` with `X` of shape ``(n_samples, dim)``.
    """
    return _core.trajectory(
        n_samples,
        topology=check_choice(topology, TOPOLOGIES, name="topology"),
        dim=dim,
        noise=noise,
        seed=seed,
    )


@beartype
def hierarchical(
    n_samples: int = 5000,
    *,
    dim: int = 50,
    n_supergroups: int = 4,
    n_subclusters: int = 5,
    supergroup_spread: float = 15.0,
    subcluster_spread: float = 3.0,
    point_std: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clusters within clusters.

    The test for whether an embedding kept two scales at once. Resolving the
    subclusters is easy; keeping the supergroups apart while doing it is the
    part that global-structure claims are usually made about.

    Args:
        n_samples: Number of points.
        dim: Ambient dimensionality.
        n_supergroups: Top-level groups.
        n_subclusters: Subclusters within each group.
        supergroup_spread: How far apart the group centres sit.
        subcluster_spread: How far apart subcluster centres sit within a group.
        point_std: Spread of points around a subcluster centre.
        seed: Fixes the centres and the sampling.

    Returns:
        ``(X, supergroup, subcluster)``.
    """
    return _core.hierarchical(
        n_samples,
        dim=dim,
        n_supergroups=n_supergroups,
        n_subclusters=n_subclusters,
        supergroup_spread=supergroup_spread,
        subcluster_spread=subcluster_spread,
        point_std=point_std,
        seed=seed,
    )
