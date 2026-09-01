"""The constructor defaults must be the crate's own.

Every estimator writes its defaults out as Python literals, because a signature
full of ``None`` tells a reader nothing. The risk that creates is drift: someone
changes a default in `src/lib.rs` and the two quietly disagree.

This catches it end to end. An untouched estimator sends its literals down; an
empty payload sends nothing and lets the crate's `Default` impl fill everything.
If those two produce the same embedding bit for bit, no literal has drifted.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import manifolds_rs as mf
from manifolds_rs import BaseEmbedding
from manifolds_rs import _manifolds as core

CASES = [
    ("UMAP", mf.UMAP, core.umap),
    ("DensMAP", mf.DensMAP, core.densmap),
    ("TSNE", mf.TSNE, core.tsne),
    ("DensNE", mf.DensNE, core.densne),
    ("PHATE", mf.PHATE, core.phate),
    ("PaCMAP", mf.PaCMAP, core.pacmap),
    ("DiffusionMaps", mf.DiffusionMaps, core.diffusion_maps),
]


@pytest.mark.parametrize(("name", "cls", "fn"), CASES)
def test_defaults_match_the_core(
    name: str, cls: type[BaseEmbedding], fn: Callable[..., np.ndarray], X: np.ndarray
) -> None:
    estimator = cls().fit_transform(X)
    bare = fn(X, {}, seed=42, verbose=0)
    assert np.array_equal(estimator, bare), (
        f"{name}: a constructor default has drifted from the crate's"
    )


@pytest.mark.skipif(not mf.gpu_available(), reason="no GPU adapter on this machine")
@pytest.mark.parametrize(
    ("name", "cls", "fn"),
    [
        ("UMAPGpu", getattr(mf, "UMAPGpu", None), getattr(core, "umap_gpu", None)),
        (
            "DensMAPGpu",
            getattr(mf, "DensMAPGpu", None),
            getattr(core, "densmap_gpu", None),
        ),
        ("TSNEGpu", getattr(mf, "TSNEGpu", None), getattr(core, "tsne_gpu", None)),
    ],
)
def test_gpu_defaults_match_the_core(
    name: str, cls: type[BaseEmbedding], fn: Callable[..., np.ndarray], X32: np.ndarray
) -> None:
    estimator = cls().fit_transform(X32)
    bare = fn(X32, {}, seed=42, verbose=0)
    assert np.array_equal(estimator, bare), (
        f"{name}: a constructor default has drifted from the crate's"
    )
