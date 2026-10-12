"""The MLX backends.

Only an `mlx` build on Apple Silicon has them, so the smoke tests skip
everywhere else. What every build can check is that an `*_mlx` name is refused
with a pointer to the build flag rather than the generic allowlist message.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import EPOCHS

import manifolds_rs as mf

needs_mlx = pytest.mark.skipif(not mf.mlx_available(), reason="not an mlx build")


@pytest.mark.skipif(mf.mlx_available(), reason="mlx build accepts the names")
def test_mlx_names_point_at_the_build_flag(X: np.ndarray) -> None:
    with pytest.raises(ValueError, match="--features mlx"):
        mf.UMAP(optimiser="adam_mlx", n_epochs=EPOCHS).fit_transform(X)


@needs_mlx
@pytest.mark.parametrize("ann", ["exhaustive_mlx", "ivf_mlx", "nndescent_mlx"])
def test_mlx_knn_drops_self(ann: str, X: np.ndarray) -> None:
    ind, dist = mf.knn_graph(X, 15, ann=ann)
    assert ind.shape == (X.shape[0], 15)
    assert not (ind == np.arange(X.shape[0])[:, None]).any()
    assert np.isfinite(dist).all()


@needs_mlx
@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("UMAP", {"optimiser": "adam_mlx", "ann": "nndescent_mlx"}),
        ("DensMAP", {"optimiser": "adam_mlx"}),
        ("TSNE", {"approx": "fft_3k_mlx"}),
        ("DREAMS", {"approx": "fft_3k_mlx"}),
    ],
)
def test_mlx_estimators_run(name: str, kwargs: dict, X32: np.ndarray) -> None:
    embedding = getattr(mf, name)(n_epochs=EPOCHS, **kwargs).fit_transform(X32)
    assert embedding.shape == (X32.shape[0], 2)
    assert np.isfinite(embedding).all()
