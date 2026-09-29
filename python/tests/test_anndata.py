"""The scanpy-style `mf.umap(adata)` / `mf.tsne(adata)` wrappers.

Scanpy is not a dependency, so ``sc.pp.neighbors`` is faked: its output is a
CSR of ``k - 1`` true distances per row in ``obsp["distances"]`` plus a small
dict in ``uns["neighbors"]``, and that is all the wrappers read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import EPOCHS

import manifolds_rs as mf

ad = pytest.importorskip("anndata")
sparse = pytest.importorskip("scipy.sparse")


@pytest.fixture
def adata(X: np.ndarray) -> ad.AnnData:
    """An AnnData shaped like the output of ``sc.pp.pca`` + ``sc.pp.neighbors``."""
    n = X.shape[0]
    a = ad.AnnData(np.zeros((n, 3), dtype=np.float32))
    a.obsm["X_pca"] = X.astype(np.float32)
    ind, dist = mf.knn_graph(a.obsm["X_pca"], k=14, seed=7)
    a.obsp["distances"] = sparse.csr_matrix(
        (dist.ravel(), ind.ravel(), np.arange(0, ind.size + 1, ind.shape[1])),
        shape=(n, n),
    )
    a.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {"n_neighbors": 15, "method": "umap", "metric": "euclidean"},
    }
    return a


def test_umap_writes_scanpy_slots(adata: ad.AnnData) -> None:
    assert mf.umap(adata, n_epochs=EPOCHS) is None
    assert adata.obsm["X_umap"].shape == (adata.n_obs, 2)
    params = adata.uns["umap"]["params"]
    assert params["n_neighbors"] == 14
    assert params["neighbors_key"] == "neighbors"


def test_tsne_writes_scanpy_slots(adata: ad.AnnData) -> None:
    mf.tsne(adata, n_epochs=EPOCHS)
    assert adata.obsm["X_tsne"].shape == (adata.n_obs, 2)
    assert adata.uns["tsne"]["params"]["perplexity"] == 30.0


def test_key_added_routes_both_slots(adata: ad.AnnData) -> None:
    mf.umap(adata, n_epochs=EPOCHS, key_added="umap_mf")
    mf.tsne(adata, n_epochs=EPOCHS, key_added="tsne_mf")
    assert {"umap_mf", "tsne_mf"} <= set(adata.obsm) & set(adata.uns)
    assert "X_umap" not in adata.obsm and "X_tsne" not in adata.obsm


def test_copy_leaves_the_input_alone(adata: ad.AnnData) -> None:
    out = mf.umap(adata, n_epochs=EPOCHS, copy=True)
    assert out is not None and "X_umap" in out.obsm
    assert "X_umap" not in adata.obsm


def test_params_survive_h5ad(adata: ad.AnnData, tmp_path: Path) -> None:
    mf.umap(adata, n_epochs=EPOCHS)
    adata.write_h5ad(tmp_path / "a.h5ad")
    back = ad.read_h5ad(tmp_path / "a.h5ad")
    assert np.array_equal(back.obsm["X_umap"], adata.obsm["X_umap"])


def test_umap_without_neighbors_raises(adata: ad.AnnData) -> None:
    del adata.uns["neighbors"]
    with pytest.raises(ValueError, match=r"sc\.pp\.neighbors"):
        mf.umap(adata)


def test_ragged_neighbour_graph_raises(adata: ad.AnnData) -> None:
    d = adata.obsp["distances"].tolil()
    d[0, d.rows[0][0]] = 0
    d = d.tocsr()
    d.eliminate_zeros()
    adata.obsp["distances"] = d
    with pytest.raises(ValueError, match="same number of neighbours"):
        mf.umap(adata)


def test_gpu_approx_on_cpu_raises(adata: ad.AnnData) -> None:
    with pytest.raises(ValueError, match="approx"):
        mf.tsne(adata, approx="fft_3k_gpu")
