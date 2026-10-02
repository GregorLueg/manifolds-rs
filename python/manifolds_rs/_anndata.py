"""Scanpy-style functional API over `AnnData`.

``mf.umap(adata)`` and ``mf.tsne(adata)`` write the same slots ``sc.tl.umap``
and ``sc.tl.tsne`` do: coordinates in ``obsm["X_umap"]`` / ``obsm["X_tsne"]``
and parameters in ``uns["umap"]`` / ``uns["tsne"]``, so ``sc.pl.umap`` and
friends work unchanged. Scanpy itself is never imported.

UMAP reuses the graph from ``sc.pp.neighbors``, as scanpy's does: the kNN
indices and distances are read back out of ``obsp`` and the fuzzy graph is
rebuilt from them by the crate. t-SNE searches on the representation itself,
also as scanpy's does, since a 15-neighbour graph is far narrower than the
``3 * perplexity`` neighbours t-SNE needs.

Imported on first access of ``mf.umap`` / ``mf.tsne``, so `anndata` stays an
optional dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from anndata import AnnData
from beartype import beartype
from scipy import sparse

from . import embeddings
from ._base import BaseEmbedding

###########
# Helpers #
###########


@beartype
def _estimator(name: str, device: str, **kwargs: Any) -> BaseEmbedding:
    """Build the CPU or GPU estimator for `name`.

    `kwargs` go to the estimator constructor, which is beartyped and rejects
    unknown names, so nothing is swallowed here.
    """
    cls: type[BaseEmbedding]
    match name, device:
        case "umap", "cpu":
            cls = embeddings.UMAP
        case "tsne", "cpu":
            cls = embeddings.TSNE
        case "umap", "gpu":
            from . import gpu

            cls = gpu.UMAPGpu
        case "tsne", "gpu":
            from . import gpu

            cls = gpu.TSNEGpu
        case _:
            raise ValueError(f"unknown device {device!r}; expected 'cpu' or 'gpu'")
    return cls(**kwargs)


@beartype
def _resolve_rep(adata: AnnData, use_rep: str | None, n_pcs: int | None) -> Any:
    """The dense matrix to embed, picked the way scanpy picks it.

    ``None`` means ``obsm["X_pca"]``, ``"X"`` means ``adata.X``, anything else
    is an ``obsm`` key. `n_pcs` slices the leading columns. There is no
    fallback PCA: run ``sc.pp.pca`` first.
    """
    match use_rep:
        case None:
            if "X_pca" not in adata.obsm:
                raise ValueError(
                    "no obsm['X_pca']; run sc.pp.pca first or pass use_rep"
                )
            x = adata.obsm["X_pca"]
        case "X":
            x = adata.X
        case str() if use_rep in adata.obsm:
            x = adata.obsm[use_rep]
        case _:
            raise ValueError(f"use_rep {use_rep!r} is not 'X' or a key of obsm")
    if x is None:
        raise ValueError("adata.X is None; pass use_rep")
    if sparse.issparse(x):
        raise ValueError(
            f"use_rep {use_rep!r} is sparse; the embeddings need a dense matrix, "
            f"usually obsm['X_pca']"
        )
    if n_pcs is not None:
        if n_pcs > x.shape[1]:
            raise ValueError(f"n_pcs={n_pcs} but the representation has {x.shape[1]}")
        x = x[:, :n_pcs]
    return x


@beartype
def _knn_from_obsp(adata: AnnData, neighbors_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the kNN graph ``sc.pp.neighbors`` stored back into dense arrays.

    ``obsp[distances_key]`` holds ``k - 1`` true distances per row, self
    excluded, which is exactly what the estimators take. Rows are re-sorted by
    distance rather than trusting the stored order.
    """
    key = adata.uns[neighbors_key].get("distances_key", "distances")
    if key not in adata.obsp:
        raise ValueError(f"no obsp[{key!r}]; rerun sc.pp.neighbors")
    dist = sparse.csr_matrix(adata.obsp[key])
    dist.sort_indices()
    nnz = np.diff(dist.indptr)
    if nnz.size == 0 or nnz.min() != nnz.max() or nnz[0] == 0:
        raise ValueError(
            f"obsp[{key!r}] does not hold the same number of neighbours on every "
            f"row; it was not built by sc.pp.neighbors with a fixed k"
        )
    k = int(nnz[0])
    ind = dist.indices.reshape(-1, k).astype(np.int64)
    d = dist.data.reshape(-1, k)
    order = np.argsort(d, axis=1, kind="stable")
    return np.take_along_axis(ind, order, axis=1), np.take_along_axis(d, order, axis=1)


@beartype
def _write(
    adata: AnnData,
    name: str,
    key_added: str | None,
    embedding: np.ndarray,
    params: dict[str, Any],
) -> None:
    """Store coordinates and parameters under scanpy's keys.

    Only scalar parameters are kept, and ``None`` is dropped, so the result
    survives ``write_h5ad``.
    """
    adata.obsm[key_added or f"X_{name}"] = embedding
    adata.uns[key_added or name] = {
        "params": {
            k: v for k, v in params.items() if isinstance(v, bool | int | float | str)
        }
    }


###############
# Entry point #
###############


@beartype
def umap(
    adata: AnnData,
    *,
    n_components: int = 2,
    min_dist: float = 0.5,
    spread: float = 1.0,
    neighbors_key: str = "neighbors",
    key_added: str | None = None,
    device: str = "cpu",
    seed: int = 42,
    verbose: int = 0,
    copy: bool = False,
    **kwargs: Any,
) -> AnnData | None:
    """UMAP on the ``sc.pp.neighbors`` graph, stored the way ``sc.tl.umap`` does.

    The neighbour graph is taken from ``obsp`` rather than searched again, so
    `n_neighbors` and the metric are whatever ``sc.pp.neighbors`` used. The
    representation (``use_rep`` / ``n_pcs``) is read from the same
    ``uns[neighbors_key]["params"]``; it feeds the initialisation.

    Args:
        adata: Annotated data matrix, after ``sc.pp.neighbors``.
        n_components: Output dimensionality.
        min_dist: How tightly points may pack. Fits the repulsion curve with
            `spread`.
        spread: Scale of the embedding relative to `min_dist`.
        neighbors_key: Where ``sc.pp.neighbors`` stored its results in ``uns``.
        key_added: Store under ``obsm[key_added]`` and ``uns[key_added]``
            instead of ``obsm["X_umap"]`` and ``uns["umap"]``.
        device: ``"cpu"`` for `manifolds_rs.UMAP`, ``"gpu"`` for
            `manifolds_rs.UMAPGpu`.
        seed: Fixes the initialisation and the negative sampling.
        verbose: ``0`` silent, ``1`` normal, ``2`` detailed.
        copy: Return a modified copy instead of writing into `adata`.
        **kwargs: Any other constructor argument of the chosen estimator, e.g.
            ``n_epochs``, ``init``, ``optim_params``. Unknown names raise.

    Returns:
        The copy if `copy`, else ``None``. Sets ``obsm["X_umap" | key_added]``
        and ``uns["umap" | key_added]["params"]``.

    Raises:
        ValueError: If ``sc.pp.neighbors`` has not run, its graph is ragged, or
            the representation is missing or sparse.
    """
    adata = adata.copy() if copy else adata
    if neighbors_key not in adata.uns:
        raise ValueError(f"no uns[{neighbors_key!r}]; run sc.pp.neighbors first")
    nn_params = adata.uns[neighbors_key].get("params", {})
    use_rep = nn_params.get("use_rep")
    n_pcs = nn_params.get("n_pcs")
    x = _resolve_rep(
        adata,
        None if use_rep is None else str(use_rep),
        None if n_pcs is None else int(n_pcs),
    )
    ind, dist = _knn_from_obsp(adata, neighbors_key)

    est = _estimator(
        "umap",
        device,
        n_components=n_components,
        min_dist=min_dist,
        spread=spread,
        n_neighbors=ind.shape[1],
        seed=seed,
        verbose=verbose,
        **kwargs,
    )
    embedding = est.fit_transform(x, knn_indices=ind, knn_distances=dist)
    # The search never ran, so its backend and metric would be misleading.
    params = {k: v for k, v in est.get_params().items() if k not in ("ann", "metric")}
    _write(
        adata,
        "umap",
        key_added,
        embedding,
        {**params, "device": device, "neighbors_key": neighbors_key},
    )
    return adata if copy else None


@beartype
def tsne(
    adata: AnnData,
    n_pcs: int | None = None,
    *,
    use_rep: str | None = None,
    perplexity: float = 30.0,
    approx: str = "barnes_hut",
    key_added: str | None = None,
    device: str = "cpu",
    seed: int = 42,
    verbose: int = 0,
    copy: bool = False,
    **kwargs: Any,
) -> AnnData | None:
    """t-SNE on a representation, stored the way ``sc.tl.tsne`` does.

    Searches its own ``3 * perplexity`` neighbours; ``sc.pp.neighbors`` is not
    used.

    Args:
        adata: Annotated data matrix.
        n_pcs: Leading columns of the representation to use. ``None`` uses all.
        use_rep: ``None`` for ``obsm["X_pca"]``, ``"X"`` for ``adata.X``, or
            any other ``obsm`` key. Must be dense.
        perplexity: Effective neighbourhood size.
        approx: ``"barnes_hut"``, ``"qd"``, or ``"fft_3k_gpu"`` with
            ``device="gpu"``.
        key_added: Store under ``obsm[key_added]`` and ``uns[key_added]``
            instead of ``obsm["X_tsne"]`` and ``uns["tsne"]``.
        device: ``"cpu"`` for `manifolds_rs.TSNE`, ``"gpu"`` for
            `manifolds_rs.TSNEGpu`.
        seed: Fixes the initialisation.
        verbose: ``0`` silent, ``1`` normal, ``2`` detailed.
        copy: Return a modified copy instead of writing into `adata`.
        **kwargs: Any other constructor argument of the chosen estimator, e.g.
            ``n_epochs``, ``learning_rate``, ``ann``, ``metric``. Unknown names
            raise.

    Returns:
        The copy if `copy`, else ``None``. Sets ``obsm["X_tsne" | key_added]``
        and ``uns["tsne" | key_added]["params"]``.

    Raises:
        ValueError: If the representation is missing or sparse, or `approx` is
            not available on `device`.
    """
    adata = adata.copy() if copy else adata
    x = _resolve_rep(adata, use_rep, n_pcs)
    est = _estimator(
        "tsne",
        device,
        perplexity=perplexity,
        approx=approx,
        seed=seed,
        verbose=verbose,
        **kwargs,
    )
    embedding = est.fit_transform(x)
    _write(
        adata,
        "tsne",
        key_added,
        embedding,
        {**est.get_params(), "device": device, "use_rep": use_rep, "n_pcs": n_pcs},
    )
    return adata if copy else None
