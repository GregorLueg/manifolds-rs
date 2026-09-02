"""Dimensionality reduction for single-cell and computational biology, in Rust.

UMAP, densMAP, t-SNE, den-SNE, PHATE, PaCMAP and diffusion maps behind one
estimator surface, with GPU variants of the first three where the neighbour
search and the Adam update move to the device.

    >>> import manifolds_rs as mf
    >>> X, labels = mf.datasets.clustered(20_000, dim=50, n_clusters=12)
    >>> embedding = mf.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(X)
    >>> embedding.shape
    (20000, 2)

The knobs people actually turn are ordinary constructor arguments. Everything
else lives in the frozen dataclasses in `manifolds_rs` alongside them, one per
group, and a field left alone keeps the crate's own default:

    >>> mf.UMAP(
    ...     n_neighbors=30,
    ...     ann="hnsw",
    ...     nn_params=mf.NeighbourParams(m=32, ef_search=200),
    ...     optim_params=mf.UmapOptim(gamma=1.5, neg_sample_rate=10),
    ... )  # doctest: +ELLIPSIS
    UMAP(...)

Running several embeddings over the same data? Build the neighbour graph once
with `knn_graph` and pass it to each `fit`; on anything large the search is most
of the runtime.

None of these algorithms projects new points, so there is no `transform`. That
is the crate's position, not a gap in the bindings: embedding new data means
refitting, which moves the existing coordinates too.
"""

from . import _manifolds, datasets
from ._base import BaseEmbedding, NotFittedError
from ._manifolds import (
    ConvergenceError,
    ManifoldsRsError,
    __core_version__,
    __version__,
    gpu_available,
    num_threads,
    set_num_threads,
)
from ._params import (
    DensParams,
    NeighbourParams,
    NeighbourParamsGpu,
    PacmapOptim,
    PhateDiffusion,
    TsneOptim,
    UmapGraph,
    UmapOptim,
)
from .embeddings import PHATE, TSNE, UMAP, DensMAP, DensNE, DiffusionMaps, PaCMAP
from .neighbours import knn_graph

__all__ = [
    "PHATE",
    "TSNE",
    "UMAP",
    "BaseEmbedding",
    "ConvergenceError",
    "DensMAP",
    "DensNE",
    "DensParams",
    "DiffusionMaps",
    "ManifoldsRsError",
    "NeighbourParams",
    "NeighbourParamsGpu",
    "NotFittedError",
    "PaCMAP",
    "PacmapOptim",
    "PhateDiffusion",
    "TsneOptim",
    "UmapGraph",
    "UmapOptim",
    "__core_version__",
    "__version__",
    "datasets",
    "gpu_available",
    "knn_graph",
    "num_threads",
    "set_num_threads",
]

# The GPU estimators exist only when the extension was built with them, which is
# fixed at wheel-build time. They are re-exported at the top level when present
# so `mf.UMAPGpu` works, and `manifolds_rs.gpu` stays importable either way for
# anyone who wants the ImportError to say why.
if hasattr(_manifolds, "umap_gpu"):  # pragma: no cover - build-dependent
    from .gpu import DensMAPGpu, TSNEGpu, UMAPGpu

    __all__ += ["DensMAPGpu", "TSNEGpu", "UMAPGpu"]
