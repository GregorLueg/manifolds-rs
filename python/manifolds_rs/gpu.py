"""GPU estimators, present only when the extension was built with them.

`manifolds_rs.gpu_available()` is the check worth making: it says whether this
machine has an adapter, which is the part that varies. The ordinary wheel has
GPU support compiled in, so importing this module only fails on a build made
with ``--no-default-features``.

Three things differ from the CPU estimators, all of them consequences of the
backend rather than choices:

- **float32 only.** WGSL has no float64, so `fit` casts rather than letting a
  float64 array fail somewhere inside a kernel. That is a narrowing conversion,
  and the only one this package performs silently.
- **Different neighbour backends.** ``"nndescent_gpu"``, ``"ivf_gpu"`` and
  ``"exhaustive_gpu"``, with their own knobs. See `NeighbourParamsGpu`.
- **Only the search and the Adam update run on the device.** Graph construction
  and the spectral initialisation stay on the CPU, so a small dataset can easily
  come out slower than the CPU path once the transfers are paid for. Measure.
- **Reproducible in structure, not in coordinates.** The device searches are not
  always bit-stable at scale, and the optimiser amplifies a small difference in
  the graph into visibly different positions. Cluster structure is preserved
  exactly; the coordinates may move. See the guide.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from beartype import beartype

from . import _manifolds as _core
from ._base import BaseEmbedding
from ._params import (
    ANN_GPU,
    INITS,
    METRICS,
    TSNE_APPROX,
    UMAP_OPTIMISERS_GPU,
    DensParams,
    NeighbourParamsGpu,
    TsneOptim,
    UmapGraph,
    UmapOptim,
    merge,
)
from ._validate import check_choice

if not hasattr(_core, "umap_gpu"):  # pragma: no cover - build-dependent
    raise ImportError(
        "this build of manifolds-rs has no GPU support compiled in. The 'gpu' "
        "feature is on by default, so this is a --no-default-features build; "
        "reinstall without that flag to get the GPU estimators."
    )

###########
# Globals #
###########

#: The only element type the GPU backend can carry.
_GPU_DTYPE: np.dtype = np.dtype(np.float32)


class UMAPGpu(BaseEmbedding):
    """UMAP with a GPU neighbour search and a GPU Adam optimiser.

    ``"nndescent_gpu"`` builds a CAGRA graph on the device and is the default
    for good reason: it is the one that scales. ``"exhaustive_gpu"`` gives exact
    neighbours and is the honest choice for ground truth, at quadratic cost.

    Args:
        n_components: Output dimensionality.
        n_neighbors: Neighbours per point.
        metric: ``"euclidean"``/``"l2"`` or ``"cosine"``.
        min_dist: How tightly points may pack. Fits the repulsion curve with
            `spread`.
        spread: Scale of the embedding relative to `min_dist`.
        n_epochs: Optimisation epochs.
        learning_rate: Initial learning rate.
        init: ``"spectral"``, ``"pca"`` or ``"random"``. Computed on the CPU
            either way.
        ann: ``"nndescent_gpu"``, ``"ivf_gpu"`` or ``"exhaustive_gpu"``.
        optimiser: ``"adam_gpu"`` keeps the update on the device. The CPU names
            still work and pull the embedding back each epoch, which is only
            worth it for debugging.
        randomised: Use randomised SVD for the PCA initialisation.
        init_range: Scale of the initial coordinates.
        seed: Fixes the initialisation and the negative sampling.
        verbose: ``0`` silent, ``1`` normal, ``2`` detailed.
        nn_params: See `NeighbourParamsGpu`.
        graph_params: See `UmapGraph`.
        optim_params: See `UmapOptim`.
    """

    _FN: ClassVar[Callable[..., Any]] = _core.umap_gpu
    _FORCE_DTYPE: ClassVar[np.dtype | None] = _GPU_DTYPE

    @beartype
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        min_dist: float = 0.5,
        spread: float = 1.0,
        n_epochs: int = 500,
        learning_rate: float = 1.0,
        init: str = "spectral",
        ann: str = "nndescent_gpu",
        optimiser: str = "adam_gpu",
        randomised: bool = False,
        init_range: float | None = None,
        seed: int = 42,
        verbose: int = 0,
        nn_params: NeighbourParamsGpu | None = None,
        graph_params: UmapGraph | None = None,
        optim_params: UmapOptim | None = None,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.min_dist = min_dist
        self.spread = spread
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.init = init
        self.ann = ann
        self.optimiser = optimiser
        self.randomised = randomised
        self.init_range = init_range
        self.seed = seed
        self.verbose = verbose
        self.nn_params = nn_params
        self.graph_params = graph_params
        self.optim_params = optim_params

    def _params(self) -> dict[str, Any]:
        return {
            "n_dim": self.n_components,
            "k": self.n_neighbors,
            "min_dist": self.min_dist,
            "spread": self.spread,
            "initialisation": check_choice(self.init, INITS, name="init"),
            "ann_type": check_choice(self.ann, ANN_GPU, name="ann"),
            "optimiser": check_choice(
                self.optimiser, UMAP_OPTIMISERS_GPU, name="optimiser"
            ),
            "randomised": self.randomised,
            "init_range": self.init_range,
            "nn_params": merge(
                self.nn_params,
                dist_metric=check_choice(self.metric, METRICS, name="metric"),
            ),
            "umap_graph_params": merge(self.graph_params),
            "optim_params": merge(
                self.optim_params, n_epochs=self.n_epochs, lr=self.learning_rate
            ),
        }


class DensMAPGpu(UMAPGpu):
    """densMAP on the GPU.

    With the default ``"adam_gpu"`` optimiser the density term runs on the
    device alongside the rest of the update.

    Args:
        lambda_: Weight on the density term. ``0`` recovers plain UMAP.
        dens_params: Remaining density knobs. See `DensParams`.

    Everything else is as `UMAPGpu`.
    """

    _FN: ClassVar[Callable[..., Any]] = _core.densmap_gpu

    @beartype
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        min_dist: float = 0.5,
        spread: float = 1.0,
        lambda_: float = 2.0,
        n_epochs: int = 500,
        learning_rate: float = 1.0,
        init: str = "spectral",
        ann: str = "nndescent_gpu",
        optimiser: str = "adam_gpu",
        randomised: bool = False,
        init_range: float | None = None,
        seed: int = 42,
        verbose: int = 0,
        nn_params: NeighbourParamsGpu | None = None,
        graph_params: UmapGraph | None = None,
        optim_params: UmapOptim | None = None,
        dens_params: DensParams | None = None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric=metric,
            min_dist=min_dist,
            spread=spread,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            init=init,
            ann=ann,
            optimiser=optimiser,
            randomised=randomised,
            init_range=init_range,
            seed=seed,
            verbose=verbose,
            nn_params=nn_params,
            graph_params=graph_params,
            optim_params=optim_params,
        )
        self.lambda_ = lambda_
        self.dens_params = dens_params

    def _params(self) -> dict[str, Any]:
        return {
            **super()._params(),
            "lambda": self.lambda_,
            "dens_params": merge(self.dens_params),
        }


class TSNEGpu(BaseEmbedding):
    """t-SNE with a GPU neighbour search.

    Only the search moves to the device here; the Barnes-Hut repulsion stays on
    the CPU. On a dataset where the search dominates that is most of the win,
    and on one where it does not you should not expect much.

    When `ann` is ``"nndescent_gpu"`` and `NeighbourParamsGpu.k` is left unset,
    the CAGRA graph degree is backfilled to ``3 * perplexity`` so it is sized for
    the query t-SNE actually makes.

    Args:
        n_components: Output dimensionality. Must be 2.
        perplexity: Effective neighbourhood size.
        metric: ``"euclidean"``/``"l2"`` or ``"cosine"``.
        n_epochs: Optimisation epochs.
        learning_rate: ``None`` applies the ``max(N / 12, 200)`` heuristic.
        init: ``"pca"``, ``"spectral"`` or ``"random"``.
        ann: ``"nndescent_gpu"``, ``"ivf_gpu"`` or ``"exhaustive_gpu"``.
        approx: Repulsion approximation. See `manifolds_rs.TSNE`.
        randomised_init: Use randomised SVD for the PCA initialisation.
        init_range: Scale of the initial coordinates.
        seed: Fixes the initialisation.
        verbose: ``0`` silent, ``1`` normal, ``2`` detailed.
        nn_params: See `NeighbourParamsGpu`.
        optim_params: See `TsneOptim`.
    """

    _FN: ClassVar[Callable[..., Any]] = _core.tsne_gpu
    _FORCE_DTYPE: ClassVar[np.dtype | None] = _GPU_DTYPE
    _EXTRA: ClassVar[tuple[str, ...]] = ("approx",)

    @beartype
    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        metric: str = "euclidean",
        n_epochs: int = 1000,
        learning_rate: float | None = None,
        init: str = "pca",
        ann: str = "nndescent_gpu",
        approx: str = "barnes_hut",
        randomised_init: bool = True,
        init_range: float | None = None,
        seed: int = 42,
        verbose: int = 0,
        nn_params: NeighbourParamsGpu | None = None,
        optim_params: TsneOptim | None = None,
    ) -> None:
        self.n_components = n_components
        self.perplexity = perplexity
        self.metric = metric
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.init = init
        self.ann = ann
        self.approx = approx
        self.randomised_init = randomised_init
        self.init_range = init_range
        self.seed = seed
        self.verbose = verbose
        self.nn_params = nn_params
        self.optim_params = optim_params

    def _params(self) -> dict[str, Any]:
        check_choice(self.approx, TSNE_APPROX, name="approx")
        return {
            "n_dim": self.n_components,
            "perplexity": self.perplexity,
            "initialisation": check_choice(self.init, INITS, name="init"),
            "ann_type": check_choice(self.ann, ANN_GPU, name="ann"),
            "randomised_init": self.randomised_init,
            "init_range": self.init_range,
            "nn_params": merge(
                self.nn_params,
                dist_metric=check_choice(self.metric, METRICS, name="metric"),
            ),
            "optim_params": merge(
                self.optim_params, n_epochs=self.n_epochs, lr=self.learning_rate
            ),
        }
