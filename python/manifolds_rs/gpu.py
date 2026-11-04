"""GPU estimators, present only when the extension was built with them.

`manifolds_rs.gpu_available()` is the check worth making: it says whether this
machine has an adapter, which is the part that varies. The ordinary wheel has
GPU support compiled in, so importing this module only fails on a build made
with ``--no-default-features``.

Four things differ from the CPU estimators, all of them consequences of the
backend rather than choices:

- **float32 at any size.** `fit` casts down to it, the only narrowing this
  package performs silently.
- **Different neighbour backends.** ``"nndescent_gpu"``, ``"ivf_gpu"`` and
  ``"exhaustive_gpu"``, with their own knobs. See `NeighbourParamsGpu`.
- **Only the search and the optimiser run on the device.** The UMAP Adam update,
  the t-SNE optimiser with ``approx="fft_3k_gpu"``, and the SVD and k-means of
  PHATE's spectral landmark selection. Graph construction and the initialisation
  stay on the CPU, so a small dataset can easily come out slower than the CPU
  path once the transfers are paid for. Measure.
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
from ._base import BaseEmbedding, ReferenceEmbedding
from ._params import (
    ANN_GPU,
    INITS,
    MDS_METHODS,
    METRICS,
    PHATE_LANDMARK_METHODS,
    PHATE_METRICS,
    SYMMETRIES,
    TSNE_APPROX_GPU,
    UMAP_OPTIMISERS_GPU,
    DensParams,
    NeighbourParamsGpu,
    PhateDiffusion,
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

#: The element type the device kernels run in.
_GPU_DTYPE: np.dtype = np.dtype(np.float32)


class UMAPGpu(BaseEmbedding):
    """UMAP with a GPU neighbour search and a GPU Adam optimiser.

    ``"nndescent_gpu"`` builds the kNN graph with NN-Descent on the device and
    is the default for good reason: it is the one that scales.
    ``"exhaustive_gpu"`` gives exact neighbours and is the honest choice for
    ground truth, at quadratic cost.

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
        verbose: ``0`` quiet, ``1`` normal, ``2`` detailed.
        nn_params: See `NeighbourParamsGpu`.
        graph_params: See `UmapGraph`.
        optim_params: See `UmapOptim`.
    """

    _FN: ClassVar[Callable[..., Any]] = _core.umap_gpu
    _BASE_DTYPE: ClassVar[np.dtype | None] = _GPU_DTYPE

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
            "graph_params": merge(self.graph_params),
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
    """t-SNE with a GPU neighbour search and, optionally, a GPU optimiser.

    With ``approx="barnes_hut"`` only the search moves to the device and the
    repulsion stays on the CPU. ``approx="fft_3k_gpu"`` runs the whole
    optimisation on the device as well: FFT-interpolated repulsion with three
    kernels, the embedding resident on the GPU throughout.

    The search defaults to ``"ivf_gpu"`` rather than ``"nndescent_gpu"``: t-SNE
    asks for ``3 * perplexity`` neighbours, and IVF holds up better at that `k`.

    Args:
        n_components: Output dimensionality. Must be 2.
        perplexity: Effective neighbourhood size.
        metric: ``"euclidean"``/``"l2"`` or ``"cosine"``.
        n_epochs: Optimisation epochs.
        learning_rate: ``None`` applies the ``max(N / 12, 200)`` heuristic.
        init: ``"pca"``, ``"spectral"`` or ``"random"``.
        ann: ``"nndescent_gpu"``, ``"ivf_gpu"`` or ``"exhaustive_gpu"``.
        approx: ``"barnes_hut"``, ``"qd"``, ``"fft"`` or ``"fft_3k"`` (CPU
            optimisers, see `TSNE`), or ``"fft_3k_gpu"`` (GPU optimiser).
        randomised: Use randomised SVD for the PCA initialisation.
        init_range: Scale of the initial coordinates.
        seed: Fixes the initialisation.
        verbose: ``0`` quiet, ``1`` normal, ``2`` detailed.
        nn_params: See `NeighbourParamsGpu`.
        optim_params: See `TsneOptim`.
    """

    _FN: ClassVar[Callable[..., Any]] = _core.tsne_gpu
    _BASE_DTYPE: ClassVar[np.dtype | None] = _GPU_DTYPE
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
        ann: str = "ivf_gpu",
        approx: str = "barnes_hut",
        randomised: bool = True,
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
        self.randomised = randomised
        self.init_range = init_range
        self.seed = seed
        self.verbose = verbose
        self.nn_params = nn_params
        self.optim_params = optim_params

    def _params(self) -> dict[str, Any]:
        check_choice(self.approx, TSNE_APPROX_GPU, name="approx")
        return {
            "n_dim": self.n_components,
            "perplexity": self.perplexity,
            "initialisation": check_choice(self.init, INITS, name="init"),
            "ann_type": check_choice(self.ann, ANN_GPU, name="ann"),
            "randomised": self.randomised,
            "init_range": self.init_range,
            "nn_params": merge(
                self.nn_params,
                dist_metric=check_choice(self.metric, METRICS, name="metric"),
            ),
            "optim_params": merge(
                self.optim_params, n_epochs=self.n_epochs, lr=self.learning_rate
            ),
        }


class DREAMSGpu(ReferenceEmbedding):
    """DREAMS with a GPU neighbour search and, optionally, a GPU optimiser.

    t-SNE regularised towards a global reference embedding, as `DREAMS`, with
    the device split of `TSNEGpu`. ``approx="fft_3k_gpu"`` runs the whole
    optimisation on the device.

    Args:
        lambda_: Regularisation strength, in ``[0, 1]``.
        init: Ignored; accepted so the signature matches `TSNEGpu`.

    Everything else is as `TSNEGpu`. The reference is passed to `fit`, see
    `fit`.
    """

    _FN: ClassVar[Callable[..., Any]] = _core.dreams_gpu
    _BASE_DTYPE: ClassVar[np.dtype | None] = _GPU_DTYPE
    _EXTRA: ClassVar[tuple[str, ...]] = ("approx",)

    @beartype
    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        metric: str = "euclidean",
        lambda_: float = 0.15,
        n_epochs: int = 1000,
        learning_rate: float | None = None,
        init: str = "pca",
        ann: str = "ivf_gpu",
        approx: str = "barnes_hut",
        randomised: bool = True,
        init_range: float | None = None,
        seed: int = 42,
        verbose: int = 0,
        nn_params: NeighbourParamsGpu | None = None,
        optim_params: TsneOptim | None = None,
    ) -> None:
        self.n_components = n_components
        self.perplexity = perplexity
        self.metric = metric
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.init = init
        self.ann = ann
        self.approx = approx
        self.randomised = randomised
        self.init_range = init_range
        self.seed = seed
        self.verbose = verbose
        self.nn_params = nn_params
        self.optim_params = optim_params

    def _params(self) -> dict[str, Any]:
        check_choice(self.approx, TSNE_APPROX_GPU, name="approx")
        return {
            "n_dim": self.n_components,
            "perplexity": self.perplexity,
            "lambda": self.lambda_,
            "initialisation": check_choice(self.init, INITS, name="init"),
            "ann_type": check_choice(self.ann, ANN_GPU, name="ann"),
            "randomised": self.randomised,
            "init_range": self.init_range,
            "nn_params": merge(
                self.nn_params,
                dist_metric=check_choice(self.metric, METRICS, name="metric"),
            ),
            "optim_params": merge(
                self.optim_params, n_epochs=self.n_epochs, lr=self.learning_rate
            ),
        }


class PHATEGpu(BaseEmbedding):
    """PHATE with the neighbour search and landmark selection on the GPU.

    Same embedding as `PHATE`. The kNN search and, for spectral landmarks, the
    randomised SVD and the k-means run on the device. The diffusion time, the
    powering, the MDS and the interpolation stay on the CPU.

    Args:
        n_components: Output dimensionality.
        k: Neighbours that set the kernel bandwidth. The search runs ``6k + 5``
            wide, so give a precomputed graph that width as well.
        metric: ``"euclidean"``/``"l2"`` or ``"cosine"``.
        decay: Alpha-decay exponent of the kernel. ``None`` gives a binary
            connectivity kernel instead.
        t: Diffusion time. ``None`` picks the VNE knee.
        gamma: Informational distance constant, in ``[-1, 1]``.
        n_landmarks: Landmarks to diffuse on. ``None``, or a value at least
            `n_samples`, runs the full operator, whose dense potential distances
            grow with the square of `n_samples`.
        landmark_method: ``"spectral"``, ``"random"`` or ``"density"``. An mlx
            build adds ``"spectral_mlx"``, which keeps the search on wgpu
            and runs the landmark selection on MLX.
        mds: ``"sgd_dense"`` (also ``"dense"``) or ``"classic"``.
        mds_iter: MDS iterations. ``None`` uses the backend default.
        ann: ``"nndescent_gpu"``, ``"ivf_gpu"`` or ``"exhaustive_gpu"``.
        randomised: Use randomised SVD for the initialisation.
        seed: Fixes the sketch, the k-means initialisation and the MDS.
        verbose: ``0`` quiet, ``1`` normal, ``2`` detailed.
        nn_params: See `NeighbourParamsGpu`.
        diffusion_params: Remaining operator knobs. See `PhateDiffusion`.
    """

    _FN: ClassVar[Callable[..., Any]] = _core.phate_gpu
    _BASE_DTYPE: ClassVar[np.dtype | None] = _GPU_DTYPE

    @beartype
    def __init__(
        self,
        n_components: int = 2,
        k: int = 5,
        metric: str = "euclidean",
        decay: float | None = 40.0,
        t: int | None = None,
        gamma: float = 1.0,
        n_landmarks: int | None = 2048,
        landmark_method: str = "spectral",
        mds: str = "sgd_dense",
        mds_iter: int | None = None,
        ann: str = "nndescent_gpu",
        randomised: bool = True,
        seed: int = 42,
        verbose: int = 0,
        nn_params: NeighbourParamsGpu | None = None,
        diffusion_params: PhateDiffusion | None = None,
    ) -> None:
        self.n_components = n_components
        self.k = k
        self.metric = metric
        self.decay = decay
        self.t = t
        self.gamma = gamma
        self.n_landmarks = n_landmarks
        self.landmark_method = landmark_method
        self.mds = mds
        self.mds_iter = mds_iter
        self.ann = ann
        self.randomised = randomised
        self.seed = seed
        self.verbose = verbose
        self.nn_params = nn_params
        self.diffusion_params = diffusion_params

    def _params(self) -> dict[str, Any]:
        diffusion = merge(self.diffusion_params, gamma=self.gamma) or {}
        # `decay` and `n_landmarks` are set rather than merged: `None` selects the
        # binary kernel and the full operator, and `merge` drops nulls as "leave
        # the default".
        diffusion["decay"] = self.decay
        diffusion["n_landmarks"] = self.n_landmarks
        if self.t is not None:
            diffusion["t"] = self.t
        if "graph_symmetry" in diffusion:
            diffusion["graph_symmetry"] = check_choice(
                diffusion["graph_symmetry"], SYMMETRIES, name="graph_symmetry"
            )
        diffusion["landmark_method"] = check_choice(
            self.landmark_method, PHATE_LANDMARK_METHODS, name="landmark_method"
        )
        return {
            "n_dim": self.n_components,
            "k": self.k,
            "ann_type": check_choice(self.ann, ANN_GPU, name="ann"),
            "mds_method": check_choice(self.mds, MDS_METHODS, name="mds"),
            "mds_iter": self.mds_iter,
            "randomised": self.randomised,
            "nn_params": merge(
                self.nn_params,
                dist_metric=check_choice(self.metric, PHATE_METRICS, name="metric"),
            ),
            "diffusion_params": diffusion,
        }
