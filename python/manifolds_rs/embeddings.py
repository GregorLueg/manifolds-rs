"""The CPU estimators.

Each class stores its constructor arguments verbatim, names the core function it
drives, and assembles the parameter payload in ``_params``. Everything else
comes from `BaseEmbedding`.

Defaults are the crate's own, and there is a test asserting exactly that: an
estimator left alone must produce the same embedding as calling the core with an
empty payload. A default that drifts here fails it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from beartype import beartype

from . import _manifolds as _core
from ._base import BaseEmbedding
from ._params import (
    ANN_CPU,
    INITS,
    LANDMARK_METHODS,
    MDS_METHODS,
    METRICS,
    PACMAP_OPTIMISERS,
    SYMMETRIES,
    TSNE_APPROX,
    UMAP_OPTIMISERS,
    DensParams,
    NeighbourParams,
    PacmapOptim,
    PhateDiffusion,
    TsneOptim,
    UmapGraph,
    UmapOptim,
    merge,
)
from ._validate import check_choice


class UMAP(BaseEmbedding):
    """Uniform manifold approximation and projection.

    The default optimiser is the multi-threaded Adam one, which is where most of
    the speed comes from; `"sgd"` reproduces the original reference behaviour
    more closely and is slower. Spectral initialisation is the default and is
    worth keeping: PCA is the fallback when the Laplacian will not converge, and
    random init throws away the global structure spectral gives you for free.

    Args:
        n_components: Output dimensionality.
        n_neighbors: Neighbours per point. The locality knob: small values chase
            fine structure, large ones preserve more of the global picture.
        metric: ``"euclidean"``/``"l2"``, ``"cosine"`` or ``"manhattan"``/``"l1"``.
            Euclidean is computed squared throughout and never rooted.
        min_dist: How tightly points may pack in the embedding. Together with
            `spread` it fits the repulsion curve; neither is used for anything
            else.
        spread: Scale of the embedding relative to `min_dist`.
        n_epochs: Optimisation epochs.
        learning_rate: Initial learning rate.
        init: ``"spectral"``, ``"pca"`` or ``"random"``.
        ann: Neighbour backend. ``"kmknn"`` is exact and fast up to a few
            hundred thousand points; ``"hnsw"`` or ``"nndescent"`` past that.
        optimiser: ``"adam_parallel"``, ``"adam"`` or ``"sgd"``.
        randomised: Use randomised SVD for the PCA initialisation. No effect
            under spectral or random init.
        init_range: Scale of the initial coordinates. ``None`` lets the core
            pick per initialisation.
        seed: Fixes the initialisation and the negative sampling.
        verbose: ``0`` silent, ``1`` normal, ``2`` detailed. Progress goes to
            the process stdout, not ``sys.stdout``. In Jupyter that lands in the
            terminal running the kernel.
        nn_params: Backend-specific neighbour knobs. See `NeighbourParams`.
        graph_params: Fuzzy simplicial set knobs. See `UmapGraph`.
        optim_params: Remaining optimiser knobs. See `UmapOptim`.
    """

    _FN: ClassVar[Callable[..., Any]] = _core.umap

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
        ann: str = "kmknn",
        optimiser: str = "adam_parallel",
        randomised: bool = False,
        init_range: float | None = None,
        seed: int = 42,
        verbose: int = 0,
        nn_params: NeighbourParams | None = None,
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
            "ann_type": check_choice(self.ann, ANN_CPU, name="ann"),
            "optimiser": check_choice(
                self.optimiser, UMAP_OPTIMISERS, name="optimiser"
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


class DensMAP(UMAP):
    """UMAP with a density-preservation term.

    Plain UMAP is free to stretch a dense region and squash a sparse one, so
    relative density in the embedding means nothing. densMAP adds a term
    correlating the local radii of the embedding with those of the original
    space, switched on only for the last `DensParams.frac` of the run so it
    corrects a settled embedding rather than fighting the layout.

    Args:
        lambda_: Weight on the density term. ``0`` recovers plain UMAP.
        dens_params: Remaining density knobs. See `DensParams`.

    Everything else is as `UMAP`.
    """

    _FN: ClassVar[Callable[..., Any]] = _core.densmap

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
        ann: str = "kmknn",
        optimiser: str = "adam_parallel",
        randomised: bool = False,
        init_range: float | None = None,
        seed: int = 42,
        verbose: int = 0,
        nn_params: NeighbourParams | None = None,
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


class TSNE(BaseEmbedding):
    """t-distributed stochastic neighbour embedding.

    Two-dimensional only, which is the core's restriction rather than this
    layer's. The learning rate defaults to the N-invariant ``max(N / 12, 200)``
    heuristic rather than a fixed 200, so it does not need retuning when the
    dataset grows.

    Args:
        n_components: Output dimensionality. Must be 2.
        perplexity: Effective neighbourhood size. The kNN search uses
            ``3 * perplexity`` neighbours, so this also sets the graph width.
        metric: ``"euclidean"``/``"l2"``, ``"cosine"`` or ``"manhattan"``/``"l1"``.
        n_epochs: Optimisation epochs.
        learning_rate: ``None`` applies the ``max(N / 12, 200)`` heuristic.
        init: ``"pca"``, ``"spectral"`` or ``"random"``. PCA is the default and
            what makes a t-SNE run reproducible in shape rather than only in
            seed.
        ann: Neighbour backend.
        approx: Repulsion approximation. ``"barnes_hut"`` unless the extension
            was built with the `fft_tsne` feature, which the published wheel is
            not: FFTW is a system library no manylinux container carries.
        randomised_init: Use randomised SVD for the PCA initialisation.
        init_range: Scale of the initial coordinates.
        seed: Fixes the initialisation.
        verbose: ``0`` silent, ``1`` normal, ``2`` detailed.
        nn_params: See `NeighbourParams`.
        optim_params: See `TsneOptim`.
    """

    _FN: ClassVar[Callable[..., Any]] = _core.tsne
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
        ann: str = "kmknn",
        approx: str = "barnes_hut",
        randomised_init: bool = True,
        init_range: float | None = None,
        seed: int = 42,
        verbose: int = 0,
        nn_params: NeighbourParams | None = None,
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
            "ann_type": check_choice(self.ann, ANN_CPU, name="ann"),
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


class DensNE(TSNE):
    """t-SNE with a density-preservation term.

    The same correction as `DensMAP`, applied to t-SNE. The default weight is
    much smaller than densMAP's because t-SNE's gradients are on a different
    scale, not because the effect is meant to be weaker.

    Args:
        lambda_: Weight on the density term. ``0`` recovers plain t-SNE.
        dens_params: Remaining density knobs. See `DensParams`.

    Everything else is as `TSNE`.
    """

    _FN: ClassVar[Callable[..., Any]] = _core.densne

    @beartype
    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        metric: str = "euclidean",
        lambda_: float = 0.1,
        n_epochs: int = 1000,
        learning_rate: float | None = None,
        init: str = "pca",
        ann: str = "kmknn",
        approx: str = "barnes_hut",
        randomised_init: bool = True,
        init_range: float | None = None,
        seed: int = 42,
        verbose: int = 0,
        nn_params: NeighbourParams | None = None,
        optim_params: TsneOptim | None = None,
        dens_params: DensParams | None = None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            perplexity=perplexity,
            metric=metric,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            init=init,
            ann=ann,
            approx=approx,
            randomised_init=randomised_init,
            init_range=init_range,
            seed=seed,
            verbose=verbose,
            nn_params=nn_params,
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


class PHATE(BaseEmbedding):
    """Potential of heat diffusion for affinity-based transition embedding.

    Built for continuous structure rather than clusters: it powers a diffusion
    operator to time `t`, takes the potential distance between the resulting
    distributions, and lays those out with MDS. Trajectories and branch points
    survive this that t-SNE and UMAP tear apart.

    `t` defaults to the von Neumann entropy knee, which is the right answer more
    often than a guess, and is worth pinning once you have looked at it.

    Args:
        n_components: Output dimensionality.
        k: Neighbours used to build the affinity graph. Smaller than UMAP's
            because the diffusion does the smoothing.
        metric: ``"euclidean"``/``"l2"``, ``"cosine"`` or ``"manhattan"``/``"l1"``.
        decay: Alpha-decay exponent of the kernel. ``None`` gives a binary
            connectivity kernel instead.
        t: Diffusion time. ``None`` picks the VNE knee.
        gamma: Informational distance constant, in ``[-1, 1]``. ``1`` is the log
            potential, ``0`` the square-root potential.
        mds: ``"sgd_dense"`` (also ``"dense"``) or ``"classic"``.
        mds_iter: MDS iterations. ``None`` uses the backend default.
        ann: Neighbour backend.
        randomised: Use randomised SVD for the initialisation.
        seed: Fixes the initialisation and the landmark sampling.
        verbose: ``0`` silent, ``1`` normal, ``2`` detailed.
        nn_params: See `NeighbourParams`.
        diffusion_params: Remaining operator knobs, landmarks included. See
            `PhateDiffusion`.
    """

    _FN: ClassVar[Callable[..., Any]] = _core.phate

    @beartype
    def __init__(
        self,
        n_components: int = 2,
        k: int = 5,
        metric: str = "euclidean",
        decay: float | None = 40.0,
        t: int | None = None,
        gamma: float = 1.0,
        mds: str = "sgd_dense",
        mds_iter: int | None = None,
        ann: str = "kmknn",
        randomised: bool = True,
        seed: int = 42,
        verbose: int = 0,
        nn_params: NeighbourParams | None = None,
        diffusion_params: PhateDiffusion | None = None,
    ) -> None:
        self.n_components = n_components
        self.k = k
        self.metric = metric
        self.decay = decay
        self.t = t
        self.gamma = gamma
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
        # `decay` is set rather than merged: `None` selects the binary
        # connectivity kernel, and `merge` drops nulls as "leave the default".
        diffusion["decay"] = self.decay
        if self.t is not None:
            diffusion["t"] = self.t
        if "graph_symmetry" in diffusion:
            diffusion["graph_symmetry"] = check_choice(
                diffusion["graph_symmetry"], SYMMETRIES, name="graph_symmetry"
            )
        if "landmark_method" in diffusion:
            diffusion["landmark_method"] = check_choice(
                diffusion["landmark_method"], LANDMARK_METHODS, name="landmark_method"
            )
        return {
            "n_dim": self.n_components,
            "k": self.k,
            "ann_type": check_choice(self.ann, ANN_CPU, name="ann"),
            "mds_method": check_choice(self.mds, MDS_METHODS, name="mds"),
            "mds_iter": self.mds_iter,
            "randomised": self.randomised,
            "nn_params": merge(
                self.nn_params,
                dist_metric=check_choice(self.metric, METRICS, name="metric"),
            ),
            "diffusion_params": diffusion,
        }


class PaCMAP(BaseEmbedding):
    """Pairwise-controlled manifold approximation and projection.

    Three kinds of pair rather than UMAP's one: near pairs pull, further pairs
    push, and mid-near pairs hold the global arrangement together while their
    weight decays over the first two phases. That decay is what lets PaCMAP keep
    global structure without the spectral initialisation UMAP leans on.

    PCA initialisation is the default and close to required: random init costs
    PaCMAP most of its global-structure advantage.

    Args:
        n_components: Output dimensionality.
        n_near: Near (attractive) pairs per point.
        n_mid_near: Mid-near pairs per point.
        n_further: Further (repulsive) pairs per point.
        mn_candidate_start: First kNN slot the mid-near sampler draws from.
        mn_candidate_end: Last such slot, and therefore the width of the kNN
            search. A precomputed graph must be at least this wide.
        metric: ``"euclidean"``/``"l2"``, ``"cosine"`` or ``"manhattan"``/``"l1"``.
        n_epochs: Optimisation epochs across all three phases.
        learning_rate: Adam learning rate.
        init: ``"pca"``, ``"spectral"`` or ``"random"``.
        ann: Neighbour backend.
        optimiser: ``"adam_parallel"`` or ``"adam"``.
        range_: Scale of the initial coordinates.
        seed: Fixes the initialisation and the pair sampling.
        verbose: ``0`` silent, ``1`` normal, ``2`` detailed.
        nn_params: See `NeighbourParams`.
        optim_params: Phase boundaries and Adam knobs. See `PacmapOptim`.
    """

    _FN: ClassVar[Callable[..., Any]] = _core.pacmap

    @beartype
    def __init__(
        self,
        n_components: int = 2,
        n_near: int = 10,
        n_mid_near: int = 5,
        n_further: int = 20,
        mn_candidate_start: int = 4,
        mn_candidate_end: int = 50,
        metric: str = "euclidean",
        n_epochs: int = 450,
        learning_rate: float = 0.01,
        init: str = "pca",
        ann: str = "kmknn",
        optimiser: str = "adam_parallel",
        range_: float | None = 0.01,
        seed: int = 42,
        verbose: int = 0,
        nn_params: NeighbourParams | None = None,
        optim_params: PacmapOptim | None = None,
    ) -> None:
        self.n_components = n_components
        self.n_near = n_near
        self.n_mid_near = n_mid_near
        self.n_further = n_further
        self.mn_candidate_start = mn_candidate_start
        self.mn_candidate_end = mn_candidate_end
        self.metric = metric
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.init = init
        self.ann = ann
        self.optimiser = optimiser
        self.range_ = range_
        self.seed = seed
        self.verbose = verbose
        self.nn_params = nn_params
        self.optim_params = optim_params

    def _params(self) -> dict[str, Any]:
        return {
            "n_dim": self.n_components,
            "n_near": self.n_near,
            "n_mid_near": self.n_mid_near,
            "n_further": self.n_further,
            "mn_candidate_start": self.mn_candidate_start,
            "mn_candidate_end": self.mn_candidate_end,
            "initialisation": check_choice(self.init, INITS, name="init"),
            "ann_type": check_choice(self.ann, ANN_CPU, name="ann"),
            "optimiser_type": check_choice(
                self.optimiser, PACMAP_OPTIMISERS, name="optimiser"
            ),
            "range": self.range_,
            "nn_params": merge(
                self.nn_params,
                dist_metric=check_choice(self.metric, METRICS, name="metric"),
            ),
            "optim_params": merge(
                self.optim_params, n_epochs=self.n_epochs, lr=self.learning_rate
            ),
        }


class DiffusionMaps(BaseEmbedding):
    """Diffusion maps.

    The spectral embedding of a diffusion operator, which is the thing PHATE
    builds on top of. `alpha` is the knob worth understanding: ``0`` gives the
    normalised graph Laplacian, ``0.5`` the Fokker-Planck operator, ``1`` the
    Laplace-Beltrami operator, which is the one that removes the influence of
    sampling density.

    Args:
        n_components: Output dimensionality, meaning eigenvectors kept.
        k: Neighbours used to build the kernel.
        metric: ``"euclidean"``/``"l2"``, ``"cosine"`` or ``"manhattan"``/``"l1"``.
        alpha: Anisotropic density-correction exponent in ``[0, 1]``.
        t: Diffusion time. ``None`` picks the VNE knee.
        bandwidth_scale: Multiplier on the adaptive kernel bandwidth.
        thresh: Kernel entries below this are zeroed.
        graph_symmetry: ``"add"``, ``"multiply"``, ``"mnn"`` or ``"none"``.
        n_landmarks: Landmarks to diffuse on instead of the full graph. ``None``
            or a value at least `n_samples` runs the full operator.
        landmark_method: ``"spectral"``, ``"random"`` or ``"density"``.
        n_svd: Components for spectral landmark selection.
        ann: Neighbour backend.
        seed: Fixes the landmark sampling.
        verbose: ``0`` silent, ``1`` normal, ``2`` detailed.
        nn_params: See `NeighbourParams`.
    """

    _FN: ClassVar[Callable[..., Any]] = _core.diffusion_maps

    @beartype
    def __init__(
        self,
        n_components: int = 2,
        k: int = 5,
        metric: str = "euclidean",
        alpha: float = 1.0,
        t: int | None = None,
        bandwidth_scale: float = 1.0,
        thresh: float = 1e-4,
        graph_symmetry: str = "add",
        n_landmarks: int | None = None,
        landmark_method: str = "spectral",
        n_svd: int | None = None,
        ann: str = "kmknn",
        seed: int = 42,
        verbose: int = 0,
        nn_params: NeighbourParams | None = None,
    ) -> None:
        self.n_components = n_components
        self.k = k
        self.metric = metric
        self.alpha = alpha
        self.t = t
        self.bandwidth_scale = bandwidth_scale
        self.thresh = thresh
        self.graph_symmetry = graph_symmetry
        self.n_landmarks = n_landmarks
        self.landmark_method = landmark_method
        self.n_svd = n_svd
        self.ann = ann
        self.seed = seed
        self.verbose = verbose
        self.nn_params = nn_params

    def _params(self) -> dict[str, Any]:
        return {
            "n_dim": self.n_components,
            "k": self.k,
            "alpha_norm": self.alpha,
            "t": self.t,
            "bandwidth_scale": self.bandwidth_scale,
            "thresh": self.thresh,
            "graph_symmetry": check_choice(
                self.graph_symmetry, SYMMETRIES, name="graph_symmetry"
            ),
            "n_landmarks": self.n_landmarks,
            "landmark_method": check_choice(
                self.landmark_method, LANDMARK_METHODS, name="landmark_method"
            ),
            "n_svd": self.n_svd,
            "ann_type": check_choice(self.ann, ANN_CPU, name="ann"),
            "nn_params": merge(
                self.nn_params,
                dist_metric=check_choice(self.metric, METRICS, name="metric"),
            ),
        }
