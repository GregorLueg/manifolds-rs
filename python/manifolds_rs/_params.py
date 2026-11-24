"""The escape-hatch parameter groups, and the allowlists for every string knob.

Each estimator takes the dozen or so parameters people actually turn as ordinary
constructor arguments. The rest, and there are a lot of them, live in the frozen
dataclasses here: pass one to the matching ``*_params`` argument and it overrides
the crate's defaults field by field.

**Every parameter has exactly one home.** A field that is a constructor argument
on the estimator is deliberately absent from the group, so there is never a
question of which of the two wins. That is why `NeighbourParams` has no
``dist_metric`` (it is `metric` on the estimator) and `UmapOptim` has no
``n_epochs``, ``lr`` or ``min_dist``.

A field left at ``None`` is not sent at all, and the Rust default applies. This
is the only reason ``None`` appears as a default here: it means "the crate
decides", not "off".
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from beartype import beartype

from . import _manifolds
from ._validate import check_choice

###########
# Globals #
###########

#: Whether the MLX names below are accepted: the extension was built with the
#: `mlx` feature and the core found mlx-c. Fixed at build time.
_MLX = _manifolds.mlx_available()

#: Whether ``"rinnd"`` is accepted: the extension was built with the `rinnd`
#: feature (on by default). Fixed at build time.
_RINND = _manifolds.rinnd_available()

#: The MLX neighbour backends, accepted by both the CPU and GPU estimators.
_MLX_ANN: frozenset[str] = frozenset({"exhaustive_mlx", "ivf_mlx", "nndescent_mlx"})

#: CPU neighbour backends, from `parse_ann_search`. The `*_mlx` ones run on the
#: Apple GPU through MLX and only exist in an `mlx` build. ``"rinnd"`` is added
#: by the bindings, not the core, and only exists in a `rinnd` build.
ANN_CPU: frozenset[str] = (
    frozenset({"exhaustive", "kmknn", "balltree", "annoy", "hnsw", "ivf", "nndescent"})
    | (_MLX_ANN if _MLX else frozenset())
    | (frozenset({"rinnd"}) if _RINND else frozenset())
)

#: GPU neighbour backends, from `parse_ann_search_gpu`. The unsuffixed aliases
#: are accepted by the core, so they are accepted here.
ANN_GPU: frozenset[str] = frozenset(
    {
        "exhaustive_gpu",
        "ivf_gpu",
        "nndescent_gpu",
        "exhaustive",
        "ivf",
        "nndescent",
    }
) | (_MLX_ANN if _MLX else frozenset())

#: Distance metrics, from `ann_search_rs::utils::dist::parse_ann_dist`. Note
#: ``"euclidean"`` and ``"l2"`` are the same metric, as are ``"manhattan"`` and
#: ``"l1"``. Distances reaching a caller are true distances in every case.
METRICS: frozenset[str] = frozenset({"euclidean", "l2", "cosine", "manhattan", "l1"})

#: As `METRICS`, for PHATE: the potential distances have no Manhattan path.
PHATE_METRICS: frozenset[str] = METRICS - {"manhattan", "l1"}

#: As `METRICS`, for the GPU estimators: the device kNN builders reject
#: Manhattan.
GPU_METRICS: frozenset[str] = METRICS - {"manhattan", "l1"}

#: k-means algorithms, from `parse_k_means`: the IVF and KmKNN centroids and the
#: spectral landmark selection of PHATE and diffusion maps.
K_MEANS: frozenset[str] = frozenset({"emeans", "e-means", "lloyd"})

#: Embedding initialisations, from `parse_initialisation`.
INITS: frozenset[str] = frozenset({"spectral", "pca", "random"})

#: As `INITS`, for UMAP and densMAP: ``"recursive"`` is the umap-learn 0.6
#: label propagation initialisation and needs the UMAP graph.
UMAP_INITS: frozenset[str] = INITS | {"recursive"}

#: NCVis initialisations: PCA, a standard normal draw or the original NCVis
#: start (uniform noise smoothed over the kNN graph), all scaled so the first
#: axis has the requested standard deviation.
NCVIS_INITS: frozenset[str] = frozenset({"pca", "random", "ncvis"})

#: GPU UMAP optimisers, from `parse_umap_optimiser_gpu`. `"adam_gpu"` and
#: `"adam_v06_gpu"` keep the update on the device; the CPU names still work and
#: pull the embedding back each epoch.
UMAP_OPTIMISERS_GPU: frozenset[str] = frozenset(
    {
        "adam",
        "sgd",
        "sgd_parallel",
        "adam_parallel",
        "adam_gpu",
        "adam_v06",
        "adam_v06_gpu",
    }
) | (frozenset({"adam_mlx", "adam_v06_mlx"}) if _MLX else frozenset())

#: UMAP optimisers, from `parse_umap_optimiser`.
UMAP_OPTIMISERS: frozenset[str] = frozenset(
    {"adam", "sgd", "sgd_parallel", "adam_parallel", "adam_v06"}
) | (frozenset({"adam_mlx", "adam_v06_mlx"}) if _MLX else frozenset())

#: The umap-learn 0.6 Adam optimisers. densMAP rejects them in the core.
V06_OPTIMISERS: frozenset[str] = frozenset({"adam_v06", "adam_v06_gpu", "adam_v06_mlx"})

#: The SGD optimisers, which drop to `SGD_EPOCHS_LARGE` epochs on large data.
SGD_OPTIMISERS: frozenset[str] = frozenset({"sgd", "sgd_parallel"})

#: Above this many samples `n_epochs=None` gives the SGD optimisers
#: `SGD_EPOCHS_LARGE` epochs instead of the crate's 500, as umap-learn does.
SGD_SMALL_N: int = 10_000

#: Epochs for the SGD optimisers above `SGD_SMALL_N` samples.
SGD_EPOCHS_LARGE: int = 200

#: PaCMAP and LocalMAP optimisers, from `parse_pacmap_optimiser`.
PACMAP_OPTIMISERS: frozenset[str] = frozenset({"adam", "adam_parallel"})

#: MDS backends for PHATE, from `parse_mds_method`.
MDS_METHODS: frozenset[str] = frozenset({"sgd_dense", "dense", "classic"})

#: Landmark selection, from `parse_landmark_method`.
LANDMARK_METHODS: frozenset[str] = frozenset({"random", "spectral", "density"})

#: As `LANDMARK_METHODS`, for PHATE: ``"spectral_mlx"`` runs the SVD and the
#: k-means of the spectral selection on MLX. Diffusion maps have no such path.
PHATE_LANDMARK_METHODS: frozenset[str] = LANDMARK_METHODS | (
    frozenset({"spectral_mlx"}) if _MLX else frozenset()
)

#: Graph symmetrisation, from `parse_phate_symmetrisation`.
SYMMETRIES: frozenset[str] = frozenset(
    {"additive", "add", "multiplicative", "mult", "multiply", "mnn", "none"}
)

#: t-SNE repulsion approximations, from `parse_tsne_optimiser`. The MLX
#: three-kernel FFT is accepted by both entry points.
TSNE_APPROX: frozenset[str] = frozenset(
    {
        "barnes_hut",
        "barnes-hut",
        "barnes hut",
        "bh",
        "bh_qd",
        "qd",
        "fft",
        "fft_3k",
        "3-kernel",
    }
) | (frozenset({"fft_3k_mlx"}) if _MLX else frozenset())

#: As `TSNE_APPROX`, plus the device-resident three-kernel FFT optimiser. Only
#: `tsne_gpu` accepts it.
TSNE_APPROX_GPU: frozenset[str] = TSNE_APPROX | {"fft_3k_gpu"}

#: ForceAtlas2 repulsion approximations, from `parse_fa2_optimiser`.
FA2_APPROX: frozenset[str] = frozenset({"barnes_hut", "barnes-hut", "barnes hut", "bh"})

#: Trajectory topologies, from `parse_topology`.
TOPOLOGIES: frozenset[str] = frozenset({"bifurcation", "linear", "combination"})


####################
# Parameter groups #
####################


@dataclass(frozen=True)
class NeighbourParams:
    """Backend-specific knobs for the CPU neighbour search.

    Only the fields belonging to the backend you chose have any effect; the rest
    are ignored by the core. `metric` is not here because it is a constructor
    argument on every estimator.

    Attributes:
        n_tree: Annoy. Trees in the forest. More means better recall and a
            slower build.
        search_budget: Annoy. Candidates inspected per query. ``None`` gives
            ``k * n_tree * 20``.
        m: HNSW. Edges per node on the upper layers, ``2 * m`` on layer 0.
        ef_construction: HNSW. Candidate list width during the build.
        ef_search: HNSW. Beam width at query time, the recall knob.
        delta: NN-Descent and rinnd. Convergence threshold, as a fraction of
            neighbours updated in an iteration.
        nnd_n_trees: NN-Descent and rinnd. Random projection trees seeding the
            graph. ``None`` gives ``5 + sqrt(n) / 20``, capped at 64, for
            ``"nndescent"`` (the umap-learn / scanpy rule) and
            ``clamp(round(2 * log10(n)), 3, 12)`` for ``"rinnd"``.
        bt_budget: Ball tree. Fraction of the dataset to visit per query.
        n_list: IVF. Voronoi cells. ``None`` gives ``sqrt(n)``.
        n_probes: IVF. Cells visited per query. ``None`` gives ``sqrt(n_list)``.
        k_means: IVF and KmKNN. k-means training the centroids, ``"emeans"``
            or ``"lloyd"``. ``None`` gives e-means.
    """

    n_tree: int | None = None
    search_budget: int | None = None
    m: int | None = None
    ef_construction: int | None = None
    ef_search: int | None = None
    delta: int | float | None = None
    nnd_n_trees: int | None = None
    bt_budget: int | float | None = None
    n_list: int | None = None
    n_probes: int | None = None
    k_means: str | None = None

    @beartype
    def __post_init__(self) -> None:
        """Reject an unknown `k_means` at construction, see `check_choice`."""
        if self.k_means is not None:
            check_choice(self.k_means, K_MEANS, name="k_means")


@dataclass(frozen=True)
class NeighbourParamsGpu:
    """Backend-specific knobs for the GPU neighbour search.

    A different set from `NeighbourParams`, not a subset: the device NN-Descent
    has its own working degree and sampling knobs.

    Attributes:
        n_list: IVF-GPU. Voronoi cells. ``None`` gives ``sqrt(n)``.
        n_probes: IVF-GPU. Cells visited per query.
        k_means: IVF-GPU. k-means training the centroids, ``"emeans"`` or
            ``"lloyd"``. ``None`` gives e-means.
        k_build: NN-Descent-GPU. Working degree of the descent. ``None`` gives
            ``1.5 * k``.
        n_tree: NN-Descent-GPU. Trees used to seed the graph.
        delta: NN-Descent-GPU. Convergence threshold.
        rho: NN-Descent-GPU. Sampling rate per iteration.
    """

    n_list: int | None = None
    n_probes: int | None = None
    k_means: str | None = None
    k_build: int | None = None
    n_tree: int | None = None
    delta: int | float | None = None
    rho: int | float | None = None

    @beartype
    def __post_init__(self) -> None:
        """Reject an unknown `k_means` at construction, see `check_choice`."""
        if self.k_means is not None:
            check_choice(self.k_means, K_MEANS, name="k_means")


@dataclass(frozen=True)
class UmapGraph:
    """Fuzzy simplicial set construction.

    Attributes:
        bandwidth: Convergence tolerance for the smooth-kNN binary search that
            finds each point's sigma.
        local_connectivity: Neighbours assumed to sit at distance zero. Raising
            it makes the local neighbourhood denser and the embedding tighter.
        mix_weight: Balance between the fuzzy union and the directed graph
            during symmetrisation. ``1.0`` is the plain union.
    """

    bandwidth: int | float | None = None
    local_connectivity: int | float | None = None
    mix_weight: int | float | None = None


@dataclass(frozen=True)
class UmapOptim:
    """UMAP optimiser knobs beyond the epochs and learning rate.

    Attributes:
        a: Repulsion curve numerator. Fitted from ``min_dist`` and ``spread``
            unless you set it, and setting one of `a` / `b` without the other
            is rarely what you want.
        b: Repulsion curve exponent. See `a`.
        gamma: Weight on the repulsive term.
        neg_sample_rate: Negative samples drawn per positive edge.
        beta1: Adam first-moment decay. The crate uses 0.5 for UMAP rather than
            the usual 0.9. Ignored by ``"adam_v06"`` / ``"adam_v06_gpu"``, which
            keep umap-learn's constants.
        beta2: Adam second-moment decay. Ignored by the 0.6 Adam, as `beta1`.
        eps: Adam denominator epsilon. Ignored by the 0.6 Adam, as `beta1`.
    """

    a: int | float | None = None
    b: int | float | None = None
    gamma: int | float | None = None
    neg_sample_rate: int | None = None
    beta1: int | float | None = None
    beta2: int | float | None = None
    eps: int | float | None = None


@dataclass(frozen=True)
class TsneOptim:
    """t-SNE optimiser knobs beyond the epochs and learning rate.

    Attributes:
        early_exag_iter: Iterations of early exaggeration.
        early_exag_factor: Multiplier on the affinities during those iterations.
        late_exag_factor: Multiplier for the remaining iterations. ``None``
            disables it. Above roughly 100k points a value near 4 keeps cluster
            structure from dispersing once early exaggeration ends.
        theta: Barnes-Hut opening angle. Larger is faster and coarser; ``0``
            makes it exact and very slow.
        n_interp_points: Interpolation points per box on the FFT path. No effect
            under Barnes-Hut.
        max_depth: Tree depth cap for ``approx="qd"``. Smaller is faster and
            coarser; qdtsne recommends 7 to 10. No effect on the other
            optimisers.
    """

    early_exag_iter: int | None = None
    early_exag_factor: int | float | None = None
    late_exag_factor: int | float | None = None
    theta: int | float | None = None
    n_interp_points: int | None = None
    max_depth: int | None = None


@dataclass(frozen=True)
class PacmapOptim:
    """PaCMAP and LocalMAP optimiser knobs beyond the epochs and learning rate.

    The three phases are what PaCMAP does instead of early exaggeration: the
    mid-near weight starts high, decays across phase 1, holds low through phase 2,
    and phase 3 is near pairs and repulsion alone.

    Attributes:
        beta1: Adam first-moment decay.
        beta2: Adam second-moment decay.
        eps: Adam denominator epsilon.
        phase1_end: Last epoch of the mid-near dominant phase.
        phase2_end: Last epoch of the decay phase.
    """

    beta1: int | float | None = None
    beta2: int | float | None = None
    eps: int | float | None = None
    phase1_end: int | None = None
    phase2_end: int | None = None


@dataclass(frozen=True)
class Fa2Optim:
    """ForceAtlas2 knobs beyond the epochs, scaling ratio, gravity and LinLog.

    Defaults follow Gephi.

    Attributes:
        strong_gravity: Distance-independent gravity, scaled by the scaling
            ratio. Pulls the periphery in much harder than the linear default.
        dissuade_hubs: Divide attraction by node mass, pushing hubs to the
            periphery.
        edge_weight_influence: Exponent applied to the edge weights. ``0``
            ignores the weights.
        jitter_tolerance: Tolerated swinging. Larger is faster and less
            precise.
        theta: Barnes-Hut opening parameter on Gephi's scale. Larger is faster
            and coarser; ``0`` makes the repulsion exact and very slow.
    """

    strong_gravity: bool | None = None
    dissuade_hubs: bool | None = None
    edge_weight_influence: int | float | None = None
    jitter_tolerance: int | float | None = None
    theta: int | float | None = None


@dataclass(frozen=True)
class NcvisOptim:
    """NCVis optimiser knobs beyond the epochs, negatives and learning rate.

    Attributes:
        a: Kernel numerator of ``q(d) = 1 / (1 + a d^(2b))``. Fitted from
            ``min_dist`` and ``spread`` unless you set it. ``a = b = 1`` is the
            Cauchy kernel, which skips a ``powf`` per sample.
        b: Kernel exponent. See `a`.
        lr_q: Step size of the learned log normalisation constant.
    """

    a: int | float | None = None
    b: int | float | None = None
    lr_q: int | float | None = None


@dataclass(frozen=True)
class DensParams:
    """Density-preservation knobs beyond the weight.

    Attributes:
        frac: Fraction of the run, at the end, over which the density term is
            active. It is switched on late so the embedding has settled first.
        var_shift: Additive shift on the variance of the embedding log-radii,
            which keeps the correlation defined when the spread is tiny.
    """

    frac: int | float | None = None
    var_shift: int | float | None = None


@dataclass(frozen=True)
class PhateDiffusion:
    """PHATE diffusion operator knobs beyond decay, gamma, t and the landmarks.

    Attributes:
        bandwidth_scale: Multiplier on the adaptive kernel bandwidth.
        thresh: Affinities below this are zeroed, which is what keeps the
            operator sparse.
        graph_symmetry: ``"add"``, ``"multiply"``, ``"mnn"`` or ``"none"``.
        n_svd: Singular vectors for spectral landmark selection.
        landmark_k_means: k-means of the spectral landmark selection (not the
            IVF one in `NeighbourParams`), ``"emeans"`` or ``"lloyd"``.
            ``None`` gives e-means.
        t_max: Largest diffusion time the VNE knee search will consider. Ignored
            when `t` is pinned.
    """

    bandwidth_scale: int | float | None = None
    thresh: int | float | None = None
    graph_symmetry: str | None = None
    n_svd: int | None = None
    landmark_k_means: str | None = None
    t_max: int | None = None

    @beartype
    def __post_init__(self) -> None:
        """Reject an unknown `landmark_k_means` at construction, see `check_choice`."""
        if self.landmark_k_means is not None:
            check_choice(self.landmark_k_means, K_MEANS, name="landmark_k_means")


#############
# Assembly  #
#############


@beartype
def group_to_dict(group: Any | None) -> dict[str, Any]:
    """Flatten a parameter group into the keys the core reads.

    Args:
        group: One of the dataclasses above, or ``None``.

    Returns:
        The fields that were set, keyed by name. Fields left at ``None`` are
        dropped rather than sent, so the crate's default applies to each of them
        independently.
    """
    if group is None:
        return {}
    return {
        f.name: value
        for f in fields(group)
        if (value := getattr(group, f.name)) is not None
    }


@beartype
def umap_epochs(n_epochs: int | None, optimiser: str, n_samples: int) -> int | None:
    """Resolve the UMAP epoch count.

    Args:
        n_epochs: What the caller passed. Anything but ``None`` wins.
        optimiser: The optimiser name.
        n_samples: Rows of the data being embedded.

    Returns:
        `SGD_EPOCHS_LARGE` for an SGD optimiser above `SGD_SMALL_N` samples,
        otherwise `n_epochs`; ``None`` leaves the crate default of 500.
    """
    if n_epochs is None and optimiser in SGD_OPTIMISERS and n_samples > SGD_SMALL_N:
        return SGD_EPOCHS_LARGE
    return n_epochs


@beartype
def merge(group: Any | None, **overrides: Any) -> dict[str, Any] | None:
    """Flatten a group and fold in the estimator's own arguments.

    Args:
        group: The parameter group, or ``None``.
        **overrides: Constructor arguments that belong in this group, for
            example ``n_epochs`` for `UmapOptim`. A ``None`` is dropped, since
            it means the caller asked for the crate default.

    Returns:
        The merged dictionary, or ``None`` when nothing was set at all, which
        keeps the key out of the payload entirely.
    """
    merged = group_to_dict(group)
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged or None
