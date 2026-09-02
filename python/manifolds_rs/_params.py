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

###########
# Globals #
###########

#: CPU neighbour backends, from `parse_ann_search`.
ANN_CPU: frozenset[str] = frozenset(
    {"exhaustive", "kmknn", "balltree", "annoy", "hnsw", "ivf", "nndescent"}
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
)

#: Distance metrics, from `ann_search_rs::utils::dist::parse_ann_dist`. Note
#: ``"euclidean"`` and ``"l2"`` are the same metric, as are ``"manhattan"`` and
#: ``"l1"``. Distances reaching a caller are true distances in every case.
METRICS: frozenset[str] = frozenset({"euclidean", "l2", "cosine", "manhattan", "l1"})

#: Embedding initialisations, from `parse_initilisation`.
INITS: frozenset[str] = frozenset({"spectral", "pca", "random"})

#: UMAP optimisers, from `parse_umap_optimiser`.
UMAP_OPTIMISERS: frozenset[str] = frozenset({"adam", "sgd", "adam_parallel"})

#: The GPU UMAP optimiser. `"adam_gpu"` keeps the update on the device; the CPU
#: names still work and pull the embedding back each epoch.
UMAP_OPTIMISERS_GPU: frozenset[str] = UMAP_OPTIMISERS | {"adam_gpu"}

#: PaCMAP optimisers, from `parse_pacmap_optimiser`.
PACMAP_OPTIMISERS: frozenset[str] = frozenset({"adam", "adam_parallel"})

#: MDS backends for PHATE, from `parse_mds_method`.
MDS_METHODS: frozenset[str] = frozenset({"sgd_dense", "dense", "classic"})

#: Landmark selection, from `parse_landmark_method`.
LANDMARK_METHODS: frozenset[str] = frozenset({"random", "spectral", "density"})

#: Graph symmetrisation, from `parse_phate_symmetrisation`.
SYMMETRIES: frozenset[str] = frozenset(
    {"additive", "add", "multiplicative", "mult", "multiply", "mnn", "none"}
)

#: t-SNE repulsion approximations, from `parse_tsne_optimiser`. ``"fft"`` needs
#: a build with the `fft_tsne` feature, which the published wheel is not: FFTW
#: is a system library no manylinux container carries. Barnes-Hut otherwise.
TSNE_APPROX: frozenset[str] = frozenset(
    {"barnes_hut", "barnes-hut", "barnes hut", "bh", "fft"}
)

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
        diversify_prob: NN-Descent. Diversification probability applied to the
            finished graph.
        delta: NN-Descent. Convergence threshold, as a fraction of neighbours
            updated in an iteration.
        ef_budget: NN-Descent. Beam budget when querying. ``None`` picks one.
            No effect when `extract_knn` is on, since no search runs.
        extract_knn: NN-Descent. Return the graph the descent already built
            instead of searching it. On by default: a self-kNN query
            re-searches a graph that is already a kNN graph. Measured on 20k
            points in 50D, identical recall and about 25% faster at ``k=15``;
            at ``k=50`` the graph is widened to cover the request, which is
            slower to build but reaches perfect recall where the beam search
            drops to 0.989.
        bt_budget: Ball tree. Fraction of the dataset to visit per query.
        n_list: IVF. Voronoi cells. ``None`` gives ``sqrt(n)``.
        n_probes: IVF. Cells visited per query. ``None`` gives ``sqrt(n_list)``.
    """

    n_tree: int | None = None
    search_budget: int | None = None
    m: int | None = None
    ef_construction: int | None = None
    ef_search: int | None = None
    diversify_prob: float | None = None
    delta: float | None = None
    ef_budget: int | None = None
    extract_knn: bool | None = None
    bt_budget: float | None = None
    n_list: int | None = None
    n_probes: int | None = None


@dataclass(frozen=True)
class NeighbourParamsGpu:
    """Backend-specific knobs for the GPU neighbour search.

    A different set from `NeighbourParams`, not a subset: the device backends
    build a CAGRA graph and search it with a beam, neither of which has a CPU
    counterpart.

    Attributes:
        n_list: IVF-GPU. Voronoi cells. ``None`` gives ``sqrt(n)``.
        n_probes: IVF-GPU. Cells visited per query.
        k: NN-Descent-GPU. Node degree after pruning. ``None`` gives 30, except
            under t-SNE where it is backfilled to ``3 * perplexity``.
        k_build: NN-Descent-GPU. Node degree before pruning.
        n_tree: NN-Descent-GPU. Trees used to seed the graph.
        delta: NN-Descent-GPU. Convergence threshold.
        rho: NN-Descent-GPU. Sampling rate per iteration.
        beam_width: NN-Descent-GPU. Beam width when querying.
        max_beam_iters: NN-Descent-GPU. Beam iterations when querying.
        n_entry_points: NN-Descent-GPU. Entry points per query.
        extract_knn: NN-Descent-GPU. Return the built CAGRA graph instead of
            searching it. **Off** by default, unlike the CPU backend: GPU
            extraction returns a different graph on two identical runs with a
            fixed seed, so turning it on costs you a reproducible embedding.
            The beam search is stable. `beam_width`, `max_beam_iters` and
            `n_entry_points` do nothing when it is on.
    """

    n_list: int | None = None
    n_probes: int | None = None
    k: int | None = None
    k_build: int | None = None
    n_tree: int | None = None
    delta: float | None = None
    rho: float | None = None
    beam_width: int | None = None
    max_beam_iters: int | None = None
    n_entry_points: int | None = None
    extract_knn: bool | None = None


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

    bandwidth: float | None = None
    local_connectivity: float | None = None
    mix_weight: float | None = None


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
            the usual 0.9.
        beta2: Adam second-moment decay.
        eps: Adam denominator epsilon.
    """

    a: float | None = None
    b: float | None = None
    gamma: float | None = None
    neg_sample_rate: int | None = None
    beta1: float | None = None
    beta2: float | None = None
    eps: float | None = None


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
    """

    early_exag_iter: int | None = None
    early_exag_factor: float | None = None
    late_exag_factor: float | None = None
    theta: float | None = None
    n_interp_points: int | None = None


@dataclass(frozen=True)
class PacmapOptim:
    """PaCMAP optimiser knobs beyond the epochs and learning rate.

    The three phases are what PaCMAP does instead of early exaggeration: the
    mid-near weight starts high, decays to zero across phase 2, and phase 3 is
    near pairs and repulsion alone.

    Attributes:
        beta1: Adam first-moment decay.
        beta2: Adam second-moment decay.
        eps: Adam denominator epsilon.
        phase1_end: Last epoch of the mid-near dominant phase.
        phase2_end: Last epoch of the decay phase.
    """

    beta1: float | None = None
    beta2: float | None = None
    eps: float | None = None
    phase1_end: int | None = None
    phase2_end: int | None = None


@dataclass(frozen=True)
class DensParams:
    """Density-preservation knobs beyond the weight.

    Attributes:
        frac: Fraction of the run, at the end, over which the density term is
            active. It is switched on late so the embedding has settled first.
        var_shift: Additive shift on the variance of the embedding log-radii,
            which keeps the correlation defined when the spread is tiny.
    """

    frac: float | None = None
    var_shift: float | None = None


@dataclass(frozen=True)
class PhateDiffusion:
    """PHATE diffusion operator knobs beyond decay, gamma and t.

    Attributes:
        bandwidth_scale: Multiplier on the adaptive kernel bandwidth.
        thresh: Affinities below this are zeroed, which is what keeps the
            operator sparse.
        graph_symmetry: ``"add"``, ``"multiply"``, ``"mnn"`` or ``"none"``.
        n_landmarks: Landmarks to diffuse on instead of the full graph. Worth
            setting above roughly 50k points.
        landmark_method: ``"spectral"``, ``"random"`` or ``"density"``.
        n_svd: Components for spectral landmark selection.
        t_max: Largest diffusion time the VNE knee search will consider. Ignored
            when `t` is pinned.
    """

    bandwidth_scale: float | None = None
    thresh: float | None = None
    graph_symmetry: str | None = None
    n_landmarks: int | None = None
    landmark_method: str | None = None
    n_svd: int | None = None
    t_max: int | None = None


#############
# Assembly  #
#############


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
