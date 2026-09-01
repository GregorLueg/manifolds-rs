"""Shape of the compiled extension.

Hand-written, because maturin does not generate stubs. The signatures here are
what `src/*.rs` declares; the docstrings live on the Rust side and reach Python
at runtime.

`umap_gpu`, `densmap_gpu` and `tsne_gpu` exist only in a build with the `gpu`
feature. They are declared unconditionally, since a stub cannot be conditional
and `manifolds_rs.gpu` already raises a clear `ImportError` when they are
missing.
"""

from typing import Any

import numpy as np

__version__: str
__core_version__: str

class ManifoldsRsError(Exception): ...
class ConvergenceError(ManifoldsRsError): ...

def set_num_threads(n: int) -> None: ...
def num_threads() -> int: ...
def gpu_available() -> bool: ...
def knn_graph(
    x: np.ndarray,
    k: int,
    *,
    ann: str = ...,
    nn_params: dict[str, Any] | None = ...,
    seed: int = ...,
    verbose: int = ...,
) -> tuple[np.ndarray, np.ndarray]: ...
def swiss_roll(
    n_samples: int = ...,
    *,
    noise: float = ...,
    density_bias: float = ...,
    seed: int = ...,
) -> tuple[np.ndarray, np.ndarray]: ...
def clustered(
    n_samples: int = ..., *, dim: int = ..., n_clusters: int = ..., seed: int = ...
) -> tuple[np.ndarray, np.ndarray]: ...
def trajectory(
    n_samples: int = ...,
    *,
    topology: str = ...,
    dim: int = ...,
    noise: float = ...,
    seed: int = ...,
) -> tuple[np.ndarray, np.ndarray]: ...
def hierarchical(
    n_samples: int = ...,
    *,
    dim: int = ...,
    n_supergroups: int = ...,
    n_subclusters: int = ...,
    supergroup_spread: float = ...,
    subcluster_spread: float = ...,
    point_std: float = ...,
    seed: int = ...,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def umap(
    x: np.ndarray,
    params: dict[str, Any],
    *,
    knn_indices: np.ndarray | None = ...,
    knn_distances: np.ndarray | None = ...,
    seed: int = ...,
    verbose: int = ...,
) -> np.ndarray: ...
def densmap(
    x: np.ndarray,
    params: dict[str, Any],
    *,
    knn_indices: np.ndarray | None = ...,
    knn_distances: np.ndarray | None = ...,
    seed: int = ...,
    verbose: int = ...,
) -> np.ndarray: ...
def tsne(
    x: np.ndarray,
    params: dict[str, Any],
    *,
    approx: str = ...,
    knn_indices: np.ndarray | None = ...,
    knn_distances: np.ndarray | None = ...,
    seed: int = ...,
    verbose: int = ...,
) -> np.ndarray: ...
def densne(
    x: np.ndarray,
    params: dict[str, Any],
    *,
    approx: str = ...,
    knn_indices: np.ndarray | None = ...,
    knn_distances: np.ndarray | None = ...,
    seed: int = ...,
    verbose: int = ...,
) -> np.ndarray: ...
def phate(
    x: np.ndarray,
    params: dict[str, Any],
    *,
    knn_indices: np.ndarray | None = ...,
    knn_distances: np.ndarray | None = ...,
    seed: int = ...,
    verbose: int = ...,
) -> np.ndarray: ...
def pacmap(
    x: np.ndarray,
    params: dict[str, Any],
    *,
    knn_indices: np.ndarray | None = ...,
    knn_distances: np.ndarray | None = ...,
    seed: int = ...,
    verbose: int = ...,
) -> np.ndarray: ...
def diffusion_maps(
    x: np.ndarray,
    params: dict[str, Any],
    *,
    knn_indices: np.ndarray | None = ...,
    knn_distances: np.ndarray | None = ...,
    seed: int = ...,
    verbose: int = ...,
) -> np.ndarray: ...
def umap_gpu(
    x: np.ndarray,
    params: dict[str, Any],
    *,
    knn_indices: np.ndarray | None = ...,
    knn_distances: np.ndarray | None = ...,
    seed: int = ...,
    verbose: int = ...,
) -> np.ndarray: ...
def densmap_gpu(
    x: np.ndarray,
    params: dict[str, Any],
    *,
    knn_indices: np.ndarray | None = ...,
    knn_distances: np.ndarray | None = ...,
    seed: int = ...,
    verbose: int = ...,
) -> np.ndarray: ...
def tsne_gpu(
    x: np.ndarray,
    params: dict[str, Any],
    *,
    approx: str = ...,
    knn_indices: np.ndarray | None = ...,
    knn_distances: np.ndarray | None = ...,
    seed: int = ...,
    verbose: int = ...,
) -> np.ndarray: ...
