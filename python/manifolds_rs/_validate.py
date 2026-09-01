"""Array and string checks at the FFI boundary."""

from typing import Any

import numpy as np
from beartype import beartype

###########
# Globals #
###########

#: The two element types the Rust core is compiled for.
_NATIVE_DTYPES = (np.float32, np.float64)


@beartype
def check_matrix(
    x: Any, *, name: str = "X", dtype: np.dtype | None = None
) -> np.ndarray:
    """Coerce an array-like into something the Rust core can borrow directly.

    float32 and float64 pass through untouched. Any other numeric type is
    promoted to float64 rather than narrowed to float32, so precision is never
    silently lost. The result is always C-contiguous, because the core borrows
    the buffer rather than copying it.

    Args:
        x: Array-like of shape ``(n_samples, n_features)``.
        name: Argument name, used in error messages.
        dtype: Element type to force, overriding the promotion rule above. The
            GPU estimators set this to float32, since WGSL has no float64 and
            the alternative is a failure deep inside a kernel.

    Returns:
        A C-contiguous 2-D array, of `dtype` when one was given.

    Raises:
        ValueError: If the input is not 2-D, is empty, or holds non-finite
            values. The core does not check finiteness and would spend a
            500-epoch optimisation producing NaN.
        TypeError: If the input does not hold numbers.
    """
    arr = np.asarray(x)

    if arr.dtype.kind not in "fiub":
        raise TypeError(f"{name} must hold numbers, got dtype {arr.dtype}")
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    elif arr.dtype.type not in _NATIVE_DTYPES:
        arr = arr.astype(np.float64)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (samples x features), got {arr.ndim}-D")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")

    return np.ascontiguousarray(arr)


@beartype
def check_knn(
    indices: Any, distances: Any, n_samples: int, dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce a precomputed neighbour graph to match the design matrix.

    Args:
        indices: Array-like of shape ``(n_samples, k)``, neighbour indices
            excluding self.
        distances: Array-like of the same shape, aligned with `indices`.
        n_samples: Rows the design matrix has.
        dtype: Element type the embedding will run in.

    Returns:
        ``(indices, distances)``, C-contiguous, int64 and `dtype`.

    Raises:
        ValueError: If the shapes disagree, do not match `n_samples`, or the
            indices hold padding. A row an approximate search could not fill is
            padded with ``-1``, and a padded graph is not one an embedding can
            be built from: drop those rows or search again with a smaller `k`.
    """
    ind = np.ascontiguousarray(np.asarray(indices), dtype=np.int64)
    dist = np.ascontiguousarray(np.asarray(distances), dtype=dtype)

    if ind.ndim != 2 or dist.ndim != 2:
        raise ValueError("knn_indices and knn_distances must both be 2-D")
    if ind.shape != dist.shape:
        raise ValueError(
            f"knn_indices has shape {ind.shape} but knn_distances has {dist.shape}"
        )
    if ind.shape[0] != n_samples:
        raise ValueError(
            f"knn arrays have {ind.shape[0]} rows but X has {n_samples} samples"
        )
    if (ind < 0).any():
        raise ValueError(
            "knn_indices contains -1 padding; an embedding graph needs every "
            "row filled. Search again with a smaller k."
        )
    return ind, dist


@beartype
def check_choice(value: str, allowed: frozenset[str], *, name: str) -> str:
    """Reject a string the core would silently fall back on.

    Every string parameter in `manifolds-rs` goes through a parser that returns
    a default and prints to the process stdout when it does not recognise the
    input. From a notebook that message goes to the terminal running the kernel,
    so a typo looks exactly like a working run with different results. Hence the
    allowlist here.

    Args:
        value: The string the caller passed.
        allowed: Names the core's parser accepts.
        name: Parameter name, used in the error message.

    Returns:
        `value`, lowercased, since every parser in the core lowercases first.

    Raises:
        ValueError: If `value` is not in `allowed`.
    """
    lowered = value.lower()
    if lowered not in allowed:
        raise ValueError(
            f"unknown {name} {value!r}; expected one of: {', '.join(sorted(allowed))}"
        )
    return lowered
