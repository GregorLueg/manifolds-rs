"""Shared estimator behaviour.

Every algorithm here embeds the data it was fitted on and nothing else. There is
no out-of-sample projection: UMAP's, t-SNE's and PHATE's all need machinery the
crate does not carry, so `transform` raises rather than quietly returning
something that is not a projection. That makes `fit_transform` the method you
actually want, and `fit` plus `embedding_` the scikit-learn shaped way to say
the same thing.

``get_params`` and ``set_params`` introspect the subclass ``__init__``, which is
all ``sklearn.base.BaseEstimator`` does. Doing it here keeps scikit-learn out of
the install requirements while ``clone`` and ``Pipeline`` still work by
duck-typing.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from beartype import beartype

from ._validate import check_knn, check_matrix


class NotFittedError(ValueError, AttributeError):
    """Raised when `embedding_` is read before `fit`.

    Inherits from both `ValueError` and `AttributeError` to match
    `sklearn.exceptions.NotFittedError`, so code catching either still works.
    """


class BaseEmbedding:
    """Common `fit` plumbing for every embedding.

    Subclasses supply an ``__init__`` that stores its arguments verbatim, a
    ``_params`` hook returning the payload the core reads, and the core function
    itself.
    """

    #: The `_manifolds` function this estimator drives.
    _FN: ClassVar[Callable[..., Any]]
    #: Element type to force on `fit`, or ``None`` to keep the caller's. The GPU
    #: estimators pin float32: WGSL has no float64.
    _FORCE_DTYPE: ClassVar[np.dtype | None] = None
    #: Extra keyword arguments passed straight through to the core, beyond the
    #: parameter payload. Only t-SNE uses it, for `approx`.
    _EXTRA: ClassVar[tuple[str, ...]] = ()

    # Every subclass takes these two in its `__init__`. Annotated but not
    # assigned, so they stay out of the class dict and `get_params` still reads
    # them off the instance.
    seed: int
    verbose: int

    # Unfitted state, as class attributes so subclasses need no
    # `super().__init__`.
    _embedding: np.ndarray | None = None
    n_features_in_: int = 0
    n_samples_fit_: int = 0

    ############
    # Subclass #
    ############

    def _params(self) -> dict[str, Any]:
        """The parameter payload for this algorithm."""
        raise NotImplementedError

    ##########
    # Params #
    ##########

    @classmethod
    def _param_names(cls) -> list[str]:
        """Constructor argument names, sorted."""
        sig = inspect.signature(cls.__init__)
        return sorted(p for p in sig.parameters if p != "self")

    @beartype
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Parameters this estimator was constructed with.

        Args:
            deep: Accepted for scikit-learn compatibility; these estimators hold
                no nested estimators, so it makes no difference. The parameter
                groups are frozen dataclasses, not estimators, and come back
                whole.

        Returns:
            Constructor parameters, keyed by name.
        """
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params: Any) -> BaseEmbedding:
        """Set constructor parameters, discarding any fitted embedding.

        Returns:
            self.

        Raises:
            ValueError: If a name is not a parameter of this estimator.
        """
        valid = set(self._param_names())
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    f"invalid parameter {key!r} for {type(self).__name__}; "
                    f"expected one of: {', '.join(sorted(valid))}"
                )
            setattr(self, key, value)
        self._embedding = None
        return self

    #######
    # Fit #
    #######

    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        knn_indices: Any = None,
        knn_distances: Any = None,
    ) -> BaseEmbedding:
        """Embed `X`.

        Args:
            X: Array-like of shape ``(n_samples, n_features)``. float32 and
                float64 are used as-is; other numeric types are promoted to
                float64. The element type picks the precision the whole
                pipeline runs in.
            y: Ignored, present for scikit-learn pipeline compatibility.
            knn_indices: Optional ``(n_samples, k)`` precomputed neighbour
                indices, excluding self. Skips the neighbour search, which on
                anything large is most of the runtime. See
                `manifolds_rs.knn_graph`.
            knn_distances: Distances matching `knn_indices`, as true distances
                in the same metric. `knn_graph` returns exactly that.

        Returns:
            self.

        Raises:
            ValueError: If only one of the two kNN arrays was given.
        """
        arr = check_matrix(X, dtype=self._FORCE_DTYPE)

        if (knn_indices is None) != (knn_distances is None):
            raise ValueError("knn_indices and knn_distances must be given together")
        knn: dict[str, Any] = {}
        if knn_indices is not None:
            ind, dist = check_knn(knn_indices, knn_distances, arr.shape[0], arr.dtype)
            knn = {"knn_indices": ind, "knn_distances": dist}

        extra = {name: getattr(self, name) for name in self._EXTRA}
        self._embedding = type(self)._FN(
            arr,
            self._params(),
            seed=self.seed,
            verbose=self.verbose,
            **extra,
            **knn,
        )
        self.n_samples_fit_, self.n_features_in_ = arr.shape
        return self

    def fit_transform(
        self,
        X: Any,
        y: Any = None,
        *,
        knn_indices: Any = None,
        knn_distances: Any = None,
    ) -> np.ndarray:
        """Embed `X` and return the result.

        Args:
            X: Array-like of shape ``(n_samples, n_features)``.
            y: Ignored, present for scikit-learn pipeline compatibility.
            knn_indices: Optional precomputed neighbour indices. See `fit`.
            knn_distances: Distances matching `knn_indices`.

        Returns:
            The embedding, ``(n_samples, n_components)``, in the same float type
            as the input.
        """
        self.fit(X, y, knn_indices=knn_indices, knn_distances=knn_distances)
        return self.embedding_

    def transform(self, X: Any) -> np.ndarray:
        """Not available: none of these algorithms projects new points.

        Raises:
            NotImplementedError: Always. Embedding new data means refitting on
                the whole set, which changes the existing coordinates too.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot project new points: the crate carries "
            f"no out-of-sample machinery for it. Refit on the combined data."
        )

    ##########
    # Output #
    ##########

    @property
    def embedding_(self) -> np.ndarray:
        """The fitted embedding, ``(n_samples, n_components)``.

        Raises:
            NotFittedError: If `fit` has not run.
        """
        if self._embedding is None:
            raise NotFittedError(
                f"{type(self).__name__} is not fitted; call fit(X) first"
            )
        return self._embedding

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in sorted(self.get_params().items()))
        return f"{type(self).__name__}({args})"
