"""
glassbox.optimization.kfold
============================

**KFold Splitter** — K-Fold cross-validation index generator.

Splits a dataset into K non-overlapping folds.  On each iteration one fold
is held out as the validation set and the remaining K-1 folds form the
training set.  This is the workhorse primitive used by both
:class:`~glassbox.optimization.grid_search.GridSearch` and
:class:`~glassbox.optimization.random_search.RandomSearch`.

Design Goals
------------
- **NumPy-only**: index arithmetic is done with pure NumPy — no Scikit-Learn.
- **Deterministic**: pass ``random_state`` for reproducible splits.
- **Lazy**: yields index arrays one fold at a time; the full dataset is never
  duplicated in memory.

Public API
----------
KFoldSplitter
    Main splitter class.
cross_val_score
    Convenience function — runs K-fold CV for a single estimator/param combo.
FoldSplit
    Lightweight metadata dataclass returned by :pymeth:`KFoldSplitter.get_fold_info`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Iterator, List, Optional, Tuple, Type

import numpy as np


# ---------------------------------------------------------------------------
# Metadata container
# ---------------------------------------------------------------------------
@dataclass
class FoldSplit:
    """Metadata for a single cross-validation fold.

    Attributes
    ----------
    fold_index : int
        Zero-based fold number.
    n_train : int
        Number of training samples in this fold.
    n_val : int
        Number of validation samples in this fold.
    """

    fold_index: int
    n_train: int
    n_val: int

    def to_dict(self) -> dict:
        """Return a JSON-safe dictionary."""
        return {
            "fold_index": self.fold_index,
            "n_train": self.n_train,
            "n_val": self.n_val,
        }


# ---------------------------------------------------------------------------
# KFoldSplitter
# ---------------------------------------------------------------------------
class KFoldSplitter:
    """Split data into K non-overlapping folds for cross-validation.

    The dataset of *n* samples is divided into K folds as evenly as possible.
    When *n* is not divisible by K the first ``n % K`` folds receive one extra
    sample (standard round-robin distribution).

    Parameters
    ----------
    n_splits : int, optional
        Number of folds K.  Must be >= 2.  Default 5.
    shuffle : bool, optional
        If *True* the sample indices are shuffled before splitting, so each
        fold contains a random (but non-overlapping) subset.  Default *True*.
    random_state : int, optional
        Seed for the NumPy random generator.  Only used when
        ``shuffle=True``.  Pass an integer for reproducible splits.

    Examples
    --------
    >>> import numpy as np
    >>> from glassbox.optimization.kfold import KFoldSplitter
    >>> X = np.arange(10).reshape(10, 1)
    >>> splitter = KFoldSplitter(n_splits=5, shuffle=False)
    >>> for train_idx, val_idx in splitter.split(X):
    ...     print("train:", train_idx, "val:", val_idx)
    train: [2 3 4 5 6 7 8 9] val: [0 1]
    train: [0 1 4 5 6 7 8 9] val: [2 3]
    ...
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_indices, val_indices)`` for each of the K folds.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.  Only
            ``X.shape[0]`` is used — the actual values are not inspected.
        y : np.ndarray, optional
            Target vector.  Accepted for API consistency but unused — this
            splitter does **not** implement stratification.

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            ``(train_indices, val_indices)`` — integer index arrays that can
            be used directly for array slicing: ``X[train_idx]``.

        Raises
        ------
        ValueError
            If the number of samples is less than ``n_splits``.
        """
        n = X.shape[0]
        if n < self.n_splits:
            raise ValueError(
                f"Cannot form {self.n_splits} folds from only {n} samples. "
                f"Either reduce n_splits or provide more data."
            )

        indices = np.arange(n)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        # Distribute remainder across the first folds (round-robin).
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            val_idx = indices[current : current + fold_size]
            train_idx = np.concatenate(
                [indices[:current], indices[current + fold_size :]]
            )
            yield train_idx, val_idx
            current += fold_size

    def get_n_splits(self) -> int:
        """Return the configured number of splits."""
        return self.n_splits

    def get_fold_info(self, X: np.ndarray) -> List[FoldSplit]:
        """Return :class:`FoldSplit` metadata for every fold without yielding data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (only row count is used).

        Returns
        -------
        list[FoldSplit]
        """
        info = []
        for i, (train_idx, val_idx) in enumerate(self.split(X)):
            info.append(
                FoldSplit(
                    fold_index=i,
                    n_train=int(len(train_idx)),
                    n_val=int(len(val_idx)),
                )
            )
        return info


# ---------------------------------------------------------------------------
# Standalone cross_val_score helper
# ---------------------------------------------------------------------------
def cross_val_score(
    estimator_class: Type,
    X: np.ndarray,
    y: np.ndarray,
    scoring_fn: Callable,
    params: Optional[Dict[str, Any]] = None,
    cv: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Evaluate an estimator using K-fold cross-validation.

    A convenience wrapper around :class:`KFoldSplitter`.  For each fold it
    instantiates ``estimator_class(**params)``, calls ``fit(X_train, y_train)``
    and ``predict(X_val)``, then scores the predictions with ``scoring_fn``.

    Parameters
    ----------
    estimator_class : type
        A class with ``__init__(**params)``, ``fit(X, y)``, and
        ``predict(X)`` methods.
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    y : np.ndarray
        Target vector of shape ``(n_samples,)``.
    scoring_fn : callable
        ``scoring_fn(y_true, y_pred) -> float`` — higher values are better.
    params : dict, optional
        Keyword arguments passed to ``estimator_class.__init__``.
        Default ``{}`` (no extra arguments).
    cv : int, optional
        Number of folds.  Default 5.
    shuffle : bool, optional
        Shuffle before splitting.  Default *True*.
    random_state : int, optional
        Reproducibility seed.

    Returns
    -------
    np.ndarray of shape ``(cv,)``
        Score for each fold.  Failed folds produce ``nan``.

    Examples
    --------
    >>> import numpy as np
    >>> from glassbox.optimization.kfold import cross_val_score
    >>> from glassbox.optimization.scoring import accuracy_score
    >>> # Assume MyClassifier(max_depth=5) has fit() and predict().
    >>> scores = cross_val_score(
    ...     MyClassifier, X, y, accuracy_score,
    ...     params={"max_depth": 5}, cv=5, random_state=42
    ... )
    >>> print(f"CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    """
    if params is None:
        params = {}

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    splitter = KFoldSplitter(n_splits=cv, shuffle=shuffle, random_state=random_state)
    fold_scores: List[float] = []

    for train_idx, val_idx in splitter.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        try:
            model = estimator_class(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            fold_scores.append(float(scoring_fn(y_val, y_pred)))
        except Exception:
            fold_scores.append(float("nan"))

    return np.array(fold_scores, dtype=float)
