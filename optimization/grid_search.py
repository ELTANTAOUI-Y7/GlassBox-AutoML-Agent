"""
glassbox.optimization.grid_search
===================================

**Grid Search** — systematic exhaustive search over a hyperparameter grid.

For every combination in the Cartesian product of ``param_grid`` values,
:func:`_cross_validate` runs K-fold cross-validation using
:class:`~glassbox.optimization.kfold.KFoldSplitter`.  The combination with
the highest mean CV score is returned as the best.

Design Goals
------------
- **NumPy-only math**: index arithmetic and statistics use pure NumPy.
- **Transparent**: every candidate combination and all fold scores are stored
  in the result for full auditability.
- **Fail-safe**: if an estimator raises an exception during a fold, that fold
  receives a ``nan`` score and an error message is recorded rather than
  crashing the entire search.
- **Composable**: :func:`_cross_validate` is a shared primitive imported by
  :mod:`glassbox.optimization.random_search`.

Public API
----------
GridSearch
    Main search class.
GridSearchResult
    JSON-serialisable result container.
CVResult
    Per-candidate cross-validation summary.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

from glassbox.optimization.kfold import KFoldSplitter


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_json(val: Any) -> Any:
    """Convert NumPy scalar types to plain Python for JSON serialisation."""
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return round(float(val), 6)
    if isinstance(val, np.bool_):
        return bool(val)
    return val


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CVResult:
    """Cross-validation result for a single hyperparameter combination.

    Attributes
    ----------
    params : dict
        The hyperparameter combination that was evaluated.
    fold_scores : list[float]
        Score produced by each fold.  Failed folds contain ``nan``.
    mean_score : float
        Mean over all fold scores (``nan``-safe).
    std_score : float
        Standard deviation of fold scores (``nan``-safe).
    elapsed_seconds : float
        Wall-clock time to run all K folds for this combination.
    error : str, optional
        Exception message if *any* fold raised an error; ``None`` otherwise.
    """

    params: Dict[str, Any]
    fold_scores: List[float]
    mean_score: float
    std_score: float
    elapsed_seconds: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Return a JSON-safe dictionary."""
        return {
            "params": {k: _safe_json(v) for k, v in self.params.items()},
            "fold_scores": [round(s, 6) if not np.isnan(s) else None
                            for s in self.fold_scores],
            "mean_score": round(self.mean_score, 6)
                          if not np.isnan(self.mean_score) else None,
            "std_score": round(self.std_score, 6)
                         if not np.isnan(self.std_score) else None,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "error": self.error,
        }


@dataclass
class GridSearchResult:
    """Full result of a :class:`GridSearch` run.

    Attributes
    ----------
    best_params : dict
        Hyperparameters of the combination with the highest mean CV score.
    best_score : float
        Mean CV score of the best combination.
    n_candidates : int
        Total number of hyperparameter combinations evaluated.
    n_splits : int
        Number of cross-validation folds used.
    elapsed_seconds : float
        Total wall-clock time for the entire search.
    scoring_fn_name : str
        ``__name__`` of the scoring function (for the report).
    all_results : list[CVResult]
        Full CV summary for every evaluated combination — sorted by
        ``mean_score`` descending.
    """

    best_params: Dict[str, Any]
    best_score: float
    n_candidates: int
    n_splits: int
    elapsed_seconds: float
    scoring_fn_name: str
    all_results: List[CVResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return the full result as a JSON-safe dictionary."""
        return {
            "best_params": {k: _safe_json(v) for k, v in self.best_params.items()},
            "best_score": round(self.best_score, 6)
                          if not np.isnan(self.best_score) else None,
            "n_candidates": self.n_candidates,
            "n_splits": self.n_splits,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "scoring_fn": self.scoring_fn_name,
            "all_results": [r.to_dict() for r in self.all_results],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the full result to a JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Shared cross-validation primitive
# ---------------------------------------------------------------------------

def _cross_validate(
    estimator_class: Type,
    params: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    scoring_fn: Callable,
    splitter: KFoldSplitter,
) -> CVResult:
    """Run K-fold cross-validation for one hyperparameter combination.

    This function is the inner loop shared by both GridSearch and RandomSearch.

    Parameters
    ----------
    estimator_class : type
        Class with ``fit(X, y)`` and ``predict(X)`` methods.
    params : dict
        Keyword arguments forwarded to ``estimator_class.__init__``.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    scoring_fn : callable
        ``scoring_fn(y_true, y_pred) -> float``.
    splitter : KFoldSplitter
        Pre-configured splitter instance (reused across candidates).

    Returns
    -------
    CVResult
    """
    fold_scores: List[float] = []
    first_error: Optional[str] = None
    t0 = time.perf_counter()

    for train_idx, val_idx in splitter.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        try:
            model = estimator_class(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = float(scoring_fn(y_val, y_pred))
        except Exception as exc:
            score = float("nan")
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"
        fold_scores.append(score)

    scores_arr = np.array(fold_scores, dtype=float)
    return CVResult(
        params=params,
        fold_scores=fold_scores,
        mean_score=float(np.nanmean(scores_arr)),
        std_score=float(np.nanstd(scores_arr)),
        elapsed_seconds=time.perf_counter() - t0,
        error=first_error,
    )


# ---------------------------------------------------------------------------
# GridSearch
# ---------------------------------------------------------------------------

class GridSearch:
    """Exhaustive search over a grid of hyperparameter values.

    For each combination in the Cartesian product of ``param_grid`` values,
    K-fold cross-validation is run.  The combination with the highest mean
    CV score is stored as the best.

    All combinations and their per-fold scores are preserved in the
    :class:`GridSearchResult`, making the search fully auditable.

    Parameters
    ----------
    scoring_fn : callable
        ``scoring_fn(y_true, y_pred) -> float``.  **Higher values are
        always considered better.**  For error metrics such as MSE use
        :func:`~glassbox.optimization.scoring.neg_mean_squared_error`.
    cv : int, optional
        Number of cross-validation folds.  Default 5.
    shuffle : bool, optional
        Shuffle samples before forming folds.  Default *True*.
    random_state : int, optional
        Reproducibility seed for the fold shuffler.
    verbose : bool, optional
        Print a one-line progress update for each candidate.  Default *False*.

    Examples
    --------
    >>> import numpy as np
    >>> from glassbox.optimization.grid_search import GridSearch
    >>> from glassbox.optimization.scoring import accuracy_score
    >>> # Assume MyClassifier has fit(X, y) and predict(X).
    >>> gs = GridSearch(scoring_fn=accuracy_score, cv=5, random_state=42)
    >>> result = gs.fit(
    ...     X, y, MyClassifier,
    ...     param_grid={"max_depth": [3, 5, 10], "min_samples": [2, 5]},
    ... )
    >>> print(result.best_params, result.best_score)
    """

    def __init__(
        self,
        scoring_fn: Callable,
        cv: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        if cv < 2:
            raise ValueError(f"cv must be >= 2, got {cv}.")
        self.scoring_fn = scoring_fn
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self._result: Optional[GridSearchResult] = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator_class: Type,
        param_grid: Dict[str, List[Any]],
    ) -> GridSearchResult:
        """Run the grid search.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.
        y : np.ndarray
            Target vector of shape ``(n_samples,)``.
        estimator_class : type
            A class whose ``__init__`` accepts the hyperparameter names in
            ``param_grid`` as keyword arguments and implements
            ``fit(X, y)`` and ``predict(X)``.
        param_grid : dict[str, list]
            Maps each hyperparameter name to the list of values to try.
            All combinations are evaluated (Cartesian product).

        Returns
        -------
        GridSearchResult
            Contains the best params, best score, and the full per-candidate
            audit trail.

        Raises
        ------
        ValueError
            If ``param_grid`` is empty or any value list is empty.
        """
        if not param_grid:
            raise ValueError("param_grid must not be empty.")
        for key, vals in param_grid.items():
            if not vals:
                raise ValueError(
                    f"param_grid['{key}'] is an empty list — "
                    f"provide at least one value."
                )

        t0 = time.perf_counter()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        splitter = KFoldSplitter(
            n_splits=self.cv,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        # Build all (param_name → value) combinations.
        param_names = list(param_grid.keys())
        all_combos: List[Tuple] = list(
            itertools.product(*[param_grid[k] for k in param_names])
        )
        n_candidates = len(all_combos)

        if self.verbose:
            total_fits = n_candidates * self.cv
            print(
                f"[GridSearch] {n_candidates} candidates × {self.cv} folds "
                f"= {total_fits} total fits"
            )

        all_results: List[CVResult] = []
        for i, combo in enumerate(all_combos):
            params = dict(zip(param_names, combo))
            if self.verbose:
                print(f"  Candidate [{i + 1}/{n_candidates}]: {params}")

            cv_result = _cross_validate(
                estimator_class, params, X, y, self.scoring_fn, splitter
            )
            all_results.append(cv_result)

        # Select the best — ignore all-nan candidates when possible.
        valid = [r for r in all_results if not np.isnan(r.mean_score)]
        ranking_pool = valid if valid else all_results
        best = max(ranking_pool, key=lambda r: r.mean_score)

        # Sort all_results by mean_score descending for readability.
        all_results.sort(
            key=lambda r: r.mean_score if not np.isnan(r.mean_score) else float("-inf"),
            reverse=True,
        )

        self._result = GridSearchResult(
            best_params=best.params,
            best_score=best.mean_score,
            n_candidates=n_candidates,
            n_splits=self.cv,
            elapsed_seconds=time.perf_counter() - t0,
            scoring_fn_name=getattr(self.scoring_fn, "__name__", "unknown"),
            all_results=all_results,
        )
        return self._result

    # ------------------------------------------------------------------ #
    # Convenience properties (available after fit)                         #
    # ------------------------------------------------------------------ #

    @property
    def best_params_(self) -> Dict[str, Any]:
        """Best hyperparameter combination found (call ``fit`` first)."""
        if self._result is None:
            raise RuntimeError("Call fit() before accessing best_params_.")
        return self._result.best_params

    @property
    def best_score_(self) -> float:
        """Mean CV score of the best combination (call ``fit`` first)."""
        if self._result is None:
            raise RuntimeError("Call fit() before accessing best_score_.")
        return self._result.best_score
