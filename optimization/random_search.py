"""
glassbox.optimization.random_search
=====================================

**Random Search** — stochastic sampling over a hyperparameter space.

Instead of evaluating every combination in a grid (combinatorial explosion),
RandomSearch draws ``n_iter`` random samples from ``param_distributions`` and
evaluates each with K-fold cross-validation.  This is dramatically faster than
GridSearch when the parameter space is large or when continuous ranges are
searched.

A ``time_budget`` (wall-clock seconds) can also be set — the search stops and
returns the best result found so far as soon as the budget is exhausted.

Parameter Distribution Specification
-------------------------------------
Each entry in ``param_distributions`` can be one of:

``list``
    Uniform discrete sampling.  One element is drawn at random on each
    iteration.  Example: ``"max_depth": [3, 5, 10, 20]``

``(low, high)``
    Continuous uniform sampling.  A float is drawn from ``Uniform[low, high]``
    on each iteration.  Example: ``"learning_rate": (0.001, 1.0)``

``(low, high, "log")``
    Log-uniform (reciprocal) sampling.  A float is drawn from
    ``exp(Uniform[log(low), log(high)])``.  Use this for hyperparameters
    that span several orders of magnitude (e.g., regularisation strength).
    Example: ``"alpha": (1e-5, 1.0, "log")``

Public API
----------
RandomSearch
    Main search class.
RandomSearchResult
    JSON-serialisable result container.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from glassbox.optimization.kfold import KFoldSplitter
from glassbox.optimization.grid_search import CVResult, _cross_validate, _safe_json


# ---------------------------------------------------------------------------
# Type alias for a single parameter distribution specification
# ---------------------------------------------------------------------------

#: A discrete list, a continuous ``(low, high)`` tuple, or a log-uniform
#: ``(low, high, "log")`` tuple.
ParamDistribution = Union[List[Any], Tuple]


# ---------------------------------------------------------------------------
# Internal sampler
# ---------------------------------------------------------------------------

def _sample_params(
    param_distributions: Dict[str, ParamDistribution],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Draw one parameter combination from the distributions.

    Parameters
    ----------
    param_distributions : dict
        Maps parameter names to distribution specifications.
    rng : np.random.Generator
        NumPy random generator for reproducible sampling.

    Returns
    -------
    dict[str, Any]
        One sampled value per parameter.

    Raises
    ------
    ValueError
        If a tuple distribution has an invalid format.
    TypeError
        If a distribution is neither a list nor a tuple.
    """
    sampled: Dict[str, Any] = {}
    for key, dist in param_distributions.items():
        if isinstance(dist, list):
            if not dist:
                raise ValueError(
                    f"param_distributions['{key}'] is an empty list."
                )
            # Discrete uniform: pick one element at random.
            idx = int(rng.integers(0, len(dist)))
            sampled[key] = dist[idx]

        elif isinstance(dist, tuple):
            if len(dist) == 3:
                # Log-uniform: (low, high, "log")
                if dist[2] != "log":
                    raise ValueError(
                        f"The third element of a tuple distribution must be "
                        f"'log', got {dist[2]!r} for parameter '{key}'."
                    )
                low, high = float(dist[0]), float(dist[1])
                if low <= 0 or high <= 0:
                    raise ValueError(
                        f"Log-uniform bounds must be positive for '{key}', "
                        f"got low={low}, high={high}."
                    )
                log_val = rng.uniform(np.log(low), np.log(high))
                sampled[key] = float(np.exp(log_val))

            elif len(dist) == 2:
                # Continuous uniform: (low, high)
                low, high = float(dist[0]), float(dist[1])
                if low > high:
                    raise ValueError(
                        f"Tuple distribution bounds must satisfy low <= high "
                        f"for '{key}', got low={low}, high={high}."
                    )
                sampled[key] = float(rng.uniform(low, high))

            else:
                raise ValueError(
                    f"Tuple distribution for '{key}' must be (low, high) or "
                    f"(low, high, 'log'), got length {len(dist)}."
                )
        else:
            raise TypeError(
                f"param_distributions['{key}'] must be a list or tuple, "
                f"got {type(dist).__name__!r}."
            )
    return sampled


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RandomSearchResult:
    """Full result of a :class:`RandomSearch` run.

    Attributes
    ----------
    best_params : dict
        Hyperparameters of the combination with the highest mean CV score.
    best_score : float
        Mean CV score of the best combination.
    n_iter : int
        Number of iterations that were actually completed (may be less than
        the requested ``n_iter`` if ``time_budget`` was hit).
    n_splits : int
        Number of cross-validation folds used.
    elapsed_seconds : float
        Total wall-clock time for the search.
    scoring_fn_name : str
        ``__name__`` of the scoring function.
    time_budget_hit : bool
        ``True`` if the search stopped early because the time budget was
        exhausted.
    all_results : list[CVResult]
        Full CV summary for every evaluated combination — sorted by
        ``mean_score`` descending.
    """

    best_params: Dict[str, Any]
    best_score: float
    n_iter: int
    n_splits: int
    elapsed_seconds: float
    scoring_fn_name: str
    time_budget_hit: bool = False
    all_results: List[CVResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return the full result as a JSON-safe dictionary."""
        return {
            "best_params": {k: _safe_json(v) for k, v in self.best_params.items()},
            "best_score": round(self.best_score, 6)
                          if not np.isnan(self.best_score) else None,
            "n_iter": self.n_iter,
            "n_splits": self.n_splits,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "scoring_fn": self.scoring_fn_name,
            "time_budget_hit": self.time_budget_hit,
            "all_results": [r.to_dict() for r in self.all_results],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the full result to a JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# RandomSearch
# ---------------------------------------------------------------------------

class RandomSearch:
    """Stochastic search over a hyperparameter space.

    On each of the ``n_iter`` iterations, one parameter combination is
    sampled from ``param_distributions`` and evaluated via K-fold
    cross-validation.  The combination with the highest mean CV score across
    all iterations is returned.

    An optional ``time_budget`` (in seconds) stops the search early when the
    wall-clock time is exceeded — useful for fixed-time AutoML pipelines.

    Parameters
    ----------
    scoring_fn : callable
        ``scoring_fn(y_true, y_pred) -> float``.  **Higher values are
        always considered better.**  For error metrics (MSE, MAE) use
        :func:`~glassbox.optimization.scoring.neg_mean_squared_error` /
        :func:`~glassbox.optimization.scoring.neg_mean_absolute_error`.
    cv : int, optional
        Number of cross-validation folds.  Default 5.
    n_iter : int, optional
        Number of random combinations to sample and evaluate.  Default 20.
    time_budget : float, optional
        Maximum wall-clock time in seconds.  The search stops as soon as
        the elapsed time reaches this value.  ``None`` means no limit.
    shuffle : bool, optional
        Shuffle samples before forming folds.  Default *True*.
    random_state : int, optional
        Seed for *both* the parameter sampler and the fold shuffler.
        Pass an integer for fully reproducible results.
    verbose : bool, optional
        Print a one-line progress update for each iteration.  Default *False*.

    Examples
    --------
    >>> import numpy as np
    >>> from glassbox.optimization.random_search import RandomSearch
    >>> from glassbox.optimization.scoring import neg_mean_squared_error
    >>> # Assume MyRegressor has fit(X, y) and predict(X).
    >>> rs = RandomSearch(
    ...     scoring_fn=neg_mean_squared_error,
    ...     cv=5,
    ...     n_iter=30,
    ...     time_budget=60.0,
    ...     random_state=0,
    ... )
    >>> result = rs.fit(
    ...     X, y, MyRegressor,
    ...     param_distributions={
    ...         "max_depth": [3, 5, 10, 20],
    ...         "learning_rate": (0.001, 0.5, "log"),
    ...         "n_estimators": (50, 500),
    ...     },
    ... )
    >>> print(result.best_params, result.best_score)
    """

    def __init__(
        self,
        scoring_fn: Callable,
        cv: int = 5,
        n_iter: int = 20,
        time_budget: Optional[float] = None,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        if cv < 2:
            raise ValueError(f"cv must be >= 2, got {cv}.")
        if n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {n_iter}.")
        if time_budget is not None and time_budget <= 0:
            raise ValueError(f"time_budget must be positive, got {time_budget}.")

        self.scoring_fn = scoring_fn
        self.cv = cv
        self.n_iter = n_iter
        self.time_budget = time_budget
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self._result: Optional[RandomSearchResult] = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator_class: Type,
        param_distributions: Dict[str, ParamDistribution],
    ) -> RandomSearchResult:
        """Run the random search.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.
        y : np.ndarray
            Target vector of shape ``(n_samples,)``.
        estimator_class : type
            A class whose ``__init__`` accepts the hyperparameter names in
            ``param_distributions`` as keyword arguments and implements
            ``fit(X, y)`` and ``predict(X)``.
        param_distributions : dict
            Maps each hyperparameter name to its distribution.  See module
            docstring for the supported specification formats.

        Returns
        -------
        RandomSearchResult
            Contains the best params, best score, time-budget metadata, and
            the full per-iteration audit trail.

        Raises
        ------
        ValueError
            If ``param_distributions`` is empty or a budget-only run
            completes zero iterations.
        RuntimeError
            If zero iterations complete (e.g. time budget too small).
        """
        if not param_distributions:
            raise ValueError("param_distributions must not be empty.")

        t0 = time.perf_counter()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        rng = np.random.default_rng(self.random_state)
        splitter = KFoldSplitter(
            n_splits=self.cv,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        if self.verbose:
            budget_str = (
                f", time_budget={self.time_budget}s"
                if self.time_budget is not None
                else ""
            )
            print(
                f"[RandomSearch] n_iter={self.n_iter}, cv={self.cv}"
                f"{budget_str}"
            )

        all_results: List[CVResult] = []
        time_budget_hit = False

        for i in range(self.n_iter):
            # Check time budget before starting each iteration.
            if self.time_budget is not None:
                if time.perf_counter() - t0 >= self.time_budget:
                    if self.verbose:
                        print(
                            f"  Time budget of {self.time_budget}s exhausted "
                            f"after {i} completed iterations."
                        )
                    time_budget_hit = True
                    break

            params = _sample_params(param_distributions, rng)
            if self.verbose:
                print(f"  Iteration [{i + 1}/{self.n_iter}]: {params}")

            cv_result = _cross_validate(
                estimator_class, params, X, y, self.scoring_fn, splitter
            )
            all_results.append(cv_result)

        if not all_results:
            raise RuntimeError(
                "RandomSearch completed zero iterations. "
                "The time_budget may be too short, or n_iter=0."
            )

        # Select the best — prefer candidates without errors.
        valid = [r for r in all_results if not np.isnan(r.mean_score)]
        ranking_pool = valid if valid else all_results
        best = max(ranking_pool, key=lambda r: r.mean_score)

        # Sort all_results by mean_score descending for readability.
        all_results.sort(
            key=lambda r: r.mean_score if not np.isnan(r.mean_score) else float("-inf"),
            reverse=True,
        )

        self._result = RandomSearchResult(
            best_params=best.params,
            best_score=best.mean_score,
            n_iter=len(all_results),
            n_splits=self.cv,
            elapsed_seconds=time.perf_counter() - t0,
            scoring_fn_name=getattr(self.scoring_fn, "__name__", "unknown"),
            time_budget_hit=time_budget_hit,
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
