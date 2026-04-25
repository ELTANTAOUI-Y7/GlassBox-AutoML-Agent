#NEW UPDATE
"""
glassbox.optimization.orchestrator
=====================================

**The Orchestrator** — top-level entry point for Phase IV: Parameter
Optimization.

The Orchestrator mirrors the design of the Phase I Inspector and Phase II
Cleaner: it wraps :class:`~glassbox.optimization.grid_search.GridSearch` or
:class:`~glassbox.optimization.random_search.RandomSearch` behind a unified
``run()`` method and returns a single JSON-serialisable
:class:`OptimizationReport`.

Typical usage in an IronClaw pipeline::

    1. Run Inspector (EDA)    → understand the data.
    2. Run Cleaner            → produce a clean float matrix X, target y.
    3. Run Orchestrator       → find the best hyperparameters via CV search.
    4. Re-train the model     → using best_params on the full dataset.
    5. Return JSON report     → to the agent for explanation.

Public API
----------
Orchestrator
    Main entry point.
OrchestratorConfig
    Configuration dataclass.
OptimizationReport
    JSON-serialisable output container.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np

from glassbox.optimization.kfold import KFoldSplitter
from glassbox.optimization.grid_search import GridSearch, GridSearchResult, _safe_json
from glassbox.optimization.random_search import (
    RandomSearch,
    RandomSearchResult,
    ParamDistribution,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorConfig:
    """Configuration for the :class:`Orchestrator`.

    Attributes
    ----------
    search_type : str
        Which search strategy to use.  ``"grid"`` for exhaustive grid search
        or ``"random"`` for stochastic random search.  Default ``"grid"``.
    cv : int
        Number of K-fold cross-validation folds.  Must be >= 2.  Default 5.
    n_iter : int
        Number of random combinations to evaluate.  Only used when
        ``search_type="random"``.  Default 20.
    time_budget : float, optional
        Wall-clock time limit in seconds for ``RandomSearch``.  The search
        stops early if this limit is reached.  ``None`` = no limit.
    shuffle : bool
        Shuffle samples before forming folds.  Default *True*.
    random_state : int, optional
        Reproducibility seed for both the fold shuffler and the parameter
        sampler (RandomSearch).
    verbose : bool
        Print per-candidate progress to stdout.  Default *False*.
    """

    search_type: str = "grid"
    cv: int = 5
    n_iter: int = 20
    time_budget: Optional[float] = None
    shuffle: bool = True
    random_state: Optional[int] = None
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.search_type not in {"grid", "random"}:
            raise ValueError(
                f"search_type must be 'grid' or 'random', "
                f"got '{self.search_type}'."
            )
        if self.cv < 2:
            raise ValueError(f"cv must be >= 2, got {self.cv}.")
        if self.n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {self.n_iter}.")
        if self.time_budget is not None and self.time_budget <= 0:
            raise ValueError(
                f"time_budget must be a positive number, got {self.time_budget}."
            )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class OptimizationReport:
    """Full output of an :class:`Orchestrator` run.

    Mirrors the structure of :class:`~glassbox.eda.inspector.EDAReport` and
    :class:`~glassbox.preprocessing.cleaner.PreprocessingReport` so the
    IronClaw agent receives a consistent JSON schema across all pipeline
    phases.

    Attributes
    ----------
    search_type : str
        Which search strategy was used (``"grid"`` or ``"random"``).
    best_params : dict
        Hyperparameter combination with the highest mean CV score.
    best_score : float
        Mean CV score of the best combination.
    n_candidates_evaluated : int
        Number of hyperparameter combinations that were evaluated.
    n_splits : int
        Number of cross-validation folds.
    elapsed_seconds : float
        Total wall-clock time for the search.
    scoring_fn_name : str
        Name of the scoring function used (for the report header).
    time_budget_hit : bool
        ``True`` if a RandomSearch stopped early due to the time budget.
    all_results : list[dict]
        Per-candidate CV summaries, sorted by mean score descending.
    warnings : list[str]
        Any noteworthy conditions (failed candidates, NaN scores, etc.).
    """

    search_type: str = "grid"
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_score: float = float("nan")
    n_candidates_evaluated: int = 0
    n_splits: int = 5
    elapsed_seconds: float = 0.0
    scoring_fn_name: str = "unknown"
    time_budget_hit: bool = False
    all_results: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return the full report as a plain dictionary."""
        return {
            "metadata": {
                "search_type": self.search_type,
                "n_candidates_evaluated": self.n_candidates_evaluated,
                "n_splits": self.n_splits,
                "elapsed_seconds": round(self.elapsed_seconds, 4),
                "scoring_fn": self.scoring_fn_name,
                "time_budget_hit": self.time_budget_hit,
            },
            "best_params": {k: _safe_json(v) for k, v in self.best_params.items()},
            "best_score": (
                round(self.best_score, 6) if not np.isnan(self.best_score) else None
            ),
            "all_results": self.all_results,
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the full report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Run the full hyperparameter optimisation pipeline.

    The Orchestrator is the Phase IV equivalent of the Cleaner (Phase II)
    and Inspector (Phase I): a single ``run()`` call chains the correct
    search strategy, runs K-fold cross-validation over the supplied parameter
    space, and returns a complete :class:`OptimizationReport`.

    Parameters
    ----------
    config : OrchestratorConfig, optional
        Pipeline configuration.  Uses sane defaults when *None*.

    Examples
    --------
    **Grid Search (classification)**

    >>> import numpy as np
    >>> from glassbox.optimization.orchestrator import Orchestrator, OrchestratorConfig
    >>> from glassbox.optimization.scoring import accuracy_score
    >>>
    >>> config = OrchestratorConfig(search_type="grid", cv=5, random_state=42)
    >>> orch = Orchestrator(config=config)
    >>> report = orch.run(
    ...     X, y,
    ...     estimator_class=MyClassifier,
    ...     param_grid={"max_depth": [3, 5, 10], "min_samples": [2, 5]},
    ...     scoring_fn=accuracy_score,
    ... )
    >>> print(report.best_params, report.best_score)
    >>> print(report.to_json())

    **Random Search (regression, with time budget)**

    >>> from glassbox.optimization.orchestrator import OrchestratorConfig
    >>> from glassbox.optimization.scoring import neg_mean_squared_error
    >>>
    >>> config = OrchestratorConfig(
    ...     search_type="random",
    ...     cv=5,
    ...     n_iter=50,
    ...     time_budget=30.0,
    ...     random_state=0,
    ... )
    >>> orch = Orchestrator(config=config)
    >>> report = orch.run(
    ...     X, y,
    ...     estimator_class=MyRegressor,
    ...     param_grid={
    ...         "max_depth": [3, 5, 10, 20],
    ...         "learning_rate": (0.001, 0.5, "log"),
    ...         "n_estimators": (50, 500),
    ...     },
    ...     scoring_fn=neg_mean_squared_error,
    ... )
    >>> print(report.best_params)
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        self.config = config or OrchestratorConfig()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator_class: Type,
        param_grid: Dict[str, Any],
        scoring_fn: Callable,
    ) -> OptimizationReport:
        """Execute the hyperparameter optimisation pipeline.

        Parameters
        ----------
        X : np.ndarray
            Cleaned feature matrix of shape ``(n_samples, n_features)``.
            This is typically the output array from
            :class:`~glassbox.preprocessing.cleaner.Cleaner`.
        y : np.ndarray
            Target vector of shape ``(n_samples,)``.  Must be numeric.
        estimator_class : type
            A class whose constructor accepts hyperparameter names from
            ``param_grid`` as keyword arguments and implements the
            two-method contract:

            - ``fit(X: np.ndarray, y: np.ndarray) -> None``
            - ``predict(X: np.ndarray) -> np.ndarray``

        param_grid : dict
            Hyperparameter search space.

            - **GridSearch**: ``{name: [val1, val2, ...]}`` — lists of
              discrete values; all combinations are evaluated.
            - **RandomSearch**: ``{name: list | (low, high) | (low, high, "log")}``
              — see :mod:`glassbox.optimization.random_search` for details.

        scoring_fn : callable
            ``scoring_fn(y_true, y_pred) -> float``.  The search **maximises**
            this score.  Use
            :func:`~glassbox.optimization.scoring.neg_mean_squared_error` for
            regression.  Ready-made functions live in
            :mod:`glassbox.optimization.scoring`.

        Returns
        -------
        OptimizationReport
            JSON-serialisable report with best params, best score, and full
            audit trail.
        """
        cfg = self.config
        warnings: List[str] = []

        if cfg.search_type == "grid":
            searcher = GridSearch(
                scoring_fn=scoring_fn,
                cv=cfg.cv,
                shuffle=cfg.shuffle,
                random_state=cfg.random_state,
                verbose=cfg.verbose,
            )
            raw: GridSearchResult = searcher.fit(X, y, estimator_class, param_grid)

            # Surface warnings for failed candidates.
            failed = [r for r in raw.all_results if r.error is not None]
            if failed:
                warnings.append(
                    f"{len(failed)} of {raw.n_candidates} candidates raised "
                    f"errors during cross-validation — check estimator and "
                    f"parameter ranges. First error: {failed[0].error}"
                )

            return OptimizationReport(
                search_type="grid",
                best_params=raw.best_params,
                best_score=raw.best_score,
                n_candidates_evaluated=raw.n_candidates,
                n_splits=raw.n_splits,
                elapsed_seconds=raw.elapsed_seconds,
                scoring_fn_name=raw.scoring_fn_name,
                time_budget_hit=False,
                all_results=[r.to_dict() for r in raw.all_results],
                warnings=warnings,
            )

        else:  # "random"
            searcher = RandomSearch(
                scoring_fn=scoring_fn,
                cv=cfg.cv,
                n_iter=cfg.n_iter,
                time_budget=cfg.time_budget,
                shuffle=cfg.shuffle,
                random_state=cfg.random_state,
                verbose=cfg.verbose,
            )
            raw: RandomSearchResult = searcher.fit(X, y, estimator_class, param_grid)

            if raw.time_budget_hit:
                warnings.append(
                    f"RandomSearch stopped early after {raw.n_iter} iterations "
                    f"— time budget of {cfg.time_budget}s was reached."
                )

            failed = [r for r in raw.all_results if r.error is not None]
            if failed:
                warnings.append(
                    f"{len(failed)} of {raw.n_iter} iterations raised errors "
                    f"during cross-validation. First error: {failed[0].error}"
                )

            return OptimizationReport(
                search_type="random",
                best_params=raw.best_params,
                best_score=raw.best_score,
                n_candidates_evaluated=raw.n_iter,
                n_splits=raw.n_splits,
                elapsed_seconds=raw.elapsed_seconds,
                scoring_fn_name=raw.scoring_fn_name,
                time_budget_hit=raw.time_budget_hit,
                all_results=[r.to_dict() for r in raw.all_results],
                warnings=warnings,
            )

    # ------------------------------------------------------------------ #
    # Convenience                                                          #
    # ------------------------------------------------------------------ #

    def fold_info(self, X: np.ndarray) -> List[Dict[str, int]]:
        """Return metadata for each cross-validation fold without running a search.

        Useful for inspecting how the data will be split before committing to
        a full search.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (only row count matters).

        Returns
        -------
        list[dict]
            One dict per fold with keys ``fold_index``, ``n_train``, ``n_val``.
        """
        splitter = KFoldSplitter(
            n_splits=self.config.cv,
            shuffle=self.config.shuffle,
            random_state=self.config.random_state,
        )
        return [fi.to_dict() for fi in splitter.get_fold_info(X)]
