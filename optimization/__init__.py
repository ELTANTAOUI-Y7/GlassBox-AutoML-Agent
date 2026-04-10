"""
GlassBox Optimization (The Orchestrator) — Automated Hyperparameter Search.

Submodules:
    - kfold:         KFoldSplitter — K-fold index generator; cross_val_score helper.
    - scoring:       Evaluation metrics from scratch (accuracy, MSE, MAE, R², F1, …).
    - grid_search:   GridSearch — exhaustive Cartesian-product search.
    - random_search: RandomSearch — stochastic sampling with optional time budget.
    - orchestrator:  Orchestrator — unified entry point returning OptimizationReport.
"""

from glassbox.optimization.kfold import KFoldSplitter, FoldSplit, cross_val_score
from glassbox.optimization.scoring import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    neg_mean_squared_error,
    neg_mean_absolute_error,
)
from glassbox.optimization.grid_search import GridSearch, GridSearchResult, CVResult
from glassbox.optimization.random_search import RandomSearch, RandomSearchResult
from glassbox.optimization.orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    OptimizationReport,
)

__all__ = [
    # KFold
    "KFoldSplitter",
    "FoldSplit",
    "cross_val_score",
    # Scoring
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "confusion_matrix",
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "neg_mean_squared_error",
    "neg_mean_absolute_error",
    # Search
    "GridSearch",
    "GridSearchResult",
    "CVResult",
    "RandomSearch",
    "RandomSearchResult",
    # Orchestrator
    "Orchestrator",
    "OrchestratorConfig",
    "OptimizationReport",
]
