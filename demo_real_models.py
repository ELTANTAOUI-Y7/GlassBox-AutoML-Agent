"""
demo_real_models.py
===================
Test the optimization module with the actual Phase III models.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from glassbox.optimization.orchestrator import Orchestrator, OrchestratorConfig
from glassbox.optimization.scoring import accuracy_score, neg_mean_squared_error
from glassbox.optimization.kfold import cross_val_score

from models.knn import KNN
from models.logistic_regression import LogisticRegression
from models.random_forest import RandomForest


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def make_classification(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=-1.0, scale=1.0, size=(n // 2, 4))
    X1 = rng.normal(loc=+1.0, scale=1.0, size=(n // 2, 4))
    X = np.vstack([X0, X1]).astype(float)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    idx = rng.permutation(n)
    return X[idx], y[idx]


def make_regression(n=200, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(n, 3))
    y = X @ np.array([2.5, -1.0, 0.8]) + rng.normal(0, 0.5, size=n)
    return X, y


def section(title):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


# ---------------------------------------------------------------------------
# 1. KNN — Grid Search (classification)
# ---------------------------------------------------------------------------

section("1. KNN  —  Grid Search (classification)")

X_cls, y_cls = make_classification()

config = OrchestratorConfig(search_type="grid", cv=5, random_state=42, verbose=True)
orch = Orchestrator(config=config)

report = orch.run(
    X_cls, y_cls,
    estimator_class=KNN,
    param_grid={
        "k":      [1, 3, 5, 7, 11],
        "metric": ["euclidean", "manhattan"],
    },
    scoring_fn=accuracy_score,
)

print(f"\n  Best params : {report.best_params}")
print(f"  Best score  : {report.best_score:.4f}  (accuracy)")
print(f"  Candidates  : {report.n_candidates_evaluated}")


# ---------------------------------------------------------------------------
# 2. LogisticRegression — Grid Search (classification)
# ---------------------------------------------------------------------------

section("2. LogisticRegression  —  Grid Search (classification)")

config2 = OrchestratorConfig(search_type="grid", cv=5, random_state=0, verbose=True)
orch2 = Orchestrator(config=config2)

report2 = orch2.run(
    X_cls, y_cls,
    estimator_class=LogisticRegression,
    param_grid={
        "lr":          [0.001, 0.01, 0.1],
        "n_iter":      [500, 1000],
        "lr_schedule": ["constant", "time_decay"],
    },
    scoring_fn=accuracy_score,
)

print(f"\n  Best params : {report2.best_params}")
print(f"  Best score  : {report2.best_score:.4f}  (accuracy)")
print(f"  Candidates  : {report2.n_candidates_evaluated}")


# ---------------------------------------------------------------------------
# 3. RandomForest — Random Search (classification)
# ---------------------------------------------------------------------------

section("3. RandomForest  —  Random Search (classification)")

config3 = OrchestratorConfig(
    search_type="random", cv=5, n_iter=15, random_state=99, verbose=True
)
orch3 = Orchestrator(config=config3)

report3 = orch3.run(
    X_cls, y_cls,
    estimator_class=RandomForest,
    param_grid={
        "n_trees":           [10, 20, 50],
        "max_depth":         [3, 5, 10],
        "min_samples_split": [2, 5, 10],
        "criterion":         ["gini", "entropy"],
    },
    scoring_fn=accuracy_score,
)

print(f"\n  Best params : {report3.best_params}")
print(f"  Best score  : {report3.best_score:.4f}  (accuracy)")


# ---------------------------------------------------------------------------
# 4. KNN regression — quick cross_val_score check
# ---------------------------------------------------------------------------

section("4. KNN Regression  —  cross_val_score sanity check")

X_reg, y_reg = make_regression()

scores = cross_val_score(
    KNN,
    X_reg, y_reg,
    scoring_fn=neg_mean_squared_error,
    params={"k": 5, "task": "regression"},
    cv=5,
    random_state=42,
)
print(f"\n  KNN(k=5, task='regression') — 5-fold neg-MSE:")
print(f"    Fold scores : {[round(s, 4) for s in scores.tolist()]}")
print(f"    Mean ± Std  : {scores.mean():.4f} ± {scores.std():.4f}")
print(f"    (MSE ≈ {-scores.mean():.4f})")
