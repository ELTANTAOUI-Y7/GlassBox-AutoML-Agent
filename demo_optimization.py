"""
demo_optimization.py
====================

End-to-end demonstration of the GlassBox Optimization module (Phase IV).

Because Phase III (Algorithm Zoo) is not yet implemented, this demo uses two
minimal stub models built from scratch with pure NumPy:

  - ``DummyClassifier``   — majority-class predictor (classification baseline).
  - ``LinearRegressor``   — closed-form Ordinary Least Squares regression.

These stubs satisfy the required two-method contract::

    model = MyModel(**hyperparams)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

Once Phase III models are ready, swap the stub classes for the real ones —
the optimization module API does not change.

This script demonstrates:
  1. KFoldSplitter  — inspect how the data will be split.
  2. cross_val_score — quick CV evaluation for a single param combo.
  3. GridSearch      — exhaustive search over a discrete grid.
  4. RandomSearch    — stochastic search with a time budget.
  5. Orchestrator    — unified entry point, JSON report.
  6. Scoring metrics — accuracy, R², MSE, MAE from scratch.

Usage
-----
    python demo_optimization.py
"""

import json
import numpy as np

from glassbox.optimization.kfold import KFoldSplitter, cross_val_score
from glassbox.optimization.scoring import (
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    neg_mean_squared_error,
    f1_score,
    confusion_matrix,
)
from glassbox.optimization.grid_search import GridSearch
from glassbox.optimization.random_search import RandomSearch
from glassbox.optimization.orchestrator import Orchestrator, OrchestratorConfig


# ---------------------------------------------------------------------------
# Minimal stub models (replace with Phase III models when ready)
# ---------------------------------------------------------------------------

class DummyClassifier:
    """Predicts the majority class regardless of input.

    Hyperparameters
    ---------------
    strategy : str
        ``"majority"`` (always predict the most frequent training label) or
        ``"minority"`` (always predict the least frequent).  For demo purposes
        to give GridSearch multiple values to compare.
    """

    def __init__(self, strategy: str = "majority") -> None:
        self.strategy = strategy
        self._label = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        classes, counts = np.unique(y, return_counts=True)
        if self.strategy == "majority":
            self._label = classes[np.argmax(counts)]
        else:
            self._label = classes[np.argmin(counts)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._label)


class LinearRegressor:
    """Ordinary Least Squares regression via the normal equation.

    .. math::
        \\hat{\\beta} = (X^T X)^{-1} X^T y

    Hyperparameters
    ---------------
    fit_intercept : bool
        Whether to add a bias column of ones.  Default *True*.
    regularization : float
        Ridge (L2) penalty added to the diagonal of X^T X to avoid
        singular matrices.  Default 0.0.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        regularization: float = 0.0,
    ) -> None:
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self._weights: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X = np.hstack([ones, X])
        n_features = X.shape[1]
        # Normal equation: β = (XᵀX + λI)⁻¹ Xᵀy
        XtX = X.T @ X
        if self.regularization > 0.0:
            XtX += self.regularization * np.eye(n_features)
        self._weights = np.linalg.solve(XtX, X.T @ y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Call fit() before predict().")
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X = np.hstack([ones, X])
        return X @ self._weights


# ---------------------------------------------------------------------------
# Helper: Optional type hint for Python < 3.10 compat
# ---------------------------------------------------------------------------
from typing import Optional


# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------

def make_classification(n: int = 200, seed: int = 42) -> tuple:
    """Binary classification dataset — two linearly separable Gaussians."""
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=-1.0, scale=1.0, size=(n // 2, 4))
    X1 = rng.normal(loc=+1.0, scale=1.0, size=(n // 2, 4))
    X = np.vstack([X0, X1]).astype(float)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    shuffle_idx = rng.permutation(n)
    return X[shuffle_idx], y[shuffle_idx]


def make_regression(n: int = 200, seed: int = 7) -> tuple:
    """Simple linear regression dataset with Gaussian noise."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(n, 3))
    true_weights = np.array([2.5, -1.0, 0.8])
    y = X @ true_weights + rng.normal(0, 0.5, size=n)
    return X, y


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    width = 64
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def subsection(title: str) -> None:
    print(f"\n  -- {title} --")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    X_cls, y_cls = make_classification(n=200, seed=42)
    X_reg, y_reg = make_regression(n=200, seed=7)

    # ------------------------------------------------------------------ #
    # 1. KFoldSplitter — inspect fold shapes                              #
    # ------------------------------------------------------------------ #
    section("1. KFoldSplitter — Fold Metadata")
    splitter = KFoldSplitter(n_splits=5, shuffle=True, random_state=42)
    fold_info = splitter.get_fold_info(X_cls)
    print(f"\n  Dataset: {X_cls.shape[0]} samples, 5-fold split")
    print(f"  {'Fold':>6}  {'Train':>8}  {'Val':>8}")
    print(f"  {'----':>6}  {'-------':>8}  {'------':>8}")
    for fi in fold_info:
        print(f"  {fi.fold_index:>6}  {fi.n_train:>8}  {fi.n_val:>8}")

    # ------------------------------------------------------------------ #
    # 2. cross_val_score — single param combo                             #
    # ------------------------------------------------------------------ #
    section("2. cross_val_score — Single Param Evaluation")
    scores = cross_val_score(
        DummyClassifier,
        X_cls, y_cls,
        scoring_fn=accuracy_score,
        params={"strategy": "majority"},
        cv=5,
        random_state=42,
    )
    print(f"\n  DummyClassifier(strategy='majority') — 5-fold accuracy:")
    print(f"    Fold scores : {[round(s, 4) for s in scores.tolist()]}")
    print(f"    Mean ± Std  : {scores.mean():.4f} ± {scores.std():.4f}")

    # ------------------------------------------------------------------ #
    # 3. Scoring metrics demo                                              #
    # ------------------------------------------------------------------ #
    section("3. Scoring Metrics (from scratch)")

    subsection("Classification metrics")
    y_true_demo = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    y_pred_demo = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    print(f"\n  y_true : {y_true_demo.tolist()}")
    print(f"  y_pred : {y_pred_demo.tolist()}")
    print(f"\n  Accuracy  : {accuracy_score(y_true_demo, y_pred_demo):.4f}")
    print(f"  F1-Score  : {f1_score(y_true_demo, y_pred_demo):.4f}")
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true_demo, y_pred_demo)
    for row in cm:
        print(f"    {row.tolist()}")

    subsection("Regression metrics")
    y_true_reg = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred_reg = np.array([2.5,  0.0, 2.0, 8.0])
    print(f"\n  y_true : {y_true_reg.tolist()}")
    print(f"  y_pred : {y_pred_reg.tolist()}")
    print(f"\n  MSE  : {mean_squared_error(y_true_reg, y_pred_reg):.4f}")
    print(f"  MAE  : {mean_absolute_error(y_true_reg, y_pred_reg):.4f}")
    print(f"  R²   : {r2_score(y_true_reg, y_pred_reg):.4f}")

    # ------------------------------------------------------------------ #
    # 4. GridSearch — classification                                       #
    # ------------------------------------------------------------------ #
    section("4. GridSearch — Classification (DummyClassifier)")

    gs = GridSearch(
        scoring_fn=accuracy_score,
        cv=5,
        shuffle=True,
        random_state=42,
        verbose=True,
    )
    gs_result = gs.fit(
        X_cls, y_cls,
        estimator_class=DummyClassifier,
        param_grid={"strategy": ["majority", "minority"]},
    )
    print(f"\n  Best params : {gs_result.best_params}")
    print(f"  Best score  : {gs_result.best_score:.4f}")
    print(f"\n  Full ranking:")
    for r in gs_result.all_results:
        err_str = f"  [ERROR: {r.error}]" if r.error else ""
        print(
            f"    {r.params}  →  mean={r.mean_score:.4f} "
            f"± {r.std_score:.4f}{err_str}"
        )

    # ------------------------------------------------------------------ #
    # 5. GridSearch — regression                                           #
    # ------------------------------------------------------------------ #
    section("5. GridSearch — Regression (LinearRegressor)")

    gs_reg = GridSearch(
        scoring_fn=neg_mean_squared_error,
        cv=5,
        shuffle=True,
        random_state=0,
        verbose=True,
    )
    gs_reg_result = gs_reg.fit(
        X_reg, y_reg,
        estimator_class=LinearRegressor,
        param_grid={
            "fit_intercept": [True, False],
            "regularization": [0.0, 0.01, 0.1, 1.0],
        },
    )
    print(f"\n  Best params : {gs_reg_result.best_params}")
    print(f"  Best neg-MSE: {gs_reg_result.best_score:.4f}  "
          f"(MSE ≈ {-gs_reg_result.best_score:.4f})")
    print(f"\n  Top-3 candidates:")
    for r in gs_reg_result.all_results[:3]:
        print(
            f"    {r.params}  →  "
            f"neg-MSE={r.mean_score:.4f} ± {r.std_score:.4f}"
        )

    # ------------------------------------------------------------------ #
    # 6. RandomSearch — regression with continuous range + time budget     #
    # ------------------------------------------------------------------ #
    section("6. RandomSearch — Regression with Log-Uniform Sampling")

    rs = RandomSearch(
        scoring_fn=neg_mean_squared_error,
        cv=5,
        n_iter=25,
        time_budget=30.0,    # stop after 30 s (will finish well before that)
        shuffle=True,
        random_state=99,
        verbose=True,
    )
    rs_result = rs.fit(
        X_reg, y_reg,
        estimator_class=LinearRegressor,
        param_distributions={
            "fit_intercept": [True, False],        # discrete list
            "regularization": (1e-4, 10.0, "log"), # log-uniform continuous
        },
    )
    print(f"\n  Iterations completed : {rs_result.n_iter}")
    print(f"  Time budget hit      : {rs_result.time_budget_hit}")
    print(f"  Best params          : {rs_result.best_params}")
    print(f"  Best neg-MSE         : {rs_result.best_score:.4f}  "
          f"(MSE ≈ {-rs_result.best_score:.4f})")

    # ------------------------------------------------------------------ #
    # 7. Orchestrator — unified entry point + JSON report                  #
    # ------------------------------------------------------------------ #
    section("7. Orchestrator — Unified Entry Point")

    subsection("Grid search via Orchestrator")
    config_grid = OrchestratorConfig(
        search_type="grid",
        cv=5,
        shuffle=True,
        random_state=42,
        verbose=False,
    )
    orch_grid = Orchestrator(config=config_grid)

    # Inspect fold layout before running.
    folds = orch_grid.fold_info(X_cls)
    print(f"\n  Fold layout (cv={config_grid.cv}):")
    for f in folds:
        print(f"    fold {f['fold_index']}: train={f['n_train']}, val={f['n_val']}")

    report_grid = orch_grid.run(
        X_cls, y_cls,
        estimator_class=DummyClassifier,
        param_grid={"strategy": ["majority", "minority"]},
        scoring_fn=accuracy_score,
    )
    print(f"\n  Best params : {report_grid.best_params}")
    print(f"  Best score  : {report_grid.best_score:.4f}")
    if report_grid.warnings:
        print("  Warnings:")
        for w in report_grid.warnings:
            print(f"    ⚠  {w}")

    subsection("Random search via Orchestrator")
    config_rand = OrchestratorConfig(
        search_type="random",
        cv=5,
        n_iter=20,
        time_budget=30.0,
        random_state=7,
        verbose=False,
    )
    orch_rand = Orchestrator(config=config_rand)
    report_rand = orch_rand.run(
        X_reg, y_reg,
        estimator_class=LinearRegressor,
        param_grid={
            "fit_intercept": [True, False],
            "regularization": (1e-4, 5.0, "log"),
        },
        scoring_fn=neg_mean_squared_error,
    )
    print(f"\n  Best params  : {report_rand.best_params}")
    print(f"  Best neg-MSE : {report_rand.best_score:.4f}")
    print(f"  Iterations   : {report_rand.n_candidates_evaluated}")

    # ------------------------------------------------------------------ #
    # 8. Full JSON report                                                  #
    # ------------------------------------------------------------------ #
    section("8. Full Optimization Report (JSON) — Grid Search")
    # Print only metadata + best_params for brevity.
    report_dict = report_grid.to_dict()
    summary = {
        "metadata": report_dict["metadata"],
        "best_params": report_dict["best_params"],
        "best_score": report_dict["best_score"],
        "warnings": report_dict["warnings"],
    }
    print(json.dumps(summary, indent=2))

    print("\n  (Full report with all_results available via report.to_json())\n")


if __name__ == "__main__":
    main()
