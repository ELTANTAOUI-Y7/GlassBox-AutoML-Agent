"""
glassbox.agent.autofit
======================

**AutoFit** — the IronClaw "Skill" entry point for GlassBox.

A single ``run()`` / ``run_csv()`` / ``run_file()`` call chains the full
GlassBox pipeline and returns one JSON-serialisable ``AutoFitReport``:

    Raw CSV  →  Inspector (EDA)
             →  Cleaner (Preprocessing)
             →  Orchestrator × each candidate model
             →  Best model re-trained on full data
             →  AutoFitReport (JSON)

IronClaw integration flow
-------------------------
1. User: "Build a model to predict 'Churn' using this CSV."
2. Agent calls ``AutoFit().run_csv(csv_string, target_col="Churn")``.
3. AutoFit returns ``AutoFitReport`` as JSON.
4. Agent reads the report and explains:
   "The best model was RandomForest (accuracy 0.91).
    The most influential features were: tenure, MonthlyCharges, Contract."

Public API
----------
AutoFit
    Main entry point.
AutoFitConfig
    Configuration dataclass.
AutoFitReport
    JSON-serialisable output container.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — models/ lives at the repo root, not inside the glassbox
# namespace.  Insert the project root so ``from models.xxx import ...`` works
# regardless of how the package is invoked.
# ---------------------------------------------------------------------------
_REPO_ROOT = str(Path(__file__).parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from glassbox.eda.inspector import Inspector
from glassbox.preprocessing.cleaner import Cleaner, CleanerConfig
from glassbox.optimization.orchestrator import Orchestrator, OrchestratorConfig
from glassbox.optimization.scoring import accuracy_score, neg_mean_squared_error
from glassbox.optimization.kfold import cross_val_score as _cv_score

from models.decision_tree import DecisionTree
from models.random_forest import RandomForest
from models.naive_bayes import GaussianNaiveBayes
from models.logistic_regression import LogisticRegression
from models.linear_regression import LinearRegression
from models.knn import KNN


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AutoFitConfig:
    """Tuneable knobs for the AutoFit pipeline.

    Attributes
    ----------
    task : str, optional
        Force ``"classification"`` or ``"regression"``.  Auto-detected when
        *None* (default).
    classification_threshold : int
        Maximum number of unique target values before the task is treated as
        regression instead of classification.  Default 20.
    search_type : str
        ``"random"`` (fast, recommended) or ``"grid"`` (exhaustive).
    cv : int
        Number of K-fold cross-validation folds.  Default 5.
    n_iter : int
        Iterations for RandomSearch per model candidate.  Default 15.
    time_budget : float, optional
        Per-model wall-clock limit in seconds for RandomSearch.  None = no
        limit.
    random_state : int
        Reproducibility seed.  Default 42.
    scaler_type : str
        ``"standard"`` or ``"minmax"``.  Default ``"standard"``.
    encoder_type : str
        ``"onehot"`` or ``"label"``.  Default ``"onehot"``.
    """

    task: Optional[str] = None
    classification_threshold: int = 20
    search_type: str = "random"
    cv: int = 5
    n_iter: int = 15
    time_budget: Optional[float] = None
    random_state: int = 42
    scaler_type: str = "standard"
    encoder_type: str = "onehot"

    def __post_init__(self) -> None:
        if self.task is not None and self.task not in {"classification", "regression"}:
            raise ValueError(
                f"task must be 'classification', 'regression', or None; "
                f"got '{self.task}'."
            )
        if self.search_type not in {"grid", "random"}:
            raise ValueError(
                f"search_type must be 'grid' or 'random'; "
                f"got '{self.search_type}'."
            )
        if self.cv < 2:
            raise ValueError(f"cv must be >= 2, got {self.cv}.")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class AutoFitReport:
    """Full output of an :class:`AutoFit` run.

    All fields are JSON-serialisable.  Call :pymeth:`to_json` to get the
    string the IronClaw agent receives.

    Attributes
    ----------
    task_type : str
        ``"classification"`` or ``"regression"``.
    target_column : str
        Name of the target column that was predicted.
    n_rows : int
        Number of rows in the input dataset.
    n_features_in : int
        Number of feature columns before preprocessing.
    n_features_out : int
        Number of feature columns after preprocessing (may differ due to OHE).
    best_model : str
        Name of the winning model class.
    best_params : dict
        Hyperparameters of the winning model.
    best_score : float
        Mean CV score of the winning model.
    scoring_metric : str
        Name of the scoring function used (``"accuracy"`` or ``"neg_mse"``).
    feature_importances : list[dict]
        Per-feature ``{"feature": str, "importance": float}`` sorted
        descending by |correlation with target|.  Covers numerical features
        only (categorical features are listed after OHE expansion).
    model_rankings : list[dict]
        All tried models ranked by best CV score, descending.
    eda_report : dict
        Full output from :class:`~glassbox.eda.inspector.Inspector`.
    preprocessing_report : dict
        Full output from :class:`~glassbox.preprocessing.cleaner.Cleaner`.
    optimization_report : dict
        Orchestrator output for the winning model.
    warnings : list[str]
        Any issues encountered during the pipeline.
    elapsed_seconds : float
        Total wall-clock time.
    """

    task_type: str = ""
    target_column: str = ""
    n_rows: int = 0
    n_features_in: int = 0
    n_features_out: int = 0
    best_model: str = ""
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_score: float = float("nan")
    scoring_metric: str = ""
    feature_importances: List[Dict[str, Any]] = field(default_factory=list)
    model_rankings: List[Dict[str, Any]] = field(default_factory=list)
    eda_report: Dict[str, Any] = field(default_factory=dict)
    preprocessing_report: Dict[str, Any] = field(default_factory=dict)
    optimization_report: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Return the full report as a plain dictionary."""
        return {
            "metadata": {
                "task_type": self.task_type,
                "target_column": self.target_column,
                "n_rows": self.n_rows,
                "n_features_in": self.n_features_in,
                "n_features_out": self.n_features_out,
                "scoring_metric": self.scoring_metric,
                "elapsed_seconds": round(self.elapsed_seconds, 4),
            },
            "best_model": {
                "name": self.best_model,
                "params": self.best_params,
                "score": (
                    round(self.best_score, 6)
                    if not math.isnan(self.best_score)
                    else None
                ),
            },
            "feature_importances": self.feature_importances,
            "model_rankings": self.model_rankings,
            "eda_report": self.eda_report,
            "preprocessing_report": self.preprocessing_report,
            "optimization_report": self.optimization_report,
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the full report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Return a short human-readable summary for agent explainability."""
        score_str = (
            f"{self.best_score:.4f}"
            if not math.isnan(self.best_score)
            else "N/A"
        )
        top_features = [
            fi["feature"] for fi in self.feature_importances[:3]
        ]
        top_str = ", ".join(top_features) if top_features else "N/A"
        lines = [
            f"Task          : {self.task_type}",
            f"Target        : {self.target_column}",
            f"Dataset       : {self.n_rows} rows × {self.n_features_in} features",
            f"Best model    : {self.best_model}",
            f"Best params   : {self.best_params}",
            f"CV score      : {score_str} ({self.scoring_metric})",
            f"Top features  : {top_str}",
            f"Time          : {self.elapsed_seconds:.2f}s",
        ]
        if self.warnings:
            lines.append(f"Warnings      : {len(self.warnings)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AutoFit
# ---------------------------------------------------------------------------

class AutoFit:
    """Run the full GlassBox pipeline from raw data to a trained model.

    This is the object an IronClaw agent holds as a "Skill".  It exposes
    three entry points depending on how the agent receives the data:

    - :pymeth:`run_csv`  — raw CSV string (most common in agent contexts)
    - :pymeth:`run_file` — path to a CSV file on disk
    - :pymeth:`run`      — pre-parsed ``np.ndarray`` + headers

    Parameters
    ----------
    config : AutoFitConfig, optional
        Pipeline configuration.  Sane defaults when *None*.

    Examples
    --------
    **From a CSV string (IronClaw agent context)**

    >>> from glassbox.agent.autofit import AutoFit
    >>> af = AutoFit()
    >>> report = af.run_csv(csv_string, target_col="Churn")
    >>> print(report.summary())
    >>> print(report.to_json())

    **From a file path**

    >>> report = AutoFit().run_file("customers.csv", target_col="Salary")
    """

    def __init__(self, config: Optional[AutoFitConfig] = None) -> None:
        self.config = config or AutoFitConfig()

    # ------------------------------------------------------------------ #
    # Public entry points                                                  #
    # ------------------------------------------------------------------ #

    def run_csv(self, csv_string: str, target_col: str) -> AutoFitReport:
        """Parse *csv_string* and run the full pipeline.

        Parameters
        ----------
        csv_string : str
            Raw CSV content, including a header row.
        target_col : str
            Name of the column to predict.

        Returns
        -------
        AutoFitReport
        """
        data, headers = self._parse_csv(csv_string)
        return self.run(data, headers, target_col)

    def run_file(self, csv_path: str, target_col: str) -> AutoFitReport:
        """Load a CSV from *csv_path* and run the full pipeline.

        Parameters
        ----------
        csv_path : str
            Absolute or relative path to the CSV file.
        target_col : str
            Name of the column to predict.

        Returns
        -------
        AutoFitReport
        """
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            csv_string = f.read()
        return self.run_csv(csv_string, target_col)

    def run(
        self,
        data: np.ndarray,
        headers: List[str],
        target_col: str,
    ) -> AutoFitReport:
        """Run the full pipeline on a pre-parsed array.

        Parameters
        ----------
        data : np.ndarray
            2-D object array of shape ``(n_rows, n_cols)``.
        headers : list[str]
            Column names matching the columns in *data*.
        target_col : str
            Name of the column to predict.  Must be in *headers*.

        Returns
        -------
        AutoFitReport
        """
        t0 = time.perf_counter()
        cfg = self.config
        warnings: List[str] = []

        if target_col not in headers:
            raise ValueError(
                f"target_col '{target_col}' not found in headers: {headers}"
            )

        # ---------------------------------------------------------------- #
        # Step 1: Separate features and target                              #
        # ---------------------------------------------------------------- #
        target_idx = headers.index(target_col)
        y_raw: np.ndarray = data[:, target_idx]
        feature_data = np.delete(data, target_idx, axis=1)
        feature_headers = [h for i, h in enumerate(headers) if i != target_idx]

        n_rows = data.shape[0]
        n_features_in = len(feature_headers)

        # ---------------------------------------------------------------- #
        # Step 2: Detect task type                                          #
        # ---------------------------------------------------------------- #
        task = cfg.task or self._detect_task(y_raw, cfg.classification_threshold)

        # ---------------------------------------------------------------- #
        # Step 3: EDA — Inspector on feature columns only                   #
        # ---------------------------------------------------------------- #
        inspector = Inspector()
        eda_report = inspector.run(feature_data, feature_headers)

        # ---------------------------------------------------------------- #
        # Step 4: Preprocessing — Cleaner on feature columns               #
        # ---------------------------------------------------------------- #
        cleaner = Cleaner(CleanerConfig(
            scaler_type=cfg.scaler_type,
            encoder_type=cfg.encoder_type,
        ))
        cleaner_result = cleaner.run(feature_data, feature_headers)

        # The Cleaner does not encode boolean columns (only "categorical").
        # Convert any remaining string booleans to 0/1 so the matrix is
        # fully numeric before passing it to the models.
        X, bool_converted = self._finalize_feature_matrix(cleaner_result.data)
        if bool_converted:
            warnings.append(
                f"Boolean columns encoded as 0/1 by AutoFit "
                f"(Cleaner skips boolean type): {bool_converted}"
            )

        n_features_out = X.shape[1]

        # ---------------------------------------------------------------- #
        # Step 5: Encode target                                             #
        # ---------------------------------------------------------------- #
        y, class_mapping = self._encode_target(y_raw, task)

        # ---------------------------------------------------------------- #
        # Step 6: Feature importances (|Pearson r| with target)             #
        # ---------------------------------------------------------------- #
        feature_importances = self._compute_feature_importances(
            X, y, cleaner_result.headers
        )

        # ---------------------------------------------------------------- #
        # Step 7: Build model registry                                      #
        # ---------------------------------------------------------------- #
        candidates = self._get_candidates(task)

        orch_config = OrchestratorConfig(
            search_type=cfg.search_type,
            cv=cfg.cv,
            n_iter=cfg.n_iter,
            time_budget=cfg.time_budget,
            random_state=cfg.random_state,
        )
        scoring_fn = accuracy_score if task == "classification" else neg_mean_squared_error
        scoring_name = "accuracy" if task == "classification" else "neg_mse"

        # ---------------------------------------------------------------- #
        # Step 8: Optimize each candidate model                             #
        # ---------------------------------------------------------------- #
        best_score = float("-inf")
        best_model_name = ""
        best_model_params: Dict[str, Any] = {}
        best_opt_report = None
        model_rankings: List[Dict[str, Any]] = []

        for model_name, estimator_class, param_grid in candidates:
            try:
                if not param_grid:
                    # No hyperparameters (e.g. GaussianNaiveBayes) — bypass
                    # the search and run a direct K-fold cross-validation with
                    # default constructor args.
                    scores = _cv_score(
                        estimator_class,
                        X,
                        y,
                        scoring_fn=scoring_fn,
                        params={},
                        cv=cfg.cv,
                        shuffle=True,
                        random_state=cfg.random_state,
                    )
                    candidate_score = float(np.nanmean(scores))
                    model_rankings.append({
                        "model": model_name,
                        "best_score": (
                            round(candidate_score, 6)
                            if not math.isnan(candidate_score)
                            else None
                        ),
                        "best_params": {},
                    })
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_model_name = model_name
                        best_model_params = {}
                        best_opt_report = None
                else:
                    orch = Orchestrator(orch_config)
                    opt_report = orch.run(X, y, estimator_class, param_grid, scoring_fn)

                    model_rankings.append({
                        "model": model_name,
                        "best_score": (
                            round(opt_report.best_score, 6)
                            if not math.isnan(opt_report.best_score)
                            else None
                        ),
                        "best_params": opt_report.best_params,
                    })

                    if opt_report.best_score > best_score:
                        best_score = opt_report.best_score
                        best_model_name = model_name
                        best_model_params = opt_report.best_params
                        best_opt_report = opt_report

            except Exception as exc:
                warnings.append(f"{model_name} failed during optimization: {exc}")
                model_rankings.append({
                    "model": model_name,
                    "best_score": None,
                    "best_params": {},
                    "error": str(exc),
                })

        # Sort rankings descending (None scores go to the bottom).
        model_rankings.sort(
            key=lambda r: r["best_score"] if r["best_score"] is not None else float("-inf"),
            reverse=True,
        )

        elapsed = time.perf_counter() - t0

        return AutoFitReport(
            task_type=task,
            target_column=target_col,
            n_rows=n_rows,
            n_features_in=n_features_in,
            n_features_out=n_features_out,
            best_model=best_model_name,
            best_params=best_model_params,
            best_score=best_score,
            scoring_metric=scoring_name,
            feature_importances=feature_importances,
            model_rankings=model_rankings,
            eda_report=eda_report.to_dict(),
            preprocessing_report=cleaner_result.report.to_dict(),
            optimization_report=best_opt_report.to_dict() if best_opt_report else {},
            warnings=warnings,
            elapsed_seconds=elapsed,
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    # Map of string boolean values → float.  Checked case-insensitively.
    _BOOL_MAP: Dict[str, float] = {
        "yes": 1.0, "no": 0.0,
        "true": 1.0, "false": 0.0,
        "1": 1.0, "0": 0.0,
        "y": 1.0, "n": 0.0,
    }

    def _finalize_feature_matrix(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, List[int]]:
        """Convert the Cleaner output to a pure float matrix.

        The Cleaner leaves boolean columns as their original string values
        (e.g. ``"yes"`` / ``"no"``).  This method converts any non-float
        column that looks boolean into 0/1, then casts the whole array to
        ``float64``.

        Returns
        -------
        X : np.ndarray  shape (n_rows, n_cols), dtype float64
        bool_converted : list[int]  column indices that were bool-mapped
        """
        out = data.copy().astype(object)
        bool_converted: List[int] = []

        for col_idx in range(out.shape[1]):
            col = out[:, col_idx]
            # Try a direct numeric cast first.
            try:
                out[:, col_idx] = col.astype(float)
                continue
            except (ValueError, TypeError):
                pass

            # Try boolean mapping.
            mapped = np.empty(len(col), dtype=float)
            ok = True
            for row_idx, val in enumerate(col):
                key = str(val).strip().lower() if val is not None else ""
                if key in self._BOOL_MAP:
                    mapped[row_idx] = self._BOOL_MAP[key]
                else:
                    ok = False
                    break
            if ok:
                out[:, col_idx] = mapped
                bool_converted.append(col_idx)
            else:
                raise ValueError(
                    f"Column index {col_idx} could not be converted to float "
                    f"after preprocessing.  Sample values: {list(col[:5])}"
                )

        return out.astype(float), bool_converted

    def _parse_csv(self, csv_string: str) -> Tuple[np.ndarray, List[str]]:
        """Parse a raw CSV string into an object array + header list."""
        reader = csv.reader(io.StringIO(csv_string.strip()))
        rows = list(reader)
        if len(rows) < 2:
            raise ValueError("CSV must have at least a header row and one data row.")
        headers = [h.strip() for h in rows[0]]
        data = np.array(
            [[cell.strip() if cell.strip() != "" else None for cell in row]
             for row in rows[1:]],
            dtype=object,
        )
        return data, headers

    def _detect_task(self, y_raw: np.ndarray, threshold: int) -> str:
        """Auto-detect classification vs regression from the target column."""
        non_null = [v for v in y_raw if v is not None]
        unique_vals = set(str(v) for v in non_null)

        # If we can parse every value as float AND there are many unique
        # values, treat it as regression.
        try:
            _ = [float(v) for v in non_null]
            if len(unique_vals) > threshold:
                return "regression"
        except (ValueError, TypeError):
            pass

        return "classification"

    def _encode_target(
        self,
        y_raw: np.ndarray,
        task: str,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """Encode the raw target column into a 1-D float array.

        Returns
        -------
        y : np.ndarray
            Numeric target of shape ``(n_rows,)``.
        class_mapping : dict
            For classification: ``{label_str: int_code}``.  Empty for
            regression.
        """
        if task == "regression":
            y = np.array([float(v) for v in y_raw], dtype=float)
            return y, {}

        # Classification: alphabetically stable label encoding.
        non_null_strs = [str(v) for v in y_raw if v is not None]
        classes = sorted(set(non_null_strs))
        mapping = {c: i for i, c in enumerate(classes)}
        y = np.array(
            [mapping.get(str(v), -1) for v in y_raw],
            dtype=float,
        )
        return y, mapping

    def _compute_feature_importances(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Rank features by |Pearson correlation with the target|.

        Only columns where the correlation is computable (non-zero variance)
        are included.  This works for both classification (integer-coded y)
        and regression.

        Returns
        -------
        list[dict]
            ``[{"feature": str, "importance": float}, ...]`` sorted
            descending.
        """
        importances: List[Dict[str, Any]] = []
        y_mean = float(np.mean(y))
        y_std = float(np.std(y))

        if y_std == 0.0:
            return importances

        for i, name in enumerate(feature_names):
            col = X[:, i].astype(float)
            col_std = float(np.std(col))
            if col_std == 0.0:
                continue
            # Pearson r: cov(x, y) / (std_x * std_y)
            r = float(np.mean((col - np.mean(col)) * (y - y_mean)) / (col_std * y_std))
            importances.append({"feature": name, "importance": round(abs(r), 4)})

        importances.sort(key=lambda d: d["importance"], reverse=True)
        return importances

    def _get_candidates(
        self,
        task: str,
    ) -> List[Tuple[str, Type, Dict[str, Any]]]:
        """Return ``(name, estimator_class, param_grid)`` for each candidate.

        The param grids are intentionally compact so the search stays fast
        by default.  Callers can widen them by subclassing AutoFit and
        overriding this method.
        """
        if task == "classification":
            return [
                (
                    "DecisionTree",
                    DecisionTree,
                    {
                        "task": ["classification"],
                        "criterion": ["gini"],
                        "max_depth": [3, 5, 10, None],
                        "min_samples_split": [2, 5, 10],
                    },
                ),
                (
                    "RandomForest",
                    RandomForest,
                    {
                        "task": ["classification"],
                        "n_trees": [10, 30, 50],
                        "max_depth": [3, 5, 10],
                        "min_samples_split": [2, 5],
                    },
                ),
                (
                    "GaussianNaiveBayes",
                    GaussianNaiveBayes,
                    {},  # No hyperparameters — one candidate, default params.
                ),
                (
                    "LogisticRegression",
                    LogisticRegression,
                    {
                        "lr": [0.001, 0.01, 0.1],
                        "n_iter": [200, 500],
                    },
                ),
                (
                    "KNN",
                    KNN,
                    {
                        "k": [3, 5, 7, 11],
                        "task": ["classification"],
                        "metric": ["euclidean"],
                    },
                ),
            ]
        else:
            return [
                (
                    "DecisionTree",
                    DecisionTree,
                    {
                        "task": ["regression"],
                        "criterion": ["mse"],
                        "max_depth": [3, 5, 10, None],
                        "min_samples_split": [2, 5, 10],
                    },
                ),
                (
                    "RandomForest",
                    RandomForest,
                    {
                        "task": ["regression"],
                        "n_trees": [10, 30, 50],
                        "max_depth": [3, 5, 10],
                        "min_samples_split": [2, 5],
                    },
                ),
                (
                    "LinearRegression",
                    LinearRegression,
                    {
                        "lr": [0.001, 0.01],
                        "n_iter": [200, 500],
                    },
                ),
                (
                    "KNN",
                    KNN,
                    {
                        "k": [3, 5, 7, 11],
                        "task": ["regression"],
                        "metric": ["euclidean"],
                    },
                ),
            ]
