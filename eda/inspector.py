"""
glassbox.eda.inspector
======================

**The Inspector** — top-level orchestrator for Automated EDA.

The Inspector chains every sub-module in the correct order:

1. **Auto-Typing** — detect column types (numerical / categorical / boolean).
2. **Statistical Profiling** — compute descriptive stats for numeric columns.
3. **Correlation Analysis** — build the Pearson correlation matrix and
   flag collinearity.
4. **Outlier Detection** — flag (and optionally cap) IQR-based outliers.

It produces a single, unified **JSON-serialisable report** suitable for
returning from an IronClaw agent tool call.

Public API
----------
Inspector
    Main entry point.  Instantiate with optional config, then call
    :pymeth:`run` passing a 2-D NumPy array and optional headers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from glassbox.eda.auto_typer import AutoTyper, ColumnTypeInfo
from glassbox.eda.correlation import CorrelationAnalyzer, CorrelationResult
from glassbox.eda.outliers import OutlierDetector, OutlierReport
from glassbox.eda.stats import ColumnStats, StatProfiler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class InspectorConfig:
    """Tuneable knobs for the Inspector pipeline.

    Attributes
    ----------
    categorical_cardinality_ratio : float
        See :class:`AutoTyper`.
    categorical_max_unique : int
        See :class:`AutoTyper`.
    correlation_threshold : float
        |r| above which a pair is flagged collinear.
    outlier_k : float
        IQR fence multiplier for outlier detection.
    cap_outliers : bool
        If *True*, the report includes a capped copy of the data.
    """
    categorical_cardinality_ratio: float = 0.05
    categorical_max_unique: int = 20
    correlation_threshold: float = 0.90
    outlier_k: float = 1.5
    cap_outliers: bool = False


# ---------------------------------------------------------------------------
# Full EDA Report
# ---------------------------------------------------------------------------
@dataclass
class EDAReport:
    """Container for the complete Inspector output.

    Convert to a JSON string via :pymeth:`to_json`.
    """
    # Metadata
    n_rows: int = 0
    n_cols: int = 0
    elapsed_seconds: float = 0.0

    # Sub-reports
    column_types: List[Dict[str, Any]] = field(default_factory=list)
    statistics: List[Dict[str, Any]] = field(default_factory=list)
    correlation: Dict[str, Any] = field(default_factory=dict)
    outliers: List[Dict[str, Any]] = field(default_factory=list)

    # Summaries
    type_summary: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return the full report as a plain dictionary."""
        return {
            "metadata": {
                "n_rows": self.n_rows,
                "n_cols": self.n_cols,
                "elapsed_seconds": round(self.elapsed_seconds, 4),
            },
            "type_summary": self.type_summary,
            "column_types": self.column_types,
            "statistics": self.statistics,
            "correlation": self.correlation,
            "outliers": self.outliers,
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the full report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------
class Inspector:
    """Run the full Automated EDA pipeline.

    Parameters
    ----------
    config : InspectorConfig, optional
        Pipeline configuration.  Uses sane defaults when *None*.

    Examples
    --------
    >>> import numpy as np
    >>> from glassbox.eda.inspector import Inspector
    >>> data = np.array([
    ...     [25, 50000, 1],
    ...     [30, 60000, 0],
    ...     [35, 55000, 1],
    ...     [40, 80000, 0],
    ...     [50, 120000, 1],
    ... ], dtype=float)
    >>> headers = ["age", "salary", "purchased"]
    >>> inspector = Inspector()
    >>> report = inspector.run(data, headers)
    >>> print(report.n_rows, report.n_cols)
    5 3
    >>> print(report.to_json()[:50])
    {
      "metadata": {
        "n_rows": 5,
        "n_cols": 3
    """

    def __init__(self, config: Optional[InspectorConfig] = None):
        self.config = config or InspectorConfig()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def run(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
    ) -> EDAReport:
        """Execute the full EDA pipeline.

        Parameters
        ----------
        data : np.ndarray
            2-D array of shape ``(n_rows, n_cols)``.  May be object dtype.
        headers : list[str], optional
            Column names.

        Returns
        -------
        EDAReport
            Complete audit of the dataset.
        """
        t0 = time.perf_counter()

        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n_rows, n_cols = data.shape
        if headers is None:
            headers = [f"col_{i}" for i in range(n_cols)]

        report = EDAReport(n_rows=n_rows, n_cols=n_cols)

        # 1. Auto-Typing ------------------------------------------------
        typer = AutoTyper(
            categorical_cardinality_ratio=self.config.categorical_cardinality_ratio,
            categorical_max_unique=self.config.categorical_max_unique,
        )
        type_infos: List[ColumnTypeInfo] = typer.detect(data, headers)
        report.column_types = [ti.to_dict() for ti in type_infos]

        # Build quick look-ups.
        type_map = {ti.name: ti.inferred_type for ti in type_infos}
        numeric_names = [ti.name for ti in type_infos if ti.inferred_type == "numerical"]
        numeric_indices = [i for i, ti in enumerate(type_infos) if ti.inferred_type == "numerical"]

        # Type summary.
        summary: Dict[str, int] = {}
        for ti in type_infos:
            summary[ti.inferred_type] = summary.get(ti.inferred_type, 0) + 1
        report.type_summary = summary

        # 2. Statistical Profiling --------------------------------------
        profiler = StatProfiler()
        col_stats: List[ColumnStats] = profiler.profile(
            data, headers, type_map=type_map
        )
        report.statistics = [cs.to_dict() for cs in col_stats]

        # 3. Correlation Analysis ---------------------------------------
        if len(numeric_indices) >= 2:
            corr_analyzer = CorrelationAnalyzer(
                threshold=self.config.correlation_threshold
            )
            corr_result: CorrelationResult = corr_analyzer.analyze(
                data, headers, numeric_indices=numeric_indices
            )
            report.correlation = corr_result.to_dict()

            # Collinearity warnings.
            for p in corr_result.high_pairs:
                report.warnings.append(
                    f"High correlation detected between '{p.col_a}' and "
                    f"'{p.col_b}' (r = {p.r:.4f}).  Consider removing one "
                    f"to reduce multicollinearity."
                )
        else:
            report.correlation = {
                "column_names": numeric_names,
                "matrix": [],
                "high_correlation_pairs": [],
            }

        # 4. Outlier Detection ------------------------------------------
        if numeric_indices:
            detector = OutlierDetector(k=self.config.outlier_k)
            outlier_reports: List[OutlierReport] = detector.detect(
                data, headers, numeric_indices=numeric_indices
            )
            report.outliers = [orep.to_dict() for orep in outlier_reports]

            # Outlier warnings.
            for orep in outlier_reports:
                if orep.outlier_pct > 5.0:
                    report.warnings.append(
                        f"Column '{orep.name}' has {orep.outlier_pct:.1f}% outliers "
                        f"({orep.n_outliers_low + orep.n_outliers_high} of {orep.n_total}).  "
                        f"Consider capping or investigating."
                    )

        # Missing-value warnings.
        for ti in type_infos:
            if ti.n_missing > 0:
                pct = (ti.n_missing / n_rows) * 100
                report.warnings.append(
                    f"Column '{ti.name}' has {ti.n_missing} missing values "
                    f"({pct:.1f}%)."
                )

        report.elapsed_seconds = time.perf_counter() - t0
        return report

    def run_json(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
    ) -> str:
        """Convenience wrapper — returns the report as a JSON string."""
        return self.run(data, headers).to_json()
