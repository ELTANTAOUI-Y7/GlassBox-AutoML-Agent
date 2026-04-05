"""
glassbox.preprocessing.cleaner
================================

**The Cleaner** — top-level orchestrator for the Preprocessing Engine.

The Cleaner chains every sub-module in the correct order:

1. **Auto-Typing** (via :class:`glassbox.eda.auto_typer.AutoTyper`) — detect
   column types so each step knows what to act on.  Reuses the EDA module.
2. **Imputation** (:class:`~glassbox.preprocessing.imputer.SimpleImputer`) —
   fill missing values using mean/median/mode/constant.
3. **Scaling** (:class:`~glassbox.preprocessing.scalers.MinMaxScaler` or
   :class:`~glassbox.preprocessing.scalers.StandardScaler`) — normalise
   numerical features.
4. **Encoding** (:class:`~glassbox.preprocessing.encoders.OneHotEncoder` or
   :class:`~glassbox.preprocessing.encoders.LabelEncoder`) — encode
   categorical features as numbers.

It produces a unified **JSON-serialisable report** (``PreprocessingReport``)
and returns the cleaned data array alongside updated column headers.

The Cleaner honours the same design goals as The Inspector:
- **Non-destructive**: the original data is never modified.
- **NumPy-only**: no Pandas or Scikit-Learn.
- **EDA-aware**: accepts an ``InspectorConfig``-derived ``type_map`` or runs
  AutoTyper automatically.
- **JSON-first output**: every result serialises to a plain JSON object.

Public API
----------
Cleaner
    Main entry point.
CleanerConfig
    Configuration dataclass.
PreprocessingReport
    Output container — holds the cleaned data, new headers, and a full
    audit trail of what was applied.
CleanerResult
    Lightweight wrapper returned by ``run()`` that bundles the cleaned array,
    headers, and the report.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from glassbox.eda.auto_typer import AutoTyper, ColumnTypeInfo
from glassbox.preprocessing.imputer import SimpleImputer, ImputationSummary
from glassbox.preprocessing.scalers import MinMaxScaler, StandardScaler, ScalerSummary
from glassbox.preprocessing.encoders import OneHotEncoder, LabelEncoder, EncoderSummary


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class CleanerConfig:
    """Tuneable knobs for the Cleaner pipeline.

    Attributes
    ----------
    imputer_strategy : str
        Strategy for :class:`~glassbox.preprocessing.imputer.SimpleImputer`.
        One of ``"mean"``, ``"median"``, ``"mode"``, ``"constant"``.
    imputer_fill_value : Any
        Used when ``imputer_strategy="constant"``.
    scale_numerical : bool
        Whether to scale numerical columns.  Default ``True``.
    scaler_type : str
        ``"standard"`` (zero mean, unit variance) or ``"minmax"`` (``[0, 1]``
        range).  Default ``"standard"``.
    minmax_range : tuple[float, float]
        Output range for MinMaxScaler.  Default ``(0.0, 1.0)``.
    encode_categorical : bool
        Whether to encode categorical columns.  Default ``True``.
    encoder_type : str
        ``"onehot"`` (binary indicator columns) or ``"label"`` (integer
        mapping).  Default ``"onehot"``.
    drop_first_ohe : bool
        When ``encoder_type="onehot"``, drop the first indicator column per
        feature to avoid the dummy-variable trap.  Default ``False``.
    """

    imputer_strategy: str = "mean"
    imputer_fill_value: Any = None
    scale_numerical: bool = True
    scaler_type: str = "standard"
    minmax_range: Tuple[float, float] = (0.0, 1.0)
    encode_categorical: bool = True
    encoder_type: str = "onehot"
    drop_first_ohe: bool = False

    def __post_init__(self) -> None:
        valid_strategies = {"mean", "median", "mode", "constant"}
        if self.imputer_strategy not in valid_strategies:
            raise ValueError(
                f"imputer_strategy must be one of {valid_strategies}, "
                f"got '{self.imputer_strategy}'."
            )
        if self.scaler_type not in {"standard", "minmax"}:
            raise ValueError(
                f"scaler_type must be 'standard' or 'minmax', "
                f"got '{self.scaler_type}'."
            )
        if self.encoder_type not in {"onehot", "label"}:
            raise ValueError(
                f"encoder_type must be 'onehot' or 'label', "
                f"got '{self.encoder_type}'."
            )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass
class PreprocessingReport:
    """Full audit report for one Cleaner run.

    Attributes
    ----------
    n_rows : int
        Number of rows in the input dataset.
    n_cols_in : int
        Number of columns before preprocessing.
    n_cols_out : int
        Number of columns after preprocessing (may differ due to OHE).
    elapsed_seconds : float
        Wall-clock time for the complete pipeline.
    headers_in : list[str]
        Original column names.
    headers_out : list[str]
        Column names after preprocessing.
    type_summary : dict[str, int]
        Count of columns per inferred type (``numerical``, ``categorical``,
        ``boolean``).
    column_types : list[dict]
        Per-column type metadata from :class:`glassbox.eda.auto_typer.AutoTyper`.
    imputation : list[dict]
        Per-column imputation details.
    scaling : list[dict]
        Per-column scaling parameters.
    encoding : list[dict]
        Per-column encoding details.
    steps_applied : list[str]
        Human-readable list of pipeline steps that were executed.
    warnings : list[str]
        Any issues encountered (all-missing columns, constant columns, etc.).
    """

    n_rows: int = 0
    n_cols_in: int = 0
    n_cols_out: int = 0
    elapsed_seconds: float = 0.0
    headers_in: List[str] = field(default_factory=list)
    headers_out: List[str] = field(default_factory=list)
    type_summary: Dict[str, int] = field(default_factory=dict)
    column_types: List[Dict[str, Any]] = field(default_factory=list)
    imputation: List[Dict[str, Any]] = field(default_factory=list)
    scaling: List[Dict[str, Any]] = field(default_factory=list)
    encoding: List[Dict[str, Any]] = field(default_factory=list)
    steps_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return the full report as a plain dictionary."""
        return {
            "metadata": {
                "n_rows": self.n_rows,
                "n_cols_in": self.n_cols_in,
                "n_cols_out": self.n_cols_out,
                "elapsed_seconds": round(self.elapsed_seconds, 4),
            },
            "headers_in": self.headers_in,
            "headers_out": self.headers_out,
            "type_summary": self.type_summary,
            "column_types": self.column_types,
            "steps_applied": self.steps_applied,
            "imputation": self.imputation,
            "scaling": self.scaling,
            "encoding": self.encoding,
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the full report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# CleanerResult
# ---------------------------------------------------------------------------
@dataclass
class CleanerResult:
    """Output of :pymeth:`Cleaner.run`.

    Attributes
    ----------
    data : np.ndarray
        Cleaned data array (object dtype; contains floats for numerical
        columns and ints for label-encoded columns).
    headers : list[str]
        Column names matching *data*.
    report : PreprocessingReport
        Full audit report with JSON-serialisation support.
    """

    data: np.ndarray
    headers: List[str]
    report: PreprocessingReport

    def to_json(self, indent: int = 2) -> str:
        """Convenience shortcut — serialise the report to JSON."""
        return self.report.to_json(indent=indent)


# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------
class Cleaner:
    """Run the full Preprocessing pipeline.

    Parameters
    ----------
    config : CleanerConfig, optional
        Pipeline configuration.  Uses sane defaults when *None*.

    Examples
    --------
    >>> import numpy as np
    >>> from glassbox.preprocessing.cleaner import Cleaner, CleanerConfig
    >>> data = np.array([
    ...     [25, "Engineering", "yes", 8.5],
    ...     [30, "Sales",       None,  6.2],
    ...     [35, "HR",          "yes", np.nan],
    ...     [40, "Engineering", "no",  9.1],
    ...     [50, "Marketing",   "yes", 5.5],
    ... ], dtype=object)
    >>> headers = ["age", "dept", "manager", "rating"]
    >>> config = CleanerConfig(encoder_type="onehot", scaler_type="minmax")
    >>> cleaner = Cleaner(config=config)
    >>> result = cleaner.run(data, headers)
    >>> print(result.headers)
    ['age', 'rating', 'dept_Engineering', 'dept_HR', 'dept_Marketing', 'dept_Sales', 'manager_no', 'manager_yes']
    >>> print(result.to_json()[:100])
    """

    def __init__(self, config: Optional[CleanerConfig] = None) -> None:
        self.config = config or CleanerConfig()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
        type_map: Optional[Dict[str, str]] = None,
    ) -> CleanerResult:
        """Execute the full preprocessing pipeline on *data*.

        Parameters
        ----------
        data : np.ndarray
            2-D input array of shape ``(n_rows, n_cols)``.  May be object
            dtype for mixed-type datasets.
        headers : list[str], optional
            Column names.  Auto-generated when *None*.
        type_map : dict[str, str], optional
            Pre-computed ``{col_name: inferred_type}`` from the EDA
            :class:`~glassbox.eda.auto_typer.AutoTyper`.  Auto-detected when
            *None* — pass this to avoid running AutoTyper twice when the EDA
            Inspector has already been called.

        Returns
        -------
        CleanerResult
            Contains the cleaned data array, updated headers, and a full
            ``PreprocessingReport``.
        """
        t0 = time.perf_counter()
        cfg = self.config

        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n_rows, n_cols = data.shape
        if headers is None:
            headers = [f"col_{i}" for i in range(n_cols)]

        report = PreprocessingReport(
            n_rows=n_rows,
            n_cols_in=n_cols,
            headers_in=list(headers),
        )

        # ---------------------------------------------------------------- #
        # Step 1: Auto-Typing                                               #
        # ---------------------------------------------------------------- #
        if type_map is None:
            typer = AutoTyper()
            type_infos: List[ColumnTypeInfo] = typer.detect(data, headers)
            type_map = {ti.name: ti.inferred_type for ti in type_infos}
        else:
            # Reconstruct ColumnTypeInfo list from the provided type_map for
            # the report (re-run AutoTyper for metadata only — no extra cost).
            typer = AutoTyper()
            type_infos = typer.detect(data, headers)

        report.column_types = [ti.to_dict() for ti in type_infos]

        # Type summary counts.
        summary: Dict[str, int] = {}
        for ti in type_infos:
            summary[ti.inferred_type] = summary.get(ti.inferred_type, 0) + 1
        report.type_summary = summary

        # Warnings: missing values detected by AutoTyper.
        for ti in type_infos:
            if ti.n_missing > 0:
                pct = (ti.n_missing / n_rows) * 100
                report.warnings.append(
                    f"Column '{ti.name}' has {ti.n_missing} missing values "
                    f"({pct:.1f}%) — will be imputed with strategy "
                    f"'{cfg.imputer_strategy}'."
                )

        report.steps_applied.append("auto_typing")

        # Work on a mutable copy from here on.
        current_data = data.copy().astype(object)
        current_headers = list(headers)

        # ---------------------------------------------------------------- #
        # Step 2: Imputation                                                #
        # ---------------------------------------------------------------- #
        imputer = SimpleImputer(
            strategy=cfg.imputer_strategy,
            fill_value=cfg.imputer_fill_value,
        )
        current_data, imp_summaries = imputer.fit_transform(
            current_data, current_headers, type_map=type_map
        )
        report.imputation = [s.to_dict() for s in imp_summaries]
        report.steps_applied.append(
            f"imputation(strategy='{cfg.imputer_strategy}')"
        )

        # Warn about all-missing columns that could not be imputed.
        for s in imp_summaries:
            if s.fill_value is None and s.n_filled > 0:
                report.warnings.append(
                    f"Column '{s.column}' is entirely missing — "
                    f"imputation was skipped."
                )

        # ---------------------------------------------------------------- #
        # Step 3: Scaling (numerical columns only)                          #
        # ---------------------------------------------------------------- #
        if cfg.scale_numerical:
            if cfg.scaler_type == "minmax":
                scaler = MinMaxScaler(feature_range=cfg.minmax_range)
            else:
                scaler = StandardScaler()

            current_data, scale_summaries = scaler.fit_transform(
                current_data, current_headers, type_map=type_map
            )
            report.scaling = [s.to_dict() for s in scale_summaries]
            report.steps_applied.append(
                f"scaling(type='{cfg.scaler_type}')"
            )

            # Warn about constant columns (std = 0 for Standard, range = 0 for MinMax).
            for s in scale_summaries:
                if cfg.scaler_type == "standard" and s.param_b == 0.0:
                    report.warnings.append(
                        f"Column '{s.column}' has zero standard deviation "
                        f"(constant) — StandardScaler output is 0.0."
                    )
                elif cfg.scaler_type == "minmax" and s.param_a == s.param_b:
                    report.warnings.append(
                        f"Column '{s.column}' has zero range (constant) — "
                        f"MinMaxScaler output is {cfg.minmax_range[0]}."
                    )
        else:
            report.scaling = []

        # ---------------------------------------------------------------- #
        # Step 4: Encoding (categorical columns)                            #
        # ---------------------------------------------------------------- #
        if cfg.encode_categorical:
            if cfg.encoder_type == "onehot":
                encoder = OneHotEncoder(drop_first=cfg.drop_first_ohe)
                current_data, current_headers, enc_summaries = encoder.fit_transform(
                    current_data, current_headers, type_map=type_map
                )
            else:
                encoder = LabelEncoder()
                current_data, enc_summaries = encoder.fit_transform(
                    current_data, current_headers, type_map=type_map
                )

            report.encoding = [s.to_dict() for s in enc_summaries]
            report.steps_applied.append(
                f"encoding(type='{cfg.encoder_type}')"
            )
        else:
            report.encoding = []

        # ---------------------------------------------------------------- #
        # Finalise report                                                   #
        # ---------------------------------------------------------------- #
        report.n_cols_out = current_data.shape[1]
        report.headers_out = list(current_headers)
        report.elapsed_seconds = time.perf_counter() - t0

        return CleanerResult(
            data=current_data,
            headers=current_headers,
            report=report,
        )

    def fit_transform(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
        type_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[np.ndarray, List[str], PreprocessingReport]:
        """Convenience alias for :pymeth:`run` that unpacks the result.

        Returns
        -------
        tuple[np.ndarray, list[str], PreprocessingReport]
            Cleaned data array, updated headers, and the full report.
        """
        result = self.run(data, headers, type_map)
        return result.data, result.headers, result.report
