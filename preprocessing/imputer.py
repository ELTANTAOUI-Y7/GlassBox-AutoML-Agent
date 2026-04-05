"""
glassbox.preprocessing.imputer
===============================

**Simple Imputer** — fills missing values using column statistics.

Strategies
----------
``"mean"``
    Arithmetic mean for numerical columns (falls back to mode for categorical).
    Uses :func:`glassbox.eda.math_utils.manual_mean` — implemented from scratch.
``"median"``
    Median for numerical columns (falls back to mode for categorical).
    Uses :func:`glassbox.eda.math_utils.manual_median` — implemented from scratch.
``"mode"``
    Most frequent value. Works for any column type.
    Uses :func:`glassbox.eda.math_utils.manual_mode` — implemented from scratch.
``"constant"``
    Fills every missing cell with a user-supplied ``fill_value``.

Design Goals
------------
- **Non-destructive**: ``transform()`` always returns a copy of the data.
- **NumPy-only**: no Pandas or Scikit-Learn.
- **EDA-aware**: accepts a ``type_map`` from :class:`glassbox.eda.auto_typer.AutoTyper`
  to choose the correct strategy per column type automatically.
- **Transformer pattern**: implements ``fit()``, ``transform()``, and
  ``fit_transform()`` for pipeline composability.

Public API
----------
SimpleImputer
    Main class.
ImputationSummary
    Per-column record of what fill value was applied and how many cells were filled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from glassbox.eda.auto_typer import AutoTyper
from glassbox.eda.math_utils import manual_mean, manual_median, manual_mode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_MISSING_STRINGS = {"", "nan", "none", "null", "na", "n/a"}


def _is_missing(val: Any) -> bool:
    """Return True if *val* should be treated as a missing value."""
    if val is None:
        return True
    try:
        if np.isnan(float(val)):
            return True
    except (ValueError, TypeError):
        pass
    if isinstance(val, str) and val.strip().lower() in _MISSING_STRINGS:
        return True
    return False


def _missing_mask(col: np.ndarray) -> np.ndarray:
    """Build a boolean mask identifying missing entries in *col*."""
    if col.dtype.kind == "f":
        return np.isnan(col)
    # Object / string arrays — check element-by-element.
    return np.array([_is_missing(v) for v in col], dtype=bool)


def _safe_json(val: Any) -> Any:
    """Convert NumPy scalars to plain Python types for JSON serialisation."""
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return round(float(val), 6)
    if isinstance(val, np.bool_):
        return bool(val)
    return val


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ImputationSummary:
    """Records the imputation applied to a single column.

    Attributes
    ----------
    column : str
        Column name.
    strategy : str
        Strategy that was used (``"mean"``, ``"median"``, ``"mode"``,
        ``"constant"``).
    fill_value : Any
        The value used to fill missing cells.
    n_filled : int
        Number of cells that were actually filled.
    """

    column: str
    strategy: str
    fill_value: Any
    n_filled: int

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dictionary."""
        return {
            "column": self.column,
            "strategy": self.strategy,
            "fill_value": _safe_json(self.fill_value),
            "n_filled": self.n_filled,
        }


# ---------------------------------------------------------------------------
# SimpleImputer
# ---------------------------------------------------------------------------
class SimpleImputer:
    """Fill missing values in every column using statistics computed on training data.

    Parameters
    ----------
    strategy : str, optional
        How to compute fill values.  One of ``"mean"``, ``"median"``,
        ``"mode"``, or ``"constant"``.  Default ``"mean"``.
    fill_value : Any, optional
        Value used when ``strategy="constant"``.  Ignored otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from glassbox.preprocessing.imputer import SimpleImputer
    >>> data = np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, 4.0]])
    >>> imputer = SimpleImputer(strategy="mean")
    >>> out, summaries = imputer.fit_transform(data, headers=["a", "b"])
    >>> out
    array([[1. , 3. ],
           [2. , 2. ],
           [3. , 4. ]])
    """

    _VALID_STRATEGIES = {"mean", "median", "mode", "constant"}

    def __init__(self, strategy: str = "mean", fill_value: Any = None) -> None:
        if strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {sorted(self._VALID_STRATEGIES)}"
            )
        self.strategy = strategy
        self.fill_value = fill_value

        # State set during fit()
        self._fill_values: Dict[str, Any] = {}
        self._headers: Optional[List[str]] = None
        self._type_map: Dict[str, str] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
        type_map: Optional[Dict[str, str]] = None,
    ) -> "SimpleImputer":
        """Compute per-column fill values from *data*.

        Parameters
        ----------
        data : np.ndarray
            2-D training data of shape ``(n_rows, n_cols)``.
        headers : list[str], optional
            Column names.  Auto-generated as ``col_0, col_1, …`` when *None*.
        type_map : dict[str, str], optional
            ``{col_name: inferred_type}`` produced by
            :class:`glassbox.eda.auto_typer.AutoTyper`.  When *None*, the
            AutoTyper is run automatically on *data*.

        Returns
        -------
        self
        """
        data, headers = _validate_input(data, headers)
        self._headers = headers

        if type_map is None:
            typer = AutoTyper()
            type_infos = typer.detect(data, headers)
            type_map = {ti.name: ti.inferred_type for ti in type_infos}
        self._type_map = type_map

        self._fill_values = {}
        for i, col_name in enumerate(headers):
            col = data[:, i]
            col_type = type_map.get(col_name, "numerical")
            self._fill_values[col_name] = self._compute_fill(col, col_type)

        self._fitted = True
        return self

    def transform(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[ImputationSummary]]:
        """Fill missing values using the statistics from ``fit()``.

        Parameters
        ----------
        data : np.ndarray
            2-D data to impute.  Must have the same number of columns as the
            training data seen during ``fit()``.
        headers : list[str], optional
            Column names.  Must match those seen during ``fit()`` when
            provided.

        Returns
        -------
        tuple[np.ndarray, list[ImputationSummary]]
            A *copy* of the data with missing values filled, and a per-column
            summary of what was applied.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        data, headers = _validate_input(data, headers)

        result = data.copy()
        summaries: List[ImputationSummary] = []

        for i, col_name in enumerate(headers):
            fill = self._fill_values.get(col_name)
            col = result[:, i]
            mask = _missing_mask(col)
            n_filled = int(np.sum(mask))

            if n_filled > 0 and fill is not None:
                result[mask, i] = fill

            summaries.append(ImputationSummary(
                column=col_name,
                strategy=self.strategy,
                fill_value=fill,
                n_filled=n_filled,
            ))

        return result, summaries

    def fit_transform(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
        type_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[np.ndarray, List[ImputationSummary]]:
        """Fit on *data* then transform it in one step.

        Parameters
        ----------
        data : np.ndarray
            Training data to fit on and transform.
        headers : list[str], optional
            Column names.
        type_map : dict[str, str], optional
            Pre-computed type map from the EDA AutoTyper.

        Returns
        -------
        tuple[np.ndarray, list[ImputationSummary]]
        """
        return self.fit(data, headers, type_map).transform(data, headers)

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _compute_fill(self, col: np.ndarray, col_type: str) -> Any:
        """Compute the fill value for one column vector.

        Uses :mod:`glassbox.eda.math_utils` primitives so the logic is fully
        transparent and consistent with the EDA module.
        """
        if self.strategy == "constant":
            return self.fill_value

        is_numerical = col_type == "numerical"

        # Build a clean array (no missing values).
        if is_numerical:
            try:
                col_f = col.astype(np.float64)
                clean = col_f[~np.isnan(col_f)]
            except (ValueError, TypeError):
                clean = np.array([], dtype=np.float64)
        else:
            mask = _missing_mask(col)
            clean = col[~mask]

        if clean.size == 0:
            return None  # All-missing column — cannot compute fill value.

        if self.strategy == "mean":
            if is_numerical and clean.size > 0:
                return manual_mean(clean.astype(np.float64))
            # Fall back to mode for categorical / boolean.
            return manual_mode(clean)

        if self.strategy == "median":
            if is_numerical and clean.size > 0:
                return manual_median(clean.astype(np.float64))
            # Fall back to mode for categorical / boolean.
            return manual_mode(clean)

        if self.strategy == "mode":
            return manual_mode(clean)

        return None  # Should never reach here given __init__ validation.


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------

def _validate_input(
    data: np.ndarray,
    headers: Optional[List[str]],
) -> Tuple[np.ndarray, List[str]]:
    """Ensure *data* is 2-D and *headers* is a list of the correct length."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_cols = data.shape[1]
    if headers is None:
        headers = [f"col_{i}" for i in range(n_cols)]
    if len(headers) != n_cols:
        raise ValueError(
            f"Length of headers ({len(headers)}) does not match "
            f"number of columns ({n_cols})."
        )
    return data, headers
