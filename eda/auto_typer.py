"""
glassbox.eda.auto_typer
=======================

Automatic column-type detection for raw datasets.

The **AutoTyper** inspects every column of a 2-D NumPy array (or
list-of-lists) and classifies it into one of three semantic types:

* **Numerical** — continuous or discrete numbers.
* **Categorical** — string / object labels with more than 2 unique values
  (or numeric columns with very low cardinality).
* **Boolean** — columns that contain exactly 2 unique non-NaN values
  (including literal ``True``/``False``, ``0``/``1``, ``"yes"``/``"no"``).

Design Goals
------------
- **Non-destructive**: the original data is never modified.
- **NumPy-only**: no Pandas or Scikit-Learn.
- **Configurable**: cardinality thresholds can be tuned.

Public API
----------
AutoTyper
    Callable class.  Instantiate with optional config, then call
    :pymethod:`detect` passing column data and headers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BOOL_TRUTHY = {"true", "yes", "1", "t", "y", "on"}
_BOOL_FALSY = {"false", "no", "0", "f", "n", "off"}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ColumnTypeInfo:
    """Stores the inferred type metadata for a single column.

    Attributes
    ----------
    name : str
        Column header / name.
    inferred_type : str
        One of ``"numerical"``, ``"categorical"``, ``"boolean"``.
    dtype : str
        The raw NumPy dtype string (e.g. ``"float64"``, ``"<U10"``).
    n_unique : int
        Number of unique non-NaN values.
    n_missing : int
        Count of NaN / None values.
    sample_values : list
        Up to 5 example values from the column.
    """
    name: str
    inferred_type: str
    dtype: str
    n_unique: int
    n_missing: int
    sample_values: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "name": self.name,
            "inferred_type": self.inferred_type,
            "dtype": self.dtype,
            "n_unique": self.n_unique,
            "n_missing": self.n_missing,
            "sample_values": [_safe_json(v) for v in self.sample_values],
        }


def _safe_json(val):
    """Convert NumPy scalars to Python builtins for JSON serialisation."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


# ---------------------------------------------------------------------------
# AutoTyper
# ---------------------------------------------------------------------------
class AutoTyper:
    """Infer the semantic type of every column in a dataset.

    Parameters
    ----------
    categorical_cardinality_ratio : float, optional
        If a numeric column has ``n_unique / n_total`` *below* this ratio
        **and** ``n_unique <= categorical_max_unique``, it is re-classified
        as categorical.  Default ``0.05`` (5 %).
    categorical_max_unique : int, optional
        Absolute cap on unique values for the cardinality heuristic above.
        Default ``20``.
    bool_values : set[str] | None, optional
        Extra string literals to recognise as boolean (case-insensitive).
        Merged with the built-in set.

    Examples
    --------
    >>> import numpy as np
    >>> typer = AutoTyper()
    >>> data = np.array([[1, "a", True], [2, "b", False], [3, "a", True]])
    >>> headers = ["id", "label", "flag"]
    >>> result = typer.detect(data, headers)
    >>> [c.inferred_type for c in result]
    ['numerical', 'categorical', 'boolean']
    """

    def __init__(
        self,
        categorical_cardinality_ratio: float = 0.05,
        categorical_max_unique: int = 20,
        bool_values: Optional[set] = None,
    ) -> None:
        self.categorical_cardinality_ratio = categorical_cardinality_ratio
        self.categorical_max_unique = categorical_max_unique
        self._bool_truthy = _BOOL_TRUTHY | (bool_values or set())
        self._bool_falsy = _BOOL_FALSY | (bool_values or set())

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def detect(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
    ) -> List[ColumnTypeInfo]:
        """Run type inference on every column of *data*.

        Parameters
        ----------
        data : np.ndarray
            2-D array of shape ``(n_rows, n_cols)``.  May be a structured
            array, an object array, or a regular numeric array.
        headers : list[str], optional
            Column names.  If *None*, columns are named ``col_0``, ``col_1``,
            etc.

        Returns
        -------
        list[ColumnTypeInfo]
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n_rows, n_cols = data.shape
        if headers is None:
            headers = [f"col_{i}" for i in range(n_cols)]

        results: List[ColumnTypeInfo] = []
        for col_idx in range(n_cols):
            col = data[:, col_idx]
            info = self._classify_column(col, headers[col_idx], n_rows)
            results.append(info)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _classify_column(
        self, col: np.ndarray, name: str, n_rows: int
    ) -> ColumnTypeInfo:
        """Classify a single column vector."""
        # --- Attempt to coerce to float for numeric check ----------------
        numeric_col, is_numeric = self._try_numeric(col)

        # --- Count missing -----------------------------------------------
        if is_numeric:
            mask_missing = np.isnan(numeric_col)
        else:
            mask_missing = np.array(
                [(v is None) or (str(v).strip().lower() in ("", "nan", "none", "null", "na", "n/a"))
                 for v in col],
                dtype=bool,
            )
        n_missing = int(np.sum(mask_missing))
        clean = col[~mask_missing]

        # --- Unique values -----------------------------------------------
        n_unique = int(np.unique(clean).size)

        # --- Sample values -----------------------------------------------
        sample_vals = list(clean[:5]) if clean.size > 0 else []

        # --- Type inference logic ----------------------------------------
        inferred = self._infer_type(
            clean, is_numeric, numeric_col, mask_missing, n_unique, n_rows
        )

        return ColumnTypeInfo(
            name=name,
            inferred_type=inferred,
            dtype=str(col.dtype),
            n_unique=n_unique,
            n_missing=n_missing,
            sample_values=sample_vals,
        )

    def _infer_type(
        self,
        clean: np.ndarray,
        is_numeric: bool,
        numeric_col: Optional[np.ndarray],
        mask_missing: np.ndarray,
        n_unique: int,
        n_rows: int,
    ) -> str:
        """Core heuristic engine."""
        # 1. Boolean check first (highest priority).
        if n_unique == 2:
            if self._looks_boolean(clean):
                return "boolean"

        # Also catch single-value boolean-ish columns (e.g. all True).
        if n_unique == 1:
            if self._looks_boolean(clean):
                return "boolean"

        # 2. Numeric columns.
        if is_numeric:
            # Low-cardinality numeric → categorical.
            n_valid = int(np.sum(~mask_missing))
            if n_valid > 0:
                ratio = n_unique / n_valid
                if (
                    ratio <= self.categorical_cardinality_ratio
                    and n_unique <= self.categorical_max_unique
                ):
                    return "categorical"
            return "numerical"

        # 3. Non-numeric → categorical.
        return "categorical"

    def _looks_boolean(self, clean: np.ndarray) -> bool:
        """Return True if the unique values look like boolean indicators."""
        uniq = set(str(v).strip().lower() for v in np.unique(clean))
        # Direct bool dtype
        if uniq <= {"true", "false"}:
            return True
        # String representations
        if uniq <= (self._bool_truthy | self._bool_falsy):
            return True
        # Numeric 0/1
        if uniq <= {"0", "1", "0.0", "1.0"}:
            return True
        return False

    @staticmethod
    def _try_numeric(col: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
        """Try to cast *col* to float64.  Return (array | None, success)."""
        if col.dtype.kind in ("f", "i", "u"):
            return col.astype(np.float64), True
        try:
            numeric = np.array(col, dtype=np.float64)
            return numeric, True
        except (ValueError, TypeError):
            return None, False
