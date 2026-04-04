"""
glassbox.eda.stats
==================

**Statistical Profiler** — computes a comprehensive statistical profile for
every numeric column in a dataset.

All statistics are computed from scratch using the primitives in
:mod:`glassbox.eda.math_utils`.  No ``np.mean``, ``np.std``, or
``scipy.stats`` calls are used.

Computed Statistics
-------------------
For each numeric column the profiler returns:

============  =============================================
Statistic     Description
============  =============================================
count         Number of non-NaN values
missing       Number of NaN / None values
mean          Arithmetic mean
median        Median (middle value)
mode          Most frequent value
std           Sample standard deviation (ddof = 1)
variance      Sample variance (ddof = 1)
min           Minimum value
max           Maximum value
range         max − min
q1            25th percentile
q3            75th percentile
iqr           Interquartile range (Q3 − Q1)
skewness      Adjusted Fisher–Pearson skewness
kurtosis      Excess (Fisher) kurtosis
============  =============================================

Public API
----------
StatProfiler
    Instantiate, then call :pymeth:`profile` with data + headers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from glassbox.eda.math_utils import (
    manual_kurtosis,
    manual_mean,
    manual_median,
    manual_mode,
    manual_percentile,
    manual_skewness,
    manual_std,
    manual_variance,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ColumnStats:
    """Statistical summary for a single numeric column.

    All fields default to ``None`` when the statistic cannot be computed
    (e.g. too few data points for skewness).
    """
    name: str
    count: int = 0
    missing: int = 0
    mean: Optional[float] = None
    median: Optional[float] = None
    mode: Optional[Any] = None
    std: Optional[float] = None
    variance: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    range: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    iqr: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dictionary."""
        return {
            "name": self.name,
            "count": self.count,
            "missing": self.missing,
            "mean": self._f(self.mean),
            "median": self._f(self.median),
            "mode": self._safe(self.mode),
            "std": self._f(self.std),
            "variance": self._f(self.variance),
            "min": self._f(self.min),
            "max": self._f(self.max),
            "range": self._f(self.range),
            "q1": self._f(self.q1),
            "q3": self._f(self.q3),
            "iqr": self._f(self.iqr),
            "skewness": self._f(self.skewness),
            "kurtosis": self._f(self.kurtosis),
        }

    @staticmethod
    def _f(v):
        if v is None:
            return None
        return round(float(v), 6)

    @staticmethod
    def _safe(v):
        if v is None:
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return round(float(v), 6)
        return v


# ---------------------------------------------------------------------------
# StatProfiler
# ---------------------------------------------------------------------------
class StatProfiler:
    """Generate descriptive statistics for every numeric column.

    Parameters
    ----------
    numeric_types : set[str], optional
        Set of ``ColumnTypeInfo.inferred_type`` values that should be
        profiled.  Defaults to ``{"numerical"}``.

    Examples
    --------
    >>> import numpy as np
    >>> from glassbox.eda.stats import StatProfiler
    >>> profiler = StatProfiler()
    >>> data = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=float)
    >>> results = profiler.profile(data, ["a", "b"])
    >>> results[0].mean
    2.5
    """

    def __init__(self, numeric_types: Optional[set] = None):
        self.numeric_types = numeric_types or {"numerical"}

    def profile(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
        type_map: Optional[Dict[str, str]] = None,
    ) -> List[ColumnStats]:
        """Compute statistics for all numeric columns.

        Parameters
        ----------
        data : np.ndarray
            2-D array of shape ``(n_rows, n_cols)``.
        headers : list[str], optional
            Column names.  Auto-generated when *None*.
        type_map : dict[str, str], optional
            Mapping ``{column_name: inferred_type}``.  When provided only
            columns whose type is in ``self.numeric_types`` are profiled.
            When *None*, every column that can be cast to float is profiled.

        Returns
        -------
        list[ColumnStats]
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n_rows, n_cols = data.shape
        if headers is None:
            headers = [f"col_{i}" for i in range(n_cols)]

        results: List[ColumnStats] = []
        for i in range(n_cols):
            name = headers[i]
            # Skip non-numeric columns when type_map is available.
            if type_map is not None and type_map.get(name) not in self.numeric_types:
                continue

            col = data[:, i]
            # Attempt numeric conversion.
            try:
                col_f = col.astype(np.float64)
            except (ValueError, TypeError):
                continue  # Not numeric — skip.

            results.append(self._compute(col_f, name))
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _compute(col: np.ndarray, name: str) -> ColumnStats:
        """Compute all statistics for a single float64 column."""
        mask = np.isnan(col)
        n_missing = int(np.sum(mask))
        clean = col[~mask]
        n = clean.size

        cs = ColumnStats(name=name, count=n, missing=n_missing)
        if n == 0:
            return cs

        cs.mean = manual_mean(clean)
        cs.median = manual_median(clean)
        try:
            cs.mode = manual_mode(clean)
        except ValueError:
            cs.mode = None

        cs.min = float(np.min(clean))
        cs.max = float(np.max(clean))
        cs.range = cs.max - cs.min

        cs.q1 = manual_percentile(clean, 25)
        cs.q3 = manual_percentile(clean, 75)
        cs.iqr = cs.q3 - cs.q1

        if n >= 2:
            cs.std = manual_std(clean, ddof=1)
            cs.variance = manual_variance(clean, ddof=1)

        if n >= 3:
            try:
                cs.skewness = manual_skewness(clean)
            except ValueError:
                cs.skewness = None

        if n >= 4:
            try:
                cs.kurtosis = manual_kurtosis(clean)
            except ValueError:
                cs.kurtosis = None

        return cs
