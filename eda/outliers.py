"""
glassbox.eda.outliers
=====================

**IQR-based Outlier Detection** — flags and optionally caps data points
that fall beyond the standard fences.

Method
------
For each numeric column the module computes:

.. math::

    Q_1 = P_{25}, \\quad Q_3 = P_{75}, \\quad IQR = Q_3 - Q_1

A data point *x* is considered an **outlier** if:

.. math::

    x < Q_1 - k \\cdot IQR \\quad \\text{or} \\quad x > Q_3 + k \\cdot IQR

where *k* = 1.5 (Tukey's default) is configurable.

Capabilities
~~~~~~~~~~~~
* **Flag** — non-destructively marks outlier indices per column.
* **Cap / Winsorise** — clips outliers to the fence boundaries.
* **Report** — returns a per-column JSON-safe summary (counts, %, bounds).

Public API
----------
OutlierDetector
    Instantiate with optional *k* factor, then call :pymeth:`detect` or
    :pymeth:`cap`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from glassbox.eda.math_utils import manual_percentile


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class OutlierReport:
    """Outlier summary for a single column.

    Attributes
    ----------
    name : str
        Column name.
    q1 : float
        25th percentile.
    q3 : float
        75th percentile.
    iqr : float
        Interquartile range.
    lower_fence : float
        Q1 − k × IQR.
    upper_fence : float
        Q3 + k × IQR.
    n_outliers_low : int
        Count of values below the lower fence.
    n_outliers_high : int
        Count of values above the upper fence.
    n_total : int
        Total non-NaN values in the column.
    outlier_indices : list[int]
        Row indices (within the non-NaN subset) of outlier values.
    outlier_pct : float
        Percentage of values that are outliers.
    """
    name: str
    q1: float = 0.0
    q3: float = 0.0
    iqr: float = 0.0
    lower_fence: float = 0.0
    upper_fence: float = 0.0
    n_outliers_low: int = 0
    n_outliers_high: int = 0
    n_total: int = 0
    outlier_indices: List[int] = field(default_factory=list)
    outlier_pct: float = 0.0

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dictionary."""
        return {
            "name": self.name,
            "q1": round(self.q1, 6),
            "q3": round(self.q3, 6),
            "iqr": round(self.iqr, 6),
            "lower_fence": round(self.lower_fence, 6),
            "upper_fence": round(self.upper_fence, 6),
            "n_outliers_low": self.n_outliers_low,
            "n_outliers_high": self.n_outliers_high,
            "n_outliers_total": self.n_outliers_low + self.n_outliers_high,
            "n_total": self.n_total,
            "outlier_pct": round(self.outlier_pct, 4),
            "outlier_indices": self.outlier_indices,
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class OutlierDetector:
    """IQR-based outlier detection and capping.

    Parameters
    ----------
    k : float, optional
        Fence multiplier.  Default ``1.5`` (Tukey's rule).
        Use ``3.0`` for "extreme" outliers only.

    Examples
    --------
    >>> import numpy as np
    >>> from glassbox.eda.outliers import OutlierDetector
    >>> det = OutlierDetector(k=1.5)
    >>> data = np.array([[1, 2], [3, 4], [5, 100], [7, 8]], dtype=float)
    >>> reports = det.detect(data, headers=["a", "b"])
    >>> reports[1].n_outliers_high
    1
    """

    def __init__(self, k: float = 1.5):
        if k <= 0:
            raise ValueError("k must be positive.")
        self.k = k

    # ------------------------------------------------------------------
    # Detection (non-destructive)
    # ------------------------------------------------------------------
    def detect(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
        numeric_indices: Optional[List[int]] = None,
    ) -> List[OutlierReport]:
        """Flag outliers for each numeric column.

        Parameters
        ----------
        data : np.ndarray
            2-D array.
        headers : list[str], optional
            Column names.
        numeric_indices : list[int], optional
            Columns to inspect.  When *None*, all float-castable columns
            are used.

        Returns
        -------
        list[OutlierReport]
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n_rows, n_cols = data.shape
        if headers is None:
            headers = [f"col_{i}" for i in range(n_cols)]
        if numeric_indices is None:
            numeric_indices = self._detect_numeric(data, n_cols)

        reports: List[OutlierReport] = []
        for idx in numeric_indices:
            col = data[:, idx].astype(np.float64)
            reports.append(self._analyze_column(col, headers[idx]))
        return reports

    # ------------------------------------------------------------------
    # Capping / Winsorisation
    # ------------------------------------------------------------------
    def cap(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
        numeric_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Return a **copy** of *data* with outliers clipped to fences.

        Parameters are identical to :pymeth:`detect`.  The original array
        is never modified (non-destructive).

        Returns
        -------
        np.ndarray
            Copy of *data* with capped values.
        """
        capped = data.copy().astype(np.float64)
        if capped.ndim == 1:
            capped = capped.reshape(-1, 1)
        n_rows, n_cols = capped.shape
        if headers is None:
            headers = [f"col_{i}" for i in range(n_cols)]
        if numeric_indices is None:
            numeric_indices = self._detect_numeric(data, n_cols)

        for idx in numeric_indices:
            col = capped[:, idx]
            mask_valid = ~np.isnan(col)
            clean = col[mask_valid]
            if clean.size == 0:
                continue

            q1 = manual_percentile(clean, 25)
            q3 = manual_percentile(clean, 75)
            iqr = q3 - q1
            lo = q1 - self.k * iqr
            hi = q3 + self.k * iqr

            # Clip in-place on the copy.
            col[mask_valid & (col < lo)] = lo
            col[mask_valid & (col > hi)] = hi
            capped[:, idx] = col

        return capped

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _analyze_column(self, col: np.ndarray, name: str) -> OutlierReport:
        """Build an OutlierReport for a single column."""
        mask_valid = ~np.isnan(col)
        clean = col[mask_valid]
        n = clean.size
        if n == 0:
            return OutlierReport(name=name)

        q1 = manual_percentile(clean, 25)
        q3 = manual_percentile(clean, 75)
        iqr = q3 - q1
        lo = q1 - self.k * iqr
        hi = q3 + self.k * iqr

        # Identify outlier positions (indices in the *original* array).
        valid_indices = np.where(mask_valid)[0]
        is_low = clean < lo
        is_high = clean > hi
        outlier_mask = is_low | is_high
        outlier_idx = valid_indices[outlier_mask].tolist()

        n_low = int(np.sum(is_low))
        n_high = int(np.sum(is_high))
        total_outliers = n_low + n_high
        pct = (total_outliers / n) * 100.0 if n > 0 else 0.0

        return OutlierReport(
            name=name,
            q1=q1,
            q3=q3,
            iqr=iqr,
            lower_fence=lo,
            upper_fence=hi,
            n_outliers_low=n_low,
            n_outliers_high=n_high,
            n_total=n,
            outlier_indices=outlier_idx,
            outlier_pct=pct,
        )

    @staticmethod
    def _detect_numeric(data: np.ndarray, n_cols: int) -> List[int]:
        indices: List[int] = []
        for i in range(n_cols):
            try:
                data[:, i].astype(np.float64)
                indices.append(i)
            except (ValueError, TypeError):
                continue
        return indices
