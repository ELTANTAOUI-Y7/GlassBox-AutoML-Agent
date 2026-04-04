"""
glassbox.eda.correlation
========================

**Pearson Correlation Matrix** — built entirely from scratch.

The Pearson correlation coefficient between two variables *X* and *Y* is:

.. math::

    r_{xy} = \\frac{\\sum_{i=1}^{n}(x_i - \\bar{x})(y_i - \\bar{y})}
             {\\sqrt{\\sum_{i=1}^{n}(x_i - \\bar{x})^2} \\;
              \\sqrt{\\sum_{i=1}^{n}(y_i - \\bar{y})^2}}

The module additionally provides:

* **Collinearity detection** — flags pairs with |r| above a configurable
  threshold (default 0.90).
* **Highly-correlated feature report** — returns a ranked list ready for
  inclusion in the Inspector JSON.

All maths rely only on NumPy element-wise operations — no ``np.corrcoef``.

Public API
----------
CorrelationAnalyzer
    Compute the full correlation matrix and extract collinearity warnings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class CorrelationPair:
    """A pair of columns and their Pearson *r* value."""
    col_a: str
    col_b: str
    r: float

    def to_dict(self) -> dict:
        return {"col_a": self.col_a, "col_b": self.col_b, "r": round(self.r, 6)}


@dataclass
class CorrelationResult:
    """Full correlation analysis output.

    Attributes
    ----------
    matrix : np.ndarray
        Square correlation matrix of shape ``(k, k)`` where *k* is the
        number of numeric columns analysed.
    column_names : list[str]
        Column names corresponding to each axis of *matrix*.
    high_pairs : list[CorrelationPair]
        Pairs whose ``|r| >= threshold`` (sorted descending by |r|).
    """
    matrix: np.ndarray
    column_names: List[str]
    high_pairs: List[CorrelationPair]

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dictionary."""
        # Convert matrix to a nested list of rounded floats.
        mat_list = [
            [round(float(self.matrix[i, j]), 6) for j in range(self.matrix.shape[1])]
            for i in range(self.matrix.shape[0])
        ]
        return {
            "column_names": self.column_names,
            "matrix": mat_list,
            "high_correlation_pairs": [p.to_dict() for p in self.high_pairs],
        }


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------
class CorrelationAnalyzer:
    """Compute a scratch-built Pearson correlation matrix.

    Parameters
    ----------
    threshold : float, optional
        Absolute *r* above which a pair is flagged as highly correlated.
        Default ``0.90``.

    Examples
    --------
    >>> import numpy as np
    >>> from glassbox.eda.correlation import CorrelationAnalyzer
    >>> ca = CorrelationAnalyzer(threshold=0.8)
    >>> data = np.random.randn(100, 3)
    >>> res = ca.analyze(data, ["a", "b", "c"])
    >>> res.matrix.shape
    (3, 3)
    """

    def __init__(self, threshold: float = 0.90):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1].")
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def analyze(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
        numeric_indices: Optional[List[int]] = None,
    ) -> CorrelationResult:
        """Compute the Pearson correlation matrix.

        Parameters
        ----------
        data : np.ndarray
            2-D array ``(n_rows, n_cols)``.
        headers : list[str], optional
            Column names.  Auto-generated when *None*.
        numeric_indices : list[int], optional
            Column indices to include.  When *None*, all columns that can
            be cast to float are used.

        Returns
        -------
        CorrelationResult
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n_rows, n_cols = data.shape
        if headers is None:
            headers = [f"col_{i}" for i in range(n_cols)]

        # Select numeric columns.
        if numeric_indices is None:
            numeric_indices = self._detect_numeric(data, n_cols)
        if len(numeric_indices) == 0:
            return CorrelationResult(
                matrix=np.array([]).reshape(0, 0),
                column_names=[],
                high_pairs=[],
            )

        sub = data[:, numeric_indices].astype(np.float64)
        col_names = [headers[i] for i in numeric_indices]
        k = sub.shape[1]

        # Build correlation matrix.
        corr = self._pearson_matrix(sub, k)

        # Detect high-correlation pairs.
        high_pairs = self._flag_high(corr, col_names, k)

        return CorrelationResult(
            matrix=corr, column_names=col_names, high_pairs=high_pairs
        )

    # ------------------------------------------------------------------
    # Core math
    # ------------------------------------------------------------------
    @staticmethod
    def _pearson_matrix(sub: np.ndarray, k: int) -> np.ndarray:
        """Build the full *k×k* Pearson correlation matrix.

        Uses pairwise NaN-aware computation: for each pair (i, j), only rows
        where *both* values are non-NaN are used.
        """
        corr = np.ones((k, k), dtype=np.float64)
        for i in range(k):
            for j in range(i + 1, k):
                r = CorrelationAnalyzer._pearson_r(sub[:, i], sub[:, j])
                corr[i, j] = r
                corr[j, i] = r
        return corr

    @staticmethod
    def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
        """Compute Pearson *r* for two 1-D float arrays (NaN-aware).

        .. math::
            r = \\frac{\\sum (x_i - \\bar x)(y_i - \\bar y)}
                {\\sqrt{\\sum (x_i-\\bar x)^2} \\sqrt{\\sum (y_i-\\bar y)^2}}
        """
        valid = ~(np.isnan(x) | np.isnan(y))
        xc = x[valid]
        yc = y[valid]
        n = xc.size
        if n < 2:
            return 0.0
        mean_x = np.sum(xc) / n
        mean_y = np.sum(yc) / n
        dx = xc - mean_x
        dy = yc - mean_y
        num = np.sum(dx * dy)
        den = np.sqrt(np.sum(dx ** 2)) * np.sqrt(np.sum(dy ** 2))
        if den == 0.0:
            return 0.0
        return float(num / den)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _flag_high(
        self, corr: np.ndarray, names: List[str], k: int
    ) -> List[CorrelationPair]:
        """Return pairs with |r| ≥ threshold, sorted by |r| descending."""
        pairs: List[CorrelationPair] = []
        for i in range(k):
            for j in range(i + 1, k):
                r = corr[i, j]
                if abs(r) >= self.threshold:
                    pairs.append(CorrelationPair(names[i], names[j], r))
        pairs.sort(key=lambda p: abs(p.r), reverse=True)
        return pairs

    @staticmethod
    def _detect_numeric(data: np.ndarray, n_cols: int) -> List[int]:
        """Return indices of columns that can be cast to float64."""
        indices: List[int] = []
        for i in range(n_cols):
            try:
                data[:, i].astype(np.float64)
                indices.append(i)
            except (ValueError, TypeError):
                continue
        return indices
