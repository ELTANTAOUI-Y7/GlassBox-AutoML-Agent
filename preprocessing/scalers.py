"""
glassbox.preprocessing.scalers
================================

**MinMaxScaler** and **StandardScaler** — normalise numerical features.

Both scalers
------------
* Only operate on columns whose ``inferred_type`` is ``"numerical"`` (detected
  automatically via :class:`glassbox.eda.auto_typer.AutoTyper` or supplied via
  a pre-computed ``type_map``).
* Leave non-numerical columns untouched.
* Return a **copy** of the data — the original array is never modified.
* Use :mod:`glassbox.eda.math_utils` primitives (``manual_mean``,
  ``manual_std``) so the math is fully transparent.
* Implement ``fit()``, ``transform()``, and ``fit_transform()`` following the
  Transformer pattern.

MinMaxScaler
------------
Scales each numerical feature to a fixed range ``[feature_range[0],
feature_range[1]]`` (default ``[0, 1]``):

.. math::

    x_{\\text{scaled}} = \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}}
    \\times (\\text{high} - \\text{low}) + \\text{low}

StandardScaler
--------------
Standardises each numerical feature to zero mean and unit variance:

.. math::

    x_{\\text{scaled}} = \\frac{x - \\bar{x}}{\\sigma}

where :math:`\\bar{x}` is the sample mean and :math:`\\sigma` is the sample
standard deviation (``ddof = 1``).

Public API
----------
MinMaxScaler
StandardScaler
ScalerSummary  — per-column record of learned parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from glassbox.eda.auto_typer import AutoTyper
from glassbox.eda.math_utils import manual_mean, manual_std


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ScalerSummary:
    """Records the parameters learned for a single scaled column.

    Attributes
    ----------
    column : str
        Column name.
    scaler_type : str
        ``"minmax"`` or ``"standard"``.
    param_a : float
        For MinMax: ``x_min``.  For Standard: learned mean (``mu``).
    param_b : float
        For MinMax: ``x_max``.  For Standard: learned std (``sigma``).
    """

    column: str
    scaler_type: str
    param_a: float  # min  (MinMax)  |  mean  (Standard)
    param_b: float  # max  (MinMax)  |  std   (Standard)

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dictionary."""
        if self.scaler_type == "minmax":
            return {
                "column": self.column,
                "scaler_type": self.scaler_type,
                "x_min": round(self.param_a, 6),
                "x_max": round(self.param_b, 6),
            }
        return {
            "column": self.column,
            "scaler_type": self.scaler_type,
            "mean": round(self.param_a, 6),
            "std": round(self.param_b, 6),
        }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _validate_input(
    data: np.ndarray,
    headers: Optional[List[str]],
) -> Tuple[np.ndarray, List[str]]:
    """Ensure *data* is 2-D and *headers* has the right length."""
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


def _get_type_map(
    data: np.ndarray,
    headers: List[str],
    type_map: Optional[Dict[str, str]],
) -> Dict[str, str]:
    """Return *type_map*, auto-detecting it via AutoTyper when *None*."""
    if type_map is not None:
        return type_map
    typer = AutoTyper()
    type_infos = typer.detect(data, headers)
    return {ti.name: ti.inferred_type for ti in type_infos}


def _col_to_float(col: np.ndarray) -> np.ndarray:
    """Cast *col* to float64, stripping NaN-aware missing values."""
    col_f = col.astype(np.float64)
    return col_f[~np.isnan(col_f)]


# ---------------------------------------------------------------------------
# MinMaxScaler
# ---------------------------------------------------------------------------
class MinMaxScaler:
    """Scale numerical features to a fixed range ``[low, high]``.

    .. math::

        x_{\\text{scaled}} = \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}}
        \\times (\\text{high} - \\text{low}) + \\text{low}

    When ``x_min == x_max`` (constant column) the scaler outputs ``low`` for
    all values to avoid division by zero.

    Parameters
    ----------
    feature_range : tuple[float, float], optional
        Desired output range ``(low, high)``.  Default ``(0.0, 1.0)``.

    Examples
    --------
    >>> import numpy as np
    >>> from glassbox.preprocessing.scalers import MinMaxScaler
    >>> data = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
    >>> scaler = MinMaxScaler()
    >>> out, summaries = scaler.fit_transform(data, headers=["a", "b"])
    >>> out
    array([[0. , 0. ],
           [0.5, 0.5],
           [1. , 1. ]])
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        low, high = feature_range
        if low >= high:
            raise ValueError(
                f"feature_range must satisfy low < high, got ({low}, {high})."
            )
        self.feature_range = feature_range

        # State set during fit()
        self._x_min: Dict[str, float] = {}
        self._x_max: Dict[str, float] = {}
        self._numerical_cols: List[str] = []
        self._fitted: bool = False

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
        type_map: Optional[Dict[str, str]] = None,
    ) -> "MinMaxScaler":
        """Learn per-column min and max from *data*.

        Parameters
        ----------
        data : np.ndarray
            2-D training data.
        headers : list[str], optional
            Column names.
        type_map : dict[str, str], optional
            ``{col_name: inferred_type}`` from :class:`glassbox.eda.auto_typer.AutoTyper`.
            Auto-detected when *None*.

        Returns
        -------
        self
        """
        data, headers = _validate_input(data, headers)
        type_map = _get_type_map(data, headers, type_map)

        self._x_min = {}
        self._x_max = {}
        self._numerical_cols = []

        for i, col_name in enumerate(headers):
            if type_map.get(col_name) != "numerical":
                continue
            clean = _col_to_float(data[:, i])
            if clean.size == 0:
                continue
            self._x_min[col_name] = float(np.min(clean))
            self._x_max[col_name] = float(np.max(clean))
            self._numerical_cols.append(col_name)

        self._fitted = True
        return self

    def transform(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[ScalerSummary]]:
        """Apply Min-Max scaling to numerical columns.

        Parameters
        ----------
        data : np.ndarray
            2-D data to scale.
        headers : list[str], optional
            Column names (must match those seen in ``fit()``).

        Returns
        -------
        tuple[np.ndarray, list[ScalerSummary]]
            A *copy* of *data* with numerical columns scaled, and a per-column
            summary of the parameters used.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        data, headers = _validate_input(data, headers)

        low, high = self.feature_range
        result = data.copy().astype(object)
        summaries: List[ScalerSummary] = []

        for i, col_name in enumerate(headers):
            if col_name not in self._x_min:
                continue
            x_min = self._x_min[col_name]
            x_max = self._x_max[col_name]

            col_f = data[:, i].astype(np.float64)
            denom = x_max - x_min

            if denom == 0.0:
                # Constant column — set everything to the lower bound.
                scaled = np.where(np.isnan(col_f), np.nan, float(low))
            else:
                # MinMax formula: (x - min) / (max - min) * (high - low) + low
                scaled = (col_f - x_min) / denom * (high - low) + low

            result[:, i] = scaled
            summaries.append(ScalerSummary(
                column=col_name,
                scaler_type="minmax",
                param_a=x_min,
                param_b=x_max,
            ))

        return result, summaries

    def fit_transform(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
        type_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[np.ndarray, List[ScalerSummary]]:
        """Fit on *data* then transform it in one step."""
        return self.fit(data, headers, type_map).transform(data, headers)


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------
class StandardScaler:
    """Standardise numerical features to zero mean and unit variance.

    .. math::

        x_{\\text{scaled}} = \\frac{x - \\bar{x}}{\\sigma}

    where :math:`\\bar{x}` is the sample mean and :math:`\\sigma` is the
    sample standard deviation (``ddof = 1``).

    Both statistics are computed using :func:`glassbox.eda.math_utils.manual_mean`
    and :func:`glassbox.eda.math_utils.manual_std` — implemented from scratch
    without ``np.mean`` or ``np.std``.

    When :math:`\\sigma = 0` (constant column) the scaler outputs ``0.0`` for
    all values to avoid division by zero.

    Examples
    --------
    >>> import numpy as np
    >>> from glassbox.preprocessing.scalers import StandardScaler
    >>> data = np.array([[2.0, 10.0], [4.0, 20.0], [6.0, 30.0]])
    >>> scaler = StandardScaler()
    >>> out, summaries = scaler.fit_transform(data, headers=["a", "b"])
    """

    def __init__(self) -> None:
        # State set during fit()
        self._mean: Dict[str, float] = {}
        self._std: Dict[str, float] = {}
        self._numerical_cols: List[str] = []
        self._fitted: bool = False

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
        type_map: Optional[Dict[str, str]] = None,
    ) -> "StandardScaler":
        """Learn per-column mean and standard deviation from *data*.

        Parameters
        ----------
        data : np.ndarray
            2-D training data.
        headers : list[str], optional
            Column names.
        type_map : dict[str, str], optional
            ``{col_name: inferred_type}`` from :class:`glassbox.eda.auto_typer.AutoTyper`.
            Auto-detected when *None*.

        Returns
        -------
        self
        """
        data, headers = _validate_input(data, headers)
        type_map = _get_type_map(data, headers, type_map)

        self._mean = {}
        self._std = {}
        self._numerical_cols = []

        for i, col_name in enumerate(headers):
            if type_map.get(col_name) != "numerical":
                continue
            clean = _col_to_float(data[:, i])
            if clean.size == 0:
                continue
            self._mean[col_name] = manual_mean(clean)
            # Standard deviation with ddof=1 (sample std) — matches the EDA stats module.
            self._std[col_name] = manual_std(clean, ddof=1) if clean.size >= 2 else 0.0
            self._numerical_cols.append(col_name)

        self._fitted = True
        return self

    def transform(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[ScalerSummary]]:
        """Apply Standard scaling to numerical columns.

        Parameters
        ----------
        data : np.ndarray
            2-D data to scale.
        headers : list[str], optional
            Column names (must match those seen in ``fit()``).

        Returns
        -------
        tuple[np.ndarray, list[ScalerSummary]]
            A *copy* of *data* with numerical columns standardised, and a
            per-column summary of the parameters used.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        data, headers = _validate_input(data, headers)

        result = data.copy().astype(object)
        summaries: List[ScalerSummary] = []

        for i, col_name in enumerate(headers):
            if col_name not in self._mean:
                continue
            mu = self._mean[col_name]
            sigma = self._std[col_name]

            col_f = data[:, i].astype(np.float64)

            if sigma == 0.0:
                # Constant column — output 0.0 (mean-centred, undefined spread).
                scaled = np.where(np.isnan(col_f), np.nan, 0.0)
            else:
                # Standard score formula: (x - mu) / sigma
                scaled = (col_f - mu) / sigma

            result[:, i] = scaled
            summaries.append(ScalerSummary(
                column=col_name,
                scaler_type="standard",
                param_a=mu,
                param_b=sigma,
            ))

        return result, summaries

    def fit_transform(
        self,
        data: np.ndarray,
        headers: Optional[List[str]] = None,
        type_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[np.ndarray, List[ScalerSummary]]:
        """Fit on *data* then transform it in one step."""
        return self.fit(data, headers, type_map).transform(data, headers)
