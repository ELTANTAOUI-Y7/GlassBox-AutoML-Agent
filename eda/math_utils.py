"""
glassbox.eda.math_utils
=======================

Low-level mathematical primitives used across the EDA module.

Every function here operates on plain NumPy arrays and implements the
formula from scratch — no calls to ``np.mean``, ``np.std``, etc.

Functions
---------
manual_mean        – Arithmetic mean.
manual_median      – Median (middle value).
manual_mode        – Mode (most frequent value).
manual_variance    – Population or sample variance.
manual_std         – Population or sample standard deviation.
manual_skewness    – Fisher's skewness (adjusted for sample size).
manual_kurtosis    – Excess kurtosis (Fisher's definition).
manual_percentile  – Linear-interpolation percentile (matches NumPy).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Mean
# ---------------------------------------------------------------------------
def manual_mean(arr: np.ndarray) -> float:
    """Compute the arithmetic mean of *arr*.

    .. math::
        \\bar{x} = \\frac{1}{n} \\sum_{i=1}^{n} x_i

    Parameters
    ----------
    arr : np.ndarray
        1-D array of numeric values (NaN values are ignored).

    Returns
    -------
    float
        The arithmetic mean.

    Raises
    ------
    ValueError
        If *arr* contains no valid (non-NaN) elements.
    """
    clean = arr[~np.isnan(arr)]
    if clean.size == 0:
        raise ValueError("Cannot compute mean of an empty / all-NaN array.")
    return float(np.sum(clean) / clean.size)


# ---------------------------------------------------------------------------
# Median
# ---------------------------------------------------------------------------
def manual_median(arr: np.ndarray) -> float:
    """Compute the median of *arr*.

    The median is the middle value of a sorted array.  For arrays with an
    even number of elements the average of the two central values is
    returned.

    Parameters
    ----------
    arr : np.ndarray
        1-D array of numeric values (NaN values are ignored).

    Returns
    -------
    float
        The median value.
    """
    clean = arr[~np.isnan(arr)]
    if clean.size == 0:
        raise ValueError("Cannot compute median of an empty / all-NaN array.")
    sorted_arr = np.sort(clean)
    n = sorted_arr.size
    mid = n // 2
    if n % 2 == 0:
        return float((sorted_arr[mid - 1] + sorted_arr[mid]) / 2.0)
    return float(sorted_arr[mid])


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------
def manual_mode(arr: np.ndarray) -> object:
    """Return the mode (most frequent value) of *arr*.

    If multiple values share the highest frequency the *smallest* one is
    returned (consistent with scipy.stats.mode behavior).

    Parameters
    ----------
    arr : np.ndarray
        1-D array.  NaN values are ignored.  Works for numeric **and**
        string arrays.

    Returns
    -------
    object
        The mode value (same dtype as input).
    """
    if arr.dtype.kind in ("U", "S", "O"):
        # String / object array — filter None and "nan" manually.
        clean = arr[arr != None]  # noqa: E711
        clean = np.array([v for v in clean if str(v) != "nan"])
    else:
        clean = arr[~np.isnan(arr.astype(float))]
    if clean.size == 0:
        raise ValueError("Cannot compute mode of an empty / all-NaN array.")

    unique_vals, counts = np.unique(clean, return_counts=True)
    max_count = np.max(counts)
    # Among ties pick the smallest value.
    candidates = unique_vals[counts == max_count]
    return candidates[0]


# ---------------------------------------------------------------------------
# Variance
# ---------------------------------------------------------------------------
def manual_variance(arr: np.ndarray, *, ddof: int = 0) -> float:
    """Compute the variance of *arr*.

    .. math::
        \\sigma^2 = \\frac{1}{n - \\text{ddof}} \\sum_{i=1}^{n}(x_i - \\bar{x})^2

    Parameters
    ----------
    arr : np.ndarray
        1-D numeric array (NaN ignored).
    ddof : int, optional
        Delta degrees of freedom.  ``0`` gives the **population** variance;
        ``1`` gives the **sample** (Bessel-corrected) variance.

    Returns
    -------
    float
    """
    clean = arr[~np.isnan(arr)]
    n = clean.size
    if n == 0:
        raise ValueError("Cannot compute variance of an empty / all-NaN array.")
    if n - ddof <= 0:
        raise ValueError("Not enough data points for the requested ddof.")
    mean = np.sum(clean) / n
    return float(np.sum((clean - mean) ** 2) / (n - ddof))


# ---------------------------------------------------------------------------
# Standard Deviation
# ---------------------------------------------------------------------------
def manual_std(arr: np.ndarray, *, ddof: int = 0) -> float:
    """Compute the standard deviation of *arr*.

    .. math::
        \\sigma = \\sqrt{\\text{Var}(x)}

    Parameters
    ----------
    arr : np.ndarray
        1-D numeric array (NaN ignored).
    ddof : int, optional
        See :func:`manual_variance`.

    Returns
    -------
    float
    """
    return float(np.sqrt(manual_variance(arr, ddof=ddof)))


# ---------------------------------------------------------------------------
# Skewness (Fisher's definition)
# ---------------------------------------------------------------------------
def manual_skewness(arr: np.ndarray) -> float:
    """Compute the **adjusted Fisher–Pearson** skewness coefficient.

    .. math::
        G_1 = \\frac{n}{(n-1)(n-2)} \\sum_{i=1}^{n}
              \\left(\\frac{x_i - \\bar{x}}{s}\\right)^3

    where *s* is the sample standard deviation (ddof = 1).

    Parameters
    ----------
    arr : np.ndarray
        1-D numeric array (NaN ignored).  Requires at least 3 values.

    Returns
    -------
    float
        Positive → right-skewed, negative → left-skewed, 0 → symmetric.
    """
    clean = arr[~np.isnan(arr)]
    n = clean.size
    if n < 3:
        raise ValueError("Skewness requires at least 3 data points.")
    mean = np.sum(clean) / n
    s = np.sqrt(np.sum((clean - mean) ** 2) / (n - 1))
    if s == 0:
        return 0.0
    m3 = np.sum(((clean - mean) / s) ** 3)
    return float((n / ((n - 1) * (n - 2))) * m3)


# ---------------------------------------------------------------------------
# Kurtosis (excess / Fisher's definition)
# ---------------------------------------------------------------------------
def manual_kurtosis(arr: np.ndarray) -> float:
    """Compute the **excess kurtosis** (Fisher's definition).

    .. math::
        K = \\frac{n(n+1)}{(n-1)(n-2)(n-3)} \\sum\\left(\\frac{x_i-\\bar{x}}{s}\\right)^4
            - \\frac{3(n-1)^2}{(n-2)(n-3)}

    A normal distribution has excess kurtosis of 0.

    Parameters
    ----------
    arr : np.ndarray
        1-D numeric array (NaN ignored).  Requires at least 4 values.

    Returns
    -------
    float
        Positive → heavy tails (leptokurtic), negative → light tails
        (platykurtic).
    """
    clean = arr[~np.isnan(arr)]
    n = clean.size
    if n < 4:
        raise ValueError("Kurtosis requires at least 4 data points.")
    mean = np.sum(clean) / n
    s = np.sqrt(np.sum((clean - mean) ** 2) / (n - 1))
    if s == 0:
        return 0.0
    m4 = np.sum(((clean - mean) / s) ** 4)
    term1 = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) * m4
    term2 = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return float(term1 - term2)


# ---------------------------------------------------------------------------
# Percentile (linear interpolation – matches NumPy default)
# ---------------------------------------------------------------------------
def manual_percentile(arr: np.ndarray, q: float) -> float:
    """Compute the *q*-th percentile of *arr* using linear interpolation.

    Parameters
    ----------
    arr : np.ndarray
        1-D numeric array (NaN ignored).
    q : float
        Percentile in the range ``[0, 100]``.

    Returns
    -------
    float
    """
    if not 0.0 <= q <= 100.0:
        raise ValueError("Percentile must be in [0, 100].")
    clean = arr[~np.isnan(arr)]
    if clean.size == 0:
        raise ValueError("Cannot compute percentile of an empty / all-NaN array.")
    sorted_arr = np.sort(clean)
    n = sorted_arr.size
    # Virtual index (linear interpolation formula used by NumPy).
    idx = (q / 100.0) * (n - 1)
    lo = int(np.floor(idx))
    hi = int(np.ceil(idx))
    if lo == hi:
        return float(sorted_arr[lo])
    frac = idx - lo
    return float(sorted_arr[lo] * (1 - frac) + sorted_arr[hi] * frac)
