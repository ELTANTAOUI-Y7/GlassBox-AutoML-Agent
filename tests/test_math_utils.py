"""
tests.test_math_utils
=====================

Unit tests for :mod:`glassbox.eda.math_utils`.

Every test compares the scratch implementation against a known analytical
value or NumPy/SciPy reference to guarantee correctness.
"""

import numpy as np
import pytest

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


# ======================================================================
# MEAN
# ======================================================================
class TestManualMean:
    def test_simple(self):
        assert manual_mean(np.array([1, 2, 3, 4, 5], dtype=float)) == 3.0

    def test_single_element(self):
        assert manual_mean(np.array([42.0])) == 42.0

    def test_with_nan(self):
        arr = np.array([1.0, np.nan, 3.0])
        assert manual_mean(arr) == pytest.approx(2.0)

    def test_all_nan_raises(self):
        with pytest.raises(ValueError):
            manual_mean(np.array([np.nan, np.nan]))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            manual_mean(np.array([], dtype=float))

    def test_large_array(self):
        arr = np.arange(1, 10001, dtype=float)
        expected = 5000.5
        assert manual_mean(arr) == pytest.approx(expected)

    def test_negative_values(self):
        arr = np.array([-3, -1, 0, 1, 3], dtype=float)
        assert manual_mean(arr) == pytest.approx(0.0)

    def test_matches_numpy(self):
        rng = np.random.default_rng(42)
        arr = rng.standard_normal(500)
        assert manual_mean(arr) == pytest.approx(np.nanmean(arr), abs=1e-10)


# ======================================================================
# MEDIAN
# ======================================================================
class TestManualMedian:
    def test_odd_count(self):
        assert manual_median(np.array([3, 1, 2], dtype=float)) == 2.0

    def test_even_count(self):
        assert manual_median(np.array([1, 2, 3, 4], dtype=float)) == 2.5

    def test_single(self):
        assert manual_median(np.array([7.0])) == 7.0

    def test_with_nan(self):
        arr = np.array([np.nan, 5, 1, 3], dtype=float)
        assert manual_median(arr) == pytest.approx(3.0)

    def test_all_nan_raises(self):
        with pytest.raises(ValueError):
            manual_median(np.array([np.nan]))

    def test_matches_numpy(self):
        rng = np.random.default_rng(99)
        arr = rng.standard_normal(201)
        assert manual_median(arr) == pytest.approx(np.nanmedian(arr), abs=1e-10)


# ======================================================================
# MODE
# ======================================================================
class TestManualMode:
    def test_clear_winner(self):
        arr = np.array([1, 2, 2, 3, 3, 3], dtype=float)
        assert manual_mode(arr) == 3.0

    def test_tie_returns_smallest(self):
        arr = np.array([1, 1, 2, 2], dtype=float)
        assert manual_mode(arr) == 1.0

    def test_strings(self):
        arr = np.array(["cat", "dog", "cat", "bird"])
        assert manual_mode(arr) == "cat"

    def test_single(self):
        arr = np.array([99.0])
        assert manual_mode(arr) == 99.0

    def test_all_nan_raises(self):
        with pytest.raises(ValueError):
            manual_mode(np.array([np.nan, np.nan]))


# ======================================================================
# VARIANCE
# ======================================================================
class TestManualVariance:
    def test_population(self):
        arr = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)
        expected = 4.0
        assert manual_variance(arr, ddof=0) == pytest.approx(expected)

    def test_sample(self):
        arr = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)
        expected = 32.0 / 7
        assert manual_variance(arr, ddof=1) == pytest.approx(expected)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            manual_variance(np.array([], dtype=float))

    def test_single_ddof1_raises(self):
        with pytest.raises(ValueError):
            manual_variance(np.array([5.0]), ddof=1)

    def test_matches_numpy(self):
        rng = np.random.default_rng(7)
        arr = rng.standard_normal(300)
        assert manual_variance(arr, ddof=0) == pytest.approx(
            np.nanvar(arr, ddof=0), abs=1e-10
        )
        assert manual_variance(arr, ddof=1) == pytest.approx(
            np.nanvar(arr, ddof=1), abs=1e-10
        )


# ======================================================================
# STANDARD DEVIATION
# ======================================================================
class TestManualStd:
    def test_basic(self):
        arr = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)
        assert manual_std(arr, ddof=0) == pytest.approx(2.0)

    def test_matches_numpy(self):
        rng = np.random.default_rng(3)
        arr = rng.standard_normal(400)
        assert manual_std(arr, ddof=1) == pytest.approx(
            float(np.nanstd(arr, ddof=1)), abs=1e-10
        )


# ======================================================================
# SKEWNESS
# ======================================================================
class TestManualSkewness:
    def test_symmetric_is_zero(self):
        arr = np.array([-2, -1, 0, 1, 2], dtype=float)
        assert manual_skewness(arr) == pytest.approx(0.0, abs=1e-10)

    def test_right_skew(self):
        arr = np.array([1, 1, 1, 1, 1, 1, 1, 1, 10, 100], dtype=float)
        assert manual_skewness(arr) > 0

    def test_left_skew(self):
        arr = np.array([100, 10, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
        skew = manual_skewness(arr)
        # Same data → same magnitude, but still positive since values
        # are the same set.  Let's use a truly left-skewed sample:
        arr2 = np.array([-100, -10, 0, 1, 1, 1, 1, 1, 1, 1], dtype=float)
        assert manual_skewness(arr2) < 0

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            manual_skewness(np.array([1, 2], dtype=float))

    def test_constant_is_zero(self):
        arr = np.array([5, 5, 5, 5, 5], dtype=float)
        assert manual_skewness(arr) == 0.0


# ======================================================================
# KURTOSIS
# ======================================================================
class TestManualKurtosis:
    def test_normal_approx_zero(self):
        rng = np.random.default_rng(12)
        arr = rng.standard_normal(100000)
        assert manual_kurtosis(arr) == pytest.approx(0.0, abs=0.1)

    def test_uniform_negative(self):
        # Uniform distribution has excess kurtosis ≈ −1.2.
        rng = np.random.default_rng(55)
        arr = rng.uniform(0, 1, 100000)
        assert manual_kurtosis(arr) < 0

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            manual_kurtosis(np.array([1, 2, 3], dtype=float))

    def test_constant_is_zero(self):
        arr = np.array([7, 7, 7, 7, 7], dtype=float)
        assert manual_kurtosis(arr) == 0.0


# ======================================================================
# PERCENTILE
# ======================================================================
class TestManualPercentile:
    def test_0_and_100(self):
        arr = np.array([10, 20, 30, 40, 50], dtype=float)
        assert manual_percentile(arr, 0) == 10.0
        assert manual_percentile(arr, 100) == 50.0

    def test_median_via_p50(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=float)
        assert manual_percentile(arr, 50) == pytest.approx(3.0)

    def test_q1_q3(self):
        arr = np.arange(1, 101, dtype=float)
        assert manual_percentile(arr, 25) == pytest.approx(np.percentile(arr, 25))
        assert manual_percentile(arr, 75) == pytest.approx(np.percentile(arr, 75))

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            manual_percentile(np.array([1.0]), -1)
        with pytest.raises(ValueError):
            manual_percentile(np.array([1.0]), 101)

    def test_matches_numpy(self):
        rng = np.random.default_rng(22)
        arr = rng.standard_normal(500)
        for q in [10, 25, 50, 75, 90]:
            assert manual_percentile(arr, q) == pytest.approx(
                float(np.percentile(arr, q)), abs=1e-10
            )
