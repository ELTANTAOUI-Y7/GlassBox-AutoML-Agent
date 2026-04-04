"""
tests.test_outliers
===================

Unit tests for :mod:`glassbox.eda.outliers`.
"""

import numpy as np
import pytest

from glassbox.eda.outliers import OutlierDetector, OutlierReport


class TestOutlierDetector:
    """Tests for IQR-based outlier detection and capping."""

    def setup_method(self):
        self.det = OutlierDetector(k=1.5)

    def test_no_outliers(self):
        data = np.arange(1, 11, dtype=float).reshape(-1, 1)
        reports = self.det.detect(data, ["x"])
        assert reports[0].n_outliers_low == 0
        assert reports[0].n_outliers_high == 0

    def test_clear_outlier_high(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype=float).reshape(-1, 1)
        reports = self.det.detect(data, ["x"])
        assert reports[0].n_outliers_high >= 1

    def test_clear_outlier_low(self):
        data = np.array([-100, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float).reshape(-1, 1)
        reports = self.det.detect(data, ["x"])
        assert reports[0].n_outliers_low >= 1

    def test_outlier_indices_correct(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000], dtype=float).reshape(-1, 1)
        reports = self.det.detect(data, ["x"])
        assert 9 in reports[0].outlier_indices  # Index of 1000.

    def test_cap_clips_values(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000], dtype=float).reshape(-1, 1)
        capped = self.det.cap(data, ["x"])
        assert capped.max() < 1000  # Should be clipped.

    def test_cap_is_non_destructive(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000], dtype=float).reshape(-1, 1)
        original = data.copy()
        _ = self.det.cap(data, ["x"])
        np.testing.assert_array_equal(data, original)

    def test_outlier_pct(self):
        # 1 outlier in 10 values → 10%.
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000], dtype=float).reshape(-1, 1)
        reports = self.det.detect(data, ["x"])
        assert reports[0].outlier_pct > 0

    def test_multiple_columns(self):
        data = np.column_stack([
            np.arange(1, 11, dtype=float),
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000], dtype=float),
        ])
        reports = self.det.detect(data, ["a", "b"])
        assert len(reports) == 2

    def test_nan_handling(self):
        data = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 1000], dtype=float).reshape(-1, 1)
        reports = self.det.detect(data, ["x"])
        assert reports[0].n_total == 9  # 10 - 1 NaN.

    def test_to_dict_keys(self):
        data = np.array([1, 2, 3, 4, 5], dtype=float).reshape(-1, 1)
        reports = self.det.detect(data, ["x"])
        d = reports[0].to_dict()
        assert "lower_fence" in d
        assert "upper_fence" in d
        assert "n_outliers_total" in d
        assert "outlier_pct" in d

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            OutlierDetector(k=0)
        with pytest.raises(ValueError):
            OutlierDetector(k=-1)

    def test_extreme_k(self):
        # k=3 should flag fewer outliers than k=1.5.
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 50], dtype=float).reshape(-1, 1)
        det15 = OutlierDetector(k=1.5)
        det30 = OutlierDetector(k=3.0)
        r15 = det15.detect(data, ["x"])[0]
        r30 = det30.detect(data, ["x"])[0]
        total15 = r15.n_outliers_low + r15.n_outliers_high
        total30 = r30.n_outliers_low + r30.n_outliers_high
        assert total30 <= total15
