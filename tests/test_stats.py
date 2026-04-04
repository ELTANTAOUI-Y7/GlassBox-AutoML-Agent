"""
tests.test_stats
================

Unit tests for :mod:`glassbox.eda.stats`.
"""

import numpy as np
import pytest

from glassbox.eda.stats import ColumnStats, StatProfiler


class TestStatProfiler:
    """Tests for the StatProfiler."""

    def setup_method(self):
        self.profiler = StatProfiler()

    def test_basic_stats(self):
        data = np.array([[1], [2], [3], [4], [5]], dtype=float)
        results = self.profiler.profile(data, ["x"])
        s = results[0]
        assert s.name == "x"
        assert s.count == 5
        assert s.missing == 0
        assert s.mean == pytest.approx(3.0)
        assert s.median == pytest.approx(3.0)
        assert s.min == pytest.approx(1.0)
        assert s.max == pytest.approx(5.0)
        assert s.range == pytest.approx(4.0)

    def test_with_nan(self):
        data = np.array([[1], [np.nan], [3], [np.nan], [5]], dtype=float)
        results = self.profiler.profile(data, ["x"])
        s = results[0]
        assert s.count == 3
        assert s.missing == 2
        assert s.mean == pytest.approx(3.0)

    def test_multiple_columns(self):
        data = np.array([
            [1, 10],
            [2, 20],
            [3, 30],
            [4, 40],
            [5, 50],
        ], dtype=float)
        results = self.profiler.profile(data, ["a", "b"])
        assert len(results) == 2
        assert results[0].mean == pytest.approx(3.0)
        assert results[1].mean == pytest.approx(30.0)

    def test_skips_non_numeric_with_type_map(self):
        data = np.array([
            ["a", "1"],
            ["b", "2"],
            ["c", "3"],
        ])
        type_map = {"name": "categorical", "value": "numerical"}
        results = self.profiler.profile(
            data, ["name", "value"], type_map=type_map
        )
        assert len(results) == 1
        assert results[0].name == "value"

    def test_all_nan_column(self):
        data = np.array([[np.nan], [np.nan], [np.nan]])
        results = self.profiler.profile(data, ["x"])
        s = results[0]
        assert s.count == 0
        assert s.missing == 3
        assert s.mean is None

    def test_iqr(self):
        data = np.arange(1, 101, dtype=float).reshape(-1, 1)
        results = self.profiler.profile(data, ["x"])
        s = results[0]
        assert s.q1 == pytest.approx(np.percentile(np.arange(1, 101), 25))
        assert s.q3 == pytest.approx(np.percentile(np.arange(1, 101), 75))
        assert s.iqr == pytest.approx(s.q3 - s.q1)

    def test_to_dict_keys(self):
        data = np.array([[1], [2], [3], [4], [5]], dtype=float)
        results = self.profiler.profile(data, ["x"])
        d = results[0].to_dict()
        expected_keys = {
            "name", "count", "missing", "mean", "median", "mode",
            "std", "variance", "min", "max", "range",
            "q1", "q3", "iqr", "skewness", "kurtosis",
        }
        assert set(d.keys()) == expected_keys
