"""
tests.test_correlation
======================

Unit tests for :mod:`glassbox.eda.correlation`.
"""

import numpy as np
import pytest

from glassbox.eda.correlation import CorrelationAnalyzer, CorrelationPair


class TestCorrelationAnalyzer:
    """Tests for the Pearson correlation analyser."""

    def test_perfect_positive(self):
        data = np.array([[1, 2], [2, 4], [3, 6], [4, 8]], dtype=float)
        ca = CorrelationAnalyzer()
        result = ca.analyze(data, ["x", "y"])
        assert result.matrix[0, 1] == pytest.approx(1.0, abs=1e-10)
        assert result.matrix[1, 0] == pytest.approx(1.0, abs=1e-10)

    def test_perfect_negative(self):
        data = np.array([[1, -2], [2, -4], [3, -6], [4, -8]], dtype=float)
        ca = CorrelationAnalyzer()
        result = ca.analyze(data, ["x", "y"])
        assert result.matrix[0, 1] == pytest.approx(-1.0, abs=1e-10)

    def test_uncorrelated(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(10000)
        y = rng.standard_normal(10000)
        data = np.column_stack([x, y])
        ca = CorrelationAnalyzer()
        result = ca.analyze(data, ["x", "y"])
        assert abs(result.matrix[0, 1]) < 0.05

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(1)
        data = rng.standard_normal((100, 4))
        ca = CorrelationAnalyzer()
        result = ca.analyze(data, ["a", "b", "c", "d"])
        for i in range(4):
            assert result.matrix[i, i] == pytest.approx(1.0, abs=1e-10)

    def test_high_pairs_flagged(self):
        x = np.arange(100, dtype=float)
        y = x * 2 + 1  # perfect correlation
        z = np.random.default_rng(5).standard_normal(100)
        data = np.column_stack([x, y, z])
        ca = CorrelationAnalyzer(threshold=0.9)
        result = ca.analyze(data, ["x", "y", "z"])
        # x-y should be flagged.
        pair_names = {(p.col_a, p.col_b) for p in result.high_pairs}
        assert ("x", "y") in pair_names

    def test_no_numeric_columns(self):
        data = np.array([["a", "b"], ["c", "d"]])
        ca = CorrelationAnalyzer()
        result = ca.analyze(data, ["x", "y"])
        assert result.matrix.size == 0
        assert result.high_pairs == []

    def test_single_numeric_column(self):
        data = np.array([[1], [2], [3]], dtype=float)
        ca = CorrelationAnalyzer()
        result = ca.analyze(data, ["x"])
        assert result.matrix.shape == (1, 1)
        assert result.matrix[0, 0] == pytest.approx(1.0)

    def test_to_dict_structure(self):
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        ca = CorrelationAnalyzer()
        result = ca.analyze(data, ["x", "y"])
        d = result.to_dict()
        assert "column_names" in d
        assert "matrix" in d
        assert "high_correlation_pairs" in d

    def test_nan_handling(self):
        data = np.array([
            [1.0, 2.0],
            [np.nan, 4.0],
            [3.0, np.nan],
            [4.0, 8.0],
            [5.0, 10.0],
        ])
        ca = CorrelationAnalyzer()
        result = ca.analyze(data, ["x", "y"])
        # Should not crash and should return a valid r.
        assert -1.0 <= result.matrix[0, 1] <= 1.0
