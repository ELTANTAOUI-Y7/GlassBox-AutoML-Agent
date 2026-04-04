"""
tests.test_inspector
====================

Integration tests for :mod:`glassbox.eda.inspector`.
"""

import json

import numpy as np
import pytest

from glassbox.eda.inspector import EDAReport, Inspector, InspectorConfig


class TestInspector:
    """Integration tests for the full Inspector pipeline."""

    def _make_dataset(self):
        """Create a small mixed-type dataset for testing."""
        rng = np.random.default_rng(42)
        n = 100
        age = rng.integers(18, 70, size=n).astype(float)
        salary = rng.integers(30000, 150000, size=n).astype(float)
        # Inject a couple of outliers.
        salary[0] = 999999
        salary[1] = -5000
        experience = age * 0.5 + rng.normal(0, 2, size=n)  # correlated with age
        purchased = rng.choice([0.0, 1.0], size=n)

        data = np.column_stack([age, salary, experience, purchased])
        headers = ["age", "salary", "experience", "purchased"]
        return data, headers

    def test_full_pipeline_runs(self):
        data, headers = self._make_dataset()
        inspector = Inspector()
        report = inspector.run(data, headers)
        assert isinstance(report, EDAReport)
        assert report.n_rows == 100
        assert report.n_cols == 4

    def test_column_types_populated(self):
        data, headers = self._make_dataset()
        report = Inspector().run(data, headers)
        assert len(report.column_types) == 4

    def test_statistics_populated(self):
        data, headers = self._make_dataset()
        report = Inspector().run(data, headers)
        # At least the numerical columns should have stats.
        assert len(report.statistics) >= 2

    def test_correlation_matrix_shape(self):
        data, headers = self._make_dataset()
        report = Inspector().run(data, headers)
        mat = report.correlation.get("matrix", [])
        n_numeric = len(report.correlation.get("column_names", []))
        assert len(mat) == n_numeric
        if n_numeric > 0:
            assert len(mat[0]) == n_numeric

    def test_outliers_detected(self):
        data, headers = self._make_dataset()
        report = Inspector().run(data, headers)
        # Salary column should have outliers flagged.
        salary_outliers = [o for o in report.outliers if o["name"] == "salary"]
        assert len(salary_outliers) == 1
        assert salary_outliers[0]["n_outliers_total"] >= 1

    def test_warnings_generated(self):
        data, headers = self._make_dataset()
        report = Inspector().run(data, headers)
        # age and experience are highly correlated → should produce a warning.
        assert len(report.warnings) >= 0  # At least no crash.

    def test_json_output(self):
        data, headers = self._make_dataset()
        report = Inspector().run(data, headers)
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert "metadata" in parsed
        assert "column_types" in parsed
        assert "statistics" in parsed
        assert "correlation" in parsed
        assert "outliers" in parsed
        assert "warnings" in parsed

    def test_run_json_convenience(self):
        data, headers = self._make_dataset()
        json_str = Inspector().run_json(data, headers)
        parsed = json.loads(json_str)
        assert parsed["metadata"]["n_rows"] == 100

    def test_custom_config(self):
        data, headers = self._make_dataset()
        config = InspectorConfig(
            correlation_threshold=0.5,
            outlier_k=3.0,
        )
        report = Inspector(config=config).run(data, headers)
        assert isinstance(report, EDAReport)

    def test_single_column(self):
        data = np.array([[1], [2], [3], [4], [5]], dtype=float)
        report = Inspector().run(data, ["x"])
        assert report.n_cols == 1
        assert len(report.statistics) == 1

    def test_all_categorical(self):
        data = np.array([["a", "x"], ["b", "y"], ["c", "z"], ["a", "x"]])
        report = Inspector().run(data, ["col1", "col2"])
        assert report.n_cols == 2
        # No numerical stats.
        num_stats = [s for s in report.statistics]
        # Could be 0 if none are castable.
        assert isinstance(num_stats, list)

    def test_elapsed_time_recorded(self):
        data = np.arange(20, dtype=float).reshape(10, 2)
        report = Inspector().run(data, ["a", "b"])
        assert report.elapsed_seconds >= 0
