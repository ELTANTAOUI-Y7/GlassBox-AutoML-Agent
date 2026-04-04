"""
tests.test_auto_typer
=====================

Unit tests for :mod:`glassbox.eda.auto_typer`.
"""

import numpy as np
import pytest

from glassbox.eda.auto_typer import AutoTyper, ColumnTypeInfo


class TestAutoTyper:
    """Tests for the AutoTyper column-type inference engine."""

    def setup_method(self):
        self.typer = AutoTyper()

    # ----- Numerical detection -----------------------------------------
    def test_float_column_detected_as_numerical(self):
        data = np.array([[1.1], [2.2], [3.3], [4.4], [5.5]])
        result = self.typer.detect(data, ["val"])
        assert result[0].inferred_type == "numerical"

    def test_integer_column_detected_as_numerical(self):
        data = np.arange(100).reshape(-1, 1).astype(float)
        result = self.typer.detect(data, ["id"])
        assert result[0].inferred_type == "numerical"

    # ----- Categorical detection ---------------------------------------
    def test_string_column_detected_as_categorical(self):
        data = np.array([["red"], ["blue"], ["green"], ["red"], ["blue"]])
        result = self.typer.detect(data, ["color"])
        assert result[0].inferred_type == "categorical"

    def test_low_cardinality_numeric_detected_as_categorical(self):
        # 2 unique values out of 100 → ratio 0.02 < 0.05 threshold.
        col = np.array([1, 2] * 50, dtype=float).reshape(-1, 1)
        typer = AutoTyper(categorical_cardinality_ratio=0.05,
                          categorical_max_unique=20)
        result = typer.detect(col, ["code"])
        # Should be boolean (2 unique) or categorical, not "numerical".
        assert result[0].inferred_type in ("categorical", "boolean")

    # ----- Boolean detection -------------------------------------------
    def test_true_false_detected_as_boolean(self):
        data = np.array([["true"], ["false"], ["true"], ["false"]])
        result = self.typer.detect(data, ["flag"])
        assert result[0].inferred_type == "boolean"

    def test_yes_no_detected_as_boolean(self):
        data = np.array([["yes"], ["no"], ["yes"], ["no"]])
        result = self.typer.detect(data, ["active"])
        assert result[0].inferred_type == "boolean"

    def test_01_detected_as_boolean(self):
        data = np.array([[0], [1], [0], [1], [1]], dtype=float)
        result = self.typer.detect(data, ["flag"])
        assert result[0].inferred_type == "boolean"

    # ----- Missing values ----------------------------------------------
    def test_missing_count(self):
        data = np.array([[1.0], [np.nan], [3.0], [np.nan]])
        result = self.typer.detect(data, ["x"])
        assert result[0].n_missing == 2

    # ----- Auto-generated headers --------------------------------------
    def test_auto_headers(self):
        data = np.array([[1, 2], [3, 4]], dtype=float)
        result = self.typer.detect(data)
        assert result[0].name == "col_0"
        assert result[1].name == "col_1"

    # ----- Multi-column -------------------------------------------------
    def test_mixed_dataset(self):
        data = np.array([
            [25,  "male",   "yes"],
            [30,  "female", "no"],
            [35,  "male",   "yes"],
            [40,  "female", "no"],
            [45,  "male",   "yes"],
            [50,  "female", "yes"],
            [55,  "male",   "no"],
            [60,  "female", "no"],
            [65,  "male",   "yes"],
            [70,  "female", "no"],
        ])
        headers = ["age", "gender", "subscriber"]
        result = self.typer.detect(data, headers)
        types = {r.name: r.inferred_type for r in result}
        assert types["gender"] == "categorical"
        assert types["subscriber"] == "boolean"

    # ----- Serialisation -----------------------------------------------
    def test_to_dict(self):
        data = np.array([[1.0], [2.0], [3.0]])
        result = self.typer.detect(data, ["x"])
        d = result[0].to_dict()
        assert "name" in d
        assert "inferred_type" in d
        assert "n_unique" in d
