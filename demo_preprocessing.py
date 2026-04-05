"""
demo_preprocessing.py
=====================

End-to-end demonstration of the GlassBox Cleaner (Automated Preprocessing).

This script:
  1. Generates a synthetic mixed-type dataset with missing values.
  2. Runs the full Inspector pipeline (EDA) to show the raw data profile.
  3. Runs the Cleaner pipeline with different configurations.
  4. Pretty-prints the JSON preprocessing report.

Usage
-----
    python demo_preprocessing.py
"""

import json
import numpy as np

from glassbox.eda.inspector import Inspector
from glassbox.preprocessing.cleaner import Cleaner, CleanerConfig


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------
def generate_dataset(n: int = 10, seed: int = 42) -> tuple:
    """Create a small mixed-type dataset with missing values.

    Columns
    -------
    age        Numerical  (20–60), 1 NaN.
    salary     Numerical  (30k–120k), 1 NaN.
    dept       Categorical ("Eng", "HR", "Sales", "Marketing").
    manager    Boolean    ("yes" / "no"), 1 None.
    rating     Numerical  (1.0–10.0), 1 NaN.
    """
    rng = np.random.default_rng(seed)

    ages    = rng.integers(20, 60, size=n).astype(float)
    salaries = rng.integers(30_000, 120_000, size=n).astype(float)
    depts   = rng.choice(["Eng", "HR", "Sales", "Marketing"], size=n)
    managers = rng.choice(["yes", "no"], size=n).astype(object)
    ratings = rng.uniform(1.0, 10.0, size=n)

    # Inject missing values.
    ages[2]     = np.nan
    salaries[5] = np.nan
    managers[7] = None
    ratings[4]  = np.nan

    data = np.empty((n, 5), dtype=object)
    data[:, 0] = ages
    data[:, 1] = salaries
    data[:, 2] = depts
    data[:, 3] = managers
    data[:, 4] = ratings

    headers = ["age", "salary", "dept", "manager", "rating"]
    return data, headers


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------
def section(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_array(data: np.ndarray, headers: list, max_rows: int = 6) -> None:
    """Pretty-print an array with column headers."""
    col_w = max(10, max(len(h) for h in headers) + 2)
    header_line = "  ".join(h.ljust(col_w) for h in headers)
    print(f"\n  {header_line}")
    print(f"  {'-' * len(header_line)}")
    for i, row in enumerate(data[:max_rows]):
        row_str = "  ".join(str(v)[:col_w].ljust(col_w) for v in row)
        print(f"  {row_str}")
    if len(data) > max_rows:
        print(f"  … ({len(data) - max_rows} more rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    data, headers = generate_dataset(n=10)

    # ------------------------------------------------------------------ #
    # 1. Show raw data                                                     #
    # ------------------------------------------------------------------ #
    section("RAW DATASET")
    print_array(data, headers)

    # ------------------------------------------------------------------ #
    # 2. EDA Inspector — leverage existing Module I                        #
    # ------------------------------------------------------------------ #
    section("EDA REPORT (Inspector — Module I)")
    inspector = Inspector()
    eda_report = inspector.run(data, headers)
    eda_dict = eda_report.to_dict()

    print("\n  Column types detected:")
    for ct in eda_dict["column_types"]:
        print(f"    {ct['name']:12s} → {ct['inferred_type']:12s}  "
              f"(missing: {ct['n_missing']})")

    if eda_dict["warnings"]:
        print("\n  EDA Warnings:")
        for w in eda_dict["warnings"]:
            print(f"    ⚠  {w}")

    # Build type_map from EDA so the Cleaner reuses it (no double AutoTyper run).
    type_map = {ct["name"]: ct["inferred_type"] for ct in eda_dict["column_types"]}

    # ------------------------------------------------------------------ #
    # 3. Cleaner — Standard Scaler + One-Hot Encoding                     #
    # ------------------------------------------------------------------ #
    section("CLEANER — Standard Scaler + One-Hot Encoding")

    config_ohe = CleanerConfig(
        imputer_strategy="mean",
        scale_numerical=True,
        scaler_type="standard",
        encode_categorical=True,
        encoder_type="onehot",
        drop_first_ohe=False,
    )
    cleaner_ohe = Cleaner(config=config_ohe)
    result_ohe = cleaner_ohe.run(data, headers, type_map=type_map)

    print(f"\n  Input shape : {data.shape[0]} rows × {data.shape[1]} cols")
    print(f"  Output shape: {result_ohe.data.shape[0]} rows × {result_ohe.data.shape[1]} cols")
    print(f"\n  Output headers: {result_ohe.headers}")
    print_array(result_ohe.data, result_ohe.headers)

    print("\n  Steps applied:")
    for step in result_ohe.report.steps_applied:
        print(f"    → {step}")

    if result_ohe.report.warnings:
        print("\n  Cleaner Warnings:")
        for w in result_ohe.report.warnings:
            print(f"    ⚠  {w}")

    print("\n  Imputation summary:")
    for imp in result_ohe.report.imputation:
        if imp["n_filled"] > 0:
            print(f"    {imp['column']:10s}: {imp['n_filled']} cell(s) filled "
                  f"with {imp['fill_value']!r} (strategy: {imp['strategy']})")

    print("\n  Scaling summary:")
    for sc in result_ohe.report.scaling:
        if sc["scaler_type"] == "standard":
            print(f"    {sc['column']:10s}: mean={sc['mean']:.4f}, std={sc['std']:.4f}")
        else:
            print(f"    {sc['column']:10s}: min={sc['x_min']:.4f}, max={sc['x_max']:.4f}")

    print("\n  Encoding summary:")
    for enc in result_ohe.report.encoding:
        print(f"    {enc['column']:10s}: {len(enc['categories'])} categories "
              f"→ {enc['output_columns']}")

    # ------------------------------------------------------------------ #
    # 4. Cleaner — MinMax Scaler + Label Encoding                         #
    # ------------------------------------------------------------------ #
    section("CLEANER — MinMax Scaler + Label Encoding")

    config_le = CleanerConfig(
        imputer_strategy="median",
        scale_numerical=True,
        scaler_type="minmax",
        minmax_range=(0.0, 1.0),
        encode_categorical=True,
        encoder_type="label",
    )
    cleaner_le = Cleaner(config=config_le)
    result_le = cleaner_le.run(data, headers, type_map=type_map)

    print(f"\n  Output shape: {result_le.data.shape[0]} rows × {result_le.data.shape[1]} cols")
    print(f"\n  Output headers: {result_le.headers}")
    print_array(result_le.data, result_le.headers)

    print("\n  Label encoding mappings:")
    for enc in result_le.report.encoding:
        print(f"    {enc['column']:10s}: {enc['mapping']}")

    # ------------------------------------------------------------------ #
    # 5. Full JSON report                                                  #
    # ------------------------------------------------------------------ #
    section("FULL PREPROCESSING REPORT (JSON)")
    print(result_ohe.to_json())


if __name__ == "__main__":
    main()
