"""
demo_eda.py
===========

End-to-end demonstration of the GlassBox Inspector (Automated EDA).

This script:
  1. Generates a synthetic dataset with mixed types, missing values, and outliers.
  2. Runs the full Inspector pipeline.
  3. Pretty-prints the JSON report to the console.

Usage
-----
    python demo_eda.py
"""

import json

import numpy as np

from glassbox.eda.inspector import Inspector, InspectorConfig


def generate_synthetic_dataset(n: int = 200, seed: int = 42) -> tuple:
    """Create a realistic mixed-type dataset.

    Columns
    -------
    age         Numerical (18–75), 5 % NaN.
    salary      Numerical (25 k–200 k) with 2 extreme outliers.
    experience  Numerical: ~0.5 × age + noise (correlated with age).
    department  Categorical: {"Engineering", "Sales", "HR", "Marketing"}.
    is_manager  Boolean: "yes" / "no".
    satisfaction Numerical: 1–10 rating.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 75, size=n).astype(float)
    # Inject missing values.
    nan_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    age[nan_idx] = np.nan

    salary = rng.integers(25000, 200000, size=n).astype(float)
    salary[0] = 999999   # outlier high
    salary[1] = -5000    # outlier low

    experience = np.round(age * 0.5 + rng.normal(0, 3, size=n), 1)

    departments = np.array(["Engineering", "Sales", "HR", "Marketing"])
    department = rng.choice(departments, size=n)

    is_manager = rng.choice(["yes", "no"], size=n)

    satisfaction = rng.integers(1, 11, size=n).astype(float)

    # Build an object array so mixed types can coexist.
    data = np.empty((n, 6), dtype=object)
    data[:, 0] = age
    data[:, 1] = salary
    data[:, 2] = experience
    data[:, 3] = department
    data[:, 4] = is_manager
    data[:, 5] = satisfaction

    headers = ["age", "salary", "experience", "department", "is_manager", "satisfaction"]
    return data, headers


def main():
    print("=" * 70)
    print("  GlassBox-AutoML · Automated EDA Demo (The Inspector)")
    print("=" * 70)
    print()

    # 1. Generate data.
    data, headers = generate_synthetic_dataset()
    print(f"Dataset shape: {data.shape[0]} rows × {data.shape[1]} columns")
    print(f"Columns: {headers}")
    print()

    # 2. Configure and run Inspector.
    config = InspectorConfig(
        correlation_threshold=0.80,  # Flag pairs with |r| >= 0.80.
        outlier_k=1.5,              # Standard Tukey fences.
    )
    inspector = Inspector(config=config)
    report = inspector.run(data, headers)

    # 3. Print the JSON report.
    print("-" * 70)
    print("  FULL EDA REPORT (JSON)")
    print("-" * 70)
    print(report.to_json(indent=2))
    print()

    # 4. Summary highlights.
    print("-" * 70)
    print("  HIGHLIGHTS")
    print("-" * 70)
    rd = report.to_dict()
    print(f"  Rows: {rd['metadata']['n_rows']}")
    print(f"  Cols: {rd['metadata']['n_cols']}")
    print(f"  Time: {rd['metadata']['elapsed_seconds']:.4f}s")
    print()
    print("  Type Summary:")
    for t, count in rd["type_summary"].items():
        print(f"    {t:15s}: {count}")
    print()
    if rd["warnings"]:
        print("  Warnings:")
        for w in rd["warnings"]:
            print(f"    ⚠ {w}")
    else:
        print("  No warnings.")
    print()
    print("=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
