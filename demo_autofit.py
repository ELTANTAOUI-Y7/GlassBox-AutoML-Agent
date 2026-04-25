"""
demo_autofit.py
===============
End-to-end IronClaw Agent Integration demo.

Simulates the full flow described in the GlassBox spec:

    User  →  "Build a model to predict 'Churn' using this CSV."
    Agent →  AutoFit().run_csv(csv_string, target_col="Churn")
    Lib   →  EDA → Cleaning → Optimization → AutoFitReport (JSON)
    Agent →  Explains: best model, score, top features.

Run:
    python demo_autofit.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import textwrap
import numpy as np
from glassbox.agent.autofit import AutoFit, AutoFitConfig

# ---------------------------------------------------------------------------
# Synthetic CSV datasets
# ---------------------------------------------------------------------------

# --- Classification: Customer Churn ---
CHURN_CSV = """\
CustomerID,Age,Tenure,MonthlyCharges,NumProducts,HasPartner,Contract,Churn
1,29,12,65.5,2,yes,month-to-month,yes
2,45,36,89.0,3,no,one_year,no
3,52,60,120.0,4,yes,two_year,no
4,23,3,45.0,1,no,month-to-month,yes
5,38,24,75.0,2,yes,one_year,no
6,61,72,95.0,3,no,two_year,no
7,27,6,55.0,1,no,month-to-month,yes
8,34,18,70.0,2,yes,month-to-month,yes
9,48,48,110.0,4,no,two_year,no
10,55,60,130.0,3,yes,two_year,no
11,31,9,60.0,1,no,month-to-month,yes
12,42,30,85.0,2,yes,one_year,no
13,25,3,40.0,1,no,month-to-month,yes
14,67,84,140.0,4,yes,two_year,no
15,36,15,72.0,2,no,month-to-month,yes
16,50,54,115.0,3,yes,two_year,no
17,28,6,50.0,1,no,month-to-month,yes
18,44,42,92.0,3,no,one_year,no
19,59,66,125.0,4,yes,two_year,no
20,33,12,68.0,2,yes,month-to-month,yes
21,46,36,88.0,3,no,one_year,no
22,26,4,42.0,1,no,month-to-month,yes
23,57,72,118.0,4,yes,two_year,no
24,39,20,78.0,2,no,month-to-month,yes
25,63,78,135.0,3,yes,two_year,no
26,30,8,58.0,1,no,month-to-month,yes
27,53,56,108.0,3,no,two_year,no
28,41,28,83.0,2,yes,one_year,no
29,24,2,38.0,1,no,month-to-month,yes
30,60,70,128.0,4,yes,two_year,no
"""

# --- Regression: House Prices ---
HOUSE_CSV = """\
SquareFeet,Bedrooms,Bathrooms,Age,Garage,Neighborhood,Price
1500,3,2,10,yes,suburban,250000
2200,4,3,5,yes,urban,420000
900,2,1,30,no,rural,120000
1800,3,2,8,yes,suburban,310000
3000,5,4,2,yes,urban,580000
1100,2,1,25,no,rural,145000
2500,4,3,3,yes,suburban,490000
1300,3,2,15,no,suburban,215000
2800,4,3,1,yes,urban,540000
750,1,1,40,no,rural,95000
1650,3,2,12,yes,suburban,275000
2100,4,2,7,no,urban,380000
1050,2,1,28,no,rural,135000
1950,3,3,6,yes,suburban,335000
3200,5,4,1,yes,urban,620000
1200,2,2,20,yes,suburban,195000
2350,4,3,4,yes,urban,455000
850,2,1,35,no,rural,108000
1750,3,2,9,yes,suburban,292000
2700,5,3,2,yes,urban,518000
1400,3,2,14,no,suburban,228000
2050,4,2,6,yes,urban,395000
950,2,1,32,no,rural,125000
1600,3,2,11,yes,suburban,262000
2900,5,4,1,yes,urban,562000
"""


# ---------------------------------------------------------------------------
# Helper: pretty section header
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Demo 1: Classification (Customer Churn)
# ---------------------------------------------------------------------------

def demo_classification() -> None:
    section("DEMO 1 — Classification: Customer Churn")

    print("\n[IronClaw Agent receives user message]")
    print('  User: "Build a model to predict \'Churn\' using this CSV."')
    print("\n[Agent calls AutoFit tool]")

    config = AutoFitConfig(
        search_type="random",
        cv=3,
        n_iter=10,
        random_state=42,
    )
    af = AutoFit(config=config)

    print("  AutoFit().run_csv(csv_string, target_col='Churn')  ...")
    report = af.run_csv(CHURN_CSV, target_col="Churn")

    print("\n[AutoFit returned AutoFitReport — Agent reads it]\n")
    print(report.summary())

    # --- Simulate agent explanation ---
    section("Agent Explanation (from JSON report)")

    top = report.feature_importances[:3]
    top_names = [f["feature"] for f in top]
    top_scores = [f['importance'] for f in top]

    print(f"\nAgent: I ran {len(report.model_rankings)} models on your dataset "
          f"({report.n_rows} rows, {report.n_features_in} features).\n")

    print(f"  Best model   : {report.best_model}")
    print(f"  CV Accuracy  : {report.best_score:.4f}")
    print(f"  Parameters   : {report.best_params}\n")

    if top_names:
        print("  Most influential features (by correlation with Churn):")
        for name, score in zip(top_names, top_scores):
            bar = "█" * int(score * 20)
            print(f"    {name:<20} {score:.4f}  {bar}")

    print("\n  Model rankings:")
    for rank in report.model_rankings:
        score_str = f"{rank['best_score']:.4f}" if rank["best_score"] is not None else "failed"
        print(f"    {rank['model']:<22} {score_str}")

    if report.warnings:
        print(f"\n  Warnings ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"    • {w[:80]}")


# ---------------------------------------------------------------------------
# Demo 2: Regression (House Prices)
# ---------------------------------------------------------------------------

def demo_regression() -> None:
    section("DEMO 2 — Regression: House Prices")

    print("\n[IronClaw Agent receives user message]")
    print('  User: "Build a model to predict \'Price\' using this CSV."')
    print("\n[Agent calls AutoFit tool]")

    config = AutoFitConfig(
        search_type="random",
        cv=3,
        n_iter=10,
        random_state=0,
    )
    af = AutoFit(config=config)

    print("  AutoFit().run_csv(csv_string, target_col='Price')  ...")
    report = af.run_csv(HOUSE_CSV, target_col="Price")

    print("\n[AutoFit returned AutoFitReport — Agent reads it]\n")
    print(report.summary())

    section("Agent Explanation (from JSON report)")

    top = report.feature_importances[:3]
    top_names = [f["feature"] for f in top]
    top_scores = [f["importance"] for f in top]

    print(f"\nAgent: I ran {len(report.model_rankings)} models on your dataset "
          f"({report.n_rows} rows, {report.n_features_in} features).\n")

    print(f"  Best model   : {report.best_model}")
    print(f"  CV neg-MSE   : {report.best_score:.4f}")
    print(f"  Parameters   : {report.best_params}\n")

    if top_names:
        print("  Most influential features (by |correlation with Price|):")
        for name, score in zip(top_names, top_scores):
            bar = "█" * int(score * 20)
            print(f"    {name:<20} {score:.4f}  {bar}")

    print("\n  Model rankings:")
    for rank in report.model_rankings:
        score_str = f"{rank['best_score']:.4f}" if rank["best_score"] is not None else "failed"
        print(f"    {rank['model']:<22} {score_str}")

    if report.warnings:
        print(f"\n  Warnings ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"    • {w[:80]}")


# ---------------------------------------------------------------------------
# Demo 3: Show raw JSON output (what the agent actually receives)
# ---------------------------------------------------------------------------

def demo_json_output() -> None:
    section("DEMO 3 — Raw JSON (what IronClaw agent receives)")

    config = AutoFitConfig(
        search_type="random",
        cv=3,
        n_iter=5,
        random_state=1,
    )
    report = AutoFit(config=config).run_csv(CHURN_CSV, target_col="Churn")

    # Show only the top-level structure (truncated for readability).
    import json
    d = report.to_dict()
    top_level = {
        "metadata": d["metadata"],
        "best_model": d["best_model"],
        "feature_importances": d["feature_importances"][:3],
        "model_rankings": d["model_rankings"],
        "warnings": d["warnings"],
        "eda_report": "{ ... full EDA JSON ... }",
        "preprocessing_report": "{ ... full preprocessing JSON ... }",
        "optimization_report": "{ ... full optimization JSON ... }",
    }
    print("\n# Top-level AutoFitReport JSON (sub-reports truncated):\n")
    print(json.dumps(top_level, indent=2, default=str))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_classification()
    demo_regression()
    demo_json_output()
    print("\n" + "=" * 60)
    print("  All demos complete.")
    print("=" * 60 + "\n")
