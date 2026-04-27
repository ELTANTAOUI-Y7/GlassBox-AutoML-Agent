"""
GlassBox vs Scikit-Learn — Telco Customer Churn Comparison
Runs both pipelines on the same data and writes comparison_report.html.

Usage:
    python compare_sklearn.py [path/to/telco.csv]
"""

from __future__ import annotations

import csv
import io
import importlib.util
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

_ROOT = str(Path(__file__).parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from glassbox.agent.autofit import AutoFitConfig

# Load _FastAutoFit from agent.py (same variant used at inference time)
_spec = importlib.util.spec_from_file_location("agent_main", Path(_ROOT) / "agent.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_FastAutoFit = _mod._FastAutoFit

# ── Config ───────────────────────────────────────────────────────────────────
CSV_PATH   = sys.argv[1] if len(sys.argv) > 1 else "/tmp/telco_churn.csv"
TARGET     = "Churn"
SAMPLE     = 500
SEED       = 42
CV_FOLDS   = 3


# ── Helpers (mirror agent.py logic for a fair comparison) ────────────────────

def _drop_id_cols(csv_string: str, target_col: str) -> tuple[str, list[str]]:
    """Drop columns with >90% unique string values (e.g. customerID)."""
    reader = csv.reader(io.StringIO(csv_string.strip()))
    rows = list(reader)
    headers, data_rows = [h.strip() for h in rows[0]], rows[1:]
    n = len(data_rows)
    drop = set()
    for ci, col in enumerate(headers):
        if col == target_col:
            continue
        vals = [data_rows[r][ci].strip() if ci < len(data_rows[r]) else "" for r in range(n)]
        try:
            floats = [float(v) for v in vals if v]
            ints   = [int(f) for f in floats]
            # consecutive integer index → row ID
            if (all(floats[i] == ints[i] for i in range(len(floats)))
                    and len(set(ints)) == n and (max(ints) - min(ints)) == n - 1):
                drop.add(ci)
                continue
        except ValueError:
            pass
        if len(set(vals)) / max(n, 1) > 0.9:
            drop.add(ci)
    keep = [i for i in range(len(headers)) if i not in drop]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([headers[i] for i in keep])
    for row in data_rows:
        w.writerow([row[i] if i < len(row) else "" for i in keep])
    return buf.getvalue(), [headers[i] for i in drop]


def _sample_csv(csv_string: str, max_rows: int, seed: int) -> tuple[str, int, int]:
    """Randomly sample up to max_rows data rows (header preserved)."""
    reader = csv.reader(io.StringIO(csv_string.strip()))
    rows = list(reader)
    header, data = rows[0], rows[1:]
    n_total = len(data)
    if n_total <= max_rows:
        return csv_string, n_total, n_total
    random.seed(seed)
    sampled = random.sample(data, max_rows)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(sampled)
    return buf.getvalue(), n_total, max_rows


# ── Load dataset ─────────────────────────────────────────────────────────────

print("Loading dataset …")
raw_csv = Path(CSV_PATH).read_text(encoding="utf-8")
n_rows_raw = raw_csv.count("\n") - 1
print(f"  {n_rows_raw} rows × {len(raw_csv.splitlines()[0].split(','))} columns")

cleaned_csv, dropped_cols = _drop_id_cols(raw_csv, TARGET)
sampled_csv, n_total, n_used = _sample_csv(cleaned_csv, SAMPLE, SEED)
print(f"  Dropped: {dropped_cols}  |  Using {n_used}/{n_total} rows")


# ── GlassBox pipeline ─────────────────────────────────────────────────────────

print("\nRunning GlassBox AutoFit …")
t0 = time.perf_counter()
gb_config = AutoFitConfig(search_type="random", cv=CV_FOLDS, n_iter=8,
                          time_budget=60, random_state=SEED)
gb_report  = _FastAutoFit(config=gb_config).run_csv(sampled_csv, target_col=TARGET)
gb_elapsed = time.perf_counter() - t0
print(f"  best={gb_report.best_model}  accuracy={gb_report.best_score:.4f}  ({gb_elapsed:.1f}s)")


# ── Scikit-Learn pipeline ─────────────────────────────────────────────────────

print("\nRunning Scikit-Learn pipeline …")
df = pd.read_csv(io.StringIO(sampled_csv))
# TotalCharges arrives as string with occasional spaces for brand-new customers
for col in df.select_dtypes("object").columns:
    converted = pd.to_numeric(df[col], errors="coerce")
    if converted.notna().mean() > 0.9:   # mostly numeric → coerce
        df[col] = converted

X_df  = df.drop(columns=[TARGET])
y_raw = df[TARGET].map({"Yes": 1, "No": 0}).values

num_cols = X_df.select_dtypes(include="number").columns.tolist()
cat_cols = X_df.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("scl", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore",
                                            sparse_output=False))]), cat_cols),
])

sk_candidates = {
    "DecisionTree":       DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "RandomForest":       RandomForestClassifier(n_estimators=10, max_depth=5, random_state=SEED),
    "LogisticRegression": SKLogisticRegression(max_iter=500, random_state=SEED),
    "GaussianNaiveBayes": GaussianNB(),
}

cv       = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
sk_results: dict = {}
t0_sk = time.perf_counter()

for name, clf in sk_candidates.items():
    pipe   = Pipeline([("pre", preprocessor), ("clf", clf)])
    scores = cross_val_score(pipe, X_df, y_raw, cv=cv, scoring="accuracy")
    sk_results[name] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
    print(f"  {name:<22} accuracy={np.mean(scores):.4f} ± {np.std(scores):.4f}")

sk_elapsed  = time.perf_counter() - t0_sk
sk_best     = max(sk_results, key=lambda k: sk_results[k]["mean"])
sk_best_score = sk_results[sk_best]["mean"]
print(f"  best={sk_best}  accuracy={sk_best_score:.4f}  ({sk_elapsed:.1f}s)")


# ── Feature importances (sklearn, tree-based) ─────────────────────────────────

sk_fi: list[dict] = []
tree_model = next((n for n in ["RandomForest", "DecisionTree"] if n in sk_results), None)
if tree_model:
    pipe = Pipeline([("pre", preprocessor), ("clf", sk_candidates[tree_model])])
    pipe.fit(X_df, y_raw)
    ohe_feats   = (pipe.named_steps["pre"]
                       .named_transformers_["cat"]
                       .named_steps["ohe"]
                       .get_feature_names_out(cat_cols).tolist())
    all_feats   = num_cols + ohe_feats
    imps        = pipe.named_steps["clf"].feature_importances_
    sk_fi = [{"feature": f, "importance": round(float(v), 4)}
             for f, v in sorted(zip(all_feats, imps), key=lambda x: x[1], reverse=True)[:10]]


# ── Print summary ─────────────────────────────────────────────────────────────

winner   = "GlassBox" if gb_report.best_score >= sk_best_score else "Scikit-Learn"
gap      = abs(gb_report.best_score - sk_best_score)
print(f"\n{'='*55}")
print(f"  GlassBox best : {gb_report.best_model:<22} {gb_report.best_score:.4f}")
print(f"  Sklearn best  : {sk_best:<22} {sk_best_score:.4f}")
print(f"  Winner        : {winner}  (gap={gap*100:.2f}pp)")
print(f"{'='*55}")


# ── Build & write HTML report ─────────────────────────────────────────────────

from report_comparison import build_html  # noqa: E402  (local module)

html = build_html(
    gb_report=gb_report,
    gb_elapsed=gb_elapsed,
    sk_results=sk_results,
    sk_elapsed=sk_elapsed,
    sk_best=sk_best,
    sk_fi=sk_fi,
    tree_model=tree_model,
    gb_config=gb_config,
    n_total=n_total,
    n_used=n_used,
)

out = Path(_ROOT) / "comparison_report.html"
out.write_text(html, encoding="utf-8")
print(f"\nReport saved → {out}")