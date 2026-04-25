"""
GlassBox AutoFit — IronClaw / NEAR AI Agent Entry Point
=========================================================

IronClaw invokes ``run(env)`` for every user message.

Flow
----
1. User: "Build a model to predict 'Churn' using this CSV."
         (CSV is either an uploaded file or pasted inline in the message)
2. Agent extracts the CSV and the target column name.
3. AutoFit runs the full GlassBox pipeline:
       EDA → Cleaning → Hyperparameter Optimisation → Best model
4. Agent replies with a human-readable summary + full JSON report.
"""

from __future__ import annotations

import os
import re
import sys
import json
import traceback
from pathlib import Path
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so glassbox.* and models.* are found
# whether the agent is run locally or inside the IronClaw container.
# ---------------------------------------------------------------------------
_ROOT = str(Path(__file__).parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from glassbox.agent.autofit import AutoFit, AutoFitConfig, AutoFitReport


# ---------------------------------------------------------------------------
# Lean candidate set — drops KNN (O(n²) distance, too slow on large data)
# and caps RandomForest trees.  Used whenever n_rows > _SAMPLE_LIMIT.
# ---------------------------------------------------------------------------

class _FastAutoFit(AutoFit):
    def _get_candidates(self, task):
        return [
            (name, cls, grid)
            for name, cls, grid in super()._get_candidates(task)
            if name != "KNN"
            and (name != "RandomForest" or self._patch_rf(grid))
        ]

    @staticmethod
    def _patch_rf(grid: dict) -> bool:
        if "n_trees" in grid:
            grid["n_trees"] = [5]
        return True


# ---------------------------------------------------------------------------
# IronClaw / NEAR AI entry point
# ---------------------------------------------------------------------------

def run(env) -> None:
    """Called by the IronClaw runtime for every user turn."""
    messages = env.list_messages()

    # No messages yet — show the welcome prompt.
    if not messages:
        env.add_reply(_WELCOME)
        return

    last = messages[-1]
    if last.get("role") != "user":
        return

    user_text: str = last.get("content", "").strip()

    # ------------------------------------------------------------------ #
    # 1.  Find CSV                                                         #
    # ------------------------------------------------------------------ #
    csv_string, csv_source = _find_csv(env, user_text)

    if csv_string is None:
        env.add_reply(
            "I couldn't find a CSV dataset in your message.\n\n"
            "Please either:\n"
            "- **Upload a `.csv` file**, or\n"
            "- **Paste the CSV text** directly in your message.\n\n"
            "Also tell me which column to predict, e.g.:\n"
            '> "Build a model to predict **Churn** from this data."'
        )
        return

    # ------------------------------------------------------------------ #
    # 2.  Extract target column                                            #
    # ------------------------------------------------------------------ #
    headers = _csv_headers(csv_string)
    target_col = _extract_target(user_text, headers)

    if target_col is None:
        header_list = ", ".join(f"`{h}`" for h in headers)
        env.add_reply(
            f"I found your dataset ({csv_source}) but couldn't determine "
            f"which column to predict.\n\n"
            f"Available columns: {header_list}\n\n"
            "Please specify, for example:\n"
            "> \"predict **Churn**\"\n"
            "> \"target column: Price\""
        )
        return

    # ------------------------------------------------------------------ #
    # 3.  Run AutoFit pipeline                                             #
    # ------------------------------------------------------------------ #
    # Pure-Python models (RF, DT) are slow at scale — cap at 500 rows so each
    # individual CV fold completes in seconds, not minutes.
    csv_string, n_total, n_used = _sample_csv(csv_string, max_rows=500)
    sampled = n_used < n_total

    status = (
        f"Running GlassBox AutoFit on `{csv_source}` → predicting **{target_col}** …\n"
        f"Dataset: {n_total} rows — using {n_used} rows for search (random sample, seed 42)."
        if sampled else
        f"Running GlassBox AutoFit on `{csv_source}` → predicting **{target_col}** …"
    )
    env.add_reply(status)

    try:
        csv_string, dropped_cols = _drop_id_columns(csv_string, target_col)
        if dropped_cols:
            env.add_reply(
                f"Dropped high-cardinality columns (likely IDs, not useful as features): "
                + ", ".join(f"`{c}`" for c in dropped_cols)
            )

        if sampled:
            config = AutoFitConfig(
                search_type="random",
                cv=2,
                n_iter=3,
                time_budget=30,
                random_state=42,
            )
        else:
            config = AutoFitConfig(
                search_type="random",
                cv=3,
                n_iter=8,
                random_state=42,
            )
        cls = _FastAutoFit if sampled else AutoFit
        report = cls(config=config).run_csv(csv_string, target_col=target_col)
        env.add_reply(_format_reply(report))

    except Exception as exc:
        tb = traceback.format_exc()
        env.add_reply(
            f"**Pipeline error:** `{exc}`\n\n"
            "<details><summary>Traceback</summary>\n\n"
            f"```\n{tb}\n```\n</details>"
        )


# ---------------------------------------------------------------------------
# CSV sampling
# ---------------------------------------------------------------------------

def _sample_csv(csv_string: str, max_rows: int) -> Tuple[str, int, int]:
    """Return (csv_string, n_total, n_used). Randomly samples if n_total > max_rows."""
    import random
    lines = csv_string.strip().splitlines()
    header, data = lines[0], lines[1:]
    n_total = len(data)
    if n_total <= max_rows:
        return csv_string, n_total, n_total
    random.seed(42)
    sampled = random.sample(data, max_rows)
    return header + "\n" + "\n".join(sampled), n_total, max_rows


# ---------------------------------------------------------------------------
# ID-column removal
# ---------------------------------------------------------------------------

def _drop_id_columns(csv_string: str, target_col: str) -> Tuple[str, list]:
    """Drop columns whose values are almost entirely unique (ID columns).

    A column is treated as an ID if:
    - it is not the target, AND
    - it cannot be parsed as numbers, AND
    - unique value ratio > 0.9 (>90% of rows have a distinct value)

    Returns the cleaned CSV string and the list of dropped column names.
    """
    lines = csv_string.strip().splitlines()
    headers = [h.strip().strip('"').strip("'") for h in lines[0].split(",")]
    rows = [line.split(",") for line in lines[1:]]
    n_rows = len(rows)
    if n_rows == 0:
        return csv_string, []

    drop_indices = set()
    dropped_names = []

    for col_idx, col_name in enumerate(headers):
        if col_name == target_col:
            continue
        values = [rows[r][col_idx].strip() if col_idx < len(rows[r]) else "" for r in range(n_rows)]
        unique_ratio = len(set(values)) / max(n_rows, 1)
        # Numeric column: only drop if values are consecutive integers (1,2,3…)
        # — the classic row-ID pattern. Real features like Age or Price have gaps.
        try:
            floats = [float(v) for v in values if v]
            ints   = [int(f) for f in floats]
            is_consecutive = (
                all(floats[i] == ints[i] for i in range(len(floats)))  # all whole numbers
                and len(set(ints)) == n_rows                            # all unique
                and (max(ints) - min(ints)) == (n_rows - 1)            # no gaps
            )
            if not is_consecutive:
                continue
        except ValueError:
            pass
        # String ID columns: drop if >90% unique
        if unique_ratio > 0.9:
            drop_indices.add(col_idx)
            dropped_names.append(col_name)

    if not drop_indices:
        return csv_string, []

    keep = [i for i in range(len(headers)) if i not in drop_indices]
    new_header = ",".join(headers[i] for i in keep)
    new_rows   = [",".join(row[i] if i < len(row) else "" for i in keep) for row in rows]
    return new_header + "\n" + "\n".join(new_rows), dropped_names


# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------

def _find_csv(env, user_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (csv_string, source_label) or (None, None)."""

    # 1. Uploaded files (preferred — user explicitly attached them).
    try:
        for fname in env.list_files():
            if fname.lower().endswith(".csv"):
                content = env.read_file(fname)
                if content and _looks_like_csv(content):
                    return content, fname
    except Exception:
        pass

    # 2. Fenced code blocks: ```csv ... ``` or ``` ... ```
    fence_match = re.search(
        r"```(?:csv)?\s*\n(.*?)```",
        user_text,
        re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        candidate = fence_match.group(1).strip()
        if _looks_like_csv(candidate):
            return candidate, "inline (code block)"

    # 3. Bare CSV pasted into the message: find the longest contiguous block
    #    of lines that all contain at least one comma.
    best_block: Optional[str] = None
    best_len = 0

    lines = user_text.splitlines()
    i = 0
    while i < len(lines):
        if "," in lines[i]:
            j = i
            while j < len(lines) and "," in lines[j]:
                j += 1
            block = "\n".join(lines[i:j]).strip()
            if _looks_like_csv(block) and (j - i) > best_len:
                best_block = block
                best_len = j - i
            i = j
        else:
            i += 1

    if best_block:
        return best_block, "inline (pasted)"

    return None, None


def _looks_like_csv(text: str) -> bool:
    """Heuristic: at least 3 lines, each with 2+ comma-separated fields."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    col_count = len(lines[0].split(","))
    if col_count < 2:
        return False
    matching = sum(1 for ln in lines[1:] if len(ln.split(",")) == col_count)
    return matching >= min(2, len(lines) - 1)


def _csv_headers(csv_string: str) -> list[str]:
    first_line = csv_string.strip().splitlines()[0]
    return [h.strip().strip('"').strip("'") for h in first_line.split(",")]


# ---------------------------------------------------------------------------
# Target column extraction
# ---------------------------------------------------------------------------

_TARGET_PATTERNS = [
    r'predict(?:ing)?\s+["‘’“”`\'*]*(\w[\w\s]*?)["‘’“”`\'*]*\s*(?:\b|$)',
    r'target\s*(?:col(?:umn)?)?\s*[=:]\s*["‘’“”`\'*]*(\w[\w\s]*?)["‘’“”`\'*]*\s*(?:\b|$)',
    r'forecast(?:ing)?\s+["‘’“”`\'*]*(\w[\w\s]*?)["‘’“”`\'*]*',
    r'label\s*[=:]\s*["‘’“”`\'*]*(\w[\w\s]*?)["‘’“”`\'*]*',
]


def _extract_target(user_text: str, headers: list[str]) -> Optional[str]:
    """Try to find the target column name in the user's instruction."""
    lower_headers = {h.lower(): h for h in headers}

    def _match(candidate: str) -> Optional[str]:
        c = candidate.strip()
        # Exact match
        if c in headers:
            return c
        # Case-insensitive
        return lower_headers.get(c.lower())

    # Regex patterns
    for pat in _TARGET_PATTERNS:
        for m in re.finditer(pat, user_text, re.IGNORECASE):
            result = _match(m.group(1))
            if result:
                return result

    # Bold markdown: **ColumnName**
    for bold in re.findall(r"\*\*(\w[\w\s]*?)\*\*", user_text):
        result = _match(bold)
        if result:
            return result

    # Inline code: `ColumnName`
    for code in re.findall(r"`(\w[\w\s]*?)`", user_text):
        result = _match(code)
        if result:
            return result

    # Last resort: any header word that appears verbatim in the user text
    # (only if exactly one such header matches — avoids false positives).
    found = [h for h in headers if re.search(rf"\b{re.escape(h)}\b", user_text, re.IGNORECASE)]
    if len(found) == 1:
        return found[0]

    return None


# ---------------------------------------------------------------------------
# Reply formatting
# ---------------------------------------------------------------------------

def _format_reply(report: AutoFitReport) -> str:
    score_str = (
        f"{report.best_score:.4f}"
        if report.best_score == report.best_score  # nan check
        else "N/A"
    )

    lines = [
        "## GlassBox AutoFit — Results",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Task | {report.task_type} |",
        f"| Target | `{report.target_column}` |",
        f"| Dataset | {report.n_rows} rows × {report.n_features_in} features |",
        f"| Best model | **{report.best_model}** |",
        f"| CV score ({report.scoring_metric}) | **{score_str}** |",
        f"| Time | {report.elapsed_seconds:.2f}s |",
        "",
    ]

    # Feature importances
    if report.feature_importances:
        lines += ["**Top features** (by |correlation with target|):", ""]
        for fi in report.feature_importances[:5]:
            bar = "█" * max(1, int(fi["importance"] * 20))
            lines.append(f"- `{fi['feature']:<20}` {fi['importance']:.4f}  {bar}")
        lines.append("")

    # Best params
    if report.best_params:
        lines += [
            f"**Best hyperparameters for {report.best_model}:**",
            f"```json\n{json.dumps(report.best_params, indent=2)}\n```",
            "",
        ]

    # Model rankings table
    lines += ["**All model rankings:**", ""]
    lines.append("| Model | CV Score |")
    lines.append("|---|---|")
    for r in report.model_rankings:
        sc = f"{r['best_score']:.4f}" if r.get("best_score") is not None else "failed"
        medal = " 🥇" if r["model"] == report.best_model else ""
        lines.append(f"| {r['model']}{medal} | {sc} |")

    lines.append("")

    # Warnings
    if report.warnings:
        lines += [f"**Warnings ({len(report.warnings)}):**", ""]
        for w in report.warnings:
            lines.append(f"- {w}")
        lines.append("")

    # Full JSON — collapsed for readability
    lines += [
        "<details>",
        "<summary>Full AutoFitReport JSON</summary>",
        "",
        "```json",
        report.to_json(indent=2),
        "```",
        "",
        "</details>",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Welcome message
# ---------------------------------------------------------------------------

_WELCOME = """\
## GlassBox AutoFit

I can build a machine-learning model from your CSV data in one message.

**How to use:**
1. Upload a `.csv` file **or** paste your CSV text directly here.
2. Tell me which column to predict.

**Example:**
> "Build a model to predict **Churn** from this CSV."

I'll run the full GlassBox pipeline:
- **EDA** — inspect column types, missing values, distributions
- **Preprocessing** — imputation, scaling, encoding
- **Model selection** — Decision Tree, Random Forest, Naive Bayes, Logistic/Linear Regression, KNN
- **Hyperparameter search** — random search with cross-validation

You'll get a ranked summary of all models and the top predictive features.
"""


# ---------------------------------------------------------------------------
# Local smoke-test (python agent.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    class _FakeEnv:
        def __init__(self, message: str, files: dict | None = None):
            self._msgs = [{"role": "user", "content": message}]
            self._files = files or {}

        def list_messages(self):
            return self._msgs

        def list_files(self):
            return list(self._files.keys())

        def read_file(self, name):
            return self._files.get(name, "")

        def add_reply(self, text):
            print("\n" + "─" * 60)
            print(text)

    _CSV = """\
CustomerID,Age,Tenure,MonthlyCharges,HasPartner,Churn
1,29,12,65.5,yes,yes
2,45,36,89.0,no,no
3,52,60,120.0,yes,no
4,23,3,45.0,no,yes
5,38,24,75.0,yes,no
6,61,72,95.0,no,no
7,27,6,55.0,no,yes
8,34,18,70.0,yes,yes
9,48,48,110.0,no,no
10,55,60,130.0,yes,no
11,31,9,60.0,no,yes
12,42,30,85.0,yes,no
13,25,3,40.0,no,yes
14,67,84,140.0,yes,no
15,36,15,72.0,no,yes
"""

    print("=== Test 1: inline CSV ===")
    run(_FakeEnv(f"Build a model to predict Churn from this data:\n\n{_CSV}"))

    print("\n=== Test 2: uploaded file ===")
    run(_FakeEnv("predict Price", files={"houses.csv": """\
SquareFeet,Bedrooms,Bathrooms,Age,Price
1500,3,2,10,250000
2200,4,3,5,420000
900,2,1,30,120000
1800,3,2,8,310000
3000,5,4,2,580000
1100,2,1,25,145000
2500,4,3,3,490000
1300,3,2,15,215000
2800,4,3,1,540000
750,1,1,40,95000
1650,3,2,12,275000
2100,4,2,7,380000
1050,2,1,28,135000
1950,3,3,6,335000
3200,5,4,1,620000
"""}))

    print("\n=== Test 3: no CSV ===")
    run(_FakeEnv("Hello, what can you do?"))

    print("\n=== Test 4: no target ===")
    run(_FakeEnv(f"Here's some data:\n\n{_CSV}"))
