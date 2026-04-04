# GlassBox-AutoML · Module I: Automated EDA (The Inspector)

## Complete Technical Documentation

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Installation](#3-installation)
4. [Module Reference](#4-module-reference)
   - 4.1 [Math Utilities (`math_utils`)](#41-math-utilities)
   - 4.2 [Auto-Typer (`auto_typer`)](#42-auto-typer)
   - 4.3 [Statistical Profiler (`stats`)](#43-statistical-profiler)
   - 4.4 [Correlation Analyzer (`correlation`)](#44-correlation-analyzer)
   - 4.5 [Outlier Detector (`outliers`)](#45-outlier-detector)
   - 4.6 [Inspector Orchestrator (`inspector`)](#46-inspector-orchestrator)
5. [Usage Examples](#5-usage-examples)
6. [JSON Report Schema](#6-json-report-schema)
7. [Testing](#7-testing)
8. [Design Decisions](#8-design-decisions)
9. [IronClaw Agent Integration](#9-ironclaw-agent-integration)

---

## 1. Overview

**The Inspector** is Module I of the GlassBox-AutoML library. It performs a
**non-destructive, automated audit** of raw datasets, producing a comprehensive
JSON report that can be consumed by an IronClaw (NEAR AI) agent or any
downstream pipeline.

### Key Principles

| Principle            | Implementation                                       |
| -------------------- | ---------------------------------------------------- |
| Zero-dependency core | Only NumPy — no Scikit-Learn, Pandas, or SciPy       |
| Transparency         | Every formula is manually implemented and documented  |
| Non-destructive      | The original data is **never** modified               |
| WASM-ready           | Pure Python + NumPy — no C extensions beyond NumPy    |
| JSON-first output    | Every result serialises to a plain JSON object        |

### What the Inspector Computes

```
Raw Data ──► Auto-Typing ──► Statistical Profiling ──► Correlation ──► Outlier Detection ──► JSON Report
```

1. **Auto-Typing** — classifies columns as Numerical, Categorical, or Boolean.
2. **Statistical Profiling** — mean, median, mode, std, skewness, kurtosis, percentiles.
3. **Pearson Correlation** — full matrix + collinearity warnings.
4. **IQR Outlier Detection** — flags/caps points beyond 1.5 × IQR fences.

---

## 2. Architecture

```
glassbox/
├── __init__.py              # Package root — exports Inspector
├── eda/
│   ├── __init__.py          # EDA sub-package
│   ├── math_utils.py        # Low-level statistical primitives
│   ├── auto_typer.py        # Column type inference engine
│   ├── stats.py             # Descriptive statistics profiler
│   ├── correlation.py       # Pearson correlation matrix
│   ├── outliers.py          # IQR-based outlier detection
│   └── inspector.py         # Orchestrator (chains all modules)
tests/
├── __init__.py
├── test_math_utils.py       # Unit tests — math primitives
├── test_auto_typer.py       # Unit tests — type detection
├── test_stats.py            # Unit tests — stat profiler
├── test_correlation.py      # Unit tests — correlation
├── test_outliers.py         # Unit tests — outlier detection
└── test_inspector.py        # Integration tests — full pipeline
demo_eda.py                  # End-to-end demonstration script
pyproject.toml               # Build & dependency configuration
README.md                    # This documentation
```

### Data Flow Diagram

```
                    ┌──────────────────────────────────────────┐
                    │            Inspector.run(data)            │
                    └──────────┬───────────────────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │     AutoTyper        │
                    │  detect(data, hdrs)  │
                    └──────────┬───────────┘
                               │ List[ColumnTypeInfo]
                    ┌──────────▼───────────┐
                    │    StatProfiler      │
                    │  profile(data, ...)  │──── uses math_utils.*
                    └──────────┬───────────┘
                               │ List[ColumnStats]
                    ┌──────────▼───────────┐
                    │ CorrelationAnalyzer   │
                    │  analyze(data, ...)  │──── uses _pearson_r()
                    └──────────┬───────────┘
                               │ CorrelationResult
                    ┌──────────▼───────────┐
                    │  OutlierDetector     │
                    │  detect(data, ...)   │──── uses manual_percentile()
                    └──────────┬───────────┘
                               │ List[OutlierReport]
                    ┌──────────▼───────────┐
                    │      EDAReport       │
                    │     .to_json()       │
                    └──────────────────────┘
```

---

## 3. Installation

```bash
# Clone or copy the project
cd Part_EDA_Ai

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run demo
python demo_eda.py
```

### Requirements

- **Python** ≥ 3.11
- **NumPy** ≥ 1.24
- **pytest** ≥ 7.0 (dev only)

---

## 4. Module Reference

### 4.1 Math Utilities

**File:** `glassbox/eda/math_utils.py`

Every function operates on 1-D NumPy arrays, ignores NaN values, and
implements the formula from scratch (no `np.mean`, `np.std`, etc.).

#### `manual_mean(arr) → float`

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

| Parameter | Type         | Description               |
| --------- | ------------ | ------------------------- |
| `arr`     | `np.ndarray` | 1-D numeric array         |
| **Return**| `float`      | Arithmetic mean           |

#### `manual_median(arr) → float`

Returns the middle value of the sorted array. For even-length arrays,
returns the average of the two central values.

#### `manual_mode(arr) → object`

Returns the most frequent value. Ties are broken by returning the smallest
value. Works for both numeric and string arrays.

#### `manual_variance(arr, *, ddof=0) → float`

$$\sigma^2 = \frac{1}{n - \text{ddof}} \sum_{i=1}^{n}(x_i - \bar{x})^2$$

| Parameter | Type  | Description                                |
| --------- | ----- | ------------------------------------------ |
| `ddof`    | `int` | 0 = population variance, 1 = sample (Bessel) |

#### `manual_std(arr, *, ddof=0) → float`

$$\sigma = \sqrt{\text{Var}(x)}$$

#### `manual_skewness(arr) → float`

Adjusted Fisher–Pearson skewness coefficient:

$$G_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{s}\right)^3$$

| Value       | Interpretation        |
| ----------- | --------------------- |
| $G_1 > 0$   | Right-skewed (tail right) |
| $G_1 = 0$   | Symmetric              |
| $G_1 < 0$   | Left-skewed (tail left)  |

Requires at least **3** data points.

#### `manual_kurtosis(arr) → float`

Excess kurtosis (Fisher's definition — normal distribution = 0):

$$K = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum\left(\frac{x_i-\bar{x}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}$$

| Value    | Interpretation                      |
| -------- | ----------------------------------- |
| $K > 0$  | Leptokurtic (heavy tails, peaked)   |
| $K = 0$  | Mesokurtic (normal-like)            |
| $K < 0$  | Platykurtic (light tails, flat)     |

Requires at least **4** data points.

#### `manual_percentile(arr, q) → float`

Linear interpolation percentile matching NumPy's default method.

| Parameter | Type    | Description                  |
| --------- | ------- | ---------------------------- |
| `q`       | `float` | Percentile in `[0, 100]`    |

---

### 4.2 Auto-Typer

**File:** `glassbox/eda/auto_typer.py`

Classifies each column into one of three semantic types:

| Type            | Detection Rule                                                     |
| --------------- | ------------------------------------------------------------------ |
| **Numerical**   | Column can be cast to `float64` and has sufficient cardinality     |
| **Categorical** | String/object column, OR low-cardinality numeric (< 5 % unique)   |
| **Boolean**     | Exactly 2 unique non-NaN values that match boolean patterns        |

#### Class: `AutoTyper`

```python
AutoTyper(
    categorical_cardinality_ratio=0.05,  # Threshold for numeric → categorical
    categorical_max_unique=20,           # Max unique values for reclassification
    bool_values=None,                    # Extra boolean string literals
)
```

**Method:** `detect(data, headers=None) → List[ColumnTypeInfo]`

#### Dataclass: `ColumnTypeInfo`

| Field           | Type   | Description                           |
| --------------- | ------ | ------------------------------------- |
| `name`          | `str`  | Column header                        |
| `inferred_type` | `str`  | `"numerical"`, `"categorical"`, `"boolean"` |
| `dtype`         | `str`  | Raw NumPy dtype string               |
| `n_unique`      | `int`  | Unique non-NaN values                |
| `n_missing`     | `int`  | NaN / None count                     |
| `sample_values` | `list` | Up to 5 example values               |

#### Boolean Recognition

Built-in patterns (case-insensitive):

- **Truthy:** `true`, `yes`, `1`, `t`, `y`, `on`
- **Falsy:** `false`, `no`, `0`, `f`, `n`, `off`
- Numeric: `0` / `1`, `0.0` / `1.0`

---

### 4.3 Statistical Profiler

**File:** `glassbox/eda/stats.py`

Computes 16 descriptive statistics for every numeric column.

#### Class: `StatProfiler`

```python
StatProfiler(numeric_types={"numerical"})
```

**Method:** `profile(data, headers=None, type_map=None) → List[ColumnStats]`

| Parameter   | Description                                              |
| ----------- | -------------------------------------------------------- |
| `type_map`  | `{col_name: inferred_type}` — filters non-numeric cols  |

#### Dataclass: `ColumnStats`

| Field      | Type           | Description                        |
| ---------- | -------------- | ---------------------------------- |
| `name`     | `str`          | Column name                        |
| `count`    | `int`          | Non-NaN values                     |
| `missing`  | `int`          | NaN count                          |
| `mean`     | `float | None` | Arithmetic mean                    |
| `median`   | `float | None` | Median                             |
| `mode`     | `Any | None`   | Most frequent value                |
| `std`      | `float | None` | Sample standard deviation (ddof=1) |
| `variance` | `float | None` | Sample variance (ddof=1)           |
| `min`      | `float | None` | Minimum                            |
| `max`      | `float | None` | Maximum                            |
| `range`    | `float | None` | max − min                          |
| `q1`       | `float | None` | 25th percentile                    |
| `q3`       | `float | None` | 75th percentile                    |
| `iqr`      | `float | None` | Interquartile range                |
| `skewness` | `float | None` | Fisher's adjusted skewness         |
| `kurtosis` | `float | None` | Excess (Fisher) kurtosis           |

Statistics are `None` when insufficient data exists (e.g., skew needs ≥ 3 points).

---

### 4.4 Correlation Analyzer

**File:** `glassbox/eda/correlation.py`

Computes the full **Pearson Correlation Matrix** from scratch.

$$r_{xy} = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2} \cdot \sqrt{\sum(y_i - \bar{y})^2}}$$

#### Class: `CorrelationAnalyzer`

```python
CorrelationAnalyzer(threshold=0.90)
```

**Method:** `analyze(data, headers=None, numeric_indices=None) → CorrelationResult`

#### Dataclass: `CorrelationResult`

| Field          | Type                    | Description                              |
| -------------- | ----------------------- | ---------------------------------------- |
| `matrix`       | `np.ndarray (k × k)`   | Full Pearson correlation matrix          |
| `column_names` | `List[str]`             | Column labels for each axis              |
| `high_pairs`   | `List[CorrelationPair]` | Pairs with $|r| \geq$ threshold          |

#### Dataclass: `CorrelationPair`

| Field   | Type    | Description                |
| ------- | ------- | -------------------------- |
| `col_a` | `str`   | First column name          |
| `col_b` | `str`   | Second column name         |
| `r`     | `float` | Pearson correlation value  |

#### NaN Handling

For each pair $(i, j)$, only rows where **both** values are non-NaN are used
(pairwise complete observations).

#### Collinearity Warning

When $|r| \geq \text{threshold}$ (default 0.90), the pair is flagged. The Inspector
generates a human-readable warning recommending removal of one feature.

---

### 4.5 Outlier Detector

**File:** `glassbox/eda/outliers.py`

Uses the **Interquartile Range (IQR)** method:

$$\text{Lower Fence} = Q_1 - k \cdot IQR$$
$$\text{Upper Fence} = Q_3 + k \cdot IQR$$

where $k = 1.5$ (Tukey's rule) by default.

A data point $x$ is an **outlier** if:
$$x < Q_1 - k \cdot IQR \quad \text{or} \quad x > Q_3 + k \cdot IQR$$

#### Class: `OutlierDetector`

```python
OutlierDetector(k=1.5)
```

| Method | Description | Destructive? |
| ------ | ----------- | ------------ |
| `detect(data, headers, numeric_indices)` | Flag outliers per column | No |
| `cap(data, headers, numeric_indices)` | Return clipped **copy** | No (copy) |

#### Dataclass: `OutlierReport`

| Field             | Type        | Description                          |
| ----------------- | ----------- | ------------------------------------ |
| `name`            | `str`       | Column name                          |
| `q1`              | `float`     | 25th percentile                      |
| `q3`              | `float`     | 75th percentile                      |
| `iqr`             | `float`     | Q3 − Q1                             |
| `lower_fence`     | `float`     | Q1 − k × IQR                        |
| `upper_fence`     | `float`     | Q3 + k × IQR                        |
| `n_outliers_low`  | `int`       | Count below lower fence              |
| `n_outliers_high` | `int`       | Count above upper fence              |
| `n_total`         | `int`       | Total non-NaN values                 |
| `outlier_indices` | `List[int]` | Row indices of outlier values        |
| `outlier_pct`     | `float`     | Percentage of values that are outliers |

---

### 4.6 Inspector Orchestrator

**File:** `glassbox/eda/inspector.py`

The **Inspector** is the top-level entry point. It chains all sub-modules and
returns a unified `EDAReport`.

#### Class: `Inspector`

```python
from glassbox import Inspector

inspector = Inspector(config=InspectorConfig(...))
report = inspector.run(data, headers)
json_str = report.to_json()
```

#### Dataclass: `InspectorConfig`

| Parameter                       | Default | Description                              |
| ------------------------------- | ------- | ---------------------------------------- |
| `categorical_cardinality_ratio` | `0.05`  | Numeric → categorical threshold          |
| `categorical_max_unique`        | `20`    | Max unique for categorical reclassification |
| `correlation_threshold`         | `0.90`  | Collinearity flag threshold              |
| `outlier_k`                     | `1.5`   | IQR fence multiplier                     |
| `cap_outliers`                  | `False` | Include capped data in report            |

#### Pipeline Steps

```
Step 1: AutoTyper.detect()      → column_types, type_map
Step 2: StatProfiler.profile()  → statistics (numeric cols only)
Step 3: CorrelationAnalyzer()   → correlation matrix + high pairs
Step 4: OutlierDetector()       → outlier reports per numeric col
Step 5: Generate warnings       → collinearity, outlier %, missing values
Step 6: Assemble EDAReport      → .to_json()
```

#### Automatic Warnings

The Inspector generates warnings for:
- **Collinearity:** Pairs with $|r| \geq$ threshold
- **Outlier density:** Columns with > 5 % outliers
- **Missing values:** Any column with NaN values (count + percentage)

---

## 5. Usage Examples

### Minimal Example

```python
import numpy as np
from glassbox import Inspector

data = np.array([
    [25, 50000],
    [30, 60000],
    [35, 55000],
    [40, 80000],
    [50, 120000],
], dtype=float)

report = Inspector().run(data, headers=["age", "salary"])
print(report.to_json())
```

### Mixed-Type Dataset

```python
import numpy as np
from glassbox.eda.inspector import Inspector, InspectorConfig

# Object array for mixed types
data = np.array([
    [25,  "Engineering", "yes", 8.5],
    [30,  "Sales",       "no",  6.2],
    [35,  "HR",          "yes", 7.8],
    [40,  "Engineering", "no",  9.1],
    [50,  "Marketing",   "yes", 5.5],
], dtype=object)

config = InspectorConfig(correlation_threshold=0.7, outlier_k=2.0)
inspector = Inspector(config=config)
report = inspector.run(data, ["age", "dept", "manager", "rating"])

# Access structured results
for ct in report.column_types:
    print(f"  {ct['name']:15s} → {ct['inferred_type']}")
```

### Individual Module Usage

```python
from glassbox.eda.auto_typer import AutoTyper
from glassbox.eda.stats import StatProfiler
from glassbox.eda.correlation import CorrelationAnalyzer
from glassbox.eda.outliers import OutlierDetector

# 1. Type detection only
typer = AutoTyper()
types = typer.detect(data, headers)

# 2. Stats only
profiler = StatProfiler()
stats = profiler.profile(data, headers)

# 3. Correlation only
ca = CorrelationAnalyzer(threshold=0.85)
corr = ca.analyze(data, headers)

# 4. Outlier detection only
det = OutlierDetector(k=1.5)
reports = det.detect(data, headers)
capped = det.cap(data, headers)  # non-destructive capping
```

---

## 6. JSON Report Schema

```json
{
  "metadata": {
    "n_rows": 200,
    "n_cols": 6,
    "elapsed_seconds": 0.0342
  },
  "type_summary": {
    "numerical": 3,
    "categorical": 1,
    "boolean": 2
  },
  "column_types": [
    {
      "name": "age",
      "inferred_type": "numerical",
      "dtype": "object",
      "n_unique": 52,
      "n_missing": 10,
      "sample_values": [25.0, 30.0, 35.0, 40.0, 50.0]
    }
  ],
  "statistics": [
    {
      "name": "age",
      "count": 190,
      "missing": 10,
      "mean": 45.263,
      "median": 45.0,
      "mode": 42.0,
      "std": 16.234,
      "variance": 263.543,
      "min": 18.0,
      "max": 74.0,
      "range": 56.0,
      "q1": 31.0,
      "q3": 59.0,
      "iqr": 28.0,
      "skewness": -0.0124,
      "kurtosis": -1.1842
    }
  ],
  "correlation": {
    "column_names": ["age", "salary", "experience"],
    "matrix": [
      [1.0, 0.12, 0.95],
      [0.12, 1.0, 0.08],
      [0.95, 0.08, 1.0]
    ],
    "high_correlation_pairs": [
      {"col_a": "age", "col_b": "experience", "r": 0.9523}
    ]
  },
  "outliers": [
    {
      "name": "salary",
      "q1": 62000.0,
      "q3": 137000.0,
      "iqr": 75000.0,
      "lower_fence": -50500.0,
      "upper_fence": 249500.0,
      "n_outliers_low": 1,
      "n_outliers_high": 1,
      "n_outliers_total": 2,
      "n_total": 200,
      "outlier_pct": 1.0,
      "outlier_indices": [0, 1]
    }
  ],
  "warnings": [
    "High correlation detected between 'age' and 'experience' (r = 0.9523). Consider removing one to reduce multicollinearity.",
    "Column 'age' has 10 missing values (5.0%)."
  ]
}
```

---

## 7. Testing

### Running Tests

```bash
# All tests
pytest

# Verbose with coverage
pytest -v --cov=glassbox --cov-report=term-missing

# Single module
pytest tests/test_math_utils.py -v
pytest tests/test_inspector.py -v
```

### Test Coverage Summary

| Test File               | Module Tested    | Tests | Focus Areas                              |
| ----------------------- | ---------------- | ----- | ---------------------------------------- |
| `test_math_utils.py`   | `math_utils`     | 30+   | All 8 functions, edge cases, NaN, NumPy comparison |
| `test_auto_typer.py`   | `auto_typer`     | 11    | Numeric/categorical/boolean detection, missing values |
| `test_stats.py`        | `stats`          | 7     | Profiling, NaN handling, type filtering  |
| `test_correlation.py`  | `correlation`    | 9     | Perfect/zero correlation, NaN, flagging  |
| `test_outliers.py`     | `outliers`       | 12    | Detection, capping, indices, k values    |
| `test_inspector.py`    | `inspector`      | 11    | Full pipeline, JSON output, config       |

### Validation Strategy

Each math function is tested against:
1. **Analytical known values** (e.g., mean of [1,2,3,4,5] = 3.0)
2. **NumPy reference** (e.g., `manual_mean(arr) ≈ np.nanmean(arr)`)
3. **Edge cases** (empty arrays, all-NaN, single element, constant arrays)
4. **Statistical properties** (normal kurtosis ≈ 0, symmetric skewness ≈ 0)

---

## 8. Design Decisions

### Why No Pandas?

Pandas is not WASM-compatible in all environments and introduces a heavy
dependency chain. By working directly with NumPy arrays:
- The library compiles cleanly to WASM via Pyodide.
- Memory footprint is minimal.
- There are no hidden type-coercion surprises.

### Why Manual Statistics?

The project spec requires "white-box" transparency. Every formula is visible
in the source code, making it possible for an IronClaw agent to:
- **Explain** how a statistic was computed.
- **Audit** the calculation step by step.
- **Cite** the exact formula used.

### Why Pairwise NaN Handling in Correlation?

Listwise deletion (dropping any row with any NaN) can dramatically reduce
sample size. Pairwise complete observations preserve maximum data per pair.

### Why Adjusted (Fisher) Skewness and Kurtosis?

The adjusted formulas correct for sample-size bias, which is important for
the small-to-medium datasets typical in agentic workflows.

### Transformer Pattern

All future modules (Phase 2+) will implement `fit()`, `transform()`, and
`fit_transform()` methods. The EDA module is read-only, so it uses `detect()`,
`profile()`, `analyze()`, and `run()` instead — but the dataclass output
follows the same serialisation pattern (`to_dict()`, `to_json()`).

---

## 9. IronClaw Agent Integration

The Inspector is designed to be called as a **tool** by an IronClaw agent:

```
User: "Analyze this CSV and tell me what's interesting."

Agent → Tool Call:
    inspector = Inspector()
    report = inspector.run_json(data, headers)

Agent → Response:
    "I found 6 columns: 3 numerical, 1 categorical, 2 boolean.
     The 'age' and 'experience' columns are highly correlated (r=0.95) —
     you may want to drop one. The 'salary' column has 2 outliers
     (one at $999,999 and one at -$5,000). I also see 10 missing
     values in the 'age' column (5%)."
```

The JSON report is designed to be:
1. **Machine-parseable** — structured keys, consistent types.
2. **Agent-explainable** — warnings are written in natural language.
3. **Deterministic** — same input always produces same output.

---

*GlassBox-AutoML v0.1.0 — Module I: Automated EDA (The Inspector)*
