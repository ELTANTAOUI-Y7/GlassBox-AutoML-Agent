# GlassBox-AutoML · Module I: Automated EDA + Module II: Preprocessing + Module IV: Optimization

## Complete Technical Documentation

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Installation](#3-installation)
4. [Module I — EDA (The Inspector)](#4-module-i--eda-the-inspector)
   - 4.1 [Math Utilities](#41-math-utilities)
   - 4.2 [Auto-Typer](#42-auto-typer)
   - 4.3 [Statistical Profiler](#43-statistical-profiler)
   - 4.4 [Correlation Analyzer](#44-correlation-analyzer)
   - 4.5 [Outlier Detector](#45-outlier-detector)
   - 4.6 [Inspector Orchestrator](#46-inspector-orchestrator)
5. [Module II — Preprocessing (The Cleaner)](#5-module-ii--preprocessing-the-cleaner)
   - 5.1 [Simple Imputer](#51-simple-imputer)
   - 5.2 [Scalers](#52-scalers)
   - 5.3 [Encoders](#53-encoders)
   - 5.4 [Cleaner Orchestrator](#54-cleaner-orchestrator)
6. [Module IV — Optimization (The Orchestrator)](#6-module-iv--optimization-the-orchestrator)
   - 6.1 [KFold Splitter](#61-kfold-splitter)
   - 6.2 [Scoring Metrics](#62-scoring-metrics)
   - 6.3 [Grid Search](#63-grid-search)
   - 6.4 [Random Search](#64-random-search)
   - 6.5 [Orchestrator](#65-orchestrator)
7. [Usage Examples](#7-usage-examples)
8. [JSON Report Schemas](#8-json-report-schemas)
9. [Testing](#9-testing)
10. [Design Decisions](#10-design-decisions)
11. [IronClaw Agent Integration](#11-ironclaw-agent-integration)

---

## 1. Overview

**GlassBox-AutoML** is a NumPy-only, white-box automated machine learning library.
Every formula is implemented from scratch — no Scikit-Learn, no Pandas.

| Module | Class | Role |
| ------ | ----- | ---- |
| **Module I** | `Inspector` | Non-destructive audit of raw data → JSON EDA report |
| **Module II** | `Cleaner` | Automated preprocessing → cleaned array + JSON audit trail |
| **Module IV** | `Orchestrator` | Hyperparameter search (Grid / Random) via K-fold CV → JSON optimization report |

### Key Principles

| Principle | Implementation |
| --------- | -------------- |
| Zero-dependency core | Only NumPy — no Scikit-Learn, Pandas, or SciPy |
| Transparency | Every formula is manually implemented and documented |
| Non-destructive | Original data is **never** modified |
| WASM-ready | Pure Python + NumPy — no C extensions beyond NumPy |
| JSON-first output | Every result serialises to a plain JSON object |

### Full Pipeline

```
Raw Data
  ──► Inspector (EDA)         ──► EDAReport (JSON)
  ──► Cleaner (Preprocessing) ──► Cleaned Array + PreprocessingReport (JSON)
  ──► Orchestrator (Phase IV) ──► Best Params + OptimizationReport (JSON)
```

---

## 2. Architecture

```
GlassBox-AutoML-Agent/
├── eda/
│   ├── math_utils.py        # Statistical primitives (mean, std, skew, …)
│   ├── auto_typer.py        # Column type inference (numerical/categorical/boolean)
│   ├── stats.py             # Descriptive statistics profiler
│   ├── correlation.py       # Pearson correlation matrix
│   ├── outliers.py          # IQR-based outlier detection
│   └── inspector.py         # EDA orchestrator → EDAReport
├── preprocessing/
│   ├── imputer.py           # SimpleImputer (mean/median/mode/constant)
│   ├── scalers.py           # MinMaxScaler + StandardScaler
│   ├── encoders.py          # OneHotEncoder + LabelEncoder
│   └── cleaner.py           # Preprocessing orchestrator → CleanerResult
├── optimization/
│   ├── kfold.py             # KFoldSplitter + cross_val_score
│   ├── scoring.py           # Evaluation metrics from scratch (accuracy, MSE, R², …)
│   ├── grid_search.py       # GridSearch — exhaustive Cartesian-product search
│   ├── random_search.py     # RandomSearch — stochastic sampling + time budget
│   └── orchestrator.py      # Optimization orchestrator → OptimizationReport
├── demo_eda.py              # End-to-end EDA demo
├── demo_preprocessing.py    # End-to-end preprocessing demo
├── demo_optimization.py     # End-to-end optimization demo
└── pyproject.toml
```

### Data Flow

```
Inspector.run(data)                     Cleaner.run(data)
      │                                       │
  AutoTyper ──► type_map ────────────► AutoTyper (or reuse type_map)
      │                                       │
  StatProfiler                         SimpleImputer
      │                                       │
  CorrelationAnalyzer                  MinMaxScaler / StandardScaler
      │                                       │
  OutlierDetector                      OneHotEncoder / LabelEncoder
      │                                       │
  EDAReport.to_json()              CleanerResult (data + PreprocessingReport)
```

---

## 3. Installation

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS / Linux

# Install the package + dev dependencies
pip install -e ".[dev]"

# Run demos
python3 demo_eda.py
python3 demo_preprocessing.py

# Run tests
pytest
```

**Requirements:** Python ≥ 3.11 · NumPy ≥ 1.24 · pytest ≥ 7.0 (dev)

---

## 4. Module I — EDA (The Inspector)

The Inspector performs a **non-destructive, automated audit** of raw data and
produces a comprehensive JSON report covering types, statistics, correlation,
and outliers.

### 4.1 Math Utilities

**File:** `eda/math_utils.py`

All functions operate on 1-D NumPy arrays, ignore NaN, and are implemented
from scratch (no `np.mean`, `np.std`, etc.).

| Function | Formula | Notes |
| -------- | ------- | ----- |
| `manual_mean(arr)` | $\bar{x} = \frac{1}{n}\sum x_i$ | |
| `manual_median(arr)` | Middle value (linear interp for even-length) | |
| `manual_mode(arr)` | Most frequent value; ties → smallest | Works on strings |
| `manual_variance(arr, ddof=0)` | $\sigma^2 = \frac{1}{n-\text{ddof}}\sum(x_i-\bar{x})^2$ | |
| `manual_std(arr, ddof=0)` | $\sqrt{\text{Var}(x)}$ | |
| `manual_skewness(arr)` | Adjusted Fisher–Pearson $G_1$ | Requires ≥ 3 points |
| `manual_kurtosis(arr)` | Excess (Fisher) kurtosis | Requires ≥ 4 points |
| `manual_percentile(arr, q)` | Linear interpolation | Matches NumPy default |

---

### 4.2 Auto-Typer

**File:** `eda/auto_typer.py`

Classifies each column into one of three semantic types:

| Type | Detection Rule |
| ---- | -------------- |
| **Numerical** | Castable to `float64` with sufficient cardinality |
| **Categorical** | String/object, or low-cardinality numeric (< 5 % unique) |
| **Boolean** | Exactly 2 unique non-NaN values matching boolean patterns |

```python
AutoTyper(
    categorical_cardinality_ratio=0.05,
    categorical_max_unique=20,
    bool_values=None,  # extra boolean literals
)
```

**Method:** `detect(data, headers) → List[ColumnTypeInfo]`

**`ColumnTypeInfo` fields:** `name`, `inferred_type`, `dtype`, `n_unique`, `n_missing`, `sample_values`

---

### 4.3 Statistical Profiler

**File:** `eda/stats.py`

Computes 16 descriptive statistics per numeric column.

```python
StatProfiler(numeric_types={"numerical"})
```

**Method:** `profile(data, headers, type_map) → List[ColumnStats]`

Fields: `name`, `count`, `missing`, `mean`, `median`, `mode`, `std`, `variance`, `min`, `max`, `range`, `q1`, `q3`, `iqr`, `skewness`, `kurtosis`

---

### 4.4 Correlation Analyzer

**File:** `eda/correlation.py`

Computes the full Pearson correlation matrix from scratch using pairwise complete observations.

$$r_{xy} = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2} \cdot \sqrt{\sum(y_i - \bar{y})^2}}$$

```python
CorrelationAnalyzer(threshold=0.90)
```

**Method:** `analyze(data, headers, numeric_indices) → CorrelationResult`

Generates a collinearity warning for any pair with $|r| \geq$ threshold.

---

### 4.5 Outlier Detector

**File:** `eda/outliers.py`

Uses Tukey's IQR rule ($k = 1.5$ by default):

$$\text{Lower Fence} = Q_1 - k \cdot IQR \qquad \text{Upper Fence} = Q_3 + k \cdot IQR$$

```python
OutlierDetector(k=1.5)
```

| Method | Description | Destructive? |
| ------ | ----------- | ------------ |
| `detect(data, headers, numeric_indices)` | Flag outliers per column | No |
| `cap(data, headers, numeric_indices)` | Return clipped copy | No (copy) |

---

### 4.6 Inspector Orchestrator

**File:** `eda/inspector.py`

```python
from eda.inspector import Inspector, InspectorConfig

config = InspectorConfig(correlation_threshold=0.9, outlier_k=1.5, cap_outliers=False)
report = Inspector(config).run(data, headers)
print(report.to_json())
```

**Pipeline steps:**
```
Step 1: AutoTyper.detect()      → column_types, type_map
Step 2: StatProfiler.profile()  → statistics
Step 3: CorrelationAnalyzer()   → correlation matrix + high pairs
Step 4: OutlierDetector()       → outlier reports
Step 5: Generate warnings       → collinearity, outlier %, missing values
Step 6: Assemble EDAReport      → .to_json()
```

**Automatic warnings:** collinearity ($|r| \geq$ threshold), outlier density (> 5 %), missing values.

---

## 5. Module II — Preprocessing (The Cleaner)

The Cleaner transforms raw data into a model-ready array. It reuses the EDA
`AutoTyper` so no type detection is duplicated when the Inspector has already run.

### 5.1 Simple Imputer

**File:** `preprocessing/imputer.py`

Fills missing values using column statistics. Follows the `fit() / transform() / fit_transform()` transformer pattern.

#### Strategies

| Strategy | Numerical | Categorical / Boolean |
| -------- | --------- | --------------------- |
| `"mean"` | Arithmetic mean | Falls back to mode |
| `"median"` | Median | Falls back to mode |
| `"mode"` | Most frequent value | Most frequent value |
| `"constant"` | User-supplied `fill_value` | User-supplied `fill_value` |

#### Class: `SimpleImputer`

```python
SimpleImputer(strategy="mean", fill_value=None)
```

| Method | Description |
| ------ | ----------- |
| `fit(data, headers, type_map)` | Compute per-column fill values |
| `transform(data, headers)` | Return a filled copy + `List[ImputationSummary]` |
| `fit_transform(data, headers, type_map)` | Fit then transform in one step |

#### Dataclass: `ImputationSummary`

| Field | Type | Description |
| ----- | ---- | ----------- |
| `column` | `str` | Column name |
| `strategy` | `str` | Strategy used |
| `fill_value` | `Any` | Value inserted |
| `n_filled` | `int` | Cells filled |

---

### 5.2 Scalers

**File:** `preprocessing/scalers.py`

Both scalers only act on `numerical` columns and return a **copy** of the data.
They use `manual_mean` and `manual_std` from `eda/math_utils.py`.

#### MinMaxScaler

Scales to a fixed range `[low, high]` (default `[0, 1]`):

$$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \times (\text{high} - \text{low}) + \text{low}$$

Constant columns (range = 0) are set to `low` to avoid division by zero.

```python
MinMaxScaler(feature_range=(0.0, 1.0))
```

#### StandardScaler

Standardises to zero mean and unit variance (sample std, `ddof=1`):

$$x_{\text{scaled}} = \frac{x - \bar{x}}{\sigma}$$

Constant columns (std = 0) are output as `0.0`.

```python
StandardScaler()
```

Both follow `fit() / transform() / fit_transform()`.

#### Dataclass: `ScalerSummary`

| Field | MinMax | Standard |
| ----- | ------ | -------- |
| `column` | column name | column name |
| `scaler_type` | `"minmax"` | `"standard"` |
| `param_a` | `x_min` | mean (μ) |
| `param_b` | `x_max` | std (σ) |

---

### 5.3 Encoders

**File:** `preprocessing/encoders.py`

Both encoders target `categorical` columns by default, return a **copy**, and
follow the transformer pattern.

#### OneHotEncoder

Creates one binary indicator column per unique category, removing the original.
Categories are sorted lexicographically for deterministic output.
Unknown categories at transform-time become all-zero rows.

```python
OneHotEncoder(drop_first=False, target_types={"categorical"})
```

```
dept = ["Eng", "HR", "Eng"]
→ dept_Eng = [1, 0, 1],  dept_HR = [0, 1, 0]
```

| Method | Returns |
| ------ | ------- |
| `fit(data, headers, type_map)` | `self` |
| `transform(data, headers)` | `(new_data, new_headers, List[EncoderSummary])` |
| `fit_transform(...)` | same as transform |
| `get_feature_names()` | all indicator column names |

#### LabelEncoder

Maps each category to a non-negative integer (alphabetical order, 0-based).
Unknown categories at transform-time are mapped to `-1`.

```python
LabelEncoder(target_types={"categorical"})
```

```
size = ["S", "M", "L", "XL"]  →  [2, 1, 0, 3]   (L=0, M=1, S=2, XL=3)
```

| Method | Returns |
| ------ | ------- |
| `fit(data, headers, type_map)` | `self` |
| `transform(data, headers)` | `(new_data, List[EncoderSummary])` |
| `fit_transform(...)` | same as transform |
| `get_mapping(col_name)` | `{category: int}` dict |

#### Dataclass: `EncoderSummary`

| Field | Type | Description |
| ----- | ---- | ----------- |
| `column` | `str` | Original column name |
| `encoder_type` | `str` | `"onehot"` or `"label"` |
| `categories` | `List[str]` | Unique categories (sorted) |
| `output_columns` | `List[str]` | Resulting column names |
| `mapping` | `dict \| None` | Integer mapping (LabelEncoder only) |

---

### 5.4 Cleaner Orchestrator

**File:** `preprocessing/cleaner.py`

The Cleaner chains all preprocessing steps and returns a `CleanerResult`.

#### Class: `CleanerConfig`

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `imputer_strategy` | `"mean"` | `"mean"`, `"median"`, `"mode"`, `"constant"` |
| `imputer_fill_value` | `None` | Used when strategy is `"constant"` |
| `scale_numerical` | `True` | Whether to scale numerical columns |
| `scaler_type` | `"standard"` | `"standard"` or `"minmax"` |
| `minmax_range` | `(0.0, 1.0)` | Output range for MinMaxScaler |
| `encode_categorical` | `True` | Whether to encode categorical columns |
| `encoder_type` | `"onehot"` | `"onehot"` or `"label"` |
| `drop_first_ohe` | `False` | Drop first OHE indicator (avoid dummy trap) |

#### Class: `Cleaner`

```python
from preprocessing.cleaner import Cleaner, CleanerConfig

config = CleanerConfig(scaler_type="minmax", encoder_type="onehot")
cleaner = Cleaner(config)
result = cleaner.run(data, headers, type_map=type_map)  # type_map from Inspector (optional)
```

**`run(data, headers, type_map) → CleanerResult`**

Pass `type_map` from the EDA Inspector to avoid running AutoTyper twice.

**`fit_transform(data, headers, type_map) → (data, headers, report)`**

Convenience alias that unpacks `CleanerResult`.

#### Pipeline Steps

```
Step 1: AutoTyper.detect()          → column_types, type_map
Step 2: SimpleImputer.fit_transform() → fill missing values
Step 3: Scaler.fit_transform()       → normalise numerical columns
Step 4: Encoder.fit_transform()      → encode categorical columns
Step 5: Assemble PreprocessingReport → .to_json()
```

#### Dataclass: `CleanerResult`

| Field | Type | Description |
| ----- | ---- | ----------- |
| `data` | `np.ndarray` | Cleaned array |
| `headers` | `List[str]` | Column names (may differ from input after OHE) |
| `report` | `PreprocessingReport` | Full audit trail |

#### Dataclass: `PreprocessingReport`

Fields: `n_rows`, `n_cols_in`, `n_cols_out`, `elapsed_seconds`, `headers_in`, `headers_out`, `type_summary`, `column_types`, `imputation`, `scaling`, `encoding`, `steps_applied`, `warnings`

Methods: `to_dict()`, `to_json()`

**Automatic warnings:** missing values (will be imputed), constant columns (zero std / zero range), all-missing columns (imputation skipped).

---

## 6. Module IV — Optimization (The Orchestrator)

The Optimization module implements Phase IV of the GlassBox pipeline: automated hyperparameter search with K-fold cross-validation. It mirrors the architecture of the Inspector and Cleaner — a single `run()` call chains all sub-components and returns a JSON-serialisable `OptimizationReport`.

It is designed to work with **any estimator** that satisfies a two-method contract:

```python
model = MyEstimator(**hyperparams)
model.fit(X_train, y_train)     # X, y are NumPy arrays
y_pred = model.predict(X_val)   # returns NumPy array
```

This means it will plug directly into the Phase III Algorithm Zoo models once they are available.

### 6.1 KFold Splitter

**File:** `optimization/kfold.py`

Splits a dataset of *n* samples into K non-overlapping folds. The last fold receives any remainder samples.

$$\text{fold\_size}_i = \lfloor n / K \rfloor + \mathbf{1}[i < n \bmod K]$$

#### Class: `KFoldSplitter`

```python
KFoldSplitter(n_splits=5, shuffle=True, random_state=None)
```

| Method | Returns | Description |
| ------ | ------- | ----------- |
| `split(X, y=None)` | `Iterator[(train_idx, val_idx)]` | Yields index arrays for each fold |
| `get_n_splits()` | `int` | Number of configured folds |
| `get_fold_info(X)` | `List[FoldSplit]` | Metadata without yielding data |

#### Standalone helper: `cross_val_score`

```python
from glassbox.optimization.kfold import cross_val_score

scores = cross_val_score(
    MyModel, X, y,
    scoring_fn=accuracy_score,
    params={"max_depth": 5},
    cv=5,
    random_state=42,
)
# → np.ndarray of shape (5,)  — one score per fold
print(f"{scores.mean():.4f} ± {scores.std():.4f}")
```

---

### 6.2 Scoring Metrics

**File:** `optimization/scoring.py`

All metrics implemented from scratch using NumPy. Every function signature is `metric(y_true, y_pred) -> float` — higher is always considered better by the search strategies. Use the negated variants for error metrics.

#### Classification

| Function | Formula | Notes |
| -------- | ------- | ----- |
| `accuracy_score(y_true, y_pred)` | $\frac{1}{n}\sum\mathbf{1}[y_i = \hat{y}_i]$ | |
| `precision_score(y_true, y_pred, pos_label=1)` | $\frac{TP}{TP+FP}$ | Returns 0.0 if no positive predictions |
| `recall_score(y_true, y_pred, pos_label=1)` | $\frac{TP}{TP+FN}$ | Returns 0.0 if no positive ground-truth |
| `f1_score(y_true, y_pred, pos_label=1)` | $\frac{2 \cdot P \cdot R}{P+R}$ | Harmonic mean of precision and recall |
| `confusion_matrix(y_true, y_pred)` | K×K count matrix | Rows = true, columns = predicted |

#### Regression

| Function | Formula | Notes |
| -------- | ------- | ----- |
| `mean_squared_error(y_true, y_pred)` | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | |
| `mean_absolute_error(y_true, y_pred)` | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | |
| `r2_score(y_true, y_pred)` | $1 - \frac{SS_{res}}{SS_{tot}}$ | 1.0 = perfect, 0.0 = mean predictor |
| `neg_mean_squared_error(y_true, y_pred)` | $-\text{MSE}$ | Use with search strategies |
| `neg_mean_absolute_error(y_true, y_pred)` | $-\text{MAE}$ | Use with search strategies |

---

### 6.3 Grid Search

**File:** `optimization/grid_search.py`

Evaluates every combination in the Cartesian product of `param_grid`. Results are sorted by mean CV score descending. Failed candidates (estimator exceptions) are recorded with their error message rather than crashing the search.

#### Class: `GridSearch`

```python
GridSearch(
    scoring_fn,           # callable — higher is better
    cv=5,                 # number of folds
    shuffle=True,
    random_state=None,
    verbose=False,
)
```

| Method | Returns | Description |
| ------ | ------- | ----------- |
| `fit(X, y, estimator_class, param_grid)` | `GridSearchResult` | Run the full grid search |
| `best_params_` | `dict` | Best combination (after `fit`) |
| `best_score_` | `float` | Mean CV score of the best combination |

#### Dataclass: `GridSearchResult`

| Field | Type | Description |
| ----- | ---- | ----------- |
| `best_params` | `dict` | Winning hyperparameter combination |
| `best_score` | `float` | Mean CV score of the winner |
| `n_candidates` | `int` | Total combinations evaluated |
| `n_splits` | `int` | Number of folds |
| `elapsed_seconds` | `float` | Total wall-clock time |
| `all_results` | `List[CVResult]` | Per-combination results, sorted best-first |

#### Dataclass: `CVResult`

| Field | Description |
| ----- | ----------- |
| `params` | The hyperparameter combination |
| `fold_scores` | Score per fold (`nan` on failure) |
| `mean_score` | NaN-safe mean |
| `std_score` | NaN-safe standard deviation |
| `error` | Exception message if any fold failed, else `None` |

---

### 6.4 Random Search

**File:** `optimization/random_search.py`

Samples `n_iter` random combinations from `param_distributions` and evaluates each with K-fold CV. Supports a `time_budget` (seconds) to stop the search early — useful for the 120-second AutoML target in the project spec.

#### Parameter Distribution Specification

| Format | Behaviour | Example |
| ------ | --------- | ------- |
| `list` | Discrete uniform — one element picked at random | `"max_depth": [3, 5, 10, 20]` |
| `(low, high)` | Continuous uniform $U[\text{low}, \text{high}]$ | `"dropout": (0.1, 0.5)` |
| `(low, high, "log")` | Log-uniform — for parameters spanning orders of magnitude | `"lr": (1e-4, 0.1, "log")` |

#### Class: `RandomSearch`

```python
RandomSearch(
    scoring_fn,           # callable — higher is better
    cv=5,
    n_iter=20,            # number of random combinations to try
    time_budget=None,     # stop early after this many seconds
    shuffle=True,
    random_state=None,
    verbose=False,
)
```

| Method | Returns | Description |
| ------ | ------- | ----------- |
| `fit(X, y, estimator_class, param_distributions)` | `RandomSearchResult` | Run the random search |
| `best_params_` | `dict` | Best combination found |
| `best_score_` | `float` | Mean CV score of the best combination |

#### Dataclass: `RandomSearchResult`

Same fields as `GridSearchResult`, plus:

| Field | Type | Description |
| ----- | ---- | ----------- |
| `n_iter` | `int` | Iterations actually completed (≤ requested `n_iter`) |
| `time_budget_hit` | `bool` | `True` if the search stopped early |

---

### 6.5 Orchestrator

**File:** `optimization/orchestrator.py`

The unified entry point — equivalent to `Inspector` (Phase I) and `Cleaner` (Phase II). A single `run()` call dispatches to the correct search strategy and returns an `OptimizationReport`.

#### Class: `OrchestratorConfig`

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `search_type` | `"grid"` | `"grid"` (exhaustive) or `"random"` (stochastic) |
| `cv` | `5` | Number of cross-validation folds |
| `n_iter` | `20` | Iterations for RandomSearch (ignored for GridSearch) |
| `time_budget` | `None` | Wall-clock limit in seconds for RandomSearch |
| `shuffle` | `True` | Shuffle samples before forming folds |
| `random_state` | `None` | Reproducibility seed |
| `verbose` | `False` | Print per-candidate progress |

#### Class: `Orchestrator`

```python
from glassbox.optimization.orchestrator import Orchestrator, OrchestratorConfig

config = OrchestratorConfig(search_type="grid", cv=5, random_state=42)
orch = Orchestrator(config)

report = orch.run(
    X, y,
    estimator_class=MyModel,
    param_grid={"max_depth": [3, 5, 10]},
    scoring_fn=accuracy_score,
)
print(report.to_json())
```

| Method | Returns | Description |
| ------ | ------- | ----------- |
| `run(X, y, estimator_class, param_grid, scoring_fn)` | `OptimizationReport` | Run the full search pipeline |
| `fold_info(X)` | `List[dict]` | Preview fold layout without running a search |

#### Dataclass: `OptimizationReport`

| Field | Type | Description |
| ----- | ---- | ----------- |
| `search_type` | `str` | `"grid"` or `"random"` |
| `best_params` | `dict` | Winning hyperparameters |
| `best_score` | `float` | Mean CV score of the winner |
| `n_candidates_evaluated` | `int` | Combinations actually evaluated |
| `n_splits` | `int` | Number of CV folds |
| `elapsed_seconds` | `float` | Total wall-clock time |
| `scoring_fn_name` | `str` | Name of the scoring function |
| `time_budget_hit` | `bool` | Early stop due to time budget |
| `all_results` | `List[dict]` | Full per-candidate audit trail |
| `warnings` | `List[str]` | Failed candidates, early-stop notices |

Methods: `to_dict()`, `to_json()`

#### Pipeline Steps

```
Step 1: OrchestratorConfig validation
Step 2: GridSearch / RandomSearch.fit()
        └─ For each candidate combination:
               KFoldSplitter.split() → K folds
               estimator.fit(X_train, y_train) + .predict(X_val)
               scoring_fn(y_val, y_pred) → fold score
           → CVResult (mean, std, fold scores, errors)
Step 3: Rank all CVResults by mean_score descending
Step 4: Surface warnings (failed candidates, time budget)
Step 5: Assemble OptimizationReport → .to_json()
```

---

## 7. Usage Examples

### Minimal EDA

```python
import numpy as np
from eda.inspector import Inspector

data = np.array([[25, 50000], [30, 60000], [35, np.nan]], dtype=object)
report = Inspector().run(data, headers=["age", "salary"])
print(report.to_json())
```

### Minimal Preprocessing

```python
import numpy as np
from preprocessing.cleaner import Cleaner, CleanerConfig

data = np.array([
    [25, "Engineering", "yes", 8.5],
    [30, "Sales",       None,  6.2],
    [35, "HR",          "yes", np.nan],
], dtype=object)
headers = ["age", "dept", "manager", "rating"]

result = Cleaner().run(data, headers)
print(result.headers)   # ['age', 'rating', 'dept_Engineering', 'dept_HR', 'dept_Sales', ...]
```

### EDA → Preprocessing (reuse type_map)

```python
from eda.inspector import Inspector
from preprocessing.cleaner import Cleaner, CleanerConfig

inspector = Inspector()
eda_report = inspector.run(data, headers)
type_map = {ct["name"]: ct["inferred_type"] for ct in eda_report.to_dict()["column_types"]}

config = CleanerConfig(
    imputer_strategy="mean",
    scaler_type="standard",
    encoder_type="onehot",
)
result = Cleaner(config).run(data, headers, type_map=type_map)
print(result.to_json())
```

### MinMax + Label Encoding

```python
config = CleanerConfig(
    imputer_strategy="median",
    scaler_type="minmax",
    minmax_range=(0.0, 1.0),
    encoder_type="label",
)
result = Cleaner(config).run(data, headers)

for enc in result.report.encoding:
    print(f"{enc['column']}: {enc['mapping']}")
```

### Individual Module Usage

```python
from preprocessing.imputer import SimpleImputer
from preprocessing.scalers import MinMaxScaler, StandardScaler
from preprocessing.encoders import OneHotEncoder, LabelEncoder

imputer = SimpleImputer(strategy="median")
data_imp, imp_summaries = imputer.fit_transform(data, headers)

scaler = StandardScaler()
data_scaled, scale_summaries = scaler.fit_transform(data_imp, headers)

encoder = OneHotEncoder(drop_first=True)
data_enc, new_headers, enc_summaries = encoder.fit_transform(data_scaled, headers)
```

### Grid Search (Classification)

```python
import numpy as np
from glassbox.optimization.grid_search import GridSearch
from glassbox.optimization.scoring import accuracy_score

# MyClassifier must implement fit(X, y) and predict(X).
gs = GridSearch(scoring_fn=accuracy_score, cv=5, random_state=42)
result = gs.fit(
    X, y,
    estimator_class=MyClassifier,
    param_grid={
        "max_depth": [3, 5, 10],
        "min_samples": [2, 5],
    },
)
print(result.best_params)   # {"max_depth": 5, "min_samples": 2}
print(result.best_score)    # 0.9350
```

### Random Search (Regression, with time budget)

```python
from glassbox.optimization.random_search import RandomSearch
from glassbox.optimization.scoring import neg_mean_squared_error

rs = RandomSearch(
    scoring_fn=neg_mean_squared_error,
    cv=5,
    n_iter=50,
    time_budget=30.0,    # stop after 30 s
    random_state=0,
)
result = rs.fit(
    X, y,
    estimator_class=MyRegressor,
    param_distributions={
        "max_depth":    [3, 5, 10, 20],           # discrete list
        "learning_rate": (0.001, 0.5, "log"),     # log-uniform
        "n_estimators":  (50, 500),               # continuous uniform
    },
)
print(result.best_params)
print(f"Best MSE ≈ {-result.best_score:.4f}")
```

### Full Pipeline (EDA → Preprocessing → Optimization)

```python
import numpy as np
from glassbox.eda.inspector import Inspector
from glassbox.preprocessing.cleaner import Cleaner, CleanerConfig
from glassbox.optimization.orchestrator import Orchestrator, OrchestratorConfig
from glassbox.optimization.scoring import accuracy_score

# 1. EDA
inspector = Inspector()
eda_report = inspector.run(data, headers)
type_map = {ct["name"]: ct["inferred_type"] for ct in eda_report.to_dict()["column_types"]}

# 2. Preprocessing → clean float matrix
result = Cleaner(CleanerConfig(scaler_type="standard", encoder_type="label")).run(
    data, headers, type_map=type_map
)
X = result.data[:, :-1].astype(float)   # features
y = result.data[:, -1].astype(int)      # target

# 3. Hyperparameter search
config = OrchestratorConfig(search_type="grid", cv=5, random_state=42)
orch = Orchestrator(config)
report = orch.run(X, y, MyClassifier, {"max_depth": [3, 5, 10]}, accuracy_score)
print(report.to_json())
```

---

## 8. JSON Report Schemas

### EDA Report (`EDAReport.to_json()`)

```json
{
  "metadata": { "n_rows": 200, "n_cols": 6, "elapsed_seconds": 0.034 },
  "type_summary": { "numerical": 3, "categorical": 1, "boolean": 2 },
  "column_types": [
    { "name": "age", "inferred_type": "numerical", "n_unique": 52, "n_missing": 10 }
  ],
  "statistics": [
    { "name": "age", "mean": 45.26, "std": 16.23, "skewness": -0.01, "kurtosis": -1.18 }
  ],
  "correlation": {
    "column_names": ["age", "salary"],
    "matrix": [[1.0, 0.12], [0.12, 1.0]],
    "high_correlation_pairs": []
  },
  "outliers": [
    { "name": "salary", "n_outliers_low": 1, "n_outliers_high": 1, "outlier_pct": 1.0 }
  ],
  "warnings": ["Column 'age' has 10 missing values (5.0%)."]
}
```

### Preprocessing Report (`PreprocessingReport.to_json()`)

```json
{
  "metadata": {
    "n_rows": 10, "n_cols_in": 5, "n_cols_out": 9, "elapsed_seconds": 0.002
  },
  "headers_in":  ["age", "salary", "dept", "manager", "rating"],
  "headers_out": ["age", "salary", "rating", "dept_Eng", "dept_HR", "dept_Marketing",
                  "dept_Sales", "manager_no", "manager_yes"],
  "type_summary": { "numerical": 3, "categorical": 1, "boolean": 1 },
  "steps_applied": [
    "auto_typing",
    "imputation(strategy='mean')",
    "scaling(type='standard')",
    "encoding(type='onehot')"
  ],
  "imputation": [
    { "column": "age", "strategy": "mean", "fill_value": 38.5, "n_filled": 1 }
  ],
  "scaling": [
    { "column": "age", "scaler_type": "standard", "mean": 38.5, "std": 12.34 }
  ],
  "encoding": [
    {
      "column": "dept", "encoder_type": "onehot",
      "categories": ["Eng", "HR", "Marketing", "Sales"],
      "output_columns": ["dept_Eng", "dept_HR", "dept_Marketing", "dept_Sales"]
    }
  ],
  "warnings": ["Column 'age' has 1 missing values (10.0%) — will be imputed with strategy 'mean'."]
}
```

### Optimization Report (`OptimizationReport.to_json()`)

```json
{
  "metadata": {
    "search_type": "grid",
    "n_candidates_evaluated": 6,
    "n_splits": 5,
    "elapsed_seconds": 0.412,
    "scoring_fn": "accuracy_score",
    "time_budget_hit": false
  },
  "best_params": { "max_depth": 5, "min_samples": 2 },
  "best_score": 0.935,
  "all_results": [
    {
      "params": { "max_depth": 5, "min_samples": 2 },
      "fold_scores": [0.925, 0.95, 0.925, 0.95, 0.925],
      "mean_score": 0.935,
      "std_score": 0.012,
      "elapsed_seconds": 0.068,
      "error": null
    },
    {
      "params": { "max_depth": 3, "min_samples": 2 },
      "fold_scores": [0.9, 0.925, 0.9, 0.925, 0.9],
      "mean_score": 0.91,
      "std_score": 0.012,
      "elapsed_seconds": 0.061,
      "error": null
    }
  ],
  "warnings": []
}
```

---

## 9. Testing

```bash
pytest                                      # all tests
pytest -v --cov=. --cov-report=term-missing # with coverage
pytest tests/test_inspector.py -v           # single file
```

| Test File | Module | Focus |
| --------- | ------ | ----- |
| `test_math_utils.py` | `math_utils` | All 8 functions, NaN, NumPy comparison |
| `test_auto_typer.py` | `auto_typer` | Numeric/categorical/boolean detection |
| `test_stats.py` | `stats` | Profiling, NaN handling |
| `test_correlation.py` | `correlation` | Perfect/zero correlation, flagging |
| `test_outliers.py` | `outliers` | Detection, capping, k values |
| `test_inspector.py` | `inspector` | Full EDA pipeline, JSON output |
| `test_kfold.py` | `kfold` | Fold sizes, no data leakage, shuffle reproducibility |
| `test_scoring.py` | `scoring` | All metrics vs known values, edge cases |
| `test_grid_search.py` | `grid_search` | Best-param selection, error handling, JSON output |
| `test_random_search.py` | `random_search` | Discrete/continuous/log-uniform sampling, time budget |
| `test_orchestrator.py` | `orchestrator` | Full pipeline, both search types, report schema |

---

## 10. Design Decisions

**Why NumPy only?** Pandas and Scikit-Learn are not WASM-compatible in all environments. NumPy compiles cleanly via Pyodide, keeps memory footprint minimal, and avoids hidden type-coercion surprises.

**Why manual statistics?** The white-box requirement means every formula must be auditable. An IronClaw agent can explain, audit, and cite each calculation step.

**Why pairwise NaN handling in correlation?** Listwise deletion can drastically shrink sample size. Pairwise complete observations preserve maximum data per pair.

**Transformer pattern** (`fit / transform / fit_transform`): all preprocessing modules follow this pattern for pipeline composability. The EDA module is read-only so it uses `detect / profile / analyze / run` instead, but the dataclass serialisation pattern (`to_dict / to_json`) is consistent across both modules.

**OHE column expansion:** One-Hot Encoding increases the column count. `CleanerResult.headers` always reflects the actual output columns, and `PreprocessingReport` records both `headers_in` and `headers_out` for full traceability.

**Maximisation convention in search:** Both GridSearch and RandomSearch always *maximise* the scoring function. Error metrics (MSE, MAE) are wrapped with `neg_*` variants so the convention is consistent regardless of task type. This eliminates a common source of bugs in AutoML pipelines.

**Random Search log-uniform sampling:** Hyperparameters like learning rate span several orders of magnitude. Sampling linearly would concentrate draws near the upper bound. Log-uniform sampling (`(low, high, "log")`) ensures equal probability of sampling from each order of magnitude: $\text{value} = e^{\mathcal{U}[\ln(\text{low}),\; \ln(\text{high})]}$.

**Time budget for agent efficiency:** The project spec requires end-to-end search to complete in < 120 seconds. `RandomSearch.time_budget` enforces this at the search level, stopping after the budget expires and returning the best result found so far.

**Shared `_cross_validate` primitive:** The inner K-fold loop is extracted into a module-level function shared by both GridSearch and RandomSearch. This avoids code duplication and ensures both strategies produce identically structured `CVResult` objects.

---

## 11. IronClaw Agent Integration

Both modules are designed to be called as **tools** by an IronClaw (NEAR AI) agent:

```
User: "Clean this CSV for me — impute missing values and encode categoricals."

Agent → Tool Call (EDA):
    report = Inspector().run(data, headers)
    type_map = {ct["name"]: ct["inferred_type"] for ct in report.to_dict()["column_types"]}

Agent → Tool Call (Preprocessing):
    result = Cleaner(CleanerConfig(...)).run(data, headers, type_map=type_map)

Agent → Response:
    "Done. I filled 3 missing values using the column mean, standardised 3
     numerical columns, and one-hot encoded 'dept' into 4 indicator columns.
     The output has 9 columns (up from 5). Full audit in the JSON report."
```

The JSON reports are designed to be:
1. **Machine-parseable** — structured keys, consistent types.
2. **Agent-explainable** — warnings are written in natural language.
3. **Deterministic** — same input always produces the same output.

---

*GlassBox-AutoML v0.3.0 — Module I: Automated EDA (The Inspector) · Module II: Automated Preprocessing (The Cleaner) · Module IV: Hyperparameter Optimization (The Orchestrator)*
