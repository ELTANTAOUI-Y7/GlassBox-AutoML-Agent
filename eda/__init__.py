"""
GlassBox EDA (The Inspector) — Automated Exploratory Data Analysis.

Submodules:
    - auto_typer:   Automatic column type detection (Numerical, Categorical, Boolean).
    - stats:        Manual statistical profiling (Mean, Median, Mode, StdDev, Skewness, Kurtosis).
    - correlation:  Pearson Correlation Matrix computed from scratch.
    - outliers:     IQR-based outlier detection, flagging, and capping.
    - inspector:    Orchestrator that runs the full EDA pipeline and returns a JSON report.
"""

from glassbox.eda.inspector import Inspector

__all__ = ["Inspector"]
