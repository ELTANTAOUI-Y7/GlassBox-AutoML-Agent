"""
GlassBox-AutoML: A Secure, Scratch-Built Agentic Framework for Automated Machine Learning.

This package provides a transparent, auditable, and lightweight AutoML engine
built entirely from scratch using only NumPy for mathematical operations.

Modules
-------
eda
    Module I — The Inspector: Automated Exploratory Data Analysis.
preprocessing
    Module II — The Cleaner: Automated Data Preprocessing.
"""

__version__ = "0.3.0"
__author__ = "GlassBox-AutoML Team"

from glassbox.eda.inspector import Inspector
from glassbox.preprocessing.cleaner import Cleaner, CleanerConfig, PreprocessingReport
from glassbox.optimization.orchestrator import Orchestrator, OrchestratorConfig, OptimizationReport
from glassbox.agent.autofit import AutoFit, AutoFitConfig, AutoFitReport
