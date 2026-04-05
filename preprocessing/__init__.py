"""
GlassBox Preprocessing (The Cleaner) — Automated Data Preprocessing Engine.

Submodules:
    - imputer:   SimpleImputer — fills missing values with mean/median/mode/constant.
    - scalers:   MinMaxScaler and StandardScaler — normalise numerical features.
    - encoders:  OneHotEncoder and LabelEncoder — encode categorical features.
    - cleaner:   Cleaner orchestrator — chains all preprocessing steps into one pipeline.
"""

from glassbox.preprocessing.imputer import SimpleImputer
from glassbox.preprocessing.scalers import MinMaxScaler, StandardScaler
from glassbox.preprocessing.encoders import OneHotEncoder, LabelEncoder
from glassbox.preprocessing.cleaner import Cleaner, CleanerConfig, PreprocessingReport

__all__ = [
    "SimpleImputer",
    "MinMaxScaler",
    "StandardScaler",
    "OneHotEncoder",
    "LabelEncoder",
    "Cleaner",
    "CleanerConfig",
    "PreprocessingReport",
]
