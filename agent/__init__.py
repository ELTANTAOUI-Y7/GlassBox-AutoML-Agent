"""
glassbox.agent
==============

Phase V — IronClaw Agent Integration.

Exposes the :class:`~glassbox.agent.autofit.AutoFit` tool, the single entry
point that an IronClaw agent calls to run the full EDA → Cleaning →
Optimization pipeline from raw CSV data.
"""

from glassbox.agent.autofit import AutoFit, AutoFitConfig, AutoFitReport

__all__ = ["AutoFit", "AutoFitConfig", "AutoFitReport"]
