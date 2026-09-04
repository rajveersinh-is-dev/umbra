"""
Umbra: MNAR-Aware Missing Data Imputation Library.

Diagnose and honestly handle Not-Missing-At-Random (MNAR) data.
Refuses to pretend that a single confident imputed point estimate is safe when it isn't.
"""

from umbra.api import UmbraImputer
from umbra.diagnostics.mcar_test import LittleMCARResult, littles_mcar_test
from umbra.diagnostics.mnar_risk_score import MNARRiskReport, assess_mnar_risk, diagnose_dataframe
from umbra.diagnostics.pattern_analysis import PatternAnalysisReport, analyze_missingness_patterns
from umbra.diagnostics.shadow_variable_finder import ShadowFinderReport, find_shadow_variables
from umbra.explain import diagnostics_to_markdown, explain_diagnostics, explain_sensitivity
from umbra.imputers.heckman_selection import HeckmanSelectionImputer
from umbra.imputers.mar_chained_equations import MARChainedEquationsImputer
from umbra.imputers.pattern_mixture import PatternMixtureImputer
from umbra.sensitivity.grid_analysis import SensitivityReport, TippingPoint, run_sensitivity_grid

__version__ = "0.1.0"

__all__ = [
    "UmbraImputer",
    "littles_mcar_test",
    "LittleMCARResult",
    "analyze_missingness_patterns",
    "PatternAnalysisReport",
    "find_shadow_variables",
    "ShadowFinderReport",
    "assess_mnar_risk",
    "diagnose_dataframe",
    "MNARRiskReport",
    "MARChainedEquationsImputer",
    "HeckmanSelectionImputer",
    "PatternMixtureImputer",
    "run_sensitivity_grid",
    "SensitivityReport",
    "TippingPoint",
    "explain_diagnostics",
    "diagnostics_to_markdown",
    "explain_sensitivity",
]
