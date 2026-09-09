"""
Umbra Diagnostics Module.

Provides empirical diagnostics for missing data mechanisms:
- littles_mcar_test: Global likelihood ratio test for Missing Completely At Random (Little 1988)
- analyze_missingness_patterns: Pattern frequency, correlation, and covariate shift analysis
- find_shadow_variables: Empirical screening for candidate auxiliary identification variables
- assess_mnar_risk: Composite risk scoring across observable diagnostic signals
- UmbraDiagnosticReport: Comprehensive structured diagnostic report
"""

from umbra.diagnostics.mcar_test import LittleMCARResult, littles_mcar_test
from umbra.diagnostics.mnar_risk_score import (
    DiagnosticSignal,
    MNARRiskReport,
    assess_mnar_risk,
    diagnose_dataframe,
)
from umbra.diagnostics.pattern_analysis import (
    CovariateShift,
    PatternAnalysisReport,
    VariablePatternReport,
    analyze_missingness_patterns,
    compute_cliffs_delta,
    compute_cohens_d,
)
from umbra.diagnostics.report import UmbraDiagnosticReport
from umbra.diagnostics.risk_calibrator import (
    ROUTER_PROFILES,
    MNARRiskCalibrator,
    RouterDecisionProfile,
    compute_bayes_optimal_threshold,
    compute_expected_losses,
    get_decision_profile,
)
from umbra.diagnostics.shadow_variable_finder import (
    AuxiliaryVariableCandidate,
    AuxiliaryVariableReport,
    ShadowFinderReport,
    ShadowVariableCandidate,
    find_shadow_variables,
)

__all__ = [
    "littles_mcar_test",
    "LittleMCARResult",
    "analyze_missingness_patterns",
    "PatternAnalysisReport",
    "VariablePatternReport",
    "CovariateShift",
    "compute_cohens_d",
    "compute_cliffs_delta",
    "find_shadow_variables",
    "ShadowFinderReport",
    "ShadowVariableCandidate",
    "AuxiliaryVariableReport",
    "AuxiliaryVariableCandidate",
    "assess_mnar_risk",
    "MNARRiskReport",
    "DiagnosticSignal",
    "diagnose_dataframe",
    "UmbraDiagnosticReport",
    "MNARRiskCalibrator",
    "RouterDecisionProfile",
    "ROUTER_PROFILES",
    "compute_bayes_optimal_threshold",
    "compute_expected_losses",
    "get_decision_profile",
]
