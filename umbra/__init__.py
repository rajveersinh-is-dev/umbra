"""
Umbra: Honest Missing Data Diagnostics, Sensitivity Analysis & MNAR Imputation.

Mathematical Principle:
True MNAR is generally unidentifiable from observed data alone.
Umbra combines empirical diagnostics and sensitivity analyses to quantify evidence
and assess how conclusions change under plausible departures from MAR.
"""

from umbra.api import UmbraImputer, diagnose, diagnose_report
from umbra.benchmark.amputation import AmputationResult, ampute_multivariate
from umbra.data.loaders import (
    load_california_housing,
    load_clinical_trial_attrition,
    load_cps_wage,
    load_nhanes_biomarkers,
)
from umbra.diagnostics.identifiability_audit import (
    IdentifiabilityCertificate,
    audit_identifiability,
)
from umbra.diagnostics.manski_bounds import (
    ManskiBoundsResult,
    compute_dataframe_manski_bounds,
    compute_manski_bounds,
)
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
from umbra.explain import diagnostics_to_markdown, explain_diagnostics, explain_sensitivity
from umbra.imputers.heckman_selection import (
    HeckmanCollinearityWarning,
    HeckmanConvergenceWarning,
    HeckmanSelectionImputer,
    HeckmanSEWarning,
    WeakInstrumentWarning,
)
from umbra.imputers.mar_chained_equations import MARChainedEquationsImputer
from umbra.imputers.pattern_mixture import PatternMixtureImputer
from umbra.imputers.rubin_pooler import RubinPooler, RubinsRulesResult, rubins_rules
from umbra.sensitivity.grid_analysis import (
    SensitivityReport,
    TippingPoint,
    run_sensitivity_grid,
)

__version__ = "0.2.0"

__all__ = [
    "UmbraImputer",
    "diagnose",
    "diagnose_report",
    "UmbraDiagnosticReport",
    "littles_mcar_test",
    "LittleMCARResult",
    "analyze_missingness_patterns",
    "PatternAnalysisReport",
    "VariablePatternReport",
    "CovariateShift",
    "compute_cohens_d",
    "compute_cliffs_delta",
    "find_shadow_variables",
    "AuxiliaryVariableCandidate",
    "AuxiliaryVariableReport",
    "ShadowVariableCandidate",
    "ShadowFinderReport",
    "assess_mnar_risk",
    "diagnose_dataframe",
    "MNARRiskReport",
    "DiagnosticSignal",
    "MARChainedEquationsImputer",
    "HeckmanSelectionImputer",
    "WeakInstrumentWarning",
    "HeckmanSEWarning",
    "HeckmanConvergenceWarning",
    "HeckmanCollinearityWarning",
    "PatternMixtureImputer",
    "rubins_rules",
    "RubinPooler",
    "RubinsRulesResult",
    "run_sensitivity_grid",
    "SensitivityReport",
    "TippingPoint",
    "explain_diagnostics",
    "diagnostics_to_markdown",
    "explain_sensitivity",
    "MNARRiskCalibrator",
    "RouterDecisionProfile",
    "ROUTER_PROFILES",
    "compute_bayes_optimal_threshold",
    "compute_expected_losses",
    "get_decision_profile",
    "ampute_multivariate",
    "AmputationResult",
    "load_cps_wage",
    "load_nhanes_biomarkers",
    "load_california_housing",
    "load_clinical_trial_attrition",
    "ManskiBoundsResult",
    "compute_manski_bounds",
    "compute_dataframe_manski_bounds",
    "IdentifiabilityCertificate",
    "audit_identifiability",
]
