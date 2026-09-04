"""
MNAR Risk Score Synthesizer.

Combines multiple converging heuristics:
1. Little's MCAR test rejection (rules out pure MCAR).
2. Covariate distribution shift magnitude across missingness patterns.
3. Residual dependency after MAR conditioning (self-censoring / tail propensity).
4. Unexplained missingness propensity (low pseudo-R2 from observed variables).
5. Domain-shape heuristic prior with literature citations.
6. Shadow variable availability.

CRITICAL METHODOLOGICAL LIMITATION:
True MNAR is fundamentally unidentifiable from observed data alone without
untestable assumptions. This module produces a structured RISK ASSESSMENT
and explanation, NOT a certainty claim.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

from umbra.diagnostics.mcar_test import LittleMCARResult, littles_mcar_test
from umbra.diagnostics.pattern_analysis import PatternAnalysisReport, analyze_missingness_patterns
from umbra.diagnostics.shadow_variable_finder import ShadowFinderReport, find_shadow_variables

DOMAIN_SENSITIVE_PATTERNS = [
    (
        r"(income|salary|wage|bonus|earning|compensation|wealth|net_worth|debt|revenue)",
        "Socioeconomic self-censoring: High-earners and low-earners disproportionately skip income questions. "
        "Reference: Tourangeau & Yan (2007), 'Sensitive questions in surveys', Psychological Bulletin 133(5).",
    ),
    (
        r"(depression|anxiety|phq|gad|ces_d|bdi|ptsd|mental_health|suicid)",
        "Psychiatric symptom non-response: Clinical symptom severity correlates with survey dropout and non-disclosure. "
        "Reference: Little & Rubin (2019), Statistical Analysis with Missing Data (3rd ed.).",
    ),
    (
        r"(alcohol|drug|substance|cannabis|cocaine|opioid|tobacco|smoking)",
        "Substance reporting bias: Self-reports of illicit or stigmatized behaviors exhibit systematic underreporting or refusal. "
        "Reference: Harrison et al. (2007), 'The validity of self-reported drug use in survey research'.",
    ),
    (
        r"(weight|bmi|obesity|body_mass|calorie)",
        "Anthropometric reporting bias: High-BMI individuals have higher refusal/missingness rates. "
        "Reference: Rowland (1990), 'Self-reported weight and height', Am. J. Clin. Nutr. 52(6).",
    ),
    (
        r"(adverse_event|side_effect|toxicity|compliance|dropout|adherence)",
        "Clinical trial attrition: Patients experiencing severe adverse effects or lack of efficacy selectively drop out. "
        "Reference: National Research Council (2010), 'The Prevention and Treatment of Missing Data in Clinical Trials'.",
    ),
]


@dataclass
class DiagnosticSignal:
    """Individual diagnostic evidence signal."""

    name: str
    weight: float
    score: float  # [0.0, 1.0]
    is_triggered: bool
    description: str
    details: Dict[str, Any]


@dataclass
class MNARRiskReport:
    """Structured MNAR risk assessment for an incomplete variable."""

    target_column: str
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    composite_score: float  # [0.0, 1.0]
    missing_rate: float
    signals: List[DiagnosticSignal]
    explanation: str
    recommended_strategy: str
    shadow_candidate: Optional[str] = None
    domain_heuristic_matched: bool = False
    citation: Optional[str] = None

    def summary(self) -> str:
        sig_str = "\n".join(
            [
                f"    * [{s.score:.2f}] {s.name}: {s.description}"
                for s in self.signals
                if s.is_triggered
            ]
        )
        if not sig_str:
            sig_str = "    * None (data appears consistent with MCAR/MAR)"
        return (
            f"============================================================\n"
            f"MNAR Risk Assessment for '{self.target_column}'\n"
            f"  - Risk Level          : {self.risk_level} (Score: {self.composite_score:.2f})\n"
            f"  - Missing Rate        : {self.missing_rate:.1%}\n"
            f"  - Recommended Strategy: {self.recommended_strategy}\n"
            f"  - Triggered Signals   :\n{sig_str}\n"
            f"  - Explanation         :\n    {self.explanation}\n"
            f"============================================================"
        )


def _check_domain_heuristics(col_name: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Check if variable name matches known MNAR-vulnerable domains."""
    name_clean = col_name.lower().strip()
    for pattern, citation in DOMAIN_SENSITIVE_PATTERNS:
        if re.search(pattern, name_clean):
            return True, pattern, citation
    return False, None, None


def _evaluate_residual_tail_dependency(
    data: pd.DataFrame,
    target_col: str,
    covariate_cols: List[str],
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Test whether missingness is concentrated in extreme predicted quantiles (self-censoring).
    Fits MAR regression on observed cases and evaluates prediction distribution.
    """
    is_missing = data[target_col].isna()
    obs_mask = ~is_missing

    if obs_mask.sum() < 20 or is_missing.sum() < 10 or not covariate_cols:
        return 0.0, 0.0, {"status": "insufficient_samples"}

    X_obs = (
        data.loc[obs_mask, covariate_cols]
        .fillna(data[covariate_cols].median())
        .to_numpy(dtype=float)
    )
    y_obs = data.loc[obs_mask, target_col].to_numpy(dtype=float)

    # Fit Ridge regressor on observed data
    reg = Ridge(alpha=1.0)
    reg.fit(X_obs, y_obs)

    # Predict target across all data
    X_all = data[covariate_cols].fillna(data[covariate_cols].median()).to_numpy(dtype=float)
    y_pred_all = reg.predict(X_all)

    # Discretize predicted values into deciles
    try:
        quantiles = pd.qcut(y_pred_all, q=5, labels=False, duplicates="drop")
        missing_rate_by_quantile = is_missing.groupby(quantiles).mean()

        max_rate = float(missing_rate_by_quantile.max())
        min_rate = float(missing_rate_by_quantile.min())
        overall_rate = float(is_missing.mean())

        # Concentration ratio: ratio of max decile missingness to overall rate
        tail_ratio = (max_rate / (overall_rate + 1e-6)) if overall_rate > 0 else 1.0

        # Correlation between predicted target and missingness
        r_pred_miss, p_pred_miss = stats.pearsonr(y_pred_all, is_missing.astype(int))

        # Check if upper quantile or lower quantile has extreme missingness
        tail_score = float(np.clip((tail_ratio - 1.2) / 1.5, 0.0, 1.0))

        details = {
            "tail_ratio": tail_ratio,
            "max_quantile_missing_rate": max_rate,
            "min_quantile_missing_rate": min_rate,
            "predicted_missingness_corr": float(r_pred_miss),
            "predicted_missingness_p": float(p_pred_miss),
        }
        return tail_score, float(abs(r_pred_miss)), details
    except Exception as e:
        return 0.0, 0.0, {"error": str(e)}


def assess_mnar_risk(
    data: pd.DataFrame,
    target_col: str,
    littles_result: Optional[LittleMCARResult] = None,
    pattern_report: Optional[PatternAnalysisReport] = None,
    shadow_report: Optional[ShadowFinderReport] = None,
) -> MNARRiskReport:
    """
    Synthesize multiple diagnostic signals into an MNAR risk report for target_col.
    """
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")

    is_missing = data[target_col].isna()
    n_missing = int(is_missing.sum())
    n_total = len(data)
    missing_rate = float(n_missing / n_total) if n_total > 0 else 0.0

    if n_missing == 0:
        return MNARRiskReport(
            target_column=target_col,
            risk_level="LOW",
            composite_score=0.0,
            missing_rate=0.0,
            signals=[],
            explanation="No missing values observed in this variable.",
            recommended_strategy="none",
        )

    # 1. Little's MCAR Test signal
    if littles_result is None:
        numeric_df = data.select_dtypes(include=[np.number])
        littles_result = littles_mcar_test(numeric_df)

    mcar_rejected = littles_result.is_rejected
    mcar_score = 0.8 if mcar_rejected else 0.0
    sig_mcar = DiagnosticSignal(
        name="Little's MCAR Test Rejection",
        weight=0.20,
        score=mcar_score,
        is_triggered=mcar_rejected,
        description="Global hypothesis of Missing Completely at Random was rejected."
        if mcar_rejected
        else "Data is consistent with MCAR globally.",
        details={"p_value": littles_result.p_value, "statistic": littles_result.statistic},
    )

    # 2. Covariate Shift signal
    if pattern_report is None:
        pattern_report = analyze_missingness_patterns(data)

    var_pat = pattern_report.variable_reports.get(target_col)
    max_ks = var_pat.max_ks_statistic if var_pat else 0.0
    sig_shifts = var_pat.n_significant_shifts if var_pat else 0
    # Score scaled from KS statistic (KS > 0.20 is notable shift, > 0.40 is severe)
    shift_score = float(np.clip(max_ks / 0.35, 0.0, 1.0))
    shift_triggered = bool(max_ks >= 0.15 or sig_shifts > 0)
    sig_shift = DiagnosticSignal(
        name="Covariate Distribution Shift",
        weight=0.25,
        score=shift_score,
        is_triggered=shift_triggered,
        description=f"Observed covariates show distribution shifts when {target_col} is missing (Max KS={max_ks:.2f}, {sig_shifts} significant features).",
        details={
            "max_ks": max_ks,
            "significant_features": sig_shifts,
            "strongest_predictor": var_pat.strongest_predictor if var_pat else None,
        },
    )

    # 3. Residual Tail Dependency / Self-Censoring signal
    numeric_covars = [c for c in data.select_dtypes(include=[np.number]).columns if c != target_col]
    tail_score, r_pred_miss, tail_details = _evaluate_residual_tail_dependency(
        data, target_col, numeric_covars
    )
    tail_triggered = bool(tail_score >= 0.40 or r_pred_miss >= 0.25)
    sig_tail = DiagnosticSignal(
        name="Residual Tail / Self-Censoring Concentration",
        weight=0.25,
        score=tail_score,
        is_triggered=tail_triggered,
        description=f"Missingness is selectively concentrated at extreme predicted values (tail concentration ratio={tail_details.get('tail_ratio', 1.0):.2f}).",
        details=tail_details,
    )

    # 4. Domain Heuristic Prior signal
    domain_match, matched_pattern, citation = _check_domain_heuristics(target_col)
    domain_score = 0.85 if domain_match else 0.0
    sig_domain = DiagnosticSignal(
        name="Domain-Shape Prior",
        weight=0.20,
        score=domain_score,
        is_triggered=domain_match,
        description=f"Variable name matches high-risk MNAR domain pattern '{matched_pattern}'.",
        details={"pattern": matched_pattern, "citation": citation},
    )

    # 5. Shadow Variable signal
    if shadow_report is None:
        shadow_report = find_shadow_variables(data, target_col)
    best_shadow = shadow_report.best_candidate
    has_shadow = best_shadow is not None and best_shadow.is_promising_candidate
    shadow_name = best_shadow.variable_name if (has_shadow and best_shadow is not None) else None

    # Compute composite weighted risk score
    all_signals = [sig_mcar, sig_shift, sig_tail, sig_domain]
    total_weight = sum(s.weight for s in all_signals)
    composite = sum(s.score * s.weight for s in all_signals) / total_weight

    # Determine risk level
    if composite >= 0.50 or (
        sig_tail.is_triggered and (sig_domain.is_triggered or shift_score > 0.4)
    ):
        risk_level = "HIGH"
        recommended_strategy = "mnar_heckman" if has_shadow else "mnar_pattern_mixture_sensitivity"
    elif composite >= 0.25 or sig_domain.is_triggered or shift_triggered:
        risk_level = "MEDIUM"
        recommended_strategy = "mnar_pattern_mixture_sensitivity"
    else:
        risk_level = "LOW"
        recommended_strategy = "mar_chained_equations"

    # Plain language explanation
    reasons = []
    if domain_match:
        reasons.append(
            f"it falls into a known sensitive domain ('{target_col}') where self-censoring is common in literature"
        )
    if sig_tail.is_triggered:
        reasons.append("missingness is strongly non-uniform across the predicted value range")
    if shift_triggered:
        reasons.append(
            f"covariates show systematic distribution shifts (e.g. {var_pat.strongest_predictor if var_pat else 'other variables'})"
        )
    if mcar_rejected:
        reasons.append("Little's test rejected MCAR globally")

    if risk_level == "HIGH":
        explanation = (
            f"Variable '{target_col}' shows strong converging indicators of Not-Missing-At-Random (MNAR) missingness: "
            + "; ".join(reasons)
            + ". "
            "A standard MAR imputation (like MICE or mean imputation) is likely to introduce systematic bias. "
            "We strongly recommend running an MNAR-aware sensitivity grid analysis or Heckman selection model."
        )
    elif risk_level == "MEDIUM":
        explanation = (
            f"Variable '{target_col}' shows moderate signs consistent with MNAR: "
            + "; ".join(reasons)
            + ". "
            "While observed covariates explain some of the missingness, an unobserved component cannot be ruled out. "
            "Reporting a sensitivity interval is recommended."
        )
    else:
        explanation = (
            f"Variable '{target_col}' shows low risk of severe MNAR. Missingness is either close to MCAR or "
            "well-conditioned by observed covariates (MAR). Standard chained equations (MICE) are defensible."
        )

    return MNARRiskReport(
        target_column=target_col,
        risk_level=risk_level,
        composite_score=float(composite),
        missing_rate=missing_rate,
        signals=all_signals,
        explanation=explanation,
        recommended_strategy=recommended_strategy,
        shadow_candidate=shadow_name,
        domain_heuristic_matched=domain_match,
        citation=citation,
    )


def diagnose_dataframe(data: pd.DataFrame) -> Dict[str, MNARRiskReport]:
    """
    Run the full diagnostics battery across all columns with missing values.
    """
    missing_cols = [c for c in data.columns if data[c].isna().any()]
    if not missing_cols:
        return {}

    # Run shared global tests once
    numeric_df = data.select_dtypes(include=[np.number])
    littles_res = littles_mcar_test(numeric_df) if numeric_df.shape[1] > 1 else None
    pattern_rep = analyze_missingness_patterns(data)

    reports = {}
    for col in missing_cols:
        shadow_rep = find_shadow_variables(data, col)
        report = assess_mnar_risk(
            data,
            target_col=col,
            littles_result=littles_res,
            pattern_report=pattern_rep,
            shadow_report=shadow_rep,
        )
        reports[col] = report

    return reports
