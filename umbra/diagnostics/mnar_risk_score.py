"""
MNAR Evidence Assessment & Risk Score Synthesizer.

Synthesizes multiple empirical diagnostic signals:
1. Little's MCAR test rejection (evaluates global departures from MCAR).
2. Covariate distribution shift magnitude across missingness patterns (evidence for MAR).
3. Residual tail dependency & self-censoring concentration after conditioning on observed covariates.
4. Candidate auxiliary variable availability (for selection model identification).
5. Domain prior context (substantive literature context, informational).

CRITICAL METHODOLOGICAL FOUNDATION (IDENTIFIABILITY LIMIT):
MNAR is generally not identifiable from observed data alone. Umbra therefore
does not claim to prove that data are MNAR; it combines diagnostics and
sensitivity analyses to quantify evidence and assess how conclusions change
under plausible departures from MAR.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

from umbra.diagnostics.mcar_test import LittleMCARResult, littles_mcar_test
from umbra.diagnostics.pattern_analysis import PatternAnalysisReport, analyze_missingness_patterns
from umbra.diagnostics.shadow_variable_finder import (
    AuxiliaryVariableReport,
    find_shadow_variables,
)

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
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "weight": float(self.weight),
            "score": float(self.score),
            "is_triggered": bool(self.is_triggered),
            "description": self.description,
            "details": self.details,
        }


@dataclass
class MNARRiskReport:
    """Structured evidence assessment for an incomplete variable.

    Attributes
    ----------
    target_column : str
        Name of the evaluated variable.
    risk_level : str
        'LOW', 'MEDIUM', or 'HIGH' evidence consistent with MNAR.
    composite_score : float
        Synthesized score in [0.0, 1.0] reflecting strength of evidence against MAR.
    missing_rate : float
        Proportion of missing values in the variable.
    signals : List[DiagnosticSignal]
        Individual evidence signals evaluated on the observed data.
    explanation : str
        Scientifically calibrated narrative explanation.
    recommended_strategy : str
        Actionable modeling recommendation based on evidence and assumptions.
    shadow_candidate : Optional[str]
        Best candidate auxiliary variable for selection models, if available.
    domain_heuristic_matched : bool
        Whether variable name matches a sensitive domain literature pattern.
    citation : Optional[str]
        Literature citation associated with domain pattern.
    """

    target_column: str
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    composite_score: float  # [0.0, 1.0]
    missing_rate: float
    signals: List[DiagnosticSignal] = field(default_factory=list)
    explanation: str = ""
    recommended_strategy: str = "mar_chained_equations"
    shadow_candidate: Optional[str] = None
    domain_heuristic_matched: bool = False
    citation: Optional[str] = None

    @property
    def covariate_shift_score(self) -> float:
        """Score contribution from covariate distribution shifts."""
        for sig in self.signals:
            if "Covariate" in sig.name or "Distribution Shift" in sig.name:
                return float(sig.score)
        return 0.0

    def summary(self) -> str:
        sig_str = "\n".join(
            [
                f"    * [{s.score:.2f}] {s.name}: {s.description}"
                for s in self.signals
                if s.is_triggered
            ]
        )
        if not sig_str:
            sig_str = "    * None (Observed data patterns are compatible with MCAR/MAR)"
        return (
            f"============================================================\n"
            f"MNAR Risk Assessment for '{self.target_column}'\n"
            f"  - Evidence Level      : {self.risk_level} (Score: {self.composite_score:.2f})\n"
            f"  - Missing Rate        : {self.missing_rate:.1%}\n"
            f"  - Recommended Action  : {self.recommended_strategy}\n"
            f"  - Candidate Auxiliary : {self.shadow_candidate or 'None'}\n"
            f"  - Triggered Signals   :\n{sig_str}\n"
            f"  - Methodological Assessment:\n    {self.explanation}\n"
            f"============================================================"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_column": self.target_column,
            "risk_level": self.risk_level,
            "composite_score": float(self.composite_score),
            "missing_rate": float(self.missing_rate),
            "recommended_strategy": self.recommended_strategy,
            "shadow_candidate": self.shadow_candidate,
            "domain_heuristic_matched": self.domain_heuristic_matched,
            "citation": self.citation,
            "explanation": self.explanation,
            "signals": [s.to_dict() for s in self.signals],
        }


def _check_domain_heuristics(col_name: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Check if variable name matches known MNAR-vulnerable domains (informational only)."""
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
    """Test whether missingness is concentrated at extreme predicted quantiles (self-censoring).
    Fits MAR regression on observed cases and evaluates prediction distribution.
    """
    is_missing = data[target_col].isna()
    obs_mask = ~is_missing

    if not pd.api.types.is_numeric_dtype(data[target_col]):
        return 0.0, 0.0, {"status": "non_numeric_target"}

    num_covars = [c for c in covariate_cols if pd.api.types.is_numeric_dtype(data[c])]
    if obs_mask.sum() < 20 or is_missing.sum() < 10 or not num_covars:
        return 0.0, 0.0, {"status": "insufficient_samples"}

    X_obs = data.loc[obs_mask, num_covars].fillna(data[num_covars].median()).to_numpy(dtype=float)
    y_obs = data.loc[obs_mask, target_col].to_numpy(dtype=float)

    # Check for constant variance in predictors
    stds = np.std(X_obs, axis=0)
    valid_pred_indices = np.where(stds > 1e-8)[0]
    if len(valid_pred_indices) == 0:
        return 0.0, 0.0, {"status": "constant_covariates"}

    X_obs = X_obs[:, valid_pred_indices]
    sub_covars = [covariate_cols[i] for i in valid_pred_indices]

    # Fit regularized Ridge regression on observed data
    try:
        reg = Ridge(alpha=1.0)
        reg.fit(X_obs, y_obs)

        # Predict target across all observations
        X_all = data[sub_covars].fillna(data[sub_covars].median()).to_numpy(dtype=float)
        y_pred_all = reg.predict(X_all)

        # Discretize predicted values into quantiles
        n_quantiles = 5 if len(y_pred_all) >= 50 else 3
        quantiles = pd.qcut(y_pred_all, q=n_quantiles, labels=False, duplicates="drop")
        missing_rate_by_quantile = is_missing.groupby(quantiles).mean()

        max_rate = float(missing_rate_by_quantile.max())
        min_rate = float(missing_rate_by_quantile.min())
        overall_rate = float(is_missing.mean())

        # Concentration ratio: max quantile missing rate vs overall rate
        tail_ratio = (max_rate / (overall_rate + 1e-6)) if overall_rate > 0 else 1.0

        # Correlation between predicted target and missingness indicator
        r_pred_miss, p_pred_miss = stats.pearsonr(y_pred_all, is_missing.astype(float))

        # Tail score is high when missingness concentrates heavily in tails
        tail_score = float(np.clip((tail_ratio - 1.25) / 1.5, 0.0, 1.0))

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
    shadow_report: Optional[AuxiliaryVariableReport] = None,
    alpha: float = 0.05,
) -> MNARRiskReport:
    """Synthesize empirical data diagnostics into an honest MNAR risk assessment.

    Decision Hierarchy:
    -------------------
    1. If data is complete for target_col -> LOW risk.
    2. If Little's test fails to reject MCAR (p > alpha) AND covariate shifts are negligible
       -> Observed data are compatible with MCAR -> LOW risk (MICE defensible).
    3. If MCAR is rejected OR covariate shifts are significant, but residual tail dependency
       is low -> Covariates account for non-response -> Consistent with MAR -> LOW-to-MEDIUM risk.
    4. If significant residual tail concentration / self-censoring persists -> Evidence
       consistent with MNAR -> HIGH risk (Sensitivity analysis required).
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
        littles_result = (
            littles_mcar_test(numeric_df, alpha=alpha) if numeric_df.shape[1] > 1 else None
        )

    mcar_rejected = littles_result.is_rejected if littles_result else False
    mcar_p = littles_result.p_value if littles_result else 1.0

    # MCAR score: 0 if p >= alpha; scales up if p is very small
    if not mcar_rejected:
        mcar_score = 0.0
    else:
        # p < alpha: score between 0.3 and 0.8 depending on p-value
        mcar_score = float(np.clip(0.4 - 0.1 * np.log10(max(1e-10, mcar_p)), 0.3, 0.8))

    sig_mcar = DiagnosticSignal(
        name="Little's MCAR Test Rejection",
        weight=0.25,
        score=mcar_score,
        is_triggered=mcar_rejected,
        description=(
            f"Global hypothesis of MCAR rejected (p={mcar_p:.2e} < {alpha})."
            if mcar_rejected
            else f"Data are statistically compatible with MCAR (p={mcar_p:.3f} >= {alpha})."
        ),
        details={
            "p_value": mcar_p,
            "statistic": littles_result.statistic if littles_result else 0.0,
        },
    )

    # 2. Covariate Shift signal
    if pattern_report is None:
        pattern_report = analyze_missingness_patterns(data, alpha=alpha)

    var_pat = pattern_report.variable_reports.get(target_col)
    max_ks = var_pat.max_ks_statistic if var_pat else 0.0
    sig_shifts = var_pat.n_significant_shifts if var_pat else 0
    shift_score = float(np.clip(max_ks / 0.35, 0.0, 1.0))
    shift_triggered = bool(max_ks >= 0.15 or sig_shifts > 0)

    sig_shift = DiagnosticSignal(
        name="Covariate Distribution Shift",
        weight=0.35,
        score=shift_score,
        is_triggered=shift_triggered,
        description=(
            f"Observed covariates show significant distribution shifts across missingness "
            f"(Max KS={max_ks:.2f}, {sig_shifts} significant features)."
            if shift_triggered
            else f"No significant covariate distribution shifts detected (Max KS={max_ks:.2f})."
        ),
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
    tail_triggered = bool(tail_score >= 0.40 or r_pred_miss >= 0.30)

    sig_tail = DiagnosticSignal(
        name="Residual Tail / Self-Censoring Concentration",
        weight=0.40,
        score=tail_score,
        is_triggered=tail_triggered,
        description=(
            f"Missingness concentrates at extreme predicted quantiles (tail ratio={tail_details.get('tail_ratio', 1.0):.2f})."
            if tail_triggered
            else "Missingness shows no extreme tail concentration after MAR conditioning."
        ),
        details=tail_details,
    )

    # 4. Domain Heuristic Prior signal (informational annotation)
    domain_match, matched_pattern, citation = _check_domain_heuristics(target_col)

    # 5. Candidate Auxiliary Variable signal
    if shadow_report is None:
        shadow_report = find_shadow_variables(data, target_col)
    best_shadow = shadow_report.best_candidate
    has_shadow = best_shadow is not None and best_shadow.is_statistically_viable_candidate
    shadow_name = best_shadow.variable_name if has_shadow and best_shadow is not None else None

    # Compute empirical composite risk score (data-driven only!)
    empirical_signals = [sig_mcar, sig_shift, sig_tail]
    total_weight = sum(s.weight for s in empirical_signals)
    composite = sum(s.score * s.weight for s in empirical_signals) / total_weight

    # SCIENTIFIC ROUTING LOGIC:
    # Under MCAR (no shifts, MCAR not rejected), risk MUST be LOW regardless of column name!
    if not mcar_rejected and not shift_triggered and not tail_triggered:
        risk_level = "LOW"
        recommended_strategy = "mar_chained_equations"
    elif tail_triggered and (mcar_rejected or shift_triggered):
        # High tail concentration plus departure from MCAR is evidence consistent with MNAR
        risk_level = "HIGH"
        recommended_strategy = "mnar_heckman" if has_shadow else "mnar_pattern_mixture_sensitivity"
    elif shift_triggered or mcar_rejected:
        # Covariates explain missingness; tail dependency is low -> MAR compatible
        if composite >= 0.50 and tail_score >= 0.25:
            risk_level = "HIGH"
            recommended_strategy = (
                "mnar_heckman" if has_shadow else "mnar_pattern_mixture_sensitivity"
            )
        elif composite >= 0.25:
            risk_level = "MEDIUM"
            recommended_strategy = "mnar_pattern_mixture_sensitivity"
        else:
            risk_level = "LOW"
            recommended_strategy = "mar_chained_equations"
    else:
        risk_level = "LOW"
        recommended_strategy = "mar_chained_equations"

    # Add domain context signal as informational annotation
    sig_domain = DiagnosticSignal(
        name="Domain Context (Literature Prior)",
        weight=0.0,  # Informational: weight 0 so it cannot game or corrupt data routing!
        score=0.85 if domain_match else 0.0,
        is_triggered=domain_match,
        description=(
            f"Variable matches sensitive reporting domain '{matched_pattern}'."
            if domain_match
            else "No sensitive domain match."
        ),
        details={"pattern": matched_pattern, "citation": citation},
    )
    all_signals = [sig_mcar, sig_shift, sig_tail, sig_domain]

    # Calibrated scientific explanation
    reasons = []
    if mcar_rejected:
        reasons.append("Little's test rejected MCAR globally")
    if shift_triggered:
        strongest = var_pat.strongest_predictor if var_pat else "covariates"
        reasons.append(f"covariates show distribution shifts (strongest: {strongest})")
    if sig_tail.is_triggered:
        reasons.append(
            "missingness is non-uniform across predicted target quantiles (self-censoring signal)"
        )
    if domain_match:
        reasons.append(
            f"domain context indicates potential sensitive reporting bias ({matched_pattern})"
        )

    if risk_level == "HIGH":
        explanation = (
            f"Variable '{target_col}' exhibits empirical evidence consistent with Not-Missing-At-Random (MNAR): "
            + "; ".join(reasons)
            + ". "
            "Because true MNAR is not identifiable from observed data alone, point estimates under MAR "
            "cannot be guaranteed unbiased. Sensitivity analysis or selection modeling is strongly recommended."
        )
    elif risk_level == "MEDIUM":
        explanation = (
            f"Variable '{target_col}' shows moderate departures from MCAR: "
            + "; ".join(reasons)
            + ". "
            "While observed covariates account for part of the missingness, an unobserved component cannot be ruled out. "
            "Reporting sensitivity intervals alongside MAR point estimates is recommended."
        )
    else:
        explanation = (
            f"Variable '{target_col}' shows evidence compatible with MCAR or well-conditioned MAR. "
            "No strong self-censoring tail patterns were detected. Standard chained equations (MICE) are defensible."
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


def diagnose_dataframe(data: pd.DataFrame, alpha: float = 0.05) -> Dict[str, MNARRiskReport]:
    """Run full diagnostic screening across all incomplete columns in data."""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    missing_cols = [c for c in data.columns if data[c].isna().any()]
    if not missing_cols:
        return {}

    # Run shared global tests once
    numeric_df = data.select_dtypes(include=[np.number])
    littles_res = littles_mcar_test(numeric_df, alpha=alpha) if numeric_df.shape[1] > 1 else None
    pattern_rep = analyze_missingness_patterns(data, alpha=alpha)

    reports: Dict[str, MNARRiskReport] = {}
    for col in missing_cols:
        shadow_rep = find_shadow_variables(data, col)
        rep = assess_mnar_risk(
            data,
            target_col=col,
            littles_result=littles_res,
            pattern_report=pattern_rep,
            shadow_report=shadow_rep,
            alpha=alpha,
        )
        reports[col] = rep

    return reports
