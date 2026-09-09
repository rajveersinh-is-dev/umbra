"""
Identifiability & Assumption Audit Certificate Engine (Phase 6).

Implements machine-verifiable audit certificates documenting:
1. Testable Empirical Implications vs. Untestable Domain Assumptions.
2. Molenberghs et al. (2008) Non-Identifiability Theorem guarantees.
3. Manski (1989, 2003) Sharp Nonparametric Partial Identification Bounds.
4. Strategy-specific identification contracts for MICE, Heckman Selection, and Pattern Mixture.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from umbra.diagnostics.manski_bounds import ManskiBoundsResult, compute_manski_bounds
from umbra.diagnostics.mnar_risk_score import MNARRiskReport


@dataclass
class IdentifiabilityCertificate:
    """Machine-verifiable audit certificate of missing data identifiability."""

    feature: str
    missing_rate: float
    manski_bounds: ManskiBoundsResult
    testable_evidence: Dict[str, Any]
    strategy_assumptions: Dict[str, Dict[str, Any]]
    audit_verdict: str
    scientific_disclaimer: str
    citations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize certificate to dictionary for JSON reports and CI/CD pipelines."""
        return {
            "feature": self.feature,
            "missing_rate": round(self.missing_rate, 4),
            "audit_verdict": self.audit_verdict,
            "manski_bounds": self.manski_bounds.to_dict(),
            "testable_evidence": self.testable_evidence,
            "strategy_assumptions": self.strategy_assumptions,
            "scientific_disclaimer": self.scientific_disclaimer,
            "citations": self.citations,
        }

    def to_markdown(self) -> str:
        """Format certificate into clean GitHub-flavored markdown."""
        lines = [
            f"### Identifiability & Assumption Audit Certificate: `{self.feature}`",
            "",
            f"**Audit Verdict**: `{self.audit_verdict}`  ",
            f"**Observed Missing Rate**: `{self.missing_rate:.1%}`  ",
            "",
            "#### 1. Nonparametric Manski Partial Identification (Zero Untestable Assumptions)",
            f"- **Domain Support**: `[{self.manski_bounds.support_lower:.2f}, {self.manski_bounds.support_upper:.2f}]`",
            f"- **Sharp Mean Bounds**: `[{self.manski_bounds.mean_lower_bound:.4f}, {self.manski_bounds.mean_upper_bound:.4f}]` (Interval width: `{self.manski_bounds.mean_interval_width:.4f}`)",
            f"- **Sharp Median Bounds**: `[{self.manski_bounds.median_lower_bound:.4f}, {self.manski_bounds.median_upper_bound:.4f}]` (Interval width: `{self.manski_bounds.median_interval_width:.4f}`)",
            f"- **Observed Point Mean**: `{self.manski_bounds.observed_mean:.4f}`",
            "",
            "#### 2. Testable Empirical Evidence",
        ]

        for k, v in self.testable_evidence.items():
            lines.append(f"- **{k.replace('_', ' ').title()}**: `{v}`")

        lines.extend(
            [
                "",
                "#### 3. Candidate Strategy Identification Contracts",
                "",
                "| Strategy | Testable Implications | Untestable Domain Assumptions | Identification Status |",
                "|---|---|---|---|",
            ]
        )

        for strat, details in self.strategy_assumptions.items():
            lines.append(
                f"| **{strat}** | {details.get('testable', 'N/A')} | {details.get('untestable', 'N/A')} | `{details.get('status', 'N/A')}` |"
            )

        lines.extend(
            [
                "",
                "> [!IMPORTANT]",
                f"> **Molenberghs Non-Identifiability Theorem**: {self.scientific_disclaimer}",
                "",
                "**Key Citations**:",
            ]
        )

        for cit in self.citations:
            lines.append(f"- *{cit}*")

        return "\n".join(lines)


def audit_identifiability(
    feature: str,
    df: pd.DataFrame,
    risk_report: Optional[MNARRiskReport] = None,
    shadow_var: Optional[str] = None,
    support: Optional[Tuple[float, float]] = None,
) -> IdentifiabilityCertificate:
    """
    Perform formal identifiability audit and generate an IdentifiabilityCertificate.

    Parameters
    ----------
    feature : str
        Target column name being audited.
    df : pd.DataFrame
        Complete DataFrame containing feature and potential predictors/instruments.
    risk_report : Optional[MNARRiskReport]
        Precomputed risk report from diagnose_dataframe or assess_mnar_risk.
    shadow_var : Optional[str]
        Auxiliary instrumental candidate variable.
    support : Optional[Tuple[float, float]]
        Domain bounds [y_L, y_U] for Manski partial identification.

    Returns
    -------
    IdentifiabilityCertificate
        Structured audit report.
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame columns.")

    series = df[feature]
    missing_rate = float(series.isna().mean())

    # 1. Compute Manski Nonparametric Bounds
    manski = compute_manski_bounds(
        y=series,
        feature_name=feature,
        support=support,
    )

    # 2. Extract testable implications
    testable_ev: Dict[str, Any] = {
        "n_total": len(series),
        "n_observed": int(series.notna().sum()),
        "n_missing": int(series.isna().sum()),
        "missing_percentage": f"{missing_rate:.1%}",
    }

    f_stat = None
    shadow_candidate = shadow_var
    risk_level = "UNKNOWN"

    if risk_report is not None:
        risk_level = risk_report.risk_level
        testable_ev["risk_level"] = risk_report.risk_level
        testable_ev["composite_risk_score"] = round(risk_report.composite_score, 3)

        if risk_report.shadow_candidate:
            shadow_candidate = risk_report.shadow_candidate
            testable_ev["shadow_instrument_candidate"] = shadow_candidate

        # Extract signal details
        for sig in risk_report.signals:
            name_clean = sig.name.lower().replace(" ", "_").replace("'", "")
            testable_ev[name_clean] = {
                "triggered": sig.is_triggered,
                "score": round(sig.score, 3),
                "details": sig.details,
            }

    # Evaluate auxiliary instrument relevance if candidate exists
    effective_shadow = shadow_candidate or shadow_var
    if effective_shadow and effective_shadow in df.columns:
        from umbra.diagnostics.shadow_variable_finder import find_shadow_variables

        s_rep = find_shadow_variables(df, feature)
        matching = [c for c in s_rep.candidates if c.variable_name == effective_shadow]
        if matching:
            cand = matching[0]
            f_stat = cand.first_stage_f_stat
            testable_ev["instrument_f_stat"] = round(f_stat, 2)
            testable_ev["stock_yogo_pass"] = bool(f_stat > 10.0)

    # 3. Formulate strategy contracts
    strategy_contracts: Dict[str, Dict[str, Any]] = {
        "Manski Nonparametric Bounds": {
            "testable": "Observed support and missing proportion",
            "untestable": "NONE (Invariant under all missingness processes)",
            "status": "GUARANTEED_VALID",
        },
        "MICE / MAR Chained Equations": {
            "testable": "Covariate shifts and non-linear associations in observed X",
            "untestable": "Missingness conditional independence: Y_mis _|_ R | X (Rubin 1976)",
            "status": "COMPATIBLE" if risk_level in ("LOW", "MEDIUM") else "AT_RISK_MNAR",
        },
        "Heckman Selection Model": {
            "testable": f"First-stage instrument relevance (F = {f_stat:.1f} > 10)"
            if f_stat is not None
            else "Exclusion restriction presence",
            "untestable": "Joint bivariate normality (eps, u) and instrument exogeneity (Z _|_ eps)",
            "status": "IDENTIFIABLE_UNDER_EXCLUSION"
            if (f_stat is not None and f_stat > 10.0)
            else "FRAGILE_FUNCTIONAL_FORM_ONLY",
        },
        "Pattern Mixture Sensitivity": {
            "testable": "Observed complete case variance and distribution",
            "untestable": "Plausible bounds on mean shift parameter delta",
            "status": "PARTIALLY_IDENTIFIABLE_UNDER_DELTA_PRIOR",
        },
    }

    # 4. Derive Audit Verdict
    if missing_rate == 0.0:
        verdict = "FULLY_OBSERVED"
    elif f_stat is not None and f_stat > 10.0 and shadow_candidate is not None:
        verdict = "HECKMAN_IDENTIFIABLE_UNDER_EXCLUSION"
    elif risk_level == "LOW":
        verdict = "COMPATIBLE_WITH_MAR"
    else:
        verdict = "PARTIALLY_IDENTIFIED_ONLY"

    disclaimer = (
        "True MNAR is fundamentally unidentifiable from observed data alone (Molenberghs et al., 2008). "
        "Any point-imputation model (MAR MICE, Heckman selection, or Pattern Mixture) substitutes untestable "
        "structural assumptions to achieve point identification. The Manski partial identification bounds "
        "provide the only mathematically guaranteed parameter interval that holds under zero untestable assumptions."
    )

    citations = [
        "Manski, C. F. (1989). Anatomy of the Selection Problem. Journal of Human Resources, 24(3), 343-360.",
        "Manski, C. F. (2003). Partial Identification of Probability Distributions. Springer New York.",
        "Molenberghs, G., Beunckens, C., Sotto, C., & Kenward, M. G. (2008). Every missingness not at random model has a missingness at random counterpart with equal fit. JRSS-B, 70(2), 371-388.",
        "Heckman, J. J. (1979). Sample Selection Bias as a Specification Error. Econometrica, 47(1), 153-161.",
    ]

    return IdentifiabilityCertificate(
        feature=feature,
        missing_rate=missing_rate,
        manski_bounds=manski,
        testable_evidence=testable_ev,
        strategy_assumptions=strategy_contracts,
        audit_verdict=verdict,
        scientific_disclaimer=disclaimer,
        citations=citations,
    )
