"""
Structured Diagnostic Report Object for Umbra.

Provides the unified entry point `umbra.diagnose(X)` and the structured
`UmbraDiagnosticReport` container.

Strictly separates:
1. OBSERVED-DATA EVIDENCE
2. MODEL-BASED INFERENCE
3. UNTESTABLE ASSUMPTIONS
4. SENSITIVITY RESULTS
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from umbra.diagnostics.mcar_test import LittleMCARResult, littles_mcar_test
from umbra.diagnostics.mnar_risk_score import MNARRiskReport, assess_mnar_risk
from umbra.diagnostics.pattern_analysis import PatternAnalysisReport, analyze_missingness_patterns
from umbra.diagnostics.shadow_variable_finder import (
    AuxiliaryVariableReport,
    find_shadow_variables,
)
from umbra.sensitivity.grid_analysis import SensitivityReport, run_sensitivity_grid


@dataclass
class UmbraDiagnosticReport:
    """Comprehensive, structured diagnostic audit report for missing data.

    Properties
    ----------
    mcar : LittleMCARResult
        Global test of Missing Completely at Random.
    covariate_shift : PatternAnalysisReport
        Two-sample KS, Mann-Whitney, and Chi-squared distribution shift tests.
    residual_diagnostics : Dict[str, Dict[str, Any]]
        Tail concentration and residual non-linear dependencies.
    shadow_variables : Dict[str, AuxiliaryVariableReport]
        Candidate auxiliary instruments and relevance/exclusion assessments.
    mnar_evidence : Dict[str, MNARRiskReport]
        Synthesized risk scores and converging evidence indicators.
    sensitivity : Dict[str, SensitivityReport]
        Sensitivity sweeps and tipping point analyses across delta grids.
    warnings : List[str]
        Methodological caveats and identifiability limitations.
    recommendations : Dict[str, str]
        Scientifically defensible strategy recommendations per incomplete variable.
    """

    mcar: LittleMCARResult
    covariate_shift: PatternAnalysisReport
    residual_diagnostics: Dict[str, Dict[str, Any]]
    shadow_variables: Dict[str, AuxiliaryVariableReport]
    mnar_evidence: Dict[str, MNARRiskReport]
    sensitivity: Dict[str, SensitivityReport] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    recommendations: Dict[str, str] = field(default_factory=dict)
    n_samples: int = 0
    n_features: int = 0
    missing_columns: List[str] = field(default_factory=list)

    @property
    def overall_missing_rate(self) -> float:
        """Compute the global fraction of missing values across all cells."""
        if self.n_samples == 0 or self.n_features == 0:
            return 0.0
        total_missing = sum(rep.n_missing for rep in self.covariate_shift.variable_reports.values())
        return float(total_missing / (self.n_samples * self.n_features))

    @property
    def missing_counts(self) -> Dict[str, int]:
        """Dictionary of missing cell counts per variable."""
        return {col: rep.n_missing for col, rep in self.covariate_shift.variable_reports.items()}

    @property
    def missing_rates(self) -> Dict[str, float]:
        """Dictionary of missing cell rates per variable."""
        return {col: rep.missing_rate for col, rep in self.covariate_shift.variable_reports.items()}

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "UMBRA SCIENTIFIC MISSING DATA DIAGNOSTIC REPORT",
            "=" * 70,
            f"Dataset Dimensions: {self.n_samples:,} observations x {self.n_features} features",
            f"Incomplete Columns: {len(self.missing_columns)} ({', '.join(self.missing_columns) if self.missing_columns else 'None'})",
            "",
            "TIER 1: OBSERVED-DATA EVIDENCE",
            "----------------------------------------------------------------------",
            f"  * Little's MCAR Test: stat={self.mcar.statistic:.2f}, df={self.mcar.degrees_of_freedom}, p={self.mcar.p_value:.2e}",
            f"    Verdict: {'REJECT MCAR (MAR or MNAR likely)' if self.mcar.is_rejected else 'FAIL TO REJECT (Compatible with MCAR)'}",
        ]
        for col, rep in self.covariate_shift.variable_reports.items():
            lines.append(
                f"  * Covariate Shift for '{col}': Max KS={rep.max_ks_statistic:.3f} "
                f"({rep.n_significant_shifts} significant shifts)"
            )

        lines.extend(
            [
                "",
                "TIER 2: MODEL-BASED INFERENCE (EVIDENCE CONSISTENT WITH MNAR)",
                "----------------------------------------------------------------------",
            ]
        )
        for col, m_rep in self.mnar_evidence.items():
            lines.append(
                f"  * '{col}': Risk={m_rep.risk_level} (Score: {m_rep.composite_score:.2f}) | "
                f"Recommended: {m_rep.recommended_strategy}"
            )
            if col in self.residual_diagnostics:
                r_diag = self.residual_diagnostics[col]
                lines.append(f"    Tail Concentration Ratio: {r_diag.get('tail_ratio', 1.0):.2f}")

        lines.extend(
            [
                "",
                "TIER 3: UNTESTABLE ASSUMPTIONS & IDENTIFIABILITY LIMITATIONS",
                "----------------------------------------------------------------------",
            ]
        )
        for w in self.warnings:
            lines.append(f"  [!] {w}")

        if self.sensitivity:
            lines.extend(
                [
                    "",
                    "TIER 4: SENSITIVITY RESULTS & TIPPING POINTS",
                    "----------------------------------------------------------------------",
                ]
            )
            for col, sens in self.sensitivity.items():
                status = "FRAGILE" if sens.is_fragile else "ROBUST"
                lines.append(
                    f"  * '{col}': Sensitivity Interval [{sens.estimate_min:.3f}, {sens.estimate_max:.3f}] | "
                    f"Spread={sens.uncertainty_spread:.3f} | Conclusion: {status}"
                )
                for tp in sens.tipping_points:
                    lines.append(
                        f"    - Tipping Point: crosses zero at delta={tp.tipping_delta:+.2f} std devs"
                    )
        lines.extend(
            [
                "",
                "EPISTEMIC BOUNDARY: KNOW / ASSUME / CANNOT KNOW",
                "----------------------------------------------------------------------",
                "  [What Umbra Observes]:",
                "    - Little's MCAR test statistic and degrees of freedom",
                "    - Observable covariate distribution shifts across missingness indicators",
                "    - Residual tail concentration and non-linear patterns on observed rows",
                "    - Candidate auxiliary variable relevance (first-stage F-statistic)",
                "  [What Umbra Assumes]:",
                "    - MAR conditional independence when applying chained equations (MICE)",
                "    - Bivariate joint normality and exclusion restrictions when applying Heckman",
                "    - Residual standard-deviation scaled shift (delta) in pattern-mixture models",
                "  [What Umbra Cannot Establish]:",
                "    - True MNAR missingness mechanism from observed data alone (Molenberghs et al., 2008)",
                "    - Validity of an exclusion restriction from observational correlation alone",
                "    - Unobserved counterfactual distribution without unverifiable structural assumptions",
            ]
        )

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_info": {
                "n_samples": self.n_samples,
                "n_features": self.n_features,
                "missing_columns": self.missing_columns,
            },
            "epistemic_boundary": {
                "what_umbra_observes": [
                    "Missingness pattern frequencies and co-occurrence matrices",
                    "Covariate distribution shifts across response indicators",
                    "Residual tail concentration after MAR regression",
                    "Candidate auxiliary variable relevance (first-stage F > 10)",
                ],
                "what_umbra_assumes": [
                    "MAR conditional exchangeability under MICE",
                    "Bivariate normality and exclusion restrictions under Heckman selection",
                    "Specified residual shift parameter delta under pattern-mixture modeling",
                ],
                "what_umbra_cannot_establish": [
                    "True MNAR status from observed data alone (non-identifiable)",
                    "Exclusion restriction validity without substantive domain knowledge",
                    "The true unobserved outcome distribution without structural assumptions",
                ],
            },
            "tier1_observed_data_evidence": {
                "mcar_test": self.mcar.to_dict(),
                "covariate_shifts": self.covariate_shift.to_dict(),
            },
            "tier2_model_based_inference": {
                "residual_diagnostics": self.residual_diagnostics,
                "mnar_evidence": {k: v.to_dict() for k, v in self.mnar_evidence.items()},
                "shadow_variable_candidates": {
                    k: v.to_dict() for k, v in self.shadow_variables.items()
                },
            },
            "tier3_untestable_assumptions": {
                "warnings": self.warnings,
                "fundamental_identifiability_limit": (
                    "True MNAR cannot be mathematically distinguished from unmeasured "
                    "confounding using observed data alone. Auxiliary variable exclusion "
                    "restrictions are inherently untestable from data."
                ),
            },
            "tier4_sensitivity_results": {
                k: {
                    "baseline_mar": float(v.mar_baseline_estimate),
                    "estimate_min": float(v.estimate_min),
                    "estimate_max": float(v.estimate_max),
                    "uncertainty_spread": float(v.uncertainty_spread),
                    "is_fragile": bool(v.is_fragile),
                    "interpretation": v.interpretation,
                    "tipping_points": [
                        {
                            "metric_name": tp.metric_name,
                            "tipping_delta": float(tp.tipping_delta),
                            "description": tp.description,
                        }
                        for tp in v.tipping_points
                    ],
                }
                for k, v in self.sensitivity.items()
            },
            "recommendations": self.recommendations,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        lines = [
            "# Umbra Missingness Diagnostic Audit Report",
            "",
            "> [!IMPORTANT]",
            "> **Fundamental Identifiability Principle**: True MNAR is fundamentally unidentifiable from observed ",
            "> data alone. Umbra combines empirical diagnostics and sensitivity analyses to quantify evidence ",
            "> and assess how substantive conclusions shift across plausible departures from MAR.",
            "",
            f"**Dataset Overview**: {self.n_samples:,} observations, {self.n_features} features, "
            f"{len(self.missing_columns)} incomplete variable(s): `{', '.join(self.missing_columns) if self.missing_columns else 'None'}`.",
            "",
            "---",
            "",
            "## 1. Observed-Data Evidence (Empirically Testable)",
            "",
            "### Little's MCAR Test (1988)",
            f"- **Chi-Squared Statistic**: `{self.mcar.statistic:.4f}`",
            f"- **Degrees of Freedom**: `{self.mcar.degrees_of_freedom}`",
            f"- **p-value**: `{self.mcar.p_value:.4e}`",
            f"- **Verdict**: **{'REJECT MCAR (Systematic non-random missingness detected)' if self.mcar.is_rejected else 'FAIL TO REJECT MCAR (Missingness is statistically compatible with MCAR)'}**",
            f"- *Caveat*: {self.mcar.note}",
            "",
            "### Covariate Distribution Shifts",
            "| Incomplete Variable | Missing Rate | Max KS Stat | Significant Shifts | Strongest Associator |",
            "| :--- | :---: | :---: | :---: | :--- |",
        ]
        for col, rep in self.covariate_shift.variable_reports.items():
            lines.append(
                f"| `{col}` | {rep.missing_rate:.1%} | {rep.max_ks_statistic:.3f} | "
                f"{rep.n_significant_shifts}/{len(rep.covariate_shifts)} | `{rep.strongest_predictor or 'None'}` |"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 2. Model-Based Inference & MNAR Evidence Assessment",
                "",
                "| Variable | Evidence Level | Composite Score | Recommended Strategy | Candidate Auxiliary (Instrument) |",
                "| :--- | :---: | :---: | :--- | :--- |",
            ]
        )
        for col, mnar_rep in self.mnar_evidence.items():
            lines.append(
                f"| `{col}` | **{mnar_rep.risk_level}** | `{mnar_rep.composite_score:.2f}` | "
                f"`{mnar_rep.recommended_strategy}` | `{mnar_rep.shadow_candidate or 'None'}` |"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 3. Untestable Assumptions & Methodological Warnings",
                "",
            ]
        )
        for w in self.warnings:
            lines.append(f"> [!WARNING]\n> {w}\n")

        if self.sensitivity:
            lines.extend(
                [
                    "---",
                    "",
                    "## 4. Sensitivity Results & Tipping-Point Analysis",
                    "",
                    "| Variable | Baseline MAR | Sensitivity Range (delta in [-1, +1]) | Spread | Fragility | Tipping Points |",
                    "| :--- | :---: | :---: | :---: | :---: | :--- |",
                ]
            )
            for col, sens in self.sensitivity.items():
                tps = (
                    ", ".join([f"delta={tp.tipping_delta:+.2f}" for tp in sens.tipping_points])
                    if sens.tipping_points
                    else "None"
                )
                frag = "**FRAGILE**" if sens.is_fragile else "ROBUST"
                lines.append(
                    f"| `{col}` | {sens.mar_baseline_estimate:.3f} | "
                    f"[{sens.estimate_min:.3f}, {sens.estimate_max:.3f}] | {sens.uncertainty_spread:.3f} | {frag} | {tps} |"
                )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 5. Epistemic Boundary: Know / Assume / Cannot Know",
                "",
                "| Category | Empirical & Theoretical Scope |",
                "| :--- | :--- |",
                "| **What Umbra Observes** | - Exact missingness pattern co-occurrence matrices<br>- Empirical covariate distribution shifts between response indicators<br>- Residual tail concentration and non-linear patterns on observed cases<br>- Statistical relevance of candidate auxiliary variables ($F > 10$) |",
                r"| **What Umbra Assumes** | - Conditional exchangeability (MAR) when applying chained equations<br>- Bivariate joint normality and valid exclusion restrictions when applying Heckman selection<br>- Residual-scaled shift magnitude ($\delta$) when exploring pattern-mixture models |",
                "| **What Umbra Cannot Establish** | - **True MNAR mechanism from observed data alone** (Molenberghs et al., 2008)<br>- Validity of exclusion restrictions from observational data without domain knowledge<br>- Counterfactual unobserved distributions without untestable structural assumptions |",
                "",
            ]
        )
        return "\n".join(lines)

    def to_html(self) -> str:
        md = self.to_markdown()
        html_lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>Umbra Diagnostic Audit</title>",
            "<style>",
            "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; line-height: 1.6; color: #24292e; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #e1e4e8; padding: 10px 14px; text-align: left; }",
            "th { background-color: #f6f8fa; font-weight: 600; }",
            "tr:nth-child(even) { background-color: #fafbfc; }",
            "code { background-color: #f0f2f5; padding: 2px 6px; border-radius: 4px; font-family: monospace; }",
            ".tier { border-left: 4px solid #0366d6; padding-left: 14px; margin: 24px 0; }",
            ".warning { background-color: #fffbdd; border-left: 4px solid #d9a406; padding: 12px 16px; margin: 16px 0; }",
            "</style>",
            "</head>",
            "<body>",
            f"<pre style='white-space: pre-wrap;'>{md}</pre>",
            "</body>",
            "</html>",
        ]
        return "\n".join(html_lines)


def diagnose_report(
    data: Union[pd.DataFrame, np.ndarray],
    target_cols: Optional[List[str]] = None,
    shadow_cols: Optional[Dict[str, str]] = None,
    alpha: float = 0.05,
    run_sensitivity: bool = True,
    random_state: int = 42,
) -> UmbraDiagnosticReport:
    """Run full, multi-tier missingness diagnostics on dataset X.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Input data containing potentially missing values.
    target_cols : Optional[List[str]], default=None
        Specific incomplete columns to target. If None, targets all columns with missing values.
    shadow_cols : Optional[Dict[str, str]], default=None
        Pre-identified candidate instrumental shadow variables per target column.
    alpha : float, default=0.05
        Significance level for hypothesis tests.
    run_sensitivity : bool, default=True
        Whether to sweep sensitivity grid for variables with evidence consistent with MNAR.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    UmbraDiagnosticReport
        Structured diagnostic report object exposing:
        - report.mcar
        - report.covariate_shift
        - report.residual_diagnostics
        - report.shadow_variables
        - report.mnar_evidence
        - report.sensitivity
        - report.warnings
        - report.recommendations
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(data.shape[1])])

    n_samples, n_features = df.shape
    if target_cols is not None:
        missing_cols = [c for c in target_cols if c in df.columns and df[c].isna().any()]
    else:
        missing_cols = [c for c in df.columns if df[c].isna().any()]

    # 1. Little's MCAR Test
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1 and len(missing_cols) > 0:
        mcar_res = littles_mcar_test(numeric_df, alpha=alpha)
    else:
        mcar_res = LittleMCARResult(
            statistic=0.0,
            p_value=1.0,
            degrees_of_freedom=0,
            n_patterns=1,
            n_samples=n_samples,
            n_features=numeric_df.shape[1],
            is_rejected=False,
            alpha=alpha,
            note="Insufficient numeric columns or zero missing values.",
        )

    # 2. Covariate shift analysis
    covar_rep = analyze_missingness_patterns(df, alpha=alpha)

    # 3. Candidate auxiliary variables and MNAR evidence
    shadow_reports: Dict[str, AuxiliaryVariableReport] = {}
    mnar_reports: Dict[str, MNARRiskReport] = {}
    residual_diags: Dict[str, Dict[str, Any]] = {}
    recommendations: Dict[str, str] = {}

    for col in missing_cols:
        s_rep = find_shadow_variables(df, col)
        shadow_reports[col] = s_rep

        m_rep = assess_mnar_risk(
            df,
            target_col=col,
            littles_result=mcar_res,
            pattern_report=covar_rep,
            shadow_report=s_rep,
            alpha=alpha,
        )
        mnar_reports[col] = m_rep
        recommendations[col] = m_rep.recommended_strategy

        # Extract residual details
        for sig in m_rep.signals:
            if "Residual Tail" in sig.name:
                residual_diags[col] = sig.details

    # 4. Sensitivity grid analysis for columns with medium/high MNAR risk
    sensitivity_reports: Dict[str, SensitivityReport] = {}
    if run_sensitivity:
        for col, rep in mnar_reports.items():
            if rep.risk_level in ("MEDIUM", "HIGH") and pd.api.types.is_numeric_dtype(df[col]):
                sens = run_sensitivity_grid(df, target_column=col, random_state=random_state)
                sensitivity_reports[col] = sens

    # Formulate explicit warnings
    warnings_list = [
        "MNAR Non-Identifiability Limit: MNAR cannot generally be distinguished from MAR using observed data alone. "
        "Umbra quantifies empirical evidence and sensitivity rather than claiming mathematical certainty.",
        "Auxiliary Variable Instrument Assumption: Candidate auxiliary variables identified empirically do NOT "
        "guarantee causal exclusion. True instruments require domain verification that the variable has no direct path to the unobserved outcome.",
    ]
    if any(r.risk_level == "HIGH" for r in mnar_reports.values()):
        high_cols = [c for c, r in mnar_reports.items() if r.risk_level == "HIGH"]
        warnings_list.append(
            f"High MNAR Risk Detected in {high_cols}: Single-value MAR point imputations (e.g. standard MICE) "
            "are susceptible to selection bias. Inspect sensitivity intervals before drawing scientific conclusions."
        )

    return UmbraDiagnosticReport(
        mcar=mcar_res,
        covariate_shift=covar_rep,
        residual_diagnostics=residual_diags,
        shadow_variables=shadow_reports,
        mnar_evidence=mnar_reports,
        sensitivity=sensitivity_reports,
        warnings=warnings_list,
        recommendations=recommendations,
        n_samples=n_samples,
        n_features=n_features,
        missing_columns=missing_cols,
    )


# Alias for backwards compatibility
diagnose = diagnose_report

__all__ = ["UmbraDiagnosticReport", "diagnose_report", "diagnose"]
