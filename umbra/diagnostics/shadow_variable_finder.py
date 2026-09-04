"""
Shadow / Auxiliary Variable Finder.

Surfaces candidate instrumental variables (exclusion restrictions) that can
help identify MNAR models (such as Heckman selection models).

A valid shadow variable Z must satisfy two conditions:
1. Relevance: Z is strongly correlated with the missingness indicator R_Y.
2. Exclusion: Z is conditionally independent of outcome Y given other observed
   covariates X (i.e. Cor(Z, Y | X) ~= 0).

CRITICAL WARNING:
Empirical correlation alone CANNOT prove that an exclusion restriction holds.
A candidate identified here MUST be substantively validated by domain experts.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ShadowVariableCandidate:
    """Evaluation metrics for a candidate shadow variable."""

    variable_name: str
    target_column: str
    missingness_correlation: float
    outcome_correlation: float
    partial_outcome_correlation: float
    relevance_p_value: float
    shadow_score: float
    is_promising_candidate: bool
    rationale: str

    def summary(self) -> str:
        status = "PROMISING CANDIDATE" if self.is_promising_candidate else "WEAK / INVALID"
        return (
            f"Candidate '{self.variable_name}' for '{self.target_column}' [{status}]:\n"
            f"  - Missingness Correlation |r(Z, R)| : {abs(self.missingness_correlation):.3f} (p={self.relevance_p_value:.3e})\n"
            f"  - Partial Outcome Correlation |r(Z, Y|X)|: {abs(self.partial_outcome_correlation):.3f}\n"
            f"  - Instrument Shadow Score             : {self.shadow_score:.2f}\n"
            f"  - Substantive Rationale               : {self.rationale}"
        )


@dataclass
class ShadowFinderReport:
    """Summary of shadow variable search for an incomplete variable."""

    target_column: str
    candidates: List[ShadowVariableCandidate]
    best_candidate: Optional[ShadowVariableCandidate]
    warning: str

    def summary(self) -> str:
        lines = [
            f"Shadow Variable Discovery Report for '{self.target_column}':",
            f"  - Candidates Evaluated: {len(self.candidates)}",
        ]
        if self.best_candidate and self.best_candidate.is_promising_candidate:
            lines.append(
                f"  - Top Candidate: {self.best_candidate.variable_name} (Score: {self.best_candidate.shadow_score:.2f})"
            )
        else:
            lines.append(
                "  - No strong candidate instrument found with clean exclusion properties."
            )
        lines.append(f"  - Methodological Warning: {self.warning}")
        return "\n".join(lines)


def _compute_partial_correlation(x: np.ndarray, y: np.ndarray, covars: np.ndarray) -> float:
    """Compute partial correlation between x and y partialling out covars using OLS residuals."""
    if covars.shape[1] == 0:
        r, _ = stats.pearsonr(x, y)
        return float(r)

    # Regress x on covars
    covars_const = np.column_stack([np.ones(len(covars)), covars])
    try:
        beta_x = np.linalg.lstsq(covars_const, x, rcond=None)[0]
        res_x = x - (covars_const @ beta_x)

        beta_y = np.linalg.lstsq(covars_const, y, rcond=None)[0]
        res_y = y - (covars_const @ beta_y)

        r, _ = stats.pearsonr(res_x, res_y)
        return float(r)
    except Exception:
        return 0.0


def find_shadow_variables(
    data: pd.DataFrame,
    target_column: str,
    min_missingness_corr: float = 0.12,
    max_partial_outcome_corr: float = 0.15,
) -> ShadowFinderReport:
    """
    Scan covariates to surface candidate shadow / instrumental variables.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing target_column and other features.
    target_column : str
        The variable with missing values under investigation.
    min_missingness_corr : float, default=0.12
        Minimum correlation with missingness indicator to consider as relevant.
    max_partial_outcome_corr : float, default=0.15
        Maximum conditional correlation with outcome to consider as conditionally independent.

    Returns
    -------
    ShadowFinderReport
    """
    warning = (
        "Shadow variable candidates are selected purely through empirical correlations. "
        "Domain knowledge is strictly required to verify the exclusion restriction "
        "(i.e., that the variable does not directly influence the unobserved outcome)."
    )

    if target_column not in data.columns:
        raise ValueError(f"Column '{target_column}' not found in data.")

    is_missing = data[target_column].isna().astype(int)
    n_missing = int(is_missing.sum())
    if n_missing == 0:
        return ShadowFinderReport(
            target_column=target_column,
            candidates=[],
            best_candidate=None,
            warning="Target variable has no missing values.",
        )

    # Observed subset for computing outcome relationships
    obs_mask = data[target_column].notna()
    y_obs = data.loc[obs_mask, target_column].to_numpy(dtype=float)

    # Candidate columns: numeric features with complete or near-complete data
    candidate_cols = [
        c
        for c in data.columns
        if c != target_column
        and pd.api.types.is_numeric_dtype(data[c])
        and data[c].isna().mean() < 0.10
    ]

    evaluated_candidates: List[ShadowVariableCandidate] = []

    for cand_col in candidate_cols:
        # Impute temporary median for minor missingness in candidate column
        cand_series = data[cand_col].fillna(data[cand_col].median()).to_numpy(dtype=float)

        # 1. Relevance: Correlation with missingness indicator R_Y
        r_miss, p_miss = stats.pearsonr(cand_series, is_missing)

        # 2. Direct outcome correlation on observed data
        cand_obs = cand_series[obs_mask]
        r_out, _ = stats.pearsonr(cand_obs, y_obs)

        # 3. Partial outcome correlation conditioning on other candidates
        other_covars = [c for c in candidate_cols if c != cand_col]
        if other_covars:
            covar_matrix = (
                data.loc[obs_mask, other_covars]
                .fillna(data[other_covars].median())
                .to_numpy(dtype=float)
            )
            partial_r = _compute_partial_correlation(cand_obs, y_obs, covar_matrix)
        else:
            partial_r = float(r_out)

        # 4. Shadow score: high missingness correlation divided by low partial outcome correlation
        eps = 0.05
        score = float(abs(r_miss) / (abs(partial_r) + eps))

        is_promising = bool(
            abs(r_miss) >= min_missingness_corr
            and abs(partial_r) <= max_partial_outcome_corr
            and p_miss < 0.05
        )

        if is_promising:
            rationale = (
                f"Correlates with missingness (|r|={abs(r_miss):.2f}, p={p_miss:.2e}) "
                f"while showing weak direct partial connection to {target_column} (|r_partial|={abs(partial_r):.2f})."
            )
        else:
            reasons = []
            if abs(r_miss) < min_missingness_corr:
                reasons.append(
                    f"low missingness correlation ({abs(r_miss):.2f} < {min_missingness_corr})"
                )
            if abs(partial_r) > max_partial_outcome_corr:
                reasons.append(
                    f"high partial outcome correlation ({abs(partial_r):.2f} > {max_partial_outcome_corr})"
                )
            if p_miss >= 0.05:
                reasons.append("not statistically significant missingness correlation")
            rationale = "Unsuitable: " + ", ".join(reasons)

        evaluated_candidates.append(
            ShadowVariableCandidate(
                variable_name=cand_col,
                target_column=target_column,
                missingness_correlation=float(r_miss),
                outcome_correlation=float(r_out),
                partial_outcome_correlation=float(partial_r),
                relevance_p_value=float(p_miss),
                shadow_score=score,
                is_promising_candidate=is_promising,
                rationale=rationale,
            )
        )

    # Sort candidates by shadow score descending
    evaluated_candidates.sort(key=lambda c: c.shadow_score, reverse=True)
    best_candidate = evaluated_candidates[0] if evaluated_candidates else None

    return ShadowFinderReport(
        target_column=target_column,
        candidates=evaluated_candidates,
        best_candidate=best_candidate,
        warning=warning,
    )
