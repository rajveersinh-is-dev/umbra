"""
Auxiliary / Candidate Shadow Variable Finder.

Surfaces candidate auxiliary variables (exclusion restrictions) that may
statistically support MNAR selection models (such as Heckman models).

CRITICAL METHODOLOGICAL WARNING (IDENTIFIABILITY LIMIT):
A valid shadow variable (instrument) Z requires:
1. Relevance: Z is strongly associated with the missingness indicator R_Y (F > 10).
2. Exclusion Restriction: Z has no direct causal path to outcome Y other than
   through the selection mechanism, conditional on covariates X (Cor(Z, Y | X) = 0).

UNTESTABLE ASSUMPTION:
Statistical correlation and partial correlation ALONE CANNOT prove that an
exclusion restriction holds. Umbra explicitly classifies these as
"candidate auxiliary variables", NOT "validated identification variables".
Domain expertise is strictly required to validate whether the exclusion
restriction causally holds.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


@dataclass
class AuxiliaryVariableCandidate:
    """Statistical evaluation metrics for a candidate auxiliary (shadow) variable.

    Attributes
    ----------
    variable_name : str
        Name of candidate auxiliary variable Z.
    target_column : str
        Name of incomplete outcome variable Y.
    relevance_correlation : float
        Correlation r(Z, R) with missingness indicator R.
    relevance_p_value : float
        P-value for H0: r(Z, R) = 0.
    first_stage_f_stat : float
        F-statistic from first-stage regression of R on Z and covariates X
        (Stock & Yogo weak instrument benchmark: F > 10).
    direct_outcome_correlation : float
        Bivariate correlation r(Z, Y) on observed subset.
    partial_outcome_correlation : float
        Partial correlation r(Z, Y | X) conditional on other observed covariates.
    partial_outcome_p_value : float
        P-value testing H0: r(Z, Y | X) = 0.
    candidate_score : float
        Heuristic ratio of relevance to conditional outcome association.
    is_statistically_viable_candidate : bool
        True if empirical thresholds are satisfied (relevance F > 10 or |r| >= 0.12,
        and |r_partial| <= 0.15).
    classification : str
        'candidate_auxiliary_variable' or 'statistically_insufficient'.
    caveat_warning : str
        Mandatory disclaimer regarding untestable exclusion restriction.
    """

    variable_name: str
    target_column: str
    relevance_correlation: float
    relevance_p_value: float
    first_stage_f_stat: float
    direct_outcome_correlation: float
    partial_outcome_correlation: float
    partial_outcome_p_value: float
    candidate_score: float
    is_statistically_viable_candidate: bool
    classification: str = "candidate_auxiliary_variable"
    caveat_warning: str = (
        "Statistical association does not establish the causal/exclusion assumptions "
        "required for identification. Domain knowledge is strictly required."
    )

    # Legacy compatibility alias
    @property
    def missingness_correlation(self) -> float:
        return self.relevance_correlation

    @property
    def outcome_correlation(self) -> float:
        return self.direct_outcome_correlation

    @property
    def shadow_score(self) -> float:
        return self.candidate_score

    @property
    def is_promising_candidate(self) -> bool:
        return self.is_statistically_viable_candidate

    @property
    def rationale(self) -> str:
        return (
            f"Associated with missingness (|r|={abs(self.relevance_correlation):.3f}, "
            f"F={self.first_stage_f_stat:.1f}) with conditional outcome correlation "
            f"|r_partial|={abs(self.partial_outcome_correlation):.3f} (p={self.partial_outcome_p_value:.3f})."
        )

    def summary(self) -> str:
        status = (
            "CANDIDATE AUXILIARY VARIABLE (Empirically Plausible)"
            if self.is_statistically_viable_candidate
            else "INSUFFICIENT STATISTICAL EVIDENCE"
        )
        return (
            f"Auxiliary Candidate '{self.variable_name}' for '{self.target_column}' [{status}]:\n"
            f"  - Missingness Relevance |r(Z, R)|       : {abs(self.relevance_correlation):.3f} (p={self.relevance_p_value:.3e})\n"
            f"  - First-Stage F-statistic               : {self.first_stage_f_stat:.2f} (Benchmark > 10)\n"
            f"  - Direct Outcome Correlation r(Z, Y)    : {self.direct_outcome_correlation:+.3f}\n"
            f"  - Partial Outcome Correlation r(Z, Y|X) : {self.partial_outcome_correlation:+.3f} (p={self.partial_outcome_p_value:.3e})\n"
            f"  - Heuristic Candidate Score             : {self.candidate_score:.2f}\n"
            f"  - Crucial Warning: {self.caveat_warning}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable_name": self.variable_name,
            "target_column": self.target_column,
            "relevance_correlation": float(self.relevance_correlation),
            "relevance_p_value": float(self.relevance_p_value),
            "first_stage_f_stat": float(self.first_stage_f_stat),
            "direct_outcome_correlation": float(self.direct_outcome_correlation),
            "partial_outcome_correlation": float(self.partial_outcome_correlation),
            "partial_outcome_p_value": float(self.partial_outcome_p_value),
            "candidate_score": float(self.candidate_score),
            "is_statistically_viable_candidate": bool(self.is_statistically_viable_candidate),
            "classification": self.classification,
            "caveat_warning": self.caveat_warning,
        }


# Legacy type alias
ShadowVariableCandidate = AuxiliaryVariableCandidate


@dataclass
class AuxiliaryVariableReport:
    """Summary of candidate auxiliary (shadow) variable discovery for an incomplete target.

    Attributes
    ----------
    target_column : str
        Target column with missing data.
    candidates : List[AuxiliaryVariableCandidate]
        List of all evaluated candidate variables.
    best_candidate : Optional[AuxiliaryVariableCandidate]
        Top candidate ranked by candidate_score.
    warning : str
        Methodological warning.
    """

    target_column: str
    candidates: List[AuxiliaryVariableCandidate] = field(default_factory=list)
    best_candidate: Optional[AuxiliaryVariableCandidate] = None
    warning: str = (
        "Statistical association does not establish the causal/exclusion assumptions "
        "required for identification. A variable correlated with missingness is NOT "
        "automatically a valid instrument. Domain expertise is strictly required."
    )

    def summary(self) -> str:
        lines = [
            f"Candidate Auxiliary Variable Report for '{self.target_column}':",
            f"  - Candidates Evaluated: {len(self.candidates)}",
        ]
        if self.best_candidate and self.best_candidate.is_statistically_viable_candidate:
            lines.append(
                f"  - Top Candidate: '{self.best_candidate.variable_name}' "
                f"(Score: {self.best_candidate.candidate_score:.2f}, F={self.best_candidate.first_stage_f_stat:.1f})"
            )
        else:
            lines.append(
                "  - No variable met statistical candidate criteria (relevance + conditional independence)."
            )
        lines.append(f"  - Methodological Warning:\n    {self.warning}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_column": self.target_column,
            "best_candidate": self.best_candidate.to_dict() if self.best_candidate else None,
            "candidates": [c.to_dict() for c in self.candidates],
            "warning": self.warning,
        }


# Legacy type alias
ShadowFinderReport = AuxiliaryVariableReport


def _compute_partial_correlation_and_p(
    x: np.ndarray, y: np.ndarray, covars: np.ndarray
) -> Tuple[float, float]:
    """Compute partial correlation between x and y partialling out covars using OLS residuals,
    along with two-tailed p-value.
    """
    n = len(x)
    k = covars.shape[1] if covars.ndim > 1 else 0

    if k == 0 or covars.size == 0:
        r, p = stats.pearsonr(x, y)
        return float(r), float(p)

    covars_const = np.column_stack([np.ones(n), covars])
    try:
        beta_x = np.linalg.lstsq(covars_const, x, rcond=None)[0]
        res_x = x - (covars_const @ beta_x)

        beta_y = np.linalg.lstsq(covars_const, y, rcond=None)[0]
        res_y = y - (covars_const @ beta_y)

        # Check for zero residual variance
        if np.std(res_x) < 1e-10 or np.std(res_y) < 1e-10:
            return 0.0, 1.0

        r, p = stats.pearsonr(res_x, res_y)
        # Degree of freedom adjustment: df = n - k - 2
        df = max(1, n - k - 2)
        t_stat = r * np.sqrt(df / max(1e-12, 1.0 - r**2))
        p_val = 2.0 * float(stats.t.sf(np.abs(t_stat), df))
        return float(r), float(p_val)
    except Exception:
        return 0.0, 1.0


def _compute_first_stage_f_stat(R: np.ndarray, Z: np.ndarray, X_covars: np.ndarray) -> float:
    """Compute first-stage F-statistic for instrument Z in predicting missingness indicator R,
    controlling for observed covariates X_covars.
    """
    n = len(R)
    try:
        # Restricted model (X only)
        if X_covars.shape[1] > 0:
            X_res = sm.add_constant(X_covars, has_constant="add")
            mod_res = sm.OLS(R, X_res).fit()
            ssr_res = mod_res.ssr
        else:
            X_res = np.ones((n, 1))
            mod_res = sm.OLS(R, X_res).fit()
            ssr_res = mod_res.ssr

        # Unrestricted model (X + Z)
        if X_covars.shape[1] > 0:
            X_unres = sm.add_constant(np.column_stack([X_covars, Z]), has_constant="add")
        else:
            X_unres = sm.add_constant(Z.reshape(-1, 1), has_constant="add")

        mod_unres = sm.OLS(R, X_unres).fit()
        ssr_unres = mod_unres.ssr
        df_unres = mod_unres.df_resid

        df_num = 1
        df_denom = max(1.0, df_unres)
        f_stat = ((ssr_res - ssr_unres) / df_num) / (ssr_unres / df_denom)
        return float(max(0.0, f_stat))
    except Exception:
        # Fallback to simple bivariate F-stat: t^2
        try:
            r, _ = stats.pearsonr(Z, R)
            df = max(1, n - 2)
            f_fallback = (r**2 / max(1e-10, 1 - r**2)) * df
            return float(f_fallback)
        except Exception:
            return 0.0


def find_shadow_variables(
    data: pd.DataFrame,
    target_column: str,
    min_missingness_corr: float = 0.12,
    max_partial_outcome_corr: float = 0.15,
) -> AuxiliaryVariableReport:
    """Scan covariates to surface candidate auxiliary / shadow variables.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing target_column and candidate auxiliary features.
    target_column : str
        The variable with missing values under investigation.
    min_missingness_corr : float, default=0.12
        Minimum correlation with missingness indicator to consider as relevant.
    max_partial_outcome_corr : float, default=0.15
        Maximum conditional correlation with outcome to consider as conditionally independent.

    Returns
    -------
    AuxiliaryVariableReport
    """
    if target_column not in data.columns:
        raise ValueError(f"Column '{target_column}' not found in data.")

    is_missing = data[target_column].isna().astype(float).to_numpy()
    n_missing = int(np.sum(is_missing))
    n_total = len(data)

    if n_missing == 0:
        return AuxiliaryVariableReport(
            target_column=target_column,
            candidates=[],
            best_candidate=None,
            warning="Target variable has no missing values.",
        )

    obs_mask = data[target_column].notna()
    y_obs = data.loc[obs_mask, target_column].to_numpy(dtype=float)

    # Candidate numeric features with reasonably complete data
    candidate_cols = [
        c
        for c in data.columns
        if c != target_column
        and pd.api.types.is_numeric_dtype(data[c])
        and data[c].isna().mean() < 0.20
    ]

    evaluated_candidates: List[AuxiliaryVariableCandidate] = []

    for cand_col in candidate_cols:
        col_series = data[cand_col].fillna(data[cand_col].median()).to_numpy(dtype=float)

        # 1. Relevance: Correlation with missingness indicator
        r_miss, p_miss = stats.pearsonr(col_series, is_missing)

        # 2. Direct outcome correlation on observed rows
        cand_obs = col_series[obs_mask]
        if len(cand_obs) < 3 or np.std(cand_obs) < 1e-10 or np.std(y_obs) < 1e-10:
            r_out, p_out = 0.0, 1.0
            partial_r, partial_p = 0.0, 1.0
            f_stat = 0.0
        else:
            r_out, p_out = stats.pearsonr(cand_obs, y_obs)

            # Other covariates for conditioning
            other_covars = [c for c in candidate_cols if c != cand_col]
            if other_covars:
                covar_obs = (
                    data.loc[obs_mask, other_covars]
                    .fillna(data[other_covars].median())
                    .to_numpy(dtype=float)
                )
                covar_all = (
                    data[other_covars].fillna(data[other_covars].median()).to_numpy(dtype=float)
                )
                partial_r, partial_p = _compute_partial_correlation_and_p(
                    cand_obs, y_obs, covar_obs
                )
                f_stat = _compute_first_stage_f_stat(is_missing, col_series, covar_all)
            else:
                partial_r, partial_p = float(r_out), float(p_out)
                f_stat = _compute_first_stage_f_stat(is_missing, col_series, np.empty((n_total, 0)))

        eps = 0.05
        score = float(abs(r_miss) / (abs(partial_r) + eps))

        # Viability: relevant (correlation or F > 10) AND weak partial correlation with outcome
        is_viable = bool(
            (abs(r_miss) >= min_missingness_corr or f_stat >= 10.0)
            and abs(partial_r) <= max_partial_outcome_corr
            and p_miss < 0.05
        )

        classification = (
            "candidate_auxiliary_variable" if is_viable else "statistically_insufficient"
        )

        cand_obj = AuxiliaryVariableCandidate(
            variable_name=cand_col,
            target_column=target_column,
            relevance_correlation=float(r_miss),
            relevance_p_value=float(p_miss),
            first_stage_f_stat=float(f_stat),
            direct_outcome_correlation=float(r_out),
            partial_outcome_correlation=float(partial_r),
            partial_outcome_p_value=float(partial_p),
            candidate_score=float(score),
            is_statistically_viable_candidate=is_viable,
            classification=classification,
        )
        evaluated_candidates.append(cand_obj)

    # Sort candidates by candidate_score descending
    evaluated_candidates.sort(
        key=lambda c: (c.is_statistically_viable_candidate, c.candidate_score), reverse=True
    )
    best = evaluated_candidates[0] if evaluated_candidates else None

    return AuxiliaryVariableReport(
        target_column=target_column,
        candidates=evaluated_candidates,
        best_candidate=best,
    )
