"""
Missingness pattern analysis and covariate distribution shift detection.

Compares observed-variable distributions across missing vs. observed groups
to identify systematic dependencies.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CovariateShift:
    """Distribution comparison for a single covariate across missing vs. observed rows."""

    covariate_name: str
    is_numeric: bool
    ks_statistic: Optional[float] = None
    ks_p_value: Optional[float] = None
    mw_statistic: Optional[float] = None
    mw_p_value: Optional[float] = None
    cohens_d: Optional[float] = None
    cliffs_delta: Optional[float] = None
    chi2_statistic: Optional[float] = None
    chi2_p_value: Optional[float] = None
    mean_observed: Optional[float] = None
    mean_missing: Optional[float] = None
    is_significant: bool = False

    def summary(self) -> str:
        if self.is_numeric:
            diff = (
                (self.mean_missing - self.mean_observed)
                if (self.mean_missing is not None and self.mean_observed is not None)
                else 0.0
            )
            return (
                f"{self.covariate_name}: KS={self.ks_statistic:.3f} (p={self.ks_p_value:.3e}), "
                f"Cohen's d={self.cohens_d:.3f}, Diff={diff:+.3f} "
                f"[{'SHIFT DETECTED' if self.is_significant else 'NO SHIFT'}]"
            )
        else:
            return (
                f"{self.covariate_name}: Chi2={self.chi2_statistic:.3f} (p={self.chi2_p_value:.3e}) "
                f"[{'SHIFT DETECTED' if self.is_significant else 'NO SHIFT'}]"
            )


@dataclass
class VariablePatternReport:
    """Missingness pattern and covariate shift report for a single variable with missing data."""

    target_column: str
    n_total: int
    n_missing: int
    missing_rate: float
    covariate_shifts: Dict[str, CovariateShift]
    max_ks_statistic: float = 0.0
    max_cohens_d: float = 0.0
    n_significant_shifts: int = 0
    strongest_predictor: Optional[str] = None

    def summary(self) -> str:
        return (
            f"Variable '{self.target_column}': {self.n_missing}/{self.n_total} missing ({self.missing_rate:.1%})\n"
            f"  - Significant Covariate Shifts: {self.n_significant_shifts}/{len(self.covariate_shifts)}\n"
            f"  - Max KS Statistic: {self.max_ks_statistic:.3f} (Strongest: {self.strongest_predictor})\n"
            f"  - Max |Cohen's d|: {self.max_cohens_d:.3f}"
        )


@dataclass
class PatternAnalysisReport:
    """Full missingness pattern and covariate shift report across all columns."""

    variable_reports: Dict[str, VariablePatternReport] = field(default_factory=dict)
    co_missingness_matrix: Optional[pd.DataFrame] = None
    pattern_table: Optional[pd.DataFrame] = None


def compute_cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_s < 1e-12:
        return 0.0
    return float((np.mean(x2) - np.mean(x1)) / pooled_s)


def compute_cliffs_delta(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Cliff's delta non-parametric effect size."""
    n1, n2 = len(x1), len(x2)
    if n1 == 0 or n2 == 0:
        return 0.0
    # For large sample sizes, use Mann-Whitney U relation: delta = 2*U / (n1*n2) - 1
    u_stat, _ = stats.mannwhitneyu(x2, x1, alternative="two-sided")
    return float((2.0 * u_stat / (n1 * n2)) - 1.0)


def analyze_missingness_patterns(
    data: pd.DataFrame,
    alpha: float = 0.05,
) -> PatternAnalysisReport:
    """
    Analyze missingness patterns and test for covariate shifts across missingness indicators.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing features with potential missing values.
    alpha : float, default=0.05
        Significance threshold for covariate shift hypothesis tests.

    Returns
    -------
    PatternAnalysisReport
    """
    missing_cols = [col for col in data.columns if data[col].isna().any()]
    if not missing_cols:
        return PatternAnalysisReport()

    n_rows = len(data)
    variable_reports = {}

    for target_col in missing_cols:
        is_missing = data[target_col].isna()
        n_mis = int(is_missing.sum())
        n_obs = n_rows - n_mis

        if n_obs < 5 or n_mis < 5:
            # Insufficient samples in one group to perform distributional tests
            continue

        covariate_shifts = {}
        max_ks = 0.0
        max_d = 0.0
        strongest_pred = None
        sig_count = 0

        for cov_col in data.columns:
            if cov_col == target_col:
                continue

            # Consider only rows where covariate itself is observed
            valid_cov = data[cov_col].notna()
            obs_group = data.loc[valid_cov & (~is_missing), cov_col]
            mis_group = data.loc[valid_cov & is_missing, cov_col]

            if len(obs_group) < 5 or len(mis_group) < 5:
                continue

            if pd.api.types.is_numeric_dtype(data[cov_col]):
                vals_obs = obs_group.to_numpy(dtype=float)
                vals_mis = mis_group.to_numpy(dtype=float)

                # Two-sample Kolmogorov-Smirnov test
                ks_res = stats.ks_2samp(vals_obs, vals_mis)
                # Mann-Whitney U test
                mw_res = stats.mannwhitneyu(vals_obs, vals_mis, alternative="two-sided")

                d = compute_cohens_d(vals_obs, vals_mis)
                delta = compute_cliffs_delta(vals_obs, vals_mis)
                is_sig = bool(ks_res.pvalue < alpha)

                if is_sig:
                    sig_count += 1
                if ks_res.statistic > max_ks:
                    max_ks = float(ks_res.statistic)
                    strongest_pred = cov_col
                if abs(d) > max_d:
                    max_d = float(abs(d))

                covariate_shifts[cov_col] = CovariateShift(
                    covariate_name=cov_col,
                    is_numeric=True,
                    ks_statistic=float(ks_res.statistic),
                    ks_p_value=float(ks_res.pvalue),
                    mw_statistic=float(mw_res.statistic),
                    mw_p_value=float(mw_res.pvalue),
                    cohens_d=float(d),
                    cliffs_delta=float(delta),
                    mean_observed=float(np.mean(vals_obs)),
                    mean_missing=float(np.mean(vals_mis)),
                    is_significant=is_sig,
                )
            else:
                # Categorical / Discrete
                contingency = pd.crosstab(data[cov_col], is_missing)
                chi2_stat, p_val, _, _ = stats.chi2_contingency(contingency)
                is_sig = bool(p_val < alpha)
                if is_sig:
                    sig_count += 1

                covariate_shifts[cov_col] = CovariateShift(
                    covariate_name=cov_col,
                    is_numeric=False,
                    chi2_statistic=float(chi2_stat),
                    chi2_p_value=float(p_val),
                    is_significant=is_sig,
                )

        variable_reports[target_col] = VariablePatternReport(
            target_column=target_col,
            n_total=n_rows,
            n_missing=n_mis,
            missing_rate=float(n_mis / n_rows),
            covariate_shifts=covariate_shifts,
            max_ks_statistic=max_ks,
            max_cohens_d=max_d,
            n_significant_shifts=sig_count,
            strongest_predictor=strongest_pred,
        )

    # Co-missingness correlation matrix
    co_missingness = None
    if len(missing_cols) > 1:
        mask_df = data[missing_cols].isna().astype(int)
        co_missingness = mask_df.corr(method="pearson")

    # Patterns frequency table
    mask_all = data[missing_cols].isna()
    pattern_signatures = mask_all.apply(
        lambda row: "".join(["1" if v else "0" for v in row]), axis=1
    )
    pat_counts = pattern_signatures.value_counts()
    pat_df = pd.DataFrame(
        {
            "pattern_bits": pat_counts.index,
            "count": pat_counts.values,
            "percentage": (pat_counts.values / n_rows) * 100.0,
        }
    )

    return PatternAnalysisReport(
        variable_reports=variable_reports,
        co_missingness_matrix=co_missingness,
        pattern_table=pat_df,
    )
