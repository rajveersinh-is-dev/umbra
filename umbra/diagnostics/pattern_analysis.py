"""
Missingness pattern analysis and covariate distribution shift detection.

Compares observed-variable distributions across missing vs. observed groups
to identify systematic dependencies (evidence for MAR / departures from MCAR).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CovariateShift:
    """Distribution comparison for a single covariate across missing vs. observed rows.

    Attributes
    ----------
    covariate_name : str
        Name of the evaluated covariate.
    is_numeric : bool
        Whether the covariate is numeric or categorical.
    ks_statistic : Optional[float]
        Kolmogorov-Smirnov two-sample test statistic (numeric).
    ks_p_value : Optional[float]
        P-value for the KS test.
    mw_statistic : Optional[float]
        Mann-Whitney U rank test statistic (numeric).
    mw_p_value : Optional[float]
        P-value for the Mann-Whitney test.
    cohens_d : Optional[float]
        Standardized mean difference (Cohen's d).
    cliffs_delta : Optional[float]
        Non-parametric effect size (Cliff's delta, in [-1, +1]).
    cramers_v : Optional[float]
        Effect size for categorical association (Cramer's V, in [0, 1]).
    chi2_statistic : Optional[float]
        Chi-squared contingency test statistic (categorical).
    chi2_p_value : Optional[float]
        P-value for Chi-squared test.
    mean_observed : Optional[float]
        Mean of covariate when target is observed.
    mean_missing : Optional[float]
        Mean of covariate when target is missing.
    is_significant : bool
        Whether any distribution test rejected at the specified alpha.
    """

    covariate_name: str
    is_numeric: bool
    ks_statistic: Optional[float] = None
    ks_p_value: Optional[float] = None
    mw_statistic: Optional[float] = None
    mw_p_value: Optional[float] = None
    cohens_d: Optional[float] = None
    cliffs_delta: Optional[float] = None
    cramers_v: Optional[float] = None
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
            ks_s = f"{self.ks_statistic:.3f}" if self.ks_statistic is not None else "N/A"
            ks_p = f"{self.ks_p_value:.3e}" if self.ks_p_value is not None else "N/A"
            d_s = f"{self.cohens_d:.3f}" if self.cohens_d is not None else "N/A"
            status = "SHIFT DETECTED" if self.is_significant else "NO SHIFT"
            return (
                f"{self.covariate_name}: KS={ks_s} (p={ks_p}), "
                f"Cohen's d={d_s}, Diff={diff:+.3f} [{status}]"
            )
        else:
            chi_s = f"{self.chi2_statistic:.3f}" if self.chi2_statistic is not None else "N/A"
            chi_p = f"{self.chi2_p_value:.3e}" if self.chi2_p_value is not None else "N/A"
            v_s = f"{self.cramers_v:.3f}" if self.cramers_v is not None else "N/A"
            status = "SHIFT DETECTED" if self.is_significant else "NO SHIFT"
            return f"{self.covariate_name}: Chi2={chi_s} (p={chi_p}), Cramer's V={v_s} [{status}]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "covariate_name": self.covariate_name,
            "is_numeric": self.is_numeric,
            "ks_statistic": self.ks_statistic,
            "ks_p_value": self.ks_p_value,
            "mw_statistic": self.mw_statistic,
            "mw_p_value": self.mw_p_value,
            "cohens_d": self.cohens_d,
            "cliffs_delta": self.cliffs_delta,
            "cramers_v": self.cramers_v,
            "chi2_statistic": self.chi2_statistic,
            "chi2_p_value": self.chi2_p_value,
            "mean_observed": self.mean_observed,
            "mean_missing": self.mean_missing,
            "is_significant": self.is_significant,
        }


@dataclass
class VariablePatternReport:
    """Missingness pattern and covariate shift report for a single incomplete variable."""

    target_column: str
    n_total: int
    n_missing: int
    missing_rate: float
    covariate_shifts: Dict[str, CovariateShift] = field(default_factory=dict)
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_column": self.target_column,
            "n_total": self.n_total,
            "n_missing": self.n_missing,
            "missing_rate": self.missing_rate,
            "max_ks_statistic": self.max_ks_statistic,
            "max_cohens_d": self.max_cohens_d,
            "n_significant_shifts": self.n_significant_shifts,
            "strongest_predictor": self.strongest_predictor,
            "covariate_shifts": {k: v.to_dict() for k, v in self.covariate_shifts.items()},
        }


@dataclass
class PatternAnalysisReport:
    """Full missingness pattern and covariate shift report across all columns."""

    variable_reports: Dict[str, VariablePatternReport] = field(default_factory=dict)
    co_missingness_matrix: Optional[pd.DataFrame] = None
    pattern_table: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable_reports": {k: v.to_dict() for k, v in self.variable_reports.items()},
            "co_missingness": (
                self.co_missingness_matrix.to_dict()
                if self.co_missingness_matrix is not None
                else None
            ),
            "patterns": (
                self.pattern_table.to_dict(orient="records")
                if self.pattern_table is not None
                else []
            ),
        }


def compute_cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups.

    Formula:
      d = (mean(x2) - mean(x1)) / s_pooled
      s_pooled = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))
    """
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / max(1, n1 + n2 - 2))
    if pooled_s < 1e-12:
        return 0.0
    return float((np.mean(x2) - np.mean(x1)) / pooled_s)


def compute_cliffs_delta(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Cliff's delta non-parametric effect size.

    Formula:
      delta = 2 * U / (n1 * n2) - 1
    where U is the Mann-Whitney U statistic.
    """
    n1, n2 = len(x1), len(x2)
    if n1 == 0 or n2 == 0:
        return 0.0
    try:
        u_stat, _ = stats.mannwhitneyu(x2, x1, alternative="two-sided")
        return float((2.0 * u_stat / (n1 * n2)) - 1.0)
    except Exception:
        return 0.0


def compute_cramers_v(contingency_table: pd.DataFrame) -> float:
    """Compute Cramer's V for a 2D contingency table."""
    try:
        chi2, _, _, _ = stats.chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        if n == 0:
            return 0.0
        r, k = contingency_table.shape
        min_dim = min(r - 1, k - 1)
        if min_dim == 0:
            return 0.0
        return float(np.sqrt((chi2 / n) / min_dim))
    except Exception:
        return 0.0


def analyze_missingness_patterns(
    data: pd.DataFrame,
    alpha: float = 0.05,
) -> PatternAnalysisReport:
    """Analyze missingness patterns and test for covariate distribution shifts.

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
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    missing_cols = [col for col in data.columns if data[col].isna().any()]
    if not missing_cols:
        return PatternAnalysisReport()

    n_rows = len(data)
    variable_reports: Dict[str, VariablePatternReport] = {}

    for target_col in missing_cols:
        is_missing = data[target_col].isna()
        n_mis = int(is_missing.sum())
        n_obs = n_rows - n_mis

        if n_obs < 3 or n_mis < 3:
            # Too few observations in one group for distributional inference
            variable_reports[target_col] = VariablePatternReport(
                target_column=target_col,
                n_total=n_rows,
                n_missing=n_mis,
                missing_rate=float(n_mis / n_rows),
            )
            continue

        covariate_shifts: Dict[str, CovariateShift] = {}
        max_ks = 0.0
        max_d = 0.0
        strongest_pred: Optional[str] = None
        sig_count = 0

        for cov_col in data.columns:
            if cov_col == target_col:
                continue

            valid_cov = data[cov_col].notna()
            obs_group = data.loc[valid_cov & (~is_missing), cov_col]
            mis_group = data.loc[valid_cov & is_missing, cov_col]

            if len(obs_group) < 3 or len(mis_group) < 3:
                continue

            if pd.api.types.is_numeric_dtype(data[cov_col]):
                vals_obs = obs_group.to_numpy(dtype=float)
                vals_mis = mis_group.to_numpy(dtype=float)

                # Check for constant arrays
                obs_constant = np.all(vals_obs == vals_obs[0])
                mis_constant = np.all(vals_mis == vals_mis[0])

                if obs_constant and mis_constant:
                    ks_stat, ks_p = 0.0, 1.0
                    mw_stat, mw_p = 0.0, 1.0
                    d = 0.0
                    delta = 0.0
                else:
                    ks_res = stats.ks_2samp(vals_obs, vals_mis)
                    ks_stat, ks_p = float(ks_res.statistic), float(ks_res.pvalue)

                    try:
                        mw_res = stats.mannwhitneyu(vals_obs, vals_mis, alternative="two-sided")
                        mw_stat, mw_p = float(mw_res.statistic), float(mw_res.pvalue)
                    except Exception:
                        mw_stat, mw_p = 0.0, 1.0

                    d = compute_cohens_d(vals_obs, vals_mis)
                    delta = compute_cliffs_delta(vals_obs, vals_mis)

                is_sig = bool(ks_p < alpha)
                if is_sig:
                    sig_count += 1
                if ks_stat > max_ks:
                    max_ks = ks_stat
                    strongest_pred = cov_col
                if abs(d) > max_d:
                    max_d = abs(d)

                covariate_shifts[cov_col] = CovariateShift(
                    covariate_name=cov_col,
                    is_numeric=True,
                    ks_statistic=ks_stat,
                    ks_p_value=ks_p,
                    mw_statistic=mw_stat,
                    mw_p_value=mw_p,
                    cohens_d=d,
                    cliffs_delta=delta,
                    mean_observed=float(np.mean(vals_obs)),
                    mean_missing=float(np.mean(vals_mis)),
                    is_significant=is_sig,
                )
            else:
                # Categorical / nominal
                contingency = pd.crosstab(data[cov_col], is_missing)
                if contingency.size > 0:
                    chi2_stat, p_val, _, _ = stats.chi2_contingency(contingency)
                    cramers = compute_cramers_v(contingency)
                    is_sig = bool(p_val < alpha)
                    if is_sig:
                        sig_count += 1

                    covariate_shifts[cov_col] = CovariateShift(
                        covariate_name=cov_col,
                        is_numeric=False,
                        chi2_statistic=float(chi2_stat),
                        chi2_p_value=float(p_val),
                        cramers_v=cramers,
                        is_significant=is_sig,
                    )

        variable_reports[target_col] = VariablePatternReport(
            target_column=target_col,
            n_total=n_rows,
            n_missing=n_mis,
            missing_rate=float(n_mis / n_rows),
            covariate_shifts=covariate_shifts,
            max_ks_statistic=float(max_ks),
            max_cohens_d=float(max_d),
            n_significant_shifts=sig_count,
            strongest_predictor=strongest_pred,
        )

    # Co-missingness correlation matrix
    co_missingness = None
    if len(missing_cols) > 1:
        mask_df = data[missing_cols].isna().astype(float)
        # Avoid zero division if any column has constant missingness
        stds = mask_df.std()
        valid_cols = stds[stds > 1e-12].index
        if len(valid_cols) > 1:
            co_missingness = mask_df[valid_cols].corr(method="pearson")

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
