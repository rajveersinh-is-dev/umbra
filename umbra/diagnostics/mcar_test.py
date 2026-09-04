"""
Little's Missing Completely at Random (MCAR) Test.

Reference:
Little, R. J. A. (1988). A Test of Missing Completely at Random for
Multivariate Data with Missing Values. Journal of the American Statistical
Association, 83(404), 1198-1202.

IMPORTANT METHODOLOGICAL NOTE:
Little's test evaluates the null hypothesis that data is MCAR against the
alternative that it is NOT MCAR. A rejection indicates that data is either
MAR or MNAR. Crucially, Little's test CANNOT distinguish MAR from MNAR.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class LittleMCARResult:
    """Results from Little's MCAR test."""

    statistic: float
    p_value: float
    degrees_of_freedom: int
    n_patterns: int
    n_samples: int
    n_features: int
    is_rejected: bool
    alpha: float
    pattern_details: List[Dict[str, Any]]
    note: str

    def summary(self) -> str:
        verdict = (
            "REJECT MCAR (Data is likely MAR or MNAR)"
            if self.is_rejected
            else "FAIL TO REJECT MCAR (Data is consistent with MCAR)"
        )
        return (
            f"Little's MCAR Test Summary:\n"
            f"  - Chi-squared Statistic : {self.statistic:.4f}\n"
            f"  - Degrees of Freedom    : {self.degrees_of_freedom}\n"
            f"  - p-value               : {self.p_value:.4e}\n"
            f"  - Significance (alpha)  : {self.alpha}\n"
            f"  - Verdict               : {verdict}\n"
            f"  - Distinct Patterns     : {self.n_patterns}\n"
            f"  - Note                  : {self.note}"
        )


def _em_multivariate_normal(
    X: np.ndarray,
    max_iter: int = 150,
    tol: float = 1e-5,
    ridge_reg: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expectation-Maximization algorithm for estimating mean and covariance of
    multivariate normal data with missing values.
    """
    n, p = X.shape
    nan_mask = np.isnan(X)

    # Initial parameter estimates from column-wise available values
    mu = np.nanmean(X, axis=0)
    # Fill any completely NaN columns with 0
    mu = np.nan_to_num(mu, nan=0.0)

    # Initial covariance using pairwise complete obs, regularized
    X_zero = np.nan_to_num(X - mu, nan=0.0)
    sigma = (X_zero.T @ X_zero) / n + ridge_reg * np.eye(p)

    for iteration in range(max_iter):
        mu_prev = mu.copy()
        sigma_prev = sigma.copy()

        # E-step
        sum_x = np.zeros(p)
        sum_xx = np.zeros((p, p))

        for i in range(n):
            obs_idx = np.where(~nan_mask[i])[0]
            mis_idx = np.where(nan_mask[i])[0]

            if len(mis_idx) == 0:
                # Fully observed
                x_i = X[i]
                sum_x += x_i
                sum_xx += np.outer(x_i, x_i)
            elif len(obs_idx) == 0:
                # Fully missing row
                sum_x += mu
                sum_xx += np.outer(mu, mu) + sigma
            else:
                x_obs = X[i, obs_idx]
                mu_obs = mu[obs_idx]
                mu_mis = mu[mis_idx]

                sigma_obs_obs = sigma[np.ix_(obs_idx, obs_idx)] + ridge_reg * np.eye(len(obs_idx))
                sigma_mis_obs = sigma[np.ix_(mis_idx, obs_idx)]
                sigma_mis_mis = sigma[np.ix_(mis_idx, mis_idx)]

                # Regression coefficients for conditional mean: Sigma_mis_obs @ inv(Sigma_obs_obs)
                try:
                    beta = np.linalg.solve(sigma_obs_obs, sigma_mis_obs.T).T
                except np.linalg.LinAlgError:
                    beta = sigma_mis_obs @ np.linalg.pinv(sigma_obs_obs)

                # Conditional mean and covariance
                cond_mu_mis = mu_mis + beta @ (x_obs - mu_obs)
                cond_cov_mis = sigma_mis_mis - beta @ sigma_mis_obs.T
                # Ensure positive semi-definite
                cond_cov_mis = 0.5 * (cond_cov_mis + cond_cov_mis.T) + ridge_reg * np.eye(
                    len(mis_idx)
                )

                # Reconstruct full completed vector expectation
                x_comp = np.zeros(p)
                x_comp[obs_idx] = x_obs
                x_comp[mis_idx] = cond_mu_mis
                sum_x += x_comp

                # Outer product + conditional covariance
                xx_comp = np.outer(x_comp, x_comp)
                xx_comp[np.ix_(mis_idx, mis_idx)] += cond_cov_mis
                sum_xx += xx_comp

        # M-step
        mu = sum_x / n
        sigma = (sum_xx / n) - np.outer(mu, mu)
        sigma = 0.5 * (sigma + sigma.T) + ridge_reg * np.eye(p)

        # Check convergence
        mu_diff = np.max(np.abs(mu - mu_prev)) / (np.max(np.abs(mu_prev)) + 1e-8)
        sigma_diff = np.max(np.abs(sigma - sigma_prev)) / (np.max(np.abs(sigma_prev)) + 1e-8)

        if max(mu_diff, sigma_diff) < tol:
            break

    return mu, sigma


def littles_mcar_test(
    data: Union[pd.DataFrame, np.ndarray],
    alpha: float = 0.05,
    ridge_reg: float = 1e-4,
) -> LittleMCARResult:
    """
    Perform Little's (1988) MCAR test on missing multivariate data.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Data with potential NaN values. Continuous or numerical columns only.
    alpha : float, default=0.05
        Significance level for hypothesis test.
    ridge_reg : float, default=1e-4
        Regularization added to covariance matrices to ensure invertibility.

    Returns
    -------
    LittleMCARResult
        Object containing test statistic, p-value, degrees of freedom, and explanation.
    """
    if isinstance(data, pd.DataFrame):
        col_names = list(data.columns)
        # Select numeric columns only
        numeric_df = data.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < data.shape[1]:
            col_names = list(numeric_df.columns)
        X = numeric_df.to_numpy(dtype=float, copy=True)
    else:
        X = np.asarray(data, dtype=float).copy()
        col_names = [f"col_{i}" for i in range(X.shape[1])]

    n, p = X.shape
    nan_mask = np.isnan(X)

    if not np.any(nan_mask):
        return LittleMCARResult(
            statistic=0.0,
            p_value=1.0,
            degrees_of_freedom=0,
            n_patterns=1,
            n_samples=n,
            n_features=p,
            is_rejected=False,
            alpha=alpha,
            pattern_details=[],
            note="No missing values present in the data. Trivially MCAR.",
        )

    # Variables that have at least one missing value
    vars_with_missing = np.where(nan_mask.any(axis=0))[0]
    p_missing = len(vars_with_missing)

    # Estimate global MLE mean and covariance via EM
    mu_mle, sigma_mle = _em_multivariate_normal(X, ridge_reg=ridge_reg)

    # Identify distinct missingness patterns
    pattern_tuples = [tuple(row) for row in nan_mask]
    unique_patterns, pattern_inverse, pattern_counts = np.unique(
        pattern_tuples, axis=0, return_inverse=True, return_counts=True
    )

    n_patterns = len(unique_patterns)
    d2_total = 0.0
    df_total = 0
    pattern_details = []

    for j, (pat, count) in enumerate(zip(unique_patterns, pattern_counts)):
        obs_cols = np.where(~pat)[0]
        p_j = len(obs_cols)

        if p_j == 0:
            # Entire row is missing
            continue

        # Rows matching pattern j
        row_indices = np.where(pattern_inverse == j)[0]
        y_j_obs = X[row_indices][:, obs_cols]
        y_bar_j = np.mean(y_j_obs, axis=0)

        mu_j = mu_mle[obs_cols]
        sigma_j = sigma_mle[np.ix_(obs_cols, obs_cols)] + ridge_reg * np.eye(p_j)

        diff = y_bar_j - mu_j
        try:
            inv_sigma_j = np.linalg.inv(sigma_j)
            d2_j = float(diff.T @ inv_sigma_j @ diff)
        except np.linalg.LinAlgError:
            inv_sigma_j = np.linalg.pinv(sigma_j)
            d2_j = float(diff.T @ inv_sigma_j @ diff)

        d2_total += count * d2_j

        # If pattern is not fully observed, it contributes to degrees of freedom
        if p_j < p:
            df_total += p_j

        pattern_details.append(
            {
                "pattern_id": j,
                "count": int(count),
                "observed_features": [col_names[c] for c in obs_cols],
                "d2_contribution": float(count * d2_j),
            }
        )

    # Little's degrees of freedom formula: sum(p_j) - p_missing
    df_total = max(1, df_total - p_missing)
    p_val = float(stats.chi2.sf(d2_total, df_total))
    is_rejected = bool(p_val < alpha)

    note = (
        "Little's test tests MCAR vs. Not-MCAR. Rejection indicates the data is NOT MCAR "
        "(i.e., it is MAR or MNAR). It CANNOT distinguish between MAR and MNAR."
    )

    return LittleMCARResult(
        statistic=float(d2_total),
        p_value=p_val,
        degrees_of_freedom=int(df_total),
        n_patterns=int(n_patterns),
        n_samples=int(n),
        n_features=int(p),
        is_rejected=is_rejected,
        alpha=alpha,
        pattern_details=pattern_details,
        note=note,
    )
