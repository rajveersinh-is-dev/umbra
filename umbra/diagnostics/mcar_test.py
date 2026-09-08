"""
Little's Missing Completely at Random (MCAR) Test.

Reference:
Little, R. J. A. (1988). A Test of Missing Completely at Random for
Multivariate Data with Missing Values. Journal of the American Statistical
Association, 83(404), 1198-1202.

CRITICAL METHODOLOGICAL FOUNDATION:
Little's test evaluates the null hypothesis H0: Data are Missing Completely at
Random (MCAR) against the alternative H1: Data are NOT MCAR.
- A statistically significant rejection (p < alpha) provides evidence that the
  missingness mechanism is either MAR or MNAR.
- A failure to reject indicates the observed patterns are compatible with MCAR
  under multivariate normality assumptions.
- CRUCIALLY, Little's test CANNOT distinguish between MAR and MNAR.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class LittleMCARResult:
    """Results from Little's (1988) MCAR test.

    Attributes
    ----------
    statistic : float
        Chi-squared test statistic (d^2 = sum_j N_j * d_j^2).
    p_value : float
        P-value computed from chi-squared distribution with degrees_of_freedom.
    degrees_of_freedom : int
        Degrees of freedom: sum_{j=1}^J p_j - p.
    n_patterns : int
        Number of distinct missingness patterns observed.
    n_samples : int
        Number of rows (observations).
    n_features : int
        Number of evaluated numeric features.
    is_rejected : bool
        Whether H0 (MCAR) is rejected at the specified significance level alpha.
    alpha : float
        Significance level used for rejection decision.
    pattern_details : List[Dict[str, Any]]
        Diagnostic metrics for each distinct missingness pattern.
    note : str
        Methodological interpretation and caveats.
    """

    statistic: float
    p_value: float
    degrees_of_freedom: int
    n_patterns: int
    n_samples: int
    n_features: int
    is_rejected: bool
    alpha: float = 0.05
    pattern_details: List[Dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def summary(self) -> str:
        verdict = (
            "REJECT MCAR (Data exhibits systematic departures consistent with MAR or MNAR)"
            if self.is_rejected
            else "FAIL TO REJECT MCAR (Observed patterns are statistically compatible with MCAR)"
        )
        return (
            f"Little's MCAR Test (Little, 1988):\n"
            f"  - Chi-squared Statistic : {self.statistic:.4f}\n"
            f"  - Degrees of Freedom    : {self.degrees_of_freedom}\n"
            f"  - p-value               : {self.p_value:.4e}\n"
            f"  - Significance Level    : {self.alpha}\n"
            f"  - Test Verdict          : {verdict}\n"
            f"  - Distinct Patterns     : {self.n_patterns}\n"
            f"  - Methodological Note   : {self.note}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "degrees_of_freedom": int(self.degrees_of_freedom),
            "n_patterns": int(self.n_patterns),
            "n_samples": int(self.n_samples),
            "n_features": int(self.n_features),
            "is_rejected": bool(self.is_rejected),
            "alpha": float(self.alpha),
            "verdict": "REJECT_MCAR" if self.is_rejected else "FAIL_TO_REJECT_MCAR",
            "pattern_details": self.pattern_details,
            "note": self.note,
        }


def _em_multivariate_normal(
    X: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-5,
    ridge_reg: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expectation-Maximization algorithm for maximum likelihood estimation of
    mean vector and covariance matrix under multivariate normality with missing data.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Data matrix with potential np.nan values.
    max_iter : int, default=200
        Maximum EM iterations.
    tol : float, default=1e-5
        Relative convergence tolerance for parameters.
    ridge_reg : float, default=1e-4
        Tikhonov regularization added to covariance diagonals for numerical stability.

    Returns
    -------
    mu : np.ndarray of shape (p,)
        Estimated MLE mean vector.
    sigma : np.ndarray of shape (p, p)
        Estimated MLE covariance matrix.
    """
    n, p = X.shape
    nan_mask = np.isnan(X)

    # Initial parameter estimates from pairwise/columnwise available values
    mu = np.zeros(p, dtype=float)
    for j in range(p):
        col_vals = X[:, j][~nan_mask[:, j]]
        mu[j] = np.mean(col_vals) if len(col_vals) > 0 else 0.0

    # Initial covariance
    X_centered = np.zeros_like(X)
    for j in range(p):
        X_centered[:, j] = np.where(nan_mask[:, j], 0.0, X[:, j] - mu[j])

    # Sample covariance with regularizer
    sigma = (X_centered.T @ X_centered) / max(1, n - 1) + ridge_reg * np.eye(p)

    # Identify distinct missingness patterns once before EM iterations
    pattern_tuples = [tuple(row) for row in nan_mask]
    unique_patterns, pattern_inverse, pattern_counts = np.unique(
        pattern_tuples, axis=0, return_inverse=True, return_counts=True
    )
    pattern_row_indices = [np.where(pattern_inverse == j)[0] for j in range(len(unique_patterns))]

    for _ in range(max_iter):
        mu_prev = mu.copy()
        sigma_prev = sigma.copy()

        sum_x = np.zeros(p, dtype=float)
        sum_xx = np.zeros((p, p), dtype=float)

        for j, pat in enumerate(unique_patterns):
            row_idx = pattern_row_indices[j]
            n_j = len(row_idx)
            obs_idx = np.where(~pat)[0]
            mis_idx = np.where(pat)[0]

            if len(mis_idx) == 0:
                # Fully observed rows
                X_obs_rows = X[row_idx]
                sum_x += np.sum(X_obs_rows, axis=0)
                sum_xx += X_obs_rows.T @ X_obs_rows
            elif len(obs_idx) == 0:
                # Fully missing rows: conditional expectation is prior mean & covariance
                sum_x += n_j * mu
                sum_xx += n_j * (np.outer(mu, mu) + sigma)
            else:
                X_obs_rows = X[row_idx][:, obs_idx]  # (n_j, len(obs_idx))
                mu_obs = mu[obs_idx]
                mu_mis = mu[mis_idx]

                sigma_obs_obs = sigma[np.ix_(obs_idx, obs_idx)] + ridge_reg * np.eye(len(obs_idx))
                sigma_mis_obs = sigma[np.ix_(mis_idx, obs_idx)]
                sigma_mis_mis = sigma[np.ix_(mis_idx, mis_idx)]

                # Regression coefficients: Sigma_mis_obs @ inv(Sigma_obs_obs)
                try:
                    beta = np.linalg.solve(sigma_obs_obs, sigma_mis_obs.T).T
                except np.linalg.LinAlgError:
                    beta = sigma_mis_obs @ np.linalg.pinv(sigma_obs_obs)

                # Conditional mean: E[X_mis | X_obs] for all rows in pattern
                diff_obs = X_obs_rows - mu_obs  # (n_j, len(obs_idx))
                cond_mu_mis = mu_mis + diff_obs @ beta.T  # (n_j, len(mis_idx))

                # Conditional covariance: Var(X_mis | X_obs) (same for all rows in pattern)
                cond_cov_mis = sigma_mis_mis - beta @ sigma_mis_obs.T
                cond_cov_mis = 0.5 * (cond_cov_mis + cond_cov_mis.T)
                # Ensure positive semi-definiteness
                eigenvals, eigenvecs = np.linalg.eigh(cond_cov_mis)
                eigenvals = np.maximum(eigenvals, 1e-8)
                cond_cov_mis = (eigenvecs * eigenvals) @ eigenvecs.T

                # Completed matrix for pattern
                X_comp = np.zeros((n_j, p), dtype=float)
                X_comp[:, obs_idx] = X_obs_rows
                X_comp[:, mis_idx] = cond_mu_mis

                sum_x += np.sum(X_comp, axis=0)
                sum_xx += X_comp.T @ X_comp
                sum_xx[np.ix_(mis_idx, mis_idx)] += n_j * cond_cov_mis

        # M-step: Update parameter estimates
        mu = sum_x / n
        sigma = (sum_xx / n) - np.outer(mu, mu)
        sigma = 0.5 * (sigma + sigma.T) + ridge_reg * np.eye(p)

        # Check convergence
        denom_mu = np.max(np.abs(mu_prev)) + 1e-8
        denom_sig = np.max(np.abs(sigma_prev)) + 1e-8
        mu_diff = np.max(np.abs(mu - mu_prev)) / denom_mu
        sigma_diff = np.max(np.abs(sigma - sigma_prev)) / denom_sig

        if max(mu_diff, sigma_diff) < tol:
            break

    return mu, sigma


def littles_mcar_test(
    data: Union[pd.DataFrame, np.ndarray],
    alpha: float = 0.05,
    ridge_reg: float = 1e-4,
) -> LittleMCARResult:
    """Perform Little's (1988) test of Missing Completely at Random (MCAR).

    Mathematical Formulation:
    -------------------------
    Under H0 (MCAR), all missingness patterns share the same underlying population
    mean vector mu and covariance matrix Sigma.
    For each distinct missingness pattern j with N_j observations, let:
      - y_bar_{obs, j}: observed sample mean vector of dimension p_j
      - mu_{obs, j}   : corresponding subvector of MLE mu
      - Sigma_{obs, j}: corresponding submatrix of MLE Sigma

    The test statistic is the sum of squared Mahalanobis distances:
      d^2 = sum_{j=1}^J N_j * (y_bar_{obs, j} - mu_{obs, j})' *
                              [Sigma_{obs, j}]^{-1} *
                              (y_bar_{obs, j} - mu_{obs, j})

    Under H0 and multivariate normality:
      d^2 ~ Chi-squared(df)
      df = sum_{j=1}^J p_j - p

    where:
      - J is the number of distinct patterns
      - p_j is the number of observed variables in pattern j
      - p is the total number of variables evaluated

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Dataset with numeric columns and missing values. Non-numeric columns
        are automatically filtered out if a DataFrame is passed.
    alpha : float, default=0.05
        Significance level for the hypothesis test.
    ridge_reg : float, default=1e-4
        Regularization added to covariance diagonals for numerical conditioning.

    Returns
    -------
    LittleMCARResult
        Comprehensive test result with test statistic, p-value, df, and pattern details.

    References
    ----------
    Little, R. J. A. (1988). A test of missing completely at random for
    multivariate data with missing values. JASA, 83(404), 1198-1202.
    """
    if isinstance(data, pd.DataFrame):
        numeric_df = data.select_dtypes(include=[np.number])
        if numeric_df.shape[1] == 0:
            raise ValueError("Input data contains no numeric columns for Little's MCAR test.")
        col_names = list(numeric_df.columns)
        X = numeric_df.to_numpy(dtype=float, copy=True)
    else:
        X = np.asarray(data, dtype=float).copy()
        col_names = [f"col_{i}" for i in range(X.shape[1])]

    n, p = X.shape
    if n < 2 or p < 1:
        raise ValueError(f"Insufficient data dimensions for Little's test: shape=({n}, {p}).")

    nan_mask = np.isnan(X)

    # Edge case 1: No missing values
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
            pattern_details=[
                {
                    "pattern_id": 0,
                    "count": n,
                    "observed_features": col_names,
                    "d2_contribution": 0.0,
                }
            ],
            note="No missing values present. Data are trivially complete (MCAR holds).",
        )

    # Edge case 2: All values missing
    if np.all(nan_mask):
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
            note="All entries are missing. Cannot compute test statistic.",
        )

    # Variables with at least some observed entries
    col_observed_counts = np.sum(~nan_mask, axis=0)
    if np.any(col_observed_counts == 0):
        # Drop columns that are 100% missing
        valid_cols = np.where(col_observed_counts > 0)[0]
        if len(valid_cols) == 0:
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
                note="No columns have observed values.",
            )
        X = X[:, valid_cols]
        col_names = [col_names[c] for c in valid_cols]
        nan_mask = np.isnan(X)
        n, p = X.shape

    # Estimate global MLE mean and covariance via EM
    mu_mle, sigma_mle = _em_multivariate_normal(X, ridge_reg=ridge_reg)

    # Identify distinct missingness patterns
    pattern_tuples = [tuple(row) for row in nan_mask]
    unique_patterns, pattern_inverse, pattern_counts = np.unique(
        pattern_tuples, axis=0, return_inverse=True, return_counts=True
    )

    n_patterns = len(unique_patterns)
    d2_total = 0.0
    sum_pj = 0
    pattern_details = []

    for j, (pat, count) in enumerate(zip(unique_patterns, pattern_counts)):
        obs_cols = np.where(~pat)[0]
        p_j = len(obs_cols)

        if p_j == 0:
            # Fully missing row does not contribute to observed Mahalanobis distance
            continue

        sum_pj += p_j

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

        pattern_details.append(
            {
                "pattern_id": j,
                "count": int(count),
                "n_observed_features": int(p_j),
                "observed_features": [col_names[c] for c in obs_cols],
                "d2_contribution": float(count * d2_j),
            }
        )

    # Exact Little (1988) degrees of freedom: df = sum(p_j) - p
    df = sum_pj - p

    if df <= 0:
        # If df <= 0, no overidentifying restrictions exist to test MCAR
        return LittleMCARResult(
            statistic=float(d2_total),
            p_value=1.0,
            degrees_of_freedom=0,
            n_patterns=int(n_patterns),
            n_samples=int(n),
            n_features=int(p),
            is_rejected=False,
            alpha=alpha,
            pattern_details=pattern_details,
            note="Degrees of freedom <= 0. Not enough distinct patterns to perform test.",
        )

    p_val = float(stats.chi2.sf(d2_total, df))
    is_rejected = bool(p_val < alpha)

    note = (
        "Little's test evaluates MCAR vs. Not-MCAR. Rejection indicates the data are NOT MCAR "
        "(i.e., evidence consistent with MAR or MNAR). It CANNOT distinguish between MAR and MNAR."
    )

    return LittleMCARResult(
        statistic=float(d2_total),
        p_value=float(p_val),
        degrees_of_freedom=int(df),
        n_patterns=int(n_patterns),
        n_samples=int(n),
        n_features=int(p),
        is_rejected=is_rejected,
        alpha=float(alpha),
        pattern_details=pattern_details,
        note=note,
    )
