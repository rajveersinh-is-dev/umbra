"""
Rubin's Multiple Imputation Pooling Engine and Barnard-Rubin (1999) Degrees of Freedom.

Combines estimates across M >= 5 multiple imputed datasets using Rubin's (1987) rules
and Barnard & Rubin's (1999) small-sample adjusted degrees of freedom.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator


@dataclass
class RubinsRulesResult:
    """Result of pooling multiple imputation estimates via Rubin's (1987) rules and Barnard-Rubin (1999)."""

    pooled_estimate: float
    within_variance: float
    between_variance: float
    total_variance: float
    standard_error: float
    df: float
    ci_lower: float
    ci_upper: float
    p_value: float = 0.0
    fmi: float = 0.0
    relative_variance: float = 0.0
    relative_efficiency: float = 1.0
    df_adjusted: Optional[float] = None
    df_complete: Optional[float] = None
    df_rubin: float = float("inf")

    @property
    def pooled_mean(self) -> float:
        return self.pooled_estimate

    @property
    def degrees_of_freedom(self) -> float:
        return self.df

    def __getitem__(self, item: str) -> float:
        val = getattr(self, item)
        return float(val)

    def to_dict(self) -> Dict[str, float]:
        return {
            "pooled_estimate": self.pooled_estimate,
            "within_variance": self.within_variance,
            "between_variance": self.between_variance,
            "total_variance": self.total_variance,
            "standard_error": self.standard_error,
            "df": self.df,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "p_value": self.p_value,
            "fmi": self.fmi,
            "relative_variance": self.relative_variance,
            "relative_efficiency": self.relative_efficiency,
            "df_adjusted": self.df_adjusted if self.df_adjusted is not None else self.df,
            "df_rubin": self.df_rubin,
        }


def rubins_rules(
    point_estimates: Union[List[float], np.ndarray],
    variance_estimates: Union[List[float], np.ndarray],
    df_complete: Optional[float] = None,
    alpha: float = 0.05,
) -> RubinsRulesResult:
    """Pool multiple imputation estimates using Rubin's (1987) Rules and Barnard-Rubin (1999).

    Parameters
    ----------
    point_estimates : Union[List[float], np.ndarray]
        Estimates Q_hat_m across M imputations.
    variance_estimates : Union[List[float], np.ndarray]
        Within-imputation variance estimates U_hat_m across M imputations.
    df_complete : Optional[float], default=None
        Complete-data degrees of freedom (nu_0 = N - k). When provided, applies
        Barnard & Rubin's (1999) small-sample adjustment.
    alpha : float, default=0.05
        Significance level for pooled confidence interval.

    Returns
    -------
    RubinsRulesResult
        Pooled estimate, within/between/total variance, standard error,
        degrees of freedom, confidence intervals, p-value, FMI, and relative efficiency.
    """
    pts = [float(x) for x in point_estimates]
    vars_ = [float(x) for x in variance_estimates]

    if len(pts) == 0 or len(vars_) == 0:
        raise ValueError("rubins_rules requires at least M=1 point and variance estimate.")
    if len(pts) != len(vars_):
        raise ValueError(
            f"Length mismatch: point_estimates has {len(pts)} items, "
            f"variance_estimates has {len(vars_)} items."
        )

    m = len(pts)
    if m == 1:
        warnings.warn(
            "rubins_rules called with M=1 imputation. Between-imputation variance is zero "
            "by construction (not because uncertainty is small). Use M>=5 for valid pooled inference.",
            UserWarning,
            stacklevel=2,
        )
        q_bar = pts[0]
        u_bar = vars_[0]
        t_var = u_bar
        se = float(np.sqrt(max(1e-12, t_var)))
        df_val = float(df_complete) if df_complete is not None and df_complete > 0 else float("inf")
        crit = (
            stats.t.ppf(1.0 - alpha / 2.0, df_val)
            if np.isfinite(df_val)
            else stats.norm.ppf(1.0 - alpha / 2.0)
        )
        t_stat = q_bar / se if se > 0 else 0.0
        p_val = (
            float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df_val)))
            if np.isfinite(df_val)
            else float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
        )

        return RubinsRulesResult(
            pooled_estimate=q_bar,
            within_variance=u_bar,
            between_variance=0.0,
            total_variance=t_var,
            standard_error=se,
            df=df_val,
            ci_lower=float(q_bar - crit * se),
            ci_upper=float(q_bar + crit * se),
            p_value=p_val,
            fmi=0.0,
            relative_variance=0.0,
            relative_efficiency=1.0,
            df_adjusted=df_val if df_complete is not None else None,
            df_complete=float(df_complete) if df_complete is not None else None,
            df_rubin=float("inf"),
        )

    q_bar = float(np.mean(pts))
    u_bar = float(np.mean(vars_))
    b_var = float(np.var(pts, ddof=1))
    t_var = float(u_bar + (1.0 + 1.0 / m) * b_var)

    # Relative increase in variance due to nonresponse
    r = float((1.0 + 1.0 / m) * b_var / max(1e-12, u_bar)) if b_var > 1e-12 else 0.0

    # Rubin (1987) large-sample degrees of freedom: nu_m = (m - 1) * (1 + 1/r)^2
    if r > 1e-12:
        df_rubin = float((m - 1) * (1.0 + 1.0 / r) ** 2)
    else:
        df_rubin = float("inf")

    # Barnard & Rubin (1999) small-sample adjustment
    df_adjusted: Optional[float] = None
    if df_complete is not None and df_complete > 0:
        nu_0 = float(df_complete)
        if b_var > 1e-12 and np.isfinite(df_rubin):
            # Fraction of missing information rate: gamma_hat = (1 + 1/m) * B / T = r / (1 + r)
            gamma_hat = float(r / (1.0 + r))
            # Observed-data degrees of freedom: nu_obs = ((nu_0 + 1) / (nu_0 + 3)) * nu_0 * (1 - gamma_hat)
            nu_obs = float(((nu_0 + 1.0) / (nu_0 + 3.0)) * nu_0 * (1.0 - gamma_hat))
            # Harmonic combination: nu_adj = (1/nu_m + 1/nu_obs)^-1
            nu_adj = float((df_rubin * nu_obs) / max(1e-12, (df_rubin + nu_obs)))
            # Bounded by complete-sample degrees of freedom
            df_val = float(np.clip(nu_adj, 1.0, nu_0))
            df_adjusted = df_val
        else:
            df_val = nu_0
            df_adjusted = nu_0
    else:
        df_val = df_rubin

    # Fraction of Missing Information (FMI)
    if r > 1e-12 and np.isfinite(df_val):
        fmi = float((r + 2.0 / (df_val + 3.0)) / (1.0 + r))
    else:
        fmi = 0.0
    fmi = float(np.clip(fmi, 0.0, 1.0))

    # Relative efficiency: RE = (1 + FMI / M)^-1
    rel_eff = float(1.0 / (1.0 + fmi / m))

    se = float(np.sqrt(max(1e-12, t_var)))
    crit = (
        float(stats.t.ppf(1.0 - alpha / 2.0, df_val))
        if np.isfinite(df_val)
        else float(stats.norm.ppf(1.0 - alpha / 2.0))
    )

    t_stat = q_bar / se if se > 0 else 0.0
    p_val = (
        float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df_val)))
        if np.isfinite(df_val)
        else float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
    )

    return RubinsRulesResult(
        pooled_estimate=q_bar,
        within_variance=u_bar,
        between_variance=b_var,
        total_variance=t_var,
        standard_error=se,
        df=df_val,
        ci_lower=float(q_bar - crit * se),
        ci_upper=float(q_bar + crit * se),
        p_value=p_val,
        fmi=fmi,
        relative_variance=r,
        relative_efficiency=rel_eff,
        df_adjusted=df_adjusted,
        df_complete=float(df_complete) if df_complete is not None else None,
        df_rubin=df_rubin,
    )


class RubinPooler(BaseEstimator):
    """Multiple Imputation Pooler via Rubin's Rules and Barnard-Rubin (1999) DOF.

    Fits downstream statistical models across M >= 5 multiple imputed datasets,
    pooling regression coefficients, variances, and hypothesis test statistics.

    Parameters
    ----------
    estimator : Optional[Any], default=None
        Scikit-learn regressor/classifier or statsmodels model class. If None,
        defaults to Ordinary Least Squares (OLS) regression with exact standard errors.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    df_complete : Optional[float], default=None
        Complete-data degrees of freedom. If None, automatically inferred as N - p - 1
        when fitting regression models.
    """

    def __init__(
        self,
        estimator: Optional[Any] = None,
        alpha: float = 0.05,
        df_complete: Optional[float] = None,
    ):
        self.estimator = estimator
        self.alpha = alpha
        self.df_complete = df_complete

        # Fitted attributes
        self.coef_: np.ndarray = np.empty((0,))
        self.intercept_: float = 0.0
        self.stderr_: np.ndarray = np.empty((0,))
        self.tvalues_: np.ndarray = np.empty((0,))
        self.pvalues_: np.ndarray = np.empty((0,))
        self.ci_: List[Tuple[float, float]] = []
        self.fmi_: np.ndarray = np.empty((0,))
        self.df_: np.ndarray = np.empty((0,))
        self.results_: Dict[str, RubinsRulesResult] = {}
        self.summary_: pd.DataFrame = pd.DataFrame()
        self.is_fitted_: bool = False

    def pool_estimates(
        self,
        point_estimates: Union[List[float], np.ndarray],
        variance_estimates: Union[List[float], np.ndarray],
        df_complete: Optional[float] = None,
        alpha: Optional[float] = None,
    ) -> Union[RubinsRulesResult, Dict[str, RubinsRulesResult]]:
        """Pool scalar or vector estimates across M multiple imputations.

        Parameters
        ----------
        point_estimates : Union[List[float], np.ndarray]
            Array of shape (M,) for a scalar parameter or (M, K) for K parameters.
        variance_estimates : Union[List[float], np.ndarray]
            Array of shape (M,) for a scalar parameter or (M, K) for K parameters.
        df_complete : Optional[float], default=None
            Complete data degrees of freedom.
        alpha : Optional[float], default=None
            Confidence alpha (defaults to self.alpha).

        Returns
        -------
        Union[RubinsRulesResult, Dict[str, RubinsRulesResult]]
            Pooled result for scalar parameter, or dictionary of results for vector.
        """
        eff_alpha = alpha if alpha is not None else self.alpha
        eff_df_comp = df_complete if df_complete is not None else self.df_complete

        pts_arr = np.asarray(point_estimates, dtype=float)
        vars_arr = np.asarray(variance_estimates, dtype=float)

        if pts_arr.ndim <= 1:
            return rubins_rules(
                point_estimates=pts_arr.tolist(),
                variance_estimates=vars_arr.tolist(),
                df_complete=eff_df_comp,
                alpha=eff_alpha,
            )

        # Multi-parameter vector: shape (M, K)
        m, k = pts_arr.shape
        pooled_dict: Dict[str, RubinsRulesResult] = {}
        for j in range(k):
            res_j = rubins_rules(
                point_estimates=pts_arr[:, j].tolist(),
                variance_estimates=vars_arr[:, j].tolist(),
                df_complete=eff_df_comp,
                alpha=eff_alpha,
            )
            pooled_dict[f"param_{j}"] = res_j

        return pooled_dict

    def fit(
        self,
        imputed_dfs: List[pd.DataFrame],
        target: str,
        feature_cols: Optional[List[str]] = None,
    ) -> "RubinPooler":
        """Fit model across M imputed datasets and pool parameter estimates.

        Parameters
        ----------
        imputed_dfs : List[pd.DataFrame]
            List of M complete imputed pandas DataFrames.
        target : str
            Name of target variable column Y.
        feature_cols : Optional[List[str]], default=None
            Names of predictor columns X. If None, uses all numeric columns except target.

        Returns
        -------
        self : RubinPooler
            Fitted pooler with pooled coefficients, standard errors, and summary table.
        """
        if not imputed_dfs:
            raise ValueError("imputed_dfs must contain at least M=1 DataFrame.")

        df_first = imputed_dfs[0]
        if target not in df_first.columns:
            raise ValueError(f"Target column '{target}' not found in imputed DataFrames.")

        if feature_cols is None:
            feature_cols = [
                c
                for c in df_first.columns
                if c != target and pd.api.types.is_numeric_dtype(df_first[c])
            ]

        if not feature_cols:
            raise ValueError("No numeric predictor feature columns found to fit model.")

        n = len(df_first)
        p = len(feature_cols)
        eff_df_complete = (
            self.df_complete if self.df_complete is not None else float(max(1, n - p - 1))
        )

        all_params: List[np.ndarray] = []
        all_variances: List[np.ndarray] = []
        param_names = ["const"] + list(feature_cols)

        for df in imputed_dfs:
            X_df = df[feature_cols].copy()
            y_arr = df[target].to_numpy(dtype=float)

            # Fit OLS model with constant intercept
            X_mat = np.column_stack([np.ones(len(df)), X_df.to_numpy(dtype=float)])

            # Analytical OLS estimation
            XtX = X_mat.T @ X_mat
            Q = np.linalg.pinv(XtX)
            beta = Q @ (X_mat.T @ y_arr)
            residuals = y_arr - (X_mat @ beta)
            s2 = float(np.sum(residuals**2) / max(1, n - len(beta)))
            cov_beta = s2 * Q
            var_beta = np.diag(cov_beta)

            all_params.append(beta)
            all_variances.append(var_beta)

        params_arr = np.array(all_params)  # (M, p + 1)
        vars_arr = np.array(all_variances)  # (M, p + 1)

        summary_rows = []
        pooled_coefs = []
        pooled_ses = []
        pooled_t = []
        pooled_p = []
        pooled_cis = []
        pooled_fmi = []
        pooled_df = []

        self.results_ = {}
        for j, name in enumerate(param_names):
            res_j = rubins_rules(
                point_estimates=params_arr[:, j].tolist(),
                variance_estimates=vars_arr[:, j].tolist(),
                df_complete=eff_df_complete,
                alpha=self.alpha,
            )
            self.results_[name] = res_j

            t_val = (
                res_j.pooled_estimate / res_j.standard_error if res_j.standard_error > 0 else 0.0
            )

            pooled_coefs.append(res_j.pooled_estimate)
            pooled_ses.append(res_j.standard_error)
            pooled_t.append(t_val)
            pooled_p.append(res_j.p_value)
            pooled_cis.append((res_j.ci_lower, res_j.ci_upper))
            pooled_fmi.append(res_j.fmi)
            pooled_df.append(res_j.df)

            summary_rows.append(
                {
                    "parameter": name,
                    "estimate": res_j.pooled_estimate,
                    "std_error": res_j.standard_error,
                    "t_statistic": t_val,
                    "p_value": res_j.p_value,
                    "ci_lower": res_j.ci_lower,
                    "ci_upper": res_j.ci_upper,
                    "df": res_j.df,
                    "fmi": res_j.fmi,
                    "relative_efficiency": res_j.relative_efficiency,
                }
            )

        self.intercept_ = float(pooled_coefs[0])
        self.coef_ = np.array(pooled_coefs[1:], dtype=float)
        self.stderr_ = np.array(pooled_ses, dtype=float)
        self.tvalues_ = np.array(pooled_t, dtype=float)
        self.pvalues_ = np.array(pooled_p, dtype=float)
        self.ci_ = pooled_cis
        self.fmi_ = np.array(pooled_fmi, dtype=float)
        self.df_ = np.array(pooled_df, dtype=float)
        self.summary_ = pd.DataFrame(summary_rows)
        self.is_fitted_ = True

        return self

    def summary(self) -> pd.DataFrame:
        """Return summary DataFrame of pooled estimates, standard errors, and diagnostics."""
        if not self.is_fitted_:
            raise ValueError("RubinPooler is not fitted yet. Call fit() first.")
        return self.summary_.copy()
