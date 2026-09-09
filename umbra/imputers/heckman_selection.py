"""
Heckman Selection Model Imputer for MNAR Data.

Implements James Heckman's (1976, 1979) two-step selection estimator and Full-Information
Maximum Likelihood (FIML) joint estimation adapted for missing data imputation and
sensitivity bounds under MNAR.

Model:
  Selection Equation: z_i* = W_i * gamma + u_i,  R_i = 1(z_i* > 0)
  Outcome Equation:   Y_i  = X_i * beta  + eps_i, observed only when R_i = 1
  Errors: (u_i, eps_i) ~ Bivariate Normal with correlation rho, Var(u_i) = 1, Var(eps_i) = sigma^2.

Imputation for unobserved cases (R_i = 0):
  E[Y_i | R_i = 0, X_i, W_i] = X_i * beta + beta_lambda * lambda_0(W_i * gamma)
  where lambda_0(eta) = -phi(eta) / (1 - Phi(eta)) = -phi(-eta) / Phi(-eta) <= 0.

Standard Error Corrections:
  - Murphy & Topel (1985) exact analytical asymptotic covariance matrix for two-step.
  - Full-Information Maximum Likelihood (FIML) joint log-likelihood optimization
    with observed Hessian inversion and Delta-method standard errors.
  - Paired nonparametric bootstrap across both stages as non-parametric alternative.

References:
Heckman, J. J. (1976). The common structure of statistical models of
truncation, sample selection and limited dependent variables.
Annals of Economic and Social Measurement, 5(4), 475-492.
Heckman, J. J. (1979). Sample selection bias as a specification error.
Econometrica, 47(1), 153-161.
Murphy, K. M., & Topel, R. H. (1985). Estimation and inference with two-step
econometric estimators. Journal of Business & Economic Statistics, 3(4), 370-379.
"""

import warnings
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, special, stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from statsmodels.tools.numdiff import approx_hess

__all__ = [
    "HeckmanSelectionImputer",
    "WeakInstrumentWarning",
    "HeckmanSEWarning",
    "HeckmanConvergenceWarning",
    "HeckmanCollinearityWarning",
]


class WeakInstrumentWarning(UserWarning):
    """Warning emitted when a candidate auxiliary instrument fails the Stock-Yogo relevance test (F <= 10)."""

    pass


class HeckmanSEWarning(UserWarning):
    """Warning emitted when Heckman standard errors cannot be reliably computed (e.g. singular design matrix, Ridge fallback) or uncorrected naive OLS is requested."""

    pass


class HeckmanConvergenceWarning(UserWarning):
    """Warning emitted when Heckman FIML estimation fails to converge or produces a non-positive-definite Hessian, falling back to two-step."""

    pass


class HeckmanCollinearityWarning(UserWarning):
    """Warning emitted when the Heckman second-stage design matrix is ill-conditioned (condition index > 30 or VIF > 10)."""

    pass


def _bootstrap_heckman_se(
    W_mat: np.ndarray,
    X_mat_obs: np.ndarray,
    y_obs: np.ndarray,
    R_obs: np.ndarray,
    n_boot: int = 200,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Paired bootstrap standard error for Heckman two-step estimator.

    Both stages are re-estimated on each bootstrap resample, correctly
    propagating first-stage estimation uncertainty into second-stage SEs.
    This implements the nonparametric bootstrap as recommended by
    Cameron & Trivedi (2005) Section 24.5 as a computationally tractable
    alternative to Murphy-Topel (1985) analytical correction.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    assert len(W_mat) == len(X_mat_obs) == len(y_obs) == len(R_obs), (
        "All inputs to _bootstrap_heckman_se must have the same number of rows (N total)"
    )
    assert not np.any(np.isnan(y_obs[R_obs == 1])), (
        "y_obs must not have NaN at positions where R_obs == 1"
    )

    n_total = len(W_mat)
    k_params = X_mat_obs.shape[1] + 1  # X coefficients (including constant) + lambda

    obs_indices = np.where(R_obs == 1)[0]
    if len(X_mat_obs) == n_total:
        X_full = X_mat_obs
        y_full = y_obs
    else:
        X_full = np.zeros((n_total, X_mat_obs.shape[1]), dtype=float)
        X_full[obs_indices] = X_mat_obs
        y_full = np.full(n_total, np.nan, dtype=float)
        y_full[obs_indices] = y_obs

    boot_params: List[np.ndarray] = []
    max_attempts = n_boot * 3
    attempts = 0

    while len(boot_params) < n_boot and attempts < max_attempts:
        attempts += 1
        boot_idx = rng.choice(n_total, size=n_total, replace=True)
        R_b = R_obs[boot_idx]

        n_obs_b = int(np.sum(R_b == 1))
        if n_obs_b < k_params + 5 or n_obs_b > n_total - 5:
            continue

        W_b = W_mat[boot_idx]

        # Step 1: Selection equation on resample
        try:
            probit_mod = sm.Probit(R_b, W_b)
            probit_res = probit_mod.fit(disp=0, maxiter=35)
            gamma_b = probit_res.params
        except Exception:
            try:
                logit_mod = sm.Logit(R_b, W_b)
                logit_res = logit_mod.fit(disp=0, maxiter=35)
                gamma_b = logit_res.params / 1.6
            except Exception:
                try:
                    ols_lpm = sm.OLS(R_b, W_b).fit()
                    gamma_b = ols_lpm.params * 2.5
                except Exception:
                    continue

        obs_mask_b = R_b == 1
        eta_b = W_b @ gamma_b
        lambda_1_b = _compute_imr_observed(eta_b[obs_mask_b])

        # Step 2: Outcome regression on resample observed cases
        X_obs_b = X_full[boot_idx][obs_mask_b]
        y_obs_b = y_full[boot_idx][obs_mask_b]

        design_b = np.column_stack([X_obs_b, lambda_1_b])

        try:
            ols_b = sm.OLS(y_obs_b, design_b).fit()
            if len(ols_b.params) == k_params and not np.any(np.isnan(ols_b.params)):
                boot_params.append(ols_b.params)
        except Exception:
            continue

    if len(boot_params) >= 10:
        se = np.std(boot_params, axis=0, ddof=1)
        return np.asarray(se, dtype=float)
    else:
        warnings.warn(
            f"Bootstrap standard error estimation completed with only {len(boot_params)}/{n_boot} successful draws. Returning NaN standard errors.",
            HeckmanSEWarning,
            stacklevel=2,
        )
        return np.full(k_params, np.nan, dtype=float)


def _compute_imr_observed(eta: np.ndarray) -> np.ndarray:
    """Compute Inverse Mills Ratio for observed cases (R=1):
    lambda_1(eta) = phi(eta) / Phi(eta) >= 0.
    """
    eta = np.clip(eta, -30.0, 30.0)
    phi = stats.norm.pdf(eta)
    Phi = stats.norm.cdf(eta)
    safe_Phi = np.where(Phi < 1e-12, 1e-12, Phi)
    imr = phi / safe_Phi

    # Tail correction for eta in [-30, -10]: direct phi/Phi ratio is numerically unstable.
    extreme_neg = eta < -10.0
    if np.any(extreme_neg):
        imr[extreme_neg] = np.exp(
            stats.norm.logpdf(eta[extreme_neg]) - stats.norm.logcdf(eta[extreme_neg])
        )
    return np.asarray(imr, dtype=float)


def _compute_imr_missing(eta: np.ndarray) -> np.ndarray:
    """Compute Inverse Mills Ratio for missing cases (R=0):
    lambda_0(eta) = -phi(eta) / (1 - Phi(eta)) = -phi(-eta) / Phi(-eta) <= 0.
    """
    eta = np.clip(eta, -30.0, 30.0)
    neg_eta = -eta
    phi_neg = stats.norm.pdf(neg_eta)
    Phi_neg = stats.norm.cdf(neg_eta)
    safe_Phi_neg = np.where(Phi_neg < 1e-12, 1e-12, Phi_neg)
    imr_0 = -(phi_neg / safe_Phi_neg)

    # Tail correction for neg_eta in [-30, -10]: direct phi/Phi ratio is numerically unstable.
    extreme_neg = neg_eta < -10.0
    if np.any(extreme_neg):
        imr_0[extreme_neg] = -np.exp(
            stats.norm.logpdf(neg_eta[extreme_neg]) - stats.norm.logcdf(neg_eta[extreme_neg])
        )
    return np.asarray(imr_0, dtype=float)


class HeckmanSelectionImputer(BaseEstimator, TransformerMixin):
    """Heckman selection model imputer for MNAR tabular data.

    Supports both Heckman (1979) two-step estimation with exact Murphy & Topel (1985)
    asymptotic covariance correction and Full-Information Maximum Likelihood (FIML)
    joint bivariate normal estimation.

    Parameters
    ----------
    target_cols : Optional[List[str]], default=None
        Columns to impute with Heckman selection model. If None, targets all
        columns with missing values.
    shadow_cols : Optional[Dict[str, str]], default=None
        Mapping of {target_col: shadow_var_col}. Shadow variables enter the
        selection equation W but are EXCLUDED from the outcome equation X
        (classic exclusion restriction / instrument).
    method : Literal["two-step", "fiml"], default="two-step"
        Estimation method for the Heckman model:
        - 'two-step': Heckman (1979) two-step estimator with Murphy-Topel (1985)
          asymptotic covariance matrix correction.
        - 'fiml': Full-Information Maximum Likelihood joint bivariate normal
          estimation initialized from two-step estimates, with automatic fallback
          to two-step on non-convergence.
    se_method : Optional[Literal["murphy-topel", "bootstrap", "naive"]], default=None
        Standard error computation method for two-step:
        - None: defaults to 'bootstrap' if n_bootstrap_se > 0, else 'murphy-topel'.
        - 'murphy-topel': Exact analytical Murphy & Topel (1985) asymptotic covariance.
        - 'bootstrap': Nonparametric paired bootstrap across both stages.
        - 'naive': Uncorrected OLS standard errors (emits HeckmanSEWarning).
    ridge_alpha : float, default=1.0
        Regularization penalty for Ridge regression fallback when the second-stage
        design matrix is ill-conditioned or singular.
    ridge_se_method : Literal["sandwich", "nan"], default="sandwich"
        Method for standard errors when Ridge fallback is triggered:
        - 'sandwich': Analytical regularized sandwich covariance matrix
          V_ridge = sigma^2 Q_alpha M Q_alpha, preserving first-stage uncertainty.
        - 'nan': Legacy mode setting standard errors to NaNs.
    stochastic : bool, default=False
        If True, draws normal noise with conditional variance Var(Y | R=0).
    n_imputations : int, default=1
        Number of stochastic imputation draws (if > 1, stochastic is enabled).
    n_bootstrap_se : int, default=0
        Number of bootstrap iterations. If > 0 and se_method is None, enables paired
        bootstrap standard error estimation.
    random_state : Optional[int], default=42
        Reproducibility seed.
    """

    def __init__(
        self,
        target_cols: Optional[List[str]] = None,
        shadow_cols: Optional[Dict[str, str]] = None,
        method: Literal["two-step", "fiml"] = "two-step",
        se_method: Optional[Literal["murphy-topel", "bootstrap", "naive"]] = None,
        ridge_alpha: float = 1.0,
        ridge_se_method: Literal["sandwich", "nan"] = "sandwich",
        stochastic: bool = False,
        n_imputations: int = 1,
        n_bootstrap_se: int = 0,
        random_state: Optional[int] = 42,
        **kwargs: Any,
    ):
        if "n_draws" in kwargs:
            warnings.warn(
                "n_draws is deprecated; use n_imputations instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            n_imputations = kwargs.pop("n_draws")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        self.target_cols = target_cols
        self.shadow_cols = shadow_cols
        self.method = method
        self.se_method = se_method
        self.ridge_alpha = ridge_alpha
        self.ridge_se_method = ridge_se_method
        self.stochastic = stochastic
        self.n_imputations = n_imputations
        self.n_bootstrap_se = n_bootstrap_se
        self.random_state = random_state

        # Fitted attributes
        self.models_: Dict[str, Dict[str, Any]] = {}
        self.col_medians_: Dict[str, float] = {}
        self.feature_names_in_: Union[List[str], np.ndarray] = []
        self.n_features_in_: int = 0

        # Primary fitted attributes (accessible directly on self)
        self.method_used_: Literal["two-step", "fiml"] = "two-step"
        self.coef_cov_: np.ndarray = np.empty((0, 0))
        self.coef_stderr_: np.ndarray = np.empty((0,))
        self.mills_stderr_: float = np.nan
        self.sigma_: float = np.nan
        self.rho_: float = np.nan
        self.log_likelihood_: Optional[float] = None
        self.collinearity_diagnostics_: Dict[str, Any] = {}

    def _compute_murphy_topel_covariance(
        self,
        X_star: np.ndarray,
        W_obs: np.ndarray,
        y_obs: np.ndarray,
        beta_star: np.ndarray,
        V1: np.ndarray,
        delta: np.ndarray,
    ) -> np.ndarray:
        """Compute exact analytical Murphy & Topel (1985) asymptotic covariance matrix.

        Formula:
          V2 = sigma^2 (X_*^T X_*)^{-1} [ X_*^T (I - rho^2 Delta) X_* + X_*^T Delta W V1 W^T Delta X_* ] (X_*^T X_*)^{-1}

        Parameters
        ----------
        X_star : np.ndarray of shape (n1, k + 1)
            Augmented design matrix [X_obs, lambda_1] for observed cases.
        W_obs : np.ndarray of shape (n1, m)
            Probit selection design matrix for observed cases.
        y_obs : np.ndarray of shape (n1,)
            Target variable values for observed cases.
        beta_star : np.ndarray of shape (k + 1,)
            Second-stage OLS coefficients [beta, beta_lambda].
        V1 : np.ndarray of shape (m, m)
            First-stage Probit parameter covariance matrix cov_params().
        delta : np.ndarray of shape (n1,)
            Selection hazard curvature delta_i = lambda_1 * (lambda_1 + eta_obs) in (0, 1).

        Returns
        -------
        V2 : np.ndarray of shape (k + 1, k + 1)
            Corrected asymptotic covariance matrix of beta_star.
        """
        e_obs = y_obs - (X_star @ beta_star)
        beta_lambda = float(beta_star[-1])
        mean_delta = float(np.mean(delta))
        sigma2_eps = float(np.mean(e_obs**2) + (beta_lambda**2) * mean_delta)
        sigma_eps = float(np.sqrt(max(1e-8, sigma2_eps)))
        rho = float(np.clip(beta_lambda / sigma_eps, -0.999, 0.999))

        # Curvature weight for second-stage heteroskedasticity
        D = 1.0 - (rho**2) * delta

        # Q = (X_*^T X_*)^{-1}
        XtX = X_star.T @ X_star
        Q = np.linalg.inv(XtX)

        # term1 = X_*^T (I - rho^2 Delta) X_*
        term1 = (X_star * D[:, np.newaxis]).T @ X_star

        # term2 = X_*^T Delta W_obs V1 W_obs^T Delta X_* = A V1 A^T
        A = X_star.T @ (delta[:, np.newaxis] * W_obs)
        term2 = A @ V1 @ A.T

        M = term1 + term2
        V2 = sigma2_eps * (Q @ M @ Q)
        V2 = 0.5 * (V2 + V2.T)
        return np.asarray(V2, dtype=float)

    def _compute_ridge_sandwich_covariance(
        self,
        X_star: np.ndarray,
        W_obs: np.ndarray,
        y_obs: np.ndarray,
        beta_star: np.ndarray,
        V1: np.ndarray,
        delta: np.ndarray,
        alpha: float = 1.0,
    ) -> np.ndarray:
        """Compute regularized Ridge sandwich covariance matrix for Heckman second stage.

        Formula:
          Q_alpha = (X_*^T X_* + alpha * I)^{-1}
          M = X_*^T (I - rho^2 Delta) X_* + X_*^T Delta W V1 W^T Delta X_*
          V_ridge = sigma^2 (Q_alpha M Q_alpha)

        Parameters
        ----------
        X_star : np.ndarray of shape (n1, k + 1)
            Augmented design matrix [X_obs, lambda_1] for observed cases.
        W_obs : np.ndarray of shape (n1, m)
            Probit selection design matrix for observed cases.
        y_obs : np.ndarray of shape (n1,)
            Target variable values for observed cases.
        beta_star : np.ndarray of shape (k + 1,)
            Regularized Ridge coefficients [beta, beta_lambda].
        V1 : np.ndarray of shape (m, m)
            First-stage Probit parameter covariance matrix cov_params().
        delta : np.ndarray of shape (n1,)
            Selection hazard curvature delta_i = lambda_1 * (lambda_1 + eta_obs).
        alpha : float, default=1.0
            Ridge regularization parameter (Tikhonov penalty).

        Returns
        -------
        V_ridge : np.ndarray of shape (k + 1, k + 1)
            Asymptotic regularized sandwich covariance matrix.
        """
        e_obs = y_obs - (X_star @ beta_star)
        beta_lambda = float(beta_star[-1])
        mean_delta = float(np.mean(delta))
        sigma2_eps = float(np.mean(e_obs**2) + (beta_lambda**2) * mean_delta)
        sigma_eps = float(np.sqrt(max(1e-8, sigma2_eps)))
        rho = float(np.clip(beta_lambda / sigma_eps, -0.999, 0.999))

        # Curvature weight for second-stage heteroskedasticity
        D = 1.0 - (rho**2) * delta

        # Q_alpha = (X_*^T X_* + alpha * I)^{-1}
        p = X_star.shape[1]
        XtX = X_star.T @ X_star
        regularized_XtX = XtX + alpha * np.eye(p)
        Q_alpha = np.linalg.inv(regularized_XtX)

        # M = X_*^T (I - rho^2 Delta) X_* + X_*^T Delta W_obs V1 W_obs^T Delta X_*
        term1 = (X_star * D[:, np.newaxis]).T @ X_star
        A = X_star.T @ (delta[:, np.newaxis] * W_obs)
        term2 = A @ V1 @ A.T
        M = term1 + term2

        V_ridge = sigma2_eps * (Q_alpha @ M @ Q_alpha)
        V_ridge = 0.5 * (V_ridge + V_ridge.T)
        return np.asarray(V_ridge, dtype=float)

    def _compute_collinearity_diagnostics(
        self,
        X_star: np.ndarray,
        col_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute Variance Inflation Factors (VIF) and Belsley condition indices for design matrix.

        Parameters
        ----------
        X_star : np.ndarray of shape (n1, p)
            Augmented design matrix [X_obs, lambda_1] for observed cases.
        col_names : Optional[List[str]], default=None
            Names of the columns in X_star.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - "condition_number": Condition number of unit-norm scaled design matrix (kappa).
            - "condition_indices": List of Belsley condition indices (eta_j).
            - "vif": Dictionary mapping column names to VIFs.
            - "vif_lambda": VIF of the inverse Mills ratio regressor lambda_1.
            - "severe_collinearity": bool indicating if kappa > 30 or VIF(lambda_1) > 10.
        """
        n, p = X_star.shape
        if col_names is None or len(col_names) != p:
            col_names = [f"col_{i}" for i in range(p)]

        # Unit-norm scaling for Belsley condition indices
        col_norms = np.linalg.norm(X_star, axis=0)
        col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
        X_scaled = X_star / col_norms

        _, s, _ = np.linalg.svd(X_scaled, full_matrices=False)
        s_max = float(s[0]) if len(s) > 0 else 1.0
        s_min = float(s[-1]) if len(s) > 0 else 1.0
        condition_number = float(s_max / max(1e-16, s_min))
        condition_indices = (s_max / np.maximum(1e-16, s)).tolist()

        # VIF computation for each column
        vif_dict: Dict[str, float] = {}
        for i, name in enumerate(col_names):
            y_i = X_star[:, i]
            cols_noti = [j for j in range(p) if j != i]
            X_noti = X_star[:, cols_noti]

            y_var = float(np.var(y_i))
            if y_var < 1e-12:
                # Intercept or constant column
                vif_dict[name] = 1.0
                continue

            try:
                coef, _, _, _ = np.linalg.lstsq(X_noti, y_i, rcond=None)
                pred = X_noti @ coef
                ss_tot = float(np.sum((y_i - np.mean(y_i)) ** 2))
                ss_res = float(np.sum((y_i - pred) ** 2))
                r2 = max(0.0, 1.0 - (ss_res / max(1e-12, ss_tot)))
                if r2 >= 1.0 - 1e-12:
                    vif = 1e12
                else:
                    vif = 1.0 / (1.0 - r2)
                vif_dict[name] = float(min(1e12, vif))
            except Exception:
                vif_dict[name] = float("inf")

        vif_lambda = float(vif_dict.get(col_names[-1], np.nan))
        severe_collinearity = bool(
            condition_number > 30.0 or (np.isfinite(vif_lambda) and vif_lambda > 10.0)
        )

        return {
            "condition_number": condition_number,
            "condition_indices": condition_indices,
            "vif": vif_dict,
            "vif_lambda": vif_lambda,
            "severe_collinearity": severe_collinearity,
        }

    @staticmethod
    def _fiml_neg_log_likelihood(
        theta: np.ndarray,
        X_obs: np.ndarray,
        y_obs: np.ndarray,
        W_obs: np.ndarray,
        W_unobs: np.ndarray,
    ) -> float:
        """Negative joint bivariate normal log-likelihood for Heckman selection model."""
        k = X_obs.shape[1]
        m = W_obs.shape[1]
        beta = theta[:k]
        gamma = theta[k : k + m]
        tau = float(theta[k + m])
        zeta = float(theta[k + m + 1])

        sigma = np.exp(np.clip(tau, -20.0, 20.0))
        rho = float(np.tanh(zeta))

        # 1. Unobserved cases: ln Phi(-W_unobs @ gamma)
        if len(W_unobs) > 0:
            eta_unobs = W_unobs @ gamma
            ll_unobs = float(np.sum(special.log_ndtr(-eta_unobs)))
        else:
            ll_unobs = 0.0

        # 2. Observed cases:
        # ln f(y_obs) + ln Phi((W_obs @ gamma + (rho/sigma)*(y_obs - X_obs @ beta)) / sqrt(1 - rho^2))
        resid = y_obs - (X_obs @ beta)
        eta_obs = W_obs @ gamma
        sqrt_1_minus_rho2 = float(np.sqrt(max(1e-12, 1.0 - rho**2)))
        arg = (eta_obs + (rho / sigma) * resid) / sqrt_1_minus_rho2

        ll_y = -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * (resid / sigma) ** 2
        ll_cond = special.log_ndtr(arg)
        ll_obs = float(np.sum(ll_y + ll_cond))

        total_ll = ll_unobs + ll_obs
        if not np.isfinite(total_ll):
            return 1e12
        return -total_ll

    def _fit_fiml(
        self,
        X_obs: np.ndarray,
        y_obs: np.ndarray,
        W_obs: np.ndarray,
        W_unobs: np.ndarray,
        beta_init: np.ndarray,
        gamma_init: np.ndarray,
        sigma_init: float,
        rho_init: float,
    ) -> Optional[Dict[str, Any]]:
        """Fit Heckman selection model via Full-Information Maximum Likelihood (FIML)."""
        k = X_obs.shape[1]
        m = W_obs.shape[1]

        tau_0 = float(np.log(max(1e-4, sigma_init)))
        zeta_0 = float(np.arctanh(np.clip(rho_init, -0.99, 0.99)))
        theta_0 = np.concatenate([beta_init, gamma_init, [tau_0], [zeta_0]])

        try:
            res = optimize.minimize(
                self._fiml_neg_log_likelihood,
                theta_0,
                args=(X_obs, y_obs, W_obs, W_unobs),
                method="L-BFGS-B",
                options={"maxiter": 500, "ftol": 1e-7, "gtol": 1e-5},
            )
            if not res.success:
                res = optimize.minimize(
                    self._fiml_neg_log_likelihood,
                    theta_0,
                    args=(X_obs, y_obs, W_obs, W_unobs),
                    method="BFGS",
                    options={"maxiter": 500, "gtol": 1e-5},
                )

            if not res.success or np.any(np.isnan(res.x)):
                return None

            # Compute observed Hessian of negative log-likelihood at minimum
            H = approx_hess(
                res.x,
                self._fiml_neg_log_likelihood,
                args=(X_obs, y_obs, W_obs, W_unobs),
            )
            if np.any(np.isnan(H)):
                return None

            eigvals = np.linalg.eigvalsh(H)
            if np.any(eigvals <= 1e-8):
                return None

            cov_theta = np.linalg.inv(H)
            cov_theta = 0.5 * (cov_theta + cov_theta.T)
            diag_cov = np.diag(cov_theta)
            if np.any(diag_cov <= 0) or np.any(np.isnan(diag_cov)):
                return None

            beta_fiml = res.x[:k]
            gamma_fiml = res.x[k : k + m]
            tau_fiml = float(res.x[k + m])
            zeta_fiml = float(res.x[k + m + 1])
            sigma_fiml = float(np.exp(tau_fiml))
            rho_fiml = float(np.tanh(zeta_fiml))
            beta_lambda_fiml = float(rho_fiml * sigma_fiml)

            se_theta = np.sqrt(diag_cov)
            beta_se = se_theta[:k]
            gamma_se = se_theta[k : k + m]
            sigma_se = float(sigma_fiml * se_theta[k + m])
            rho_se = float((1.0 - rho_fiml**2) * se_theta[k + m + 1])

            # Jacobian mapping (beta, tau, zeta) -> (beta, beta_lambda)
            # beta_lambda = rho * sigma = tanh(zeta) * exp(tau)
            # d(beta_lambda)/d(tau) = rho * sigma
            # d(beta_lambda)/d(zeta) = (1 - rho^2) * sigma
            J = np.zeros((k + 1, k + 2), dtype=float)
            J[:k, :k] = np.eye(k)
            J[k, k] = rho_fiml * sigma_fiml
            J[k, k + 1] = (1.0 - rho_fiml**2) * sigma_fiml

            sub_idx = list(range(k)) + [k + m, k + m + 1]
            V_sub = cov_theta[np.ix_(sub_idx, sub_idx)]
            V_out = J @ V_sub @ J.T
            V_out = 0.5 * (V_out + V_out.T)
            mills_se = float(np.sqrt(max(1e-16, V_out[-1, -1])))

            return {
                "beta": beta_fiml,
                "gamma": gamma_fiml,
                "beta_lambda": beta_lambda_fiml,
                "params": np.concatenate([beta_fiml, [beta_lambda_fiml]]),
                "sigma_eps": sigma_fiml,
                "rho": rho_fiml,
                "std_errors": np.concatenate([beta_se, [mills_se]]),
                "coef_stderr": beta_se,
                "gamma_se": gamma_se,
                "mills_stderr": mills_se,
                "coef_cov": V_out,
                "fiml_cov_theta": cov_theta,
                "log_likelihood": float(-res.fun),
                "sigma_se": sigma_se,
                "rho_se": rho_se,
                "method_used": "fiml",
            }
        except Exception:
            return None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Any = None) -> "HeckmanSelectionImputer":
        """Fit Heckman selection models for target columns."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            self.n_features_in_ = len(self.feature_names_in_)
        else:
            self.n_features_in_ = int(X.shape[1])
            if hasattr(self, "feature_names_in_"):
                delattr(self, "feature_names_in_")

        df = self._to_dataframe(X).copy()

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
                self.col_medians_[col] = float(df[col].median())
            else:
                self.col_medians_[col] = 0.0

        targets = self.target_cols or [c for c in df.columns if df[c].isna().any()]
        shadow_map = self.shadow_cols or {}

        if self.ridge_se_method not in ("sandwich", "nan"):
            raise ValueError(
                f"ridge_se_method must be 'sandwich' or 'nan', got {self.ridge_se_method!r}"
            )
        if self.ridge_alpha <= 0:
            raise ValueError(f"ridge_alpha must be positive, got {self.ridge_alpha}")

        # Resolve effective standard error method for two-step
        if self.se_method is not None:
            eff_se_method = self.se_method
        elif self.n_bootstrap_se > 0:
            eff_se_method = "bootstrap"
        else:
            eff_se_method = "murphy-topel"

        for target in targets:
            if target not in df.columns or not df[target].isna().any():
                continue

            obs_mask = df[target].notna()
            R = obs_mask.astype(int).to_numpy()

            if obs_mask.sum() < 10 or (~obs_mask).sum() < 3:
                continue

            covar_cols = [
                c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])
            ]
            if not covar_cols:
                continue

            # Check exclusion restriction
            shadow_var = shadow_map.get(target)
            if shadow_var and shadow_var in covar_cols:
                X_cols = [c for c in covar_cols if c != shadow_var]
                W_cols = covar_cols.copy()
                try:
                    z_data = df[[shadow_var]].fillna(df[shadow_var].median()).to_numpy(dtype=float)
                    z_mat = sm.add_constant(z_data, has_constant="add")
                    ols_first = sm.OLS(R, z_mat).fit()
                    f_stat = float(ols_first.fvalue)
                    if f_stat <= 10.0:
                        warnings.warn(
                            f"Candidate auxiliary instrument '{shadow_var}' for target '{target}' has weak relevance "
                            f"(first-stage F = {f_stat:.2f} <= 10.0, Stock-Yogo benchmark). Heckman selection estimates "
                            "may suffer from variance inflation and numerical instability.",
                            WeakInstrumentWarning,
                        )
                except Exception:
                    pass
            else:
                X_cols = covar_cols.copy()
                W_cols = covar_cols.copy()
                warnings.warn(
                    f"Heckman model for '{target}' has no exclusion restriction (shadow variable). "
                    "Identification relies purely on the Probit functional form, which may induce "
                    "high collinearity and estimator instability.",
                    UserWarning,
                )

            # Impute median for missing values in predictors before fitting
            W_df = df[W_cols].fillna(df[W_cols].median())
            W_mat = sm.add_constant(W_df.to_numpy(dtype=float), has_constant="add")

            # Step 1: Probit model for selection equation P(R=1 | W)
            try:
                probit_mod = sm.Probit(R, W_mat)
                probit_res = probit_mod.fit(disp=False, maxiter=100)
                gamma = probit_res.params
                V1 = probit_res.cov_params()
            except Exception:
                try:
                    logit_mod = sm.Logit(R, W_mat)
                    logit_res = logit_mod.fit(disp=False, maxiter=100)
                    gamma = logit_res.params / 1.6  # Standard Probit approximation
                    V1 = (logit_res.cov_params()) / (1.6**2)
                except Exception:
                    # Fallback linear probability model
                    ols_lpm = sm.OLS(R, W_mat).fit()
                    gamma = ols_lpm.params * 2.5
                    V1 = (ols_lpm.cov_params()) * (2.5**2)

            eta = W_mat @ gamma
            lambda_1 = _compute_imr_observed(eta[obs_mask])

            # Step 2: Outcome regression on observed cases
            X_df = df.loc[obs_mask, X_cols].fillna(df[X_cols].median())
            X_mat_obs = sm.add_constant(X_df.to_numpy(dtype=float), has_constant="add")

            # Augmented regression design: [1, X, lambda_1]
            design_obs = np.column_stack([X_mat_obs, lambda_1])
            y_obs = df.loc[obs_mask, target].to_numpy(dtype=float)

            # Collinearity diagnostics for second stage
            col_names = ["const"] + list(X_cols) + ["mills_ratio"]
            collin_diag = self._compute_collinearity_diagnostics(design_obs, col_names=col_names)
            if collin_diag["severe_collinearity"]:
                warnings.warn(
                    f"Severe collinearity detected in Heckman second stage for target '{target}'. "
                    f"Condition index: {collin_diag['condition_number']:.1f} (threshold > 30), "
                    f"VIF(lambda): {collin_diag['vif_lambda']:.1f} (threshold > 10). "
                    "Second-stage standard errors and estimates may be unstable due to weak exclusion restriction.",
                    HeckmanCollinearityWarning,
                    stacklevel=2,
                )

            ridge_fallback_used = False
            try:
                if np.linalg.matrix_rank(design_obs) < design_obs.shape[1]:
                    raise np.linalg.LinAlgError(
                        "Near-singular design matrix in Heckman second stage."
                    )
                ols_res = sm.OLS(y_obs, design_obs).fit()
                params = ols_res.params
                beta = params[:-1]
                beta_lambda = float(params[-1])

                # Residual variance estimation with Heckman (1979) adjustment
                e_obs = y_obs - (design_obs @ params)
                delta_1 = lambda_1 * (lambda_1 + eta[obs_mask])
                mean_delta_1 = float(np.mean(delta_1))
                sigma2_eps = float(np.mean(e_obs**2) + (beta_lambda**2) * mean_delta_1)
                sigma_eps = float(np.sqrt(max(1e-8, sigma2_eps)))
                rho = float(np.clip(beta_lambda / sigma_eps, -0.999, 0.999))

                # Exact Murphy & Topel (1985) asymptotic covariance matrix
                V2 = self._compute_murphy_topel_covariance(
                    X_star=design_obs,
                    W_obs=W_mat[obs_mask],
                    y_obs=y_obs,
                    beta_star=params,
                    V1=V1,
                    delta=delta_1,
                )
                mt_se = np.sqrt(np.maximum(1e-16, np.diag(V2)))

                # Resolve standard errors for two-step
                bootstrap_se = None
                if eff_se_method == "bootstrap" or self.n_bootstrap_se > 0:
                    X_df_all = df[X_cols].fillna(df[X_cols].median())
                    X_mat_all = sm.add_constant(X_df_all.to_numpy(dtype=float), has_constant="add")
                    rng_seed = (
                        (self.random_state + 999) % (2**31 - 1)
                        if self.random_state is not None
                        else None
                    )
                    rng_se = np.random.RandomState(rng_seed)
                    n_boot_draws = self.n_bootstrap_se if self.n_bootstrap_se > 0 else 200
                    bootstrap_se = _bootstrap_heckman_se(
                        W_mat=W_mat,
                        X_mat_obs=X_mat_all,
                        y_obs=df[target].to_numpy(dtype=float),
                        R_obs=R,
                        n_boot=n_boot_draws,
                        rng=rng_se,
                    )

                if eff_se_method == "bootstrap" and bootstrap_se is not None:
                    std_errors = bootstrap_se
                    coef_cov = np.diag(std_errors**2)
                elif eff_se_method == "naive":
                    warnings.warn(
                        "Heckman selection imputer fitted with naive OLS standard errors (se_method='naive'). "
                        "Second-stage standard errors are naive OLS standard errors that ignore "
                        "first-stage estimation uncertainty (Murphy-Topel 1985 bias) and understate uncertainty.",
                        HeckmanSEWarning,
                        stacklevel=2,
                    )
                    std_errors = np.asarray(ols_res.bse)
                    coef_cov = ols_res.cov_params()
                else:
                    # Murphy-Topel analytical SE
                    std_errors = mt_se
                    coef_cov = V2

            except Exception:
                # Regularized ridge fallback
                from sklearn.linear_model import Ridge

                ridge_fallback_used = True
                r_est = Ridge(alpha=self.ridge_alpha, fit_intercept=False).fit(design_obs, y_obs)
                params = r_est.coef_
                beta = params[:-1]
                beta_lambda = float(params[-1])
                bootstrap_se = None

                if self.ridge_se_method == "sandwich":
                    warnings.warn(
                        "Heckman second-stage OLS failed due to near-singular design matrix. "
                        f"Ridge regression fallback was used (alpha={self.ridge_alpha}) with regularized "
                        "sandwich covariance. Standard errors reflect regularized estimation uncertainty.",
                        HeckmanSEWarning,
                        stacklevel=2,
                    )
                    e_obs = y_obs - (design_obs @ params)
                    delta_1 = lambda_1 * (lambda_1 + eta[obs_mask])
                    mean_delta_1 = float(np.mean(delta_1))
                    sigma2_eps = float(np.mean(e_obs**2) + (beta_lambda**2) * mean_delta_1)
                    sigma_eps = float(np.sqrt(max(1e-8, sigma2_eps)))
                    rho = float(np.clip(beta_lambda / sigma_eps, -0.999, 0.999))

                    V_ridge = self._compute_ridge_sandwich_covariance(
                        X_star=design_obs,
                        W_obs=W_mat[obs_mask],
                        y_obs=y_obs,
                        beta_star=params,
                        V1=V1,
                        delta=delta_1,
                        alpha=self.ridge_alpha,
                    )
                    std_errors = np.sqrt(np.maximum(1e-16, np.diag(V_ridge)))
                    mt_se = std_errors
                    coef_cov = V_ridge
                    V2 = V_ridge
                else:
                    warnings.warn(
                        "Heckman second-stage OLS failed due to near-singular design matrix. "
                        "Ridge regression fallback was used. Standard errors are set to NaN, "
                        "not zero, to prevent silent zero-width confidence intervals. "
                        "Imputed point estimates may still be reasonable but uncertainty "
                        "quantification is unavailable for this column.",
                        HeckmanSEWarning,
                        stacklevel=2,
                    )
                    sigma_eps = float(max(1e-4, np.std(y_obs)))
                    rho = 0.0
                    std_errors = np.full_like(params, np.nan)
                    mt_se = np.full_like(params, np.nan)
                    V2 = np.full((len(params), len(params)), np.nan)
                    coef_cov = V2

            method_used: Literal["two-step", "fiml"] = "two-step"
            log_likelihood: Optional[float] = None
            fiml_cov_theta: Optional[np.ndarray] = None

            # FIML estimation branch
            if self.method == "fiml" and not ridge_fallback_used:
                fiml_dict = self._fit_fiml(
                    X_obs=X_mat_obs,
                    y_obs=y_obs,
                    W_obs=W_mat[obs_mask],
                    W_unobs=W_mat[~obs_mask],
                    beta_init=beta,
                    gamma_init=gamma,
                    sigma_init=sigma_eps,
                    rho_init=rho,
                )
                if fiml_dict is not None:
                    method_used = "fiml"
                    beta = fiml_dict["beta"]
                    gamma = fiml_dict["gamma"]
                    beta_lambda = fiml_dict["beta_lambda"]
                    params = fiml_dict["params"]
                    sigma_eps = fiml_dict["sigma_eps"]
                    rho = fiml_dict["rho"]
                    std_errors = fiml_dict["std_errors"]
                    coef_cov = fiml_dict["coef_cov"]
                    log_likelihood = fiml_dict["log_likelihood"]
                    fiml_cov_theta = fiml_dict["fiml_cov_theta"]
                else:
                    warnings.warn(
                        "FIML optimization did not converge or yielded singular Hessian; falling back to two-step estimator.",
                        HeckmanConvergenceWarning,
                        stacklevel=2,
                    )
                    method_used = "two-step"

            # Store fitted model metadata
            model_info: Dict[str, Any] = {
                "gamma": gamma,
                "beta": beta,
                "beta_lambda": beta_lambda,
                "params": params,
                "std_errors": std_errors,
                "murphy_topel_stderr": mt_se,
                "murphy_topel_cov": V2,
                "coef_cov": coef_cov,
                "sigma_eps": sigma_eps,
                "rho": rho,
                "method_used": method_used,
                "log_likelihood": log_likelihood,
                "fiml_cov_theta": fiml_cov_theta,
                "W_cols": W_cols,
                "X_cols": X_cols,
                "shadow_var": shadow_var,
                "collinearity_diagnostics": collin_diag,
                "ridge_fallback_used": ridge_fallback_used,
            }
            if bootstrap_se is not None:
                model_info["bootstrap_stderr"] = bootstrap_se

            self.models_[target] = model_info

            # Expose primary fitted attributes directly on self
            self.method_used_ = method_used
            self.coef_cov_ = coef_cov
            self.coef_stderr_ = std_errors[:-1]
            self.mills_stderr_ = float(std_errors[-1])
            self.sigma_ = sigma_eps
            self.rho_ = rho
            self.log_likelihood_ = log_likelihood
            self.collinearity_diagnostics_ = collin_diag
            self.ridge_fallback_used_ = ridge_fallback_used
            if bootstrap_se is not None:
                self.bootstrap_stderr_ = bootstrap_se

        self.is_fitted_ = True
        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray], return_all_imputations: bool = False
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Impute missing values using the fitted Heckman selection model."""
        check_is_fitted(self, "is_fitted_")
        n_features = (
            X.shape[1]
            if hasattr(X, "shape") and len(X.shape) > 1
            else len(getattr(X, "columns", []))
        )
        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, but {self.__class__.__name__} is expecting {self.n_features_in_} features as input."
            )
        if (
            isinstance(X, pd.DataFrame)
            and hasattr(self, "feature_names_in_")
            and self.feature_names_in_ is not None
        ):
            if list(X.columns) != list(self.feature_names_in_):
                raise ValueError(
                    f"The feature names should match those that were passed during fit. "
                    f"Expected {list(self.feature_names_in_)}, got {list(X.columns)}"
                )

        if not self.models_:
            df = self._to_dataframe(X).copy()
            return [df] if return_all_imputations else df

        rng = np.random.RandomState(self.random_state)
        n_draws = max(1, self.n_imputations if self.stochastic or self.n_imputations > 1 else 1)
        imputed_dfs = []

        for draw in range(n_draws):
            df_cur = self._to_dataframe(X).copy()

            for target, model_info in self.models_.items():
                if target not in df_cur.columns:
                    continue

                mis_mask = df_cur[target].isna()
                if mis_mask.sum() == 0:
                    continue

                W_cols = model_info["W_cols"]
                X_cols = model_info["X_cols"]
                gamma = model_info["gamma"]
                beta = model_info["beta"]
                beta_lambda = model_info["beta_lambda"]
                sigma_eps = model_info["sigma_eps"]
                rho = model_info["rho"]

                # Missing subset design
                W_mis = (
                    df_cur.loc[mis_mask, W_cols]
                    .fillna(df_cur[W_cols].median())
                    .to_numpy(dtype=float)
                )
                W_mat_mis = sm.add_constant(W_mis, has_constant="add")
                eta_mis = W_mat_mis @ gamma

                # Compute lambda_0 for missing cases (unobserved selection)
                lambda_0 = _compute_imr_missing(eta_mis)

                X_mis = (
                    df_cur.loc[mis_mask, X_cols]
                    .fillna(df_cur[X_cols].median())
                    .to_numpy(dtype=float)
                )
                X_mat_mis = sm.add_constant(X_mis, has_constant="add")

                # Conditional expectation under truncated normal selection:
                # E[Y | R=0] = X * beta + beta_lambda * lambda_0
                expected_y = (X_mat_mis @ beta) + (beta_lambda * lambda_0)

                if self.stochastic or n_draws > 1:
                    delta_0 = -lambda_0 * (eta_mis - lambda_0)
                    cond_var = (sigma_eps**2) * np.clip(1.0 - (rho**2) * delta_0, 1e-4, None)
                    noise = rng.normal(0.0, np.sqrt(cond_var))
                    expected_y += noise

                df_cur.loc[mis_mask, target] = expected_y

            imputed_dfs.append(df_cur)

        if return_all_imputations:
            return imputed_dfs
        return imputed_dfs[0]

    def fit_transform_multiple(self, X: Union[pd.DataFrame, np.ndarray]) -> List[pd.DataFrame]:
        """Fit Heckman selection models and generate all M stochastic multiple imputations."""
        self.fit(X)
        return self.transform(X, return_all_imputations=True)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        if input_features is not None:
            if len(input_features) != self.n_features_in_:
                raise ValueError(
                    f"input_features should have length equal to number of features ({self.n_features_in_}), "
                    f"got {len(input_features)}"
                )
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_") and self.feature_names_in_ is not None:
            return np.asarray(self.feature_names_in_, dtype=object)
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)

    def _to_dataframe(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if hasattr(self, "feature_names_in_") and self.feature_names_in_ is not None:
            cols = [str(c) for c in self.feature_names_in_]
        else:
            cols = [f"x{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols)
