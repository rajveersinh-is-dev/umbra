"""
Heckman Selection Model Imputer for MNAR Data.

Implements James Heckman's (1976, 1979) two-step selection estimator adapted
for missing data imputation and sensitivity bounds under MNAR.

Model:
  Selection Equation: z_i* = W_i * gamma + u_i,  R_i = 1(z_i* > 0)
  Outcome Equation:   Y_i  = X_i * beta  + eps_i, observed only when R_i = 1
  Errors: (u_i, eps_i) ~ Bivariate Normal with correlation rho and Var(u_i) = 1.

Imputation for unobserved cases (R_i = 0):
  E[Y_i | R_i = 0, X_i, W_i] = X_i * beta + beta_lambda * lambda_0(W_i * gamma)
  where lambda_0(eta) = -phi(eta) / (1 - Phi(eta)) = -phi(-eta) / Phi(-eta) <= 0.

References:
Heckman, J. J. (1976). The common structure of statistical models of
truncation, sample selection and limited dependent variables.
Annals of Economic and Social Measurement, 5(4), 475-492.
Heckman, J. J. (1979). Sample selection bias as a specification error.
Econometrica, 47(1), 153-161.
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin


def _compute_imr_observed(eta: np.ndarray) -> np.ndarray:
    """Compute Inverse Mills Ratio for observed cases (R=1):
    lambda_1(eta) = phi(eta) / Phi(eta) >= 0.
    """
    eta = np.clip(eta, -30.0, 30.0)
    phi = stats.norm.pdf(eta)
    Phi = stats.norm.cdf(eta)
    safe_Phi = np.where(Phi < 1e-12, 1e-12, Phi)
    imr = phi / safe_Phi

    # Tail correction for extreme negative eta
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

    # Tail correction for extreme positive eta (negative -eta)
    extreme_neg = neg_eta < -10.0
    if np.any(extreme_neg):
        imr_0[extreme_neg] = -np.exp(
            stats.norm.logpdf(neg_eta[extreme_neg]) - stats.norm.logcdf(neg_eta[extreme_neg])
        )
    return np.asarray(imr_0, dtype=float)


class HeckmanSelectionImputer(BaseEstimator, TransformerMixin):
    """Heckman two-step selection model imputer for MNAR tabular data.

    Parameters
    ----------
    target_cols : Optional[List[str]], default=None
        Columns to impute with Heckman selection model. If None, targets all
        columns with missing values.
    shadow_cols : Optional[Dict[str, str]], default=None
        Mapping of {target_col: shadow_var_col}. Shadow variables enter the
        selection equation W but are EXCLUDED from the outcome equation X
        (classic exclusion restriction / instrument).
    stochastic : bool, default=False
        If True, draws normal noise with conditional variance Var(Y | R=0).
    n_imputations : int, default=1
        Number of stochastic imputation draws (if > 1, stochastic is enabled).
    random_state : Optional[int], default=42
        Reproducibility seed.
    """

    def __init__(
        self,
        target_cols: Optional[List[str]] = None,
        shadow_cols: Optional[Dict[str, str]] = None,
        stochastic: bool = False,
        n_imputations: int = 1,
        n_draws: Optional[int] = None,
        random_state: Optional[int] = 42,
    ):
        self.target_cols = target_cols
        self.shadow_cols = shadow_cols
        self.stochastic = stochastic
        self.n_imputations = n_draws if n_draws is not None else n_imputations
        self.n_draws = n_draws
        self.random_state = random_state

        # Fitted attributes
        self.models_: Dict[str, Dict[str, Any]] = {}
        self.col_medians_: Dict[str, float] = {}
        self.feature_names_in_: List[str] = []
        self.n_features_in_: int = 0

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Fit Heckman selection models for target columns."""
        df = self._to_dataframe(X).copy()
        self.feature_names_in_ = list(df.columns)
        self.n_features_in_ = len(self.feature_names_in_)

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
                self.col_medians_[col] = float(df[col].median())
            else:
                self.col_medians_[col] = 0.0

        targets = self.target_cols or [c for c in df.columns if df[c].isna().any()]
        shadow_map = self.shadow_cols or {}

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
            except Exception:
                try:
                    logit_mod = sm.Logit(R, W_mat)
                    logit_res = logit_mod.fit(disp=False, maxiter=100)
                    gamma = logit_res.params / 1.6  # Standard Probit approximation
                except Exception:
                    # Fallback linear probability model
                    ols_lpm = sm.OLS(R, W_mat).fit()
                    gamma = ols_lpm.params * 2.5

            eta = W_mat @ gamma
            lambda_1 = _compute_imr_observed(eta[obs_mask])

            # Step 2: Outcome regression on observed cases
            X_df = df.loc[obs_mask, X_cols].fillna(df[X_cols].median())
            X_mat_obs = sm.add_constant(X_df.to_numpy(dtype=float), has_constant="add")

            # Augmented regression design: [1, X, lambda_1]
            design_obs = np.column_stack([X_mat_obs, lambda_1])
            y_obs = df.loc[obs_mask, target].to_numpy(dtype=float)

            try:
                ols_res = sm.OLS(y_obs, design_obs).fit()
                params = ols_res.params
                std_errors = ols_res.bse
            except Exception:
                # Regularized ridge fallback
                from sklearn.linear_model import Ridge

                r_est = Ridge(alpha=1.0).fit(design_obs, y_obs)
                params = np.concatenate([[r_est.intercept_], r_est.coef_[1:]])
                std_errors = np.zeros_like(params)

            beta = params[:-1]
            beta_lambda = float(params[-1])

            # Residual variance estimation with Heckman adjustment
            e_obs = y_obs - (design_obs @ params)
            delta_1 = lambda_1 * (lambda_1 + eta[obs_mask])
            mean_delta_1 = float(np.mean(delta_1))
            sigma2_eps = float(np.mean(e_obs**2) + (beta_lambda**2) * mean_delta_1)
            sigma_eps = float(np.sqrt(max(1e-6, sigma2_eps)))
            rho = float(np.clip(beta_lambda / sigma_eps, -0.99, 0.99))

            self.models_[target] = {
                "gamma": gamma,
                "beta": beta,
                "beta_lambda": beta_lambda,
                "params": params,
                "std_errors": std_errors,
                "sigma_eps": sigma_eps,
                "rho": rho,
                "W_cols": W_cols,
                "X_cols": X_cols,
                "shadow_var": shadow_var,
            }

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray], return_all_imputations: bool = False
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Impute missing values using the fitted Heckman selection model."""
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

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_)

    def _to_dataframe(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        cols = (
            self.feature_names_in_
            if self.feature_names_in_
            else [f"x_{i}" for i in range(X.shape[1])]
        )
        return pd.DataFrame(X, columns=cols)
