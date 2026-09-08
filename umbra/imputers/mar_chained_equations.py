"""
MAR Chained Equations Imputer (MICE Baseline).

Implements Multiple Imputation by Chained Equations under the Missing at Random (MAR) assumption.
Supports Predictive Mean Matching (PMM), Bayesian Ridge regression posterior draws,
and Rubin's Rules for pooling multiple imputations and calculating confidence intervals.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.neighbors import NearestNeighbors


@dataclass
class RubinsRulesResult:
    """Result of pooling multiple imputation estimates via Rubin's (1987) rules."""

    pooled_estimate: float
    within_variance: float
    between_variance: float
    total_variance: float
    standard_error: float
    df: float
    ci_lower: float
    ci_upper: float

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
        }


def rubins_rules(
    point_estimates: List[float],
    variance_estimates: List[float],
    alpha: float = 0.05,
) -> RubinsRulesResult:
    """Pool multiple imputation estimates using Rubin's (1987) Rules.

    Parameters
    ----------
    point_estimates : List[float]
        Estimates Q_hat_m across M imputations.
    variance_estimates : List[float]
        Within-imputation variance estimates U_hat_m across M imputations.
    alpha : float, default=0.05
        Significance level for pooled confidence interval.

    Returns
    -------
    RubinsRulesResult
        Pooled estimate, within variance, between variance, total variance,
        degrees of freedom, and confidence bounds.
    """
    m = len(point_estimates)
    if m == 1:
        warnings.warn(
            "rubins_rules called with M=1 imputation. Between-imputation variance is zero "
            "by construction (not because uncertainty is small). Use M>=5 for valid pooled inference.",
            UserWarning,
            stacklevel=2,
        )
        q_bar = point_estimates[0]
        t_var = variance_estimates[0]
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        return RubinsRulesResult(
            pooled_estimate=float(q_bar),
            within_variance=float(t_var),
            between_variance=0.0,
            total_variance=float(t_var),
            standard_error=float(np.sqrt(max(1e-12, t_var))),
            df=float("inf"),
            ci_lower=float(q_bar - z * np.sqrt(max(1e-12, t_var))),
            ci_upper=float(q_bar + z * np.sqrt(max(1e-12, t_var))),
        )

    q_bar = np.mean(point_estimates)
    u_bar = np.mean(variance_estimates)
    b_var = np.var(point_estimates, ddof=1)
    t_var = u_bar + (1.0 + 1.0 / m) * b_var

    # Barnard & Rubin (1999) degrees of freedom
    df_val: float
    if b_var > 1e-12:
        r = (1.0 + 1.0 / m) * b_var / max(1e-12, u_bar)
        df_val = float((m - 1) * (1.0 + 1.0 / r) ** 2)
    else:
        df_val = float("inf")

    se = np.sqrt(max(1e-12, t_var))
    crit = (
        stats.t.ppf(1.0 - alpha / 2.0, df_val)
        if np.isfinite(df_val)
        else stats.norm.ppf(1.0 - alpha / 2.0)
    )

    return RubinsRulesResult(
        pooled_estimate=float(q_bar),
        within_variance=float(u_bar),
        between_variance=float(b_var),
        total_variance=float(t_var),
        standard_error=float(se),
        df=float(df_val),
        ci_lower=float(q_bar - crit * se),
        ci_upper=float(q_bar + crit * se),
    )


class MARChainedEquationsImputer(BaseEstimator, TransformerMixin):
    """MICE-style chained equations imputer under the Missing at Random (MAR) assumption.

    Parameters
    ----------
    max_iter : int, default=10
        Number of chained equations iteration cycles.
    imputation_method : str, default='pmm'
        - 'pmm': Predictive Mean Matching (draws from k nearest observed donors)
        - 'bayesian_ridge': Posterior predictive draws from Bayesian Ridge regression
        - 'ridge': Deterministic Ridge regression prediction
    n_donors : int, default=5
        Number of nearest neighbor donors for PMM.
    n_imputations : int, default=1
        Number of imputed datasets generated (M).
    random_state : Optional[int], default=42
        Seed for reproducibility.
    """

    def __init__(
        self,
        max_iter: int = 10,
        imputation_method: str = "pmm",
        n_donors: int = 5,
        n_imputations: int = 1,
        random_state: Optional[int] = 42,
        **kwargs,
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

        self.max_iter = max_iter
        self.imputation_method = imputation_method
        self.n_donors = n_donors
        self.n_imputations = n_imputations
        self.random_state = random_state

        # Fitted attributes
        self.columns_: List[str] = []
        self.incomplete_cols_: List[str] = []
        self.models_: Dict[str, Union[BayesianRidge, Ridge]] = {}
        self.col_medians_: Dict[str, float] = {}
        self.feature_names_in_: List[str] = []
        self.n_features_in_: int = 0

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Fit chained equations models on available data."""
        df = self._to_dataframe(X).copy()
        self.feature_names_in_ = list(df.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        self.columns_ = list(df.columns)

        for col in self.columns_:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
                self.col_medians_[col] = float(df[col].median())
            else:
                self.col_medians_[col] = 0.0

        self.incomplete_cols_ = [c for c in df.columns if df[c].isna().any()]
        self.incomplete_cols_.sort(key=lambda c: df[c].isna().sum())

        if not self.incomplete_cols_:
            return self

        # Initial working matrix via median imputation
        working_df = df.copy()
        for col in self.incomplete_cols_:
            working_df[col] = working_df[col].fillna(self.col_medians_[col])

        rng = np.random.RandomState(self.random_state)

        # Chained equations iterations
        for _ in range(self.max_iter):
            for target in self.incomplete_cols_:
                obs_mask = df[target].notna()
                if obs_mask.sum() < 3:
                    continue

                predictor_cols = [
                    c for c in self.columns_ if c != target and pd.api.types.is_numeric_dtype(df[c])
                ]
                if not predictor_cols:
                    continue

                X_train = working_df.loc[obs_mask, predictor_cols].to_numpy(dtype=float)
                y_train = df.loc[obs_mask, target].to_numpy(dtype=float)

                if self.imputation_method == "bayesian_ridge":
                    model: Union[BayesianRidge, Ridge] = BayesianRidge()
                else:
                    model = Ridge(alpha=1.0)

                try:
                    model.fit(X_train, y_train)
                    self.models_[target] = model
                except Exception:
                    continue

                mis_mask = ~obs_mask
                if mis_mask.sum() > 0:
                    X_mis = working_df.loc[mis_mask, predictor_cols].to_numpy(dtype=float)
                    if self.imputation_method == "pmm" and obs_mask.sum() >= 2:
                        preds_obs = model.predict(X_train)
                        preds_mis = model.predict(X_mis)
                        imputed_vals = self._pmm_draw(preds_obs, y_train, preds_mis, rng)
                    elif self.imputation_method == "bayesian_ridge" and isinstance(
                        model, BayesianRidge
                    ):
                        preds, std = model.predict(X_mis, return_std=True)
                        imputed_vals = rng.normal(preds, np.maximum(1e-6, std))
                    else:
                        imputed_vals = model.predict(X_mis)

                    working_df.loc[mis_mask, target] = imputed_vals

        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray], return_all_imputations: bool = False
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Impute missing values using the fitted chained equations."""
        df_base = self._to_dataframe(X).copy()
        if not self.incomplete_cols_:
            return [df_base] if return_all_imputations else df_base

        rng = np.random.RandomState(self.random_state)
        n_draws = max(1, self.n_imputations)
        imputed_dfs = []

        for draw in range(n_draws):
            df = df_base.copy()
            # Note: predictor values in working_df may differ from those seen during fit()
            # because earlier columns in this draw have been filled. This approximates
            # a second chained-equations iteration at transform time, improving imputation
            # quality for multivariate missing patterns but does not change model parameters.
            working_df = df.copy()
            for col in self.columns_:
                if col in self.col_medians_:
                    working_df[col] = working_df[col].fillna(self.col_medians_[col])

            for target in self.incomplete_cols_:
                obs_mask = df[target].notna()
                mis_mask = ~obs_mask

                if mis_mask.sum() == 0:
                    continue

                if target in self.models_:
                    model = self.models_[target]
                    predictor_cols = [
                        c
                        for c in self.columns_
                        if c != target and pd.api.types.is_numeric_dtype(df[c])
                    ]
                    X_mis = working_df.loc[mis_mask, predictor_cols].to_numpy(dtype=float)

                    if self.imputation_method == "pmm" and obs_mask.sum() >= 2:
                        X_obs = working_df.loc[obs_mask, predictor_cols].to_numpy(dtype=float)
                        y_obs = df.loc[obs_mask, target].to_numpy(dtype=float)
                        preds_obs = model.predict(X_obs)
                        preds_mis = model.predict(X_mis)
                        imputed = self._pmm_draw(preds_obs, y_obs, preds_mis, rng)
                    elif self.imputation_method == "bayesian_ridge" and isinstance(
                        model, BayesianRidge
                    ):
                        preds, std = model.predict(X_mis, return_std=True)
                        imputed = rng.normal(preds, np.maximum(1e-6, std))
                    else:
                        imputed = model.predict(X_mis)

                    df.loc[mis_mask, target] = imputed
                    working_df.loc[mis_mask, target] = imputed
                else:
                    df.loc[mis_mask, target] = self.col_medians_.get(target, 0.0)

            # Final check for any remaining NaNs
            for col in df.columns:
                if df[col].isna().any():
                    df[col] = df[col].fillna(self.col_medians_.get(col, 0.0))

            imputed_dfs.append(df)

        if return_all_imputations:
            return imputed_dfs
        return imputed_dfs[0]

    def fit_transform_multiple(self, X: Union[pd.DataFrame, np.ndarray]) -> List[pd.DataFrame]:
        """Fit chained equations and generate all M stochastic multiple imputations."""
        self.fit(X)
        return self.transform(X, return_all_imputations=True)

    def _pmm_draw(
        self,
        y_pred_obs: np.ndarray,
        y_obs: np.ndarray,
        y_pred_mis: np.ndarray,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Match missing predictions to k closest observed donors."""
        k = min(self.n_donors, len(y_pred_obs))
        if k == 0:
            return np.zeros(len(y_pred_mis), dtype=float)

        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(y_pred_obs.reshape(-1, 1))
        _, indices = nbrs.kneighbors(y_pred_mis.reshape(-1, 1))

        chosen_offsets = rng.randint(0, k, size=len(y_pred_mis))
        selected_donor_indices = indices[np.arange(len(y_pred_mis)), chosen_offsets]
        return np.asarray(y_obs[selected_donor_indices], dtype=float)

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
