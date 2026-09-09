"""
MAR Chained Equations Imputer (MICE Baseline).

Implements Multiple Imputation by Chained Equations under the Missing at Random (MAR) assumption.
Supports Predictive Mean Matching (PMM), Bayesian Ridge regression posterior draws,
and Rubin's Rules for pooling multiple imputations and calculating confidence intervals.
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted

from umbra.imputers.rubin_pooler import RubinsRulesResult, rubins_rules

__all__ = [
    "MARChainedEquationsImputer",
    "RubinsRulesResult",
    "rubins_rules",
]


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
        self.feature_names_in_: Union[List[str], np.ndarray] = []
        self.n_features_in_: int = 0

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Any = None
    ) -> "MARChainedEquationsImputer":
        """Fit chained equations models on available data."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            self.n_features_in_ = len(self.feature_names_in_)
        else:
            self.n_features_in_ = int(X.shape[1])
            if hasattr(self, "feature_names_in_"):
                delattr(self, "feature_names_in_")

        df = self._to_dataframe(X).copy()
        self.columns_ = list(df.columns)

        for col in self.columns_:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
                self.col_medians_[col] = float(df[col].median())
            else:
                self.col_medians_[col] = 0.0

        self.incomplete_cols_ = [c for c in df.columns if df[c].isna().any()]
        self.incomplete_cols_.sort(key=lambda c: df[c].isna().sum())

        if not self.incomplete_cols_:
            self.is_fitted_ = True
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

        self.is_fitted_ = True
        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray], return_all_imputations: bool = False
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Impute missing values using the fitted chained equations."""
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

    def transform_multiple(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        m: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        """Generate multiple complete imputed datasets from the fitted imputer.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Data matrix containing missing values.
        m : Optional[int], default=None
            Number of multiple imputations (M). If None, defaults to self.n_imputations
            if > 1, else 5 (standard Rubin multiple imputation minimum).
        random_state : Optional[int], default=None
            Optional random seed for reproducibility.

        Returns
        -------
        List[pd.DataFrame]
            List of M complete imputed pandas DataFrames.
        """
        check_is_fitted(self, "is_fitted_")
        target_m = m if m is not None else (self.n_imputations if self.n_imputations > 1 else 5)
        old_m = self.n_imputations
        old_seed = self.random_state
        try:
            self.n_imputations = target_m
            if random_state is not None:
                self.random_state = random_state
            res = self.transform(X, return_all_imputations=True)
            return res if isinstance(res, list) else [res]
        finally:
            self.n_imputations = old_m
            self.random_state = old_seed

    def fit_transform_multiple(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        m: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        """Fit chained equations and generate M stochastic multiple imputations."""
        if random_state is not None:
            self.random_state = random_state
        self.fit(X)
        return self.transform_multiple(X, m=m, random_state=random_state)

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
