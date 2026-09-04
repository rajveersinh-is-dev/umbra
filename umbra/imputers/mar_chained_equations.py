"""
MAR Chained Equations Imputer (MICE Baseline).

Implements Multiple Imputation by Chained Equations under the Missing at Random (MAR) assumption.
Supports Predictive Mean Matching (PMM) and Bayesian / Ridge regression draws.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.neighbors import NearestNeighbors


class MARChainedEquationsImputer(BaseEstimator, TransformerMixin):
    """
    Standard MICE-style chained equations imputer under the MAR assumption.

    Parameters
    ----------
    max_iter : int, default=10
        Number of cycles of chained equations.
    imputation_method : str, default='pmm'
        Method for generating imputed values:
        - 'pmm': Predictive Mean Matching (draws from k nearest observed donors)
        - 'bayesian_ridge': Draws from posterior distribution of Bayesian Ridge model
        - 'ridge': Deterministic Ridge regression prediction
    n_donors : int, default=5
        Number of nearest neighbors for Predictive Mean Matching.
    random_state : Optional[int], default=42
        Seed for reproducibility.
    """

    def __init__(
        self,
        max_iter: int = 10,
        imputation_method: str = "pmm",
        n_donors: int = 5,
        random_state: Optional[int] = 42,
    ):
        self.max_iter = max_iter
        self.imputation_method = imputation_method
        self.n_donors = n_donors
        self.random_state = random_state
        self.columns_: List[str] = []
        self.incomplete_cols_: List[str] = []
        self.models_: Dict[str, Union[BayesianRidge, Ridge]] = {}
        self.col_medians_: Dict[str, float] = {}

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Fit chained equations models on available data."""
        df = self._to_dataframe(X)
        self.columns_ = list(df.columns)
        self.incomplete_cols_ = [c for c in df.columns if df[c].isna().any()]
        self.incomplete_cols_.sort(key=lambda c: df[c].isna().sum())

        for col in self.columns_:
            med = float(df[col].median()) if df[col].notna().any() else 0.0
            self.col_medians_[col] = med

        if not self.incomplete_cols_:
            return self

        # Initial imputation using median
        working_df = df.copy()
        for col in self.incomplete_cols_:
            working_df[col] = working_df[col].fillna(self.col_medians_[col])

        rng = np.random.RandomState(self.random_state)

        # Chained equations iterations
        for _ in range(self.max_iter):
            for target in self.incomplete_cols_:
                obs_mask = df[target].notna()
                if obs_mask.sum() < 2:
                    continue

                predictor_cols = [c for c in self.columns_ if c != target]
                if not predictor_cols:
                    continue

                X_train = working_df.loc[obs_mask, predictor_cols].to_numpy(dtype=float)
                y_train = df.loc[obs_mask, target].to_numpy(dtype=float)

                if self.imputation_method == "bayesian_ridge":
                    model = BayesianRidge()
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
                        imputed_vals = rng.normal(preds, std)
                    else:
                        imputed_vals = model.predict(X_mis)

                    working_df.loc[mis_mask, target] = imputed_vals

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Impute missing values using the fitted chained equations."""
        df = self._to_dataframe(X).copy()
        if not self.incomplete_cols_:
            return df

        rng = np.random.RandomState(self.random_state)
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
                predictor_cols = [c for c in self.columns_ if c != target]
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
                    imputed = rng.normal(preds, std)
                else:
                    imputed = model.predict(X_mis)

                df.loc[mis_mask, target] = imputed
                working_df.loc[mis_mask, target] = imputed
            else:
                # Fallback to column median if model could not be fitted
                df.loc[mis_mask, target] = self.col_medians_.get(target, 0.0)

        # Final guarantee against any remaining NaNs
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(self.col_medians_.get(col, 0.0))

        return df

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

    def _to_dataframe(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        cols = self.columns_ if self.columns_ else [f"col_{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols)
