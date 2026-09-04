"""
Pattern-Mixture Model Imputer for MNAR Data.

Implements Pattern-Mixture Models (Little 1993, 1994) with explicit sensitivity shift
parameters (delta).

Model:
  P(Y, R | X) = P(Y | X, R) * P(R | X)
  Observed Data (R=1): Y | (X, R=1) ~ N(X * beta, sigma^2)
  Missing Data  (R=0): Y | (X, R=0) ~ N(X * beta + delta * sigma, sigma^2)

When delta = 0, this reduces exactly to the standard MAR baseline.
When delta != 0, it models systematic unobserved differences between responders
and non-responders with identical observed covariates.

References:
Little, R. J. A. (1993). Pattern-mixture models for multivariate incomplete data.
Journal of the American Statistical Association, 88(421), 125-134.
Hedeker, D., & Gibbons, R. D. (1997). Application of random-effects pattern-mixture
models for missing data in longitudinal studies. Psychological Methods, 2(1), 64.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge


class PatternMixtureImputer(BaseEstimator, TransformerMixin):
    """
    Pattern-Mixture Model imputer with explicit MNAR sensitivity shift parameter.

    Parameters
    ----------
    delta : float, default=0.0
        Sensitivity parameter in standard deviation units of the residual.
        - delta = 0.0 : Standard MAR assumption
        - delta > 0.0 : Missing cases are systematically higher than observed cases
        - delta < 0.0 : Missing cases are systematically lower than observed cases
    shift_type : str, default='standardized'
        - 'standardized': delta * residual_sigma added to predictions
        - 'percentage': predictions multiplied by (1 + delta)
        - 'raw': delta added directly to predictions in original units
    target_cols : Optional[List[str]], default=None
        Columns to apply pattern-mixture shift to. If None, applies to all incomplete columns.
    stochastic : bool, default=False
        If True, draws random normal residuals around the shifted mean.
    random_state : Optional[int], default=42
        Seed for reproducibility.
    """

    def __init__(
        self,
        delta: float = 0.0,
        shift_type: str = "standardized",
        target_cols: Optional[List[str]] = None,
        stochastic: bool = False,
        random_state: Optional[int] = 42,
    ):
        self.delta = delta
        self.shift_type = shift_type
        self.target_cols = target_cols
        self.stochastic = stochastic
        self.random_state = random_state
        self.models_: Dict[str, Ridge] = {}
        self.residual_sigmas_: Dict[str, float] = {}
        self.feature_names_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Fit regression models on observed patterns."""
        df = self._to_dataframe(X).copy()
        self.feature_names_ = list(df.columns)
        targets = self.target_cols or [c for c in df.columns if df[c].isna().any()]

        for target in targets:
            if target not in df.columns or not df[target].isna().any():
                continue

            obs_mask = df[target].notna()
            if obs_mask.sum() < 5:
                continue

            predictor_cols = [
                c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])
            ]
            if not predictor_cols:
                continue

            X_obs = (
                df.loc[obs_mask, predictor_cols]
                .fillna(df[predictor_cols].median())
                .to_numpy(dtype=float)
            )
            y_obs = df.loc[obs_mask, target].to_numpy(dtype=float)

            reg = Ridge(alpha=1.0)
            reg.fit(X_obs, y_obs)
            preds_obs = reg.predict(X_obs)

            residuals = y_obs - preds_obs
            sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 1.0

            self.models_[target] = reg
            self.residual_sigmas_[target] = max(1e-6, sigma)

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Impute missing values applying the pattern-mixture delta shift."""
        df = self._to_dataframe(X).copy()
        rng = np.random.RandomState(self.random_state)

        for target, model in self.models_.items():
            if target not in df.columns:
                continue

            mis_mask = df[target].isna()
            if mis_mask.sum() == 0:
                continue

            predictor_cols = [
                c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])
            ]
            X_mis = (
                df.loc[mis_mask, predictor_cols]
                .fillna(df[predictor_cols].median())
                .to_numpy(dtype=float)
            )

            base_pred = model.predict(X_mis)
            sigma = self.residual_sigmas_.get(target, 1.0)

            # Compute MNAR pattern shift
            if self.shift_type == "standardized":
                shift = self.delta * sigma
                shifted_pred = base_pred + shift
            elif self.shift_type == "percentage":
                shifted_pred = base_pred * (1.0 + self.delta)
            elif self.shift_type == "raw":
                shifted_pred = base_pred + self.delta
            else:
                raise ValueError(f"Unknown shift_type: {self.shift_type}")

            if self.stochastic:
                noise = rng.normal(0, sigma, size=len(shifted_pred))
                shifted_pred += noise

            df.loc[mis_mask, target] = shifted_pred

        return df

    def _to_dataframe(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=[f"x_{i}" for i in range(X.shape[1])])
