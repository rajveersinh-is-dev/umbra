"""
Pattern-Mixture Model Imputer for MNAR Data.

Implements Pattern-Mixture Models (Little 1993, 1994) with explicit sensitivity shift
parameters (delta).

Model:
  P(Y, R | X) = P(Y | X, R) * P(R | X)
  Observed Data (R=1): Y | (X, R=1) ~ N(X * beta, sigma^2)
  Missing Data  (R=0): Y | (X, R=0) ~ N(X * beta + delta * sigma, sigma^2)

When delta = 0, this reduces to the standard MAR baseline.
When delta != 0, it models systematic unobserved departures between responders
and non-responders with identical observed covariates.

References:
Little, R. J. A. (1993). Pattern-mixture models for multivariate incomplete data.
Journal of the American Statistical Association, 88(421), 125-134.
Hedeker, D., & Gibbons, R. D. (1997). Application of random-effects pattern-mixture
models for missing data in longitudinal studies. Psychological Methods, 2(1), 64.
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted


class PatternMixtureImputer(BaseEstimator, TransformerMixin):
    """Pattern-Mixture Model imputer with explicit MNAR sensitivity shift parameter.

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
    n_imputations : int, default=1
        Number of stochastic draws generated.
    random_state : Optional[int], default=42
        Seed for reproducibility.
    """

    def __init__(
        self,
        delta: float = 0.0,
        shift_type: str = "standardized",
        target_cols: Optional[List[str]] = None,
        stochastic: bool = False,
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

        self.delta = delta
        self.shift_type = shift_type
        self.target_cols = target_cols
        self.stochastic = stochastic
        self.n_imputations = n_imputations
        self.random_state = random_state

        # Fitted attributes
        self.models_: Dict[str, Ridge] = {}
        self.residual_sigmas_: Dict[str, float] = {}
        self.col_medians_: Dict[str, float] = {}
        self.feature_names_in_: Union[List[str], np.ndarray] = []
        self.n_features_in_: int = 0

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Any = None) -> "PatternMixtureImputer":
        """Fit regression models on observed patterns."""
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

        for target in targets:
            if (
                target not in df.columns
                or not df[target].isna().any()
                or not pd.api.types.is_numeric_dtype(df[target])
            ):
                continue

            obs_mask = df[target].notna()
            if obs_mask.sum() < 3:
                continue

            predictor_cols = [
                c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])
            ]
            if not predictor_cols:
                # If no predictors, use intercept model
                y_obs = df.loc[obs_mask, target].to_numpy(dtype=float)
                std_val = float(np.std(y_obs, ddof=1)) if len(y_obs) > 1 else 1.0
                self.residual_sigmas_[target] = max(1e-6, std_val)
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

        self.is_fitted_ = True
        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray], return_all_imputations: bool = False
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Impute missing values applying the pattern-mixture delta shift."""
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
        rng = np.random.RandomState(self.random_state)
        n_draws = max(1, self.n_imputations if self.stochastic or self.n_imputations > 1 else 1)
        imputed_dfs = []

        for draw in range(n_draws):
            df = df_base.copy()

            for target in list(self.target_cols or self.models_.keys()):
                if target not in df.columns:
                    continue

                mis_mask = df[target].isna()
                if mis_mask.sum() == 0:
                    continue

                sigma = self.residual_sigmas_.get(target, 1.0)

                if target in self.models_:
                    model = self.models_[target]
                    predictor_cols = [
                        c
                        for c in df.columns
                        if c != target and pd.api.types.is_numeric_dtype(df[c])
                    ]
                    X_mis = (
                        df.loc[mis_mask, predictor_cols]
                        .fillna(df[predictor_cols].median())
                        .to_numpy(dtype=float)
                    )
                    base_pred = model.predict(X_mis)
                else:
                    base_pred = np.full(int(mis_mask.sum()), self.col_medians_.get(target, 0.0))

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

                if self.stochastic or n_draws > 1:
                    noise = rng.normal(0, sigma, size=len(shifted_pred))
                    shifted_pred += noise

                df.loc[mis_mask, target] = shifted_pred

            # Final check for any remaining NaNs
            for col in df.columns:
                if df[col].isna().any():
                    df[col] = df[col].fillna(self.col_medians_.get(col, 0.0))

            imputed_dfs.append(df)

        if return_all_imputations:
            return imputed_dfs
        return imputed_dfs[0]

    def fit_transform_multiple(self, X: Union[pd.DataFrame, np.ndarray]) -> List[pd.DataFrame]:
        """Fit pattern mixture model and generate all M stochastic multiple imputations."""
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
