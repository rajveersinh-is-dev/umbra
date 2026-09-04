"""
Unified Scikit-Learn Compatible API for Umbra.

Provides `UmbraImputer`, a drop-in scikit-learn transformer that diagnoses
missingness mechanisms during `fit()` and executes honest, MNAR-aware
imputation during `transform()`.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from umbra.diagnostics.mnar_risk_score import MNARRiskReport, diagnose_dataframe
from umbra.explain import explain_diagnostics
from umbra.imputers.heckman_selection import HeckmanSelectionImputer
from umbra.imputers.mar_chained_equations import MARChainedEquationsImputer
from umbra.imputers.pattern_mixture import PatternMixtureImputer
from umbra.sensitivity.grid_analysis import SensitivityReport, run_sensitivity_grid


class UmbraImputer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible MNAR-aware missing data imputer.

    Parameters
    ----------
    strategy : str, default='auto'
        Imputation strategy:
        - 'auto': Diagnoses each variable's missingness risk. Plausibly MCAR/MAR variables
                  are imputed using chained equations (MICE). Variables flagged as high MNAR risk
                  use Heckman selection (if shadow variables exist) or pattern-mixture models,
                  and attach sensitivity intervals.
        - 'mar': Forces standard MICE chained equations for all variables.
        - 'heckman': Uses Heckman selection models for all incomplete variables.
        - 'pattern_mixture': Uses pattern-mixture models with specified delta.
    delta : float, default=0.0
        Sensitivity parameter for pattern-mixture models (in standard deviations).
    shadow_cols : Optional[Dict[str, str]], default=None
        Mapping of {target_col: shadow_var} providing exclusion restrictions for Heckman models.
        If None and strategy='auto', promising candidates discovered by diagnostics are used.
    run_sensitivity : bool, default=True
        Whether to compute sensitivity grid analysis for medium/high MNAR risk variables.
    random_state : Optional[int], default=42
        Reproducibility seed.
    verbose : bool, default=False
        Whether to print diagnostic summaries upon fitting.
    """

    def __init__(
        self,
        strategy: str = "auto",
        delta: float = 0.0,
        shadow_cols: Optional[Dict[str, str]] = None,
        run_sensitivity: bool = True,
        random_state: Optional[int] = 42,
        verbose: bool = False,
    ):
        self.strategy = strategy
        self.delta = delta
        self.shadow_cols = shadow_cols
        self.run_sensitivity = run_sensitivity
        self.random_state = random_state
        self.verbose = verbose

        # Fitted attributes
        self.diagnostics_: Dict[str, MNARRiskReport] = {}
        self.sensitivity_reports_: Dict[str, SensitivityReport] = {}
        self.imputers_: Dict[str, BaseEstimator] = {}
        self.columns_: List[str] = []
        self.is_fitted_: bool = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """
        Fit UmbraImputer on data:
        1. Runs diagnostic battery (Little's test, covariate shift, tail dependency).
        2. Assigns appropriate imputer per column based on empirical evidence.
        3. Runs sensitivity grid analysis for MNAR-flagged columns.
        """
        df = self._to_dataframe(X).copy()
        self.columns_ = list(df.columns)
        self.diagnostics_ = {}
        self.sensitivity_reports_ = {}
        self.imputers_ = {}

        incomplete_cols = [c for c in df.columns if df[c].isna().any()]
        if not incomplete_cols:
            self.is_fitted_ = True
            return self

        # 1. Run full diagnostics
        self.diagnostics_ = diagnose_dataframe(df)

        if self.verbose:
            explain_diagnostics(self.diagnostics_)

        # Prepare shadow variables mapping
        effective_shadows = dict(self.shadow_cols or {})
        for col, rep in self.diagnostics_.items():
            if col not in effective_shadows and rep.shadow_candidate:
                effective_shadows[col] = rep.shadow_candidate

        # 2. Select and fit imputers per column
        if self.strategy == "mar":
            mar_imputer = MARChainedEquationsImputer(random_state=self.random_state)
            mar_imputer.fit(df)
            self.imputers_["_all_mar"] = mar_imputer

        elif self.strategy == "heckman":
            heck_imputer = HeckmanSelectionImputer(
                shadow_cols=effective_shadows,
                random_state=self.random_state,
            )
            heck_imputer.fit(df)
            self.imputers_["_all_heckman"] = heck_imputer

        elif self.strategy == "pattern_mixture":
            pm_imputer = PatternMixtureImputer(
                delta=self.delta,
                random_state=self.random_state,
            )
            pm_imputer.fit(df)
            self.imputers_["_all_pattern_mixture"] = pm_imputer

        elif self.strategy == "auto":
            # Hybrid routing:
            mar_cols = []
            heckman_cols = []
            pattern_cols = []

            for col in incomplete_cols:
                col_rep = self.diagnostics_.get(col)
                if col_rep is None or getattr(col_rep, "risk_level", None) == "LOW":
                    mar_cols.append(col)
                elif col_rep.risk_level in ("MEDIUM", "HIGH"):
                    if effective_shadows.get(col):
                        heckman_cols.append(col)
                    else:
                        pattern_cols.append(col)

            # Fit MAR imputer for low-risk columns
            if mar_cols:
                mar_imp = MARChainedEquationsImputer(random_state=self.random_state)
                mar_imp.fit(df)
                self.imputers_["mar"] = mar_imp

            # Fit Heckman imputer for high-risk columns with instruments
            if heckman_cols:
                heck_imp = HeckmanSelectionImputer(
                    target_cols=heckman_cols,
                    shadow_cols=effective_shadows,
                    random_state=self.random_state,
                )
                heck_imp.fit(df)
                self.imputers_["heckman"] = heck_imp

            # Fit Pattern Mixture imputer for high-risk columns without instruments
            if pattern_cols:
                pm_imp = PatternMixtureImputer(
                    delta=self.delta,
                    target_cols=pattern_cols,
                    random_state=self.random_state,
                )
                pm_imp.fit(df)
                self.imputers_["pattern_mixture"] = pm_imp

            # Warn user if any column is high risk so they know not to treat point estimates as confident
            high_risk_cols = [c for c, r in self.diagnostics_.items() if r.risk_level == "HIGH"]
            if high_risk_cols:
                warnings.warn(
                    f"Umbra detected HIGH risk of Not-Missing-At-Random (MNAR) in columns: {high_risk_cols}. "
                    "Single-value point imputation cannot eliminate selection bias. "
                    "Inspect `imputer.sensitivity_reports_` for honest bounds.",
                    UserWarning,
                )

        # 3. Compute sensitivity grids for medium/high risk columns
        if self.run_sensitivity:
            for col, rep in self.diagnostics_.items():
                if rep.risk_level in ("MEDIUM", "HIGH"):
                    self.sensitivity_reports_[col] = run_sensitivity_grid(
                        df, target_column=col, random_state=self.random_state or 42
                    )

        self.is_fitted_ = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_diagnostics: bool = False,
    ) -> Union[
        pd.DataFrame, np.ndarray, Tuple[Union[pd.DataFrame, np.ndarray], Dict[str, MNARRiskReport]]
    ]:
        """
        Impute missing values in X.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Data to impute.
        return_diagnostics : bool, default=False
            If True, returns (X_imputed, diagnostics_dict).

        Returns
        -------
        X_imputed : pd.DataFrame or np.ndarray
            Completed dataset.
        diagnostics : Dict[str, MNARRiskReport] (optional)
            Diagnostic assessments per variable.
        """
        if not self.is_fitted_:
            raise RuntimeError("UmbraImputer must be fitted before calling transform().")

        is_numpy = isinstance(X, np.ndarray)
        df = self._to_dataframe(X).copy()

        # Apply fitted imputers
        if "_all_mar" in self.imputers_:
            df = self.imputers_["_all_mar"].transform(df)
        elif "_all_heckman" in self.imputers_:
            df = self.imputers_["_all_heckman"].transform(df)
        elif "_all_pattern_mixture" in self.imputers_:
            df = self.imputers_["_all_pattern_mixture"].transform(df)
        else:
            # Auto strategy combination
            if "mar" in self.imputers_:
                df = self.imputers_["mar"].transform(df)
            if "heckman" in self.imputers_:
                df = self.imputers_["heckman"].transform(df)
            if "pattern_mixture" in self.imputers_:
                df = self.imputers_["pattern_mixture"].transform(df)

        result = df.to_numpy() if is_numpy else df

        if return_diagnostics:
            return result, self.diagnostics_
        return result

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y=None,
        return_diagnostics: bool = False,
    ):
        """Fit and transform in a single call."""
        return self.fit(X, y).transform(X, return_diagnostics=return_diagnostics)

    def explain(self):
        """Print rich diagnostic and sensitivity report to terminal."""
        explain_diagnostics(self.diagnostics_)

    def get_sensitivity(self, column: str) -> Optional[SensitivityReport]:
        """Get the sensitivity analysis report for a specific column."""
        return self.sensitivity_reports_.get(column)

    def _to_dataframe(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        cols = self.columns_ if self.columns_ else [f"col_{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols)
