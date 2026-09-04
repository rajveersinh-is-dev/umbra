"""
Unified Scikit-Learn Native API for Umbra.

Provides `UmbraImputer`, a drop-in scikit-learn transformer that executes
statistically grounded diagnostics during `fit()` and executes honest,
MNAR-aware imputation during `transform()`.

Also exposes the standalone `diagnose(X)` function returning an
`UmbraDiagnosticReport`.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from umbra.diagnostics.mnar_risk_score import MNARRiskReport, diagnose_dataframe
from umbra.diagnostics.report import diagnose
from umbra.explain import explain_diagnostics
from umbra.imputers.heckman_selection import HeckmanSelectionImputer
from umbra.imputers.mar_chained_equations import MARChainedEquationsImputer
from umbra.imputers.pattern_mixture import PatternMixtureImputer
from umbra.sensitivity.grid_analysis import SensitivityReport, run_sensitivity_grid

__all__ = ["UmbraImputer", "diagnose"]


class UmbraImputer(BaseEstimator, TransformerMixin):
    """Scikit-learn compliant MNAR-aware missing data imputer.

    Parameters
    ----------
    strategy : str, default='auto'
        Imputation strategy:
        - 'auto': Evidence-conditioned missing-data analysis policy. Variables with observed
                  patterns compatible with MCAR/MAR are routed to chained equations (MICE).
                  Variables with empirical evidence consistent with departure from MAR
                  use Heckman selection (if a candidate auxiliary variable satisfying F > 10
                  is available) or pattern-mixture models with sensitivity exploration.
                  NOTE: This policy conditions on observable signals and stated assumptions;
                  it does NOT claim to establish true MNAR from observed data alone.
        - 'mar': Standard MICE chained equations for all variables.
        - 'heckman': Heckman selection model for incomplete variables.
        - 'pattern_mixture': Pattern-mixture model with specified delta shift.
    delta : float, default=0.0
        Sensitivity parameter for pattern-mixture models (in residual standard deviations).
    shadow_cols : Optional[Dict[str, str]], default=None
        Mapping of {target_col: shadow_var} providing exclusion restrictions for Heckman models.
        If None and strategy='auto', candidate auxiliary variables discovered by diagnostics are used.
    run_sensitivity : bool, default=True
        Whether to compute sensitivity grid analysis for variables flagged with MNAR evidence.
    n_imputations : int, default=1
        Number of stochastic multiple imputations generated.
    random_state : Optional[int], default=42
        Seed for reproducibility.
    verbose : bool, default=False
        Whether to print diagnostic summaries upon fitting.
    """

    def __init__(
        self,
        strategy: str = "auto",
        delta: float = 0.0,
        shadow_cols: Optional[Dict[str, str]] = None,
        run_sensitivity: bool = True,
        n_imputations: int = 1,
        n_bootstrap_se: int = 200,
        random_state: Optional[int] = 42,
        verbose: bool = False,
    ):
        self.strategy = strategy
        self.delta = delta
        self.shadow_cols = shadow_cols
        self.run_sensitivity = run_sensitivity
        self.n_imputations = n_imputations
        self.n_bootstrap_se = n_bootstrap_se
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Fit UmbraImputer on data:
        1. Runs empirical diagnostic battery (Little's test, covariate shift, tail dependency).
        2. Assigns appropriate imputer per column based on empirical evidence.
        3. Runs sensitivity grid analysis for MNAR-flagged columns.
        """
        # Validate parameters as per sklearn conventions
        valid_strategies = ["auto", "mar", "heckman", "pattern_mixture"]
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy '{self.strategy}'. Must be one of {valid_strategies}."
            )

        df = self._to_dataframe(X).copy()
        self.feature_names_in_ = list(df.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        self.diagnostics_: Dict[str, MNARRiskReport] = {}
        self.sensitivity_reports_: Dict[str, SensitivityReport] = {}
        self.imputers_: Dict[str, BaseEstimator] = {}
        self.routing_decisions_: Dict[str, str] = {}

        incomplete_cols = [c for c in df.columns if df[c].isna().any()]
        if not incomplete_cols:
            self.is_fitted_ = True
            return self

        non_numeric_incomplete = [
            c for c in incomplete_cols if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if non_numeric_incomplete:
            raise TypeError(
                f"UmbraImputer requires incomplete features to be numeric. Found non-numeric column(s): {non_numeric_incomplete}. "
                "Please encode categorical variables (e.g. using OrdinalEncoder, OneHotEncoder, or pd.get_dummies) before imputing."
            )

        # 1. Run empirical diagnostics
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
            mar_imputer = MARChainedEquationsImputer(
                n_imputations=self.n_imputations, random_state=self.random_state
            )
            mar_imputer.fit(df)
            self.imputers_["_all_mar"] = mar_imputer
            for col in incomplete_cols:
                self.routing_decisions_[col] = "mar_chained_equations"

        elif self.strategy == "heckman":
            heck_imputer = HeckmanSelectionImputer(
                shadow_cols=effective_shadows,
                n_imputations=self.n_imputations,
                n_bootstrap_se=self.n_bootstrap_se,
                random_state=self.random_state,
            )
            heck_imputer.fit(df)
            self.imputers_["_all_heckman"] = heck_imputer
            for col in incomplete_cols:
                self.routing_decisions_[col] = "heckman_selection"

        elif self.strategy == "pattern_mixture":
            pm_imputer = PatternMixtureImputer(
                delta=self.delta,
                n_imputations=self.n_imputations,
                random_state=self.random_state,
            )
            pm_imputer.fit(df)
            self.imputers_["_all_pattern_mixture"] = pm_imputer
            for col in incomplete_cols:
                self.routing_decisions_[col] = "pattern_mixture"

        elif self.strategy == "auto":
            # Data-driven scientific routing:
            # Under MCAR / MAR evidence -> MICE
            # Under MNAR evidence -> Heckman (if instrument exists) or Pattern Mixture
            mar_cols = []
            heckman_cols = []
            pattern_cols = []

            for col in incomplete_cols:
                col_rep = self.diagnostics_.get(col)
                if col_rep is None or col_rep.risk_level in ("LOW", "MEDIUM"):
                    # Low or moderate risk without extreme tail dependency -> MAR chained equations
                    mar_cols.append(col)
                    self.routing_decisions_[col] = "mar_chained_equations"
                else:
                    # HIGH evidence consistent with MNAR
                    if effective_shadows.get(col):
                        heckman_cols.append(col)
                        self.routing_decisions_[col] = "heckman_selection"
                    else:
                        pattern_cols.append(col)
                        self.routing_decisions_[col] = "pattern_mixture"

            # Fit MAR imputer for low/medium risk columns
            if mar_cols:
                mar_imp = MARChainedEquationsImputer(
                    n_imputations=self.n_imputations, random_state=self.random_state
                )
                mar_imp.fit(df)
                self.imputers_["mar"] = mar_imp

            # Fit Heckman imputer for high-risk columns with instruments
            if heckman_cols:
                heck_imp = HeckmanSelectionImputer(
                    target_cols=heckman_cols,
                    shadow_cols=effective_shadows,
                    n_imputations=self.n_imputations,
                    n_bootstrap_se=self.n_bootstrap_se,
                    random_state=self.random_state,
                )
                heck_imp.fit(df)
                self.imputers_["heckman"] = heck_imp

            # Fit Pattern Mixture imputer for high-risk columns without instruments
            if pattern_cols:
                pm_imp = PatternMixtureImputer(
                    delta=self.delta,
                    target_cols=pattern_cols,
                    n_imputations=self.n_imputations,
                    random_state=self.random_state,
                )
                pm_imp.fit(df)
                self.imputers_["pattern_mixture"] = pm_imp

            # Inform user if high MNAR risk was detected
            high_risk_cols = [c for c, r in self.diagnostics_.items() if r.risk_level == "HIGH"]
            if high_risk_cols:
                warnings.warn(
                    f"Umbra detected evidence consistent with Not-Missing-At-Random (MNAR) in: {high_risk_cols}. "
                    "Because true MNAR is not identifiable from observed data alone, point estimates cannot "
                    "eliminate selection bias. Inspect `imputer.sensitivity_reports_` for honest bounds.",
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

    @property
    def strategy_map_(self) -> Dict[str, str]:
        """Map of column names to selected imputation strategy shorthand."""
        out = {}
        for col, decision in self.routing_decisions_.items():
            if "heckman" in decision:
                out[col] = "heckman"
            elif "mar" in decision:
                out[col] = "mar"
            else:
                out[col] = "pattern_mixture"
        return out

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_diagnostics: bool = False,
    ) -> Union[
        pd.DataFrame,
        np.ndarray,
        Tuple[Union[pd.DataFrame, np.ndarray], Dict[str, MNARRiskReport]],
    ]:
        """Impute missing values in X."""
        check_is_fitted(self, "is_fitted_")
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

    def fit_transform_multiple(self, X: Union[pd.DataFrame, np.ndarray]) -> List[pd.DataFrame]:
        """Fit UmbraImputer and return all M stochastic multiple imputations."""
        self.fit(X)
        df_base = self._to_dataframe(X).copy()
        n_m = max(1, self.n_imputations)
        if n_m == 1:
            res = self.transform(df_base)
            return [
                res if isinstance(res, pd.DataFrame) else pd.DataFrame(res, columns=df_base.columns)
            ]

        sub_m = {}
        for key, imp in self.imputers_.items():
            if hasattr(imp, "transform"):
                try:
                    sub_m[key] = imp.transform(df_base, return_all_imputations=True)
                except Exception:
                    sub_m[key] = [imp.transform(df_base)] * n_m

        imputed_dfs = []
        for i in range(n_m):
            df_i = df_base.copy()
            for key, dfs in sub_m.items():
                imp_frame = dfs[i % len(dfs)]
                if isinstance(imp_frame, np.ndarray):
                    imp_frame = pd.DataFrame(imp_frame, columns=df_base.columns)
                for col in self.routing_decisions_:
                    if col in imp_frame.columns and imp_frame[col].notna().all():
                        df_i[col] = imp_frame[col]
            imputed_dfs.append(df_i)
        return imputed_dfs

    def explain(self):
        """Print rich diagnostic and sensitivity report to terminal."""
        explain_diagnostics(self.diagnostics_)

    def get_sensitivity(self, column: str) -> Optional[SensitivityReport]:
        """Get the sensitivity analysis report for a specific column."""
        return self.sensitivity_reports_.get(column)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        check_is_fitted(self, "is_fitted_")
        return np.asarray(self.feature_names_in_)

    def _to_dataframe(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if hasattr(self, "feature_names_in_") and self.feature_names_in_:
            cols = self.feature_names_in_
        else:
            cols = [f"col_{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols)
