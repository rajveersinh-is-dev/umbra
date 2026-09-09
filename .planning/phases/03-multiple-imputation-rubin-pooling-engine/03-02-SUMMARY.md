# Plan 03-02 Summary: RubinPooler & Barnard-Rubin (1999) Small-Sample Degrees of Freedom

## Execution Details
- **Phase:** 03 (Multiple Imputation & Rubin Pooling Engine)
- **Plan:** 03-02 (Wave 2)
- **Status:** COMPLETED
- **Requirements Satisfied:** POOL-02, POOL-03

## What Changed
1. **`umbra/imputers/rubin_pooler.py`**:
   - Created `RubinsRulesResult` dataclass with complete diagnostics:
     - `pooled_estimate`, `within_variance`, `between_variance`, `total_variance`, `standard_error`.
     - `df`, `df_adjusted`, `df_complete`, `df_rubin`.
     - `fmi` (Fraction of Missing Information).
     - `relative_variance` ($r$), `relative_efficiency` ($\text{RE}$).
     - `ci_lower`, `ci_upper`, `p_value`.
   - Implemented `rubins_rules(point_estimates, variance_estimates, df_complete=None, alpha=0.05)`:
     - Exact Rubin (1987) variance pooling: $T = \bar{U} + (1 + M^{-1}) B$.
     - Barnard & Rubin (1999) small-sample adjustment:
       $$\nu_{\text{obs}} = \frac{\nu_0 + 1}{\nu_0 + 3} \nu_0 (1 - \hat{\gamma})$$
       $$\nu_{\text{adj}} = \left(\frac{1}{\nu_m} + \frac{1}{\nu_{\text{obs}}}\right)^{-1} = \frac{\nu_m \nu_{\text{obs}}}{\nu_m + \nu_{\text{obs}}}$$
       guaranteeing $\nu_{\text{adj}} \le \nu_0$ and avoiding overconfidence in small samples.
   - Implemented `RubinPooler` class:
     - `pool_estimates()`: pools scalar or vector parameter estimates.
     - `fit(imputed_dfs, target, feature_cols=None)`: fits downstream regression across $M$ imputed DataFrames and pools parameters.
     - Exposes `coef_`, `intercept_`, `stderr_`, `tvalues_`, `pvalues_`, `ci_`, `fmi_`, `df_`, `summary_`.
2. **Backward Compatibility & Exports**:
   - `umbra/imputers/mar_chained_equations.py` imports `RubinsRulesResult` and `rubins_rules` from `rubin_pooler` and re-exports them.
   - Exported `RubinPooler` and `RubinsRulesResult` in `umbra` and `umbra.imputers`.

## Verification
- Unit tests in `test_rubin_pooling.py` and `test_coverage.py` pass.
- Barnard-Rubin bounds verified: $\nu_{\text{adj}} \le \nu_0$ and $\nu_{\text{adj}} \to \nu_m$ as $\nu_0 \to \infty$.
