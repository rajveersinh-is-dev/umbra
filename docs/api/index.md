# Umbra API Reference

Complete reference for all public classes, functions, and diagnostics in Umbra.

---

## 1. Top-Level Functions & Transformers

### `umbra.diagnose`
```python
umbra.diagnose(data, alpha=0.05, run_sensitivity=True, random_state=42) -> UmbraDiagnosticReport
```
Executes full multi-tier missingness diagnostics across all incomplete columns in `data`.

### `umbra.UmbraImputer`
```python
class umbra.UmbraImputer(
    strategy='auto',
    delta=0.0,
    shadow_cols=None,
    run_sensitivity=True,
    n_imputations=1,
    random_state=42,
    verbose=False
)
```
Scikit-learn compliant transformer implementing `fit`, `transform`, `fit_transform`, and `get_feature_names_out`.

---

## 2. Diagnostics Module (`umbra.diagnostics`)

### `littles_mcar_test`
```python
umbra.littles_mcar_test(data, alpha=0.05, ridge_reg=1e-4) -> LittleMCARResult
```
Performs Little's (1988) multivariate test of Missing Completely at Random.

### `analyze_missingness_patterns`
```python
umbra.analyze_missingness_patterns(data, alpha=0.05) -> PatternAnalysisReport
```
Evaluates covariate distribution shifts (KS test, Mann-Whitney U, Cohen's d, Cliff's delta, Chi-squared).

### `find_shadow_variables`
```python
umbra.find_shadow_variables(data, target_column, min_missingness_corr=0.12, max_partial_outcome_corr=0.15) -> AuxiliaryVariableReport
```
Screens covariates to surface candidate auxiliary variables (instruments) with first-stage $F$-statistics.

### `assess_mnar_risk`
```python
umbra.assess_mnar_risk(data, target_col, ...) -> MNARRiskReport
```
Synthesizes empirical evidence signals into an MNAR risk assessment.

---

## 3. Imputers Module (`umbra.imputers`)

### `MARChainedEquationsImputer`
```python
umbra.MARChainedEquationsImputer(
    max_iter=10, imputation_method="pmm", n_donors=5, n_imputations=1, random_state=42
)
```
MICE chained equations under MAR. Supports PMM, Bayesian Ridge, and Ridge regression.

### `HeckmanSelectionImputer`
```python
umbra.HeckmanSelectionImputer(
    target_cols=None, shadow_cols=None, stochastic=False, n_imputations=1, random_state=42
)
```
Heckman two-step selection model imputer with Inverse Mills Ratio correction.

### `PatternMixtureImputer`
```python
umbra.PatternMixtureImputer(
    delta=0.0,
    shift_type="standardized",
    target_cols=None,
    stochastic=False,
    n_imputations=1,
    random_state=42,
)
```
Pattern-mixture model imputer with sensitivity shift $\delta$.

---

## 4. Sensitivity & Uncertainty (`umbra.sensitivity`)

### `run_sensitivity_grid`
```python
umbra.run_sensitivity_grid(data, target_column, delta_grid=None, downstream_evaluator=None, ...) -> SensitivityReport
```
Sweeps a grid of sensitivity parameters and detects sign flips and significance tipping points.

### `rubins_rules`
```python
umbra.rubins_rules(point_estimates, variance_estimates, alpha=0.05) -> Dict[str, float]
```
Pools multiple imputation point estimates and calculates valid standard errors and confidence intervals.
