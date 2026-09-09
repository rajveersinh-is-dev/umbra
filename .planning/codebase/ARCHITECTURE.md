# Architecture

**Analysis Date:** 2026-09-09

## Pattern Overview

**Overall Architecture:** Scikit-Learn Estimator Pattern with Unified Empirical Routing and Methodological Hardening.

**Key Characteristics:**
- Drop-in scikit-learn transformer (`fit()`, `transform()`, `fit_transform()`, `fit_transform_multiple()`).
- Data-driven mechanism diagnosis (Little's MCAR test + covariate distribution shifts + tail self-censoring concentration + auxiliary shadow variable discovery).
- Honest MNAR handling: Explicit separation between point estimation and non-identifiable sensitivity bounds.
- Dual inference support: Single plug-in imputation or Rubin-pooled multiple imputation ($M \ge 5$) with Barnard-Rubin small-sample degrees of freedom adjustments.

## Layers

### 1. API & Orchestration Layer
- **Location:** `umbra/api.py`, `umbra/__init__.py`
- **Responsibilities:**
  - Exposes `UmbraImputer`, `diagnose`, `diagnose_report`.
  - Coordinates diagnosis, column routing, imputer fitting, and selective column transformation.
  - Manages scikit-learn state transitions and fitted attributes (`is_fitted_`, `imputers_`, `diagnostics_`, `sensitivity_reports_`).

### 2. Empirical Diagnostics Layer
- **Location:** `umbra/diagnostics/`
- **Responsibilities:**
  - `mcar_test.py`: Implements Little's (1988) MCAR chi-squared test with vectorized EM multivariate normal parameter estimation.
  - `pattern_analysis.py`: Computes Kolmogorov-Smirnov, Mann-Whitney U, Cohen's d, and Cliff's delta across observed vs missing subsets.
  - `shadow_variable_finder.py`: Discovers candidate auxiliary instrumental variables via two-stage LPM exclusion tests and Stock-Yogo $F$-statistic benchmarks.
  - `mnar_risk_score.py`: Combines statistical signals into composite scores and categorizes risk into `LOW`, `MEDIUM`, `HIGH`.
  - `report.py`: Encapsulates multi-variable diagnostic reports with export to JSON, Markdown, and HTML.

### 3. Imputation Estimators Layer
- **Location:** `umbra/imputers/`
- **Responsibilities:**
  - `mar_chained_equations.py`: Multivariate Imputation by Chained Equations (MICE) using Predictive Mean Matching (PMM) and Ridge regression.
  - `heckman_selection.py`: Heckman two-stage selection model with Probit first-stage selection, Inverse Mills Ratio generation, second-stage OLS, and paired bootstrap standard errors (`n_bootstrap_se`).
  - `pattern_mixture.py`: Pattern-mixture sensitivity model applying explicit delta shifts (standardized, raw, or percentage) to unobserved units.
  - `deep_generative_mnar.py`: Variational autoencoder-based imputation for complex non-linear missingness.

### 4. Sensitivity & Tipping-Point Layer
- **Location:** `umbra/sensitivity/`
- **Responsibilities:**
  - `grid_analysis.py`: Sweeps departure parameter $\delta \in [-1.5, +1.5]$; calculates downstream metric shifts and automated tipping points where conclusions reverse or confidence intervals cross zero.

### 5. Visualization & Reporting Layer
- **Location:** `umbra/explain.py`, `umbra/visualization/`
- **Responsibilities:**
  - Terminal-based rich narrative explanations with risk badges and clear methodological guidance.
  - Publication-quality Matplotlib/Seaborn figures (`figures.py`) depicting missingness heatmaps, covariate shifts, and sensitivity curves.

### 6. Benchmarking & Empirical Validation Layer
- **Location:** `benchmarks/`, `scripts/`
- **Responsibilities:**
  - Ground-truth data generating processes across MCAR, MAR, MNAR-selection, MNAR-self-masking, and MNAR-tails regimes (`benchmarks/dgps.py`).
  - Monte Carlo simulation runner with Rubin-pooled confidence interval coverage evaluation (`benchmarks/simulation_runner.py`).
  - Auto-router policy matrix benchmark (`benchmarks/router_benchmark.py`).

## Data Flow

### 1. `UmbraImputer.fit(X)` Flow:
1. Input validation: Convert input to DataFrame, verify all incomplete features are numeric.
2. Empirical Diagnosis: Execute `diagnose_dataframe(df)`:
   - Run Little's MCAR EM test.
   - Run pattern analysis and covariate shift tests.
   - Run shadow variable search for candidate exclusion instruments.
   - Run residual tail dependency checks.
3. Strategy Resolution:
   - If `strategy == "auto"`: Route each column based on empirical evidence:
     - Clear MCAR / MAR evidence -> `mar_chained_equations`.
     - High MNAR risk + valid shadow variable ($F > 10$) -> `heckman_selection`.
     - High/Medium MNAR risk without instrument -> `pattern_mixture` + sensitivity analysis.
   - If explicit strategy (`"mar"`, `"heckman"`, `"pattern_mixture"`): Route accordingly.
4. Fit Estimators: Fit the resolved estimators on target columns.
5. Sensitivity Computation: Compute sensitivity grids for any columns diagnosed with `MEDIUM` or `HIGH` risk.
6. Set `self.is_fitted_ = True`.

### 2. `UmbraImputer.transform(X)` Flow:
1. Verify `is_fitted_`.
2. Convert input to working DataFrame.
3. For each fitted sub-imputer, call `transform()` and selectively merge ONLY the columns that were routed to that specific imputer.
4. Confirm strictly zero observed non-missing values were modified (invariant preservation).
5. Return imputed DataFrame or NumPy ndarray matching input type.

## Key Abstractions

- `UmbraImputer`: Primary scikit-learn transformer interface.
- `MNARRiskReport`: Comprehensive diagnostic record containing risk level, composite score, diagnostic signals, and recommended strategy.
- `SensitivityReport` & `TippingPoint`: Quantitative sensitivity container with delta grids and break-even boundaries.
- `LittleMCARResult`: Chi-squared statistic, degrees of freedom, and p-value for Little's test.
- `SyntheticBenchmark`: Benchmark scenario container with complete data, observed data, missingness mask, and ground truth parameters.

---

*Architecture analysis: 2026-09-09*
*Update when major subsystems or data flows change*
