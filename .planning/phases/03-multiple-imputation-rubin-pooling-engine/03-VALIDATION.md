# Phase 3: Multiple Imputation & Rubin Pooling Engine - Validation Plan

**Created:** 2026-09-09
**Domain:** Multiple Imputation (Rubin 1987, Barnard-Rubin 1999)
**Confidence:** HIGH

## Verification Map

| Requirement | Validation Method | Acceptance Criteria |
|-------------|-------------------|---------------------|
| `POOL-01` | Unit test all 4 stochastic imputers | Calling `transform_multiple(X, m=5)` returns list of exactly 5 unique DataFrames with no remaining missing values |
| `POOL-02` | Unit test `RubinPooler` on scalar and vector estimates | Total variance $T = \bar{U} + (1 + 1/M)B$, $\bar{Q} = \text{mean}(Q_m)$, correctly handles scalar and 1D vector parameters |
| `POOL-03` | Analytical unit test for Barnard-Rubin (1999) | When $\nu_0$ is small (e.g. 20), $\nu_{\text{adj}} \le \nu_0 < \nu_m$; when $\nu_0 \to \infty$, $\nu_{\text{adj}} \to \nu_m$; CIs are wider with smaller $\nu_0$ |
| `POOL-04` | Simulation coverage comparison test | 95% CI coverage under Rubin pooling ($M=5$) is nominal ($\ge 90\%$) whereas single plug-in imputation undercovers |

## Test Scenarios

1. `test_imputers_transform_multiple_interface`:
   Verify `MARChainedEquationsImputer`, `HeckmanSelectionImputer`, `PatternMixtureImputer`, and `DeepGenerativeMNARImputer` all implement `transform_multiple(X, m=5)` and `fit_transform_multiple(X, m=5)`.
2. `test_rubin_pooler_scalar_and_vector`:
   Verify `RubinPooler.pool_estimates()` handles scalar point/variance estimates and vector parameter matrices.
3. `test_barnard_rubin_small_sample_adjustment`:
   Verify Barnard-Rubin $\nu_{\text{adj}}$ is bounded strictly by complete-data degrees of freedom $\nu_0$.
4. `test_rubin_pooler_model_fit`:
   Verify `RubinPooler(estimator=LinearRegression()).fit(imputed_dfs, target='y')` pools coefficients, computes SEs, $t$-stats, $p$-values, FMI, and relative efficiency.
5. `test_single_vs_multiple_imputation_coverage`:
   Run Monte Carlo simulation comparing single plug-in imputation vs Rubin pooled multiple imputation ($M=5$).
