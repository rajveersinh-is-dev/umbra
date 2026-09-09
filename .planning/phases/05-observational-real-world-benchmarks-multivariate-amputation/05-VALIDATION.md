# Phase 5: Observational Real-World Benchmarks & Multivariate Amputation - Validation Plan

**Domain:** Multivariate Amputation, Observational Dataset Loaders, Comparative Benchmarking Suite

## Validation Criteria

### Mathematical & Algorithmic Invariants
1. **Amputation Invariants (`EVAL-01`)**:
   - Empirical missingness rate matches requested `prop` within $\pm 0.05$ tolerance across $N \ge 1000$.
   - Under MAR: weights for incomplete variables in pattern $k$ are strictly zero: $w_{kj} = 0$ if $P_{kj} = 0$.
   - Under MNAR: weights for incomplete variables can be non-zero ($w_{kj} \ne 0$ if $P_{kj} = 0$).
   - Odds types (`RIGHT`, `LEFT`, `MID`, `TAIL`) modulate missingness distribution predictably:
     - `RIGHT`: Pearson correlation $r(S, R) > 0$.
     - `LEFT`: Pearson correlation $r(S, R) < 0$.
     - `TAIL`: Missingness concentrated in extreme quantiles $|S| > q_{75}$.
2. **Dataset Loader Invariants (`EVAL-02`)**:
   - `load_cps_wage()` returns valid numeric DataFrame with $> 500$ rows, containing target income/wage and instrumental contact/children columns.
   - `load_nhanes_biomarkers()` returns valid numeric DataFrame with $> 500$ rows, containing target glucose and instrumental phlebotomy difficulty.
   - `load_california_housing()` returns valid numeric DataFrame with $> 500$ rows.
   - Splits `'observed'`, `'complete'`, and `'both'` execute cleanly without network dependency.
3. **Comparative Benchmark Invariants (`EVAL-03`)**:
   - `run_observational_benchmark()` evaluates all baselines (CCA, Mean, MICE) and Umbra models (Auto, Heckman, Pattern Mixture).
   - Generates finite, non-NaN RMSE, MAE, and slope error metrics.
   - Outputs publication-ready summary table.

### Automated Test Suite (`tests/test_amputation_and_benchmarks.py`)
- `test_ampute_multivariate_prop_and_patterns`: Verify empirical missing proportion and pattern assignment.
- `test_ampute_multivariate_mar_vs_mnar_weights`: Verify MAR strictly excludes amputed variables from weights, MNAR includes them.
- `test_ampute_multivariate_odds_types`: Verify `RIGHT`, `LEFT`, `MID`, `TAIL` shift functions.
- `test_load_cps_wage`: Verify schema, rows, missingness rate, and split options.
- `test_load_nhanes_biomarkers`: Verify schema, rows, missingness rate, and split options.
- `test_load_california_housing`: Verify schema, rows, missingness rate, and split options.
- `test_observational_benchmark_pipeline`: Run quick benchmark across observational and amputated datasets, verify all metrics finite.

### Regression Checks
- Full pytest suite (151+ tests) passes with 0 failures.
- Zero mypy type errors across all checked files.
- Zero ruff lint errors.
