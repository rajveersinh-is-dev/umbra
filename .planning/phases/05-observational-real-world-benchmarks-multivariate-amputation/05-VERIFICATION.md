# Phase 5: Observational Real-World Benchmarks & Multivariate Amputation - Verification

**Date:** 2026-09-09
**Status:** PASSED
**Score:** 5/5

## Verification Checklist

| Requirement | Description | Status | Evidence |
|-------------|-------------|--------|----------|
| `EVAL-01` | Schouten et al. (2018) multivariate amputation engine (`ampute_multivariate`) supporting MCAR, MAR, MNAR, and continuous odds types (`RIGHT`, `LEFT`, `MID`, `TAIL`) | PASSED | `umbra/benchmark/amputation.py`, `tests/test_amputation_and_benchmarks.py::test_ampute_multivariate_*` |
| `EVAL-02` | Curated real-world and observational benchmark dataset loaders (`load_cps_wage`, `load_nhanes_biomarkers`, `load_california_housing`, `load_clinical_trial_attrition`) | PASSED | `umbra/data/loaders.py`, `tests/test_amputation_and_benchmarks.py::test_load_*` |
| `EVAL-03` | Comparative benchmark suite evaluating 6 methods across observational datasets with automated markdown reporting | PASSED | `benchmarks/observational_benchmark.py`, `benchmarks/observational_results.md` |
| Invariants | MAR zero-weight constraint on incomplete variables strictly enforced; MNAR allows non-zero weights; target prop within $\pm 0.05$ | PASSED | `test_ampute_multivariate_mar_vs_mnar_weights`, `test_ampute_multivariate_prop_and_patterns` |
| Test Suite | 100% test pass rate with 0 regressions, clean static analysis | PASSED | 161/161 tests passing across test suite |

## Benchmark Results Snapshot

| Dataset | Best Baseline Cell RMSE | Umbra Selected Strategy | Umbra Cell RMSE | Parameter Recovery Error (Delta Beta) |
|---|---|---|---|---|
| **CPS Wage** | 16,594.68 (MICE) | Heckman Selection | **13,017.55** (-21.6% RMSE) | 900.14 vs 1,378.89 (MICE) |
| **NHANES Biomarkers** | 30.82 (MICE) | Heckman Selection | **15.94** (-48.3% RMSE) | 0.25 vs 2.25 (MICE) |
| **California Housing** | 1.97 (MICE) | Pattern Mixture | **1.45** (-26.4% RMSE) | 1.61 vs 2.82 (MICE) |
| **Clinical Trial Attrition** | 9.85 (MICE) | Heckman Selection | **6.55** (-33.5% RMSE) | 0.99 vs 2.00 (MICE) |
