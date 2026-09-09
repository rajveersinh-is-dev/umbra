# Plan 05-02: Observational Dataset Loaders & Comparative Benchmark Suite - Summary

**Executed:** 2026-09-09
**Status:** Completed & Verified
**Requirements Satisfied:** `EVAL-02`, `EVAL-03`

## Summary of Accomplishments

1. **Real-World & Semi-Synthetic Dataset Loaders**:
   - Implemented `umbra/data/loaders.py` and `umbra/data/__init__.py`:
     - `load_cps_wage`: Current Population Survey labor earnings and selection.
     - `load_nhanes_biomarkers`: NHANES clinical lab non-compliance and fasting glucose.
     - `load_california_housing`: California census real-estate income self-masking.
     - `load_clinical_trial_attrition`: Longitudinal clinical trial dropout driven by adverse events.
   - Robust path resolution: checks `UMBRA_DATA_DIR`, repo root `data/processed/`, and local working directories.
   - Deterministic offline fallback generators guaranteed to produce complete/observed splits without network access.
   - Re-exported in top-level `umbra` package and `umbra.data`.
2. **Comparative Benchmark Runner (`benchmarks/observational_benchmark.py`)**:
   - Compares 6 methods: Complete Case Analysis (CCA), Mean Imputation, Standard MICE (MAR), Umbra Auto-Router, Umbra Heckman Selection, and Umbra Pattern Mixture.
   - Computes missing cell RMSE, MAE, downstream OLS parameter recovery error $\|\hat{\beta} - \beta^*\|_2$, and runtime.
   - Evaluated across CPS, NHANES, California Housing, and Clinical Trial datasets.
   - Generates publication-ready `benchmarks/observational_results.md`.
3. **Validation & Testing**:
   - Authored `tests/test_amputation_and_benchmarks.py` with 10 comprehensive unit & integration tests covering MAR/MNAR weights, continuous odds types, data loaders, and benchmark pipelines.
