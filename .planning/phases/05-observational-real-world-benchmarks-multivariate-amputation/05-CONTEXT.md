# Phase 5: Observational Real-World Benchmarks & Multivariate Amputation - Context

**Created:** 2026-09-09
**Domain:** Empirical Evaluation, Multivariate Amputation (Schouten et al. 2018), Observational Benchmarking (CPS, NHANES)
**Goal:** Implement a principled multivariate amputation engine (`ampute_multivariate`), automated observational dataset loaders (`load_cps_wage`, `load_nhanes_biomarkers`), and a comparative benchmarking pipeline evaluating Umbra against standard imputation baselines, resolving Limitation #6 in `docs/limitations.md`.

## Background & Problem Statement

In `docs/limitations.md`:
> "6. Synthetic Data Generators vs. Observational Real-World Datasets: The validation suites in Umbra are semi-synthetic data-generating processes whose marginal distributions and missingness functions are parametrically simulated. They demonstrate algorithm behavior under controlled, mathematically known ground-truth mechanisms, but do not represent unconstrained observational datasets where the true mechanism is unknown and unverified. No claim of empirical real-world validation should be inferred from synthetic data generators alone."

In v0.2.0:
1. Missingness was introduced only via stylized synthetic DGPs in `benchmarks/dgps.py` (`MCAR`, `MAR`, `MNAR_SELECTION`, etc.) generating synthetic multivariate normal covariates.
2. There was no general-purpose multivariate amputation engine following established biostatistical standards (e.g. `mice::ampute` from Schouten et al. 2018).
3. The observational case studies in `data/processed/` lacked Python loader APIs (`load_cps_wage`, `load_nhanes_biomarkers`) and automated evaluation suites.
4. Users could not easily ampute their own empirical datasets under controlled MAR, MNAR, and MCAR patterns with customized logit odds functions (`RIGHT`, `LEFT`, `MID`, `TAIL`).

## Architectural Decisions

### D-17: Schouten et al. (2018) Multivariate Amputation (`EVAL-01`)
Implement `ampute_multivariate` in `umbra/benchmark/amputation.py`:
- Input complete matrix/DataFrame $\mathbf{X} \in \mathbb{R}^{N \times P}$.
- Parameterizes:
  - `patterns`: $K \times P$ matrix indicating incomplete variable subsets ($0 = \text{missing}$, $1 = \text{observed}$).
  - `weights`: $K \times P$ coefficients producing weighted sum scores $S_i = \sum_{j=1}^P w_{kj} X_{ij}$.
    - Under MAR: weights for incomplete variables in pattern $k$ are strictly zeroed out ($w_{kj} = 0$ if $\text{pattern}_{kj} = 0$).
    - Under MNAR: weights for incomplete variables can be non-zero (self-masking / tail dropout).
    - Under MCAR: uniform missingness assignment independent of predictor scores.
  - `odds_type`: Odds logit probability shifts:
    - `"RIGHT"`: $\sigma(S_i - \theta)$ (higher scores have higher missingness).
    - `"LEFT"`: $\sigma(-(S_i - \theta))$ (lower scores have higher missingness).
    - `"MID"`: Missingness concentrated in the middle of the distribution.
    - `"TAIL"`: Missingness concentrated in extreme tails (U-shaped dropout).
  - Returns `AmputationResult` dataclass with `data_amputed`, `data_complete`, `mask`, `patterns`, `weights`, `probabilities`, and `empirical_prop`.

### D-18: Observational Dataset Loaders (`EVAL-02`)
Implement `umbra/data/loaders.py`:
- `load_cps_wage(split='observed', as_frame=True)`: Loads Current Population Survey wage dataset with labor force participation selection.
- `load_nhanes_biomarkers(split='observed', as_frame=True)`: Loads National Health and Nutrition Examination Survey clinical biomarker dataset.
- `load_california_housing(split='observed', as_frame=True)`: Loads California Census housing dataset.
- Resilient offline fallback: Reads from `data/processed/` if present, with synthetic reference fallbacks if run in isolated pip environments.

### D-19: Comparative Observational Benchmark Pipeline (`EVAL-03`)
Implement `benchmarks/observational_benchmark.py`:
- Evaluates Complete Case Analysis, Mean Imputation, Standard MICE, Umbra Auto-Router, Heckman Selection, and Pattern Mixture.
- Metrics: RMSE, MAE, downstream regression slope error, Rubin-pooled coverage, and execution runtime.
- Produces clean tabular outputs and updates markdown benchmarks.
