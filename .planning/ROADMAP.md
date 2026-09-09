# Roadmap: Umbra Limitation Solutions (v0.3.0)

## Overview

This roadmap resolves the open econometric, statistical, and benchmarking limitations documented in `docs/limitations.md`. Through 6 focused phases, Umbra will gain exact Murphy-Topel asymptotic standard errors, FIML Heckman estimation, generalized Ridge uncertainty quantification, Rubin multiple imputation pooling with Barnard-Rubin (1999) small-sample corrections, cost-sensitive probability-calibrated auto-routing, observational benchmarks with multivariate amputation, and Manski partial identifiability auditing.

## Phases

- [x] **Phase 1: Heckman Selection Asymptotic SE Correction (Murphy-Topel & FIML)** - Exact analytical two-step standard errors and full-information maximum likelihood estimation.
- [x] **Phase 2: Ill-Conditioned Design Matrices (Ridge Sandwich SE & VIF Collinearity)** - Sandwich covariance for Ridge fallback and collinearity diagnostics for selection equations.
- [x] **Phase 3: Multiple Imputation & Rubin Pooling Engine** - Multi-draw stochastic imputation interface and Rubin pooling with Barnard-Rubin small-sample degrees of freedom.
- [x] **Phase 4: Cost-Sensitive Risk Routing & Empirical Calibration** - Bayes-optimal decision thresholds under asymmetric loss matrices and Platt/Isotonic score calibration.
- [ ] **Phase 5: Observational Real-World Benchmarks & Multivariate Amputation** - Real-world dataset loaders (CPS, NHANES) and Schouten et al. multivariate amputation engine.
- [ ] **Phase 6: Identifiability & Sensitivity Auditing (Manski Bounds & Audit Certificates)** - Worst-case partial identifiability bounds and machine-verifiable assumption certificates.

## Phase Details

### Phase 1: Heckman Selection Asymptotic SE Correction (Murphy-Topel & FIML)

**Goal**: Implement exact analytical Murphy & Topel (1985) two-step asymptotic covariance correction and optional Full-Information Maximum Likelihood (FIML) for Heckman selection imputation.
**Mode**: mvp
**Depends on**: Nothing (first phase)
**Requirements**: HECK-01, HECK-02
**Success Criteria** (what must be TRUE):

  1. `HeckmanSelectionImputer` calculates asymptotic standard errors matching Murphy & Topel (1985) without requiring bootstrap resampling.
  2. Full-Information Maximum Likelihood (`method='fiml'`) converges on bivariate normal selection problems and produces consistent estimates and joint parameter covariance.
  3. All tests pass with strict type compliance and zero regression in existing test suite.

**Plans**: 2/2 plans executed

Plans:

- [x] 01-01-PLAN.md
- [x] 01-02-PLAN.md
- [x] 01-01: Implement Murphy-Topel analytical asymptotic covariance correction in `HeckmanSelectionImputer`.
- [x] 01-02: Implement Full-Information Maximum Likelihood (FIML) Heckman estimation and selection model switcher.

### Phase 2: Ill-Conditioned Design Matrices (Ridge Sandwich SE & VIF Collinearity)

**Goal**: Resolve uncertainty quantification when design matrices are near-singular or exclusion restrictions are collinear with the inverse Mills ratio.
**Mode**: mvp
**Depends on**: Phase 1
**Requirements**: HECK-03, HECK-04
**Success Criteria** (what must be TRUE):

  1. Ridge fallback produces valid sandwich covariance and standard errors rather than setting standard errors to NaN.
  2. Condition number and Variance Inflation Factor (VIF) diagnostics detect and report collinearity between regressors and the inverse Mills ratio.
  3. Paired bootstrap standard errors execute cleanly without crashing even when Ridge regularization is active.

**Plans**: 1/1 plan executed

Plans:

- [x] 02-01-PLAN.md
- [x] 02-01: Implement Ridge regularized sandwich covariance and VIF/condition index diagnostics for Heckman selection models.

### Phase 3: Multiple Imputation & Rubin Pooling Engine

**Goal**: Enable multi-draw stochastic imputation ($M \ge 5$) across all stochastic imputers and pool parameters via Rubin's rules with Barnard-Rubin (1999) small-sample corrections.
**Mode**: mvp
**Depends on**: Phase 1
**Requirements**: POOL-01, POOL-02, POOL-03, POOL-04
**Success Criteria** (what must be TRUE):

  1. All stochastic imputers implement `transform_multiple(X, m=5, random_state=...)` returning $M$ complete imputed datasets.
  2. `RubinPooler` combines point and variance estimates across $M$ draws, computing fraction of missing information (FMI) and Barnard-Rubin (1999) degrees of freedom.
  3. Simulation runner benchmarks report Rubin-pooled coverage and confidence interval widths alongside single-imputation metrics.

**Plans**: 3/3 plans executed

Plans:

- [x] 03-01-PLAN.md
- [x] 03-02-PLAN.md
- [x] 03-03-PLAN.md
- [x] 03-01: Add `transform_multiple()` across all stochastic imputers in Umbra.
- [x] 03-02: Implement `RubinPooler` with Barnard-Rubin (1999) small-sample degrees-of-freedom corrections.
- [x] 03-03: Update benchmarking simulation runner to compute multi-imputation coverage.

### Phase 4: Cost-Sensitive Risk Routing & Empirical Calibration

**Goal**: Upgrade the auto-router with Bayes-optimal decision thresholds under user-specified loss matrices and empirical Platt/Isotonic score calibration.
**Mode**: mvp
**Depends on**: Phase 1
**Requirements**: ROUT-01, ROUT-02, ROUT-03
**Success Criteria** (what must be TRUE):

  1. Users can supply an asymmetric cost/loss matrix for false-negative MNAR vs unnecessary sensitivity exploration.
  2. Diagnostic composite scores are empirically calibrated to represent valid posterior probabilities $P(\text{MNAR} \mid \text{diagnostics})$.
  3. Pluggable decision profiles (`conservative_mnar`, `balanced`, `permissive_mar`) are available with serialization support.

**Plans**: 2/2 plans executed

Plans:

- [x] 04-01-PLAN.md
- [x] 04-02-PLAN.md
- [x] 04-01: Implement cost-sensitive decision engine with custom loss matrices.
- [x] 04-02: Implement Platt and Isotonic empirical probability calibration for MNAR risk scores.

### Phase 5: Observational Real-World Benchmarks & Multivariate Amputation

**Goal**: Build a realistic evaluation suite using authentic observational datasets (CPS, NHANES) and multivariate amputation (Schouten et al. 2018).
**Mode**: mvp
**Depends on**: Phase 3
**Requirements**: EVAL-01, EVAL-02, EVAL-03
**Success Criteria** (what must be TRUE):

  1. `ampute_multivariate` implements MAR and MNAR logistic and probit amputation patterns per established econometric and biostatistical standards.
  2. Automated dataset loaders cleanly fetch and parse observational datasets (CPS wage data, NHANES biomarkers).
  3. Comparative benchmarks evaluate Umbra against standard baselines across diverse real and amputed missingness regimes.

**Plans**: 2/2 plans executed

Plans:

- [x] 05-01-PLAN.md
- [x] 05-02-PLAN.md
- [x] 05-01: Implement `ampute_multivariate` engine supporting MAR and MNAR mechanism patterns.
- [x] 05-02: Add observational dataset loaders and comparative benchmark suite.

### Phase 6: Identifiability & Sensitivity Auditing (Manski Bounds & Audit Certificates)

**Goal**: Provide transparent partial identifiability intervals (Manski bounds) and machine-verifiable Identifiability & Assumption Audit certificates.
**Mode**: mvp
**Depends on**: Phase 4
**Requirements**: IDENT-01, IDENT-02
**Success Criteria** (what must be TRUE):

  1. `UmbraDiagnosticReport` emits Manski worst-case bounds for missing variables under zero untestable assumptions.
  2. Diagnostic reports include an Identifiability & Assumption Audit certificate detailing what testable implications hold and what requires domain assumptions.

**Plans**: 2 plans

Plans:

- [ ] 06-01: Implement Manski partial identifiability bounds in diagnostic suite.
- [ ] 06-02: Implement machine-verifiable Identifiability & Assumption Audit certificate in `UmbraDiagnosticReport`.

---
*Roadmap defined: 2026-09-09*
*Last updated: 2026-09-09 after initialization*
