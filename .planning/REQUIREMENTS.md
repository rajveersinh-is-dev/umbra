# Requirements: Umbra Limitation Solutions (v0.3.0)

**Defined:** 2026-09-09
**Core Value:** Provide mathematically rigorous, scikit-learn native imputation and transparent diagnostics that resolve known econometric and statistical estimation boundaries without overclaiming identifiability.

## v1 Requirements

Requirements for resolving open limitations in `docs/limitations.md`.

### Heckman Selection & Second-Stage Variance (HECK)

- [ ] **HECK-01**: Exact Murphy-Topel (1985) two-step asymptotic covariance correction for Heckman second stage estimation
- [ ] **HECK-02**: Full-Information Maximum Likelihood (FIML) Heckman estimation as an alternative to two-step estimation
- [ ] **HECK-03**: Generalized Ridge sandwich covariance and Bayesian posterior standard error estimation for ill-conditioned design matrices
- [ ] **HECK-04**: Condition number and VIF collinearity diagnostics for the inverse Mills ratio regressor against covariates

### Multiple Imputation & Rubin Pooling (POOL)

- [ ] **POOL-01**: Multi-draw stochastic imputation interface (`transform_multiple(X, m=5, random_state=...)`) across stochastic imputers
- [ ] **POOL-02**: `RubinPooler` combining $M$ completed datasets with within- and between-imputation variance pooling
- [ ] **POOL-03**: Barnard & Rubin (1999) small-sample degrees-of-freedom adjustment for Rubin-pooled standard errors and hypothesis testing
- [ ] **POOL-04**: Simulation runner support for $M \ge 5$ multiple imputation coverage and interval width metrics

### Calibrated Risk Routing & Cost-Sensitive Decisions (ROUT)

- [ ] **ROUT-01**: Cost-sensitive decision engine accepting user-defined asymmetric loss matrices for MNAR vs MAR misclassification
- [ ] **ROUT-02**: Empirical probability calibration (Platt scaling / isotonic regression) mapping diagnostic scores to calibrated posterior probabilities
- [ ] **ROUT-03**: Pluggable router decision profiles (`conservative_mnar`, `balanced`, `permissive_mar`) and threshold calibration persistence

### Observational Benchmarks & Multivariate Amputation (EVAL)

- [ ] **EVAL-01**: Multivariate amputation engine (`ampute_multivariate`) supporting MAR/MNAR mechanisms per Schouten et al. (2018)
- [ ] **EVAL-02**: Automated dataset loaders for observational benchmark datasets (CPS labor wage, NHANES biomarkers)
- [ ] **EVAL-03**: Comparative benchmark pipeline evaluating Umbra against standard baselines on authentic observational and amputated datasets

### Identifiability & Sensitivity Auditing (IDENT)

- [ ] **IDENT-01**: Manski-type worst-case bounds / partial identifiability intervals for missing outcome variables
- [ ] **IDENT-02**: Machine-verifiable Identifiability & Assumption Audit certificates in `UmbraDiagnosticReport`

## v2 Requirements

### Extended Econometrics & Non-Parametric Selection

- **ECON-01**: Non-parametric / semi-parametric single-index selection models (Klein & Spady / Ichimura)
- **ECON-02**: Copula-based non-Gaussian selection models

## Out of Scope

| Feature | Reason |
|---------|--------|
| Claiming pure non-parametric MNAR identifiability from observed data alone | Mathematically impossible (Molenberghs et al. 2008) — must state assumptions and provide sensitivity bounds |
| Mandatory GPU requirements | Keep Umbra lightweight, pure-Python CPU compatible |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| HECK-01 | Phase 1 | Pending |
| HECK-02 | Phase 1 | Pending |
| HECK-03 | Phase 2 | Pending |
| HECK-04 | Phase 2 | Pending |
| POOL-01 | Phase 3 | Pending |
| POOL-02 | Phase 3 | Pending |
| POOL-03 | Phase 3 | Pending |
| POOL-04 | Phase 3 | Pending |
| ROUT-01 | Phase 4 | Pending |
| ROUT-02 | Phase 4 | Pending |
| ROUT-03 | Phase 4 | Pending |
| EVAL-01 | Phase 5 | Pending |
| EVAL-02 | Phase 5 | Pending |
| EVAL-03 | Phase 5 | Pending |
| IDENT-01 | Phase 6 | Pending |
| IDENT-02 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0

---
*Requirements defined: 2026-09-09*
*Last updated: 2026-09-09 after initialization*
