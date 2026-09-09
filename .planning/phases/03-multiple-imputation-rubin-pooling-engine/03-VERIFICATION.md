---
phase: 03-multiple-imputation-rubin-pooling-engine
verified: 2026-09-09T22:42:00Z
status: passed
score: 6/6 must-haves verified
---

# Phase 3: Multiple Imputation & Rubin Pooling Engine (Rubin MI vs Single Plug-in) Verification Report

**Phase Goal:** Standardize multi-draw stochastic imputation across all imputers, implement Rubin's combination rules engine (`rubins_rules`, `RubinPooler`) with between/within variance decomposition, and implement Barnard & Rubin (1999) small-sample adjusted degrees of freedom with FMI and relative efficiency diagnostics.
**Verified:** 2026-09-09T22:42:00Z
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Multi-draw stochastic interface (`transform_multiple`, `fit_transform_multiple`) returns $M$ unique, properly-dimensioned datasets across all 4 imputers | ✓ VERIFIED | Tested in `tests/test_rubin_pooling.py::test_transform_multiple_stochasticity_across_imputers` |
| 2 | Rubin's combination rules pool point estimates and decompose total variance into within-imputation $\bar{U}$ and between-imputation $B$ | ✓ VERIFIED | Tested in `tests/test_rubin_pooling.py::test_rubins_rules_analytical_accuracy` |
| 3 | Barnard & Rubin (1999) small-sample adjustment bounds degrees of freedom ($\nu_{\text{adj}} \le \nu_0$) under small complete-sample df | ✓ VERIFIED | Tested in `tests/test_rubin_pooling.py::test_barnard_rubin_small_sample_dof` |
| 4 | Large sample asymptotic recovery converges to standard Rubin (1987) large-sample degrees of freedom ($\nu_{\text{adj}} \to \nu_m$) as $\nu_0 \to \infty$ | ✓ VERIFIED | Tested in `tests/test_rubin_pooling.py::test_rubins_rules_large_sample_recovery` |
| 5 | Single plug-in imputation produces anti-conservative confidence intervals compared to Rubin pooling which reflects between-imputation uncertainty | ✓ VERIFIED | Tested in `tests/test_rubin_pooling.py::test_rubin_vs_single_plugin_interval_coverage` |
| 6 | `RubinPooler` pools scalar and vector estimators and downstream model fits across $M$ imputed datasets | ✓ VERIFIED | Tested in `tests/test_rubin_pooling.py::test_rubin_pooler_scalar_and_vector` & `test_rubin_pooler_downstream_estimator` |

**Score:** 6/6 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `umbra/imputers/rubin_pooler.py` | `rubins_rules`, `RubinsRulesResult`, `RubinPooler` | ✓ EXISTS + SUBSTANTIVE | Full implementation of Rubin (1987) & Barnard & Rubin (1999) pooling |
| `umbra/imputers/mar_chained_equations.py` | `transform_multiple`, `fit_transform_multiple` | ✓ EXISTS + SUBSTANTIVE | Multi-draw stochastic draws from predictive distributions |
| `umbra/imputers/heckman_selection.py` | `transform_multiple`, `fit_transform_multiple` | ✓ EXISTS + SUBSTANTIVE | Multi-draw stochastic perturbation draws |
| `umbra/imputers/pattern_mixture.py` | `transform_multiple`, `fit_transform_multiple` | ✓ EXISTS + SUBSTANTIVE | Multi-draw stochastic draws from delta-shifted distributions |
| `umbra/imputers/deep_generative_mnar.py` | `transform_multiple`, `fit_transform_multiple` | ✓ EXISTS + SUBSTANTIVE | Reparameterized latent stochastic draws $z \sim \mathcal{N}(\mu, \sigma^2)$ |
| `tests/test_rubin_pooling.py` | Unit test suite | ✓ EXISTS + SUBSTANTIVE | 8 unit tests covering all mathematical properties and edge cases |
| `umbra/__init__.py` & `umbra/imputers/__init__.py` | Public exports | ✓ EXISTS + SUBSTANTIVE | Exports `rubins_rules`, `RubinsRulesResult`, `RubinPooler` |

**Artifacts:** 7/7 verified

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `RubinPooler.pool()` | `rubins_rules()` | function call | ✓ WIRED | Invokes `rubins_rules` across scalar/vector point and variance estimates |
| `RubinPooler.fit_pool()` | Imputer `transform_multiple()` | method call | ✓ WIRED | Imputes $M$ datasets, fits estimator, and pools parameters via `rubins_rules` |
| `umbra.rubins_rules` | `umbra.imputers.rubin_pooler.rubins_rules` | re-export | ✓ WIRED | Accessible at top-level `umbra` namespace |

**Wiring:** 3/3 connections verified

## Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| POOL-01: Multi-draw stochastic imputation interface across all imputers | ✓ SATISFIED | — |
| POOL-02: Rubin's combination rules engine with variance decomposition | ✓ SATISFIED | — |
| POOL-03: Barnard-Rubin small-sample df adjustment and FMI diagnostics | ✓ SATISFIED | — |

**Coverage:** 3/3 requirements satisfied

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | - | None | - | No stubs, TODOs, or placeholder returns |

**Anti-patterns:** 0 found

## Human Verification Required

None — mathematical invariants, small-sample degrees of freedom bounds, and between-imputation variance properties verified programmatically via automated tests.

## Gaps Summary

**No gaps found.** Phase goal achieved. Ready to proceed to Phase 4.

## Verification Metadata

**Verification approach:** Goal-backward (derived from phase goal and statistical MI theory)
**Must-haves source:** `03-01-PLAN.md`, `03-02-PLAN.md`, `03-03-PLAN.md`
**Automated checks:** 139 passed, 0 failed
**Human checks required:** 0
**Total verification time:** 25s

---
*Verified: 2026-09-09T22:42:00Z*
*Verifier: Antigravity*
