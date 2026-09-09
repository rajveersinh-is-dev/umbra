---
phase: 04-cost-sensitive-risk-routing-empirical-calibration
verified: 2026-09-09T22:50:00Z
status: passed
score: 6/6 must-haves verified
---

# Phase 4: Cost-Sensitive Risk Routing & Empirical Calibration Verification Report

**Phase Goal:** Implement Bayes-optimal decision thresholds under user-specified asymmetric loss matrices, empirical probability calibration (Platt scaling & Isotonic regression), pluggable decision profiles (`conservative_mnar`, `balanced`, `permissive_mar`), and full integration into diagnostics and `UmbraImputer`.
**Verified:** 2026-09-09T22:50:00Z
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Closed-form Bayes-optimal threshold $\tau^* = \frac{C_{\text{FA}}}{C_{\text{FA}} + C_{\text{FN}}}$ strictly decreases as penalty for missed MNAR ($C_{\text{FN}}$) increases | ✓ VERIFIED | Tested in `tests/test_router_calibration.py::test_bayes_optimal_threshold_computation` |
| 2 | Decision profiles (`conservative_mnar`, `balanced`, `permissive_mar`) enforce strict threshold ordering ($\tau^*_{\text{cons}} < \tau^*_{\text{bal}} < \tau^*_{\text{perm}}$) and resolve custom loss matrices | ✓ VERIFIED | Tested in `tests/test_router_calibration.py::test_router_decision_profiles` |
| 3 | `MNARRiskCalibrator` fits Platt scaling and Isotonic regression, enforcing monotonicity and producing probabilities strictly in $[0, 1]$ with Brier and ECE metrics | ✓ VERIFIED | Tested in `tests/test_router_calibration.py::test_platt_calibrator_fit_and_predict` & `test_isotonic_calibrator_fit_and_predict` |
| 4 | JSON serialization round-trips for profiles and calibrators preserve weights, thresholds, and predictions within numerical precision | ✓ VERIFIED | Tested in `tests/test_router_calibration.py::test_router_profile_serialization` & `test_calibrator_serialization_round_trip` |
| 5 | `assess_mnar_risk` and `diagnose_dataframe` emit `calibrated_p_mnar`, `decision_profile`, `expected_loss`, and `loss_matrix` in structured reports | ✓ VERIFIED | Tested in `tests/test_router_calibration.py::test_assess_mnar_risk_with_profiles` & `test_diagnose_dataframe_with_profiles_and_calibration` |
| 6 | `UmbraImputer` routes ambiguous data differentially under `conservative_mnar` vs `permissive_mar`, exposes `calibrated_p_mnar_` and `routing_expected_losses_`, and preserves scikit-learn cloning | ✓ VERIFIED | Tested in `tests/test_router_calibration.py::test_umbra_imputer_differential_routing`, `test_umbra_imputer_custom_loss_matrix`, & `test_sklearn_cloning_and_get_params_with_profiles` |

**Score:** 6/6 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `umbra/diagnostics/risk_calibrator.py` | `MNARRiskCalibrator`, `RouterDecisionProfile`, `ROUTER_PROFILES`, `compute_bayes_optimal_threshold` | ✓ EXISTS + SUBSTANTIVE | Mathematical implementation of decision theory and calibration |
| `umbra/diagnostics/mnar_risk_score.py` | Profile-aware thresholds, calibrated probabilities, expected losses | ✓ EXISTS + SUBSTANTIVE | Extended `MNARRiskReport`, `assess_mnar_risk`, and `diagnose_dataframe` |
| `umbra/api.py` | `decision_profile`, `loss_matrix`, `calibrator` in `UmbraImputer` | ✓ EXISTS + SUBSTANTIVE | Drop-in scikit-learn transformer with cost-sensitive routing |
| `tests/test_router_calibration.py` | Comprehensive test suite | ✓ EXISTS + SUBSTANTIVE | 12 unit tests verifying all mathematical and behavioral properties |
| `umbra/__init__.py` & `umbra/diagnostics/__init__.py` | Public re-exports | ✓ EXISTS + SUBSTANTIVE | Exports calibrator and decision profiles across package hierarchy |

**Artifacts:** 5/5 verified

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `UmbraImputer.fit()` | `diagnose_dataframe()` | direct function call | ✓ WIRED | Line 146: passes `decision_profile`, `loss_matrix`, `calibrator` |
| `assess_mnar_risk()` | `MNARRiskCalibrator.calibrate()` | method call | ✓ WIRED | Line 500: computes calibrated probability for composite score |
| `assess_mnar_risk()` | `compute_expected_losses()` | function call | ✓ WIRED | Line 504: computes expected losses under active loss matrix |
| `assess_mnar_risk()` | `MNARRiskReport` | dataclass instantiation | ✓ WIRED | Line 567: attaches calibration, expected loss, and profile metadata |

**Wiring:** 4/4 connections verified

## Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| ROUT-01: Cost-sensitive decision engine with asymmetric loss matrices | ✓ SATISFIED | — |
| ROUT-02: Empirical probability calibration mapping diagnostic scores to P(MNAR) | ✓ SATISFIED | — |
| ROUT-03: Pluggable decision profiles and calibration persistence | ✓ SATISFIED | — |

**Coverage:** 3/3 requirements satisfied

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | - | None | - | No stubs, TODOs, or placeholder returns |

**Anti-patterns:** 0 found

## Human Verification Required

None — decision boundary monotonicity, probability bounds, and cost-matrix invariants verified programmatically via automated tests.

## Gaps Summary

**No gaps found.** Phase goal achieved. Ready to proceed to Phase 5.

## Verification Metadata

**Verification approach:** Goal-backward (derived from phase goal and statistical decision theory)
**Must-haves source:** `04-01-PLAN.md`, `04-02-PLAN.md`
**Automated checks:** 151 passed, 0 failed
**Human checks required:** 0
**Total verification time:** 35s

---
*Verified: 2026-09-09T22:50:00Z*
*Verifier: Antigravity*
