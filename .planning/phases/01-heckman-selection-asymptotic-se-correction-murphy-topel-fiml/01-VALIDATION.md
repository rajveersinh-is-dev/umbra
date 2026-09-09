---
phase: "01"
slug: "heckman-selection-asymptotic-se-correction-murphy-topel-fiml"
status: draft
nyquist_compliant: true
wave_0_complete: true
created: "2026-09-09"
---

# Phase 01 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|---|---|
| **Framework** | pytest >= 7.2.0 |
| **Config file** | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| **Quick run command** | `.venv/Scripts/python -m pytest tests/test_heckman.py -x --tb=short` |
| **Full suite command** | `.venv/Scripts/python -m pytest tests/ -x --tb=short` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `.venv/Scripts/python -m pytest tests/test_heckman.py -x --tb=short`
- **After every plan wave:** Run `.venv/Scripts/python -m pytest tests/ -x --tb=short`
- **Before `/gsd-verify-work`:** Full suite must be green (115+ tests, 93%+ coverage, mypy clean, ruff clean)
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---|---|---|---|---|---|---|---|---|---|
| 01-01-01 | 01 | 1 | HECK-01 | — | N/A | unit | `.venv/Scripts/python -m pytest tests/test_heckman.py -k test_murphy_topel` | ✅ | ⬜ pending |
| 01-01-02 | 01 | 1 | HECK-01 | — | N/A | unit | `.venv/Scripts/python -m pytest tests/test_heckman.py -k test_standard_errors` | ✅ | ⬜ pending |
| 01-02-01 | 02 | 2 | HECK-02 | — | N/A | unit | `.venv/Scripts/python -m pytest tests/test_heckman.py -k test_fiml_estimation` | ✅ | ⬜ pending |
| 01-02-02 | 02 | 2 | HECK-02 | — | N/A | unit | `.venv/Scripts/python -m pytest tests/test_heckman.py -k test_fiml_fallback` | ✅ | ⬜ pending |

---

## Wave 0 Requirements

Existing test infrastructure in `tests/test_imputers.py` and `tests/test_sklearn_compatibility.py` covers basic Heckman execution. Dedicated test module `tests/test_heckman.py` will be created during Phase 1 for in-depth verification of Murphy-Topel variance matrices and FIML likelihood convergence.

---

## Manual-Only Verifications

None — all econometric properties (parameter consistency, covariance positive-definiteness, fallback warnings) are verified via automated unit and property-based pytest tests.
