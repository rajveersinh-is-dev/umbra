---
phase: "02"
slug: "ill-conditioned-design-matrices-ridge-sandwich-se-vif-collinearity"
status: draft
nyquist_compliant: true
wave_0_complete: true
created: "2026-09-09"
---

# Phase 02 — Validation Strategy

## Test Infrastructure
| Property | Value |
|---|---|
| **Framework** | pytest >= 7.2.0 |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `.venv/Scripts/python -m pytest tests/test_heckman_collinearity.py -x --tb=short` |
| **Full suite command** | `.venv/Scripts/python -m pytest tests/ -x --tb=short` |
| **Estimated runtime** | ~15 seconds |

## Per-Task Verification Map
| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---|---|---|---|---|---|---|
| 02-01-01 | 01 | 1 | HECK-03 | unit | `.venv/Scripts/python -m pytest tests/test_heckman_collinearity.py -k test_ridge_sandwich` | ⬜ pending |
| 02-01-02 | 01 | 1 | HECK-04 | unit | `.venv/Scripts/python -m pytest tests/test_heckman_collinearity.py -k test_collinearity` | ⬜ pending |
