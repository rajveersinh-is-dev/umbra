# Umbra

## What This Is

Umbra is a scikit-learn compatible Python library for principled missing-data imputation and diagnostics under Missing Not at Random (MNAR) and Missing at Random (MAR) mechanisms. It provides data scientists, econometricians, and biostatisticians with mathematically principled imputers, statistical diagnostic batteries, and sensitivity analyses.

## Core Value

Provide mathematically rigorous, scikit-learn native imputation and transparent diagnostics that resolve known econometric and statistical estimation boundaries without overclaiming identifiability.

## Requirements

### Validated

- ✓ Scikit-learn native estimator and transformer protocols (`fit`, `transform`, `get_feature_names_out`, `check_is_fitted`) — v0.2.0
- ✓ Diagnostic battery: Little's MCAR test, auxiliary shadow variable detection, covariate shift profiling, and composite MNAR risk score — v0.2.0
- ✓ Four specialized imputation engines: Heckman Selection, Pattern Mixture, MAR Chained Equations (MICE), and Deep Generative MNAR (VAE) — v0.2.0
- ✓ Sensitivity grid analysis: Tipping-point boundary identification across plausible perturbation ranges — v0.2.0
- ✓ Strict static typing (mypy `disallow_untyped_defs = true`) and zero-lint code quality (ruff) — v0.2.0

### Active

- [ ] Exact Murphy-Topel asymptotic covariance matrix correction and optional Full-Information Maximum Likelihood (FIML) for Heckman Selection (Limitation #2)
- [ ] Generalized Ridge sandwich covariance / Bayesian posterior standard errors & VIF collinearity diagnostics under near-singular design matrices (Limitation #3)
- [ ] Multi-draw stochastic imputation interface (`transform_multiple(m=5)`) and Rubin pooling engine with Barnard-Rubin (1999) small-sample degrees-of-freedom corrections (Limitation #4)
- [ ] Cost-sensitive risk decision engine and empirical Platt/Isotonic probability calibration in Auto-Router (Limitation #5)
- [ ] Empirical real-world observational benchmark suite (CPS, NHANES) and Multivariate Amputation engine (Schouten et al.) (Limitation #6)
- [ ] Partial identifiability Manski-type worst-case bounds and machine-verifiable Identifiability & Assumption Audit certificates (Limitation #1)

### Out of Scope

- Claiming non-parametric identifiability of pure latent self-masking MNAR from single-sample observed data alone without auxiliary assumptions (ruled out by Molenberghs et al. 2008)
- Mandatory GPU hardware dependencies (keep core lightweight, CPU-compatible with optional PyTorch extra)

## Context

- **Theoretical Foundation**: Grounded in Rubin (1976, 1987), Little (1988), Heckman (1979), Murphy & Topel (1985), Barnard & Rubin (1999), and Molenberghs et al. (2008).
- **Codebase Baseline**: Fully tested (115/115 tests passing, 93% coverage), strict typing, clean packaging with `uv`.
- **Limitation Tracking**: Catalogued in `docs/limitations.md`, addressing open econometric, statistical, and benchmarking challenges.

## Constraints

- **Compatibility**: Must adhere strictly to the scikit-learn estimator/transformer API conventions.
- **Typing & Linting**: Mypy strict mode (`disallow_untyped_defs = true`) and ruff format/check must pass across all new modules.
- **Dependencies**: Rely on standard scientific stack (NumPy, SciPy, Pandas, Scikit-Learn, Statsmodels) without introducing heavy mandatory dependencies.
- **Numerical Stability**: Graceful fallbacks and defensible parameter uncertainty when design matrices are ill-conditioned.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Murphy-Topel Asymptotic SE | Provides exact analytical standard errors for Heckman second stage, eliminating necessity of slow bootstrap | — Pending |
| Rubin Small-Sample Pooling | Barnard-Rubin (1999) adjusted degrees of freedom prevents undercoverage in small-sample multiple imputation | — Pending |
| Calibrated Cost-Sensitive Router | Replaces heuristic cutoffs with Bayes-optimal decision thresholds and calibrated probability scores | — Pending |
| Multivariate Amputation Engine | Enables benchmarking on authentic observational datasets using established Schouten et al. protocol | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-09-09 after initialization*
