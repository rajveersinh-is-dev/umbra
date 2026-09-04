# Umbra Mathematical & Scientific Claims Audit

This document audits every theoretical, statistical, and empirical claim across the Umbra repository, classifying them into formal epistemic categories to ensure rigorous scientific integrity.

---

## 1. Classification Taxonomy

- **Theorem**: Formally proven in peer-reviewed mathematical statistics literature.
- **Empirical Finding**: Demonstrated through reproducible simulation or real-world benchmark data.
- **Assumption-Dependent Claim**: True mathematically conditional on explicit, non-testable structural assumptions.
- **Heuristic**: Computationally or pragmatically motivated rule of thumb without formal asymptotic optimality guarantees.
- **Unsupported (Purged)**: Any claim lacking mathematical or empirical justification (purged from the codebase).

---

## 2. Claim-by-Claim Audit Matrix

| Major Claim in Repository | Epistemic Status | Governing Reference / Code Proof | Verification & Notes |
| :--- | :---: | :--- | :--- |
| *True MNAR cannot be identified from observed data alone.* | **Theorem** | Molenberghs et al. (2008), *Stat. Sci.* | Governs all diagnostic outputs and disclaimers. |
| *Little's MCAR test statistic follows a $\chi^2$ distribution with $\text{df} = \sum p_j - p$.* | **Theorem** | Little (1988), *JASA* | Exact algebraic df implemented in `umbra/diagnostics/mcar_test.py`. Verified in `tests/test_diagnostics.py`. |
| *Standard MICE 95% confidence interval coverage collapses to ~0% under severe MNAR.* | **Empirical Finding** | `benchmarks/results.md` (Table 1) | Replicated across 20 Monte Carlo runs ($N=2,500$); empirical coverage drops to 0.0% under self-masking. |
| *Heckman selection consistently estimates outcome parameters under joint bivariate normality and valid exclusion.* | **Assumption-Dependent Claim** | Heckman (1979), *Econometrica* | Implemented in `umbra/imputers/heckman_selection.py`. Validated in `benchmarks/results.md` (cell bias drops by 85%). |
| *Violating the exclusion restriction ($Z \to Y$ direct path) causes severe omitted variable bias in Heckman estimates.* | **Empirical Finding & Theorem** | Bound, Jaeger, & Baker (1995); `benchmarks/misspecification_results.md` | Demonstrated in misspecification battery: bias reaches $+1.087$ with 0% coverage when direct path exists. |
| *Weak auxiliary instruments ($F \le 10$) cause variance inflation and coverage loss in two-step selection models.* | **Empirical Finding & Theory** | Stock, Wright, & Yogo (2002), *JBES* | Evaluated in `benchmarks/misspecification_results.md`. Formal `WeakInstrumentWarning` raised in code. |
| *Multiple imputation standard errors must combine within- and between-imputation variance with Barnard-Rubin $\nu$.* | **Theorem** | Rubin (1987); Barnard & Rubin (1999) | Exact formulas implemented in `umbra/imputers/mar_chained_equations.py`. Verified in `tests/test_coverage.py`. |
| *The composite Missingness Concern Score thresholds ($0.35, 0.65$) separate low, medium, and high concern regimes.* | **Heuristic** | `umbra/diagnostics/mnar_risk_score.py` | Empirically calibrated decision boundary; sensitivity evaluated in `benchmarks/router_benchmark.py`. |
| *First-stage instrument relevance screening can establish a candidate auxiliary variable.* | **Heuristic** | `umbra/diagnostics/shadow_variable_finder.py` | Flags observable correlation ($F > 10$); explicitly labeled as *candidate variable*, never *proven instrument*. |
| *Umbra detects whether data are MNAR from raw data alone.* | **Unsupported (PURGED)** | *Refuted by Molenberghs et al. (2008)* | **Permanently purged.** Reframed as an *evidence-conditioned missing-data analysis policy*. |
| *Regex matching on column names ('income') indicates proven MNAR.* | **Unsupported (PURGED)** | *Refuted by statistical methodology* | **Permanently purged.** Domain regex weight set to 0.00 in composite score calculation. |

---

## 3. Epistemic Guardrails Enforced in Code

1. **No Claims of Omniscience**: `UmbraImputer` never claims to uncover ground truth under arbitrary MNAR.
2. **Mandatory Uncertainty Intervals**: All routing benchmark numbers report 95% Wilson score confidence intervals.
3. **Transparent Failure Modes**: The repository includes dedicated documentation (`docs/failure_modes.md`) and a reproducible benchmark battery (`benchmarks/misspecification_benchmark.py`) detailing exactly where each model breaks down.
