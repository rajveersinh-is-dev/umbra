# Changelog

All notable changes to Umbra will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-09-04

### Added
- **Formal Epistemic Reframing**: Repositioned `strategy="auto"` as an evidence-conditioned missing-data analysis policy governed by the Molenberghs et al. (2008) Non-Identifiability Theorem.
- **Epistemic Tri-Partition**: Added explicit "What Umbra Observes / What Umbra Assumes / What Umbra Cannot Establish" boundary sections to all diagnostic summaries, markdown reports, and JSON schemas.
- **Uncertainty Quantification for Routing**: Wilson score 95% binomial confidence intervals implemented for all routing benchmark metrics (overall accuracy, false alarm, missed risk).
- **Model Misspecification Battery**: 6 boundary failure regimes (weak instruments $F \le 10$, direct exclusion restriction violations $Z \to Y$, non-normal Student-$t$ errors, nonlinear polynomial selection, U-shaped tail dropout) in `benchmarks/misspecification_benchmark.py`.
- **Negative Control Tests**: Specificity validation suite in `tests/test_negative_controls.py` ensuring noisy MCAR and strong observable MAR do not falsely escalate to selection models.
- **Single-Command Reproducibility Entry Point**: `python -m benchmarks.reproduce_all` supporting `--quick` (~30s) and `--full` (~3m) execution.
- **Comprehensive Documentation Suite**:
  - `docs/identifiability.md`: The three-tier identifiability map and mathematical analysis of why blind Manski bounds break down on unbounded variables.
  - `docs/method_selection_matrix.md`: Rigorous selection matrix mapping methods to appropriate conditions, warning signs, and failure modes.
  - `docs/claims_audit.md`: Formal categorization of all theoretical claims into Theorem, Empirical finding, Assumption-dependent claim, or Heuristic.
  - `CITATION.cff`: Academic citation metadata.
- **Degenerate Input Hardening**: 16 explicit edge cases tested in `tests/test_edge_cases.py` (constant columns, all-missing, collinear design matrices, high dimensions $p > n$, non-numeric features).

### Changed
- **Candidate Variable Terminology**: Purged all claims of "discovering" or "proving" instruments from observational data alone. Auxiliary variables are strictly labeled as candidate variables, and exclusion restrictions are documented as inherently untestable without domain knowledge.
- **Stock-Yogo Weak Instrument Screening**: Added formal `WeakInstrumentWarning` when candidate instrument first-stage $F \le 10$.
- **Router Policy Comparison**: Evaluated Umbra Auto side-by-side with Always MICE, Always Heckman, Complete-Case, and Oracle routes across all regimes.
- **Code Coverage**: Increased test suite to 77 tests passing at >88% coverage with zero mypy or ruff errors.
- **Rubin-Pooled Variance**: Implemented Rubin-pooled multiple imputation variance with Barnard-Rubin (1999) small-sample degrees of freedom.
- **Heckman Bootstrap Standard Errors**: Implemented paired bootstrap standard errors (`n_bootstrap_se`) propagating first-stage probit estimation uncertainty, with explicit `HeckmanSEWarning` and NaN propagation on Ridge fallback.
- **Strict Router Evaluation**: Implemented dispatch-level correctness criteria and dispatch precision metrics in router benchmarks.
- **Reproducible Artifact Generation**: Added `scripts/generate_paper_tables.py` and `scripts/calibrate_thresholds.py`.
- **Known Limitations**: Replaced self-assessment documentation with `docs/limitations.md`.

### Fixed
- Non-numeric and categorical column crashes in `shadow_variable_finder`, `mnar_risk_score`, and `pattern_mixture`.
- Regex name matching ('income') weight permanently locked to 0.00 in composite score calculation to prevent heuristic biasing of MCAR/MAR regimes.
- Synchronized package version to `0.2.0` across `pyproject.toml` and `umbra/__init__.py`.

---

## [0.1.0] - 2026-08-20

### Added
- Initial scaffold: Little's MCAR test, pattern analysis, two-step Heckman selection imputer, pattern-mixture model, and scikit-learn pipeline wrapper.
