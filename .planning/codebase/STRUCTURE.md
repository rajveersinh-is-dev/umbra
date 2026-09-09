# Codebase Structure

**Analysis Date:** 2026-09-09

## Directory Layout

```
umbra/
├── umbra/                      # Primary library package source code
│   ├── diagnostics/            # Statistical diagnostic tests & risk scoring
│   ├── imputers/               # Imputation estimators (MICE, Heckman, Pattern Mixture, VAE)
│   ├── sensitivity/            # Sensitivity analysis and tipping point calculations
│   ├── visualization/          # Publication-quality plotting and figures
│   ├── __init__.py             # Public top-level exports
│   ├── api.py                  # Scikit-learn UmbraImputer estimator interface
│   ├── cli.py                  # Click CLI interface (umbra diagnose, umbra impute)
│   └── explain.py              # Rich console explanation formatting
├── benchmarks/                 # Empirical evaluation suite and Monte Carlo runners
│   ├── dgps.py                 # Ground truth synthetic DGPs across missingness regimes
│   ├── figures/                # Benchmark figures and calibration plots
│   ├── leaderboard/            # Benchmark markdown tables and results
│   ├── performance_scaling.py  # Runtime scaling vs sample size and feature dimension
│   ├── router_benchmark.py     # Independent router classification policy benchmark
│   ├── run_all.py              # Comprehensive benchmark runner CLI
│   └── simulation_runner.py    # Monte Carlo replication evaluator & Rubin's rules pooling
├── tests/                      # Pytest suite with 104 tests (93% coverage)
├── scripts/                    # Research reproducibility and calibration scripts
├── paper/                      # Scientific paper manuscript and generated tables
├── docs/                       # Architectural documentation, guides, and limitations
├── pyproject.toml              # Build backend and dependency configuration
└── README.md                   # Quickstart, API examples, and project overview
```

## Directory Purposes

**`umbra/`:**
- **Purpose:** Core Python package source code.
- **Contains:** Python source files (`.py`) implementing estimators, tests, and models.
- **Key files:**
  - `umbra/api.py`: Main entry point providing `UmbraImputer`, `diagnose`, `diagnose_report`.
  - `umbra/cli.py`: Click CLI commands.
  - `umbra/explain.py`: Rich formatted explanations.

**`umbra/diagnostics/`:**
- **Purpose:** Empirical statistical diagnostics to detect departures from MCAR and MAR.
- **Contains:**
  - `mcar_test.py`: Vectorized Little's (1988) MCAR test via multivariate normal EM algorithm.
  - `pattern_analysis.py`: Univariate and bivariate distribution shift tests (KS, Mann-Whitney, Cohen's d).
  - `shadow_variable_finder.py`: Auxiliary instrument discovery for Heckman selection identification.
  - `mnar_risk_score.py`: Rule-based decision routing and composite MNAR risk score assessment.
  - `report.py`: `UmbraDiagnosticReport` serialization to Markdown, JSON, and HTML.

**`umbra/imputers/`:**
- **Purpose:** Algorithm implementations for MAR and MNAR imputation.
- **Contains:**
  - `mar_chained_equations.py`: MICE using Predictive Mean Matching (PMM) and Ridge regression.
  - `heckman_selection.py`: Heckman two-stage selection model with paired bootstrap standard errors.
  - `pattern_mixture.py`: Pattern-mixture sensitivity model with mean/percentage/raw delta shifts.
  - `deep_generative_mnar.py`: PyTorch-based variational autoencoder for non-linear missingness.

**`umbra/sensitivity/`:**
- **Purpose:** Sensitivity bounds and tipping-point analysis.
- **Contains:**
  - `grid_analysis.py`: Sweeps delta parameters, tracks downstream regression/mean estimates, and identifies decision-boundary tipping points.

**`benchmarks/`:**
- **Purpose:** Rigorous empirical evaluation across synthetic DGPs and baseline comparisons.
- **Contains:**
  - `dgps.py`: Ground truth generators for MCAR, MAR, MNAR-selection, MNAR-self-masking, MNAR-tails.
  - `simulation_runner.py`: Monte Carlo replication engine with Rubin's rules pooling and coverage evaluation.
  - `router_benchmark.py`: Router accuracy and dispatch precision evaluation.
  - `run_all.py`: Orchestrates full benchmark suite with dynamic leaderboard output.

**`tests/`:**
- **Purpose:** Comprehensive automated testing with 104 tests covering invariants, estimators, diagnostics, and CLI.
- **Contains:** 13 test files matching `test_*.py`.

## Key File Locations

**Estimator Interface:**
- `umbra/api.py`: Implements `UmbraImputer(BaseEstimator, TransformerMixin)`

**CLI Entry Point:**
- `umbra/cli.py`: Registered as `umbra` in `pyproject.toml`

**Configuration:**
- `pyproject.toml`: Single source of truth for build requirements, dependencies, and tooling configs (ruff, mypy, pytest)

**Documentation & Manuscripts:**
- `paper/paper.md`: JOSS/NeurIPS-style academic paper manuscript
- `docs/limitations.md`: Complete disclosure of theoretical assumptions and non-identifiability limits

## Naming Conventions

**Files:**
- `snake_case.py` for all modules (`mar_chained_equations.py`, `grid_analysis.py`).
- `test_*.py` for test modules (`test_imputers.py`, `test_invariants.py`).

**Classes:**
- `PascalCase` for classes (`UmbraImputer`, `HeckmanSelectionImputer`, `LittleMCARResult`, `SensitivityReport`).

**Variables & Functions:**
- `snake_case` for all functions and methods.
- Trailing underscore for scikit-learn fitted attributes: `self.is_fitted_`, `self.imputers_`, `self.diagnostics_`, `self.sensitivity_reports_`.
- Leading underscore for internal/private helper routines: `_em_multivariate_normal`, `_compute_imr_observed`.

## Where to Add New Code

**Adding a New Imputation Estimator:**
- Create `umbra/imputers/<new_estimator>.py` inheriting from `BaseEstimator` and `TransformerMixin`.
- Export from `umbra/imputers/__init__.py`.
- Add strategy option to `UmbraImputer` in `umbra/api.py`.
- Add unit tests in `tests/test_imputers.py`.

**Adding a New Diagnostic Test:**
- Implement the test in `umbra/diagnostics/<test_name>.py`.
- Incorporate signal into `assess_mnar_risk` in `umbra/diagnostics/mnar_risk_score.py`.
- Add test coverage in `tests/test_diagnostics.py`.

---

*Structure analysis: 2026-09-09*
*Update when directories or module layouts change*
