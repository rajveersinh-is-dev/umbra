# Technology Stack

**Analysis Date:** 2026-09-09

## Languages

**Primary:**
- Python >=3.10, <3.13 (audited and verified on Python 3.12.14) — all core library code, diagnostics, imputers, sensitivity routines, and benchmark infrastructure.

**Secondary:**
- Shell / PowerShell — automation and test commands.
- Markdown / LaTeX — documentation, research paper manuscript (`paper/paper.md`), and mathematical specification.

## Runtime

**Environment:**
- CPython 3.10, 3.11, 3.12
- Cross-platform support: Windows, Linux, macOS.

**Package Manager:**
- `uv` / `pip`
- Build backend: `hatchling` via PEP 517 / PEP 621.
- Configuration: `pyproject.toml`.

## Frameworks

**Core:**
- `scikit-learn` (>=1.2.0) — base transformer interface (`BaseEstimator`, `TransformerMixin`), validation helpers (`check_is_fitted`), and pipeline compatibility.
- `numpy` (>=1.23.0) — array operations, linear algebra, Monte Carlo sampling.
- `scipy` (>=1.9.0) — statistical distributions, cumulative functions, KS test, Mann-Whitney U, Pearson correlation.
- `pandas` (>=1.5.0) — DataFrame structures, missing data masks, quantile binning.
- `statsmodels` (>=0.14.0) — Probit selection models, OLS second-stage estimation, Little's MCAR chi-squared statistics.

**CLI & UI:**
- `click` (>=8.0.0) — CLI argument parsing, subcommands (`diagnose`, `impute`).
- `rich` (>=12.0.0) — terminal tables, markdown rendering, styled console diagnostics.

**Testing:**
- `pytest` (>=7.2.0) — test runner with test discovery and assertion introspection.
- `pytest-cov` (>=4.1.0) — line and branch coverage analysis.

**Code Quality & Linting:**
- `ruff` (>=0.8.2) — extremely fast linter and code formatter.
- `mypy` (>=1.7.0) — static type checking with strict type annotations.

## Key Dependencies

**Critical:**
- `scikit-learn` >= 1.2.0 — estimator protocol, parameter tracking, pipeline integration.
- `statsmodels` >= 0.14.0 — econometrics models (Probit regression for Heckman first stage).
- `scipy` >= 1.9.0 — distribution functions, inverse Mills ratio tail computations.
- `pandas` >= 1.5.0 — table manipulation, missingness masks, quantile binning.
- `numpy` >= 1.23.0 — high-performance array operations and EM algorithms.

**Optional Extras:**
- `matplotlib` (>=3.6.0) & `seaborn` (>=0.12.0) — publication figure generation (`[vis]` extra).
- `torch` (>=2.0.0) — deep generative VAE imputation (`[deep]` extra).
- `nbconvert` (>=7.0.0) — notebook execution and verification (`[all]` extra).

## Configuration

**Build Configuration:**
- `pyproject.toml` — project metadata, dependency declarations, tool configurations (`[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]`).

**Type Checking Configuration:**
- `[tool.mypy]` in `pyproject.toml`:
  - `python_version = "3.10"`
  - `disallow_untyped_defs = true`
  - `disallow_incomplete_defs = true`
  - `warn_redundant_casts = true`
  - `warn_unused_ignores = true`
  - `exclude = ["notebooks/", "scripts/"]`

**Linter Configuration:**
- `[tool.ruff]` in `pyproject.toml`:
  - `line-length = 100`
  - `target-version = "py310"`
  - `select = ["E", "F", "W", "I", "UP", "B", "C4"]`

## Platform Requirements

**Development:**
- Python 3.10+ virtual environment (`.venv`).
- Standard dependencies installed via `pip install -e ".[all]"`.

**Production:**
- Standard Python wheel distribution (`dist/umbra_impute-0.2.0-py3-none-any.whl`).
- Pure-Python package with compiled C-extension dependencies provided by binary wheels (NumPy, SciPy).

---

*Stack analysis: 2026-09-09*
*Update after major dependency changes*
