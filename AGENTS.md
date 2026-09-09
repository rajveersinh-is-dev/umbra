<!-- GSD:project-start source:PROJECT.md -->

## Project

**Umbra**

Umbra is a scikit-learn compatible Python library for principled missing-data imputation and diagnostics under Missing Not at Random (MNAR) and Missing at Random (MAR) mechanisms. It provides data scientists, econometricians, and biostatisticians with mathematically principled imputers, statistical diagnostic batteries, and sensitivity analyses.

**Core Value:** Provide mathematically rigorous, scikit-learn native imputation and transparent diagnostics that resolve known econometric and statistical estimation boundaries without overclaiming identifiability.

### Constraints

- **Compatibility**: Must adhere strictly to the scikit-learn estimator/transformer API conventions.
- **Typing & Linting**: Mypy strict mode (`disallow_untyped_defs = true`) and ruff format/check must pass across all new modules.
- **Dependencies**: Rely on standard scientific stack (NumPy, SciPy, Pandas, Scikit-Learn, Statsmodels) without introducing heavy mandatory dependencies.
- **Numerical Stability**: Graceful fallbacks and defensible parameter uncertainty when design matrices are ill-conditioned.

<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

## Languages

- Python >=3.10, <3.13 (audited and verified on Python 3.12.14) — all core library code, diagnostics, imputers, sensitivity routines, and benchmark infrastructure.
- Shell / PowerShell — automation and test commands.
- Markdown / LaTeX — documentation, research paper manuscript (`paper/paper.md`), and mathematical specification.

## Runtime

- CPython 3.10, 3.11, 3.12
- Cross-platform support: Windows, Linux, macOS.
- `uv` / `pip`
- Build backend: `hatchling` via PEP 517 / PEP 621.
- Configuration: `pyproject.toml`.

## Frameworks

- `scikit-learn` (>=1.2.0) — base transformer interface (`BaseEstimator`, `TransformerMixin`), validation helpers (`check_is_fitted`), and pipeline compatibility.
- `numpy` (>=1.23.0) — array operations, linear algebra, Monte Carlo sampling.
- `scipy` (>=1.9.0) — statistical distributions, cumulative functions, KS test, Mann-Whitney U, Pearson correlation.
- `pandas` (>=1.5.0) — DataFrame structures, missing data masks, quantile binning.
- `statsmodels` (>=0.14.0) — Probit selection models, OLS second-stage estimation, Little's MCAR chi-squared statistics.
- `click` (>=8.0.0) — CLI argument parsing, subcommands (`diagnose`, `impute`).
- `rich` (>=12.0.0) — terminal tables, markdown rendering, styled console diagnostics.
- `pytest` (>=7.2.0) — test runner with test discovery and assertion introspection.
- `pytest-cov` (>=4.1.0) — line and branch coverage analysis.
- `ruff` (>=0.8.2) — extremely fast linter and code formatter.
- `mypy` (>=1.7.0) — static type checking with strict type annotations.

## Key Dependencies

- `scikit-learn` >= 1.2.0 — estimator protocol, parameter tracking, pipeline integration.
- `statsmodels` >= 0.14.0 — econometrics models (Probit regression for Heckman first stage).
- `scipy` >= 1.9.0 — distribution functions, inverse Mills ratio tail computations.
- `pandas` >= 1.5.0 — table manipulation, missingness masks, quantile binning.
- `numpy` >= 1.23.0 — high-performance array operations and EM algorithms.
- `matplotlib` (>=3.6.0) & `seaborn` (>=0.12.0) — publication figure generation (`[vis]` extra).
- `torch` (>=2.0.0) — deep generative VAE imputation (`[deep]` extra).
- `nbconvert` (>=7.0.0) — notebook execution and verification (`[all]` extra).

## Configuration

- `pyproject.toml` — project metadata, dependency declarations, tool configurations (`[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]`).
- `[tool.mypy]` in `pyproject.toml`:
- `[tool.ruff]` in `pyproject.toml`:

## Platform Requirements

- Python 3.10+ virtual environment (`.venv`).
- Standard dependencies installed via `pip install -e ".[all]"`.
- Standard Python wheel distribution (`dist/umbra_impute-0.2.0-py3-none-any.whl`).
- Pure-Python package with compiled C-extension dependencies provided by binary wheels (NumPy, SciPy).

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Naming Patterns

- Python modules: `snake_case.py` (e.g., `shadow_variable_finder.py`, `grid_analysis.py`).
- Test files: `tests/test_<subsystem>.py` (e.g., `test_invariants.py`, `test_sklearn_compatibility.py`).
- Benchmark files: `benchmarks/<module>.py` (e.g., `dgps.py`, `simulation_runner.py`).
- `PascalCase` for all classes and dataclasses (`UmbraImputer`, `MNARRiskReport`, `CovariateShift`, `TippingPoint`).
- Custom Warning classes: `PascalCase` ending in `Warning` (`HeckmanSEWarning`, `WeakInstrumentWarning`).
- Public methods: `snake_case` (e.g., `fit()`, `transform()`, `fit_transform_multiple()`, `assess_mnar_risk()`).
- Private / internal methods: `_leading_underscore_snake_case` (e.g., `_to_dataframe()`, `_compute_imr_observed()`, `_em_multivariate_normal()`).
- Local variables: `snake_case`.
- Scikit-learn estimator attributes: Trailing underscore for fitted state (`self.is_fitted_`, `self.feature_names_in_`, `self.n_features_in_`, `self.models_`, `self.diagnostics_`).
- Constant variables: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_DELTA_GRID`, `STOCK_YOGO_WEAK_BENCHMARK = 10.0`).

## Code Style & Formatting

- Tool: `ruff` (>=0.8.2) configured in `pyproject.toml`.
- Formatting rules:
- Run commands:
- Tool: `mypy` (strict mode).
- Rules:
- Run command:

## Import Organization

- No circular imports.
- Explicit `__all__` list exported in module `__init__.py` files.
- Grouping: Enforced via `ruff check --select I`.

## Error Handling & Warnings

- Validate required input types: raise `TypeError` with informative messages if categorical non-numeric features are passed to `UmbraImputer` without encoding.
- Raise `ValueError` if invalid argument combinations or unknown strategy strings are provided.
- Estimator state validation: Enforce `sklearn.utils.validation.check_is_fitted(self, "is_fitted_")` before running inference or retrieving post-fit attributes (`explain()`, `get_sensitivity()`, `get_feature_names_out()`).
- Use domain-specific warnings rather than generic prints:

## Mathematical Invariants

- **Imputation Preserves Observed Values:** Imputation transformers must never modify non-missing observed data points ($y_{ij}^{imp} = y_{ij}^{obs}$ whenever $R_{ij} = 1$).
- **Monotonic Sensitivity Curves:** Parameter shift operations must exhibit monotonic or non-decreasing shifts with respect to the departure magnitude $\delta$.
- **Finite Sample Bounds:** Inverse Mills Ratio computations clip latent indices $\eta \in [-30, 30]$ and apply log-space tail adjustments (`exp(logpdf - logcdf)`) for $\eta < -10$ to prevent numerical floating-point overflows.

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## Pattern Overview

- Drop-in scikit-learn transformer (`fit()`, `transform()`, `fit_transform()`, `fit_transform_multiple()`).
- Data-driven mechanism diagnosis (Little's MCAR test + covariate distribution shifts + tail self-censoring concentration + auxiliary shadow variable discovery).
- Honest MNAR handling: Explicit separation between point estimation and non-identifiable sensitivity bounds.
- Dual inference support: Single plug-in imputation or Rubin-pooled multiple imputation ($M \ge 5$) with Barnard-Rubin small-sample degrees of freedom adjustments.

## Layers

### 1. API & Orchestration Layer

- **Location:** `umbra/api.py`, `umbra/__init__.py`
- **Responsibilities:**

### 2. Empirical Diagnostics Layer

- **Location:** `umbra/diagnostics/`
- **Responsibilities:**

### 3. Imputation Estimators Layer

- **Location:** `umbra/imputers/`
- **Responsibilities:**

### 4. Sensitivity & Tipping-Point Layer

- **Location:** `umbra/sensitivity/`
- **Responsibilities:**

### 5. Visualization & Reporting Layer

- **Location:** `umbra/explain.py`, `umbra/visualization/`
- **Responsibilities:**

### 6. Benchmarking & Empirical Validation Layer

- **Location:** `benchmarks/`, `scripts/`
- **Responsibilities:**

## Data Flow

### 1. `UmbraImputer.fit(X)` Flow:

### 2. `UmbraImputer.transform(X)` Flow:

## Key Abstractions

- `UmbraImputer`: Primary scikit-learn transformer interface.
- `MNARRiskReport`: Comprehensive diagnostic record containing risk level, composite score, diagnostic signals, and recommended strategy.
- `SensitivityReport` & `TippingPoint`: Quantitative sensitivity container with delta grids and break-even boundaries.
- `LittleMCARResult`: Chi-squared statistic, degrees of freedom, and p-value for Little's test.
- `SyntheticBenchmark`: Benchmark scenario container with complete data, observed data, missingness mask, and ground truth parameters.

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.agents/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
