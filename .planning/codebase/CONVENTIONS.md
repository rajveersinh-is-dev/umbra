# Coding Conventions

**Analysis Date:** 2026-09-09

## Naming Patterns

**Files & Directories:**
- Python modules: `snake_case.py` (e.g., `shadow_variable_finder.py`, `grid_analysis.py`).
- Test files: `tests/test_<subsystem>.py` (e.g., `test_invariants.py`, `test_sklearn_compatibility.py`).
- Benchmark files: `benchmarks/<module>.py` (e.g., `dgps.py`, `simulation_runner.py`).

**Classes:**
- `PascalCase` for all classes and dataclasses (`UmbraImputer`, `MNARRiskReport`, `CovariateShift`, `TippingPoint`).
- Custom Warning classes: `PascalCase` ending in `Warning` (`HeckmanSEWarning`, `WeakInstrumentWarning`).

**Functions & Methods:**
- Public methods: `snake_case` (e.g., `fit()`, `transform()`, `fit_transform_multiple()`, `assess_mnar_risk()`).
- Private / internal methods: `_leading_underscore_snake_case` (e.g., `_to_dataframe()`, `_compute_imr_observed()`, `_em_multivariate_normal()`).

**Variables & Attributes:**
- Local variables: `snake_case`.
- Scikit-learn estimator attributes: Trailing underscore for fitted state (`self.is_fitted_`, `self.feature_names_in_`, `self.n_features_in_`, `self.models_`, `self.diagnostics_`).
- Constant variables: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_DELTA_GRID`, `STOCK_YOGO_WEAK_BENCHMARK = 10.0`).

## Code Style & Formatting

**Linter & Formatter:**
- Tool: `ruff` (>=0.8.2) configured in `pyproject.toml`.
- Formatting rules:
  - `line-length = 100`
  - `target-version = "py310"`
  - `select = ["E", "F", "W", "I", "UP", "B", "C4"]`
- Run commands:
  ```bash
  ruff check umbra/ tests/ benchmarks/
  ruff format --check umbra/ tests/ benchmarks/
  ```

**Type Annotations:**
- Tool: `mypy` (strict mode).
- Rules:
  - `disallow_untyped_defs = true`
  - `disallow_incomplete_defs = true`
  - All public and internal functions in `umbra/` must be fully typed.
  - Polymorphic methods (`transform`, `fit_transform`) use `@overload` signatures and generic `TypeVar("ArrayOrDataFrame", pd.DataFrame, np.ndarray)` to preserve DataFrame vs ndarray return types.
- Run command:
  ```bash
  mypy umbra/
  ```

## Import Organization

**Import Order:**
1. Standard library imports (`import warnings`, `import math`, `from dataclasses import dataclass`, `from typing import ...`).
2. Third-party library imports (`import numpy as np`, `import pandas as pd`, `import scipy.stats as stats`, `import statsmodels.api as sm`, `from sklearn.base import ...`).
3. Local / first-party imports (`from umbra.diagnostics.mnar_risk_score import ...`, `from umbra.explain import ...`).

**Import Conventions:**
- No circular imports.
- Explicit `__all__` list exported in module `__init__.py` files.
- Grouping: Enforced via `ruff check --select I`.

## Error Handling & Warnings

**Input Validation:**
- Validate required input types: raise `TypeError` with informative messages if categorical non-numeric features are passed to `UmbraImputer` without encoding.
- Raise `ValueError` if invalid argument combinations or unknown strategy strings are provided.
- Estimator state validation: Enforce `sklearn.utils.validation.check_is_fitted(self, "is_fitted_")` before running inference or retrieving post-fit attributes (`explain()`, `get_sensitivity()`, `get_feature_names_out()`).

**Warning Protocol:**
- Use domain-specific warnings rather than generic prints:
  - `HeckmanSEWarning`: Emitted when Heckman second-stage models run without paired bootstrap variance propagation.
  - `WeakInstrumentWarning`: Emitted when first-stage instrument $F \le 10.0$ (Stock-Yogo benchmark).
  - `UserWarning`: Emitted when high MNAR risk is detected to caution users that point estimates cannot eliminate selection bias without sensitivity bounds.

## Mathematical Invariants

- **Imputation Preserves Observed Values:** Imputation transformers must never modify non-missing observed data points ($y_{ij}^{imp} = y_{ij}^{obs}$ whenever $R_{ij} = 1$).
- **Monotonic Sensitivity Curves:** Parameter shift operations must exhibit monotonic or non-decreasing shifts with respect to the departure magnitude $\delta$.
- **Finite Sample Bounds:** Inverse Mills Ratio computations clip latent indices $\eta \in [-30, 30]$ and apply log-space tail adjustments (`exp(logpdf - logcdf)`) for $\eta < -10$ to prevent numerical floating-point overflows.

---

*Conventions analysis: 2026-09-09*
*Update when formatting standards or coding practices change*
