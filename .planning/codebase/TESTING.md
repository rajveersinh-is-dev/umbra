# Testing Patterns

**Analysis Date:** 2026-09-09

## Test Framework

**Runner & Plugins:**
- Runner: `pytest` (>=7.2.0)
- Plugins:
  - `pytest-cov` (>=4.1.0) — coverage tracking and missing line reports
  - `anyio` — asynchronous testing support

**Configuration:**
- Defined in `pyproject.toml`:
  ```toml
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  python_files = ["test_*.py"]
  addopts = "-v --tb=short"
  xfail_strict = true
  ```

**Run Commands:**
```bash
# Run all tests
.venv/Scripts/python -m pytest tests/

# Run fast with short tracebacks
.venv/Scripts/python -m pytest tests/ -x -q --tb=short

# Run full test suite with branch coverage
.venv/Scripts/python -m pytest tests/ --cov=umbra --cov-report=term-missing

# Run a specific test module
.venv/Scripts/python -m pytest tests/test_router.py -v

# Run a single targeted test
.venv/Scripts/python -m pytest tests/test_invariants.py -k test_transform_preserves_observed_values
```

## Test File Organization

**Location:** All tests live under `tests/` at the repository root.

**Current Test Inventory (104 Tests Across 13 Modules):**
- `tests/test_api_strategies.py`: Explicit strategy modes (`mar`, `heckman`, `pattern_mixture`), single vs multiple imputation draws.
- `tests/test_audit_hardening_coverage.py`: Edge cases, serialization formats, exception handlers, CLI default paths.
- `tests/test_benchmarks.py`: Smoke tests for runtime performance scaling and benchmark generation.
- `tests/test_cli.py`: Click runner tests for `umbra diagnose` and `umbra impute` with and without sensitivity flags.
- `tests/test_coverage.py`: Rubin's rules pooling variance, multiple draws imputer verification, coverage benchmarking.
- `tests/test_diagnostics.py`: Little's MCAR EM test, KS/Mann-Whitney distribution shifts, shadow variable search, risk scoring.
- `tests/test_edge_cases.py`: Empty datasets, 1-row datasets, all-missing columns, zero-variance columns, singular design matrices, extreme values.
- `tests/test_explain.py`: Diagnostic narrative generation across LOW, MEDIUM, and HIGH risk badges, markdown tables.
- `tests/test_imputers.py`: Mathematical correctness of MICE, Heckman selection, pattern mixture shifts, and bootstrap SE estimation.
- `tests/test_invariants.py`: Preservation of observed data points, sensitivity report filtering, monotonicity checks.
- `tests/test_router.py`: Auto-router classification accuracy, expected dispatch matching, tail-concentration routing.
- `tests/test_sensitivity.py`: Tipping point calculations, decision threshold crossings, summary dictionary export.
- `tests/test_sklearn_compatibility.py`: Scikit-learn estimator checks, pipeline integration, cloning, unfitted method guards.

## Test Patterns & Guidelines

**Deterministic Reproducibility:**
- All synthetic tests must instantiate explicit `np.random.RandomState(seed)` instances to avoid cross-test stochastic flakiness.
- No global seeds that contaminate other tests.

**Assertion Best Practices:**
- Invariant Testing: Test mathematical properties rather than arbitrary hardcoded floating point numbers.
- Tolerances: Use `np.testing.assert_allclose(actual, desired, rtol=1e-5, atol=1e-8)` for floating point evaluations.
- Warnings Testing: Use `pytest.warns(HeckmanSEWarning)` or `pytest.warns(UserWarning)` to verify warning emission.
- Exception Testing: Use `with pytest.raises(ExpectedException, match="pattern"):` to verify defensive guards.

## Coverage Expectations

- Minimum required total coverage: **$\ge 92\%$** (current baseline: **93%**).
- Zero regression policy: Any PR or feature addition must maintain green tests, strict mypy typing, and clean ruff linting.

---

*Testing analysis: 2026-09-09*
*Update when test tooling or testing frameworks change*
