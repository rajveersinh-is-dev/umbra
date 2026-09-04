# Contributing to Umbra

Thank you for your interest in contributing to **Umbra**!

Umbra is dedicated to bringing honest, diagnostic-first, and sensitivity-aware missing data imputation to Python practitioners. We welcome contributions, bug reports, and research enhancements.

---

## 1. Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Raj123-0/umbra.git
   cd umbra
   ```

2. Create a virtual environment and install dependencies in editable mode:
   ```bash
   uv venv --python 3.12 .venv
   source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
   uv pip install -e ".[all]"
   ```

3. Verify tests pass:
   ```bash
   pytest tests/ -v
   ```

---

## 2. Code Quality & Standards

Before opening a pull request, verify that all checks pass:

- **Linting & Formatting**:
  ```bash
  ruff check umbra tests scripts
  ruff format --check umbra tests scripts
  ```
- **Type Checking**:
  ```bash
  mypy umbra
  ```
- **Unit Tests**:
  ```bash
  pytest tests/ --cov=umbra
  ```

---

## 3. Core Methodological Principles & Guardrails

Contributors must strictly adhere to the following principles:

1. **Honesty Over Point Precision**: Never present an MNAR-path imputed value with false confidence. Always attach risk diagnostics and sensitivity intervals.
2. **Identifiability Limit**: True MNAR is not identifiable from observed data alone. Do not claim in comments, docstrings, or documentation that any test or model "proves" or "detects" MNAR with certainty.
3. **No Metric Gaming**: Do not adjust synthetic benchmark parameters post-hoc to make MNAR methods appear artificially superior to standard MAR baselines. Report all empirical trade-offs transparently.
