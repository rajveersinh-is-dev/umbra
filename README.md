# Umbra: MNAR-Aware Missing Data Imputation & Diagnostics

[![CI](https://github.com/Raj123-0/umbra/actions/workflows/ci.yml/badge.svg)](https://github.com/Raj123-0/umbra/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type_checked-mypy-blue.svg)](http://mypy-lang.org/)

**Umbra** (`umbra-impute` on PyPI) is an open-source Python library that does something almost no widely-used imputation tool does honestly: **diagnose when missing data is Not-Missing-At-Random (MNAR), and refuse to pretend a single confident point estimate is safe when it isn't.**

Mainstream tools (scikit-learn's `IterativeImputer`, R's `mice`, `missForest`, `fancyimpute`) implicitly assume **Missing at Random (MAR)**: that conditional on observed variables, non-response is independent of the missing value. In real data, this assumption is routinely false:
- **Income** is missing *because* it is unusually high or low.
- **Symptom severity** is missing *because* the patient felt too sick or dropped out.
- **Sensitive survey questions** are skipped *because* of the true answer.

Silently imputing under a false MAR assumption produces confident point estimates that are systematically biased.

---

## The Fundamental Identifiability Limit

> [!IMPORTANT]
> **True MNAR is mathematically unidentifiable from observed data alone.** 
> You cannot definitively distinguish "MAR after conditioning on these variables" from "MNAR" using observed data alone without untestable assumptions (an exclusion restriction, a selection model, or an explicit sensitivity parameter).
>
> **Umbra's honest response is not to promise a magic solution for MNAR, but to:**
> 1. Screen for converging signals that suggest MNAR is likely.
> 2. Surface candidate auxiliary shadow variables (instruments) to identify selection models.
> 3. Provide an automated **sensitivity analysis grid** to quantify how conclusions shift across plausible MNAR departures, instead of hiding behind a single false-confidence number.

---

## Key Features

- **Multi-Signal Diagnostic Screening**:
  - **Little's MCAR Test (1988)**: EM-based multivariate test to evaluate whether data is consistent with MCAR globally.
  - **Covariate Distribution Shifts**: Two-sample Kolmogorov-Smirnov (KS) tests and standardized effect sizes (Cohen's $d$, Cliff's $\delta$) comparing observed variables between missing vs. present rows.
  - **Residual Tail Dependency & Self-Censoring Checks**: Tests whether missingness propensity concentrates sharply at extreme predicted quantiles.
  - **Domain-Shape Heuristics**: Automatic literature-backed prior checks for sensitive columns (income, psychiatric symptoms, substance use, weight/BMI, clinical dropout) with bibliographic citations.
  - **Shadow Variable Candidate Finder**: Detects candidate auxiliary variables (instruments) that correlate with missingness but show weak conditional connection to the outcome.
- **Dual-Path Imputation**:
  - **Solid MAR Baseline**: MICE-style Chained Equations with Predictive Mean Matching (PMM) and Bayesian Ridge regression for variables passing MAR checks.
  - **Heckman Selection Imputer**: Classic two-step selection model adapted for machine learning pipelines, leveraging shadow variables to resolve selection bias.
  - **Pattern-Mixture Models**: Models responders and non-responders separately under explicit sensitivity shift parameters ($\delta$).
  - **Deep Generative MNAR (`not-MIWAE`)**: Joint variational autoencoder modeling data $X$ and missingness mask $M$ concurrently (stretch goal).
- **Automated Sensitivity Grid Analysis & Tipping Points**:
  - Sweeps a grid of plausible MNAR parameters ($\delta \in [-1.5, +1.5]$ std devs).
  - Automatically flags **tipping points** where downstream regression coefficients flip sign or lose significance.
- **Scikit-Learn Compatible**:
  - `UmbraImputer` implements `fit` and `transform`, drop-in ready for `sklearn.pipeline.Pipeline`.
- **Rich CLI & Interactive Streamlit Web App**:
  - Instant terminal audits with rich colorized summaries.
  - Web UI for drag-and-drop CSV diagnostics and interactive sensitivity curve visualization.

---

## Installation

Install from PyPI:

```bash
pip install umbra-impute
```

Or install with optional extras:

```bash
# With PyTorch for deep generative MNAR
pip install "umbra-impute[deep]"

# With Streamlit demo UI
pip install "umbra-impute[demo]"

# Full installation
pip install "umbra-impute[all]"
```

---

## Quickstart

### 1. Python Scikit-Learn API

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from umbra import UmbraImputer

# Drop-in scikit-learn transformer
imputer = UmbraImputer(strategy="auto")

# Fit diagnostics and impute
X_imputed, diagnostics = imputer.fit_transform(X, return_diagnostics=True)

# Inspect per-variable MNAR risk reports
imputer.explain()

# Access honest sensitivity bounds for flagged columns
sens_report = imputer.get_sensitivity("income")
print(sens_report.summary())

# Compatible with scikit-learn Pipelines
pipeline = Pipeline([
    ("imputer", UmbraImputer(strategy="auto")),
    ("regressor", Ridge()),
])
pipeline.fit(X_train, y_train)
```

### 2. Command-Line Interface (CLI)

```bash
# 1. Diagnose missingness mechanisms in a CSV file
umbra diagnose dataset.csv

# Export diagnostic report to Markdown
umbra diagnose dataset.csv --output-markdown audit_report.md

# 2. Impute with automated sensitivity grid
umbra impute dataset.csv --strategy auto --sensitivity --output completed_data.csv
```

### 3. Interactive Streamlit Web App

Launch the interactive UI:

```bash
streamlit run demo/app.py
```

---

## Empirical Benchmark Leaderboard

The benchmark below evaluates all imputers across ground-truth simulated datasets ($N=2,500$) with known missingness generation mechanisms:

| Regime | Method | Overall Bias | Missing Cell Bias | Missing RMSE | Total Beta Error |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MCAR** | Naive Mean | +0.001 | +0.003 | 1.599 | 0.254 |
| **MCAR** | MAR MICE (PMM) | -0.015 | -0.050 | 1.353 | 0.184 |
| **MCAR** | MAR MICE (Ridge) | -0.012 | -0.039 | 1.003 | 0.191 |
| **MCAR** | Heckman Selection | +0.166 | +0.545 | 1.141 | 0.186 |
| **MCAR** | **Umbra (Auto)** | -0.012 | -0.039 | 1.003 | 0.191 |
| **MAR** | Naive Mean | +0.120 | +0.301 | 1.610 | 0.394 |
| **MAR** | MAR MICE (PMM) | -0.024 | -0.059 | 1.360 | 0.167 |
| **MAR** | MAR MICE (Ridge) | -0.008 | -0.019 | 0.977 | 0.205 |
| **MAR** | Heckman Selection | +0.135 | +0.339 | 1.035 | 0.193 |
| **MAR** | **Umbra (Auto)** | -0.008 | -0.019 | 0.977 | 0.205 |
| **MNAR LOW** | Naive Mean | -0.374 | -0.974 | 1.823 | 0.469 |
| **MNAR LOW** | MAR MICE (PMM) | -0.130 | -0.338 | 1.400 | 0.116 |
| **MNAR LOW** | **Heckman Selection** | -0.042 | -0.108 | 0.981 | 0.158 |
| **MNAR LOW** | **Umbra (Auto)** | -0.042 | -0.108 | 0.981 | 0.158 |
| **MNAR MEDIUM**| Naive Mean | -0.618 | -1.752 | 2.220 | 0.606 |
| **MNAR MEDIUM**| MAR MICE (PMM) | -0.249 | -0.706 | 1.505 | 0.050 |
| **MNAR MEDIUM**| **Heckman Selection** | -0.068 | -0.193 | 0.952 | 0.124 |
| **MNAR MEDIUM**| **Umbra (Auto)** | -0.068 | -0.193 | 0.952 | 0.124 |
| **MNAR HIGH** | Naive Mean | -0.730 | -2.261 | 2.544 | 0.722 |
| **MNAR HIGH** | MAR MICE (PMM) | -0.388 | -1.200 | 1.695 | 0.210 |
| **MNAR HIGH** | **Heckman Selection** | -0.044 | -0.136 | 0.851 | 0.112 |
| **MNAR HIGH** | **Umbra (Auto)** | -0.044 | -0.136 | 0.851 | 0.112 |

### Key Benchmark Findings
1. **MAR Methods Break Down Under MNAR**: As MNAR severity increases, standard MICE exhibits severe bias (up to -0.388 overall, -1.200 on missing cells) because it ignores self-censoring.
2. **Heckman Selection Recovers Ground Truth**: By utilizing an instrumental shadow variable (exclusion restriction), Heckman selection cuts missing-cell bias by **85%** and reduces RMSE from 1.695 to 0.851.
3. **Umbra Auto Adaptivity**: When missingness is MCAR or MAR, Umbra routes to chained equations to preserve efficiency; when MNAR risk is high, it activates selection modeling and sensitivity bounds.

Full reproducible benchmark code and living tables are located in [`benchmarks/results.md`](benchmarks/results.md).

---

## Explicit Limitations & Guardrails

- **Shadow Variables are Statistical Candidates, Not Proved Instruments**: The `shadow_variable_finder` ranks candidate features based on empirical correlation with missingness and low conditional correlation with the outcome. Substantive domain expertise is strictly required to verify that the exclusion restriction causally holds.
- **Deep Generative MNAR is Experimental**: The `DeepGenerativeMNARImputer` (`not-MIWAE`) requires larger sample sizes ($N > 1,000$) and is less battle-tested than classical Heckman or Pattern-Mixture models.
- **Categorical Cardinality**: Umbra is optimized for continuous and mixed numerical tabular data. Datasets dominated by high-cardinality nominal text strings should be preprocessed before running selection models.
- **When to Just Use Standard MICE**: If Little's test fails to reject MCAR, covariate shifts are negligible (KS $< 0.10$), and the variable does not involve sensitive self-reporting, standard MICE chained equations (`strategy='mar'`) are completely defensible.

---

## Citations & Prior Art

If you use Umbra in your research or production pipelines, please cite the underlying foundational works:

- **Heckman Selection Model**:  
  Heckman, J. J. (1979). Sample selection bias as a specification error. *Econometrica*, 47(1), 153-161.
- **Little's MCAR Test**:  
  Little, R. J. A. (1988). A test of missing completely at random for multivariate data with missing values. *JASA*, 83(404), 1198-1202.
- **Pattern-Mixture Models**:  
  Little, R. J. A. (1993). Pattern-mixture models for multivariate incomplete data. *JASA*, 88(421), 125-134.
- **Sensitive Non-Response in Surveys**:  
  Tourangeau, R., & Yan, T. (2007). Sensitive questions in surveys. *Psychological Bulletin*, 133(5), 859.
- **not-MIWAE Deep Generative Approach**:  
  Ipsen, N. B., Mattei, P. A., & Frellsen, J. (2021). How to deal with missing not at random data: A missingness-agnostic deep generative approach. *NeurIPS*, 34, 19694-19707.

---

## License

Umbra is released under the [MIT License](LICENSE).\n
