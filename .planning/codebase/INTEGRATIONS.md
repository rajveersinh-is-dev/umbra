# External Integrations

**Analysis Date:** 2026-09-09

## APIs & External Services

Umbra is a fully local, self-contained statistical computing and machine learning package. It does **NOT** rely on external web APIs, cloud microservices, SaaS authentication, or remote network endpoints.

**Network Access:**
- None required. All estimation, diagnostic tests, simulation runners, and sensitivity analyses run 100% offline and locally.

## Data Storage & Formats

**Tabular Ingestion:**
- `pandas.DataFrame` — first-class supported tabular data structure with mixed numeric types and NaN indicators.
- `numpy.ndarray` — 2D floating-point arrays for high-throughput numeric workflows.
- `CSV` — standard comma-separated tabular files read via `pd.read_csv()` in CLI tools (`umbra diagnose`, `umbra impute`).

**Export Formats:**
- Imputed Datasets: CSV format (`.to_csv()`) or returned in-memory as DataFrames / ndarrays.
- Diagnostic Reports: Serialized via `UmbraDiagnosticReport` to:
  - JSON (`.to_json(path=...)`)
  - GitHub-flavored Markdown (`.to_markdown(path=...)`)
  - Self-contained HTML with CSS styling (`.to_html(path=...)`)
- Sensitivity Grids: Exported as CSV containing per-delta point estimates, standard errors, and confidence intervals.

## Machine Learning Framework Integrations

**Scikit-Learn Ecosystem:**
- `sklearn.base.BaseEstimator`, `sklearn.base.TransformerMixin`: `UmbraImputer` adheres to the scikit-learn estimator interface.
- Pipeline integration: Can be inserted directly into `sklearn.pipeline.Pipeline([('imputer', UmbraImputer()), ('classifier', ...)])`.
- Cross-validation: Compatible with `cross_val_score`, `GridSearchCV`, and `RandomizedSearchCV`.

**PyTorch (Optional Deep Generative Module):**
- Integrated in `umbra.imputers.deep_generative_mnar.DeepGenerativeMNARImputer`.
- Uses PyTorch modules (`torch.nn.Module`, `torch.optim.Adam`) to train a variational autoencoder for high-dimensional MNAR patterns when `torch` is available.
- Gracefully degrades with clear error messaging if PyTorch is not installed.

## CLI & Shell Integration

**CLI Framework:**
- `click` and `rich`: Command-line executable `umbra` registered via `[project.scripts]` entry point in `pyproject.toml`.
- Commands:
  - `umbra diagnose <path>`: Runs Little's test, covariate shifts, and self-censoring checks; renders a rich console table and optional Markdown output.
  - `umbra impute <path>`: Executes auto-routing or explicit imputation, exports imputed CSV and optional sensitivity analysis tables.

## CI/CD & Automation

**GitHub Actions:**
- Workflow file: `.github/workflows/ci.yml` (configured for multi-OS, multi-Python matrix testing, coverage upload, and artifact validation).

---

*Integrations analysis: 2026-09-09*
*Update when external integrations or data protocols change*
