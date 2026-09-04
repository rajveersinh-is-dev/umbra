# Getting Started with Umbra

Umbra is an open-source Python library for **honest missing data imputation and diagnostics**.
It provides scikit-learn compatible transformers, multi-signal missingness mechanism diagnostics, and automated sensitivity analysis.

---

## Installation

### Standard Installation
```bash
pip install umbra-impute
```

### Installation with Optional Extras
```bash
# With PyTorch support for deep generative latent models
pip install "umbra-impute[deep]"

# With Streamlit interactive visual web demo
pip install "umbra-impute[demo]"

# Full research installation (all extras, benchmarking, testing)
pip install "umbra-impute[all]"
```

---

## 30-Second Quickstart

```python
import numpy as np
import pandas as pd
from umbra import UmbraImputer, diagnose

# 1. Load your dataset with missing values
df = pd.read_csv("dataset.csv")

# 2. Run multi-tier empirical diagnostics
report = diagnose(df)
print(report.summary())

# Access structured tier properties:
print("Little's MCAR Test p-value:", report.mcar.p_value)
print("Recommended Strategies:", report.recommendations)

# Export report to markdown or JSON
report_md = report.to_markdown()
report_dict = report.to_dict()

# 3. Scikit-learn Compatible Imputation
imputer = UmbraImputer(strategy="auto", random_state=42)
df_imputed = imputer.fit_transform(df)

# 4. Inspect sensitivity reports for columns with MNAR risk
for col, sens in imputer.sensitivity_reports_.items():
    print(sens.summary())
```

---

## Core Principles to Keep in Mind

1. **MNAR is Not Identifiable**: Umbra evaluates converging empirical evidence, not infallible proofs.
2. **Sensitivity Over False Confidence**: When data are suspected to be MNAR, report the sensitivity interval $[\theta_{\min}, \theta_{\max}]$ and tipping point $\delta^*$, rather than a single point estimate under an unverified MAR assumption.
3. **Instruments Must Be Verified**: Auxiliary shadow variables must be justified using substantive domain knowledge, not merely statistical correlations.
