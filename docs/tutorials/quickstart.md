# Tutorial: Quickstart & Basic Workflow

This tutorial walks through diagnosing and imputing an incomplete dataset using Umbra and integrating with scikit-learn pipelines.

---

## 1. Load Data with Missing Values

```python
import numpy as np
import pandas as pd
from umbra import UmbraImputer, diagnose

# Create a sample DataFrame with missing values
np.random.seed(42)
n = 1000
age = np.random.normal(45, 10, size=n)
education = 0.4 * age + np.random.normal(12, 3, size=n)
income = 20000 + 800 * age + 1500 * education + np.random.normal(0, 5000, size=n)

df = pd.DataFrame({"age": age, "education": education, "income": income})

# Knock out values in income based on age and income (MNAR self-censoring)
p_miss = 1.0 / (1.0 + np.exp(-(0.0001 * (income - np.mean(income)) + 0.05 * (age - 45))))
mask = np.random.uniform(0, 1, size=n) < p_miss
df.loc[mask, "income"] = np.nan

print(f"Missing values in income: {df['income'].isna().sum()} / {n} ({df['income'].isna().mean():.1%})")
```

---

## 2. Run Comprehensive Diagnostics

```python
# Run the multi-tier diagnostic battery
report = diagnose(df)

# Print clean human-readable summary
print(report.summary())

# Inspect individual tiers
print("Little's MCAR Statistic:", report.mcar.statistic)
print("Little's MCAR p-value  :", report.mcar.p_value)
print("Is MCAR rejected?      :", report.mcar.is_rejected)

# Check covariate distribution shifts
income_shifts = report.covariate_shift.variable_reports["income"]
print("Max KS statistic on covariates:", income_shifts.max_ks_statistic)
```

---

## 3. Scikit-Learn Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# Drop-in transformer inside a scikit-learn Pipeline
pipeline = Pipeline([
    ("imputer", UmbraImputer(strategy="auto", random_state=42)),
    ("regressor", Ridge()),
])

# Fit on training data and predict
y = df["age"] * 2.5 + np.random.normal(0, 1, size=n)
pipeline.fit(df, y)
predictions = pipeline.predict(df)
print("Predictions generated successfully! Shape:", predictions.shape)
```
