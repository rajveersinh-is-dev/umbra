# Tutorial: Real-World Case Study (Labor Economics Earnings Survey)

This tutorial applies Umbra to the **Current Population Survey (CPS)** labor economics benchmark to detect selective wage non-response and compare standard MICE against Heckman selection models.

---

## 1. Background & Problem

In survey research, wage questions suffer from non-response rates exceeding 30%. High earners and low earners disproportionately refuse to state their earnings.
Standard imputation tools assume that conditioning on age and education resolves the non-response (the Missing at Random assumption). If non-response depends on unobserved wage determinants, standard MICE point estimates are systematically biased upward or downward.

---

## 2. Load the CPS Benchmark Data

```python
import pandas as pd
from umbra import UmbraImputer, diagnose

# Load the observed survey sample
df_obs = pd.read_csv("data/processed/cps_income_observed.csv")
df_true = pd.read_csv("data/processed/cps_income_complete.csv")

print("Dataset Columns:", list(df_obs.columns))
print(f"Missing values in annual_income: {df_obs['annual_income'].isna().sum()} / {len(df_obs)}")
```

---

## 3. Diagnose the Missingness Mechanism

```python
report = diagnose(df_obs)
print(report.summary())

# Verify candidate auxiliary variable (instrument)
s_rep = report.shadow_variables["annual_income"]
print(s_rep.summary())
```
Notice that `contact_attempts` is surfaced as a strong candidate auxiliary variable:
- It correlates with non-response ($F > 15$).
- It has near-zero conditional correlation with observed earnings given demographics ($r = 0.02$, $p = 0.41$).

---

## 4. Run Imputation Models & Compare Estimates

```python
# Model 1: Standard MAR MICE
imputer_mar = UmbraImputer(strategy="mar", random_state=42)
df_mar = imputer_mar.fit_transform(df_obs)

# Model 2: Heckman Selection Model using contact_attempts as instrument
imputer_heck = UmbraImputer(
    strategy="heckman",
    shadow_cols={"annual_income": "contact_attempts"},
    random_state=42,
)
df_heck = imputer_heck.fit_transform(df_obs)

# Compare estimated population mean income
true_mean = df_true["annual_income"].mean()
obs_mean = df_obs["annual_income"].dropna().mean()
mar_mean = df_mar["annual_income"].mean()
heck_mean = df_heck["annual_income"].mean()

print(f"True Population Mean       : ${true_mean:,.2f}")
print(f"Complete-Case Observed Mean: ${obs_mean:,.2f} (Selection Bias: ${obs_mean - true_mean:+,.2f})")
print(f"Standard MAR MICE Mean     : ${mar_mean:,.2f} (Residual Bias: ${mar_mean - true_mean:+,.2f})")
print(f"Heckman Selection Mean     : ${heck_mean:,.2f} (Residual Bias: ${heck_mean - true_mean:+,.2f})")
```

---

## 5. Conclusion
Standard MICE leaves substantial residual bias because it cannot adjust for self-censoring in the unobserved tail. Heckman selection, identified by the interviewer contact instrument, recovers the ground truth mean.
