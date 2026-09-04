# Tutorial: Automated Sensitivity Grid & Tipping-Point Analysis

This tutorial demonstrates how to quantify the fragility of substantive research findings across unobserved MNAR departures using Umbra's sensitivity grid machinery.

---

## 1. When Should You Run Sensitivity Analysis?

Whenever you suspect that data are Not-Missing-At-Random (MNAR) and **no valid instrumental variable is available**, you cannot identify a single point estimate.
Instead of reporting a single overconfident MAR point estimate, report:
1. The **Sensitivity Interval** $[\theta_{\min}, \theta_{\max}]$ across plausible departures $\delta \in [-1.0, +1.0]$.
2. The **Tipping Point** $\delta^*$: "How large an unobserved departure between responders and non-responders is required to reverse the qualitative finding?"

---

## 2. Running a Sensitivity Grid Analysis

```python
import pandas as pd
from umbra.sensitivity.grid_analysis import run_sensitivity_grid

# Load clinical trial data where depression score has patient attrition
df = pd.read_csv("data/processed/clinical_trial_attrition_observed.csv")

# Custom evaluator: Difference in mean endpoint score between treatment arms
def treatment_effect_evaluator(imputed_df: pd.DataFrame) -> float:
    mean_active = imputed_df.loc[imputed_df["treatment_arm"] == 1, "endpoint_score"].mean()
    mean_placebo = imputed_df.loc[imputed_df["treatment_arm"] == 0, "endpoint_score"].mean()
    return float(mean_active - mean_placebo)

# Run sensitivity grid sweep
report = run_sensitivity_grid(
    data=df,
    target_column="endpoint_score",
    delta_grid=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    downstream_evaluator=treatment_effect_evaluator,
)

print(report.summary())
```

---

## 3. Interpreting Tipping Points

```python
# Check if the conclusion is robust or fragile
if report.is_fragile:
    print("WARNING: Finding is fragile to plausible MNAR departures!")
    for tp in report.tipping_points:
        print(f"  Tipping event: {tp.description}")
else:
    print("ROBUST: The qualitative conclusion holds across all plausible departures [-1.0, +1.0].")
```

---

## 4. Visualizing Sensitivity Curves

```python
from umbra.visualization import plot_sensitivity_curve

# Plot sensitivity curve theta(delta) with 95% confidence bands and tipping points
fig = plot_sensitivity_curve(report, save_path="sensitivity_curve.png")
print("Sensitivity figure saved to sensitivity_curve.png!")
```
