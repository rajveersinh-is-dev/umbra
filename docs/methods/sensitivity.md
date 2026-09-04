# Method Specification: Sensitivity Grid & Tipping-Point Analysis

## 1. Mathematical Definition

Sensitivity analysis evaluates the functional mapping from an untestable MNAR assumption parameter $\delta$ to a downstream scientific estimand $\theta$:

$$\delta \mapsto \theta(\delta)$$

and its associated confidence band:
$$\delta \mapsto [\theta_L(\delta), \theta_U(\delta)]$$

where $\theta$ may be a population mean, a regression slope coefficient, a treatment effect, or a policy impact.

---

## 2. Automated Tipping Point Detection

A **Tipping Point** $\delta^*$ is the critical value of the sensitivity parameter at which the qualitative substantive conclusion changes:

### A. Sign Flip Tipping Point
$$\theta(\delta^*) = 0$$
Found by linear interpolation between adjacent grid points $(\delta_i, \theta_i)$ and $(\delta_{i+1}, \theta_{i+1})$ where $\theta_i \cdot \theta_{i+1} < 0$:
$$\delta^* = \delta_i - \theta_i \frac{\delta_{i+1} - \delta_i}{\theta_{i+1} - \theta_i}$$

### B. Loss of Statistical Significance (CI Zero-Crossing)
$$\theta_L(\delta^*) = 0 \quad \text{or} \quad \theta_U(\delta^*) = 0$$
Identifies the exact MNAR departure where the 95% confidence interval includes zero, rendering the effect statistically insignificant at $\alpha = 0.05$.

### C. Policy / Decision Threshold Crossing
$$\theta(\delta^*) = \tau$$
Identifies when the estimate crosses an external clinical, financial, or regulatory decision threshold $\tau$.

---

## 3. Substantive Conclusion Classification

Umbra classifies findings into two categories:

- **ROBUST FINDING**:
  No tipping point exists within the plausible range $\delta \in [-1.0, +1.0]$ standard deviations. Even under substantial unobserved non-response departures, the qualitative conclusion holds.
- **FRAGILE FINDING**:
  A tipping point exists within $|\delta^*| \le 1.0$ standard deviations. The conclusion depends critically on the unverifiable assumption that missingness is MAR. Reporting a single point estimate without sensitivity bounds is scientifically indefensible.

---

## 4. Inputs & Outputs
- `run_sensitivity_grid(data, target_column, delta_grid=None, downstream_evaluator=None, ...)`
- Returns `SensitivityReport` exposing:
  - `grid_df`: Full table of $\delta$, $\theta(\delta)$, standard errors, and confidence bounds.
  - `mar_baseline_estimate`: $\theta(0)$.
  - `estimate_min`, `estimate_max`: Range across plausible grid.
  - `uncertainty_spread`: $\theta_{\max} - \theta_{\min}$.
  - `tipping_points`: List of `TippingPoint` objects.
  - `is_fragile`: Boolean indicator of conclusion stability.
  - `interpretation`: Calibrated narrative.
