# Phase 3: Multiple Imputation & Rubin Pooling Engine - Research

**Researched:** 2026-09-09
**Domain:** Rubin's Multiple Imputation Rules, Barnard-Rubin (1999) Degrees of Freedom, Fraction of Missing Information (FMI)
**Confidence:** HIGH

<research_summary>
## Summary

Multiple imputation (Rubin 1987) accounts for the uncertainty inherent in missing data by creating $M \ge 5$ complete datasets, fitting the target model to each dataset independently, and combining the results according to Rubin's pooling rules.

### 1. Point and Variance Pooling (Rubin 1987)
For scalar or vector parameter $Q$ estimated by $\hat{Q}_m$ with variance $U_m$ on imputation $m \in \{1, \dots, M\}$:
- **Pooled Point Estimate**:
  $$\bar{Q} = \frac{1}{M} \sum_{m=1}^M \hat{Q}_m$$
- **Within-Imputation Variance**:
  $$\bar{U} = \frac{1}{M} \sum_{m=1}^M U_m$$
- **Between-Imputation Variance**:
  $$B = \frac{1}{M - 1} \sum_{m=1}^M (\hat{Q}_m - \bar{Q})^2$$
- **Total Variance**:
  $$T = \bar{U} + \left(1 + \frac{1}{M}\right) B$$
- **Pooled Standard Error**:
  $$\text{SE} = \sqrt{T}$$

### 2. Diagnostic Quantities
- **Relative Increase in Variance due to Nonresponse ($r$)**:
  $$r = \frac{(1 + M^{-1}) B}{\bar{U}}$$
- **Fraction of Missing Information ($\hat{\lambda}$)**:
  $$\hat{\lambda} = \frac{r + \frac{2}{\nu + 3}}{1 + r}$$
- **Relative Efficiency of $M$ Imputations vs $M = \infty$**:
  $$\text{RE} = \left(1 + \frac{\hat{\lambda}}{M}\right)^{-1}$$

### 3. Degrees of Freedom Adjustments
- **Rubin (1987) Large-Sample Degrees of Freedom**:
  $$\nu_m = (M - 1) \left(1 + \frac{1}{r}\right)^2 = (M - 1) \left(1 + \frac{\bar{U}}{(1 + M^{-1}) B}\right)^2$$
  *Limitation*: Assumes complete-sample degrees of freedom $\nu_0 = \infty$. When $N$ is small, $\nu_m$ can exceed $\nu_0$, which is physically impossible.
- **Barnard & Rubin (1999) Small-Sample Adjustment**:
  Given complete-data degrees of freedom $\nu_0 = N - k$:
  $$\hat{\gamma} = \frac{(1 + M^{-1}) B}{T} = \frac{r}{1 + r}$$
  $$\nu_{\text{obs}} = \frac{\nu_0 + 1}{\nu_0 + 3} \nu_0 (1 - \hat{\gamma})$$
  The adjusted small-sample degrees of freedom is the harmonic combination:
  $$\nu_{\text{adj}} = \left(\frac{1}{\nu_m} + \frac{1}{\nu_{\text{obs}}}\right)^{-1} = \frac{\nu_m \nu_{\text{obs}}}{\nu_m + \nu_{\text{obs}}}$$
  Properties:
  - If $\nu_0 \to \infty$, $\nu_{\text{obs}} \to \infty$, so $\nu_{\text{adj}} \to \nu_m$.
  - $\nu_{\text{adj}} \le \nu_0$ strictly always holds.
  - $\nu_{\text{adj}} \le \nu_m$ strictly always holds.
  - If $B = 0$ (no between-imputation variation), $\hat{\gamma} = 0$, $\nu_m = \infty$, so $\nu_{\text{adj}} = \frac{\nu_0 + 1}{\nu_0 + 3} \nu_0 \approx \nu_0$.

### 4. Confidence Intervals and Hypothesis Testing
$$t = \frac{\bar{Q} - Q_0}{\text{SE}}$$
$$(1 - \alpha)\text{ CI} = \left[ \bar{Q} - t_{\nu_{\text{adj}}, 1 - \alpha/2} \text{SE}, \quad \bar{Q} + t_{\nu_{\text{adj}}, 1 - \alpha/2} \text{SE} \right]$$
$$p\text{-value} = 2 \left(1 - F_t(|t|, \nu_{\text{adj}})\right)$$
</research_summary>
