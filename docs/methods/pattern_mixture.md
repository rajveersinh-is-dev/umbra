# Method Specification: Pattern-Mixture Models

## 1. Mathematical Definition

Implements Pattern-Mixture Models (Little, 1993, 1994) with explicit sensitivity shift parameters ($\delta$).

### Model Factorization
Rather than modeling selection $P(M \mid X)$, pattern-mixture models factor the joint distribution as:
$$P(Y, R \mid X) = P(Y \mid X, R) P(R \mid X)$$

### Distributional Specifications
1. **Observed Population ($R = 1$)**:
   $$Y \mid (X, R = 1) \sim \mathcal{N}(X \beta, \sigma^2)$$
   Parameters $\beta$ and $\sigma$ are identified directly from observed responders.
2. **Missing Population ($R = 0$)**:
   $$Y \mid (X, R = 0) \sim \mathcal{N}(X \beta + \Delta, \sigma^2)$$
   where $\Delta$ is an **unidentifiable sensitivity offset**.

### Offset Parametrizations ($\Delta$)
- **Standardized Shift (`shift_type='standardized'`)**:
  $$\Delta = \delta \cdot \sigma$$
  where $\delta$ is expressed in units of residual standard deviations (e.g. $\delta \in [-1.5, +1.5]$).
  - $\delta = 0.0$: Equivalent to standard MAR.
  - $\delta > 0.0$: Missing units are systematically higher than responders with identical covariates.
  - $\delta < 0.0$: Missing units are systematically lower than responders with identical covariates.
- **Percentage Shift (`shift_type='percentage'`)**:
  $$\hat{Y}_{\text{mis}} = (X \hat{\beta}) \cdot (1 + \delta)$$
- **Raw Shift (`shift_type='raw'`)**:
  $$\Delta = \delta \quad \text{(in native outcome units)}$$

---

## 2. Epistemological Role
Because $\delta$ is **fundamentally unidentifiable from observed data**, it is treated strictly as an **external sensitivity parameter**. It is never "estimated" or "optimized" from observed data alone.

---

## 3. Inputs
- `delta`: Sensitivity parameter (float, default 0.0).
- `shift_type`: `'standardized'`, `'percentage'`, or `'raw'`.
- `target_cols`: Target column(s) to shift.
- `stochastic`: Whether to draw random normal residuals.
- `n_imputations`: Number of draws ($M$).
- `random_state`: Reproducibility seed.

---

## 4. Outputs
- Completed DataFrame (or List of DataFrames).

---

## 5. References
- Little, R. J. A. (1993). "Pattern-mixture models for multivariate incomplete data." *JASA*, 88(421), 125-134.
- Little, R. J. A. (1994). "A class of pattern-mixture models for normal incomplete data." *Biometrika*, 81(3), 471-483.
- Hedeker, D., & Gibbons, R. D. (1997). "Application of random-effects pattern-mixture models for missing data in longitudinal studies." *Psychological Methods*, 2(1), 64-78.
