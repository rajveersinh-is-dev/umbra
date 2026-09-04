# Method Specification: MAR Chained Equations Imputer (MICE)

## 1. Mathematical Definition

Implements Multiple Imputation by Chained Equations (MICE), also known as Fully Conditional Specification (FCS) (van Buuren, 2018), under the Missing at Random (MAR) assumption.

### Iterative Algorithm
For a dataset with incomplete variables $Y_1, \dots, Y_p$:
1. **Initialization**: Fill missing values with column medians.
2. **Cycle $t = 1, \dots, T$**:
   For each incomplete variable $Y_j$:
   - Temporarily set imputed values in $Y_j$ back to missing.
   - Regress $Y_{j, \text{obs}}$ on all other variables $Y_{-j}^{(t)}$.
   - Impute missing entries $Y_{j, \text{mis}}$ by drawing from the predictive distribution:
     - **Predictive Mean Matching (PMM)**: Match predicted values $\hat{Y}_{j, \text{mis}}$ to the $k$ nearest observed donors $\hat{Y}_{j, \text{obs}}$ and draw an observed donor's true value.
     - **Bayesian Ridge**: Draw parameter vector $\tilde{\beta}$ and residual variance $\tilde{\sigma}^2$ from their normal-inverse-gamma posterior, and draw $\hat{Y}_{j, \text{mis}} \sim \mathcal{N}(X_{\text{mis}} \tilde{\beta}, \tilde{\sigma}^2)$.
     - **Ridge Regression**: Conditional expectation prediction.
3. Repeat for $T$ cycles until convergence.
4. If $M > 1$, repeat across $M$ independent Markov chains to produce $M$ completed datasets.

### Rubin's (1987) Rules for Multiple Imputation
For a scalar estimand $Q$ (e.g. mean or regression coefficient):
- **Pooled Point Estimate**:
  $$\bar{Q} = \frac{1}{M} \sum_{m=1}^M \hat{Q}_m$$
- **Within-Imputation Variance**:
  $$\bar{U} = \frac{1}{M} \sum_{m=1}^M U_m$$
- **Between-Imputation Variance**:
  $$B = \frac{1}{M - 1} \sum_{m=1}^M (\hat{Q}_m - \bar{Q})^2$$
- **Total Variance**:
  $$T = \bar{U} + \left(1 + \frac{1}{M}\right) B$$
- **Degrees of Freedom** (Barnard & Rubin, 1999):
  $$\nu = (M - 1) \left(1 + \frac{1}{r}\right)^2, \quad r = \frac{(1 + 1/M) B}{\bar{U}}$$

---

## 2. Assumptions
- Missingness is Missing at Random (MAR) conditional on the included predictors.
- The chained regression equations are mutually compatible and define a valid joint distribution.

---

## 3. Inputs
- `max_iter`: Cycles of chained equations (default 10).
- `imputation_method`: `'pmm'`, `'bayesian_ridge'`, or `'ridge'`.
- `n_donors`: Nearest neighbors for PMM (default 5).
- `n_imputations`: Number of imputed datasets ($M$, default 1).
- `random_state`: Reproducibility seed.

---

## 4. Outputs
- Imputed DataFrame (or List of DataFrames if $M > 1$).

---

## 5. Failure Modes
- Under MNAR, MICE produces biased estimates with false confidence (coverage collapses).
- Collinear predictors can cause instability; handled via ridge regularization $\alpha = 1.0$.

---

## 6. References
- van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.). Chapman & Hall/CRC.
- Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. John Wiley & Sons.
- Barnard, J., & Rubin, D. B. (1999). "Small-sample degrees of freedom with multiple imputation." *Biometrika*, 86(4), 948-955.
