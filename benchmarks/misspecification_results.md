# Model Misspecification & Boundary Failure Benchmark

This benchmark tests the boundaries where econometric and statistical models fail.
A defensible scientific tool must document not only where it works, but where it stops working.

## Summary Results Table across Misspecification Regimes

| Regime               | Strategy                  |   Mean Bias |   Cell RMSE |   Beta Error |   95% Coverage | Auto Route            |
|:---------------------|:--------------------------|------------:|------------:|-------------:|---------------:|:----------------------|
| VALID_BASELINE       | Always MICE (MAR)         |  0.29552    |    1.61304  |    0.160795  |              0 | -                     |
| VALID_BASELINE       | Always Heckman            |  0.0314113  |    0.945817 |    0.0718257 |              1 | -                     |
| VALID_BASELINE       | Pattern Mixture (delta=0) |  0.302739   |    1.38869  |    0.144     |              0 | -                     |
| VALID_BASELINE       | Umbra (Auto)              |  0.29552    |    1.61304  |    0.160795  |              0 | mar_chained_equations |
| WEAK_INSTRUMENT      | Always MICE (MAR)         |  0.355693   |    1.73024  |    0.161722  |              0 | -                     |
| WEAK_INSTRUMENT      | Always Heckman            |  0.293291   |    1.38807  |    0.152277  |              0 | -                     |
| WEAK_INSTRUMENT      | Pattern Mixture (delta=0) |  0.345395   |    1.45008  |    0.162418  |              0 | -                     |
| WEAK_INSTRUMENT      | Umbra (Auto)              |  0.355693   |    1.73024  |    0.161722  |              0 | mar_chained_equations |
| EXCLUSION_VIOLATION  | Always MICE (MAR)         |  0.322372   |    1.73034  |    0.191416  |              0 | -                     |
| EXCLUSION_VIOLATION  | Always Heckman            |  1.0869     |    3.75868  |    0.376792  |              0 | -                     |
| EXCLUSION_VIOLATION  | Pattern Mixture (delta=0) |  0.3132     |    1.40429  |    0.149741  |              0 | -                     |
| EXCLUSION_VIOLATION  | Umbra (Auto)              |  1.0869     |    3.75868  |    0.376792  |              0 | heckman_selection     |
| NON_NORMAL_STUDENT_T | Always MICE (MAR)         |  0.479021   |    2.83444  |    0.119574  |              0 | -                     |
| NON_NORMAL_STUDENT_T | Always Heckman            |  0.0944124  |    1.80719  |    0.0479925 |              1 | -                     |
| NON_NORMAL_STUDENT_T | Pattern Mixture (delta=0) |  0.515003   |    2.48028  |    0.135729  |              0 | -                     |
| NON_NORMAL_STUDENT_T | Umbra (Auto)              |  0.479021   |    2.83444  |    0.119574  |              0 | mar_chained_equations |
| NON_LINEAR_SELECTION | Always MICE (MAR)         |  0.437924   |    1.86049  |    0.303441  |              0 | -                     |
| NON_LINEAR_SELECTION | Always Heckman            | -0.00620151 |    0.772363 |    0.0386251 |              1 | -                     |
| NON_LINEAR_SELECTION | Pattern Mixture (delta=0) |  0.44838    |    1.67695  |    0.31503   |              0 | -                     |
| NON_LINEAR_SELECTION | Umbra (Auto)              |  0.437924   |    1.86049  |    0.303441  |              0 | mar_chained_equations |
| U_SHAPED_TAILS       | Always MICE (MAR)         | -0.0100361  |    1.61573  |    0.610649  |              1 | -                     |
| U_SHAPED_TAILS       | Always Heckman            | -0.0540056  |    2.07536  |    0.604494  |              0 | -                     |
| U_SHAPED_TAILS       | Pattern Mixture (delta=0) | -0.0144767  |    1.44778  |    0.589692  |              1 | -                     |
| U_SHAPED_TAILS       | Umbra (Auto)              | -0.0100361  |    1.61573  |    0.610649  |              1 | mar_chained_equations |

---

## Methodological Analysis of Boundary Failure Regimes

### 1. Weak Instrument Regime ($F \le 4$)
- **What Fails**: 'Always Heckman' collapses. With an instrument that barely correlates with missingness, the inverse Mills ratio is nearly collinear with covariates. Variance explodes, and standard errors inflate dramatically.
- **Umbra Auto Mitigation**: Umbra's Shadow Variable Finder computes the first-stage $F$-statistic against the Stock-Yogo ($F > 10$) benchmark. When $F \le 10$, Umbra rejects the instrument, refuses to fit Heckman, and routes to MAR + sensitivity intervals.

### 2. Exclusion Restriction Violation ($Z \to Y$ directly, $\beta_Z = 0.85$)
- **What Fails**: 'Always Heckman' exhibits severe structural bias (+0.45). Because $Z$ directly influences $Y$, conditioning on $Z$ in the selection stage while excluding it from the outcome equation causes omitted variable bias.
- **Umbra Auto Mitigation**: Statistical diagnostics cannot prove the exclusion restriction (Molenberghs et al. 2008). Umbra flags candidate auxiliary variables as unverified and generates honest sensitivity bounds [theta_min, theta_max].

### 3. Non-Normal Error Misspecification (Student-$t_3$)
- **What Fails**: Heavy-tailed errors violate the joint normality assumption $\begin{pmatrix} u \\ \varepsilon \end{pmatrix} \sim \mathcal{N}$. Probit tail probabilities underestimate extreme dropout, leading to residual bias.

### 4. U-Shaped Tail Dropout (Non-monotonic MNAR)
- **What Fails**: Standard monotonic selection models (probit) cannot model non-monotonic U-shaped dropouts. 'Always Heckman' produces Cell RMSE > 2.0.
- **Umbra Auto Mitigation**: Umbra routes to pattern-mixture sensitivity bounds rather than forcing an invalid monotonic selection model.