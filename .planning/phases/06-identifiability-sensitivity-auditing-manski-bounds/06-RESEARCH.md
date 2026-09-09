# Phase 6: Identifiability & Sensitivity Auditing - Research & Mathematical Formalism

**Researched:** 2026-09-09
**Domain:** Nonparametric Partial Identification, Manski Bounds, Econometric Auditing

## 1. Manski Partial Identification Bounds (Manski 1989, 2003)

### A. Sharp Population Mean Bounds
Let $Y$ be a real random variable bounded in support $[y_L, y_U] \subset \mathbb{R}$.
Let $R \in \{0, 1\}$ be the observation indicator ($R=1$ if $Y$ is observed, $R=0$ if $Y$ is missing).
Let $p = P(R=1)$ and $1-p = P(R=0) = p_{\text{miss}}$.

By the Law of Total Probability:
$$\mathbb{E}[Y] = \mathbb{E}[Y \mid R=1] P(R=1) + \mathbb{E}[Y \mid R=0] P(R=0)$$
The observed conditional expectation $\mu_{\text{obs}} = \mathbb{E}[Y \mid R=1]$ is identifiable from sample data.
However, $\mu_{\text{mis}} = \mathbb{E}[Y \mid R=0]$ is completely unobserved.

Since $Y \in [y_L, y_U]$ almost surely:
$$y_L \le \mathbb{E}[Y \mid R=0] \le y_U$$
Multiplying by $p_{\text{miss}}$ and adding $\mu_{\text{obs}} (1 - p_{\text{miss}})$ yields the sharp Manski lower and upper bounds:
$$\text{LB}_{\text{mean}} = \mu_{\text{obs}} (1 - p_{\text{miss}}) + y_L \cdot p_{\text{miss}}$$
$$\text{UB}_{\text{mean}} = \mu_{\text{obs}} (1 - p_{\text{miss}}) + y_U \cdot p_{\text{miss}}$$

**Properties**:
- Width of the Manski interval: $\Delta_{\text{mean}} = \text{UB}_{\text{mean}} - \text{LB}_{\text{mean}} = p_{\text{miss}} (y_U - y_L)$.
- As $p_{\text{miss}} \to 0$, $[\text{LB}, \text{UB}] \to \{\mu_{\text{obs}}\}$.
- As $p_{\text{miss}} \to 1$, $[\text{LB}, \text{UB}] \to [y_L, y_U]$.
- These bounds are **sharp**: every point in $[\text{LB}, \text{UB}]$ corresponds to a valid joint distribution $(Y, R)$ consistent with the observed data (Manski, 2003, Chapter 1).

### B. Sharp Quantile Bounds
For any quantile level $\alpha \in (0, 1)$, let $q_\alpha$ be the $\alpha$-quantile of $Y$:
$$P(Y \le q_\alpha) = \alpha$$
Decomposing into observed and unobserved components:
$$P(Y \le t) = P(Y \le t \mid R=1)(1 - p_{\text{miss}}) + P(Y \le t \mid R=0) p_{\text{miss}}$$
Because $0 \le P(Y \le t \mid R=0) \le 1$:
$$P(Y \le t \mid R=1)(1 - p_{\text{miss}}) \le P(Y \le t) \le P(Y \le t \mid R=1)(1 - p_{\text{miss}}) + p_{\text{miss}}$$

Let $F_{\text{obs}}(t) = P(Y \le t \mid R=1)$ be the empirical CDF of observed cases.
Then $q_\alpha \in [\text{LB}_\alpha, \text{UB}_\alpha]$ where:
- If $\alpha \le p_{\text{miss}}$: $\text{LB}_\alpha = y_L$ (since all missing cases could be $< t$).
  Otherwise: $\text{LB}_\alpha = F_{\text{obs}}^{-1}\left(\frac{\alpha - p_{\text{miss}}}{1 - p_{\text{miss}}}\right)$.
- If $\alpha \ge 1 - p_{\text{miss}}$: $\text{UB}_\alpha = y_U$ (since all missing cases could be $> t$).
  Otherwise: $\text{UB}_\alpha = F_{\text{obs}}^{-1}\left(\frac{\alpha}{1 - p_{\text{miss}}}\right)$.

For the median ($\alpha = 0.5$):
- If $p_{\text{miss}} \ge 0.5$: the median is uninformative without further assumptions: $[y_L, y_U]$.
- If $p_{\text{miss}} < 0.5$:
  $$\text{LB}_{\text{med}} = F_{\text{obs}}^{-1}\left(\frac{0.5 - p_{\text{miss}}}{1 - p_{\text{miss}}}\right), \quad \text{UB}_{\text{med}} = F_{\text{obs}}^{-1}\left(\frac{0.5}{1 - p_{\text{miss}}}\right)$$

---

## 2. Taxonomy of Assumptions for Audit Certificates

| Strategy / Imputer | Testable Empirical Implications | Required Untestable Domain Assumptions | Sharp Bounds Under Zero Assumptions |
|---|---|---|---|
| **MICE / MAR** | Little's MCAR test $p$-value, Covariate shift (Cohen's $d$, Cliff's $\delta$) | Missingness conditionally independent of $Y$ given $X$: $Y \perp R \mid X$ | Manski Mean & Median Intervals |
| **Heckman Selection** | First-stage instrument relevance ($F > 10$, Stock-Yogo) | 1. Bivariate normal error disturbances $(\varepsilon, u) \sim \mathcal{N}_2$<br>2. Instrument exogeneity: $\text{Cov}(Z, \varepsilon) = 0$ | Manski Bounds; Instrument-Assisted Bounds |
| **Pattern Mixture** | Plausible baseline parameter variance | Specific offset magnitude $\delta$ or prior distribution $P(\delta)$ | Sensitivity Tipping Point Curve & Manski Bounds |
