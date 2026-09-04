# Missing Not at Random (MNAR)

## 1. Mathematical Definition

Data are **Missing Not at Random (MNAR)** when the probability of missingness depends directly on the unobserved missing values themselves, even after conditioning on all observed variables $X_{\text{obs}}$:

$$P(M \mid X_{\text{obs}}, X_{\text{mis}}, \psi) \neq P(M \mid X_{\text{obs}}, \psi)$$

Under MNAR, responders and non-responders with identical observed characteristics differ systematically in their unobserved values:
$$P(X_{\text{mis}} \mid X_{\text{obs}}, M = 1) \neq P(X_{\text{mis}} \mid X_{\text{obs}}, M = 0)$$

---

## 2. Real-World Mechanisms of MNAR

In empirical research, MNAR arises naturally from human behavior and clinical phenomena:
- **Income Non-Response**: High earners withhold earnings due to privacy concerns; low-income workers withhold earnings due to social desirability bias (Bollinger et al., 2019).
- **Depression / Psychiatric Trials**: Patients experiencing severe symptom exacerbation or adverse events drop out of clinical trials prior to follow-up measurement (NRC, 2010).
- **Substance Use & Stigmatized Behaviors**: Respondents engaging in high-frequency illicit behaviors refuse to answer sensitive questionnaire items (Harrison et al., 2007).
- **Body Mass Index (BMI)**: Individuals with severe obesity disproportionately refuse weigh-in protocols (Rowland, 1990).

---

## 3. Why Standard Imputation Breaks Down Under MNAR

Standard imputation tools (scikit-learn `IterativeImputer`, R `mice`, `missForest`) draw imputations from the conditional distribution of observed cases:
$$\hat{Y}_{\text{mis}} \sim \hat{P}(Y \mid X_{\text{obs}}, M = 0)$$
Under MNAR, this conditional distribution is systematically shifted relative to the true non-responder distribution.

### Empirical Consequences
1. **Severe Bias**: In Monte Carlo simulations, standard MICE introduces bias up to -0.60 standard deviations on missing cells under moderate self-censoring.
2. **Catastrophic Coverage Collapse**: Because standard MICE produces falsely confident standard errors around biased point estimates, empirical 95% confidence interval coverage collapses from the nominal 95% to **0%–15%**.

---

## 4. Umbra's Two-Pronged Response to MNAR

Because true MNAR is not identifiable from observed data alone, Umbra provides two honest methodological paths:

### Path A: Identification via Candidate Auxiliary Variables (Heckman Selection)
When a valid auxiliary instrument $Z$ is available that satisfies:
1. **Relevance**: $Z$ correlates with missingness ($F > 10$).
2. **Exclusion Restriction**: $Z$ has no direct path to outcome $Y$ conditional on $X$.

Umbra fits a Heckman selection model that explicitly models the truncation distribution.

### Path B: Sensitivity Analysis Grid & Tipping Points
When no instrument exists, Umbra rejects false confidence. It sweeps a grid of plausible unobserved departures:
$$E[Y \mid X_{\text{obs}}, M = 1] = E[Y \mid X_{\text{obs}}, M = 0] + \delta \cdot \sigma$$
and reports the sensitivity interval $[\theta_{\min}, \theta_{\max}]$ alongside the exact **tipping point** $\delta^*$ required to reverse the scientific conclusion.
