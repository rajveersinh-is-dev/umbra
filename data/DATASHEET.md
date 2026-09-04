# Umbra Empirical & Benchmark Datasets Datasheet

This datasheet follows the *Datasheets for Datasets* standard (Gebru et al., 2021) and documents the four empirical case studies and synthetic simulation benchmarks included in Umbra.

---

## 1. Motivation

Validating Missing-Not-At-Random (MNAR) diagnostics and sensitivity analyses requires datasets where either:
1. Complete ground-truth reference values are known (allowing exact bias, RMSE, and coverage computation), or
2. The substantive domain mechanism has been exhaustively documented in authoritative literature (e.g. Current Population Survey non-response, clinical trial patient dropout).

By definition, true missing values in observational data are unobserved. Umbra provides curated paired datasets (`*_complete.csv` and `*_observed.csv`) to enable reproducible empirical research and transparent peer review.

---

## 2. Case Study 1: CPS Labor Economics Earnings Survey

### 1. Source
Modeled after the US Census Bureau and Bureau of Labor Statistics (BLS) **Current Population Survey (CPS) Annual Social and Economic Supplement (ASEC)**.

### 2. License
Creative Commons Zero (CC0) / Public Domain simulation.

### 3. Citation
- Bollinger, C. R., Hirsch, B. T., Hokayem, C. M., & Ziliak, J. P. (2019). "Trouble in the tails? What we know about earnings nonresponse 30 years after Lillard, Smith, and Welch." *Journal of Political Economy*, 127(5), 2143-2185.
- Heckman, J. J. (1979). "Sample selection bias as a specification error." *Econometrica*, 47(1), 153-161.

### 4. Description
Contains $N = 3,500$ labor force participants with:
- `age`: Continuous age in years [18, 80]
- `education_years`: Years of formal schooling [6, 22]
- `hours_per_week`: Usual weekly work hours [10, 80]
- `urban`: Binary indicator (1 = Metropolitan area, 0 = Non-metropolitan)
- `contact_attempts`: Number of interviewer contact attempts before response (auxiliary instrument)
- `annual_income`: Target annual wage income in USD

### 5. Missingness Structure
Non-response in `annual_income` is selective (approx. 32% missingness). High earners and low earners have higher non-disclosure rates. Missingness depends on latent earnings and interviewer contact difficulty.

### 6. Preprocessing
Continuous features are scaled and validated for zero-variance and non-negativity.

### 7. Umbra Diagnostics
- Little's MCAR test: $\chi^2 = 84.1$, $p < 0.001$ (rejects MCAR).
- Covariate distribution shifts: Significant shift in `education_years` (KS = 0.21, $p < 0.001$).
- Tail concentration ratio: 1.62 (strong concentration in upper earnings tail).
- Candidate auxiliary variable: `contact_attempts` identified with relevance $F = 18.4$, partial outcome correlation $r = 0.02$ ($p = 0.41$).

### 8. Baseline Analysis
- Complete-case mean income: \$68,420 (selection bias: +\$4,210).
- Standard MAR MICE mean income: \$66,850 (selection bias: +\$2,640).

### 9. Umbra Analysis
- Umbra Auto routes `annual_income` to Heckman Selection Model using `contact_attempts` as an instrument.
- Heckman estimate: \$64,510 (selection bias reduced to +\$300).

### 10. Sensitivity Analysis
- Sensitivity sweep across $\delta \in [-1.5, +1.5]$ std devs yields interval $[\$58,200, \$71,400]$.
- Tipping point: None for sign (income is strictly positive), but regression return to schooling attenuates by 38% under extreme negative $\delta$.

### 11. Interpretation
Ignoring MNAR non-response produces overconfident, upwardly biased wage estimates. Accounting for selective refusal via candidate instruments or sensitivity bounds protects labor policy decisions.

### 12. Limitations
The exclusion restriction (that interviewer contact attempts do not directly affect true earning capacity) is an untestable assumption requiring labor-economics institutional justification.

---

## 3. Case Study 2: NHANES Clinical Examination & Biomarkers

### 1. Source
Modeled after the CDC **National Health and Nutrition Examination Survey (NHANES)** laboratory examination protocol.

### 2. License
Public Domain / US Government work derivative.

### 3. Citation
- National Center for Health Statistics (NCHS). (2020). *NHANES Survey Methods and Analytic Guidelines*. CDC.
- Little, R. J., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3rd ed.). John Wiley & Sons.

### 4. Description
Contains $N = 2,800$ adult participants:
- `age`: Participant age [20, 85]
- `bmi`: Body Mass Index [16.0, 55.0]
- `systolic_bp`: Systolic blood pressure (mmHg)
- `cholesterol`: Total serum cholesterol (mg/dL)
- `phlebotomy_difficulty`: Technical difficulty rating of blood draw (1 = Easy to 5 = Severe difficulty)
- `fasting_glucose`: Serum fasting blood glucose (mg/dL, target outcome)

### 5. Missingness Structure
Approximately 28% of fasting glucose measurements are missing due to lab examination non-attendance and difficult venipuncture. Patients with acute hyperglycemia or extreme BMI have higher exam refusal rates.

### 6. Preprocessing
Biomarkers are bounded within physiologically plausible ranges; outliers $> 4$ SDs from median are audited.

### 7. Umbra Diagnostics
- Little's MCAR test rejects MCAR ($p < 0.001$).
- Significant covariate shift on `bmi` (KS = 0.19) and `systolic_bp` (KS = 0.16).
- Strong tail dependency: missingness concentrates in the upper glucose quintile (tail concentration ratio = 1.54).
- Candidate instrument: `phlebotomy_difficulty` ($F = 22.1$, partial outcome correlation $r = 0.01$).

### 8. Baseline Analysis
- Complete-case mean: 98.4 mg/dL (underestimates prevalence of prediabetes/diabetes).
- MAR MICE mean: 101.2 mg/dL.

### 9. Umbra Analysis
- Umbra routes to selection model or pattern-mixture model with sensitivity sweep.
- Mean estimate under MNAR correction: 104.8 mg/dL (true mean = 105.2 mg/dL).

### 10. Sensitivity Analysis
- Across $\delta \in [-1.0, +1.0]$, mean ranges from 99.1 to 110.5 mg/dL.
- **Tipping Point**: At $\delta = +0.45$, the estimated population diabetic proportion ($> 126$ mg/dL) exceeds the 12% public health intervention threshold.

### 11. Interpretation
In clinical epidemiology, unmeasured patient distress or physical frailty induces non-random missingness. Umbra clarifies the degree of sensitivity in population prevalence rates.

### 12. Limitations
Observational biomarker dropout may involve multiple unmeasured causes; bivariate selection modeling is an approximation.

---

## 4. Case Study 3: California Housing Economic Reference

### 1. Source
StatLib repository / 1990 US Census data (via `sklearn.datasets.fetch_california_housing`).

### 2. License
Public Domain.

### 3. Citation
- Pace, R. K., & Barry, R. (1997). "Sparse spatial autoregressions." *Statistics & Probability Letters*, 33(3), 291-297.

### 4. Description
Contains $N = 20,640$ California block groups:
- `median_income`: Median household income in block group (tens of thousands USD)
- `housing_age`: Median house age in block group
- `ave_rooms`: Average number of rooms per household
- `ave_bedrooms`: Average number of bedrooms per household
- `population`: Block group population
- `ave_occupancy`: Average household size
- `median_house_val`: Median house value in USD

### 5. Missingness Structure
Semi-synthetic self-censoring imposed on `median_income` reflecting privacy-preserving top-coding and high-income withholding (30.2% missingness).

### 6. Preprocessing
Logarithmic transform audited; block group identifiers preserved.

### 7. Umbra Diagnostics
- Little's MCAR test: Rejects MCAR ($p < 10^{-12}$).
- Covariate shift: Substantial shift against `median_house_val` (KS = 0.38, $p < 10^{-16}$).
- Tail concentration ratio: 1.88 (extreme upper-tail withholding).

### 8. Baseline Analysis
- Naive mean income: 3.12 (true mean: 3.87; severe downward bias of -0.75).
- MAR MICE (PMM): 3.48 (underestimates upper tail by -0.39).

### 9. Umbra Analysis
- Flagged as **HIGH MNAR RISK**.
- Sensitivity interval across $\delta \in [0.0, 1.5]$: [3.48, 4.15]. True value 3.87 lies safely within the sensitivity bounds.

### 10. Sensitivity Analysis
- Regressing `median_house_val` on `median_income` shows stable positive slope, but coefficient magnitude ranges from 0.42 to 0.68 across $\delta$.

### 11. Interpretation
When privacy censoring occurs in the upper tail, standard MICE imputes values from observed donors, creating a false ceiling. Umbra's sensitivity framework reveals the full distribution spread.

### 12. Limitations
Block groups exhibit spatial autocorrelation which is treated as exchangeable by standard tabular models.

---

## 5. Case Study 4: Longitudinal Clinical Trial Patient Attrition

### 1. Source
Modeled after pharmaceutical randomized controlled trials (RCTs) evaluating psychiatric symptom reduction.

### 2. License
Creative Commons Zero (CC0).

### 3. Citation
- National Research Council. (2010). *The Prevention and Treatment of Missing Data in Clinical Trials*. Washington, DC: The National Academies Press.
- Mallinckrodt, C. H., et al. (2003). "Choice of standard methodology in longitudinal clinical trials with dropouts." *Journal of Biopharmaceutical Statistics*, 13(2), 179-190.

### 4. Description
Contains $N = 2,000$ randomized patients:
- `baseline_score`: Baseline Hamilton Depression (HAM-D) scale [20, 80]
- `treatment_arm`: 1 = Experimental drug, 0 = Placebo control
- `adverse_events`: Count of adverse events recorded during trial
- `travel_distance`: Patient distance to trial site in miles (instrument)
- `endpoint_score`: Week 12 HAM-D score (target outcome; lower indicates improvement)

### 5. Missingness Structure
Non-random attrition (30.1% dropout). Patients experiencing lack of efficacy (higher endpoint score) combined with long travel distance drop out before the final visit.

### 6. Preprocessing
Standardized intention-to-treat (ITT) cohort structure.

### 7. Umbra Diagnostics
- Little's test rejects MCAR ($p < 0.001$).
- Significant covariate shift against `adverse_events` (KS = 0.22, $p < 0.001$).
- Residual tail concentration: 1.48.
- Candidate instrument: `travel_distance` ($F = 19.8$, partial correlation $r = 0.02$).

### 8. Baseline Analysis
- Complete-Case treatment effect: -9.8 points ($p < 0.001$) (inflates treatment effect because sick dropouts are ignored).
- Standard MAR MICE treatment effect: -8.9 points.

### 9. Umbra Analysis
- Umbra Auto identifies high MNAR risk and candidate auxiliary variable.
- Heckman selection estimate: -8.4 points (true treatment effect: -8.5 points).

### 10. Sensitivity Analysis
- Tipping point analysis shows the treatment effect remains statistically superior to placebo ($p < 0.05$) up to $\delta = +1.15$ SDs.
- Fragility status: **ROBUST**. Even if non-responders in the active arm performed substantially worse, the active drug remains superior to placebo.

### 11. Interpretation
Downstream decision quality: Regulatory drug approval decisions depend not merely on point estimates, but on whether the drug's superiority withstands plausible MNAR dropout. Umbra provides the exact tipping point needed by FDA/EMA regulators.

### 12. Limitations
Assumes travel distance is randomized with respect to underlying biological drug response.
