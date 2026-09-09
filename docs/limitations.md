# Known Limitations and Open Problems

This document catalogues open methodological, statistical, and engineering limitations in Umbra v0.2.0. Rather than asserting completeness or artificial ratings, this record outlines where current assumptions or heuristics operate, how the six core limitation areas have been systematically resolved in v0.2.0, where asymptotic theory requires care, and where ongoing research is needed.

---

## 1. Non-Identifiability of MNAR from Observed Data Alone

* **Theoretical Foundation**: Molenberghs et al. (2008), *Every missingness not at random model has a missingness at random counterpart with equal fit*, JRSS-B 70(2): 371–388; Manski, C. F. (1989, 2003), *Partial Identification of Probability Distributions*, Springer.
* **Code Reference**: `umbra/diagnostics/manski_bounds.py`, `umbra/diagnostics/identifiability_audit.py`, `umbra/diagnostics/report.py`
* **Limitation**: Observed data distributions contain information only about observable implications (e.g., covariate shifts between responders and non-responders, or global departures from MCAR via Little's test). If an MNAR process induces missingness purely as a function of the unobserved value $Y_{\text{mis}}$ without affecting observable margins (e.g., pure unconfounded latent self-masking or symmetric tail dropout with balanced margins), observable diagnostics cannot distinguish this from MAR or even MCAR.
* **Implemented Solution (v0.2.0 / Phase 6)**:
  - **Manski Partial Identification Bounds**: `compute_manski_bounds` and `compute_dataframe_manski_bounds` compute sharp nonparametric bounds for population mean and quantiles under zero untestable assumptions ($\text{LB} = \bar{Y}_{\text{obs}}(1-p) + y_L p$, $\text{UB} = \bar{Y}_{\text{obs}}(1-p) + y_U p$, exact width $\Delta = p(y_U - y_L)$).
  - **Identifiability & Assumption Audit Certificates**: `IdentifiabilityCertificate` in `UmbraDiagnosticReport` explicitly documents what testable implications hold and what untestable structural assumptions are required for each candidate strategy (Manski, MICE, Heckman, Pattern Mixture).
* **Ongoing Epistemic Boundary**: Low risk does not prove MAR, and point imputations cannot eliminate selection bias under unobservable MNAR without untestable structural assumptions. Umbra provides transparent partial identifiability bounds rather than claiming impossible point certainty.

---

## 2. Heckman Selection Two-Step Standard Error Adjustments

* **Theoretical Foundation**: Murphy & Topel (1985), *Estimation and Inference with Two-Step Econometric Estimators*, JBES 3(4): 370–379; Heckman, J. J. (1979), *Sample Selection Bias as a Specification Error*, Econometrica 47(1): 153–161.
* **Code Reference**: `umbra/imputers/heckman_selection.py`
* **Limitation**: The classical Heckman (1979) two-step estimator introduces a generated regressor—the estimated inverse Mills ratio $\hat{\lambda}_i = \lambda(w_i \hat{\gamma})$—from the first-stage probit into the second-stage outcome regression. Ordinary least squares (OLS) standard errors from the second stage ignore the estimation variance of $\hat{\gamma}$ from the first stage and assume homoskedastic second-stage disturbances, which is violated under selection.
* **Implemented Solution (v0.2.0 / Phase 1)**:
  - **Murphy-Topel Asymptotic SE Correction**: Second-stage covariance is corrected via the full asymptotic Murphy-Topel sandwich formulation $\Sigma_{\beta} = V_2 + V_2 C V_1 C' V_2 - V_2 (C R' + R C') V_2$, fully accounting for the estimation variance of the first-stage Probit parameters.
  - **Full-Information Maximum Likelihood (FIML)**: Simultaneous joint optimization of selection and outcome equations via BFGS with numerical Hessian inversion and automatic fallback to Murphy-Topel two-step on non-convergence.
  - **Paired Bootstrap Resampling**: Nonparametric bootstrap resampling across both selection and outcome stages when requested (`n_bootstrap_se > 0`).
* **Ongoing Boundary**: FIML estimation requires numerical optimization over a non-convex likelihood surface that can fail to converge under severe multicollinearity or weak instruments; two-step estimation remains sensitive to exclusion restriction strength.

---

## 3. Near-Singular Design Matrices and Ridge Fallback Uncertainty

* **Theoretical Foundation**: Belsley, D. A., Kuh, E., & Welsch, R. E. (1980), *Regression Diagnostics: Identifying Influential Data and Sources of Collinearity*, John Wiley & Sons.
* **Code Reference**: `umbra/imputers/heckman_selection.py`
* **Limitation**: In small samples ($N < 100$) or when severe multicollinearity arises between covariates $X$ and the inverse Mills ratio $\lambda$ (frequently occurring when exclusion restrictions are weak or absent), second-stage OLS estimation can fail or produce unstable inversions.
* **Implemented Solution (v0.2.0 / Phase 2)**:
  - **Ridge Sandwich Standard Errors**: Implemented asymptotic sandwich covariance $V = (X'X + \lambda I)^{-1} X' \Omega X (X'X + \lambda I)^{-1}$ for regularized second-stage estimation (`se_method='sandwich'`), preventing unhandled runtime crashes while providing valid regularized uncertainty quantification.
  - **Collinearity Diagnostics Suite**: Automated computation of Belsley-Kuh-Welsch condition indices and Variance Inflation Factors ($\text{VIF}(\lambda)$) for the inverse Mills ratio.
  - **HeckmanCollinearityWarning**: Emitted automatically when condition index $> 30$ or $\text{VIF}(\lambda) > 10$.
* **Ongoing Boundary**: Ridge regularization introduces finite-sample shrinkage bias in exchange for numerical stability; parameter standard errors reflect variability conditioned on the regularization parameter.

---

## 4. Single-Imputation vs. Rubin-Pooled Multiple Imputation Variance in Benchmarking

* **Theoretical Foundation**: Rubin, D. B. (1987), *Multiple Imputation for Nonresponse in Surveys*, John Wiley & Sons; Barnard, J., & Rubin, D. B. (1999), *Small-sample degrees of freedom with multiple imputation*, Biometrika 86(4): 948–955.
* **Code Reference**: `umbra/imputers/rubin_pooler.py`, `umbra/api.py`, `benchmarks/`
* **Limitation**: When evaluating single-draw imputations ($M = 1$), naive plug-in standard errors ($\hat{\sigma} / \sqrt{N}$) underestimate sampling variability because they treat imputed values as observed data without accounting for between-imputation variance.
* **Implemented Solution (v0.2.0 / Phase 3)**:
  - **Rubin Pooling Engine**: Implemented `RubinPooler` and `rubins_rules` computing total variance $T = \bar{U} + (1 + 1/M) B$ and Barnard-Rubin (1999) small-sample adjusted degrees of freedom $\nu_m$.
  - **Multi-Draw Support Across All Imputers**: `impute_multiple(M)` is natively implemented in `MARChainedEquationsImputer`, `HeckmanSelectionImputer`, and `PatternMixtureImputer`.
  - **Honest Uncertainty Reporting**: Benchmark suites and coverage diagnostics explicitly contrast Rubin-pooled intervals ($M \ge 5$) against single-draw plug-in estimates.
* **Ongoing Boundary**: Multiple imputation requires proper draws from the posterior predictive distribution; with misspecified imputation models, pooled intervals may deviate from nominal frequentist coverage.

---

## 5. Uncalibrated Decision Threshold Heuristics in the Auto Router

* **Theoretical Foundation**: Platt, J. (1999), *Probabilistic Outputs for Support Vector Machines*; Zadrozny, B., & Elkan, C. (2002), *Transforming classifier scores into accurate multiclass probability estimates*, KDD.
* **Code Reference**: `umbra/diagnostics/risk_calibrator.py`, `umbra/api.py`
* **Limitation**: The composite missingness concern score synthesized by Umbra originally mapped into discrete risk tiers (LOW, MEDIUM, HIGH) using heuristic cutoffs (0.25 and 0.50).
* **Implemented Solution (v0.2.0 / Phase 4)**:
  - **Empirical Probability Calibration**: Implemented `MNARRiskCalibrator` supporting both Platt scaling (logistic sigmoid) and Isotonic regression calibration.
  - **Cost-Sensitive Bayes-Optimal Routing**: Implemented decision engine computing Bayes-optimal thresholds $\tau^* = \frac{L_{01}}{L_{01} + L_{10}}$ minimizing expected domain loss under asymmetric misclassification costs.
  - **Pluggable Decision Profiles**: Preconfigured profiles (`conservative_mnar`, `balanced`, `permissive_mar`) with JSON/YAML persistence and custom loss matrix specification.
* **Ongoing Boundary**: Empirical calibration is contingent on the calibration dataset's missingness distribution; users operating in novel domains should validate calibrated thresholds against domain loss functions.

---

## 6. Synthetic Data Generators vs. Observational Real-World Datasets

* **Theoretical Foundation**: Schouten, R. M., Lugtig, P., & Vink, G. (2018), *Generating missing values for simulation studies: A tool for generating realistic nonresponse*, Sociological Methods & Research 50(3): 1243–1277.
* **Code Reference**: `umbra/benchmark/amputation.py`, `umbra/data/loaders.py`, `benchmarks/observational_benchmark.py`
* **Limitation**: Validation suites relying purely on stylized synthetic data generators can overfit to parametric assumptions and fail to reflect complex dependencies in real-world data.
* **Implemented Solution (v0.2.0 / Phase 5)**:
  - **Multivariate Amputation Engine**: Implemented `ampute_multivariate` per Schouten et al. (2018) supporting MAR zero-weight invariants, MNAR self-masking, MCAR uniform dropout, and continuous odds distributions (`RIGHT`, `LEFT`, `MID`, `TAIL`) with Brent's method root-finding probability calibration.
  - **Authentic Observational Dataset Loaders**: Curated loaders (`load_cps_wage`, `load_nhanes_biomarkers`, `load_california_housing`, `load_clinical_trial_attrition`) supporting observed, complete, and both splits with deterministic offline fallbacks.
* **Ongoing Boundary**: Real-world observational datasets without ground-truth verification cannot definitively benchmark true imputation error; multivariate amputation on complete cases serves as the gold-standard evaluation protocol.
