# Known Limitations and Open Problems

This document catalogues open methodological, statistical, and engineering limitations in Umbra v0.2.0. Rather than asserting completeness or artificial ratings, this record outlines where current assumptions or heuristics can degrade, where asymptotic theory requires care, and where ongoing research is needed.

---

## 1. Non-Identifiability of MNAR from Observed Data Alone

* **Theoretical Foundation**: Molenberghs et al. (2008), *Every missingness not at random model has a missingness at random counterpart with equal fit*, JRSS-B 70(2): 371–388.
* **Code Reference**: umbra/diagnostics/mnar_risk_score.py (lines 11–16, 385–430)
* **Limitation**: Observed data distributions contain information only about observable implications (e.g., covariate shifts between responders and non-responders, or global departures from MCAR via Little\'s test). If an MNAR process induces missingness purely as a function of the unobserved value {\\text{mis}}$ without affecting observable margins (e.g., pure unconfounded latent self-masking or symmetric tail dropout with balanced margins), observable diagnostics cannot distinguish this from MAR or even MCAR.
* **Practical Implication**: Low risk does not prove MAR. High risk indicates empirical tension with simple MAR/MCAR assumptions, but the true mechanism remains untestable without auxiliary assumptions or ground-truth follow-up data.

---

## 2. Heckman Selection Two-Step Standard Error Adjustments

* **Theoretical Foundation**: Murphy & Topel (1985), *Estimation and Inference with Two-Step Econometric Estimators*, JBES 3(4): 370–379; Cameron & Trivedi (2005), *Microeconometrics: Methods and Applications*, Cambridge University Press, Section 24.5.
* **Code Reference**: umbra/imputers/heckman_selection.py (lines 210–245)
* **Limitation**: The classical Heckman (1979) two-step estimator introduces a generated regressor—the estimated inverse Mills ratio $\\hat{\\lambda}_i = \\lambda(w_i \\hat{\\gamma})$—from the first-stage probit into the second-stage outcome regression. Ordinary least squares (OLS) standard errors from the second stage ignore the estimation variance of $\\hat{\\gamma}$ from the first stage and assume homoskedastic second-stage disturbances, which is violated under selection.
* **Mitigation & Remaining Boundary**: Paired bootstrap resampling across both stages is used to estimate parameter standard errors when requested (
_bootstrap_se > 0). Full-information maximum likelihood (FIML) is not implemented in the current release; two-step estimation remains sensitive to exclusion restriction strength and bivariate normality misspecification.

---

## 3. Near-Singular Design Matrices and Ridge Fallback Uncertainty

* **Code Reference**: umbra/imputers/heckman_selection.py (lines 215–225)
* **Limitation**: In small samples ( < 100$) or when severe multicollinearity arises between covariates $ and the inverse Mills ratio $\\lambda$ (frequently occurring when exclusion restrictions are weak or absent), second-stage OLS estimation can fail or produce unstable inversions.
* **Behavior**: Umbra provides an $-regularized Ridge fallback for point estimation to prevent unhandled runtime crashes, but emits a HeckmanSEWarning and sets parameter standard errors to NaN (rather than misleading zeros). Users must recognize that in this regime, formal parametric uncertainty quantification is unavailable.

---

## 4. Single-Imputation vs. Rubin-Pooled Multiple Imputation Variance in Benchmarking

* **Theoretical Foundation**: Rubin, D.B. (1987), *Multiple Imputation for Nonresponse in Surveys*, John Wiley & Sons; Barnard & Rubin (1999), *Small-sample degrees of freedom with multiple imputation*, Biometrika 86(4): 948–955.
* **Code Reference**: enchmarks/simulation_runner.py (lines 125–165)
* **Limitation**: When evaluating single-draw imputations ( = 1$), naive plug-in standard errors ($\\hat{\\sigma} / \\sqrt{N}$) underestimate sampling variability because they treat imputed values as observed data without accounting for between-imputation variance.
* **Standard**: Formal multiple-imputation benchmarks must use  \\ge 5$ stochastic draws pooled via Rubin\'s combining rules with small-sample degrees-of-freedom corrections. Benchmark comparisons that contrast single-imputation plug-in coverage with multiple-imputation literature are methodologically distinct and must be labeled explicitly.

---

## 5. Uncalibrated Decision Threshold Heuristics in the Auto Router

* **Code Reference**: umbra/diagnostics/mnar_risk_score.py (lines 390–415)
* **Limitation**: The composite missingness concern score synthesized by Umbra maps into discrete risk tiers (LOW, MEDIUM, HIGH) and strategy selections (mar_chained_equations, mnar_heckman, mnar_pattern_mixture_sensitivity) using heuristic score cutoffs (e.g., 0.25 and 0.50) and decision rules.
* **Impact**: While these thresholds achieve high separation on stylized synthetic data generators, they are heuristics rather than Bayes-optimal decision boundaries derived from an empirical risk minimization objective or calibrated under real-world cost matrices. Different domain loss functions (e.g., asymmetric penalties for missed MNAR vs. unnecessary sensitivity exploration) warrant user-specified recalibration via enchmarks/router_benchmark.py and scripts/calibrate_thresholds.py.

---

## 6. Synthetic Data Generators vs. Observational Real-World Datasets

* **Code Reference**: enchmarks/dgps.py
* **Limitation**: The validation suites in Umbra (including scenarios named after CPS labor economics, NHANES clinical biomarkers, and California Housing) are semi-synthetic data-generating processes whose marginal distributions and missingness functions are parametrically simulated.
* **Distinction**: They demonstrate algorithm behavior under controlled, mathematically known ground-truth mechanisms, but do not represent unconstrained observational datasets where the true mechanism is unknown and unverified. No claim of empirical real-world validation should be inferred from synthetic data generators alone.
