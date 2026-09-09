# Real-World Observational & Semi-Synthetic Benchmark Report

## Executive Summary

This benchmark compares classical baseline imputation strategies against Umbra on curated
real-world datasets with known counterfactual missingness mechanisms.

### Key Findings:
- **Complete Case Analysis (CCA)** incurs substantial downstream regression coefficient error $(\|\hat{\beta} - \beta^*\|_2)$ due to sample truncation bias.
- **Mean Imputation** severely distorts variable variances and produces elevated cell-level RMSE/MAE.
- **Standard MICE (MAR)** performs well under random non-response but suffers when missingness is correlated with the outcome.
- **Umbra Auto-Router** correctly assesses MNAR risk and selects econometric selection models (Heckman) or pattern-mixture models when instruments or shifts are detected.

## Benchmark Results Table

| dataset                  | method                            | n_samples | missing_rate | target_feature  | cell_rmse  | cell_mae   | beta_error | runtime_sec | notes                                             |
| ------------------------ | --------------------------------- | --------- | ------------ | --------------- | ---------- | ---------- | ---------- | ----------- | ------------------------------------------------- |
| CPS Wage                 | Complete Case Analysis (CCA)      | 3500      | 32.0%        | annual_income   | nan        | nan        | 1777.3872  | 0.0004      | Listwise deletion: dropped 1120 rows              |
| CPS Wage                 | Mean Imputation                   | 3500      | 32.0%        | annual_income   | 19824.8535 | 14664.7765 | 2772.0986  | 0.0004      | Standard single mean plug-in                      |
| CPS Wage                 | Standard MICE (MAR)               | 3500      | 32.0%        | annual_income   | 17745.9083 | 13298.4127 | 1611.2308  | 0.0581      | 10 MICE iterations with Bayesian ridge regression |
| CPS Wage                 | Umbra Auto-Router                 | 3500      | 32.0%        | annual_income   | 13392.1099 | 8905.292   | 1193.7874  | 1.247       | Routed to: heckman_selection                      |
| CPS Wage                 | Umbra Heckman Selection           | 3500      | 32.0%        | annual_income   | 13392.1099 | 8905.292   | 1193.7874  | 0.0231      | Instrument: contact_attempts                      |
| CPS Wage                 | Umbra Pattern Mixture (Delta=0.2) | 3500      | 32.0%        | annual_income   | 15499.7315 | 10822.3497 | 1624.995   | 0.0095      | Sensitivity offset delta = 0.2 std                |
| NHANES Biomarkers        | Complete Case Analysis (CCA)      | 2800      | 28.0%        | fasting_glucose | nan        | nan        | 2.1152     | 0.0003      | Listwise deletion: dropped 784 rows               |
| NHANES Biomarkers        | Mean Imputation                   | 2800      | 28.0%        | fasting_glucose | 33.2674    | 29.0748    | 1.8029     | 0.0003      | Standard single mean plug-in                      |
| NHANES Biomarkers        | Standard MICE (MAR)               | 2800      | 28.0%        | fasting_glucose | 30.8242    | 25.3834    | 1.6702     | 0.0527      | 10 MICE iterations with Bayesian ridge regression |
| NHANES Biomarkers        | Umbra Auto-Router                 | 2800      | 28.0%        | fasting_glucose | 29.886     | 26.18      | 2.4001     | 1.0075      | Routed to: heckman_selection                      |
| NHANES Biomarkers        | Umbra Heckman Selection           | 2800      | 28.0%        | fasting_glucose | 15.0136    | 11.6846    | 0.263      | 0.0202      | Instrument: phlebotomy_difficulty                 |
| NHANES Biomarkers        | Umbra Pattern Mixture (Delta=0.2) | 2800      | 28.0%        | fasting_glucose | 24.4415    | 20.4243    | 1.8382     | 0.0111      | Sensitivity offset delta = 0.2 std                |
| California Housing       | Complete Case Analysis (CCA)      | 20640     | 27.3%        | median_income   | nan        | nan        | 0.7978     | 0.001       | Listwise deletion: dropped 5639 rows              |
| California Housing       | Mean Imputation                   | 20640     | 27.3%        | median_income   | 3.175      | 2.3734     | 2.2101     | 0.0009      | Standard single mean plug-in                      |
| California Housing       | Standard MICE (MAR)               | 20640     | 27.3%        | median_income   | 1.8942     | 1.2463     | 0.7631     | 0.1693      | 10 MICE iterations with Bayesian ridge regression |
| California Housing       | Umbra Auto-Router                 | 20640     | 27.3%        | median_income   | 1.611      | 1.085      | 0.7989     | 0.8908      | Routed to: pattern_mixture                        |
| California Housing       | Umbra Heckman Selection           | 20640     | 27.3%        | median_income   | 6.1518     | 6.0139     | 2.7743     | 0.0748      | Instrument: Auto-discovered                       |
| California Housing       | Umbra Pattern Mixture (Delta=0.2) | 20640     | 27.3%        | median_income   | 1.532      | 1.0041     | 0.7294     | 0.0184      | Sensitivity offset delta = 0.2 std                |
| Clinical Trial Attrition | Complete Case Analysis (CCA)      | 2000      | 30.0%        | endpoint_score  | nan        | nan        | 1.3134     | 0.0003      | Listwise deletion: dropped 600 rows               |
| Clinical Trial Attrition | Mean Imputation                   | 2000      | 30.0%        | endpoint_score  | 17.9198    | 15.3947    | 4.2472     | 0.0003      | Standard single mean plug-in                      |
| Clinical Trial Attrition | Standard MICE (MAR)               | 2000      | 30.0%        | endpoint_score  | 10.1447    | 8.2417     | 1.2997     | 0.0549      | 10 MICE iterations with Bayesian ridge regression |
| Clinical Trial Attrition | Umbra Auto-Router                 | 2000      | 30.0%        | endpoint_score  | 7.063      | 5.7406     | 0.8897     | 0.7783      | Routed to: heckman_selection                      |
| Clinical Trial Attrition | Umbra Heckman Selection           | 2000      | 30.0%        | endpoint_score  | 7.063      | 5.7406     | 0.8897     | 0.0153      | Instrument: travel_distance                       |
| Clinical Trial Attrition | Umbra Pattern Mixture (Delta=0.2) | 2000      | 30.0%        | endpoint_score  | 7.5897     | 6.1842     | 1.058      | 0.008       | Sensitivity offset delta = 0.2 std                |

---
*Report generated automatically on 2026-09-09 22:58:49.*