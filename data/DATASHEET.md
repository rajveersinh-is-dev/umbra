# Umbra Benchmark Datasets Datasheet

## 1. Motivation
Validating Not-Missing-At-Random (MNAR) imputation requires known ground truth because, by definition, the true missing values in unaugmented real-world datasets are unobserved. This datasheet documents the ground-truth benchmark datasets and synthetic missingness mechanisms used to evaluate Umbra.

## 2. Dataset Overview

### A. Synthetic Benchmark Battery (`scripts/build_synthetic_benchmarks.py`)
- **Sample Size**: 2,000 observations per regime.
- **Features**:
  - `age`: Continuous demographic feature ~ N(0, 1).
  - `education`: Correlated education factor (0.5 * age + N(0, 0.87)).
  - `health`: Self-reported health score (0.3 * age + 0.4 * education + N(0, 0.8)).
  - `shadow_z`: Auxiliary instrument (interviewer contact difficulty), independent of true income given demographics.
  - `income`: Continuous target variable with ground-truth linear structure $Y = 10.0 + 0.5 \cdot \text{age} + 0.8 \cdot \text{education} + 0.3 \cdot \text{health} + \epsilon$.
- **Mechanisms Imposed**:
  1. `MCAR`: Completely random Bernoulli dropout (~30% missing).
  2. `MAR`: Missingness logistic probability depends on `age` and `education`.
  3. `MNAR (Low, Medium, High)`: Missingness logistic probability depends directly on unobserved `income` with varying slope $\beta_Y \in \{0.6, 1.5, 2.8\}$ and instrument `shadow_z`.
  4. `MNAR Tails`: U-shaped missingness where both extreme low and extreme high values skip reporting.

### B. California Housing Complete Reference (`data/processed/california_housing_complete.csv`)
- **Source**: StatLib repository / US Census 1990 (via `sklearn.datasets.fetch_california_housing`).
- **License**: Public domain.
- **Observations**: 20,640 block groups.
- **Features**: `median_income`, `housing_age`, `ave_rooms`, `ave_bedrooms`, `population`, `ave_occupancy`, `median_house_val`.

### C. CPS-Style Income Survey (`data/processed/cps_income_survey_ground_truth.csv`)
- **Motivation**: Mimics the US Current Population Survey (CPS) Annual Social and Economic Supplement (ASEC), where earnings non-response has been extensively studied (e.g., Bollinger et al. 2019, Hokayem et al. 2015).
- **Features**: `age`, `education_years`, `hours_per_week`, `urban`, `contact_attempts` (instrument), `annual_income`.

## 3. Privacy and Ethical Considerations
No real personal identifiable information (PII) or confidential patient records are stored in this repository. All benchmarks are either public aggregate benchmarks or synthetically generated distributions based on published econometric specifications.
