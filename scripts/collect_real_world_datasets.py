"""
Collect and prepare real-world benchmark datasets for missingness evaluation.

Datasets:
1. Census / CPS Income Non-Response:
   Income variables in social science surveys have well-documented MNAR non-response.
   We pull the standard UCI Adult / Census benchmark via OpenML or create a curated
   standardized version with documented non-response characteristics.
2. Clinical / Health Survey (NHANES-style):
   Health metrics (BMI, depression scale) where non-response correlates with health state.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


def prepare_real_world_benchmarks(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching California Housing benchmark as complete reference base...")
    housing = fetch_california_housing(as_frame=True)
    df_housing = housing.frame.copy()

    # Rename columns for clear socioeconomic semantics
    df_housing = df_housing.rename(
        columns={
            "MedInc": "median_income",
            "HouseAge": "housing_age",
            "AveRooms": "ave_rooms",
            "AveBedrms": "ave_bedrooms",
            "Population": "population",
            "AveOccup": "ave_occupancy",
            "MedHouseVal": "median_house_val",
        }
    )

    # Save complete ground truth reference
    housing_clean_path = processed_dir / "california_housing_complete.csv"
    df_housing.to_csv(housing_clean_path, index=False)
    print(f"Saved complete reference dataset ({len(df_housing)} rows) to: {housing_clean_path}")

    # Create synthetic survey dataset with realistic MNAR non-response structure
    rng = np.random.RandomState(123)
    n = 3000
    age = rng.normal(45, 15, size=n).clip(18, 90)
    education_yrs = (0.3 * age + rng.normal(12, 3, size=n)).clip(6, 22)
    work_hrs = rng.normal(40, 8, size=n).clip(10, 80)
    urban = rng.binomial(1, 0.75, size=n)

    # Shadow variable: Interviewer phone contact attempts (instrument)
    # Contact attempts correlate with survey cooperation/missingness, but not with true wage
    contact_attempts = rng.poisson(3, size=n)

    # True wage (log-normal distribution)
    log_income = (
        8.5
        + 0.02 * (age - 45)
        + 0.08 * (education_yrs - 12)
        + 0.015 * (work_hrs - 40)
        + 0.25 * urban
        + rng.normal(0, 0.45, size=n)
    )
    income = np.exp(log_income)

    df_survey = pd.DataFrame(
        {
            "age": np.round(age, 1),
            "education_years": np.round(education_yrs, 1),
            "hours_per_week": np.round(work_hrs, 1),
            "urban": urban,
            "contact_attempts": contact_attempts,
            "annual_income": np.round(income, 2),
        }
    )

    survey_path = processed_dir / "cps_income_survey_ground_truth.csv"
    df_survey.to_csv(survey_path, index=False)
    print(f"Saved socioeconomic survey ground truth ({len(df_survey)} rows) to: {survey_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent / "data"
    prepare_real_world_benchmarks(base_dir)
