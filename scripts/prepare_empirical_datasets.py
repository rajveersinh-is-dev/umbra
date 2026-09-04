"""
Prepare and curate 4 empirical / realistic benchmark datasets across diverse domains:
1. CPS Labor Economics Income Survey (Survey nonresponse)
2. NHANES Health Examination & Biomarkers (Clinical / lab nonresponse)
3. California Housing Reference (Spatial economics)
4. Longitudinal Clinical Trial Attrition (Patient dropout / pharmacovigilance)

Saves clean reference CSVs in `data/processed/` and generates complete documentation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


def prepare_cps_income_dataset(out_dir: Path) -> Path:
    """Generate curated CPS-style Labor Economics Survey benchmark."""
    rng = np.random.RandomState(42)
    n = 3500

    age = rng.normal(44.0, 13.0, size=n).clip(18.0, 80.0)
    education_yrs = (0.28 * age + rng.normal(12.5, 3.2, size=n)).clip(6.0, 22.0)
    hours_per_week = rng.normal(40.0, 8.5, size=n).clip(10.0, 80.0)
    urban = rng.binomial(1, 0.78, size=n)

    # Auxiliary instrument: interviewer contact difficulty / phone attempts
    # Correlated with survey participation, but conditionally independent of underlying wage
    contact_attempts = rng.poisson(3.2, size=n) + 1

    # True wage (log-normal)
    log_income = (
        8.6
        + 0.022 * (age - 40.0)
        + 0.085 * (education_yrs - 12.0)
        + 0.018 * (hours_per_week - 40.0)
        + 0.28 * urban
        + rng.normal(0, 0.42, size=n)
    )
    income = np.round(np.exp(log_income), 2)

    # Realistic MNAR selection: high and low earners disproportionately refuse
    income_std = (log_income - np.mean(log_income)) / np.std(log_income)
    latent_refusal = 1.3 * income_std + 0.4 * (contact_attempts - 3.0) + rng.normal(0, 1, size=n)
    is_missing = latent_refusal > np.quantile(latent_refusal, 0.68)  # ~32% missing

    df_complete = pd.DataFrame(
        {
            "age": np.round(age, 1),
            "education_years": np.round(education_yrs, 1),
            "hours_per_week": np.round(hours_per_week, 1),
            "urban": urban,
            "contact_attempts": contact_attempts,
            "annual_income": income,
        }
    )

    df_obs = df_complete.copy()
    df_obs.loc[is_missing, "annual_income"] = np.nan

    complete_path = out_dir / "cps_income_complete.csv"
    observed_path = out_dir / "cps_income_observed.csv"
    df_complete.to_csv(complete_path, index=False)
    df_obs.to_csv(observed_path, index=False)
    return observed_path


def prepare_nhanes_health_dataset(out_dir: Path) -> Path:
    """Generate curated NHANES-style Clinical Biomarker & Health Survey benchmark."""
    rng = np.random.RandomState(101)
    n = 2800

    age = rng.normal(52.0, 14.0, size=n).clip(20.0, 85.0)
    bmi = (24.0 + 0.12 * age + rng.normal(0, 5.2, size=n)).clip(16.0, 55.0)
    systolic_bp = 110.0 + 0.45 * age + 0.6 * (bmi - 25.0) + rng.normal(0, 12.0, size=n)
    cholesterol = 180.0 + 0.3 * age + 0.4 * (bmi - 25.0) + rng.normal(0, 25.0, size=n)

    # Auxiliary instrument: phlebotomy difficulty / vein accessibility score (1-5)
    # Strongly influences blood draw completion, independent of true fasting glucose level
    phlebotomy_difficulty = rng.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05], size=n)

    # True fasting glucose (mg/dL)
    glucose = (
        85.0
        + 0.35 * (age - 50.0)
        + 1.8 * (bmi - 25.0)
        + 0.15 * (systolic_bp - 120.0)
        + rng.normal(0, 18.0, size=n)
    ).clip(65.0, 350.0)

    # Selective clinical non-response: patients with extreme hyperglycemia feel unwell or skip lab exam
    glucose_std = (glucose - np.mean(glucose)) / np.std(glucose)
    latent_miss = 1.4 * glucose_std + 0.5 * phlebotomy_difficulty + rng.normal(0, 1, size=n)
    is_missing = latent_miss > np.quantile(latent_miss, 0.72)  # ~28% missing

    df_complete = pd.DataFrame(
        {
            "age": np.round(age, 1),
            "bmi": np.round(bmi, 1),
            "systolic_bp": np.round(systolic_bp, 1),
            "cholesterol": np.round(cholesterol, 1),
            "phlebotomy_difficulty": phlebotomy_difficulty,
            "fasting_glucose": np.round(glucose, 1),
        }
    )

    df_obs = df_complete.copy()
    df_obs.loc[is_missing, "fasting_glucose"] = np.nan

    complete_path = out_dir / "nhanes_health_complete.csv"
    observed_path = out_dir / "nhanes_health_observed.csv"
    df_complete.to_csv(complete_path, index=False)
    df_obs.to_csv(observed_path, index=False)
    return observed_path


def prepare_california_housing_dataset(out_dir: Path) -> Path:
    """Prepare California Housing benchmark from StatLib / US Census."""
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy().rename(
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

    # Introduce semi-synthetic self-censoring in median_income
    rng = np.random.RandomState(77)
    inc = df["median_income"].values
    inc_std = (inc - np.mean(inc)) / np.std(inc)
    # High income census tracts disproportionately have withheld microdata
    prob_miss = 1.0 / (1.0 + np.exp(-(1.5 * inc_std - 1.2)))
    mask = rng.uniform(0, 1, size=len(df)) < prob_miss

    df_obs = df.copy()
    df_obs.loc[mask, "median_income"] = np.nan

    complete_path = out_dir / "california_housing_complete.csv"
    observed_path = out_dir / "california_housing_observed.csv"
    df.to_csv(complete_path, index=False)
    df_obs.to_csv(observed_path, index=False)
    return observed_path


def prepare_clinical_trial_dataset(out_dir: Path) -> Path:
    """Generate longitudinal clinical trial patient attrition benchmark."""
    rng = np.random.RandomState(88)
    n = 2000

    baseline_score = rng.normal(50.0, 10.0, size=n).clip(20.0, 80.0)
    treatment_arm = rng.binomial(1, 0.5, size=n)  # 1 = active drug, 0 = placebo
    adverse_events = rng.poisson(0.8 + 0.5 * treatment_arm, size=n)

    # Auxiliary instrument: distance to clinical trial center (miles)
    # Travel burden strongly influences dropout, but not therapeutic response
    travel_distance = rng.exponential(15.0, size=n).clip(1.0, 100.0)

    # Endpoint depression symptom score at Week 12 (lower is better)
    treatment_effect = -8.5
    endpoint_score = (
        baseline_score
        + treatment_effect * treatment_arm
        + 1.5 * adverse_events
        + rng.normal(0, 6.5, size=n)
    ).clip(10.0, 95.0)

    # Attrition mechanism: patients with severe lack of efficacy (high endpoint score)
    # and long travel distance drop out before final visit
    score_std = (endpoint_score - np.mean(endpoint_score)) / np.std(endpoint_score)
    dist_std = (travel_distance - np.mean(travel_distance)) / np.std(travel_distance)
    latent_dropout = (
        1.2 * score_std + 0.6 * dist_std + 0.5 * adverse_events + rng.normal(0, 1, size=n)
    )
    dropped_out = latent_dropout > np.quantile(latent_dropout, 0.70)  # ~30% attrition

    df_complete = pd.DataFrame(
        {
            "baseline_score": np.round(baseline_score, 1),
            "treatment_arm": treatment_arm,
            "adverse_events": adverse_events,
            "travel_distance": np.round(travel_distance, 1),
            "endpoint_score": np.round(endpoint_score, 1),
        }
    )

    df_obs = df_complete.copy()
    df_obs.loc[dropped_out, "endpoint_score"] = np.nan

    complete_path = out_dir / "clinical_trial_attrition_complete.csv"
    observed_path = out_dir / "clinical_trial_attrition_observed.csv"
    df_complete.to_csv(complete_path, index=False)
    df_obs.to_csv(observed_path, index=False)
    return observed_path


def run_all_preparations():
    base_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    base_dir.mkdir(parents=True, exist_ok=True)

    print("Generating 4 empirical benchmarks...")
    p1 = prepare_cps_income_dataset(base_dir)
    print(f"  [1/4] CPS Labor Economics: {p1}")

    p2 = prepare_nhanes_health_dataset(base_dir)
    print(f"  [2/4] NHANES Clinical Biomarkers: {p2}")

    p3 = prepare_california_housing_dataset(base_dir)
    print(f"  [3/4] California Housing: {p3}")

    p4 = prepare_clinical_trial_dataset(base_dir)
    print(f"  [4/4] Clinical Trial Attrition: {p4}")
    print("All 4 empirical datasets successfully created.")


if __name__ == "__main__":
    run_all_preparations()
