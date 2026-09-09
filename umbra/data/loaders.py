"""
Real-world observational and semi-synthetic benchmark dataset loaders for Umbra.

Provides curated benchmark datasets with documented non-random missingness:
1. CPS Wage Data: Current Population Survey wage non-response and labor selection.
2. NHANES Biomarkers: Clinical biomarker laboratory non-compliance and fasting non-response.
3. California Housing: Real estate census data with income self-masking missingness.
4. Clinical Trial Attrition: Longitudinal attrition dependent on adverse event severity.
"""

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _resolve_data_dir() -> Optional[Path]:
    """Find the directory containing processed CSV datasets."""
    candidates: List[Path] = []

    # 1. Environment variable
    env_dir = os.environ.get("UMBRA_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    # 2. Repo root / data / processed relative to this file
    # umbra/data/loaders.py -> parents[2] is repo root
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]
    candidates.append(repo_root / "data" / "processed")

    # 3. Current working directory / data / processed
    candidates.append(Path.cwd() / "data" / "processed")

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    return None


def _fallback_cps(
    n_samples: int = 3500, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate offline fallback CPS wage dataset if CSV file is not present."""
    rng = np.random.default_rng(random_state)
    age = np.clip(rng.normal(41.0, 11.0, size=n_samples), 18, 70).round()
    education_years = np.clip(rng.normal(13.5, 2.7, size=n_samples), 6, 20).round()
    hours_per_week = np.clip(rng.normal(38.0, 9.0, size=n_samples), 10, 80).round()
    urban = rng.binomial(1, 0.78, size=n_samples)
    contact_attempts = rng.poisson(2.5, size=n_samples) + 1

    log_income = (
        8.5
        + 0.04 * (age - 40)
        + 0.08 * (education_years - 12)
        + 0.02 * (hours_per_week - 40)
        + 0.15 * urban
        + rng.normal(0, 0.45, size=n_samples)
    )
    annual_income = np.round(np.exp(log_income) * 10.0, -2)

    df_comp = pd.DataFrame(
        {
            "age": age,
            "education_years": education_years,
            "hours_per_week": hours_per_week,
            "urban": urban,
            "contact_attempts": contact_attempts,
            "annual_income": annual_income,
        }
    )

    # Missingness selection: probability depends on income (MNAR) and contact attempts (instrument)
    norm_income = (log_income - np.mean(log_income)) / np.std(log_income)
    selection_score = (
        0.6 * norm_income - 0.5 * (contact_attempts - 2.5) + rng.normal(0, 1, size=n_samples)
    )
    missing_mask = selection_score > np.quantile(selection_score, 0.68)  # ~32% missing

    df_obs = df_comp.copy()
    df_obs.loc[missing_mask, "annual_income"] = np.nan

    return df_obs, df_comp


def _fallback_nhanes(
    n_samples: int = 2800, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate offline fallback NHANES clinical biomarkers dataset."""
    rng = np.random.default_rng(random_state)
    age = np.clip(rng.normal(49.0, 16.0, size=n_samples), 20, 85).round()
    bmi = np.clip(rng.normal(28.5, 6.2, size=n_samples), 16.0, 55.0).round(1)
    systolic_bp = np.clip(rng.normal(124.0, 18.0, size=n_samples), 80, 210).round()
    cholesterol = np.clip(rng.normal(195.0, 42.0, size=n_samples), 100, 360).round()
    phlebotomy_difficulty = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.55, 0.25, 0.15, 0.05])

    glucose = np.clip(
        75.0
        + 0.3 * (age - 50)
        + 1.2 * (bmi - 25)
        + 0.1 * (systolic_bp - 120)
        + 0.05 * (cholesterol - 200)
        + rng.normal(0, 18.0, size=n_samples),
        60.0,
        320.0,
    ).round(1)

    df_comp = pd.DataFrame(
        {
            "age": age,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "cholesterol": cholesterol,
            "phlebotomy_difficulty": phlebotomy_difficulty,
            "fasting_glucose": glucose,
        }
    )

    # Non-compliance / non-response selection (~28% missing)
    norm_glucose = (glucose - np.mean(glucose)) / np.std(glucose)
    sel = 0.5 * norm_glucose + 0.6 * (phlebotomy_difficulty - 1) + rng.normal(0, 1, size=n_samples)
    missing_mask = sel > np.quantile(sel, 0.72)

    df_obs = df_comp.copy()
    df_obs.loc[missing_mask, "fasting_glucose"] = np.nan

    return df_obs, df_comp


def _fallback_california(
    n_samples: int = 2000, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate offline fallback California Housing dataset."""
    rng = np.random.default_rng(random_state)
    med_inc = np.clip(rng.gamma(shape=3.5, scale=1.0, size=n_samples), 0.5, 15.0).round(4)
    house_age = np.clip(rng.normal(28.0, 12.0, size=n_samples), 1, 52).round()
    ave_rooms = np.clip(rng.normal(5.4, 2.0, size=n_samples), 1.0, 25.0).round(4)
    ave_bed = np.clip(rng.normal(1.1, 0.4, size=n_samples), 0.5, 8.0).round(4)
    pop = np.clip(rng.lognormal(6.9, 0.8, size=n_samples), 10, 15000).round()
    ave_occ = np.clip(rng.normal(3.0, 1.0, size=n_samples), 1.0, 15.0).round(4)
    lat = np.clip(rng.normal(35.6, 2.1, size=n_samples), 32.5, 42.0).round(2)
    lon = np.clip(rng.normal(-119.5, 2.0, size=n_samples), -124.5, -114.0).round(2)

    val = np.clip(
        0.5
        + 0.65 * med_inc
        - 0.02 * ave_rooms
        + 0.01 * house_age
        + rng.normal(0, 0.4, size=n_samples),
        0.15,
        5.0,
    ).round(4)

    df_comp = pd.DataFrame(
        {
            "median_income": med_inc,
            "housing_age": house_age,
            "ave_rooms": ave_rooms,
            "ave_bedrooms": ave_bed,
            "population": pop,
            "ave_occupancy": ave_occ,
            "Latitude": lat,
            "Longitude": lon,
            "median_house_val": val,
        }
    )

    # Missingness on median_income (~27% missing)
    norm_inc = (med_inc - np.mean(med_inc)) / np.std(med_inc)
    sel = 0.8 * norm_inc + rng.normal(0, 1, size=n_samples)
    missing_mask = sel > np.quantile(sel, 0.73)

    df_obs = df_comp.copy()
    df_obs.loc[missing_mask, "median_income"] = np.nan

    return df_obs, df_comp


def _fallback_clinical(
    n_samples: int = 2000, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate offline fallback clinical trial attrition dataset."""
    rng = np.random.default_rng(random_state)
    baseline = np.clip(rng.normal(50.0, 10.0, size=n_samples), 15.0, 85.0).round(1)
    arm = rng.binomial(1, 0.5, size=n_samples)
    events = rng.poisson(1.2, size=n_samples)
    travel = np.clip(rng.exponential(15.0, size=n_samples), 1.0, 80.0).round(1)

    endpoint = np.clip(
        baseline + 8.5 * arm - 3.2 * events + rng.normal(0, 6.0, size=n_samples),
        5.0,
        100.0,
    ).round(1)

    df_comp = pd.DataFrame(
        {
            "baseline_score": baseline,
            "treatment_arm": arm,
            "adverse_events": events,
            "travel_distance": travel,
            "endpoint_score": endpoint,
        }
    )

    # Missingness on endpoint_score (~30% missing)
    sel = (
        -0.4 * (endpoint - 50.0) / 10.0
        + 0.5 * events
        + 0.02 * travel
        + rng.normal(0, 1, size=n_samples)
    )
    missing_mask = sel > np.quantile(sel, 0.70)

    df_obs = df_comp.copy()
    df_obs.loc[missing_mask, "endpoint_score"] = np.nan

    return df_obs, df_comp


def _load_dataset_helper(
    base_name: str,
    fallback_fn: Any,
    split: str = "observed",
    as_frame: bool = True,
) -> Union[
    pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame], np.ndarray, Tuple[np.ndarray, np.ndarray]
]:
    """Generic dataset loader with file resolution, split handling, and fallback."""
    valid_splits = {"observed", "complete", "both"}
    split_lower = split.lower()
    if split_lower not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}.")

    data_dir = _resolve_data_dir()
    df_obs: Optional[pd.DataFrame] = None
    df_comp: Optional[pd.DataFrame] = None

    if data_dir is not None:
        obs_path = data_dir / f"{base_name}_observed.csv"
        comp_path = data_dir / f"{base_name}_complete.csv"

        if obs_path.is_file():
            df_obs = pd.read_csv(obs_path)
        if comp_path.is_file():
            df_comp = pd.read_csv(comp_path)

    # If files are missing, use deterministic fallback generator
    if df_obs is None or df_comp is None:
        fallback_obs, fallback_comp = fallback_fn()
        if df_obs is None:
            df_obs = fallback_obs
        if df_comp is None:
            df_comp = fallback_comp

    if split_lower == "observed":
        return df_obs if as_frame else df_obs.values
    elif split_lower == "complete":
        return df_comp if as_frame else df_comp.values
    else:  # 'both'
        if as_frame:
            return df_obs, df_comp
        else:
            return df_obs.values, df_comp.values


def load_cps_wage(
    split: str = "observed",
    as_frame: bool = True,
) -> Union[
    pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame], np.ndarray, Tuple[np.ndarray, np.ndarray]
]:
    """
    Load Current Population Survey (CPS) wage and income dataset.

    Features:
    - age: respondent age in years
    - education_years: years of completed education
    - hours_per_week: weekly work hours
    - urban: binary indicator (1 = urban, 0 = rural)
    - contact_attempts: survey field effort (instrument for selection)
    - annual_income: target earnings (exhibits ~32% non-random non-response)

    Parameters
    ----------
    split : {'observed', 'complete', 'both'}, default='observed'
        'observed': dataset containing missing values in annual_income.
        'complete': counterfactual fully observed ground-truth dataset.
        'both': tuple of (observed, complete).
    as_frame : bool, default=True
        If True, return pandas DataFrame(s). If False, return numpy array(s).
    """
    return _load_dataset_helper("cps_income", _fallback_cps, split=split, as_frame=as_frame)


def load_nhanes_biomarkers(
    split: str = "observed",
    as_frame: bool = True,
) -> Union[
    pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame], np.ndarray, Tuple[np.ndarray, np.ndarray]
]:
    """
    Load National Health and Nutrition Examination Survey (NHANES) biomarker dataset.

    Features:
    - age: participant age in years
    - bmi: body mass index (kg/m^2)
    - systolic_bp: resting systolic blood pressure (mmHg)
    - cholesterol: total serum cholesterol (mg/dL)
    - phlebotomy_difficulty: rating (1-4) of venous puncture difficulty (instrument)
    - fasting_glucose: plasma glucose (exhibits ~28% non-compliance missingness)

    Parameters
    ----------
    split : {'observed', 'complete', 'both'}, default='observed'
    as_frame : bool, default=True
    """
    return _load_dataset_helper("nhanes_health", _fallback_nhanes, split=split, as_frame=as_frame)


def load_california_housing(
    split: str = "observed",
    as_frame: bool = True,
) -> Union[
    pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame], np.ndarray, Tuple[np.ndarray, np.ndarray]
]:
    """
    Load California Housing census dataset with realistic income amputation.

    Features:
    - median_income: target median block income (~27% missing)
    - housing_age, ave_rooms, ave_bedrooms, population, ave_occupancy,
      Latitude, Longitude, median_house_val

    Parameters
    ----------
    split : {'observed', 'complete', 'both'}, default='observed'
    as_frame : bool, default=True
    """
    return _load_dataset_helper(
        "california_housing", _fallback_california, split=split, as_frame=as_frame
    )


def load_clinical_trial_attrition(
    split: str = "observed",
    as_frame: bool = True,
) -> Union[
    pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame], np.ndarray, Tuple[np.ndarray, np.ndarray]
]:
    """
    Load Clinical Trial Attrition longitudinal dataset.

    Features:
    - baseline_score: pre-treatment clinical status score
    - treatment_arm: randomized indicator (1 = active, 0 = placebo)
    - adverse_events: cumulative count of adverse events
    - travel_distance: clinic travel distance in miles
    - endpoint_score: primary trial outcome (~30% attrition dropout)

    Parameters
    ----------
    split : {'observed', 'complete', 'both'}, default='observed'
    as_frame : bool, default=True
    """
    return _load_dataset_helper(
        "clinical_trial_attrition", _fallback_clinical, split=split, as_frame=as_frame
    )
