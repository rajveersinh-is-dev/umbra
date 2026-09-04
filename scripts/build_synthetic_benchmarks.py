"""
Synthetic benchmark data generator for Missingness regimes:
- MCAR: Missing Completely at Random
- MAR: Missing at Random (conditioned on observed covariates)
- MNAR: Missing Not at Random (conditioned on unobserved target value)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SyntheticBenchmark:
    """Container for synthetic benchmark data with ground truth."""

    name: str
    mechanism: str
    severity: str
    data_complete: pd.DataFrame
    data_observed: pd.DataFrame
    mask: pd.Series
    target_col: str
    true_params: Dict[str, float]
    shadow_col: Optional[str] = None

    @property
    def missing_rate(self) -> float:
        return float(self.mask.mean())


def expit(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def generate_ground_truth_data(
    n_samples: int = 2000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Generate fully observed data with realistic structure.
    Mimics an economics / public health survey (e.g. income, age, education, health score).
    """
    rng = np.random.RandomState(random_state)

    # Correlated covariates
    # X1: Age (standardized ~ N(0, 1), e.g. 20-70)
    age = rng.normal(0, 1, size=n_samples)

    # X2: Education level (continuous latent factor)
    education = 0.5 * age + rng.normal(0, 0.866, size=n_samples)

    # X3: Health / Well-being score
    health = 0.3 * age + 0.4 * education + rng.normal(0, 0.8, size=n_samples)

    # Auxiliary / Shadow variable Z:
    # E.g., Interviewer contact difficulty or survey wave / form version
    # Correlated with missingness propensity, independent of income given age/education
    shadow_z = rng.normal(0, 1, size=n_samples)

    # True outcome equation: Target Y (e.g., Log-Income or symptom score)
    # Y = 10.0 + 0.5 * age + 0.8 * education + 0.3 * health + noise
    beta_0 = 10.0
    beta_age = 0.5
    beta_edu = 0.8
    beta_health = 0.3
    sigma_eps = 1.0

    epsilon = rng.normal(0, sigma_eps, size=n_samples)
    y = beta_0 + beta_age * age + beta_edu * education + beta_health * health + epsilon

    df = pd.DataFrame(
        {
            "age": age,
            "education": education,
            "health": health,
            "shadow_z": shadow_z,
            "income": y,
        }
    )

    true_params = {
        "true_mean": float(y.mean()),
        "true_std": float(y.std()),
        "beta_0": beta_0,
        "beta_age": beta_age,
        "beta_education": beta_edu,
        "beta_health": beta_health,
        "sigma": sigma_eps,
    }

    return df, true_params


def apply_missingness(
    df: pd.DataFrame,
    target_col: str = "income",
    mechanism: str = "MCAR",
    severity: str = "medium",
    missing_rate_target: float = 0.30,
    shadow_col: Optional[str] = "shadow_z",
    random_state: int = 42,
) -> SyntheticBenchmark:
    """
    Knock out values in target_col according to specified mechanism and severity.
    """
    rng = np.random.RandomState(random_state)
    df_obs = df.copy()
    y = df[target_col].values
    age = df["age"].values
    edu = df["education"].values
    z = df[shadow_col].values if shadow_col and shadow_col in df else np.zeros(len(df))

    n = len(df)

    if mechanism == "MCAR":
        # Missingness is completely independent of all data
        probs = np.full(n, missing_rate_target)
        mask = rng.uniform(0, 1, size=n) < probs

    elif mechanism == "MAR":
        # Missingness depends on observed covariates (age, education), NOT on income directly
        # Higher age and lower education increase probability of missingness
        latent = 0.8 * age - 0.9 * edu
        # calibrate intercept to achieve approx missing_rate_target
        intercept = np.quantile(latent, 1.0 - missing_rate_target)
        probs = expit(latent - intercept)
        mask = rng.uniform(0, 1, size=n) < probs

    elif mechanism == "MNAR":
        # Missingness depends directly on the unobserved target (e.g. high income earners hide income)
        # Severity controls slope on Y
        slope_map = {"low": 0.6, "medium": 1.5, "high": 2.8}
        gamma_y = slope_map.get(severity, 1.5)

        # Self-censoring: high income or extreme income skips reporting
        # Standardize Y for calibrated scaling
        y_std = (y - np.mean(y)) / np.std(y)

        # In realistic MNAR, there may also be an instrument / shadow effect
        latent = gamma_y * y_std + 0.3 * age + 0.7 * z
        intercept = np.quantile(latent, 1.0 - missing_rate_target)
        probs = expit(latent - intercept)
        mask = rng.uniform(0, 1, size=n) < probs

    elif mechanism == "MNAR_TAILS":
        # U-shaped missingness: both very low and very high income hide values
        y_std = (y - np.mean(y)) / np.std(y)
        latent = 1.2 * (y_std**2) + 0.5 * z
        intercept = np.quantile(latent, 1.0 - missing_rate_target)
        probs = expit(latent - intercept)
        mask = rng.uniform(0, 1, size=n) < probs

    else:
        raise ValueError(f"Unknown missingness mechanism: {mechanism}")

    # Set knocked out values to NaN
    df_obs.loc[mask, target_col] = np.nan

    true_params = {
        "true_mean": float(y.mean()),
        "true_std": float(y.std()),
        "observed_mean": float(df_obs[target_col].dropna().mean()),
        "missing_count": int(mask.sum()),
        "missing_rate": float(mask.mean()),
        "selection_bias": float(df_obs[target_col].dropna().mean() - y.mean()),
    }

    return SyntheticBenchmark(
        name=f"synthetic_{mechanism.lower()}_{severity}",
        mechanism=mechanism,
        severity=severity,
        data_complete=df,
        data_observed=df_obs,
        mask=pd.Series(mask, index=df.index, name=f"{target_col}_missing"),
        target_col=target_col,
        true_params=true_params,
        shadow_col=shadow_col,
    )


def generate_benchmark_battery(
    n_samples: int = 2000,
    random_state: int = 42,
) -> Dict[str, SyntheticBenchmark]:
    """Generate comprehensive battery of benchmarks across all regimes."""
    df_complete, _ = generate_ground_truth_data(n_samples=n_samples, random_state=random_state)

    battery = {}

    # 1. MCAR
    battery["MCAR"] = apply_missingness(
        df_complete, mechanism="MCAR", severity="none", random_state=random_state
    )

    # 2. MAR
    battery["MAR"] = apply_missingness(
        df_complete, mechanism="MAR", severity="standard", random_state=random_state
    )

    # 3. MNAR across severity levels
    for sev in ["low", "medium", "high"]:
        battery[f"MNAR_{sev.upper()}"] = apply_missingness(
            df_complete, mechanism="MNAR", severity=sev, random_state=random_state
        )

    # 4. MNAR Tails (U-shaped)
    battery["MNAR_TAILS"] = apply_missingness(
        df_complete, mechanism="MNAR_TAILS", severity="u_shaped", random_state=random_state
    )

    return battery


if __name__ == "__main__":
    print("Generating synthetic benchmarks battery...")
    battery = generate_benchmark_battery()
    for name, bench in battery.items():
        print(
            f"[{name}] Missing rate: {bench.missing_rate:.1%}, True Mean: {bench.true_params['true_mean']:.3f}, "
            f"Observed Mean: {bench.true_params['observed_mean']:.3f}, "
            f"Selection Bias: {bench.true_params['selection_bias']:+.3f}"
        )
