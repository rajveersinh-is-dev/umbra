"""
Scientifically Rigorous Data Generating Processes (DGPs) for Missing Data Evaluation.

Defines mathematically documented simulation mechanisms:
1. MCAR: Missing Completely at Random
2. MAR: Missing at Random (logistic selection on observed covariates)
3. MNAR_SELF_MASKING: Missingness depends directly on unobserved target
4. MNAR_SELECTION: Heckman-style bivariate latent threshold model with instrument
5. MNAR_PATTERN_MIXTURE: Distinct response groups with systematic mean shift delta
6. MNAR_TAILS: Non-linear U-shaped tail dropout
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def expit(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid function."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


@dataclass
class SimulationDataset:
    """Container for a generated simulation dataset with known ground truth parameters."""

    name: str
    mechanism: str
    n_samples: int
    missing_rate_nominal: float
    missing_rate_empirical: float
    data_complete: pd.DataFrame
    data_observed: pd.DataFrame
    mask: np.ndarray  # True if missing, False if observed
    target_col: str
    shadow_col: Optional[str]
    true_params: Dict[str, Any]
    dgp_formula: str


def generate_structural_data(
    n_samples: int = 2500,
    p_covariates: int = 3,
    correlation: float = 0.35,
    snr: float = 3.0,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate multivariate ground truth data with known linear structural outcome.

    Outcome Model:
      Y = beta_0 + beta_1 * X_1 + beta_2 * X_2 + beta_3 * X_3 + eps
      eps ~ N(0, sigma^2)
    Auxiliary Instrument:
      Z ~ N(0, 1), independent of eps conditional on X.
    """
    rng = np.random.RandomState(random_state)

    # Covariance matrix with compound symmetry / Toeplitz structure
    cov_matrix = np.eye(p_covariates)
    for i in range(p_covariates):
        for j in range(p_covariates):
            if i != j:
                cov_matrix[i, j] = correlation ** abs(i - j)

    X_covars = rng.multivariate_normal(mean=np.zeros(p_covariates), cov=cov_matrix, size=n_samples)

    # Shadow / Auxiliary instrument Z
    # Independent of outcome noise, correlated with covariates if desired
    shadow_z = rng.normal(0, 1, size=n_samples)

    # True coefficients
    beta_0 = 10.0
    beta_weights = np.array([0.5, 0.8, 0.3][:p_covariates])
    signal = X_covars @ beta_weights
    signal_var = np.var(signal)
    noise_var = signal_var / max(0.1, snr)
    sigma_eps = float(np.sqrt(noise_var))

    eps = rng.normal(0, sigma_eps, size=n_samples)
    y = beta_0 + signal + eps

    df = pd.DataFrame(
        {
            "age": X_covars[:, 0],
            "education": X_covars[:, 1],
            "health": X_covars[:, 2],
            "shadow_z": shadow_z,
            "target": y,
        }
    )

    true_params = {
        "true_mean": float(np.mean(y)),
        "true_std": float(np.std(y, ddof=1)),
        "beta_0": beta_0,
        "beta_age": float(beta_weights[0]),
        "beta_education": float(beta_weights[1]),
        "beta_health": float(beta_weights[2]),
        "sigma_eps": sigma_eps,
        "snr": snr,
    }

    return df, true_params


def generate_simulation_dataset(
    mechanism: str = "MCAR",
    n_samples: int = 2500,
    missing_rate: float = 0.30,
    severity: str = "medium",
    snr: float = 3.0,
    target_col: str = "target",
    shadow_col: str = "shadow_z",
    random_state: int = 42,
) -> SimulationDataset:
    """Generate a reproducible simulation dataset for a specified missingness mechanism.

    Supported mechanisms:
    - 'MCAR': Missing Completely at Random
    - 'MAR': Missing at Random (logistic on age and education)
    - 'MNAR_SELF_MASKING': Logistic on target Y (self-censoring)
    - 'MNAR_SELECTION': Heckman selection bivariate latent threshold model
    - 'MNAR_PATTERN_MIXTURE': Subpopulation mean shift delta
    - 'MNAR_TAILS': U-shaped non-linear tail dropout
    """
    rng = np.random.RandomState(random_state)
    df_complete, true_params = generate_structural_data(
        n_samples=n_samples, snr=snr, random_state=random_state
    )
    df_obs = df_complete.copy()
    y = df_complete[target_col].values
    age = df_complete["age"].values
    edu = df_complete["education"].values
    z = df_complete[shadow_col].values
    n = n_samples

    y_std = (y - np.mean(y)) / (np.std(y) + 1e-8)

    if mechanism.upper() == "MCAR":
        # Pure Bernoulli dropout
        mask = rng.uniform(0, 1, size=n) < missing_rate
        dgp_formula = f"P(R=0) = {missing_rate:.2f}"

    elif mechanism.upper() == "MAR":
        # Logistic probability dependent on age and education
        latent = 0.8 * age - 0.9 * edu
        intercept = np.quantile(latent, 1.0 - missing_rate)
        probs = expit(latent - intercept)
        mask = rng.uniform(0, 1, size=n) < probs
        dgp_formula = "logit P(R=0) = alpha_0 + 0.8 * age - 0.9 * education"

    elif mechanism.upper() in ("MNAR_SELF_MASKING", "MNAR_SELF"):
        # Self-censoring on unobserved target Y
        slope_map = {"low": 0.7, "medium": 1.6, "high": 3.0}
        gamma_y = slope_map.get(severity.lower(), 1.6)
        latent = gamma_y * y_std + 0.3 * age
        intercept = np.quantile(latent, 1.0 - missing_rate)
        probs = expit(latent - intercept)
        mask = rng.uniform(0, 1, size=n) < probs
        dgp_formula = f"logit P(R=0) = alpha_0 + {gamma_y:.1f} * Y_std + 0.3 * age"

    elif mechanism.upper() in ("MNAR_SELECTION", "MNAR_HECKMAN"):
        # Bivariate normal selection equation
        # z* = alpha_0 + 0.4 * age + 1.2 * shadow_z + u
        # (u, eps) have correlation rho
        rho_map = {"low": 0.35, "medium": 0.65, "high": 0.85}
        rho = rho_map.get(severity.lower(), 0.65)

        # Generate correlated bivariate error (u, eps)
        # Note: eps was already generated in y; we construct u correlated with eps
        sigma_eps = true_params["sigma_eps"]
        eps_std = (
            y - (true_params["beta_0"] + 0.5 * age + 0.8 * edu + 0.3 * df_complete["health"].values)
        ) / sigma_eps

        # u = rho * eps_std + sqrt(1 - rho^2) * eta
        eta = rng.normal(0, 1, size=n)
        u = rho * eps_std + np.sqrt(1.0 - rho**2) * eta

        # Latent selection index
        z_star = 0.4 * age + 1.2 * z + u
        # Calibrate intercept so proportion with z* <= 0 equals missing_rate
        cutoff = np.quantile(z_star, missing_rate)
        mask = z_star < cutoff
        dgp_formula = (
            f"R = 1(z* > 0), z* = alpha + 0.4*age + 1.2*shadow_z + u, Cor(u, eps)={rho:.2f}"
        )
        true_params["rho"] = rho

    elif mechanism.upper() in ("MNAR_PATTERN_MIXTURE", "MNAR_PM"):
        # Explicit group departure: non-responders have shifted mean
        delta_map = {"low": -0.4, "medium": -0.8, "high": -1.5}
        delta = delta_map.get(severity.lower(), -0.8)

        # Missing indicator assigned via latent trait
        latent = 0.5 * age + rng.normal(0, 1, size=n)
        cutoff = np.quantile(latent, 1.0 - missing_rate)
        mask = latent >= cutoff

        # In non-responders, ground truth Y is shifted by delta * sigma
        y_shifted = y.copy()
        y_shifted[mask] += delta * true_params["sigma_eps"]
        df_complete[target_col] = y_shifted
        true_params["true_mean"] = float(np.mean(y_shifted))
        true_params["delta"] = delta
        dgp_formula = f"Y | R=0 ~ N(X*beta + {delta:.1f}*sigma, sigma^2)"

    elif mechanism.upper() in ("MNAR_TAILS", "MNAR_NONLINEAR"):
        # U-shaped tail dropout
        latent = 1.4 * (y_std**2) + 0.3 * age
        intercept = np.quantile(latent, 1.0 - missing_rate)
        probs = expit(latent - intercept)
        mask = rng.uniform(0, 1, size=n) < probs
        dgp_formula = "logit P(R=0) = alpha_0 + 1.4 * (Y_std)^2 + 0.3 * age"

    elif mechanism.upper() in ("MNAR_WEAK_SIGNAL", "MNAR_WEAK"):
        # Weak selection signal: subtle departure from MAR, hard to distinguish from observables
        latent = 0.40 * y_std + 0.85 * age - 0.75 * edu
        intercept = np.quantile(latent, 1.0 - missing_rate)
        probs = expit(latent - intercept)
        mask = rng.uniform(0, 1, size=n) < probs
        dgp_formula = "logit P(R=0) = alpha_0 + 0.40 * Y_std + 0.85 * age - 0.75 * edu (Weak MNAR)"

    elif mechanism.upper() in ("MNAR_STRONG_SIGNAL", "MNAR_STRONG"):
        # Strong selection signal: massive unobserved self-masking
        latent = 2.80 * y_std + 0.35 * age
        intercept = np.quantile(latent, 1.0 - missing_rate)
        probs = expit(latent - intercept)
        mask = rng.uniform(0, 1, size=n) < probs
        dgp_formula = "logit P(R=0) = alpha_0 + 2.80 * Y_std + 0.35 * age (Strong MNAR)"

    elif mechanism.upper() in ("MNAR_MISSPECIFIED", "MNAR_NON_NORMAL"):
        # Misspecified selection model: heavy-tailed t(3) shocks and non-linear root transformation
        t_shock = rng.standard_t(df=3, size=n)
        z_star = 0.5 * age + 1.2 * z + 1.2 * np.sign(y_std) * np.sqrt(np.abs(y_std)) + 0.8 * t_shock
        cutoff = np.quantile(z_star, missing_rate)
        mask = z_star < cutoff
        dgp_formula = (
            "z* = 0.5*age + 1.2*z + 1.2*sign(Y)*sqrt(|Y|) + 0.8*t_3 (Misspecified Selection)"
        )

    else:
        raise ValueError(f"Unknown missingness mechanism: '{mechanism}'")

    # Apply missingness to observed copy
    df_obs.loc[mask, target_col] = np.nan

    obs_mean = float(df_obs[target_col].dropna().mean()) if (~mask).sum() > 0 else 0.0
    true_mean = float(true_params["true_mean"])
    selection_bias = obs_mean - true_mean

    true_params["observed_mean"] = obs_mean
    true_params["selection_bias"] = selection_bias
    true_params["missing_count"] = int(np.sum(mask))

    return SimulationDataset(
        name=f"{mechanism.lower()}_{severity.lower()}_n{n_samples}_p{int(missing_rate * 100)}",
        mechanism=mechanism.upper(),
        n_samples=n_samples,
        missing_rate_nominal=missing_rate,
        missing_rate_empirical=float(np.mean(mask)),
        data_complete=df_complete,
        data_observed=df_obs,
        mask=mask,
        target_col=target_col,
        shadow_col=shadow_col,
        true_params=true_params,
        dgp_formula=dgp_formula,
    )
