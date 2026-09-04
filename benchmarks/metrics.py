"""
Statistical & Uncertainty Quantification Metrics for Imputation Benchmarks.

Calculates:
- Mean Bias, Median Bias, Empirical Standard Error
- Root Mean Squared Error (RMSE), Mean Absolute Error (MAE)
- Empirical Coverage Probability for 80%, 90%, 95% Confidence Intervals
- Average Interval Width
- Downstream Parameter Recovery (regression beta bias, RMSE, coverage)
- Convergence / Failure Rate & Runtime
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ReplicationResult:
    """Metrics collected from a single Monte Carlo replication."""

    rep_idx: int
    imputed_mean: float
    imputed_cell_mean: float
    mean_bias: float
    cell_bias: float
    cell_rmse: float
    cell_mae: float
    ci_80_covered: bool
    ci_90_covered: bool
    ci_95_covered: bool
    ci_width_95: float
    beta_age: float
    beta_edu: float
    beta_age_covered: bool
    beta_edu_covered: bool
    runtime_sec: float
    converged: bool = True
    error_msg: Optional[str] = None


@dataclass
class MonteCarloSummary:
    """Aggregated Monte Carlo benchmark summary across R replications."""

    method_name: str
    regime: str
    n_samples: int
    missing_rate: float
    n_replications: int
    mean_bias: float
    median_bias: float
    empirical_sd: float
    cell_bias: float
    cell_rmse: float
    cell_mae: float
    coverage_80: float
    coverage_90: float
    coverage_95: float
    avg_ci_width_95: float
    beta_age_bias: float
    beta_edu_bias: float
    beta_total_rmse: float
    beta_age_coverage: float
    beta_edu_coverage: float
    convergence_rate: float
    avg_runtime_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method_name,
            "regime": self.regime,
            "n_samples": self.n_samples,
            "missing_rate": f"{self.missing_rate:.1%}",
            "n_replications": self.n_replications,
            "mean_bias": round(self.mean_bias, 4),
            "median_bias": round(self.median_bias, 4),
            "empirical_sd": round(self.empirical_sd, 4),
            "cell_bias": round(self.cell_bias, 4),
            "cell_rmse": round(self.cell_rmse, 4),
            "cell_mae": round(self.cell_mae, 4),
            "coverage_80": round(self.coverage_80, 3),
            "coverage_90": round(self.coverage_90, 3),
            "coverage_95": round(self.coverage_95, 3),
            "avg_ci_width_95": round(self.avg_ci_width_95, 4),
            "beta_age_bias": round(self.beta_age_bias, 4),
            "beta_edu_bias": round(self.beta_edu_bias, 4),
            "beta_total_rmse": round(self.beta_total_rmse, 4),
            "beta_age_coverage": round(self.beta_age_coverage, 3),
            "beta_edu_coverage": round(self.beta_edu_coverage, 3),
            "convergence_rate": round(self.convergence_rate, 3),
            "avg_runtime_sec": round(self.avg_runtime_sec, 4),
        }


def aggregate_replications(
    replications: List[ReplicationResult],
    method_name: str,
    regime: str,
    n_samples: int,
    missing_rate: float,
    true_beta_age: float = 0.5,
    true_beta_edu: float = 0.8,
) -> MonteCarloSummary:
    """Aggregate replication results into Monte Carlo statistics with coverage."""
    valid_reps = [r for r in replications if r.converged]
    n_valid = len(valid_reps)
    n_total = len(replications)

    if n_valid == 0:
        return MonteCarloSummary(
            method_name=method_name,
            regime=regime,
            n_samples=n_samples,
            missing_rate=missing_rate,
            n_replications=n_total,
            mean_bias=np.nan,
            median_bias=np.nan,
            empirical_sd=np.nan,
            cell_bias=np.nan,
            cell_rmse=np.nan,
            cell_mae=np.nan,
            coverage_80=0.0,
            coverage_90=0.0,
            coverage_95=0.0,
            avg_ci_width_95=np.nan,
            beta_age_bias=np.nan,
            beta_edu_bias=np.nan,
            beta_total_rmse=np.nan,
            beta_age_coverage=0.0,
            beta_edu_coverage=0.0,
            convergence_rate=0.0,
            avg_runtime_sec=np.nan,
        )

    biases = np.array([r.mean_bias for r in valid_reps])
    cell_biases = np.array([r.cell_bias for r in valid_reps])
    cell_rmses = np.array([r.cell_rmse for r in valid_reps])
    cell_maes = np.array([r.cell_mae for r in valid_reps])

    cov_80 = np.mean([r.ci_80_covered for r in valid_reps])
    cov_90 = np.mean([r.ci_90_covered for r in valid_reps])
    cov_95 = np.mean([r.ci_95_covered for r in valid_reps])
    avg_ci_w = np.mean([r.ci_width_95 for r in valid_reps])

    beta_ages = np.array([r.beta_age for r in valid_reps])
    beta_edus = np.array([r.beta_edu for r in valid_reps])
    beta_age_cov = np.mean([r.beta_age_covered for r in valid_reps])
    beta_edu_cov = np.mean([r.beta_edu_covered for r in valid_reps])

    beta_age_bias = np.mean(beta_ages - true_beta_age)
    beta_edu_bias = np.mean(beta_edus - true_beta_edu)
    beta_rmse = np.sqrt(
        np.mean((beta_ages - true_beta_age) ** 2 + (beta_edus - true_beta_edu) ** 2)
    )

    runtimes = np.array([r.runtime_sec for r in valid_reps])

    return MonteCarloSummary(
        method_name=method_name,
        regime=regime,
        n_samples=n_samples,
        missing_rate=missing_rate,
        n_replications=n_total,
        mean_bias=float(np.mean(biases)),
        median_bias=float(np.median(biases)),
        empirical_sd=float(np.std(biases, ddof=1)) if len(biases) > 1 else 0.0,
        cell_bias=float(np.mean(cell_biases)),
        cell_rmse=float(np.mean(cell_rmses)),
        cell_mae=float(np.mean(cell_maes)),
        coverage_80=float(cov_80),
        coverage_90=float(cov_90),
        coverage_95=float(cov_95),
        avg_ci_width_95=float(avg_ci_w),
        beta_age_bias=float(beta_age_bias),
        beta_edu_bias=float(beta_edu_bias),
        beta_total_rmse=float(beta_rmse),
        beta_age_coverage=float(beta_age_cov),
        beta_edu_coverage=float(beta_edu_cov),
        convergence_rate=float(n_valid / n_total),
        avg_runtime_sec=float(np.mean(runtimes)),
    )
