"""
MNAR Sensitivity Grid Analysis and Tipping Point Detection.

Evaluates how summary statistics and downstream model conclusions shift across a
spectrum of untestable MNAR assumptions.

Rather than providing a single false-confidence point estimate under an unverified
MAR assumption, this module computes the honest sensitivity interval and tests
whether findings are fragile to unobserved selection.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import pandas as pd
from sklearn.linear_model import LinearRegression

from umbra.imputers.pattern_mixture import PatternMixtureImputer


@dataclass
class TippingPoint:
    """Represents a critical assumption threshold where a conclusion changes."""

    metric_name: str
    tipping_delta: float
    original_value: float
    tipping_value: float
    description: str


@dataclass
class SensitivityReport:
    """Comprehensive sensitivity report across MNAR assumption grid."""

    target_column: str
    grid_df: pd.DataFrame
    mar_baseline_estimate: float
    estimate_min: float
    estimate_max: float
    uncertainty_spread: float
    tipping_points: List[TippingPoint]
    is_fragile: bool
    interpretation: str

    def summary(self) -> str:
        lines = [
            "==================================================================",
            f"Sensitivity Grid Analysis for '{self.target_column}'",
            f"  - MAR Baseline Estimate (delta=0)   : {self.mar_baseline_estimate:.4f}",
            f"  - Plausible Sensitivity Interval    : [{self.estimate_min:.4f}, {self.estimate_max:.4f}]",
            f"  - Uncertainty Spread (Max - Min)    : {self.uncertainty_spread:.4f}",
            f"  - Conclusion Fragility              : {'FRAGILE (Overturned within plausible grid)' if self.is_fragile else 'ROBUST (Stable across plausible grid)'}",
        ]
        if self.tipping_points:
            lines.append("  - Identified Tipping Points:")
            for tp in self.tipping_points:
                lines.append(
                    f"    * [{tp.metric_name}] flips at delta = {tp.tipping_delta:+.2f} ({tp.description})"
                )
        else:
            lines.append(
                "  - No tipping points found: qualitative conclusion holds across entire grid."
            )
        lines.append(f"  - Interpretation:\n    {self.interpretation}")
        lines.append("==================================================================")
        return "\n".join(lines)


def run_sensitivity_grid(
    data: pd.DataFrame,
    target_column: str,
    delta_grid: Optional[List[float]] = None,
    downstream_evaluator: Optional[Callable[[pd.DataFrame], float]] = None,
    downstream_feature: Optional[str] = None,
    downstream_outcome: Optional[str] = None,
    shift_type: str = "standardized",
    random_state: int = 42,
) -> SensitivityReport:
    """
    Run an MNAR sensitivity sweep across a grid of delta values.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with missing values in target_column.
    target_column : str
        The variable to evaluate sensitivity for.
    delta_grid : Optional[List[float]], default=None
        Grid of delta values to sweep. Defaults to 11 points in [-1.5, +1.5] standard deviations.
    downstream_evaluator : Optional[Callable[[pd.DataFrame], float]]
        Custom function that takes an imputed DataFrame and returns a scalar metric
        (e.g., a regression coefficient, policy estimate, or difference in means).
    downstream_feature : Optional[str]
        If provided along with downstream_outcome, fits an OLS regression predicting
        downstream_outcome from downstream_feature and returns the slope coefficient.
    downstream_outcome : Optional[str]
        Outcome variable for default regression evaluation.
    shift_type : str, default='standardized'
        Shift type for PatternMixtureImputer ('standardized', 'percentage', 'raw').
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    SensitivityReport
    """
    if target_column not in data.columns:
        raise ValueError(f"Column '{target_column}' not found in data.")

    if delta_grid is None:
        delta_grid = [-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]

    # Sort grid
    delta_grid = sorted(delta_grid)

    records = []
    mar_estimate = None

    for delta in delta_grid:
        imputer = PatternMixtureImputer(
            delta=delta,
            shift_type=shift_type,
            target_cols=[target_column],
            stochastic=False,
            random_state=random_state,
        )
        df_imputed = imputer.fit_transform(data)

        target_vals = df_imputed[target_column]
        mean_val = float(target_vals.mean())
        median_val = float(target_vals.median())
        q10_val = float(target_vals.quantile(0.10))
        q90_val = float(target_vals.quantile(0.90))
        std_val = float(target_vals.std())

        # Evaluate downstream metric
        downstream_metric = None
        if downstream_evaluator is not None:
            downstream_metric = float(downstream_evaluator(df_imputed))
        elif downstream_outcome and downstream_feature:
            # Fit OLS
            X = df_imputed[[downstream_feature]].to_numpy()
            y = df_imputed[downstream_outcome].to_numpy()
            reg = LinearRegression().fit(X, y)
            downstream_metric = float(reg.coef_[0])
        else:
            # Default downstream metric is the imputed mean itself
            downstream_metric = mean_val

        if abs(delta) < 1e-6:
            mar_estimate = downstream_metric

        records.append(
            {
                "delta": delta,
                "target_mean": mean_val,
                "target_median": median_val,
                "target_std": std_val,
                "target_q10": q10_val,
                "target_q90": q90_val,
                "downstream_metric": downstream_metric,
            }
        )

    grid_df = pd.DataFrame(records)

    if mar_estimate is None:
        mar_estimate = float(grid_df.loc[grid_df["delta"].abs().idxmin(), "downstream_metric"])

    # Extract bounds within plausible range [-1.0, +1.0]
    plausible_mask = grid_df["delta"].between(-1.0, 1.0)
    plausible_metrics = grid_df.loc[plausible_mask, "downstream_metric"]

    est_min = float(plausible_metrics.min())
    est_max = float(plausible_metrics.max())
    spread = float(est_max - est_min)

    # Detect tipping points:
    # 1. Sign flip in downstream metric
    tipping_points = []
    is_fragile = False

    metrics = grid_df["downstream_metric"].values
    deltas = grid_df["delta"].values

    for i in range(len(deltas) - 1):
        m1, m2 = metrics[i], metrics[i + 1]
        d1, d2 = deltas[i], deltas[i + 1]
        # Check for zero crossing / sign flip
        if (m1 * m2 < 0) or (m1 == 0 and m2 != 0):
            # Linear interpolation for zero crossing
            tipping_d = d1 if (m2 - m1) == 0 else d1 - m1 * (d2 - d1) / (m2 - m1)
            is_fragile = bool(abs(tipping_d) <= 1.0)
            tipping_points.append(
                TippingPoint(
                    metric_name="sign_flip",
                    tipping_delta=float(tipping_d),
                    original_value=mar_estimate,
                    tipping_value=0.0,
                    description=f"Downstream metric crosses zero at delta={tipping_d:+.2f}",
                )
            )

    if is_fragile:
        interpretation = (
            f"CRITICAL WARNING: The downstream conclusion flips sign within the plausible "
            f"MNAR range (at delta={tipping_points[0].tipping_delta:+.2f} std devs). "
            f"Under a standard MAR assumption, the estimate is {mar_estimate:.3f}, but if non-responders "
            f"differ by even modest unobserved amounts, the true sign reverses. "
            f"Do not report a single point estimate."
        )
    elif spread > 0.5 * abs(mar_estimate + 1e-6):
        interpretation = (
            f"MODERATE SENSITIVITY: While the sign remains consistent, the magnitude varies "
            f"substantially from {est_min:.3f} to {est_max:.3f} across plausible MNAR assumptions. "
            f"Presenting this sensitivity interval alongside the MAR estimate is strongly advised."
        )
    else:
        interpretation = (
            f"ROBUST CONCLUSION: The estimate ranges tightly between {est_min:.3f} and {est_max:.3f} "
            f"across the [-1.0, +1.0] delta grid. The finding is resilient against moderate MNAR mechanisms."
        )

    return SensitivityReport(
        target_column=target_column,
        grid_df=grid_df,
        mar_baseline_estimate=mar_estimate,
        estimate_min=est_min,
        estimate_max=est_max,
        uncertainty_spread=spread,
        tipping_points=tipping_points,
        is_fragile=is_fragile,
        interpretation=interpretation,
    )
