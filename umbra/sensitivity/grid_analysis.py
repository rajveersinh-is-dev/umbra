"""
MNAR Sensitivity Grid Analysis & Tipping Point Detection.

Evaluates how summary statistics and downstream scientific conclusions shift across
a spectrum of untestable MNAR assumptions.

Rather than providing a single false-confidence point estimate under an unverified
MAR assumption, this module computes the honest sensitivity interval [theta_min, theta_max]
and identifies tipping points: "How much MNAR departure is required to overturn the conclusion?"
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from umbra.imputers.pattern_mixture import PatternMixtureImputer


@dataclass
class TippingPoint:
    """Represents a critical assumption threshold where a scientific conclusion changes.

    Attributes
    ----------
    metric_name : str
        Type of tipping event ('sign_flip', 'ci_crosses_zero', 'policy_threshold').
    tipping_delta : float
        The delta value at which the tipping event occurs.
    original_value : float
        Baseline estimate under MAR (delta=0).
    tipping_value : float
        Metric value at the tipping threshold (e.g. 0.0 or decision threshold).
    description : str
        Human-readable explanation of the tipping event.
    """

    metric_name: str
    tipping_delta: float
    original_value: float
    tipping_value: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "tipping_delta": float(self.tipping_delta),
            "original_value": float(self.original_value),
            "tipping_value": float(self.tipping_value),
            "description": self.description,
        }


@dataclass
class SensitivityReport:
    """Comprehensive sensitivity analysis report across an MNAR assumption grid.

    Attributes
    ----------
    target_column : str
        The variable evaluated for MNAR sensitivity.
    grid_df : pd.DataFrame
        Table of delta values, estimates, standard errors, and confidence bounds.
    mar_baseline_estimate : float
        Point estimate under MAR (delta = 0.0).
    estimate_min : float
        Minimum estimate across the plausible grid [-1.0, +1.0].
    estimate_max : float
        Maximum estimate across the plausible grid [-1.0, +1.0].
    uncertainty_spread : float
        Difference (estimate_max - estimate_min).
    tipping_points : List[TippingPoint]
        List of identified tipping points.
    is_fragile : bool
        True if the conclusion flips sign or significance within plausible grid [-1.0, +1.0].
    interpretation : str
        Scientifically calibrated narrative.
    """

    target_column: str
    grid_df: pd.DataFrame
    mar_baseline_estimate: float
    estimate_min: float
    estimate_max: float
    uncertainty_spread: float
    tipping_points: List[TippingPoint] = field(default_factory=list)
    is_fragile: bool = False
    interpretation: str = ""

    def summary(self) -> str:
        lines = [
            "==================================================================",
            f"MNAR Sensitivity Grid Analysis for '{self.target_column}'",
            f"  - MAR Baseline Estimate (delta=0)   : {self.mar_baseline_estimate:.4f}",
            f"  - Plausible Sensitivity Interval    : [{self.estimate_min:.4f}, {self.estimate_max:.4f}]",
            f"  - Uncertainty Spread (Max - Min)    : {self.uncertainty_spread:.4f}",
            f"  - Conclusion Stability              : {'FRAGILE (Tipping point within plausible grid)' if self.is_fragile else 'ROBUST (Stable across plausible grid)'}",
        ]
        if self.tipping_points:
            lines.append("  - Identified Tipping Points:")
            for tp in self.tipping_points:
                lines.append(
                    f"    * [{tp.metric_name}] at delta = {tp.tipping_delta:+.2f} ({tp.description})"
                )
        else:
            lines.append(
                "  - No tipping points found: qualitative conclusion holds across entire grid."
            )
        lines.append(f"  - Methodological Interpretation:\n    {self.interpretation}")
        lines.append("==================================================================")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_column": self.target_column,
            "mar_baseline_estimate": float(self.mar_baseline_estimate),
            "estimate_min": float(self.estimate_min),
            "estimate_max": float(self.estimate_max),
            "uncertainty_spread": float(self.uncertainty_spread),
            "is_fragile": bool(self.is_fragile),
            "interpretation": self.interpretation,
            "tipping_points": [tp.to_dict() for tp in self.tipping_points],
            "grid_df": self.grid_df.to_dict(orient="records"),
        }


def run_sensitivity_grid(
    data: pd.DataFrame,
    target_column: str,
    delta_grid: Optional[List[float]] = None,
    downstream_evaluator: Optional[Callable[[pd.DataFrame], float]] = None,
    downstream_feature: Optional[str] = None,
    downstream_outcome: Optional[str] = None,
    decision_threshold: Optional[float] = None,
    shift_type: str = "standardized",
    alpha: float = 0.05,
    random_state: int = 42,
) -> SensitivityReport:
    """Run an MNAR sensitivity sweep across a grid of delta values.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with missing values in target_column.
    target_column : str
        The variable to evaluate sensitivity for.
    delta_grid : Optional[List[float]], default=None
        Grid of delta values to sweep. Defaults to 13 points in [-1.5, +1.5] standard deviations.
    downstream_evaluator : Optional[Callable[[pd.DataFrame], float]]
        Custom evaluator returning a scalar metric for an imputed DataFrame.
    downstream_feature : Optional[str]
        If provided with downstream_outcome, evaluates the regression slope.
    downstream_outcome : Optional[str]
        Outcome variable for regression evaluation.
    decision_threshold : Optional[float]
        Optional substantive policy or decision threshold for tipping analysis.
    shift_type : str, default='standardized'
        Shift type for PatternMixtureImputer ('standardized', 'percentage', 'raw').
    alpha : float, default=0.05
        Significance level for confidence intervals.
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    SensitivityReport
    """
    if target_column not in data.columns:
        raise ValueError(f"Column '{target_column}' not found in data.")

    if delta_grid is None:
        delta_grid = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

    delta_grid = sorted(list(set(delta_grid)))
    records = []
    mar_estimate: Optional[float] = None
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

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
        std_val = float(target_vals.std())
        se_val = float(std_val / np.sqrt(max(1, len(target_vals))))

        # Downstream metric evaluation
        if downstream_evaluator is not None:
            metric_val = float(downstream_evaluator(df_imputed))
            metric_se = se_val  # Approximation if not provided by custom evaluator
        elif downstream_outcome and downstream_feature:
            X_mat = sm.add_constant(df_imputed[[downstream_feature]].to_numpy(), has_constant="add")
            y_vec = df_imputed[downstream_outcome].to_numpy()
            ols = sm.OLS(y_vec, X_mat).fit()
            metric_val = float(ols.params[1])
            metric_se = float(ols.bse[1])
        else:
            metric_val = mean_val
            metric_se = 0.0  # Sensitivity curve is deterministic; SE of mean does not represent delta sensitivity

        ci_lower = metric_val - z_crit * metric_se
        ci_upper = metric_val + z_crit * metric_se

        if abs(delta) < 1e-6:
            mar_estimate = metric_val

        records.append(
            {
                "delta": delta,
                "target_mean": mean_val,
                "target_median": median_val,
                "target_std": std_val,
                "downstream_metric": metric_val,
                "metric_se": metric_se,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
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

    # Tipping points detection
    tipping_points: List[TippingPoint] = []
    is_fragile = False

    metrics = grid_df["downstream_metric"].values
    ci_lowers = grid_df["ci_lower"].values
    ci_uppers = grid_df["ci_upper"].values
    deltas = grid_df["delta"].values

    for i in range(len(deltas) - 1):
        m1, m2 = metrics[i], metrics[i + 1]
        d1, d2 = deltas[i], deltas[i + 1]

        # 1. Sign flip
        if (m1 * m2 < 0) or (m1 == 0 and m2 != 0):
            tipping_d = d1 if (m2 - m1) == 0 else d1 - m1 * (d2 - d1) / (m2 - m1)
            if abs(tipping_d) <= 1.0:
                is_fragile = True
            tipping_points.append(
                TippingPoint(
                    metric_name="sign_flip",
                    tipping_delta=float(tipping_d),
                    original_value=mar_estimate,
                    tipping_value=0.0,
                    description=f"Estimate flips sign (crosses 0) at delta = {tipping_d:+.2f} std devs.",
                )
            )

        # 2. Confidence interval crosses zero (loss of significance)
        l1, l2 = ci_lowers[i], ci_lowers[i + 1]
        u1, u2 = ci_uppers[i], ci_uppers[i + 1]
        if (l1 > 0 and l2 <= 0) or (l1 <= 0 and l2 > 0):
            # Lower bound crosses zero
            tip_d = d1 if (l2 - l1) == 0 else d1 - l1 * (d2 - d1) / (l2 - l1)
            if abs(tip_d) <= 1.0:
                is_fragile = True
            tipping_points.append(
                TippingPoint(
                    metric_name="ci_crosses_zero",
                    tipping_delta=float(tip_d),
                    original_value=mar_estimate,
                    tipping_value=0.0,
                    description=f"95% CI includes zero (loss of statistical significance) at delta = {tip_d:+.2f}.",
                )
            )
        elif (u1 >= 0 and u2 < 0) or (u1 < 0 and u2 >= 0):
            tip_d = d1 if (u2 - u1) == 0 else d1 - u1 * (d2 - d1) / (u2 - u1)
            if abs(tip_d) <= 1.0:
                is_fragile = True
            tipping_points.append(
                TippingPoint(
                    metric_name="ci_crosses_zero",
                    tipping_delta=float(tip_d),
                    original_value=mar_estimate,
                    tipping_value=0.0,
                    description=f"95% CI includes zero at delta = {tip_d:+.2f}.",
                )
            )

        # 3. Policy / Decision threshold crossing
        if decision_threshold is not None:
            if (m1 < decision_threshold <= m2) or (m2 < decision_threshold <= m1):
                tip_d = d1 + (decision_threshold - m1) * (d2 - d1) / max(1e-12, m2 - m1)
                tipping_points.append(
                    TippingPoint(
                        metric_name="policy_threshold",
                        tipping_delta=float(tip_d),
                        original_value=mar_estimate,
                        tipping_value=decision_threshold,
                        description=f"Estimate crosses decision threshold {decision_threshold:.2f} at delta = {tip_d:+.2f}.",
                    )
                )

    if is_fragile:
        tip_str = ", ".join([f"delta={tp.tipping_delta:+.2f}" for tp in tipping_points])
        interpretation = (
            f"CRITICAL FRAGILITY: Substantive conclusions are fragile to plausible MNAR departures "
            f"({tip_str}). Under MAR (delta=0), the estimate is {mar_estimate:.3f}, but if non-responders "
            f"differ systematically by plausible magnitudes (|delta| <= 1.0 std dev), the conclusion reverses. "
            f"Do not present a single MAR point estimate as definitive."
        )
    elif spread > 0.5 * abs(mar_estimate + 1e-6):
        interpretation = (
            f"MODERATE SENSITIVITY: While the qualitative direction is preserved, the effect magnitude "
            f"varies considerably across plausible MNAR shifts (from {est_min:.3f} to {est_max:.3f}). "
            f"Report this sensitivity range alongside the baseline estimate."
        )
    else:
        interpretation = (
            f"ROBUST CONCLUSION: The estimate ranges tightly between {est_min:.3f} and {est_max:.3f} "
            f"across the entire [-1.0, +1.0] delta spectrum. The substantive finding is resilient to moderate MNAR."
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
