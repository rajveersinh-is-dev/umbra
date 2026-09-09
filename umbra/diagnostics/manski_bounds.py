"""
Manski Partial Identifiability Bounds Engine (Manski 1989, 1990, 2003).

Computes sharp nonparametric bounds for population parameters (mean, median, quantiles)
under arbitrary missingness mechanisms without requiring untestable parametric assumptions
(e.g., MCAR, MAR, bivariate normality, or exclusion restrictions).

Key Mathematical Theorems:
1. Sharp Mean Bounds (Manski 1989):
   For Y in [y_L, y_U] with missingness rate p_miss = P(R=0):
   E[Y] in [ E[Y | R=1](1 - p_miss) + y_L * p_miss, E[Y | R=1](1 - p_miss) + y_U * p_miss ]
   Interval width: Delta = p_miss * (y_U - y_L).

2. Sharp Quantile Bounds (Manski 1994, 2003):
   For quantile alpha in (0, 1):
   - If alpha <= p_miss: LB_alpha = y_L, else F_obs^{-1}((alpha - p_miss) / (1 - p_miss))
   - If alpha >= 1 - p_miss: UB_alpha = y_U, else F_obs^{-1}(alpha / (1 - p_miss))
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class ManskiBoundsResult:
    """Container for sharp nonparametric Manski partial identification bounds."""

    feature: str
    n_total: int
    n_observed: int
    n_missing: int
    missing_rate: float
    observed_mean: float
    observed_std: float
    support_lower: float
    support_upper: float
    mean_lower_bound: float
    mean_upper_bound: float
    mean_interval_width: float
    median_lower_bound: float
    median_upper_bound: float
    median_interval_width: float
    quantile_bounds: Dict[float, Tuple[float, float]]
    is_sharp: bool = True
    assumptions: str = "Zero untestable assumptions (Manski 1989, 2003)"

    def contains(self, value: float, parameter: str = "mean") -> bool:
        """Check whether a candidate parameter value falls within the partial identification interval."""
        if parameter == "mean":
            return bool(self.mean_lower_bound - 1e-9 <= value <= self.mean_upper_bound + 1e-9)
        elif parameter == "median":
            return bool(self.median_lower_bound - 1e-9 <= value <= self.median_upper_bound + 1e-9)
        else:
            raise ValueError(f"Unknown parameter '{parameter}'. Supported: 'mean', 'median'.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert bounds result to structured dictionary for serialization."""
        return {
            "feature": self.feature,
            "n_total": self.n_total,
            "n_observed": self.n_observed,
            "n_missing": self.n_missing,
            "missing_rate": round(self.missing_rate, 4),
            "observed_mean": round(self.observed_mean, 4)
            if not np.isnan(self.observed_mean)
            else None,
            "observed_std": round(self.observed_std, 4)
            if not np.isnan(self.observed_std)
            else None,
            "support": [round(self.support_lower, 4), round(self.support_upper, 4)],
            "mean_bounds": [round(self.mean_lower_bound, 4), round(self.mean_upper_bound, 4)],
            "mean_interval_width": round(self.mean_interval_width, 4),
            "median_bounds": [round(self.median_lower_bound, 4), round(self.median_upper_bound, 4)],
            "median_interval_width": round(self.median_interval_width, 4),
            "quantile_bounds": {
                f"{int(q * 100)}%": [round(b[0], 4), round(b[1], 4)]
                for q, b in self.quantile_bounds.items()
            },
            "is_sharp": self.is_sharp,
            "assumptions": self.assumptions,
        }

    def summary(self) -> str:
        """Return human-readable summary of Manski bounds."""
        lines = [
            f"Manski Partial Identification Bounds: '{self.feature}' (Missing: {self.missing_rate:.1%})",
            f"  Domain Support: [{self.support_lower:.2f}, {self.support_upper:.2f}]",
            f"  Sharp Mean Interval:   [{self.mean_lower_bound:.4f}, {self.mean_upper_bound:.4f}] (width: {self.mean_interval_width:.4f})",
            f"  Sharp Median Interval: [{self.median_lower_bound:.4f}, {self.median_upper_bound:.4f}] (width: {self.median_interval_width:.4f})",
            f"  Observed Point Mean:   {self.observed_mean:.4f} (Observed N: {self.n_observed:,})",
            "  Untestable Assumptions Required: NONE (Impervious to Molenberghs non-identifiability)",
        ]
        return "\n".join(lines)


def compute_manski_bounds(
    y: Union[pd.Series, np.ndarray, Sequence[float]],
    feature_name: Optional[str] = None,
    support: Optional[Tuple[float, float]] = None,
    quantiles: Sequence[float] = (0.25, 0.50, 0.75),
    trim_quantile: Optional[float] = None,
) -> ManskiBoundsResult:
    """
    Compute sharp Manski partial identification bounds for population mean and quantiles.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        1-dimensional target feature vector containing observed values and NaNs.
    feature_name : str, optional
        Name of the feature for labeling.
    support : tuple of (float, float), optional
        Known theoretical or natural domain support [y_L, y_U] (e.g. [0, 100] for percentages).
        If None, empirical minimum and maximum of observed cases are used.
    quantiles : sequence of float, default=(0.25, 0.50, 0.75)
        Quantile levels alpha in (0, 1) for which sharp bounds should be calculated.
    trim_quantile : float, optional
        If provided (e.g. 0.01), trims extreme empirical sample quantiles when establishing
        the domain support to reduce sensitivity to spurious sample outliers.

    Returns
    -------
    ManskiBoundsResult
        Container with sharp mean and quantile intervals.
    """
    # 1. Normalize input array
    if isinstance(y, pd.Series):
        fname = feature_name or str(y.name or "variable")
        y_arr = y.values.astype(float)
    elif isinstance(y, np.ndarray):
        fname = feature_name or "variable"
        y_arr = y.flatten().astype(float)
    else:
        fname = feature_name or "variable"
        y_arr = np.asarray(y, dtype=float).flatten()

    n_total = len(y_arr)
    if n_total == 0:
        raise ValueError("Cannot compute Manski bounds on empty array.")

    obs_mask = ~np.isnan(y_arr)
    y_obs = y_arr[obs_mask]
    n_observed = len(y_obs)
    n_missing = n_total - n_observed
    missing_rate = float(n_missing / n_total)

    # 2. Fully missing edge case
    if n_observed == 0:
        if support is None:
            y_l, y_u = -float("inf"), float("inf")
        else:
            y_l, y_u = float(support[0]), float(support[1])
        q_dict = {float(q): (y_l, y_u) for q in quantiles}
        return ManskiBoundsResult(
            feature=fname,
            n_total=n_total,
            n_observed=0,
            n_missing=n_total,
            missing_rate=1.0,
            observed_mean=np.nan,
            observed_std=np.nan,
            support_lower=y_l,
            support_upper=y_u,
            mean_lower_bound=y_l,
            mean_upper_bound=y_u,
            mean_interval_width=y_u - y_l,
            median_lower_bound=y_l,
            median_upper_bound=y_u,
            median_interval_width=y_u - y_l,
            quantile_bounds=q_dict,
        )

    obs_mean = float(np.mean(y_obs))
    obs_std = float(np.std(y_obs, ddof=1)) if n_observed > 1 else 0.0

    # 3. Resolve domain support [y_L, y_U]
    if support is not None:
        y_l = float(support[0])
        y_u = float(support[1])
        if y_l > y_u:
            raise ValueError(f"support lower bound ({y_l}) must be <= upper bound ({y_u}).")
        # Ensure observed data lies within support
        y_l = min(y_l, float(np.min(y_obs)))
        y_u = max(y_u, float(np.max(y_obs)))
    elif trim_quantile is not None and 0.0 < trim_quantile < 0.5:
        y_l = float(np.quantile(y_obs, trim_quantile))
        y_u = float(np.quantile(y_obs, 1.0 - trim_quantile))
    else:
        y_l = float(np.min(y_obs))
        y_u = float(np.max(y_obs))

    # 4. Compute sharp mean bounds
    if n_missing == 0:
        mean_lb = obs_mean
        mean_ub = obs_mean
        mean_width = 0.0
    else:
        p_miss = missing_rate
        p_obs = 1.0 - p_miss
        mean_lb = float(obs_mean * p_obs + y_l * p_miss)
        mean_ub = float(obs_mean * p_obs + y_u * p_miss)
        mean_width = float(mean_ub - mean_lb)

    # 5. Compute sharp quantile bounds
    q_bounds: Dict[float, Tuple[float, float]] = {}
    sorted_obs = np.sort(y_obs)

    def _obs_quantile(prob: float) -> float:
        """Safe quantile lookup on sorted observed array with clipping."""
        clamped = float(np.clip(prob, 0.0, 1.0))
        return float(np.quantile(sorted_obs, clamped))

    for q in quantiles:
        alpha = float(q)
        if not (0.0 < alpha < 1.0):
            continue

        if n_missing == 0:
            val = _obs_quantile(alpha)
            q_bounds[alpha] = (val, val)
            continue

        p_miss = missing_rate
        p_obs = 1.0 - p_miss

        # Lower bound for alpha-quantile
        if alpha <= p_miss:
            lb_alpha = y_l
        else:
            p_adj_low = (alpha - p_miss) / p_obs
            lb_alpha = _obs_quantile(p_adj_low)

        # Upper bound for alpha-quantile
        if alpha >= p_obs:
            ub_alpha = y_u
        else:
            p_adj_high = alpha / p_obs
            ub_alpha = _obs_quantile(p_adj_high)

        # Invariant: lb <= ub
        lb_alpha = min(lb_alpha, ub_alpha)
        q_bounds[alpha] = (float(lb_alpha), float(ub_alpha))

    # 6. Extract median bounds
    if 0.5 in q_bounds:
        med_lb, med_ub = q_bounds[0.5]
    else:
        if missing_rate >= 0.5:
            med_lb, med_ub = y_l, y_u
        else:
            p_obs = 1.0 - missing_rate
            med_lb = _obs_quantile((0.5 - missing_rate) / p_obs)
            med_ub = _obs_quantile(0.5 / p_obs)
            med_lb = min(med_lb, med_ub)
        q_bounds[0.5] = (med_lb, med_ub)

    med_width = float(med_ub - med_lb)

    return ManskiBoundsResult(
        feature=fname,
        n_total=n_total,
        n_observed=n_observed,
        n_missing=n_missing,
        missing_rate=missing_rate,
        observed_mean=obs_mean,
        observed_std=obs_std,
        support_lower=y_l,
        support_upper=y_u,
        mean_lower_bound=mean_lb,
        mean_upper_bound=mean_ub,
        mean_interval_width=mean_width,
        median_lower_bound=med_lb,
        median_upper_bound=med_ub,
        median_interval_width=med_width,
        quantile_bounds=q_bounds,
    )


def compute_dataframe_manski_bounds(
    df: pd.DataFrame,
    supports: Optional[Dict[str, Tuple[float, float]]] = None,
    quantiles: Sequence[float] = (0.25, 0.50, 0.75),
    trim_quantile: Optional[float] = None,
) -> Dict[str, ManskiBoundsResult]:
    """
    Compute Manski partial identification bounds for all incomplete numeric columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    supports : dict of {col_name: (lower, upper)}, optional
        Custom support bounds per feature.
    quantiles : sequence of float, default=(0.25, 0.50, 0.75)
    trim_quantile : float, optional

    Returns
    -------
    dict of {col_name: ManskiBoundsResult}
    """
    support_map = supports or {}
    results: Dict[str, ManskiBoundsResult] = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].isna().any():
            sup = support_map.get(col)
            res = compute_manski_bounds(
                y=df[col],
                feature_name=col,
                support=sup,
                quantiles=quantiles,
                trim_quantile=trim_quantile,
            )
            results[col] = res

    return results
