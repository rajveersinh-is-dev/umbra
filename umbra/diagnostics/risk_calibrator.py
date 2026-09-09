"""
Empirical Probability Calibration and Cost-Sensitive Risk Routing for Umbra.

Provides:
1. `compute_bayes_optimal_threshold`: Closed-form Bayes-optimal decision threshold
   under asymmetric loss matrix penalties.
2. `compute_expected_losses`: Expected loss evaluation for actions under posterior MNAR risk.
3. `RouterDecisionProfile`: Pluggable decision profiles with configurable loss ratios.
4. `ROUTER_PROFILES`: Standard profiles ('balanced', 'conservative_mnar', 'permissive_mar').
5. `MNARRiskCalibrator`: Platt scaling, Isotonic regression, and linear calibration mapping
   raw composite diagnostic scores to calibrated posterior probabilities P(MNAR | diagnostics).
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression


def compute_bayes_optimal_threshold(
    c_fn: float = 1.0,
    c_fa: float = 1.0,
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
) -> float:
    """Compute Bayes-optimal decision threshold for MNAR classification under asymmetric losses.

    In binary statistical decision theory (Elkan 2001), let true state be y in {0, 1}
    where 0 is MAR/MCAR and 1 is MNAR.
    Let C_FA be the cost of a false alarm (predicting MNAR when MAR holds, incurring
    unnecessary sensitivity analysis or Heckman complexity).
    Let C_FN be the cost of a false negative (predicting MAR when MNAR holds, incurring
    bias and anti-conservative inference).

    Given posterior probability p = P(MNAR | diagnostics), predicting MNAR is optimal when:
        E[L | MNAR] <= E[L | MAR]
        (1 - p) * C_FA <= p * C_FN
        p >= C_FA / (C_FA + C_FN) = tau*

    Parameters
    ----------
    c_fn : float, default=1.0
        Cost of false negative (missed MNAR).
    c_fa : float, default=1.0
        Cost of false alarm (false MNAR).
    min_threshold : float, default=0.01
        Lower bound clamp.
    max_threshold : float, default=0.99
        Upper bound clamp.

    Returns
    -------
    float
        Bayes-optimal threshold tau* in [min_threshold, max_threshold].
    """
    if c_fn <= 0.0 or c_fa <= 0.0:
        raise ValueError(f"Costs must be strictly positive. Got c_fn={c_fn}, c_fa={c_fa}.")

    tau_star = c_fa / (c_fa + c_fn)
    return float(np.clip(tau_star, min_threshold, max_threshold))


def compute_expected_losses(
    p_mnar: float,
    c_fn: float = 1.0,
    c_fa: float = 1.0,
) -> Dict[str, float]:
    """Compute expected losses for decision actions given calibrated P(MNAR).

    Parameters
    ----------
    p_mnar : float
        Calibrated posterior probability of MNAR in [0.0, 1.0].
    c_fn : float, default=1.0
        Cost of false negative (action MAR when true state is MNAR).
    c_fa : float, default=1.0
        Cost of false alarm (action MNAR when true state is MAR).

    Returns
    -------
    Dict[str, float]
        Dictionary with 'loss_mar', 'loss_mnar', and 'expected_regret'.
    """
    p = float(np.clip(p_mnar, 0.0, 1.0))
    loss_mar = p * c_fn
    loss_mnar = (1.0 - p) * c_fa
    min_loss = min(loss_mar, loss_mnar)
    return {
        "loss_mar": float(loss_mar),
        "loss_mnar": float(loss_mnar),
        "expected_loss": float(min_loss),
        "expected_regret": float(abs(loss_mar - loss_mnar)),
    }


@dataclass
class RouterDecisionProfile:
    """Decision configuration specifying misclassification loss matrix and risk cutoffs.

    Attributes
    ----------
    name : str
        Profile identifier (e.g., 'balanced', 'conservative_mnar', 'permissive_mar').
    c_fn : float
        Penalty for missed MNAR (treating MNAR as MAR).
    c_fa : float
        Penalty for false alarm (treating MAR as MNAR).
    high_risk_threshold : float
        Cutoff for HIGH MNAR risk tier (triggers Heckman or Pattern Mixture).
    medium_risk_threshold : float
        Cutoff for MEDIUM risk tier (triggers Pattern Mixture sensitivity).
    tail_risk_threshold : float
        Cutoff for residual tail dependency signal.
    description : str
        Human-readable rationale for the profile.
    """

    name: str
    c_fn: float = 1.0
    c_fa: float = 1.0
    high_risk_threshold: float = 0.50
    medium_risk_threshold: float = 0.25
    tail_risk_threshold: float = 0.25
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouterDecisionProfile":
        return cls(**data)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize profile to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "RouterDecisionProfile":
        """Load profile from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# Built-in Decision Profiles
ROUTER_PROFILES: Dict[str, RouterDecisionProfile] = {
    "balanced": RouterDecisionProfile(
        name="balanced",
        c_fn=1.0,
        c_fa=1.0,
        high_risk_threshold=0.50,
        medium_risk_threshold=0.25,
        tail_risk_threshold=0.25,
        description="Balanced operational profile treating false alarms and missed MNAR equally.",
    ),
    "conservative_mnar": RouterDecisionProfile(
        name="conservative_mnar",
        c_fn=4.0,
        c_fa=1.0,
        high_risk_threshold=0.20,
        medium_risk_threshold=0.10,
        tail_risk_threshold=0.15,
        description="Risk-averse profile penalizing missed MNAR 4x (for clinical, regulatory, or causal studies).",
    ),
    "permissive_mar": RouterDecisionProfile(
        name="permissive_mar",
        c_fn=1.0,
        c_fa=4.0,
        high_risk_threshold=0.80,
        medium_risk_threshold=0.40,
        tail_risk_threshold=0.35,
        description="Permissive profile penalizing false alarms 4x (for exploratory pipelines favoring MICE).",
    ),
}


def get_decision_profile(
    profile_or_name: Optional[Union[str, RouterDecisionProfile]] = None,
    loss_matrix: Optional[Dict[str, float]] = None,
) -> RouterDecisionProfile:
    """Resolve a RouterDecisionProfile from name, instance, or custom loss matrix."""
    if loss_matrix is not None:
        c_fn = float(loss_matrix.get("c_fn", loss_matrix.get("loss_mar_when_mnar", 1.0)))
        c_fa = float(loss_matrix.get("c_fa", loss_matrix.get("loss_mnar_when_mar", 1.0)))
        tau_star = compute_bayes_optimal_threshold(c_fn=c_fn, c_fa=c_fa)
        tau_med = round(tau_star / 2.0, 3)
        return RouterDecisionProfile(
            name="custom_loss_matrix",
            c_fn=c_fn,
            c_fa=c_fa,
            high_risk_threshold=tau_star,
            medium_risk_threshold=tau_med,
            tail_risk_threshold=round(min(0.25, tau_med * 1.2), 3),
            description=f"Custom profile with c_fn={c_fn}, c_fa={c_fa} yielding tau*={tau_star:.3f}.",
        )

    if isinstance(profile_or_name, RouterDecisionProfile):
        return profile_or_name

    if isinstance(profile_or_name, str):
        key = profile_or_name.lower().strip()
        if key in ROUTER_PROFILES:
            return ROUTER_PROFILES[key]
        raise ValueError(
            f"Unknown decision profile '{profile_or_name}'. Available: {list(ROUTER_PROFILES.keys())}."
        )

    return ROUTER_PROFILES["balanced"]


class MNARRiskCalibrator:
    """Empirical Probability Calibrator for MNAR Missingness Concern Scores.

    Maps uncalibrated composite risk scores s in [0.0, 1.0] to empirical posterior
    probabilities P(MNAR | diagnostics) using Platt scaling (logistic sigmoid) or
    non-parametric Isotonic regression.

    Parameters
    ----------
    method : str, default='platt'
        Calibration algorithm:
        - 'platt': Regularized monotonic logistic sigmoid P = expit(w * s + b) with w >= 0.
        - 'isotonic': Non-parametric isotonic regression via Pool Adjacent Violators Algorithm.
        - 'linear': Min-max linear scaling with boundary clipping.
    alpha : float, default=1.0
        L2 regularization strength for Platt scaling.
    """

    # Pre-calibrated default coefficients fitted across synthetic benchmark battery
    DEFAULT_PLATT_WEIGHT = 6.0
    DEFAULT_PLATT_INTERCEPT = -3.0

    def __init__(self, method: str = "platt", alpha: float = 1.0):
        valid_methods = ("platt", "isotonic", "linear")
        if method not in valid_methods:
            raise ValueError(
                f"Unknown calibration method '{method}'. Must be one of {valid_methods}."
            )
        self.method = method
        self.alpha = float(alpha)
        self.is_fitted_ = False
        self.weight_: float = self.DEFAULT_PLATT_WEIGHT
        self.intercept_: float = self.DEFAULT_PLATT_INTERCEPT
        self.isotonic_model_: Optional[IsotonicRegression] = None
        self.linear_min_: float = 0.0
        self.linear_max_: float = 1.0
        self.metrics_: Dict[str, float] = {}

    def fit(
        self,
        scores: Union[List[float], np.ndarray, pd.Series],
        y_true: Union[List[int], np.ndarray, pd.Series],
    ) -> "MNARRiskCalibrator":
        """Fit probability calibrator on validation diagnostic scores and true labels."""
        s = np.asarray(scores, dtype=float).ravel()
        y = np.asarray(y_true, dtype=int).ravel()

        if len(s) != len(y):
            raise ValueError(f"Length mismatch: len(scores)={len(s)}, len(y_true)={len(y)}.")
        if len(s) < 4:
            raise ValueError(f"At least 4 samples required for calibration. Got {len(s)}.")
        if len(np.unique(y)) < 2:
            raise ValueError("y_true must contain both classes (0: MAR, 1: MNAR).")

        if self.method == "platt":
            # Minimize binary cross entropy with L2 penalty on w, enforcing w >= 0 (monotonicity)
            def loss_func(params: np.ndarray) -> float:
                w, b = params
                logits = w * s + b
                p = expit(logits)
                eps = 1e-12
                p = np.clip(p, eps, 1.0 - eps)
                bce = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
                reg = 0.5 * self.alpha * (w**2)
                return float(bce + reg)

            res = minimize(
                loss_func,
                x0=np.array([self.DEFAULT_PLATT_WEIGHT, self.DEFAULT_PLATT_INTERCEPT]),
                bounds=[(0.0, 50.0), (-50.0, 50.0)],
                method="L-BFGS-B",
            )
            self.weight_ = float(res.x[0])
            self.intercept_ = float(res.x[1])

        elif self.method == "isotonic":
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(s, y)
            self.isotonic_model_ = iso

        elif self.method == "linear":
            s_min = float(np.min(s))
            s_max = float(np.max(s))
            self.linear_min_ = s_min
            self.linear_max_ = s_max if s_max > s_min else s_min + 1.0

        self.is_fitted_ = True
        preds = self.predict_proba(s)
        self.metrics_ = {
            "brier_score": float(np.mean((preds - y) ** 2)),
            "ece": self.expected_calibration_error(s, y),
        }
        return self

    def predict_proba(self, scores: Union[float, List[float], np.ndarray, pd.Series]) -> np.ndarray:
        """Compute calibrated probabilities P(MNAR | score) in [0.0, 1.0]."""
        is_scalar = np.isscalar(scores)
        s = np.asarray([scores] if is_scalar else scores, dtype=float).ravel()

        if self.method == "platt":
            logits = self.weight_ * s + self.intercept_
            probs = expit(logits)
        elif self.method == "isotonic":
            if self.isotonic_model_ is None:
                # Fallback to default Platt if unfitted
                logits = self.DEFAULT_PLATT_WEIGHT * s + self.DEFAULT_PLATT_INTERCEPT
                probs = expit(logits)
            else:
                probs = self.isotonic_model_.predict(s)
        elif self.method == "linear":
            denom = self.linear_max_ - self.linear_min_
            probs = (s - self.linear_min_) / (denom if denom > 1e-8 else 1.0)

        probs = np.clip(probs, 0.0, 1.0)
        return np.asarray(probs, dtype=float)

    def calibrate(self, score: float) -> float:
        """Calibrate a single scalar score into probability P(MNAR)."""
        return float(self.predict_proba(score)[0])

    def brier_score(
        self,
        scores: Union[List[float], np.ndarray, pd.Series],
        y_true: Union[List[int], np.ndarray, pd.Series],
    ) -> float:
        """Compute Brier score (mean squared error of probability predictions)."""
        y = np.asarray(y_true, dtype=int).ravel()
        p = self.predict_proba(scores)
        return float(np.mean((p - y) ** 2))

    def expected_calibration_error(
        self,
        scores: Union[List[float], np.ndarray, pd.Series],
        y_true: Union[List[int], np.ndarray, pd.Series],
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error (ECE) across binned confidence intervals."""
        s = np.asarray(scores, dtype=float).ravel()
        y = np.asarray(y_true, dtype=int).ravel()
        p = self.predict_proba(s)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = len(y)

        for i in range(n_bins):
            bin_lower = bin_edges[i]
            bin_upper = bin_edges[i + 1]
            mask = (p >= bin_lower) & (p <= bin_upper if i == n_bins - 1 else p < bin_upper)
            n_in_bin = int(mask.sum())
            if n_in_bin > 0:
                conf = float(np.mean(p[mask]))
                acc = float(np.mean(y[mask]))
                ece += (n_in_bin / n_total) * abs(acc - conf)

        return float(ece)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize calibrator state to dictionary."""
        d: Dict[str, Any] = {
            "method": self.method,
            "alpha": self.alpha,
            "is_fitted": self.is_fitted_,
            "weight": self.weight_,
            "intercept": self.intercept_,
            "linear_min": self.linear_min_,
            "linear_max": self.linear_max_,
            "metrics": self.metrics_,
        }
        if self.isotonic_model_ is not None and hasattr(self.isotonic_model_, "X_thresholds_"):
            d["isotonic_x_thresholds"] = self.isotonic_model_.X_thresholds_.tolist()
            d["isotonic_y_thresholds"] = self.isotonic_model_.y_thresholds_.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MNARRiskCalibrator":
        """Reconstruct calibrator from dictionary."""
        cal = cls(method=d.get("method", "platt"), alpha=d.get("alpha", 1.0))
        cal.is_fitted_ = bool(d.get("is_fitted", False))
        cal.weight_ = float(d.get("weight", cls.DEFAULT_PLATT_WEIGHT))
        cal.intercept_ = float(d.get("intercept", cls.DEFAULT_PLATT_INTERCEPT))
        cal.linear_min_ = float(d.get("linear_min", 0.0))
        cal.linear_max_ = float(d.get("linear_max", 1.0))
        cal.metrics_ = d.get("metrics", {})

        if cal.method == "isotonic" and "isotonic_x_thresholds" in d:
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.X_thresholds_ = np.asarray(d["isotonic_x_thresholds"], dtype=float)
            iso.y_thresholds_ = np.asarray(d["isotonic_y_thresholds"], dtype=float)
            iso.X_min_ = float(iso.X_thresholds_[0]) if len(iso.X_thresholds_) > 0 else 0.0
            iso.X_max_ = float(iso.X_thresholds_[-1]) if len(iso.X_thresholds_) > 0 else 1.0
            cal.isotonic_model_ = iso

        return cal

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize calibrator model to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "MNARRiskCalibrator":
        """Load calibrator model from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)
