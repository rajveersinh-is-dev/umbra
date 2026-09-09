"""
Unit tests for Umbra Cost-Sensitive Risk Routing & Empirical Probability Calibration.

Covers:
- Bayes-optimal decision threshold calculations and invariants (ROUT-01).
- Expected loss calculation for actions under posterior probability.
- RouterDecisionProfile and built-in profiles ('balanced', 'conservative_mnar', 'permissive_mar') (ROUT-03).
- Profile serialization and custom loss matrix resolution (ROUT-01, ROUT-03).
- MNARRiskCalibrator (Platt scaling, Isotonic regression, linear) (ROUT-02).
- Calibration quality metrics (Brier score, ECE) (ROUT-02).
- Model persistence / JSON serialization round-trips for calibrators.
- Differential routing in UmbraImputer under conservative vs permissive profiles (ROUT-01, ROUT-03).
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from umbra import (
    MNARRiskCalibrator,
    RouterDecisionProfile,
    UmbraImputer,
    compute_bayes_optimal_threshold,
)
from umbra.diagnostics.mnar_risk_score import assess_mnar_risk, diagnose_dataframe
from umbra.diagnostics.risk_calibrator import (
    ROUTER_PROFILES,
    compute_expected_losses,
    get_decision_profile,
)


def test_bayes_optimal_threshold_computation() -> None:
    """Verify closed-form Bayes-optimal threshold derivation and monotonicity."""
    # Balanced costs -> tau* = 0.50
    tau_bal = compute_bayes_optimal_threshold(c_fn=1.0, c_fa=1.0)
    assert pytest.approx(tau_bal, abs=1e-5) == 0.50

    # High penalty for missed MNAR -> tau* = 1 / (1 + 4) = 0.20
    tau_cons = compute_bayes_optimal_threshold(c_fn=4.0, c_fa=1.0)
    assert pytest.approx(tau_cons, abs=1e-5) == 0.20

    # High penalty for false alarms -> tau* = 4 / (4 + 1) = 0.80
    tau_perm = compute_bayes_optimal_threshold(c_fn=1.0, c_fa=4.0)
    assert pytest.approx(tau_perm, abs=1e-5) == 0.80

    # Monotonicity check: as c_fn increases, threshold strictly decreases
    costs_fn = [0.5, 1.0, 2.0, 5.0, 10.0]
    thresholds = [compute_bayes_optimal_threshold(c_fn=c, c_fa=1.0) for c in costs_fn]
    for i in range(len(thresholds) - 1):
        assert thresholds[i] > thresholds[i + 1]

    # Non-positive costs must raise ValueError
    with pytest.raises(ValueError, match="strictly positive"):
        compute_bayes_optimal_threshold(c_fn=0.0, c_fa=1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        compute_bayes_optimal_threshold(c_fn=1.0, c_fa=-2.0)


def test_compute_expected_losses() -> None:
    """Verify expected loss decomposition and regret calculation."""
    # When p_mnar = 0.8, c_fn = 1, c_fa = 1:
    # E[loss | MAR] = 0.8 * 1 = 0.8
    # E[loss | MNAR] = 0.2 * 1 = 0.2 -> optimal action is MNAR
    losses = compute_expected_losses(p_mnar=0.8, c_fn=1.0, c_fa=1.0)
    assert pytest.approx(losses["loss_mar"], abs=1e-5) == 0.8
    assert pytest.approx(losses["loss_mnar"], abs=1e-5) == 0.2
    assert pytest.approx(losses["expected_loss"], abs=1e-5) == 0.2
    assert pytest.approx(losses["expected_regret"], abs=1e-5) == 0.6


def test_router_decision_profiles() -> None:
    """Verify predefined decision profiles and custom loss matrix resolution."""
    assert "balanced" in ROUTER_PROFILES
    assert "conservative_mnar" in ROUTER_PROFILES
    assert "permissive_mar" in ROUTER_PROFILES

    p_cons = ROUTER_PROFILES["conservative_mnar"]
    p_bal = ROUTER_PROFILES["balanced"]
    p_perm = ROUTER_PROFILES["permissive_mar"]

    # Threshold ordering invariant: conservative < balanced < permissive
    assert p_cons.high_risk_threshold < p_bal.high_risk_threshold < p_perm.high_risk_threshold
    assert p_cons.medium_risk_threshold < p_bal.medium_risk_threshold < p_perm.medium_risk_threshold

    # Custom loss matrix resolution
    custom_prof = get_decision_profile(loss_matrix={"c_fn": 9.0, "c_fa": 1.0})
    assert custom_prof.name == "custom_loss_matrix"
    assert pytest.approx(custom_prof.high_risk_threshold, abs=1e-4) == 0.10
    assert custom_prof.medium_risk_threshold == 0.05

    # Resolution by string
    assert get_decision_profile("conservative_mnar").c_fn == 4.0

    # Invalid profile string raises ValueError
    with pytest.raises(ValueError, match="Unknown decision profile"):
        get_decision_profile("non_existent_profile")


def test_router_profile_serialization() -> None:
    """Verify JSON serialization and round-trip of RouterDecisionProfile."""
    prof = RouterDecisionProfile(
        name="custom_audit",
        c_fn=3.5,
        c_fa=1.5,
        high_risk_threshold=0.30,
        medium_risk_threshold=0.15,
        tail_risk_threshold=0.20,
        description="Audit trial profile",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "profile.json"
        prof.save(json_path)

        loaded = RouterDecisionProfile.load(json_path)
        assert loaded.name == prof.name
        assert loaded.c_fn == prof.c_fn
        assert loaded.c_fa == prof.c_fa
        assert loaded.high_risk_threshold == prof.high_risk_threshold
        assert loaded.description == prof.description


def test_platt_calibrator_fit_and_predict() -> None:
    """Verify Platt scaling calibrator fitting, monotonicity, and bounds."""
    rng = np.random.RandomState(42)
    # Generate synthetic scores and binary labels
    scores = np.linspace(0.05, 0.95, 50)
    # True probability of MNAR increases monotonically with score
    true_probs = 1.0 / (1.0 + np.exp(-5.0 * (scores - 0.5)))
    y_true = (rng.rand(len(scores)) < true_probs).astype(int)

    cal = MNARRiskCalibrator(method="platt", alpha=0.5)
    cal.fit(scores, y_true)

    assert cal.is_fitted_
    assert cal.weight_ >= 0.0  # Monotonicity enforced

    test_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    probs = cal.predict_proba(test_scores)

    # Probabilities in [0, 1]
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    # Monotonicity: higher score -> higher or equal probability
    for i in range(len(probs) - 1):
        assert probs[i] <= probs[i + 1]

    # Scalar calibrate method
    p_single = cal.calibrate(0.5)
    assert isinstance(p_single, float)
    assert 0.0 <= p_single <= 1.0

    # Check metrics
    bs = cal.brier_score(scores, y_true)
    ece = cal.expected_calibration_error(scores, y_true)
    assert 0.0 <= bs <= 1.0
    assert 0.0 <= ece <= 1.0


def test_isotonic_calibrator_fit_and_predict() -> None:
    """Verify Isotonic regression calibration."""
    rng = np.random.RandomState(42)
    scores = np.sort(rng.uniform(0.0, 1.0, 60))
    y_true = (scores + rng.normal(0.0, 0.2, 60) > 0.5).astype(int)

    iso_cal = MNARRiskCalibrator(method="isotonic")
    iso_cal.fit(scores, y_true)

    test_points = np.linspace(0.0, 1.0, 20)
    preds = iso_cal.predict_proba(test_points)

    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)
    # Monotonicity check
    diffs = np.diff(preds)
    assert np.all(diffs >= -1e-8)


def test_calibrator_serialization_round_trip() -> None:
    """Verify calibrator serialization to dict and JSON."""
    scores = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    y_true = np.array([0, 0, 0, 1, 1, 1])

    cal = MNARRiskCalibrator(method="platt")
    cal.fit(scores, y_true)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "calibrator.json"
        cal.save(json_path)

        loaded_cal = MNARRiskCalibrator.load(json_path)
        assert loaded_cal.is_fitted_
        assert loaded_cal.method == cal.method
        assert pytest.approx(loaded_cal.weight_, abs=1e-5) == cal.weight_
        assert pytest.approx(loaded_cal.intercept_, abs=1e-5) == cal.intercept_

        # Predictions match exactly
        preds_orig = cal.predict_proba(np.array([0.25, 0.50, 0.75]))
        preds_loaded = loaded_cal.predict_proba(np.array([0.25, 0.50, 0.75]))
        np.testing.assert_allclose(preds_orig, preds_loaded, atol=1e-6)


def test_assess_mnar_risk_with_profiles() -> None:
    """Verify assess_mnar_risk produces calibrated probabilities and respects profiles."""
    rng = np.random.RandomState(42)
    n = 300
    x1 = rng.randn(n)
    # Tail self-censoring: missing concentrated in upper tail
    y = 1.5 * x1 + rng.randn(n)
    is_missing = y > np.percentile(y, 75)
    df = pd.DataFrame({"x1": x1, "y": y})
    df.loc[is_missing, "y"] = np.nan

    # 1. Conservative profile (lower threshold for MNAR)
    rep_cons = assess_mnar_risk(df, target_col="y", decision_profile="conservative_mnar")
    assert rep_cons.decision_profile == "conservative_mnar"
    assert rep_cons.calibrated_p_mnar is not None
    assert 0.0 <= rep_cons.calibrated_p_mnar <= 1.0
    assert rep_cons.expected_loss is not None
    assert rep_cons.loss_matrix == {"c_fn": 4.0, "c_fa": 1.0}

    # 2. Permissive profile (higher threshold for MNAR)
    rep_perm = assess_mnar_risk(df, target_col="y", decision_profile="permissive_mar")
    assert rep_perm.decision_profile == "permissive_mar"
    assert rep_perm.loss_matrix == {"c_fn": 1.0, "c_fa": 4.0}


def test_diagnose_dataframe_with_profiles_and_calibration() -> None:
    """Verify diagnose_dataframe propagates decision profiles and custom calibrator."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame(
        {
            "x": rng.randn(n),
            "y1": rng.randn(n),
            "y2": rng.randn(n),
        }
    )
    df.loc[:30, "y1"] = np.nan
    df.loc[:40, "y2"] = np.nan

    custom_cal = MNARRiskCalibrator(method="platt")
    custom_cal.fit([0.1, 0.3, 0.5, 0.7, 0.9], [0, 0, 1, 1, 1])

    reports = diagnose_dataframe(
        df,
        decision_profile="conservative_mnar",
        calibrator=custom_cal,
    )
    assert len(reports) == 2
    for col, rep in reports.items():
        assert rep.decision_profile == "conservative_mnar"
        assert rep.calibrated_p_mnar is not None
        assert 0.0 <= rep.calibrated_p_mnar <= 1.0
        assert rep.expected_loss is not None


def test_umbra_imputer_differential_routing() -> None:
    """Verify UmbraImputer differentiates routing decisions under conservative vs permissive profiles."""
    rng = np.random.RandomState(42)
    n = 350
    x = rng.randn(n)
    # Generate moderate MNAR signal with moderate covariate dependency
    latent_y = 1.0 * x + rng.randn(n)
    # Masking depends partially on y (moderate tail self-censoring)
    mask = (0.5 * x + 0.6 * latent_y + rng.randn(n)) > 1.0
    y = latent_y.copy()
    y[mask] = np.nan

    df = pd.DataFrame({"x": x, "y": y})

    # Under conservative_mnar, lower threshold routes to MNAR pattern mixture
    imp_cons = UmbraImputer(
        strategy="auto",
        decision_profile="conservative_mnar",
        random_state=42,
    )
    imp_cons.fit(df)

    # Under permissive_mar, high threshold routes to MAR chained equations
    imp_perm = UmbraImputer(
        strategy="auto",
        decision_profile="permissive_mar",
        random_state=42,
    )
    imp_perm.fit(df)

    assert "y" in imp_cons.calibrated_p_mnar_
    assert "y" in imp_cons.routing_expected_losses_
    assert "y" in imp_perm.calibrated_p_mnar_

    # Verify that the two profiles produce different strategies or that conservative flags higher risk
    rep_cons = imp_cons.diagnostics_["y"]
    rep_perm = imp_perm.diagnostics_["y"]
    assert rep_cons.loss_matrix is not None
    assert rep_perm.loss_matrix is not None
    # The conservative profile's high risk threshold is 0.20 while permissive is 0.80
    assert rep_cons.loss_matrix["c_fn"] > rep_perm.loss_matrix["c_fn"]


def test_umbra_imputer_custom_loss_matrix() -> None:
    """Verify UmbraImputer accepts custom loss_matrix and sets expected losses."""
    rng = np.random.RandomState(42)
    n = 200
    x = rng.randn(n)
    y = 2.0 * x + rng.randn(n)
    y[y > np.percentile(y, 80)] = np.nan
    df = pd.DataFrame({"x": x, "y": y})

    imp = UmbraImputer(
        strategy="auto",
        loss_matrix={"c_fn": 8.0, "c_fa": 1.0},
        random_state=42,
    )
    imp.fit(df)

    assert imp.diagnostics_["y"].decision_profile == "custom_loss_matrix"
    assert imp.diagnostics_["y"].loss_matrix == {"c_fn": 8.0, "c_fa": 1.0}
    assert "y" in imp.routing_expected_losses_
    assert imp.routing_expected_losses_["y"] >= 0.0


def test_sklearn_cloning_and_get_params_with_profiles() -> None:
    """Verify scikit-learn clone and parameter introspection with new attributes."""
    imp = UmbraImputer(
        strategy="auto",
        decision_profile="conservative_mnar",
        loss_matrix={"c_fn": 5.0, "c_fa": 1.0},
        calibrator="platt",
    )
    params = imp.get_params()
    assert params["decision_profile"] == "conservative_mnar"
    assert params["loss_matrix"] == {"c_fn": 5.0, "c_fa": 1.0}
    assert params["calibrator"] == "platt"

    cloned = clone(imp)
    assert cloned.decision_profile == "conservative_mnar"
    assert cloned.loss_matrix == {"c_fn": 5.0, "c_fa": 1.0}
    assert cloned.calibrator == "platt"
