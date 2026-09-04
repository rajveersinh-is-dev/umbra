"""
Negative-Control Experiments for Umbra.

Verifies test specificity:
1. Noisy MCAR: High sampling variance must NOT trigger false MNAR escalation.
2. Strong MAR: Substantial observable covariate shifts must route to MICE, not selection models.
3. Weak MNAR: Subtle departures from MAR are handled gracefully without numerical explosion.
"""

import numpy as np
import pandas as pd

from benchmarks.dgps import generate_simulation_dataset
from umbra.api import UmbraImputer


def test_negative_control_noisy_mcar():
    """Verify that noisy MCAR data does not trigger false positive MNAR escalation."""
    rng = np.random.RandomState(123)
    n = 1000
    X1 = rng.normal(0, 2.5, size=n)
    X2 = rng.exponential(1.5, size=n)
    # Target is generated with substantial noise
    target = 10.0 + 0.8 * X1 - 0.5 * X2 + rng.normal(0, 3.0, size=n)

    # Pure Bernoulli dropout (MCAR)
    mask = rng.uniform(0, 1, size=n) < 0.25
    df = pd.DataFrame({"x1": X1, "x2": X2, "target": target})
    df.loc[mask, "target"] = np.nan

    imputer = UmbraImputer(strategy="auto", random_state=42, run_sensitivity=False)
    imputer.fit(df)

    decision = imputer.routing_decisions_["target"]
    risk = imputer.diagnostics_["target"].risk_level

    # Negative control requirement: Must NOT escalate to Heckman or flag HIGH risk
    assert decision == "mar_chained_equations", f"Expected MICE for noisy MCAR, got {decision}"
    assert risk in ("LOW", "MEDIUM"), f"Expected LOW/MEDIUM risk for MCAR, got {risk}"


def test_negative_control_strong_mar():
    """Verify that strong MAR with pronounced covariate shifts routes to MICE, NOT selection models."""
    sim = generate_simulation_dataset(
        mechanism="MAR",
        n_samples=1500,
        missing_rate=0.35,
        random_state=42,
    )

    imputer = UmbraImputer(
        strategy="auto",
        shadow_cols={"target": "shadow_z"},
        random_state=42,
        run_sensitivity=False,
    )
    imputer.fit(sim.data_observed)

    decision = imputer.routing_decisions_["target"]
    risk = imputer.diagnostics_["target"].risk_level

    # Negative control requirement: Since observables explain missingness,
    # the router must preserve standard MICE rather than misapplying a selection model.
    assert decision == "mar_chained_equations", f"Expected MICE for strong MAR, got {decision}"
    assert risk in ("LOW", "MEDIUM"), f"Expected non-HIGH risk for MAR, got {risk}"


def test_negative_control_weak_mnar_graceful_handling():
    """Verify that subtle MNAR signals are imputed without numerical failure or variance explosion."""
    sim = generate_simulation_dataset(
        mechanism="MNAR_WEAK_SIGNAL",
        n_samples=1000,
        missing_rate=0.20,
        random_state=42,
    )

    imputer = UmbraImputer(
        strategy="auto",
        random_state=42,
        run_sensitivity=True,
    )
    imputer.fit(sim.data_observed)
    X_imp = imputer.transform(sim.data_observed)

    # Output must be complete and finite
    assert not X_imp["target"].isna().any()
    assert np.all(np.isfinite(X_imp["target"]))

    # Bias should remain bounded
    imputed_mean = float(X_imp["target"].mean())
    true_mean = float(sim.true_params["true_mean"])
    abs_bias = abs(imputed_mean - true_mean)
    assert abs_bias < 1.0, f"Bias {abs_bias:.3f} unexpectedly large for weak MNAR"
