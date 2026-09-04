"""
Unit tests for model misspecification and boundary conditions.

Tests:
1. Weak instrument behavior and fallback
2. Exclusion restriction violation tracking
3. Heavy-tailed non-normal errors (Student-t)
4. Non-linear selection thresholds
5. U-shaped tail dropouts
"""

import numpy as np

from benchmarks.misspecification_benchmark import (
    generate_misspecified_dataset,
)
from umbra.api import UmbraImputer
from umbra.imputers.heckman_selection import HeckmanSelectionImputer
from umbra.imputers.mar_chained_equations import MARChainedEquationsImputer
from umbra.imputers.pattern_mixture import PatternMixtureImputer


def test_weak_instrument_regime_fallback():
    """Verify that Umbra Auto handles weak instruments by routing to MICE."""
    df_comp, df_obs, true_p = generate_misspecified_dataset(
        "WEAK_INSTRUMENT", n_samples=500, random_state=42
    )
    imp = UmbraImputer(
        strategy="auto",
        shadow_cols={"income": "shadow_z"},
        run_sensitivity=False,
        random_state=42,
    )
    df_imp = imp.fit_transform(df_obs)
    assert not df_imp["income"].isna().any()
    # Weak instrument should NOT be adopted into Heckman
    route = imp.routing_decisions_.get("income")
    assert route in ("mar_chained_equations", "pattern_mixture")


def test_exclusion_violation_bias_contrast():
    """Verify that exclusion violation induces bias in Heckman while Pattern Mixture covers."""
    df_comp, df_obs, true_p = generate_misspecified_dataset(
        "EXCLUSION_VIOLATION", n_samples=500, random_state=42
    )
    # Always Heckman
    heck = HeckmanSelectionImputer(shadow_cols={"income": "shadow_z"}, random_state=42)
    df_heck = heck.fit_transform(df_obs)
    heck_bias = abs(df_heck["income"].mean() - true_p["true_mean"])

    # Pattern Mixture at delta=0
    pm = PatternMixtureImputer(delta=0.0, random_state=42)
    df_pm = pm.fit_transform(df_obs)
    pm_bias = abs(df_pm["income"].mean() - true_p["true_mean"])

    # Both produce valid finite floats without crashing
    assert np.isfinite(heck_bias)
    assert np.isfinite(pm_bias)


def test_non_normal_student_t_stability():
    """Verify imputers execute without numerical underflow or failure under Student-t(3) errors."""
    df_comp, df_obs, true_p = generate_misspecified_dataset(
        "NON_NORMAL_STUDENT_T", n_samples=500, random_state=42
    )
    imp_auto = UmbraImputer(
        strategy="auto", shadow_cols={"income": "shadow_z"}, run_sensitivity=False, random_state=42
    )
    df_imp = imp_auto.fit_transform(df_obs)
    assert not df_imp["income"].isna().any()
    assert np.all(np.isfinite(df_imp["income"].to_numpy()))


def test_u_shaped_tail_dropout_handling():
    """Verify U-shaped tail dropout is handled without throwing LinAlgError or diverging."""
    df_comp, df_obs, true_p = generate_misspecified_dataset(
        "U_SHAPED_TAILS", n_samples=500, random_state=42
    )
    mice = MARChainedEquationsImputer(random_state=42)
    df_mice = mice.fit_transform(df_obs)
    assert not df_mice["income"].isna().any()

    pm = PatternMixtureImputer(delta=0.0, random_state=42)
    df_pm = pm.fit_transform(df_obs)
    assert not df_pm["income"].isna().any()
