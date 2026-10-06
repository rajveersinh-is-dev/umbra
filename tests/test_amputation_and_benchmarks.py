"""
Unit & Integration Tests for Multivariate Amputation and Observational Loaders (Phase 5).

Verifies:
1. Schouten et al. (2018) multivariate amputation engine (ampute_multivariate).
2. Missingness mechanism invariants (MCAR, MAR, MNAR) and weight constraints.
3. Continuous odds shift functions (RIGHT, LEFT, MID, TAIL).
4. Real-world dataset loaders (CPS, NHANES, California Housing, Clinical Trial).
5. Observational benchmark execution pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from benchmarks.observational_benchmark import run_observational_benchmark
from umbra.benchmark.amputation import AmputationResult, ampute_multivariate
from umbra.data.loaders import (
    load_california_housing,
    load_clinical_trial_attrition,
    load_cps_wage,
    load_nhanes_biomarkers,
)


def test_ampute_multivariate_prop_and_patterns() -> None:
    """Verify empirical missingness rate matches target prop within tolerance and patterns match."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(2000, 4))

    res = ampute_multivariate(
        X,
        prop=0.30,
        mechanisms="MAR",
        odds_type="RIGHT",
        random_state=42,
    )

    assert isinstance(res, AmputationResult)
    assert res.data_amputed.shape == (2000, 4)
    assert res.mask.shape == (2000, 4)
    assert res.probabilities.shape == (2000,)
    assert res.patterns.shape == (4, 4)

    # Invariant: empirical row missing rate within +-0.05 of 0.30
    assert abs(res.empirical_prop - 0.30) < 0.05
    assert res.n_incomplete == int(res.mask.any(axis=1).sum())

    # Verify each pattern has exactly one missing column by default
    for k in range(4):
        assert np.sum(res.patterns[k] == 0) == 1


def test_ampute_multivariate_mar_vs_mnar_weights() -> None:
    """Verify MAR strictly forbids weights on incomplete variables, while MNAR allows them."""
    rng = np.random.default_rng(101)
    df = pd.DataFrame(rng.normal(0, 1, size=(500, 3)), columns=["x1", "x2", "x3"])

    # Pattern: column x3 is incomplete (0), x1 and x2 are observed (1)
    custom_pattern = np.array([[1, 1, 0]])

    # Under MAR: weight on incomplete column (x3) must be 0
    valid_mar_weights = np.array([[0.5, 0.5, 0.0]])
    res_mar = ampute_multivariate(
        df,
        prop=0.25,
        patterns=custom_pattern,
        mechanisms="MAR",
        weights=valid_mar_weights,
        random_state=42,
    )
    assert res_mar.data_amputed["x1"].isna().sum() == 0
    assert res_mar.data_amputed["x2"].isna().sum() == 0
    assert res_mar.data_amputed["x3"].isna().sum() > 0

    # Under MAR: non-zero weight on incomplete variable MUST raise ValueError
    invalid_mar_weights = np.array([[0.5, 0.0, 0.8]])
    with pytest.raises(ValueError, match="weights for incomplete variables must be strictly zero"):
        ampute_multivariate(
            df,
            prop=0.25,
            patterns=custom_pattern,
            mechanisms="MAR",
            weights=invalid_mar_weights,
            random_state=42,
        )

    # Under MNAR: non-zero weight on incomplete variable (self-masking) is valid
    valid_mnar_weights = np.array([[0.0, 0.0, 1.0]])
    res_mnar = ampute_multivariate(
        df,
        prop=0.35,
        patterns=custom_pattern,
        mechanisms="MNAR",
        weights=valid_mnar_weights,
        random_state=42,
    )
    assert res_mnar.data_amputed["x3"].isna().sum() > 0


def test_ampute_multivariate_odds_types() -> None:
    """Verify RIGHT, LEFT, MID, and TAIL probability functions shape missingness."""
    rng = np.random.default_rng(2024)
    # Create single predictor x1 and outcome x2
    x1 = rng.normal(0, 1, size=3000)
    x2 = 0.8 * x1 + rng.normal(0, 0.5, size=3000)
    df = pd.DataFrame({"x1": x1, "x2": x2})

    # Pattern: amputate x2 based on x1
    pattern = np.array([[1, 0]])
    weights = np.array([[1.0, 0.0]])

    # 1. RIGHT: positive correlation between x1 and missingness of x2
    res_right = ampute_multivariate(
        df,
        prop=0.30,
        patterns=pattern,
        mechanisms="MAR",
        weights=weights,
        odds_type="RIGHT",
        random_state=42,
    )
    r_right = res_right.mask["x2"].astype(float).values
    corr_right = float(np.corrcoef(x1, r_right)[0, 1])
    assert corr_right > 0.15, f"Expected positive correlation for RIGHT, got {corr_right}"

    # 2. LEFT: negative correlation between x1 and missingness of x2
    res_left = ampute_multivariate(
        df,
        prop=0.30,
        patterns=pattern,
        mechanisms="MAR",
        weights=weights,
        odds_type="LEFT",
        random_state=42,
    )
    r_left = res_left.mask["x2"].astype(float).values
    corr_left = float(np.corrcoef(x1, r_left)[0, 1])
    assert corr_left < -0.15, f"Expected negative correlation for LEFT, got {corr_left}"

    # 3. TAIL: missingness concentrated in extreme |x1| > 1.0
    res_tail = ampute_multivariate(
        df,
        prop=0.30,
        patterns=pattern,
        mechanisms="MAR",
        weights=weights,
        odds_type="TAIL",
        random_state=42,
    )
    r_tail = res_tail.mask["x2"].values
    mean_abs_x1_missing = float(np.mean(np.abs(x1[r_tail])))
    mean_abs_x1_observed = float(np.mean(np.abs(x1[~r_tail])))
    assert mean_abs_x1_missing > mean_abs_x1_observed

    # 4. MID: missingness concentrated near center |x1| < 0.8
    res_mid = ampute_multivariate(
        df,
        prop=0.30,
        patterns=pattern,
        mechanisms="MAR",
        weights=weights,
        odds_type="MID",
        random_state=42,
    )
    r_mid = res_mid.mask["x2"].values
    mean_abs_x1_mid_missing = float(np.mean(np.abs(x1[r_mid])))
    mean_abs_x1_mid_observed = float(np.mean(np.abs(x1[~r_mid])))
    assert mean_abs_x1_mid_missing < mean_abs_x1_mid_observed


def test_ampute_multivariate_mcar() -> None:
    """Verify MCAR produces uniform random dropout uncorrelated with any feature."""
    rng = np.random.default_rng(777)
    df = pd.DataFrame(rng.normal(0, 1, size=(2000, 3)), columns=["a", "b", "c"])
    res = ampute_multivariate(df, prop=0.25, mechanisms="MCAR", random_state=42)

    assert abs(res.empirical_prop - 0.25) < 0.05
    # All probabilities should be exactly 0.25
    assert np.allclose(res.probabilities, 0.25)


def test_ampute_multivariate_invalid_inputs() -> None:
    """Verify input validation handles invalid dimensions, props, and types gracefully."""
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        ampute_multivariate(np.zeros((50, 2)), prop=1.5)

    with pytest.raises(ValueError, match="at least 2 features"):
        ampute_multivariate(np.zeros((50, 1)), prop=0.2)

    with pytest.raises(ValueError, match="Invalid mechanism"):
        ampute_multivariate(np.zeros((50, 2)), prop=0.2, mechanisms="INVALID")

    with pytest.raises(ValueError, match="Invalid odds_type"):
        ampute_multivariate(np.zeros((50, 2)), prop=0.2, odds_type="UNKNOWN")


def test_load_cps_wage() -> None:
    """Verify load_cps_wage returns valid schema, correct splits, and array outputs."""
    # 1. Observed split
    df_obs = load_cps_wage(split="observed")
    assert isinstance(df_obs, pd.DataFrame)
    assert len(df_obs) >= 500
    assert "annual_income" in df_obs.columns
    assert "contact_attempts" in df_obs.columns
    assert df_obs["annual_income"].isna().mean() > 0.15

    # 2. Complete split
    df_comp = load_cps_wage(split="complete")
    assert isinstance(df_comp, pd.DataFrame)
    assert df_comp["annual_income"].isna().sum() == 0

    # 3. Both split
    res_both = load_cps_wage(split="both")
    assert isinstance(res_both, tuple)
    assert len(res_both) == 2
    assert isinstance(res_both[0], pd.DataFrame)
    assert isinstance(res_both[1], pd.DataFrame)

    # 4. Numpy array format
    arr_obs = load_cps_wage(split="observed", as_frame=False)
    assert isinstance(arr_obs, np.ndarray)


def test_load_nhanes_biomarkers() -> None:
    """Verify load_nhanes_biomarkers schema and missingness properties."""
    df_obs, df_comp = load_nhanes_biomarkers(split="both")  # type: ignore[misc]
    assert isinstance(df_obs, pd.DataFrame)
    assert isinstance(df_comp, pd.DataFrame)
    assert len(df_obs) >= 500
    assert "fasting_glucose" in df_obs.columns
    assert "phlebotomy_difficulty" in df_obs.columns
    assert df_obs["fasting_glucose"].isna().mean() > 0.15
    assert df_comp["fasting_glucose"].isna().sum() == 0


def test_load_california_housing() -> None:
    """Verify load_california_housing schema and missingness properties."""
    df_obs, df_comp = load_california_housing(split="both")  # type: ignore[misc]
    assert isinstance(df_obs, pd.DataFrame)
    assert isinstance(df_comp, pd.DataFrame)
    assert len(df_obs) >= 500
    assert "median_income" in df_obs.columns
    assert df_obs["median_income"].isna().mean() > 0.10
    assert df_comp["median_income"].isna().sum() == 0


def test_load_clinical_trial_attrition() -> None:
    """Verify load_clinical_trial_attrition schema and missingness properties."""
    df_obs, df_comp = load_clinical_trial_attrition(split="both")  # type: ignore[misc]
    assert isinstance(df_obs, pd.DataFrame)
    assert isinstance(df_comp, pd.DataFrame)
    assert len(df_obs) >= 500
    assert "endpoint_score" in df_obs.columns
    assert "travel_distance" in df_obs.columns
    assert df_obs["endpoint_score"].isna().mean() > 0.15
    assert df_comp["endpoint_score"].isna().sum() == 0


def test_run_observational_benchmark_smoke() -> None:
    """Verify observational benchmark runs cleanly and returns valid finite metrics."""
    df_res = run_observational_benchmark(datasets=["cps"], quick=True, random_state=42)
    assert isinstance(df_res, pd.DataFrame)
    assert len(df_res) == 6  # 6 methods evaluated
    expected_cols = [
        "dataset",
        "method",
        "n_samples",
        "missing_rate",
        "target_feature",
        "cell_rmse",
        "beta_error",
    ]
    for col in expected_cols:
        assert col in df_res.columns

    # Verify imputed methods have finite cell_rmse
    imputed_rows = df_res[df_res["method"] != "Complete Case Analysis (CCA)"]
    assert np.all(np.isfinite(imputed_rows["cell_rmse"].values))
    assert np.all(np.isfinite(imputed_rows["cell_mae"].values))
    assert np.all(np.isfinite(df_res["beta_error"].values))


def test_ampute_multivariate_additional_coverage() -> None:
    """Verify input validation handles invalid inputs for coverage."""
    rng = np.random.default_rng(42)
    valid_data = rng.normal(0, 1, size=(50, 3))

    # 1. Invalid data type
    with pytest.raises(TypeError, match="data must be a pandas DataFrame or 2D numpy array"):
        ampute_multivariate([1, 2, 3], prop=0.3)

    # 2. Invalid data shape
    with pytest.raises(ValueError, match="Input data array must be 2-dimensional"):
        ampute_multivariate(np.zeros((50, 3, 2)), prop=0.3)

    # 3. Invalid patterns shape
    with pytest.raises(ValueError, match="patterns must be a 2D array"):
        ampute_multivariate(valid_data, prop=0.3, patterns=np.array([1, 1, 0]))

    # 4. Invalid patterns columns
    with pytest.raises(ValueError, match="patterns column count"):
        ampute_multivariate(valid_data, prop=0.3, patterns=np.array([[1, 0]]))

    # 5. Invalid patterns elements
    with pytest.raises(ValueError, match="patterns matrix elements must be binary"):
        ampute_multivariate(valid_data, prop=0.3, patterns=np.array([[1, 2, 0]]))

    # 6. Empty patterns
    with pytest.raises(ValueError, match="patterns matrix must contain at least 1 pattern"):
        ampute_multivariate(valid_data, prop=0.3, patterns=np.zeros((0, 3)))

    # 7. Invalid freq length
    with pytest.raises(ValueError, match="freq length"):
        ampute_multivariate(valid_data, prop=0.3, freq=[0.5])

    # 8. Negative freq values
    with pytest.raises(ValueError, match="freq values must be non-negative"):
        ampute_multivariate(valid_data, prop=0.3, freq=[-0.5, 1.5, 0.0])

    # 9. freq sum to 0
    with pytest.raises(ValueError, match="freq sum must be strictly positive"):
        ampute_multivariate(valid_data, prop=0.3, freq=[0.0, 0.0, 0.0])

    # 10. Invalid mechanisms sequence length
    with pytest.raises(ValueError, match="mechanisms list length"):
        ampute_multivariate(valid_data, prop=0.3, mechanisms=["MAR", "MCAR"])

    # 11. Invalid odds_type sequence length
    with pytest.raises(ValueError, match="odds_type list length"):
        ampute_multivariate(valid_data, prop=0.3, odds_type=["RIGHT", "LEFT"])

    # 12. Invalid odds_type in sequence
    with pytest.raises(ValueError, match="Invalid odds_type"):
        ampute_multivariate(valid_data, prop=0.3, odds_type=["RIGHT", "UNKNOWN", "LEFT"])

    # 13. Invalid weights dimensions
    with pytest.raises(ValueError, match="weights must be a 2D array"):
        ampute_multivariate(valid_data, prop=0.3, weights=np.array([1.0, 0.0, 0.0]))

    # 14. Invalid weights shape
    with pytest.raises(ValueError, match="weights shape"):
        ampute_multivariate(valid_data, prop=0.3, weights=np.zeros((3, 2)))


def test_ampute_multivariate_calibrate_and_rng() -> None:
    """Verify extreme probabilities, rng types, std_scores=False and summary."""
    rng = np.random.default_rng(42)
    valid_data = rng.normal(0, 1, size=(50, 3))

    # 1. Extreme target prop to hit bracket expansion and fallback
    res1 = ampute_multivariate(valid_data, prop=0.999, mechanisms="MAR", odds_type="RIGHT")
    assert res1.empirical_prop > 0.9

    res2 = ampute_multivariate(valid_data, prop=0.001, mechanisms="MAR", odds_type="RIGHT")
    assert res2.empirical_prop < 0.1

    # 2. std_s < 1e-12 check
    res3 = ampute_multivariate(np.zeros((10, 3)), prop=0.3, mechanisms="MAR", random_state=42)
    assert abs(res3.empirical_prop - 0.3) < 0.3  # small n so large variance, just check no error
    assert np.allclose(res3.probabilities, 0.3)

    # 3. rng as Generator
    ampute_multivariate(valid_data, prop=0.3, random_state=np.random.default_rng(42))

    # 4. rng as RandomState
    ampute_multivariate(valid_data, prop=0.3, random_state=np.random.RandomState(42))

    # 5. std_scores=False
    res_no_std = ampute_multivariate(valid_data, prop=0.3, std_scores=False, random_state=42)
    assert res_no_std.probabilities is not None

    # 6. summary method
    summary = res_no_std.summary()
    assert summary["n_samples"] == 50
    assert summary["n_features"] == 3
    assert summary["n_patterns"] == 3


def test_ampute_multivariate_additional_coverage_2() -> None:
    """Verify input validation handles more edge cases for coverage."""
    rng = np.random.default_rng(42)
    valid_data = rng.normal(0, 1, size=(50, 3))

    # 1. Invalid mechanism string in list
    with pytest.raises(ValueError, match="Invalid mechanism 'UNKNOWN'"):
        ampute_multivariate(valid_data, prop=0.3, mechanisms=["MAR", "UNKNOWN", "MCAR"])

    # 2. MNAR default weights when no incomplete variables
    # We pass pattern with all 1s (observed).
    res_mnar = ampute_multivariate(
        valid_data, prop=0.3, patterns=np.array([[1, 1, 1]]), mechanisms="MNAR"
    )
    assert res_mnar.weights.shape == (1, 3)

    # 3. Fallback binary search if brentq fails
    # Let's force an extreme target prop with an odds type that might cause issues, or just a small subset.
    # A single observation raises earlier, but what if std_s > 0 and brentq fails?
    # We will construct an edge case where target_prop is very close to 0 or 1
    # and bounds expansion triggers.
    res_fallback = ampute_multivariate(
        valid_data, prop=0.999999, mechanisms="MAR", odds_type="RIGHT", random_state=42
    )
    assert res_fallback.empirical_prop > 0.9


def test_ampute_multivariate_additional_coverage_3() -> None:
    """Trigger missing lines in _calibrate_logit_shift."""
    from umbra.benchmark.amputation import _calibrate_logit_shift

    # Empty scores
    probs, b = _calibrate_logit_shift(np.array([]), 0.5, "RIGHT")
    assert len(probs) == 0
    assert b == 0.0

    # 2. brentq failure fallback
    # To force brentq failure, we need to create an objective function that doesn't
    # cross 0 or raises an error, or we can use mock.
    # Alternatively, brentq can fail if maxiter is reached or if the root is not bracketed
    # even after expansion. Let's use mock for reliability to hit those lines.

    import unittest.mock as mock

    with mock.patch("umbra.benchmark.amputation.brentq") as mock_brentq:
        mock_brentq.side_effect = ValueError("Mock brentq error")
        scores = np.array([1.0, 2.0, 3.0])
        probs, b = _calibrate_logit_shift(scores, 0.99, "RIGHT")
        assert len(probs) == 3


def test_ampute_multivariate_additional_coverage_4() -> None:
    from umbra.benchmark.amputation import _calibrate_logit_shift

    # 1. To hit `if val_low > 0` and `if val_high < 0`, we need val_low * val_high > 0.
    # If target_prop is very small, objective(lower) and objective(upper) will both be positive.
    # Actually objective = mean(probs) - target_prop.
    # If target_prop is very small (e.g. 1e-10), probs is bounded by 0.
    # At lower = -30, probs ~ 0, target_prop ~ 0, objective ~ 0.
    # Let's mock the objective function internally to return specific values if possible,
    # or just use very small/large arrays.

    # Let's try target_prop = 1e-15 and 1 - 1e-15
    scores = np.array([-100.0, 100.0])
    probs, b = _calibrate_logit_shift(scores, 1e-15, "RIGHT")
    assert len(probs) == 2
    probs2, b2 = _calibrate_logit_shift(scores, 1.0 - 1e-15, "RIGHT")
    assert len(probs2) == 2

    # hit the invalid odds_type fallback `else: arg = centered`
    # The public API validates odds_type, but _calibrate_logit_shift might be called internally
    probs3, b3 = _calibrate_logit_shift(scores, 0.5, "UNKNOWN")
    assert len(probs3) == 2


def test_ampute_multivariate_additional_coverage_5() -> None:
    rng = np.random.default_rng(42)
    valid_data = rng.normal(0, 1, size=(50, 3))

    # To hit 181 (MNAR with all missing in pattern, i.e., incomp_mask is all False which means all are observed)
    # Actually pat == 0 is incomplete. np.any(incomp_mask) is False if ALL are 1 (observed).
    res_1 = ampute_multivariate(
        valid_data, prop=0.3, patterns=np.array([[1, 1, 1]]), mechanisms="MNAR"
    )
    assert res_1 is not None

    # To hit 186 (MAR with all missing in pattern? wait, MAR with ALL incomplete?)
    # pat == 1 is observed. If ALL are incomplete (0), obs_mask is all False.
    res_2 = ampute_multivariate(
        valid_data, prop=0.3, patterns=np.array([[0, 0, 0]]), mechanisms="MAR"
    )
    assert res_2 is not None

    # To hit 370 (pattern with no samples assigned)
    # We provide freq = [1.0, 0.0] to force the second pattern to have no assignments.
    res_3 = ampute_multivariate(
        valid_data, prop=0.3, patterns=np.array([[0, 1, 1], [1, 0, 1]]), freq=[1.0, 0.0]
    )
    assert res_3 is not None
