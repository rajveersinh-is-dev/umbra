"""
Explicit Data Leakage & Information Boundary Tests.

Verifies that:
1. Diagnostics strictly evaluate observable data and never leak complete ground truth.
2. Imputers respect train/test boundaries and do not fit on test distributions.
3. Auxiliary (shadow) variable discovery uses only observed rows for outcome correlations.
4. Imputer transform does not leak test statistics into fitted estimator state.
"""

import numpy as np
import pandas as pd

import umbra
from umbra.api import UmbraImputer
from umbra.diagnostics.shadow_variable_finder import find_shadow_variables


def test_diagnostics_contain_no_ground_truth_leakage():
    """Verify diagnostics only access observed values and never require or inspect ground truth."""
    rng = np.random.RandomState(42)
    n = 200
    df_obs = pd.DataFrame(
        {
            "age": rng.normal(40, 10, n),
            "income": rng.normal(50000, 15000, n),
            "education": rng.normal(14, 2, n),
        }
    )
    # Mask half of income
    df_obs.loc[df_obs["age"] > 45, "income"] = np.nan

    report = umbra.diagnose(df_obs, target_cols=["income"])

    # Diagnostics report must only contain observable empirical metrics
    assert report.mcar.statistic is not None
    assert "income" in report.covariate_shift.variable_reports
    # The report must not contain any ground truth attributes
    assert not hasattr(report, "ground_truth")
    assert not hasattr(report, "true_mean")


def test_train_test_split_isolation():
    """Verify that fit(X_train) and transform(X_test) maintain strict separation without test leakage."""
    rng = np.random.RandomState(42)
    n = 300
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(5, 2, n),
            "target": rng.normal(10, 3, n),
        }
    )
    df.loc[0:50, "target"] = np.nan

    train_df = df.iloc[:150].copy()
    test_df = df.iloc[150:].copy()

    imputer = UmbraImputer(strategy="mar", random_state=42)
    imputer.fit(train_df)

    # Snapshot fitted state
    fitted_cols = list(imputer.feature_names_in_)
    fitted_strategy = imputer.strategy_map_.copy()

    # Transform test set
    test_imputed = imputer.transform(test_df)

    # Assert test transformation did not mutate fitted parameters or columns
    assert list(imputer.feature_names_in_) == fitted_cols
    assert imputer.strategy_map_ == fitted_strategy
    assert isinstance(test_imputed, pd.DataFrame)
    assert len(test_imputed) == len(test_df)


def test_shadow_variable_outcome_leakage_boundary():
    """Verify find_shadow_variables computes correlations strictly on observed rows."""
    rng = np.random.RandomState(42)
    n = 200
    z = rng.randn(n)
    y = 2.0 * z + rng.randn(n)

    # Missingness on y
    is_missing = z > 0.0
    y_obs = y.copy()
    y_obs[is_missing] = np.nan

    df = pd.DataFrame({"z": z, "y": y_obs})

    rep = find_shadow_variables(df, target_column="y")

    # The direct outcome correlation must match the correlation strictly on observed cases
    if rep.best_candidate:
        cand = rep.best_candidate
        expected_r = float(np.corrcoef(z[~is_missing], y_obs[~is_missing])[0, 1])
        assert np.isclose(cand.direct_outcome_correlation, expected_r, atol=1e-5)


def test_benchmark_simulation_dataset_isolation():
    """Verify simulation datasets clearly separate complete data from observed data."""
    from benchmarks.dgps import generate_simulation_dataset

    sim = generate_simulation_dataset(
        mechanism="MNAR_SELECTION", n_samples=500, missing_rate=0.30, random_state=42
    )

    # Check observed data contains NaNs
    assert sim.data_observed[sim.target_col].isna().sum() > 0
    # Check complete data has zero NaNs
    assert sim.data_complete[sim.target_col].isna().sum() == 0

    # Ensure imputers receive only observed data
    imputer = UmbraImputer(strategy="auto", random_state=42)
    imputer.fit(sim.data_observed)

    # Verify imputer does not access complete data
    assert not hasattr(imputer, "data_complete")
