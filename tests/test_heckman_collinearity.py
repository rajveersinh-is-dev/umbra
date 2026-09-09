"""
Unit tests for Heckman selection ill-conditioned design matrix handling,
Ridge regularized sandwich covariance matrix, and VIF / condition index diagnostics.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from umbra.imputers.heckman_selection import (
    HeckmanCollinearityWarning,
    HeckmanSelectionImputer,
    HeckmanSEWarning,
)


def _generate_collinear_heckman_data(n: int = 150, seed: int = 42) -> pd.DataFrame:
    """Generate dataset where outcome covariates are near-collinear."""
    rng = np.random.RandomState(seed)
    w1 = rng.randn(n)
    w2 = rng.randn(n)

    # Selection equation
    z_star = 0.5 + 0.8 * w1 - 0.6 * w2 + rng.randn(n)
    R = (z_star > 0).astype(int)

    # Outcome covariates: x1 and x2 are almost collinear
    x1 = rng.randn(n)
    x2 = x1 + 1e-5 * rng.randn(n)  # extreme collinearity

    # Outcome equation
    y = 1.5 + 2.0 * x1 - 1.0 * x2 + rng.randn(n)
    y[R == 0] = np.nan

    return pd.DataFrame({"w1": w1, "w2": w2, "x1": x1, "x2": x2, "y": y})


def test_collinearity_vif_and_condition_index() -> None:
    """Verify condition index and VIF computation on known collinear design matrix."""
    imp = HeckmanSelectionImputer()
    rng = np.random.RandomState(42)
    n = 100
    x1 = rng.randn(n)
    x2 = 2.0 * x1 + 1e-4 * rng.randn(n)
    x_const = np.ones(n)
    lambda_1 = rng.randn(n)
    design = np.column_stack([x_const, x1, x2, lambda_1])

    diag = imp._compute_collinearity_diagnostics(design, col_names=["const", "x1", "x2", "lambda"])

    assert "condition_number" in diag
    assert "condition_indices" in diag
    assert "vif" in diag
    assert "vif_lambda" in diag
    assert "severe_collinearity" in diag

    # Extreme collinearity between x1 and x2
    assert diag["condition_number"] > 30.0, (
        f"Expected condition number > 30, got {diag['condition_number']}"
    )
    assert diag["vif"]["x1"] > 10.0, f"Expected VIF(x1) > 10, got {diag['vif']['x1']}"
    assert diag["vif"]["x2"] > 10.0, f"Expected VIF(x2) > 10, got {diag['vif']['x2']}"
    assert diag["severe_collinearity"] is True


def test_collinearity_warning_emitted() -> None:
    """Verify HeckmanCollinearityWarning is emitted when collinearity is severe."""
    df = _generate_collinear_heckman_data(n=120, seed=42)
    imp = HeckmanSelectionImputer(
        target_cols=["y"],
        shadow_cols={"y": "w2"},
        ridge_se_method="sandwich",
        random_state=42,
    )

    with pytest.warns(HeckmanCollinearityWarning, match="Severe collinearity detected"):
        imp.fit(df)

    assert imp.collinearity_diagnostics_["severe_collinearity"] is True
    assert imp.collinearity_diagnostics_["condition_number"] > 30.0


def test_ridge_sandwich_covariance_properties() -> None:
    """Verify Ridge sandwich covariance is symmetric, positive semi-definite, and finite."""
    rng = np.random.RandomState(42)
    n1 = 60
    p = 4
    X_star = rng.randn(n1, p)
    # Make rank deficient
    X_star[:, 2] = X_star[:, 1]

    W_obs = rng.randn(n1, 3)
    y_obs = rng.randn(n1)
    beta_star = np.array([1.0, 2.0, 2.0, 0.5])
    V1 = np.eye(3) * 0.1
    delta = rng.uniform(0.1, 0.9, size=n1)

    imp = HeckmanSelectionImputer()
    V_ridge = imp._compute_ridge_sandwich_covariance(
        X_star=X_star,
        W_obs=W_obs,
        y_obs=y_obs,
        beta_star=beta_star,
        V1=V1,
        delta=delta,
        alpha=1.0,
    )

    # 1. Finite and correct shape
    assert V_ridge.shape == (p, p)
    assert np.all(np.isfinite(V_ridge))

    # 2. Symmetry
    np.testing.assert_allclose(V_ridge, V_ridge.T, atol=1e-10)

    # 3. Positive semi-definiteness (all eigenvalues >= 0)
    eigvals = np.linalg.eigvalsh(V_ridge)
    assert np.all(eigvals >= -1e-12), f"Expected all eigenvalues >= 0, got {eigvals}"


def test_ridge_sandwich_replaces_nans() -> None:
    """Verify that under Ridge fallback, ridge_se_method='sandwich' produces finite positive standard errors."""
    rng = np.random.RandomState(42)
    N = 100
    x1 = rng.randn(N)
    x2 = x1.copy()  # Perfect collinearity triggers Ridge fallback
    z = rng.randn(N)
    y = 1.0 + 2.0 * x1 + rng.randn(N)
    y[:30] = np.nan

    df_sing = pd.DataFrame({"x1": x1, "x2": x2, "z": z, "y": y})

    imp_sandwich = HeckmanSelectionImputer(
        target_cols=["y"],
        shadow_cols={"y": "z"},
        ridge_alpha=1.0,
        ridge_se_method="sandwich",
        random_state=42,
    )

    with pytest.warns(HeckmanSEWarning, match="Ridge regression fallback was used"):
        imp_sandwich.fit(df_sing)

    se_sandwich = imp_sandwich.models_["y"]["std_errors"]
    assert not np.any(np.isnan(se_sandwich)), f"Expected finite SEs, got {se_sandwich}"
    assert np.all(se_sandwich > 0.0), f"Expected strictly positive SEs, got {se_sandwich}"
    assert imp_sandwich.ridge_fallback_used_ is True
    assert imp_sandwich.coef_cov_.shape == (len(se_sandwich), len(se_sandwich))
    assert not np.any(np.isnan(imp_sandwich.coef_cov_))

    # Imputation transforms successfully
    df_imputed = imp_sandwich.transform(df_sing)
    assert isinstance(df_imputed, pd.DataFrame)
    assert not df_imputed["y"].isna().any()


def test_ridge_se_method_nan_legacy_mode() -> None:
    """Verify that ridge_se_method='nan' preserves legacy NaN standard errors."""
    rng = np.random.RandomState(42)
    N = 100
    x1 = rng.randn(N)
    x2 = x1.copy()
    z = rng.randn(N)
    y = 1.0 + 2.0 * x1 + rng.randn(N)
    y[:30] = np.nan

    df_sing = pd.DataFrame({"x1": x1, "x2": x2, "z": z, "y": y})

    imp_nan = HeckmanSelectionImputer(
        target_cols=["y"],
        shadow_cols={"y": "z"},
        ridge_se_method="nan",
        random_state=42,
    )

    with pytest.warns(HeckmanSEWarning, match="Standard errors are set to NaN"):
        imp_nan.fit(df_sing)

    se_nan = imp_nan.models_["y"]["std_errors"]
    assert np.all(np.isnan(se_nan)), f"Expected all NaN std_errors, got {se_nan}"


def test_ridge_alpha_regularization_effect() -> None:
    """Verify that increasing ridge_alpha conditions the covariance matrix."""
    rng = np.random.RandomState(42)
    n1 = 50
    p = 3
    X_star = rng.randn(n1, p)
    X_star[:, 1] = X_star[:, 0] + 1e-4 * rng.randn(n1)

    W_obs = rng.randn(n1, 2)
    y_obs = rng.randn(n1)
    beta_star = np.array([1.0, 1.0, 0.5])
    V1 = np.eye(2) * 0.1
    delta = np.full(n1, 0.5)

    imp = HeckmanSelectionImputer()
    V_small_alpha = imp._compute_ridge_sandwich_covariance(
        X_star=X_star, W_obs=W_obs, y_obs=y_obs, beta_star=beta_star, V1=V1, delta=delta, alpha=0.1
    )
    V_large_alpha = imp._compute_ridge_sandwich_covariance(
        X_star=X_star, W_obs=W_obs, y_obs=y_obs, beta_star=beta_star, V1=V1, delta=delta, alpha=10.0
    )

    # Larger alpha shrinks the inverse (X'X + alpha I)^-1, reducing standard error magnitudes
    trace_small = np.trace(V_small_alpha)
    trace_large = np.trace(V_large_alpha)
    assert trace_large < trace_small, (
        f"Expected trace with large alpha ({trace_large}) < small alpha ({trace_small})"
    )


def test_heckman_sklearn_clone_and_parameter_validation() -> None:
    """Verify scikit-learn clone compatibility and parameter validation for Ridge settings."""
    imp = HeckmanSelectionImputer(ridge_alpha=2.5, ridge_se_method="sandwich")
    cloned = clone(imp)
    assert cloned.ridge_alpha == 2.5
    assert cloned.ridge_se_method == "sandwich"

    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1.0, np.nan, 2.0, 3.0]})

    bad_method = HeckmanSelectionImputer(ridge_se_method="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ridge_se_method must be 'sandwich' or 'nan'"):
        bad_method.fit(df)

    bad_alpha = HeckmanSelectionImputer(ridge_alpha=-1.0)
    with pytest.raises(ValueError, match="ridge_alpha must be positive"):
        bad_alpha.fit(df)
