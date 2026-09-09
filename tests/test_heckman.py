"""
Unit tests for Heckman selection model asymptotic standard errors and FIML estimation.

Tests:
  - Murphy & Topel (1985) two-step asymptotic covariance matrix correction
  - Positive semi-definiteness, symmetry, and variance inflation over naive OLS
  - Comparison between Murphy-Topel and paired bootstrap standard errors
  - Full-Information Maximum Likelihood (FIML) joint bivariate normal estimation
  - FIML parameter recovery on synthetic DGPs with known ground truth
  - FIML Hessian inversion and Delta-method standard errors
  - Graceful fallback from FIML to two-step with HeckmanConvergenceWarning
  - Scikit-learn Pipeline compatibility
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from umbra.imputers.heckman_selection import (
    HeckmanConvergenceWarning,
    HeckmanSelectionImputer,
    HeckmanSEWarning,
)


def _generate_heckman_dgp(
    n_samples: int = 800,
    beta: tuple = (1.0, 2.0),
    gamma: tuple = (0.3, 0.8, -0.6),
    sigma: float = 1.5,
    rho: float = 0.6,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic selection DGP with known ground truth parameters."""
    rng = np.random.RandomState(random_state)

    w1 = rng.randn(n_samples)
    w2 = rng.randn(n_samples)
    x1 = w1 + 0.4 * rng.randn(n_samples)

    u = rng.randn(n_samples)
    e = rho * u + np.sqrt(max(1e-6, 1.0 - rho**2)) * rng.randn(n_samples)
    eps = sigma * e

    z_star = gamma[0] + gamma[1] * w1 + gamma[2] * w2 + u
    R = (z_star > 0).astype(int)

    y = beta[0] + beta[1] * x1 + eps
    y[R == 0] = np.nan

    return pd.DataFrame({"x1": x1, "w2": w2, "y": y})


def test_murphy_topel_covariance_shape_and_positivity() -> None:
    """Verify Murphy-Topel covariance matrix is symmetric, positive semi-definite, and correct shape."""
    df = _generate_heckman_dgp(n_samples=600, random_state=42)

    imp = HeckmanSelectionImputer(
        target_cols=["y"],
        shadow_cols={"y": "w2"},
        method="two-step",
        se_method="murphy-topel",
        random_state=42,
    )
    imp.fit(df)

    assert imp.method_used_ == "two-step"
    cov = imp.coef_cov_

    # Design has constant + x1 + lambda = 3 parameters
    assert cov.shape == (3, 3), f"Expected cov shape (3, 3), got {cov.shape}"
    assert np.allclose(cov, cov.T, atol=1e-10), "Murphy-Topel covariance matrix must be symmetric"

    # Eigenvalues must be non-negative (positive semi-definite)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -1e-10), f"Covariance matrix has negative eigenvalues: {eigvals}"

    # Check standard errors
    se_coef = imp.coef_stderr_
    assert len(se_coef) == 2, f"Expected 2 outcome coefficients, got {len(se_coef)}"
    assert np.all(se_coef > 0), "All outcome coefficient SEs must be strictly positive"
    assert imp.mills_stderr_ > 0, "Mills ratio coefficient SE must be strictly positive"

    # Check model dict consistency
    model_y = imp.models_["y"]
    assert np.allclose(model_y["std_errors"][:-1], se_coef)
    assert np.isclose(model_y["std_errors"][-1], imp.mills_stderr_)


def test_murphy_topel_greater_than_naive_ols() -> None:
    """Verify Murphy-Topel standard errors account for first-stage estimation uncertainty and exceed naive OLS."""
    df = _generate_heckman_dgp(n_samples=700, random_state=123)

    # 1. Fit with Murphy-Topel
    imp_mt = HeckmanSelectionImputer(
        target_cols=["y"],
        shadow_cols={"y": "w2"},
        method="two-step",
        se_method="murphy-topel",
        random_state=123,
    )
    imp_mt.fit(df)
    se_mt = imp_mt.models_["y"]["std_errors"]

    # 2. Fit with naive OLS
    imp_naive = HeckmanSelectionImputer(
        target_cols=["y"],
        shadow_cols={"y": "w2"},
        method="two-step",
        se_method="naive",
        random_state=123,
    )
    with pytest.warns(HeckmanSEWarning):
        imp_naive.fit(df)
    se_naive = imp_naive.models_["y"]["std_errors"]

    # Point estimates should be identical (same two-step estimator)
    assert np.allclose(imp_mt.models_["y"]["params"], imp_naive.models_["y"]["params"]), (
        "Parameters must be identical between Murphy-Topel and naive OLS"
    )

    # Mills ratio standard error must reflect variance inflation from generated regressor
    assert se_mt[-1] > se_naive[-1], (
        f"Expected Murphy-Topel SE for lambda ({se_mt[-1]:.4f}) > naive OLS SE ({se_naive[-1]:.4f})"
    )

    # Overall average standard error must be strictly larger
    assert np.mean(se_mt) > np.mean(se_naive), (
        f"Expected mean Murphy-Topel SE ({np.mean(se_mt):.4f}) > mean naive SE ({np.mean(se_naive):.4f})"
    )


def test_murphy_topel_matches_bootstrap_order_of_magnitude() -> None:
    """Verify analytical Murphy-Topel standard errors are consistent with paired bootstrap standard errors."""
    df = _generate_heckman_dgp(n_samples=500, random_state=789)

    imp_mt = HeckmanSelectionImputer(
        target_cols=["y"],
        shadow_cols={"y": "w2"},
        method="two-step",
        se_method="murphy-topel",
        random_state=789,
    )
    imp_mt.fit(df)
    se_mt = imp_mt.models_["y"]["std_errors"]

    imp_boot = HeckmanSelectionImputer(
        target_cols=["y"],
        shadow_cols={"y": "w2"},
        method="two-step",
        se_method="bootstrap",
        n_bootstrap_se=120,
        random_state=789,
    )
    imp_boot.fit(df)
    se_boot = imp_boot.models_["y"]["std_errors"]

    assert not np.any(np.isnan(se_boot)), "Bootstrap SEs must not contain NaNs"

    # Both should be of the same order of magnitude: ratio between 0.4 and 2.5
    ratios = se_mt / se_boot
    assert np.all((ratios > 0.4) & (ratios < 2.5)), (
        f"Murphy-Topel and bootstrap standard errors diverge significantly: "
        f"se_mt={se_mt}, se_boot={se_boot}, ratios={ratios}"
    )


def test_fiml_estimation_accuracy() -> None:
    """Verify HeckmanSelectionImputer(method='fiml') recovers true parameters on synthetic DGP."""
    true_beta = (1.0, 2.0)
    true_sigma = 1.5
    true_rho = 0.6
    df = _generate_heckman_dgp(
        n_samples=1000,
        beta=true_beta,
        sigma=true_sigma,
        rho=true_rho,
        random_state=101,
    )

    imp_fiml = HeckmanSelectionImputer(
        target_cols=["y"],
        shadow_cols={"y": "w2"},
        method="fiml",
        random_state=101,
    )
    imp_fiml.fit(df)

    assert imp_fiml.method_used_ == "fiml"
    assert imp_fiml.log_likelihood_ is not None
    assert np.isfinite(imp_fiml.log_likelihood_)
    assert imp_fiml.log_likelihood_ < 0  # Log-likelihood is negative

    beta_hat = imp_fiml.models_["y"]["beta"]
    sigma_hat = imp_fiml.sigma_
    rho_hat = imp_fiml.rho_

    # Parameter recovery within sampling error (sample size 1000)
    assert np.isclose(beta_hat[0], true_beta[0], atol=0.35), (
        f"Intercept recovery error: {beta_hat[0]} vs {true_beta[0]}"
    )
    assert np.isclose(beta_hat[1], true_beta[1], atol=0.35), (
        f"Slope recovery error: {beta_hat[1]} vs {true_beta[1]}"
    )
    assert np.isclose(sigma_hat, true_sigma, atol=0.35), (
        f"Sigma recovery error: {sigma_hat} vs {true_sigma}"
    )
    assert np.isclose(rho_hat, true_rho, atol=0.35), f"Rho recovery error: {rho_hat} vs {true_rho}"


def test_fiml_standard_errors_positive() -> None:
    """Verify FIML Hessian inversion produces strictly positive standard errors."""
    df = _generate_heckman_dgp(n_samples=600, random_state=55)

    imp = HeckmanSelectionImputer(
        target_cols=["y"],
        shadow_cols={"y": "w2"},
        method="fiml",
        random_state=55,
    )
    imp.fit(df)

    assert imp.method_used_ == "fiml"
    assert np.all(imp.coef_stderr_ > 0), "All FIML coefficient standard errors must be positive"
    assert imp.mills_stderr_ > 0, "FIML mills ratio standard error must be positive"

    cov = imp.coef_cov_
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T, atol=1e-8), "FIML covariance matrix must be symmetric"
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0), f"FIML covariance matrix must be positive definite: {eigvals}"


def test_fiml_convergence_fallback() -> None:
    """Verify FIML gracefully falls back to two-step with HeckmanConvergenceWarning on degenerate data."""
    rng = np.random.RandomState(42)
    n = 150
    # Construct collinear data that causes FIML optimization / Hessian failure
    x1 = rng.randn(n)
    w2 = x1 + 1e-7 * rng.randn(n)  # Nearly collinear
    y = 2.0 * x1 + 0.1 * rng.randn(n)
    y[:40] = np.nan

    df_collinear = pd.DataFrame({"x1": x1, "w2": w2, "y": y})

    imp = HeckmanSelectionImputer(
        target_cols=["y"],
        shadow_cols={"y": "w2"},
        method="fiml",
        random_state=42,
    )

    # Should catch non-convergence or singular Hessian, emit warning, and fall back to two-step
    with pytest.warns(HeckmanConvergenceWarning, match="falling back to two-step"):
        imp.fit(df_collinear)

    assert imp.method_used_ == "two-step"
    assert not np.isnan(imp.sigma_)

    # Imputation should still complete successfully
    df_imputed = imp.transform(df_collinear)
    assert isinstance(df_imputed, pd.DataFrame)
    assert not df_imputed["y"].isna().any()


def test_fiml_sklearn_pipeline_compatibility() -> None:
    """Verify HeckmanSelectionImputer with method='fiml' functions in scikit-learn Pipeline."""
    df = _generate_heckman_dgp(n_samples=500, random_state=42)

    pipeline = Pipeline(
        [
            (
                "imputer",
                HeckmanSelectionImputer(
                    target_cols=["y"],
                    shadow_cols={"y": "w2"},
                    method="fiml",
                    random_state=42,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    transformed = pipeline.fit_transform(df)
    assert transformed.shape == df.shape
    assert not np.any(np.isnan(transformed))

    imputer_step = pipeline.named_steps["imputer"]
    feature_names = imputer_step.get_feature_names_out()
    assert list(feature_names) == ["x1", "w2", "y"]


def test_se_method_naive_emits_warning() -> None:
    """Verify HeckmanSelectionImputer(se_method='naive') emits HeckmanSEWarning."""
    df = _generate_heckman_dgp(n_samples=300, random_state=42)

    imp = HeckmanSelectionImputer(
        target_cols=["y"],
        shadow_cols={"y": "w2"},
        se_method="naive",
        random_state=42,
    )
    with pytest.warns(HeckmanSEWarning, match="naive OLS standard errors"):
        imp.fit(df)

    assert imp.method_used_ == "two-step"
    assert len(imp.coef_stderr_) == 2


def test_heckman_stochastic_imputation_draws_vary() -> None:
    """Verify stochastic=True and multiple imputations generate varying non-identical draws."""
    df = _generate_heckman_dgp(n_samples=400, random_state=42)

    imp = HeckmanSelectionImputer(
        target_cols=["y"],
        shadow_cols={"y": "w2"},
        stochastic=True,
        n_imputations=3,
        random_state=42,
    )
    draws = imp.fit_transform_multiple(df)

    assert len(draws) == 3
    for d in draws:
        assert not d["y"].isna().any()

    # The missing values should differ between draw 0 and draw 1
    mis_mask = df["y"].isna()
    diff = np.abs(draws[0].loc[mis_mask, "y"].values - draws[1].loc[mis_mask, "y"].values)
    assert np.all(diff > 0), "Stochastic draws should not produce identical values for missing rows"
