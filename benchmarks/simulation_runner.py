"""
Monte Carlo Simulation Engine for Imputation Methods and Coverage Probability.

Executes repeated replications across missingness mechanisms, sample sizes, and rates.
Measures:
- Mean bias & Median bias
- Cell RMSE & MAE
- Empirical Coverage Probability (80%, 90%, 95%)
- 95% Confidence Interval Width
- Downstream Parameter Recovery (beta_age, beta_edu bias & coverage)
- Runtime & Convergence
"""

import time
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from benchmarks.dgps import SimulationDataset, generate_simulation_dataset
from benchmarks.metrics import MonteCarloSummary, ReplicationResult, aggregate_replications
from umbra.api import UmbraImputer
from umbra.imputers.heckman_selection import HeckmanSelectionImputer
from umbra.imputers.mar_chained_equations import (
    MARChainedEquationsImputer,
)
from umbra.imputers.pattern_mixture import PatternMixtureImputer


class CompleteCaseBaseline:
    """Baseline: Listwise deletion (complete-case analysis)."""

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Replaces NaNs by propagating observed rows (or dropping for evaluation)
        return df.dropna().copy()


class MeanImputerBaseline:
    """Baseline: Naive mean imputation."""

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in out.columns:
            if out[c].isna().any():
                out[c] = out[c].fillna(out[c].mean())
        return out


def evaluate_imputer_replication(
    imputer_factory: Callable[[], Any],
    sim_data: SimulationDataset,
    rep_idx: int,
    true_beta_age: float = 0.5,
    true_beta_edu: float = 0.8,
) -> ReplicationResult:
    """Execute and evaluate a single Monte Carlo replication."""
    t0 = time.perf_counter()
    df_obs = sim_data.data_observed.copy()
    df_true = sim_data.data_complete.copy()
    target = sim_data.target_col
    mask = sim_data.mask
    n_mis = int(np.sum(mask))

    y_true_all = df_true[target].to_numpy()
    y_true_mis = y_true_all[mask]
    true_mean = sim_data.true_params["true_mean"]

    try:
        imputer = imputer_factory()
        if isinstance(imputer, CompleteCaseBaseline):
            # Complete-case evaluation
            df_cc = df_obs.dropna()
            runtime = time.perf_counter() - t0
            y_cc = df_cc[target].to_numpy()
            cc_mean = float(np.mean(y_cc))
            cc_se = float(np.std(y_cc, ddof=1) / np.sqrt(max(1, len(y_cc))))

            # Regression on complete cases
            X_cc = df_cc[["age", "education"]].to_numpy()
            reg_cc = LinearRegression().fit(X_cc, y_cc)
            b_age = float(reg_cc.coef_[0])
            b_edu = float(reg_cc.coef_[1])

            ci_80 = (cc_mean - 1.282 * cc_se, cc_mean + 1.282 * cc_se)
            ci_90 = (cc_mean - 1.645 * cc_se, cc_mean + 1.645 * cc_se)
            ci_95 = (cc_mean - 1.960 * cc_se, cc_mean + 1.960 * cc_se)

            # Complete case doesn't impute missing cells, so cell metrics are based on mean
            return ReplicationResult(
                rep_idx=rep_idx,
                imputed_mean=cc_mean,
                imputed_cell_mean=cc_mean,
                mean_bias=cc_mean - true_mean,
                cell_bias=cc_mean - float(np.mean(y_true_mis)),
                cell_rmse=float(np.sqrt(np.mean((cc_mean - y_true_mis) ** 2))),
                cell_mae=float(np.mean(np.abs(cc_mean - y_true_mis))),
                ci_80_covered=bool(ci_80[0] <= true_mean <= ci_80[1]),
                ci_90_covered=bool(ci_90[0] <= true_mean <= ci_90[1]),
                ci_95_covered=bool(ci_95[0] <= true_mean <= ci_95[1]),
                ci_width_95=float(ci_95[1] - ci_95[0]),
                beta_age=b_age,
                beta_edu=b_edu,
                beta_age_covered=bool(abs(b_age - true_beta_age) < 1.96 * 0.05),
                beta_edu_covered=bool(abs(b_edu - true_beta_edu) < 1.96 * 0.05),
                runtime_sec=runtime,
                converged=True,
            )

        # Standard transformer imputer
        df_imp = imputer.fit_transform(df_obs)
        runtime = time.perf_counter() - t0

        if isinstance(df_imp, np.ndarray):
            col_idx = list(df_obs.columns).index(target)
            y_imp_all = df_imp[:, col_idx]
        else:
            y_imp_all = df_imp[target].to_numpy()

        y_imp_mis = y_imp_all[mask]

        imp_mean = float(np.mean(y_imp_all))
        imp_cell_mean = float(np.mean(y_imp_mis)) if n_mis > 0 else imp_mean

        mean_bias = imp_mean - true_mean
        cell_bias = imp_cell_mean - float(np.mean(y_true_mis)) if n_mis > 0 else 0.0
        cell_rmse = float(np.sqrt(np.mean((y_imp_mis - y_true_mis) ** 2))) if n_mis > 0 else 0.0
        cell_mae = float(np.mean(np.abs(y_imp_mis - y_true_mis))) if n_mis > 0 else 0.0

        # Variance of sample mean
        imp_se = float(np.std(y_imp_all, ddof=1) / np.sqrt(len(y_imp_all)))
        ci_80 = (imp_mean - 1.282 * imp_se, imp_mean + 1.282 * imp_se)
        ci_90 = (imp_mean - 1.645 * imp_se, imp_mean + 1.645 * imp_se)
        ci_95 = (imp_mean - 1.960 * imp_se, imp_mean + 1.960 * imp_se)

        # Downstream regression
        if isinstance(df_imp, pd.DataFrame):
            X_reg = df_imp[["age", "education"]].to_numpy()
        else:
            age_idx = list(df_obs.columns).index("age")
            edu_idx = list(df_obs.columns).index("education")
            X_reg = df_imp[:, [age_idx, edu_idx]]

        reg = LinearRegression().fit(X_reg, y_imp_all)
        b_age = float(reg.coef_[0])
        b_edu = float(reg.coef_[1])

        # Regression SE estimation
        n = len(y_imp_all)
        residuals = y_imp_all - reg.predict(X_reg)
        s_err = np.sqrt(np.sum(residuals**2) / max(1, n - 3))
        X_design = np.column_stack([np.ones(n), X_reg])
        try:
            cov_beta = s_err**2 * np.linalg.inv(X_design.T @ X_design)
            se_age = float(np.sqrt(cov_beta[1, 1]))
            se_edu = float(np.sqrt(cov_beta[2, 2]))
        except Exception:
            se_age, se_edu = 0.05, 0.05

        b_age_cov = bool(b_age - 1.96 * se_age <= true_beta_age <= b_age + 1.96 * se_age)
        b_edu_cov = bool(b_edu - 1.96 * se_edu <= true_beta_edu <= b_edu + 1.96 * se_edu)

        return ReplicationResult(
            rep_idx=rep_idx,
            imputed_mean=imp_mean,
            imputed_cell_mean=imp_cell_mean,
            mean_bias=mean_bias,
            cell_bias=cell_bias,
            cell_rmse=cell_rmse,
            cell_mae=cell_mae,
            ci_80_covered=bool(ci_80[0] <= true_mean <= ci_80[1]),
            ci_90_covered=bool(ci_90[0] <= true_mean <= ci_90[1]),
            ci_95_covered=bool(ci_95[0] <= true_mean <= ci_95[1]),
            ci_width_95=float(ci_95[1] - ci_95[0]),
            beta_age=b_age,
            beta_edu=b_edu,
            beta_age_covered=b_age_cov,
            beta_edu_covered=b_edu_cov,
            runtime_sec=runtime,
            converged=True,
        )
    except Exception as e:
        runtime = time.perf_counter() - t0
        return ReplicationResult(
            rep_idx=rep_idx,
            imputed_mean=np.nan,
            imputed_cell_mean=np.nan,
            mean_bias=np.nan,
            cell_bias=np.nan,
            cell_rmse=np.nan,
            cell_mae=np.nan,
            ci_80_covered=False,
            ci_90_covered=False,
            ci_95_covered=False,
            ci_width_95=np.nan,
            beta_age=np.nan,
            beta_edu=np.nan,
            beta_age_covered=False,
            beta_edu_covered=False,
            runtime_sec=runtime,
            converged=False,
            error_msg=str(e),
        )


def run_monte_carlo_regime(
    regime: str,
    imputer_factories: Dict[str, Callable[[], Any]],
    n_replications: int = 25,
    n_samples: int = 2500,
    missing_rate: float = 0.30,
    severity: str = "medium",
    base_seed: int = 42,
) -> List[MonteCarloSummary]:
    """Execute Monte Carlo simulation across multiple imputers on one regime."""
    results: List[MonteCarloSummary] = []

    print(
        f"\n--- Monte Carlo: {regime} (N={n_samples}, Missing={missing_rate:.0%}, R={n_replications}) ---"
    )

    for method_name, factory in imputer_factories.items():
        replications: List[ReplicationResult] = []
        for rep in range(n_replications):
            seed = base_seed + rep * 17
            sim_data = generate_simulation_dataset(
                mechanism=regime,
                n_samples=n_samples,
                missing_rate=missing_rate,
                severity=severity,
                random_state=seed,
            )
            res = evaluate_imputer_replication(factory, sim_data, rep_idx=rep)
            replications.append(res)

        summary = aggregate_replications(
            replications=replications,
            method_name=method_name,
            regime=regime,
            n_samples=n_samples,
            missing_rate=missing_rate,
        )
        print(
            f"  [{method_name:24s}] Bias: {summary.mean_bias:+.3f} | "
            f"RMSE: {summary.cell_rmse:.3f} | 95% Cov: {summary.coverage_95:.1%} | "
            f"Beta Total RMSE: {summary.beta_total_rmse:.3f} | Time: {summary.avg_runtime_sec:.3f}s"
        )
        results.append(summary)

    return results


def get_standard_imputer_suite(
    shadow_col: str = "shadow_z", random_state: int = 42
) -> Dict[str, Callable[[], Any]]:
    """Return factory dict for standard baseline suite and Umbra."""
    return {
        "Complete-Case": lambda: CompleteCaseBaseline(),
        "Naive Mean": lambda: MeanImputerBaseline(),
        "MAR MICE (PMM)": lambda: MARChainedEquationsImputer(
            imputation_method="pmm", random_state=random_state
        ),
        "MAR MICE (Ridge)": lambda: MARChainedEquationsImputer(
            imputation_method="ridge", random_state=random_state
        ),
        "Heckman Selection": lambda: HeckmanSelectionImputer(
            shadow_cols={"target": shadow_col}, random_state=random_state
        ),
        "Pattern Mixture (delta=0)": lambda: PatternMixtureImputer(
            delta=0.0, random_state=random_state
        ),
        "Umbra (Auto)": lambda: UmbraImputer(
            strategy="auto",
            shadow_cols={"target": shadow_col},
            run_sensitivity=False,
            random_state=random_state,
        ),
    }
