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
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
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


def _rubin_pool_mean_ci(
    means: List[float], within_vars: List[float], n_obs: int
) -> Tuple[float, float, Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Compute Rubin (1987) pooled mean and confidence intervals with Barnard-Rubin (1999) df."""
    M = len(means)
    q_bar = float(np.mean(means))
    u_bar = float(np.mean(within_vars))
    if M > 1:
        B = float(np.var(means, ddof=1))
        T = u_bar + (1.0 + 1.0 / M) * B
        r = ((1.0 + 1.0 / M) * B) / max(1e-12, u_bar)
        nu_rubin = (M - 1) * (1.0 + 1.0 / r) ** 2 if r > 1e-12 else 1e6
        nu_com = max(1, n_obs - 1)
        lambda_hat = ((1.0 + 1.0 / M) * B) / max(1e-12, T)
        nu_obs = ((nu_com + 1) / (nu_com + 3)) * nu_com * (1.0 - lambda_hat)
        nu = (nu_rubin * nu_obs) / max(1e-6, nu_rubin + nu_obs)
        nu = max(1.0, nu)
    else:
        T = u_bar
        nu = max(1.0, float(n_obs - 1))

    se = float(np.sqrt(max(1e-12, T)))
    t_80 = float(stats.t.ppf(0.90, df=nu))
    t_90 = float(stats.t.ppf(0.95, df=nu))
    t_95 = float(stats.t.ppf(0.975, df=nu))

    ci_80 = (q_bar - t_80 * se, q_bar + t_80 * se)
    ci_90 = (q_bar - t_90 * se, q_bar + t_90 * se)
    ci_95 = (q_bar - t_95 * se, q_bar + t_95 * se)
    return q_bar, se, ci_80, ci_90, ci_95


def _rubin_pool_regression(
    betas: List[np.ndarray], cov_betas: List[np.ndarray], n_obs: int, n_pred: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Rubin (1987) pooled regression coefficients and SEs with Barnard-Rubin df."""
    M = len(betas)
    beta_mat = np.array(betas)  # shape (M, k)
    q_bar = np.mean(beta_mat, axis=0)
    u_bar = np.mean(np.array(cov_betas), axis=0)  # shape (k, k)

    if M > 1:
        B = np.cov(beta_mat, rowvar=False, ddof=1)
        if B.ndim == 0:
            B = np.array([[float(B)]])
        T_diag = np.diag(u_bar) + (1.0 + 1.0 / M) * np.diag(B)
        se = np.sqrt(np.maximum(1e-12, T_diag))
        nu_com = max(1, n_obs - n_pred - 1)
        t_crits = []
        for j in range(len(q_bar)):
            uj = max(1e-12, u_bar[j, j])
            bj = max(0.0, B[j, j])
            r = ((1.0 + 1.0 / M) * bj) / uj
            nu_rubin = (M - 1) * (1.0 + 1.0 / r) ** 2 if r > 1e-12 else 1e6
            lambda_hat = ((1.0 + 1.0 / M) * bj) / max(1e-12, T_diag[j])
            nu_obs = ((nu_com + 1) / (nu_com + 3)) * nu_com * (1.0 - lambda_hat)
            nu_j = (nu_rubin * nu_obs) / max(1e-6, nu_rubin + nu_obs)
            nu_j = max(1.0, nu_j)
            t_crits.append(float(stats.t.ppf(0.975, df=nu_j)))
        t_crit = np.array(t_crits)
    else:
        se = np.sqrt(np.maximum(1e-12, np.diag(u_bar)))
        nu_com = max(1, n_obs - n_pred - 1)
        t_crit = np.full(len(q_bar), float(stats.t.ppf(0.975, df=nu_com)))

    return q_bar, se, t_crit


def evaluate_imputer_replication(
    imputer_factory: Callable[[], Any],
    sim_data: SimulationDataset,
    rep_idx: int,
    true_beta_age: float = 0.5,
    true_beta_edu: float = 0.8,
    n_imputations: int = 1,
) -> ReplicationResult:
    """Execute and evaluate a single Monte Carlo replication.

    Supports single-imputation plug-in evaluation (n_imputations=1) and
    Rubin (1987) multiple imputation pooling with Barnard-Rubin (1999)
    degrees of freedom (n_imputations > 1).
    """
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

        # Imputation draws
        if n_imputations > 1:
            if hasattr(imputer, "fit_transform_multiple"):
                dfs_imp = imputer.fit_transform_multiple(df_obs)
            elif hasattr(imputer, "transform"):
                imputer.fit(df_obs)
                try:
                    dfs_imp = imputer.transform(df_obs, return_all_imputations=True)
                except TypeError:
                    dfs_imp = [imputer.transform(df_obs)]
            else:
                dfs_imp = [imputer.fit_transform(df_obs)]
        else:
            dfs_imp = [imputer.fit_transform(df_obs)]
        runtime = time.perf_counter() - t0

        n_total_obs = len(df_obs)
        all_means: List[float] = []
        all_within_vars: List[float] = []
        all_cell_means: List[float] = []
        all_cell_rmses: List[float] = []
        all_cell_maes: List[float] = []
        all_betas: List[np.ndarray] = []
        all_cov_betas: List[np.ndarray] = []

        for df_imp in dfs_imp:
            if isinstance(df_imp, np.ndarray):
                col_idx = list(df_obs.columns).index(target)
                y_imp_all = df_imp[:, col_idx]
            else:
                y_imp_all = df_imp[target].to_numpy()

            y_imp_mis = y_imp_all[mask]

            m_k = float(np.mean(y_imp_all))
            v_k = float(np.var(y_imp_all, ddof=1) / n_total_obs)
            cm_k = float(np.mean(y_imp_mis)) if n_mis > 0 else m_k
            rmse_k = float(np.sqrt(np.mean((y_imp_mis - y_true_mis) ** 2))) if n_mis > 0 else 0.0
            mae_k = float(np.mean(np.abs(y_imp_mis - y_true_mis))) if n_mis > 0 else 0.0

            all_means.append(m_k)
            all_within_vars.append(v_k)
            all_cell_means.append(cm_k)
            all_cell_rmses.append(rmse_k)
            all_cell_maes.append(mae_k)

            # Downstream regression
            if isinstance(df_imp, pd.DataFrame):
                X_reg = df_imp[["age", "education"]].to_numpy()
            else:
                age_idx = list(df_obs.columns).index("age")
                edu_idx = list(df_obs.columns).index("education")
                X_reg = df_imp[:, [age_idx, edu_idx]]

            reg = LinearRegression().fit(X_reg, y_imp_all)
            b_vec = np.array([float(reg.coef_[0]), float(reg.coef_[1])])
            residuals = y_imp_all - reg.predict(X_reg)
            s_err = np.sqrt(np.sum(residuals**2) / max(1, n_total_obs - 3))
            X_design = np.column_stack([np.ones(n_total_obs), X_reg])
            try:
                cov_b = s_err**2 * np.linalg.inv(X_design.T @ X_design)
                cov_sub = cov_b[1:3, 1:3]
            except Exception:
                cov_sub = np.eye(2) * 0.0025

            all_betas.append(b_vec)
            all_cov_betas.append(cov_sub)

        # Rubin pooled mean and confidence intervals
        imp_mean, imp_se, ci_80, ci_90, ci_95 = _rubin_pool_mean_ci(
            all_means, all_within_vars, n_total_obs
        )
        imp_cell_mean = float(np.mean(all_cell_means))
        cell_rmse = float(np.mean(all_cell_rmses))
        cell_mae = float(np.mean(all_cell_maes))

        mean_bias = imp_mean - true_mean
        cell_bias = imp_cell_mean - float(np.mean(y_true_mis)) if n_mis > 0 else 0.0

        # Rubin pooled regression
        pooled_beta, se_beta, t_crit_beta = _rubin_pool_regression(
            all_betas, all_cov_betas, n_total_obs, n_pred=2
        )
        b_age = float(pooled_beta[0])
        b_edu = float(pooled_beta[1])
        b_age_cov = bool(
            b_age - t_crit_beta[0] * se_beta[0]
            <= true_beta_age
            <= b_age + t_crit_beta[0] * se_beta[0]
        )
        b_edu_cov = bool(
            b_edu - t_crit_beta[1] * se_beta[1]
            <= true_beta_edu
            <= b_edu + t_crit_beta[1] * se_beta[1]
        )

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
        import sys
        import traceback

        traceback.print_exc(file=sys.stderr)
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
    n_imputations: int = 1,
) -> List[MonteCarloSummary]:
    """Execute Monte Carlo simulation across multiple imputers on one regime."""
    results: List[MonteCarloSummary] = []

    cov_protocol = (
        f"Multiple-Imputation (M={n_imputations}, Rubin 1987)"
        if n_imputations > 1
        else "Single-Imputation (Plug-in)"
    )
    print(
        f"\n--- Monte Carlo: {regime} (N={n_samples}, Missing={missing_rate:.0%}, R={n_replications}, Coverage={cov_protocol}) ---"
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
            res = evaluate_imputer_replication(
                factory, sim_data, rep_idx=rep, n_imputations=n_imputations
            )
            replications.append(res)

        summary = aggregate_replications(
            replications=replications,
            method_name=method_name,
            regime=regime,
            n_samples=n_samples,
            missing_rate=missing_rate,
        )
        if summary.convergence_rate < 0.5:
            raise RuntimeError(
                f"High failure rate for method '{method_name}' on regime '{regime}': "
                f"convergence rate is {summary.convergence_rate:.1%} (< 50%)."
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
