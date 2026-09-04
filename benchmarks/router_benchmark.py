"""
Independent Benchmark Suite for Umbra's Auto Strategy Router.

Evaluates:
- Correct Strategy Selection Rate across MCAR, MAR, MNAR regimes
- Incorrect Strategy Selection Rate
- False Alarm Rate (MCAR/MAR flagged as severe MNAR)
- Missed Risk Rate (MNAR classified as pure MCAR/low risk)
- Sensitivity to Sample Size N in {250, 500, 1000, 2500, 5000, 10000}
- Sensitivity to Missingness Rate in {10%, 20%, 30%, 40%, 50%}
- Sensitivity to Noise / SNR
- Full Confusion Matrix
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from benchmarks.dgps import generate_simulation_dataset
from umbra.api import UmbraImputer


def wilson_score_interval(
    successes: int, total: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion.

    References:
    Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference.
    Journal of the American Statistical Association, 22(158), 209-212.
    """
    if total <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # 95% two-sided normal critical value
    p_hat = successes / total
    denom = 1.0 + (z**2) / total
    center = (p_hat + (z**2) / (2.0 * total)) / denom
    margin = (z / denom) * np.sqrt((p_hat * (1.0 - p_hat) / total) + ((z**2) / (4.0 * (total**2))))
    lower = float(max(0.0, center - margin))
    upper = float(min(1.0, center + margin))
    return (lower, upper)


@dataclass
class RouterEvaluationSummary:
    """Performance summary of the Auto Strategy Router with rigorous confidence intervals."""

    overall_accuracy: float
    overall_accuracy_ci: Tuple[float, float]
    mcar_correct_rate: float
    mcar_correct_rate_ci: Tuple[float, float]
    mar_correct_rate: float
    mar_correct_rate_ci: Tuple[float, float]
    mnar_correct_rate: float
    mnar_correct_rate_ci: Tuple[float, float]
    false_alarm_rate: float
    false_alarm_rate_ci: Tuple[float, float]
    missed_risk_rate: float
    missed_risk_rate_ci: Tuple[float, float]
    n_total: int
    n_correct: int
    confusion_matrix: pd.DataFrame
    results_by_sample_size: pd.DataFrame
    results_by_missing_rate: pd.DataFrame


def evaluate_single_routing(
    mechanism: str,
    n_samples: int = 1000,
    missing_rate: float = 0.30,
    severity: str = "medium",
    snr: float = 3.0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Generate a single dataset and determine router classification."""
    sim = generate_simulation_dataset(
        mechanism=mechanism,
        n_samples=n_samples,
        missing_rate=missing_rate,
        severity=severity,
        snr=snr,
        random_state=random_state,
    )

    imputer = UmbraImputer(
        strategy="auto",
        shadow_cols={sim.target_col: sim.shadow_col} if sim.shadow_col else None,
        run_sensitivity=False,
        random_state=random_state,
    )

    imputer.fit(sim.data_observed)
    decision = imputer.routing_decisions_.get(sim.target_col, "mar_chained_equations")
    risk_rep = imputer.diagnostics_.get(sim.target_col)
    risk_level = risk_rep.risk_level if risk_rep else "LOW"

    # Define Ground Truth Expectations:
    # - MCAR: Expected strategy is 'mar_chained_equations' (risk LOW)
    # - MAR: Expected strategy is 'mar_chained_equations' (risk LOW or MEDIUM without tail concentration)
    is_mcar = mechanism.upper() == "MCAR"
    is_mar = mechanism.upper() == "MAR"

    if is_mcar:
        is_correct = decision == "mar_chained_equations"
        is_false_alarm = risk_level == "HIGH"
        is_missed_risk = False
    elif is_mar:
        is_correct = decision == "mar_chained_equations"
        is_false_alarm = risk_level == "HIGH"
        is_missed_risk = False
    else:
        # MNAR
        is_correct = decision in ("heckman_selection", "pattern_mixture") or risk_level in (
            "HIGH",
            "MEDIUM",
        )
        is_false_alarm = False
        is_missed_risk = risk_level == "LOW"

    return {
        "mechanism": mechanism,
        "n_samples": n_samples,
        "missing_rate": missing_rate,
        "severity": severity,
        "decision": decision,
        "risk_level": risk_level,
        "is_correct": bool(is_correct),
        "is_false_alarm": bool(is_false_alarm),
        "is_missed_risk": bool(is_missed_risk),
    }


def benchmark_auto_router(
    n_replications_per_cell: int = 10,
    sample_sizes: Optional[List[int]] = None,
    missing_rates: Optional[List[float]] = None,
    base_seed: int = 42,
) -> RouterEvaluationSummary:
    """Benchmark Auto router systematically across sample sizes, missingness rates, and mechanisms."""
    if sample_sizes is None:
        sample_sizes = [250, 500, 1000, 2500, 5000]
    if missing_rates is None:
        missing_rates = [0.10, 0.20, 0.30, 0.40, 0.50]

    mechanisms = [
        "MCAR",
        "MAR",
        "MNAR_SELF_MASKING",
        "MNAR_SELECTION",
        "MNAR_PATTERN_MIXTURE",
        "MNAR_TAILS",
    ]

    records = []
    run_id = 0

    print("Running Auto Router benchmark matrix...")
    for mech in mechanisms:
        for n in sample_sizes:
            for rate in missing_rates:
                for rep in range(n_replications_per_cell):
                    run_id += 1
                    seed = base_seed + run_id
                    res = evaluate_single_routing(
                        mechanism=mech,
                        n_samples=n,
                        missing_rate=rate,
                        severity="medium",
                        random_state=seed,
                    )
                    records.append(res)

    df = pd.DataFrame(records)

    # Calculate overall rates
    mcar_mask = df["mechanism"] == "MCAR"
    mar_mask = df["mechanism"] == "MAR"
    mnar_mask = df["mechanism"].str.startswith("MNAR")

    # Calculate overall rates and Wilson score 95% confidence intervals
    n_total = len(df)
    n_correct = int(df["is_correct"].sum())
    overall_ci = wilson_score_interval(n_correct, n_total)

    n_mcar = int(mcar_mask.sum())
    n_mcar_correct = int(df.loc[mcar_mask, "is_correct"].sum())
    mcar_ci = wilson_score_interval(n_mcar_correct, n_mcar)

    n_mar = int(mar_mask.sum())
    n_mar_correct = int(df.loc[mar_mask, "is_correct"].sum())
    mar_ci = wilson_score_interval(n_mar_correct, n_mar)

    n_mnar = int(mnar_mask.sum())
    n_mnar_correct = int(df.loc[mnar_mask, "is_correct"].sum())
    mnar_ci = wilson_score_interval(n_mnar_correct, n_mnar)

    n_mcar_mar = int((mcar_mask | mar_mask).sum())
    n_false_alarm = int(df.loc[mcar_mask | mar_mask, "is_false_alarm"].sum())
    false_alarm_ci = wilson_score_interval(n_false_alarm, n_mcar_mar)

    n_missed_risk = int(df.loc[mnar_mask, "is_missed_risk"].sum())
    missed_risk_ci = wilson_score_interval(n_missed_risk, n_mnar)

    mcar_correct = float(n_mcar_correct / n_mcar) if n_mcar > 0 else 1.0
    mar_correct = float(n_mar_correct / n_mar) if n_mar > 0 else 1.0
    mnar_correct = float(n_mnar_correct / n_mnar) if n_mnar > 0 else 1.0
    overall_acc = float(n_correct / n_total) if n_total > 0 else 1.0
    false_alarm = float(n_false_alarm / n_mcar_mar) if n_mcar_mar > 0 else 0.0
    missed_risk = float(n_missed_risk / n_mnar) if n_mnar > 0 else 0.0

    # Confusion matrix: Ground Truth vs Chosen Decision
    conf_matrix = pd.crosstab(df["mechanism"], df["decision"], margins=True, normalize="index")

    # Sensitivity to sample size
    by_n = df.groupby(["n_samples", "mechanism"])["is_correct"].mean().unstack()

    # Sensitivity to missingness rate
    by_rate = df.groupby(["missing_rate", "mechanism"])["is_correct"].mean().unstack()

    return RouterEvaluationSummary(
        overall_accuracy=overall_acc,
        overall_accuracy_ci=overall_ci,
        mcar_correct_rate=mcar_correct,
        mcar_correct_rate_ci=mcar_ci,
        mar_correct_rate=mar_correct,
        mar_correct_rate_ci=mar_ci,
        mnar_correct_rate=mnar_correct,
        mnar_correct_rate_ci=mnar_ci,
        false_alarm_rate=false_alarm,
        false_alarm_rate_ci=false_alarm_ci,
        missed_risk_rate=missed_risk,
        missed_risk_rate_ci=missed_risk_ci,
        n_total=n_total,
        n_correct=n_correct,
        confusion_matrix=conf_matrix,
        results_by_sample_size=by_n,
        results_by_missing_rate=by_rate,
    )


def compare_router_against_baselines(
    n_replications: int = 10,
    n_samples: int = 1500,
    missing_rate: float = 0.30,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Compare Umbra Auto router against fixed strategies:
    - Always MICE
    - Always Heckman
    - Oracle Strategy
    - Complete-Case
    """
    from benchmarks.simulation_runner import CompleteCaseBaseline, evaluate_imputer_replication
    from umbra.imputers.heckman_selection import HeckmanSelectionImputer
    from umbra.imputers.mar_chained_equations import MARChainedEquationsImputer
    from umbra.imputers.pattern_mixture import PatternMixtureImputer

    regimes = [
        "MCAR",
        "MAR",
        "MNAR_SELECTION",
        "MNAR_PATTERN_MIXTURE",
        "MNAR_WEAK_SIGNAL",
        "MNAR_STRONG_SIGNAL",
        "MNAR_SELF_MASKING",
        "MNAR_TAILS",
    ]

    records = []
    for reg in regimes:
        for rep in range(n_replications):
            seed = base_seed + rep * 101
            sim = generate_simulation_dataset(
                mechanism=reg,
                n_samples=n_samples,
                missing_rate=missing_rate,
                random_state=seed,
            )

            res_cc = evaluate_imputer_replication(lambda: CompleteCaseBaseline(), sim, rep_idx=rep)
            res_mice = evaluate_imputer_replication(
                lambda: MARChainedEquationsImputer(imputation_method="pmm", random_state=seed),
                sim,
                rep_idx=rep,
            )
            res_heck = evaluate_imputer_replication(
                lambda: HeckmanSelectionImputer(
                    shadow_cols={sim.target_col: sim.shadow_col} if sim.shadow_col else None,
                    random_state=seed,
                ),
                sim,
                rep_idx=rep,
            )
            res_auto = evaluate_imputer_replication(
                lambda: UmbraImputer(
                    strategy="auto",
                    shadow_cols={sim.target_col: sim.shadow_col} if sim.shadow_col else None,
                    run_sensitivity=False,
                    random_state=seed,
                ),
                sim,
                rep_idx=rep,
            )

            if reg in ["MCAR", "MAR"]:
                oracle_fn = lambda: MARChainedEquationsImputer(  # noqa: E731
                    imputation_method="pmm", random_state=seed
                )
            elif reg == "MNAR_SELECTION":
                oracle_fn = lambda: HeckmanSelectionImputer(  # noqa: E731
                    shadow_cols={sim.target_col: sim.shadow_col} if sim.shadow_col else None,
                    random_state=seed,
                )
            elif reg == "MNAR_PATTERN_MIXTURE":
                true_delta = sim.true_params.get("delta", -0.8)
                oracle_fn = lambda: PatternMixtureImputer(  # noqa: E731
                    delta=true_delta, random_state=seed
                )
            elif reg in ["MNAR_SELF_MASKING", "MNAR_STRONG_SIGNAL"]:
                # Under self-masking, the selection depends directly on unobserved Y
                bias = sim.true_params.get("selection_bias", 0.0)
                sigma = max(0.1, sim.true_params.get("sigma_eps", 1.0))
                calibrated_delta = -bias / sigma
                oracle_fn = lambda: PatternMixtureImputer(  # noqa: E731
                    delta=calibrated_delta, random_state=seed
                )
            else:
                oracle_fn = lambda: MARChainedEquationsImputer(  # noqa: E731
                    imputation_method="pmm", random_state=seed
                )

            res_oracle = evaluate_imputer_replication(oracle_fn, sim, rep_idx=rep)

            for strat_name, r in [
                ("Complete-Case", res_cc),
                ("Always MICE", res_mice),
                ("Always Heckman", res_heck),
                ("Umbra Auto", res_auto),
                ("Oracle Route", res_oracle),
            ]:
                records.append(
                    {
                        "Regime": reg,
                        "Strategy": strat_name,
                        "Beta Total Error": abs(r.beta_age - 0.5) + abs(r.beta_edu - 0.8),
                        "Cell RMSE": r.cell_rmse,
                        "Mean Bias": r.mean_bias,
                        "Coverage 95": float(r.ci_95_covered),
                    }
                )

    df = pd.DataFrame(records)
    summary = (
        df.groupby(["Regime", "Strategy"])
        .agg(
            {
                "Beta Total Error": "mean",
                "Cell RMSE": "mean",
                "Mean Bias": "mean",
                "Coverage 95": "mean",
            }
        )
        .reset_index()
    )
    return summary


def investigate_missed_risk_cases(
    n_replications: int = 10,
    base_seed: int = 100,
) -> pd.DataFrame:
    """Systematically investigate subtle MNAR cases where observable signals
    are indistinguishable from MAR in finite samples.

    Evaluates:
    - Subtle MNAR departures (MNAR_WEAK_SIGNAL, small N=250, missing_rate=0.10)
    - Observable diagnostics: Little's p-value, KS max stat, tail ratio
    - Demonstrates why these cases are observationally consistent with MAR in finite samples.
    """
    records = []
    cases = [
        ("MNAR_WEAK_SIGNAL", 250, 0.10, "weak"),
        ("MNAR_WEAK_SIGNAL", 500, 0.20, "weak"),
        ("MNAR_SELF_MASKING", 250, 0.10, "low"),
        ("MNAR_PATTERN_MIXTURE", 250, 0.10, "low"),
    ]

    for mech, n, rate, sev in cases:
        for rep in range(n_replications):
            seed = base_seed + rep * 17
            sim = generate_simulation_dataset(
                mechanism=mech,
                n_samples=n,
                missing_rate=rate,
                severity=sev,
                random_state=seed,
            )

            imputer = UmbraImputer(
                strategy="auto",
                shadow_cols={sim.target_col: sim.shadow_col} if sim.shadow_col else None,
                run_sensitivity=False,
                random_state=seed,
            )
            imputer.fit(sim.data_observed)
            rep_diag = imputer.diagnostics_.get(sim.target_col)
            score = rep_diag.composite_score if rep_diag else 0.0
            level = rep_diag.risk_level if rep_diag else "LOW"
            decision = imputer.routing_decisions_.get(sim.target_col, "mar_chained_equations")

            # Check if this case was missed (i.e. classified as LOW risk)
            is_missed = level == "LOW"

            records.append(
                {
                    "DGP Mechanism": mech,
                    "Sample Size N": n,
                    "Missing Rate": rate,
                    "Severity": sev,
                    "Concern Score": score,
                    "Assigned Risk": level,
                    "Routing Decision": decision,
                    "Missed Risk Flag": is_missed,
                    "Selection Bias": sim.true_params.get("selection_bias", 0.0),
                }
            )

    return pd.DataFrame(records)


def evaluate_threshold_sensitivity(
    thresholds: Optional[List[float]] = None,
    n_replications_per_cell: int = 4,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Evaluate router performance across a grid of decision thresholds."""
    if thresholds is None:
        thresholds = [0.20, 0.35, 0.50, 0.65, 0.80]

    mechanisms = ["MCAR", "MAR", "MNAR_SELF_MASKING", "MNAR_SELECTION", "MNAR_TAILS"]
    records = []

    for t in thresholds:
        n_mcar_mar = 0
        n_fa = 0
        n_mnar = 0
        n_mr = 0

        for mech in mechanisms:
            is_mnar = mech.startswith("MNAR")
            for rep in range(n_replications_per_cell):
                seed = base_seed + rep * 31
                sim = generate_simulation_dataset(
                    mechanism=mech,
                    n_samples=1000,
                    missing_rate=0.30,
                    random_state=seed,
                )
                imputer = UmbraImputer(strategy="auto", random_state=seed, run_sensitivity=False)
                imputer.fit(sim.data_observed)
                rep_diag = imputer.diagnostics_.get(sim.target_col)
                score = rep_diag.composite_score if rep_diag else 0.0

                # Simulated routing with threshold t:
                # If score >= t -> MNAR escalated; else -> MAR
                escalated = score >= t

                if not is_mnar:
                    n_mcar_mar += 1
                    if escalated:
                        n_fa += 1
                else:
                    n_mnar += 1
                    if not escalated:
                        n_mr += 1

        fa_rate = n_fa / n_mcar_mar if n_mcar_mar > 0 else 0.0
        mr_rate = n_mr / n_mnar if n_mnar > 0 else 0.0
        records.append(
            {
                "Threshold": t,
                "False Alarm Rate": fa_rate,
                "Missed Risk Rate": mr_rate,
                "F1 Score (Balanced)": 2
                * (1 - fa_rate)
                * (1 - mr_rate)
                / max(0.01, (2 - fa_rate - mr_rate)),
            }
        )

    return pd.DataFrame(records)


def evaluate_held_out_validation(
    n_replications_per_cell: int = 5,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """Validate threshold calibration on unseen simulation regimes to prevent overfitting."""
    tuning_regimes = ["MCAR", "MAR", "MNAR_SELF_MASKING", "MNAR_SELECTION"]
    held_out_regimes = ["MNAR_PATTERN_MIXTURE", "MNAR_WEAK_SIGNAL", "MNAR_TAILS"]

    def evaluate_regimes(
        reg_list: List[str], seed_offset: int
    ) -> Tuple[float, Tuple[float, float], int]:
        correct = 0
        total = 0
        for reg in reg_list:
            for rep in range(n_replications_per_cell):
                seed = base_seed + seed_offset + rep * 47
                res = evaluate_single_routing(
                    mechanism=reg,
                    n_samples=1500,
                    missing_rate=0.30,
                    random_state=seed,
                )
                total += 1
                if res["is_correct"]:
                    correct += 1
        acc = correct / total if total > 0 else 0.0
        ci = wilson_score_interval(correct, total)
        return acc, ci, total

    tune_acc, tune_ci, tune_n = evaluate_regimes(tuning_regimes, 0)
    test_acc, test_ci, test_n = evaluate_regimes(held_out_regimes, 1000)

    return {
        "tuning_regimes": tuning_regimes,
        "tuning_accuracy": tune_acc,
        "tuning_ci": tune_ci,
        "tuning_n": tune_n,
        "held_out_regimes": held_out_regimes,
        "held_out_accuracy": test_acc,
        "held_out_ci": test_ci,
        "held_out_n": test_n,
    }


if __name__ == "__main__":
    summary = benchmark_auto_router(
        n_replications_per_cell=3,
        sample_sizes=[500, 1000, 2500],
        missing_rates=[0.20, 0.30, 0.40],
    )
    print("\n--- AUTO ROUTER BENCHMARK RESULTS ---")
    print(
        f"Overall Accuracy    : {summary.overall_accuracy:.1%} (95% CI: [{summary.overall_accuracy_ci[0]:.1%}, {summary.overall_accuracy_ci[1]:.1%}])"
    )
    print(
        f"MCAR Correct Rate   : {summary.mcar_correct_rate:.1%} (95% CI: [{summary.mcar_correct_rate_ci[0]:.1%}, {summary.mcar_correct_rate_ci[1]:.1%}])"
    )
    print(
        f"MAR Correct Rate    : {summary.mar_correct_rate:.1%} (95% CI: [{summary.mar_correct_rate_ci[0]:.1%}, {summary.mar_correct_rate_ci[1]:.1%}])"
    )
    print(
        f"MNAR Correct Rate   : {summary.mnar_correct_rate:.1%} (95% CI: [{summary.mnar_correct_rate_ci[0]:.1%}, {summary.mnar_correct_rate_ci[1]:.1%}])"
    )
    print(
        f"False Alarm Rate    : {summary.false_alarm_rate:.1%} (95% CI: [{summary.false_alarm_rate_ci[0]:.1%}, {summary.false_alarm_rate_ci[1]:.1%}])"
    )
    print(
        f"Missed Risk Rate    : {summary.missed_risk_rate:.1%} (95% CI: [{summary.missed_risk_rate_ci[0]:.1%}, {summary.missed_risk_rate_ci[1]:.1%}])"
    )
    print("\nConfusion Matrix:")
    print(summary.confusion_matrix.to_string())

    print("\nEvaluating Threshold Sensitivity:")
    sens_df = evaluate_threshold_sensitivity(n_replications_per_cell=3)
    print(sens_df.to_string(index=False))

    print("\nEvaluating Held-Out Validation Regimes:")
    val_res = evaluate_held_out_validation(n_replications_per_cell=4)
    print(
        f"Tuning Set Accuracy   : {val_res['tuning_accuracy']:.1%} (95% CI: [{val_res['tuning_ci'][0]:.1%}, {val_res['tuning_ci'][1]:.1%}])"
    )
    print(
        f"Held-Out Set Accuracy : {val_res['held_out_accuracy']:.1%} (95% CI: [{val_res['held_out_ci'][0]:.1%}, {val_res['held_out_ci'][1]:.1%}])"
    )

    print("\nComparing Auto Router against Fixed Strategies...")
    cmp_df = compare_router_against_baselines(n_replications=3, n_samples=1000)
    print(cmp_df.to_string(index=False))
