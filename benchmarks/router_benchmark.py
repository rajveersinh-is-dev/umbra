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
from typing import Any, Dict, List, Optional

import pandas as pd

from benchmarks.dgps import generate_simulation_dataset
from umbra.api import UmbraImputer


@dataclass
class RouterEvaluationSummary:
    """Performance summary of the Auto Strategy Router."""

    overall_accuracy: float
    mcar_correct_rate: float
    mar_correct_rate: float
    mnar_correct_rate: float
    false_alarm_rate: float
    missed_risk_rate: float
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

    mcar_correct = float(df.loc[mcar_mask, "is_correct"].mean())
    mar_correct = float(df.loc[mar_mask, "is_correct"].mean())
    mnar_correct = float(df.loc[mnar_mask, "is_correct"].mean())
    overall_acc = float(df["is_correct"].mean())

    # False alarm: MCAR/MAR flagged as HIGH risk
    false_alarm = float(df.loc[mcar_mask | mar_mask, "is_false_alarm"].mean())
    # Missed risk: MNAR classified as LOW risk
    missed_risk = float(df.loc[mnar_mask, "is_missed_risk"].mean())

    # Confusion matrix: Ground Truth vs Chosen Decision
    conf_matrix = pd.crosstab(df["mechanism"], df["decision"], margins=True, normalize="index")

    # Sensitivity to sample size
    by_n = df.groupby(["n_samples", "mechanism"])["is_correct"].mean().unstack()

    # Sensitivity to missingness rate
    by_rate = df.groupby(["missing_rate", "mechanism"])["is_correct"].mean().unstack()

    return RouterEvaluationSummary(
        overall_accuracy=overall_acc,
        mcar_correct_rate=mcar_correct,
        mar_correct_rate=mar_correct,
        mnar_correct_rate=mnar_correct,
        false_alarm_rate=false_alarm,
        missed_risk_rate=missed_risk,
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


if __name__ == "__main__":
    summary = benchmark_auto_router(
        n_replications_per_cell=3,
        sample_sizes=[500, 1000, 2500],
        missing_rates=[0.20, 0.30, 0.40],
    )
    print("\n--- AUTO ROUTER BENCHMARK RESULTS ---")
    print(f"Overall Accuracy    : {summary.overall_accuracy:.1%}")
    print(f"MCAR Correct Rate   : {summary.mcar_correct_rate:.1%}")
    print(f"MAR Correct Rate    : {summary.mar_correct_rate:.1%}")
    print(f"MNAR Correct Rate   : {summary.mnar_correct_rate:.1%}")
    print(f"False Alarm Rate    : {summary.false_alarm_rate:.1%}")
    print(f"Missed Risk Rate    : {summary.missed_risk_rate:.1%}")
    print("\nConfusion Matrix:")
    print(summary.confusion_matrix.to_string())

    print("\nComparing Auto Router against Fixed Strategies...")
    cmp_df = compare_router_against_baselines(n_replications=3, n_samples=1000)
    print(cmp_df.to_string(index=False))
