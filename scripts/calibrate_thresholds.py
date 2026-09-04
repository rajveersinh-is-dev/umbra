"""
Threshold Calibration and Pareto Analysis for Umbra MNAR Risk Scoring.

Sweeps the composite risk score threshold over tau in [0.10, 0.90] (step 0.05)
and analyzes the tradeoff between:
  - False Alarm Rate: MCAR/MAR classified as high-risk MNAR
  - Missed Risk Rate: MNAR classified as low-risk / standard MICE

Generates:
  - benchmarks/figures/threshold_calibration.png
  - Summary table of Pareto-optimal operational points

Usage:
  python scripts/calibrate_thresholds.py [--quick]
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.dgps import generate_simulation_dataset
from umbra.diagnostics.mcar_test import littles_mcar_test
from umbra.diagnostics.mnar_risk_score import assess_mnar_risk
from umbra.diagnostics.pattern_analysis import analyze_missingness_patterns
from umbra.diagnostics.shadow_variable_finder import find_shadow_variables


def collect_calibration_cases(
    n_replications: int = 5, n_samples: int = 500
) -> List[Dict[str, Any]]:
    """Generate benchmark datasets and extract precomputed diagnostic components."""
    mechanisms = [
        ("MCAR", False),
        ("MAR", False),
        ("MNAR_SELECTION", True),
        ("MNAR_TAILS", True),
        ("MNAR_PATTERN_MIXTURE", True),
        ("MNAR_SELF_MASKING", True),
        ("MNAR_STRONG_SIGNAL", True),
        ("MNAR_WEAK_SIGNAL", True),
    ]

    cases = []
    for mech, is_mnar in mechanisms:
        for rep in range(n_replications):
            seed = 42 + rep * 100
            sim = generate_simulation_dataset(
                mechanism=mech,
                n_samples=n_samples,
                missing_rate=0.30,
                random_state=seed,
            )
            df = sim.data_observed
            col = sim.target_col

            # Run diagnostics once
            numeric_df = df.select_dtypes(include=[np.number])
            littles_res = (
                littles_mcar_test(numeric_df, alpha=0.05) if numeric_df.shape[1] > 1 else None
            )
            pattern_rep = analyze_missingness_patterns(df, alpha=0.05)
            shadow_rep = find_shadow_variables(df, col)

            cases.append(
                {
                    "mechanism": mech,
                    "is_mnar": is_mnar,
                    "df": df,
                    "col": col,
                    "littles_res": littles_res,
                    "pattern_rep": pattern_rep,
                    "shadow_rep": shadow_rep,
                }
            )

    return cases


def evaluate_threshold(
    cases: List[Dict[str, Any]],
    high_threshold: float,
    med_threshold: float,
    tail_threshold: float = 0.25,
) -> Dict[str, float]:
    """Evaluate routing accuracy, false alarm rate, and missed risk rate for a threshold."""
    n_mcar_mar = 0
    n_false_alarms = 0
    n_mnar = 0
    n_missed_risks = 0
    n_correct = 0

    for case in cases:
        report = assess_mnar_risk(
            data=case["df"],
            target_col=case["col"],
            littles_result=case["littles_res"],
            pattern_report=case["pattern_rep"],
            shadow_report=case["shadow_rep"],
            high_risk_composite_threshold=high_threshold,
            medium_risk_composite_threshold=med_threshold,
            tail_risk_threshold=tail_threshold,
        )

        risk = report.risk_level
        strat = report.recommended_strategy
        is_mnar = case["is_mnar"]

        if not is_mnar:
            n_mcar_mar += 1
            if risk == "HIGH" or strat != "mar_chained_equations":
                n_false_alarms += 1
            else:
                n_correct += 1
        else:
            n_mnar += 1
            if risk == "LOW" or strat == "mar_chained_equations":
                n_missed_risks += 1
            else:
                n_correct += 1

    far = n_false_alarms / n_mcar_mar if n_mcar_mar > 0 else 0.0
    mrr = n_missed_risks / n_mnar if n_mnar > 0 else 0.0
    acc = n_correct / len(cases) if cases else 0.0

    return {
        "threshold": high_threshold,
        "false_alarm_rate": far,
        "missed_risk_rate": mrr,
        "accuracy": acc,
        "pareto_dist": float(np.sqrt(far**2 + mrr**2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate MNAR risk score thresholds.")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick calibration with fewer samples"
    )
    args = parser.parse_args()

    n_reps = 2 if args.quick else 5
    n_samples = 300 if args.quick else 800

    print(f"Collecting calibration cases (R={n_reps}, N={n_samples} per regime)...")
    cases = collect_calibration_cases(n_replications=n_reps, n_samples=n_samples)
    print(f"Generated {len(cases)} benchmark datasets.")

    threshold_grid = np.round(np.arange(0.10, 0.91, 0.05), 2)
    results = []

    for tau in threshold_grid:
        med_tau = round(tau / 2.0, 2)
        res = evaluate_threshold(cases, high_threshold=float(tau), med_threshold=float(med_tau))
        results.append(res)

    df_res = pd.DataFrame(results)

    # Find Pareto-optimal point minimizing Euclidean distance to ideal (FAR=0, MRR=0)
    best_idx = df_res["pareto_dist"].idxmin()
    best_row = df_res.loc[best_idx]

    print("\n" + "=" * 70)
    print("THRESHOLD CALIBRATION RESULTS")
    print("=" * 70)
    print(
        df_res.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "false_alarm_rate": "{:.1%}".format,
                "missed_risk_rate": "{:.1%}".format,
                "accuracy": "{:.1%}".format,
                "pareto_dist": "{:.4f}".format,
            },
        )
    )
    print("=" * 70)
    print(
        f"Pareto-optimal threshold (min Euclidean dist to (0,0)): tau = {best_row['threshold']:.2f}"
    )
    print(
        f"  False Alarm Rate: {best_row['false_alarm_rate']:.1%}, Missed Risk Rate: {best_row['missed_risk_rate']:.1%}, Accuracy: {best_row['accuracy']:.1%}"
    )
    default_row = df_res[df_res["threshold"] == 0.50].iloc[0]
    print("Default threshold tau = 0.50:")
    print(
        f"  False Alarm Rate: {default_row['false_alarm_rate']:.1%}, Missed Risk Rate: {default_row['missed_risk_rate']:.1%}, Accuracy: {default_row['accuracy']:.1%}"
    )

    # Plotting
    out_dir = Path("benchmarks/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "threshold_calibration.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Plot 1: Tradeoff Curve (Missed Risk Rate vs False Alarm Rate)
    ax1.plot(
        df_res["false_alarm_rate"],
        df_res["missed_risk_rate"],
        "o-",
        color="#1f77b4",
        linewidth=2,
        label="Threshold Sweep",
    )
    ax1.scatter(
        [best_row["false_alarm_rate"]],
        [best_row["missed_risk_rate"]],
        color="#2ca02c",
        s=120,
        zorder=5,
        label=f"Pareto-Optimal (tau={best_row['threshold']:.2f})",
    )
    ax1.scatter(
        [default_row["false_alarm_rate"]],
        [default_row["missed_risk_rate"]],
        color="#d62728",
        s=100,
        marker="s",
        zorder=5,
        label="Default (tau=0.50)",
    )

    for _, row in df_res.iterrows():
        if row["threshold"] in [0.20, 0.35, 0.50, 0.65, 0.80]:
            ax1.annotate(
                f"{row['threshold']:.2f}",
                (row["false_alarm_rate"], row["missed_risk_rate"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )

    ax1.set_xlabel("False Alarm Rate (MCAR/MAR flagged as MNAR)", fontsize=10)
    ax1.set_ylabel("Missed Risk Rate (MNAR classified as MAR/MCAR)", fontsize=10)
    ax1.set_title("MNAR Router Operating Characteristic Curve", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    # Plot 2: Error Rates vs Threshold tau
    ax2.plot(
        df_res["threshold"],
        df_res["false_alarm_rate"],
        "r--",
        label="False Alarm Rate",
        linewidth=2,
    )
    ax2.plot(
        df_res["threshold"],
        df_res["missed_risk_rate"],
        "b-.",
        label="Missed Risk Rate",
        linewidth=2,
    )
    ax2.plot(
        df_res["threshold"],
        df_res["accuracy"],
        "g-",
        label="Overall Routing Accuracy",
        linewidth=2,
    )
    ax2.axvline(
        x=best_row["threshold"],
        color="#2ca02c",
        linestyle=":",
        label=f"Pareto tau={best_row['threshold']:.2f}",
    )
    ax2.axvline(x=0.50, color="#d62728", linestyle=":", label="Default tau=0.50")

    ax2.set_xlabel("High-Risk Threshold tau", fontsize=10)
    ax2.set_ylabel("Rate", fontsize=10)
    ax2.set_title("Router Error Rates across Threshold Sweep", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="center right", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nCalibration plot saved to: {out_path}")


if __name__ == "__main__":
    main()
