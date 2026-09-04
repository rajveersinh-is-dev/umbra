"""
Unified Master Reproduction Entry Point for Umbra Empirical Evidence.

CLI Usage:
  python -m benchmarks.reproduce_all --quick   # Fast verification (~30s)
  python -m benchmarks.reproduce_all --full    # Full publication battery (~3m)
"""

import argparse
import sys
import time
from pathlib import Path

from benchmarks.misspecification_benchmark import run_misspecification_battery
from benchmarks.router_benchmark import (
    evaluate_held_out_validation,
    evaluate_threshold_sensitivity,
    investigate_missed_risk_cases,
)
from benchmarks.run_all import format_markdown_leaderboard, run_full_benchmark_suite
from scripts.generate_figures import generate_all_figures
from umbra import __version__ as umbra_ver


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce all Umbra benchmarks, leaderboards, and figures."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fast smoke-test reproduction (fewer replications, smaller N).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run complete, publication-grade Monte Carlo battery.",
    )
    args = parser.parse_args()

    # Default to quick if neither or quick specified
    is_quick = args.quick or (not args.full)

    t0_start = time.perf_counter()
    benchmarks_dir = Path(__file__).resolve().parent
    results_md_path = benchmarks_dir / "results.md"

    print("=" * 75)
    print(
        f"UMBRA REPRODUCIBILITY SUITE — {'QUICK MODE' if is_quick else 'FULL PUBLICATION BATTERY'}"
    )
    print(f"Version: Umbra v{umbra_ver} | Python: {sys.version.split()[0]}")
    print("=" * 75)

    if is_quick:
        n_reps = 3
        n_samples = 800
        router_reps = 2
        misspec_reps = 3
    else:
        n_reps = 20
        n_samples = 2500
        router_reps = 5
        misspec_reps = 20

    # 1. Master Benchmark Suite
    print("\n[Step 1/5] Running Master Monte Carlo Benchmark Suite...")
    summaries, router_summary, router_vs_baselines, scaling_n, scaling_p = run_full_benchmark_suite(
        n_replications=n_reps,
        n_samples=n_samples,
        missing_rate=0.30,
        base_seed=42,
        quick=is_quick,
    )

    # 2. Write Leaderboard
    print("\n[Step 2/5] Formatting living leaderboard (results.md)...")
    format_markdown_leaderboard(
        summaries, router_summary, router_vs_baselines, scaling_n, scaling_p, results_md_path
    )

    # 3. Router In-Depth Diagnostics
    print("\n[Step 3/5] Running Router Uncertainty & Threshold Sensitivity...")
    sens_df = evaluate_threshold_sensitivity(n_replications_per_cell=router_reps, base_seed=42)
    held_out_res = evaluate_held_out_validation(n_replications_per_cell=router_reps, base_seed=42)
    missed_df = investigate_missed_risk_cases(n_replications=router_reps, base_seed=100)

    print(
        f"  * Router Overall Accuracy: {router_summary.overall_accuracy:.1%} "
        f"(95% CI: [{router_summary.overall_accuracy_ci[0]:.1%}, {router_summary.overall_accuracy_ci[1]:.1%}])"
    )
    print(
        f"  * Held-Out Generalization: {held_out_res['held_out_accuracy']:.1%} "
        f"(95% CI: [{held_out_res['held_out_ci'][0]:.1%}, {held_out_res['held_out_ci'][1]:.1%}])"
    )
    print(f"  * Investigated {len(missed_df)} subtle MNAR edge cases across N in [250, 500].")
    print(f"  * Threshold sensitivity evaluated over {len(sens_df)} decision cutoffs.")

    # 4. Misspecification Battery
    print("\n[Step 4/5] Executing Model Misspecification Stress Battery...")
    df_misspec = run_misspecification_battery(
        n_replications=misspec_reps,
        n_samples=1500 if is_quick else 2000,
        base_seed=42,
    )
    print(f"  * Misspecification battery completed ({len(df_misspec)} condition records).")

    # 5. Publication Figures
    print("\n[Step 5/5] Regenerating Publication-Quality Figures...")
    generate_all_figures()

    elapsed = time.perf_counter() - t0_start
    print("\n" + "=" * 75)
    print(f"REPRODUCTION COMPLETE in {elapsed:.1f} seconds.")
    print("Artifacts Updated:")
    print(f"  - Leaderboard       : {results_md_path}")
    print(f"  - Misspecification  : {benchmarks_dir / 'misspecification_results.md'}")
    print(f"  - Figures Directory : {benchmarks_dir / 'figures'}")
    print("=" * 75)


if __name__ == "__main__":
    main()
