"""
Independent Reproduction CLI for Umbra Benchmark Suite.

Usage:
  python scripts/reproduce_benchmarks.py --quick
  python scripts/reproduce_benchmarks.py --full
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from benchmarks.run_all import format_markdown_leaderboard, run_full_benchmark_suite  # noqa: E402
from umbra import __version__ as umbra_ver  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=f"Reproduce Umbra v{umbra_ver} Benchmarks & Leaderboard"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fast reproduction mode (N=1,000, R=2 replications per regime, ~60s)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full research-grade benchmark battery (N=2,500, R=20 replications per regime)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output markdown file path (defaults to benchmarks/results.md)",
    )

    args = parser.parse_args()

    if not args.quick and not args.full:
        print("Defaulting to --quick mode. Use --full for the complete R=20 Monte Carlo battery.")
        quick_mode = True
    else:
        quick_mode = args.quick

    if quick_mode:
        n_samples = 500
        n_replications = 2
        print(f"--> Running QUICK verification: N={n_samples}, R={n_replications} per regime...")
    else:
        n_samples = 2500
        n_replications = 20
        print(f"--> Running FULL research suite: N={n_samples}, R={n_replications} per regime...")

    t0 = time.perf_counter()
    summaries, router_summary, router_vs_baselines, scaling_n, scaling_p = run_full_benchmark_suite(
        n_replications=n_replications,
        n_samples=n_samples,
        missing_rate=0.30,
        base_seed=args.seed,
        quick=quick_mode,
    )
    elapsed = time.perf_counter() - t0

    out_file = (
        Path(args.output)
        if args.output
        else root_dir / "benchmarks" / ("quick_results.md" if quick_mode else "results.md")
    )
    format_markdown_leaderboard(
        summaries, router_summary, router_vs_baselines, scaling_n, scaling_p, out_file
    )

    print("\n" + "=" * 60)
    print(f"Reproduction complete in {elapsed:.1f}s.")
    print(f"Results successfully saved to: {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
