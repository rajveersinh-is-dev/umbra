"""
Publication-Quality Benchmark Figures Generator for Umbra.

Generates 7 publication-ready figures in both 300 DPI PNG and vector PDF formats:
1. bias_vs_missingness_rate: Parameter bias vs missingness rate (10%-50%)
2. rmse_vs_missingness_rate: Cell RMSE vs missingness rate across methods
3. coverage_probability_vs_sample_size: 95% CI empirical coverage vs N (illustrating MICE coverage collapse vs Umbra stability)
4. router_confusion_matrix: Mechanism vs selected strategy confusion matrix
5. error_by_mnar_mechanism_strength: Bias and RMSE as a function of MNAR selection parameter gamma
6. sensitivity_tipping_point_curve: Pattern mixture delta-adjustment sensitivity curve with tipping point
7. runtime_scaling_n_p: Runtime scaling vs sample size N and dimensionality p

Usage:
  python scripts/generate_figures.py [--output-dir benchmarks/figures]
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.dgps import generate_simulation_dataset
from benchmarks.performance_scaling import benchmark_runtime_vs_dimension, benchmark_runtime_vs_n
from benchmarks.router_benchmark import benchmark_auto_router
from umbra.api import UmbraImputer
from umbra.imputers.heckman_selection import HeckmanSelectionImputer
from umbra.imputers.mar_chained_equations import MARChainedEquationsImputer
from umbra.sensitivity.grid_analysis import run_sensitivity_grid

# Publication aesthetic styling
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.color": "#cccccc",
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.labelweight": "medium",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
    }
)

# Colorblind-safe palette
COLORS = {
    "Umbra (Auto)": "#1f77b4",  # Navy blue
    "Heckman Selection": "#2ca02c",  # Forest green
    "MAR MICE (PMM)": "#ff7f0e",  # Deep orange
    "MAR MICE (Ridge)": "#9467bd",  # Purple
    "Pattern Mixture": "#8c564b",  # Brown
    "Unadjusted (Observed)": "#d62728",  # Crimson red
}


def ensure_output_dir(output_dir: str) -> Path:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def save_plot(fig: plt.Figure, base_name: str, output_dir: Path) -> None:
    png_path = output_dir / f"{base_name}.png"
    pdf_path = output_dir / f"{base_name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path.name} & {pdf_path.name}")


def plot_bias_vs_missingness(output_dir: Path) -> None:
    """Figure 1: Mean parameter bias across missingness rates (10% to 50%)."""
    print("[1/7] Generating bias_vs_missingness_rate...")
    missing_rates = [0.10, 0.20, 0.30, 0.40, 0.50]
    n_reps = 5
    n_samples = 1500

    results = {
        "Unadjusted (Observed)": [],
        "MAR MICE (PMM)": [],
        "MAR MICE (Ridge)": [],
        "Heckman Selection": [],
        "Umbra (Auto)": [],
    }

    for mr in missing_rates:
        cell_biases = {k: [] for k in results.keys()}
        for rep in range(n_reps):
            seed = 42 + rep * 101
            dgp = generate_simulation_dataset(
                mechanism="MNAR_SELECTION", n_samples=n_samples, missing_rate=mr, random_state=seed
            )
            df_obs = dgp.data_observed.copy()
            y_true_mean = float(dgp.data_complete["target"].mean())

            # Unadjusted observed mean
            cell_biases["Unadjusted (Observed)"].append(
                float(df_obs["target"].mean()) - y_true_mean
            )

            # MICE PMM
            imp_pmm = MARChainedEquationsImputer(imputation_method="pmm", random_state=seed)
            df_pmm = imp_pmm.fit_transform(df_obs)
            cell_biases["MAR MICE (PMM)"].append(float(df_pmm["target"].mean()) - y_true_mean)

            # MICE Ridge
            imp_ridge = MARChainedEquationsImputer(imputation_method="ridge", random_state=seed)
            df_ridge = imp_ridge.fit_transform(df_obs)
            cell_biases["MAR MICE (Ridge)"].append(float(df_ridge["target"].mean()) - y_true_mean)

            # Heckman
            imp_heck = HeckmanSelectionImputer(
                shadow_cols={"target": "shadow_z"}, random_state=seed
            )
            df_heck = imp_heck.fit_transform(df_obs)
            cell_biases["Heckman Selection"].append(float(df_heck["target"].mean()) - y_true_mean)

            # Umbra Auto
            imp_auto = UmbraImputer(
                strategy="auto",
                shadow_cols={"target": "shadow_z"},
                run_sensitivity=False,
                random_state=seed,
            )
            df_auto = imp_auto.fit_transform(df_obs)
            cell_biases["Umbra (Auto)"].append(float(df_auto["target"].mean()) - y_true_mean)

        for k in results:
            results[k].append(float(np.mean(cell_biases[k])))

    rates_pct = [int(mr * 100) for mr in missing_rates]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for name, biases in results.items():
        style = "-o" if "Umbra" in name or "Heckman" in name else "--s"
        lw = 2.2 if "Umbra" in name else 1.6
        ms = 6 if "Umbra" in name else 5
        ax.plot(
            rates_pct, biases, style, label=name, color=COLORS[name], linewidth=lw, markersize=ms
        )

    ax.axhline(0.0, color="#555555", linestyle=":", linewidth=1.2, label="Zero Bias Reference")
    ax.set_xlabel("Missingness Rate (%)")
    ax.set_ylabel("Parameter Bias: E[Ŷ] - E[Y_true]")
    ax.set_title("Figure 1: Mean Parameter Bias vs Missingness Rate (MNAR Selection DGP)")
    ax.set_xticks(rates_pct)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left")

    save_plot(fig, "bias_vs_missingness_rate", output_dir)


def plot_rmse_vs_missingness(output_dir: Path) -> None:
    """Figure 2: Cell-level RMSE across missingness rates (10% to 50%)."""
    print("[2/7] Generating rmse_vs_missingness_rate...")
    missing_rates = [0.10, 0.20, 0.30, 0.40, 0.50]
    n_reps = 5
    n_samples = 1500

    results = {
        "MAR MICE (PMM)": [],
        "MAR MICE (Ridge)": [],
        "Heckman Selection": [],
        "Umbra (Auto)": [],
    }

    for mr in missing_rates:
        cell_rmses = {k: [] for k in results.keys()}
        for rep in range(n_reps):
            seed = 42 + rep * 101
            dgp = generate_simulation_dataset(
                mechanism="MNAR_SELECTION", n_samples=n_samples, missing_rate=mr, random_state=seed
            )
            df_obs = dgp.data_observed.copy()
            mask = df_obs["target"].isna()
            y_true_mis = dgp.data_complete.loc[mask, "target"].to_numpy()

            # MICE PMM
            df_pmm = MARChainedEquationsImputer(
                imputation_method="pmm", random_state=seed
            ).fit_transform(df_obs)
            cell_rmses["MAR MICE (PMM)"].append(
                float(np.sqrt(np.mean((df_pmm.loc[mask, "target"].to_numpy() - y_true_mis) ** 2)))
            )

            # MICE Ridge
            df_ridge = MARChainedEquationsImputer(
                imputation_method="ridge", random_state=seed
            ).fit_transform(df_obs)
            cell_rmses["MAR MICE (Ridge)"].append(
                float(np.sqrt(np.mean((df_ridge.loc[mask, "target"].to_numpy() - y_true_mis) ** 2)))
            )

            # Heckman
            df_heck = HeckmanSelectionImputer(
                shadow_cols={"target": "shadow_z"}, random_state=seed
            ).fit_transform(df_obs)
            cell_rmses["Heckman Selection"].append(
                float(np.sqrt(np.mean((df_heck.loc[mask, "target"].to_numpy() - y_true_mis) ** 2)))
            )

            # Umbra Auto
            df_auto = UmbraImputer(
                strategy="auto",
                shadow_cols={"target": "shadow_z"},
                run_sensitivity=False,
                random_state=seed,
            ).fit_transform(df_obs)
            cell_rmses["Umbra (Auto)"].append(
                float(np.sqrt(np.mean((df_auto.loc[mask, "target"].to_numpy() - y_true_mis) ** 2)))
            )

        for k in results:
            results[k].append(float(np.mean(cell_rmses[k])))

    rates_pct = [int(mr * 100) for mr in missing_rates]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for name, rmses in results.items():
        style = "-o" if "Umbra" in name or "Heckman" in name else "--s"
        lw = 2.2 if "Umbra" in name else 1.6
        ms = 6 if "Umbra" in name else 5
        ax.plot(
            rates_pct, rmses, style, label=name, color=COLORS[name], linewidth=lw, markersize=ms
        )

    ax.set_xlabel("Missingness Rate (%)")
    ax.set_ylabel("Cell RMSE (Unobserved Counterfactuals)")
    ax.set_title("Figure 2: Missing Cell RMSE vs Missingness Rate (MNAR Selection DGP)")
    ax.set_xticks(rates_pct)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left")

    save_plot(fig, "rmse_vs_missingness_rate", output_dir)


def plot_coverage_vs_sample_size(output_dir: Path) -> None:
    """Figure 3: Empirical 95% CI coverage vs sample size N (illustrating MICE coverage collapse)."""
    print("[3/7] Generating coverage_probability_vs_sample_size...")
    sample_sizes = [250, 500, 1000, 2500, 5000]
    n_reps = 20

    cov_mice = []
    cov_heckman = []
    cov_auto = []

    for n in sample_sizes:
        covered_mice = 0
        covered_heck = 0
        covered_auto = 0

        for rep in range(n_reps):
            seed = 100 + rep * 37
            dgp = generate_simulation_dataset(
                mechanism="MNAR_SELECTION", n_samples=n, missing_rate=0.30, random_state=seed
            )
            df_obs = dgp.data_observed.copy()
            mu_true = float(dgp.data_complete["target"].mean())

            # MICE
            df_mice = MARChainedEquationsImputer(
                imputation_method="ridge", random_state=seed
            ).fit_transform(df_obs)
            y_m = df_mice["target"].to_numpy()
            se_m = float(np.std(y_m, ddof=1) / np.sqrt(n))
            mean_m = float(np.mean(y_m))
            if (mean_m - 1.96 * se_m) <= mu_true <= (mean_m + 1.96 * se_m):
                covered_mice += 1

            # Heckman
            df_heck = HeckmanSelectionImputer(
                shadow_cols={"target": "shadow_z"}, random_state=seed
            ).fit_transform(df_obs)
            y_h = df_heck["target"].to_numpy()
            se_h = float(np.std(y_h, ddof=1) / np.sqrt(n))
            mean_h = float(np.mean(y_h))
            if (mean_h - 1.96 * se_h) <= mu_true <= (mean_h + 1.96 * se_h):
                covered_heck += 1

            # Auto
            df_auto = UmbraImputer(
                strategy="auto",
                shadow_cols={"target": "shadow_z"},
                run_sensitivity=False,
                random_state=seed,
            ).fit_transform(df_obs)
            y_a = df_auto["target"].to_numpy()
            se_a = float(np.std(y_a, ddof=1) / np.sqrt(n))
            mean_a = float(np.mean(y_a))
            if (mean_a - 1.96 * se_a) <= mu_true <= (mean_a + 1.96 * se_a):
                covered_auto += 1

        cov_mice.append(covered_mice / n_reps)
        cov_heckman.append(covered_heck / n_reps)
        cov_auto.append(covered_auto / n_reps)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(
        sample_sizes,
        cov_auto,
        "-o",
        color=COLORS["Umbra (Auto)"],
        label="Umbra (Auto)",
        linewidth=2.4,
        markersize=7,
    )
    ax.plot(
        sample_sizes,
        cov_heckman,
        "--^",
        color=COLORS["Heckman Selection"],
        label="Heckman Selection",
        linewidth=1.8,
        markersize=6,
    )
    ax.plot(
        sample_sizes,
        cov_mice,
        "--s",
        color=COLORS["MAR MICE (Ridge)"],
        label="MAR MICE (Ridge)",
        linewidth=1.8,
        markersize=6,
    )

    ax.axhline(0.95, color="#e41a1c", linestyle="--", linewidth=1.5, label="Nominal 95% Coverage")
    ax.axhspan(0.90, 1.00, color="#e41a1c", alpha=0.08)

    ax.set_xscale("log")
    ax.set_xticks(sample_sizes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Sample Size N (Log Scale)")
    ax.set_ylabel("Empirical 95% CI Coverage Probability")
    ax.set_title("Figure 3: 95% CI Coverage Probability vs Sample Size N (MNAR Regime)")
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="lower left")

    # Annotate MICE collapse
    ax.annotate(
        "MICE Coverage Collapse\n(Shrinking SE around biased mean)",
        xy=(sample_sizes[-1], cov_mice[-1]),
        xytext=(1200, 0.25),
        arrowprops=dict(facecolor="#555555", shrink=0.05, width=1, headwidth=6),
        fontsize=9,
        backgroundcolor="white",
    )

    save_plot(fig, "coverage_probability_vs_sample_size", output_dir)


def plot_router_confusion_matrix(output_dir: Path) -> None:
    """Figure 4: Router confusion matrix across missingness regimes."""
    print("[4/7] Generating router_confusion_matrix...")
    router_summary = benchmark_auto_router(
        n_replications_per_cell=5,
        sample_sizes=[500, 1000, 2500],
        missing_rates=[0.20, 0.30, 0.40],
        base_seed=42,
    )
    cm = router_summary.confusion_matrix

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    regimes = list(cm.index)
    actions = list(cm.columns)

    # Normalize rows to show proportions
    matrix_vals = cm.to_numpy()
    row_sums = matrix_vals.sum(axis=1, keepdims=True)
    matrix_props = np.divide(
        matrix_vals, row_sums, out=np.zeros_like(matrix_vals, dtype=float), where=row_sums != 0
    )

    im = ax.imshow(matrix_props, cmap="Blues", vmin=0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Routing Proportion (Row Normalized)", rotation=270, labelpad=15)

    ax.set_xticks(np.arange(len(actions)))
    ax.set_yticks(np.arange(len(regimes)))
    ax.set_xticklabels([a.replace("_", " ").title() for a in actions], rotation=20, ha="right")
    ax.set_yticklabels([r.replace("_", " ") for r in regimes])

    for i in range(len(regimes)):
        for j in range(len(actions)):
            count = matrix_vals[i, j]
            prop = matrix_props[i, j]
            color = "white" if prop > 0.5 else "black"
            text = f"{count}\n({prop:.0%})" if count > 0 else "0\n(0%)"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=color,
                fontsize=9.5,
                weight="bold" if prop > 0.5 else "normal",
            )

    ax.set_title("Figure 4: Umbra Auto-Router Strategy Selection Confusion Matrix")
    ax.set_xlabel("Selected Imputation Strategy")
    ax.set_ylabel("True Generative Mechanism")
    fig.tight_layout()

    save_plot(fig, "router_confusion_matrix", output_dir)


def plot_error_by_mechanism_strength(output_dir: Path) -> None:
    """Figure 5: Parameter bias and RMSE vs MNAR selection parameter gamma."""
    print("[5/7] Generating error_by_mnar_mechanism_strength...")
    gammas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    n_reps = 5
    n = 1500

    biases_mice = []
    biases_heckman = []
    biases_auto = []
    rmses_mice = []
    rmses_heckman = []
    rmses_auto = []

    for g in gammas:
        b_m, b_h, b_a = [], [], []
        r_m, r_h, r_a = [], [], []

        for rep in range(n_reps):
            seed = 42 + rep * 101
            rng = np.random.RandomState(seed)
            x1 = rng.normal(0, 1, size=n)
            shadow_z = rng.normal(0, 1, size=n)
            y_latent = 0.6 * x1 + rng.normal(0, 1, size=n)

            # Selection equation: R* = 0.5*x1 + 1.0*shadow_z + gamma*y_latent + e
            z_star = 0.5 * x1 + 1.0 * shadow_z + g * y_latent + rng.normal(0, 1, size=n)
            cutoff = np.quantile(z_star, 0.30)
            is_missing = z_star < cutoff

            df_comp = pd.DataFrame({"x1": x1, "shadow_z": shadow_z, "income": y_latent})
            df_obs = df_comp.copy()
            df_obs.loc[is_missing, "income"] = np.nan

            mu_true = float(df_comp["income"].mean())
            y_mis_true = df_comp.loc[is_missing, "income"].to_numpy()

            # MICE
            df_m = MARChainedEquationsImputer(
                imputation_method="ridge", random_state=seed
            ).fit_transform(df_obs)
            b_m.append(float(df_m["income"].mean()) - mu_true)
            r_m.append(
                float(
                    np.sqrt(np.mean((df_m.loc[is_missing, "income"].to_numpy() - y_mis_true) ** 2))
                )
            )

            # Heckman
            df_h = HeckmanSelectionImputer(
                shadow_cols={"income": "shadow_z"}, random_state=seed
            ).fit_transform(df_obs)
            b_h.append(float(df_h["income"].mean()) - mu_true)
            r_h.append(
                float(
                    np.sqrt(np.mean((df_h.loc[is_missing, "income"].to_numpy() - y_mis_true) ** 2))
                )
            )

            # Auto
            df_a = UmbraImputer(
                strategy="auto",
                shadow_cols={"income": "shadow_z"},
                run_sensitivity=False,
                random_state=seed,
            ).fit_transform(df_obs)
            b_a.append(float(df_a["income"].mean()) - mu_true)
            r_a.append(
                float(
                    np.sqrt(np.mean((df_a.loc[is_missing, "income"].to_numpy() - y_mis_true) ** 2))
                )
            )

        biases_mice.append(float(np.mean(b_m)))
        biases_heckman.append(float(np.mean(b_h)))
        biases_auto.append(float(np.mean(b_a)))
        rmses_mice.append(float(np.mean(r_m)))
        rmses_heckman.append(float(np.mean(r_h)))
        rmses_auto.append(float(np.mean(r_a)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # Left: Bias
    ax1.plot(
        gammas, biases_auto, "-o", color=COLORS["Umbra (Auto)"], label="Umbra (Auto)", linewidth=2.2
    )
    ax1.plot(
        gammas,
        biases_heckman,
        "--^",
        color=COLORS["Heckman Selection"],
        label="Heckman Selection",
        linewidth=1.8,
    )
    ax1.plot(
        gammas,
        biases_mice,
        "--s",
        color=COLORS["MAR MICE (Ridge)"],
        label="MAR MICE (Ridge)",
        linewidth=1.8,
    )
    ax1.axhline(0.0, color="#555555", linestyle=":", linewidth=1.2)
    ax1.set_xlabel(r"MNAR Mechanism Strength ($\gamma$)")
    ax1.set_ylabel("Mean Parameter Bias")
    ax1.set_title("A: Bias vs MNAR Mechanism Strength")
    ax1.grid(True)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

    # Right: RMSE
    ax2.plot(
        gammas, rmses_auto, "-o", color=COLORS["Umbra (Auto)"], label="Umbra (Auto)", linewidth=2.2
    )
    ax2.plot(
        gammas,
        rmses_heckman,
        "--^",
        color=COLORS["Heckman Selection"],
        label="Heckman Selection",
        linewidth=1.8,
    )
    ax2.plot(
        gammas,
        rmses_mice,
        "--s",
        color=COLORS["MAR MICE (Ridge)"],
        label="MAR MICE (Ridge)",
        linewidth=1.8,
    )
    ax2.set_xlabel(r"MNAR Mechanism Strength ($\gamma$)")
    ax2.set_ylabel("Cell RMSE (Missing Values)")
    ax2.set_title("B: Cell RMSE vs MNAR Mechanism Strength")
    ax2.grid(True)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

    fig.tight_layout()
    save_plot(fig, "error_by_mnar_mechanism_strength", output_dir)


def plot_sensitivity_tipping_point(output_dir: Path) -> None:
    """Figure 6: Sensitivity analysis tipping point curve under pattern mixture delta shifts."""
    print("[6/7] Generating sensitivity_tipping_point_curve...")
    rng = np.random.RandomState(42)
    n = 1000
    x = rng.normal(0, 1, size=n)
    y = 0.5 * x + rng.normal(0.4, 1.0, size=n)  # True mean = 0.40
    mask = rng.uniform(0, 1, size=n) < 0.30
    y_obs = y.copy()
    y_obs[mask] = np.nan

    df = pd.DataFrame({"x": x, "response": y_obs})

    # Run grid sensitivity analysis across delta in [-2.0, 2.0]
    deltas = list(np.linspace(-2.0, 2.0, 41))
    grid_res = run_sensitivity_grid(
        data=df,
        target_column="response",
        delta_grid=deltas,
        random_state=42,
    )

    df_grid = grid_res.grid_df
    est = df_grid["target_mean"].to_numpy()
    ci_low = df_grid["ci_lower"].to_numpy()
    ci_high = df_grid["ci_upper"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    ax.plot(
        deltas,
        est,
        "-",
        color="#1f77b4",
        linewidth=2.2,
        label=r"Adjusted Mean Estimate $\hat{\mu}(\delta)$",
    )
    ax.fill_between(
        deltas, ci_low, ci_high, color="#1f77b4", alpha=0.22, label="95% Bootstrap Confidence Band"
    )

    # Null hypothesis threshold
    null_val = 0.0
    ax.axhline(
        null_val, color="#d62728", linestyle="--", linewidth=1.5, label="Null Hypothesis (Mean = 0)"
    )

    # Find tipping point where 95% CI crosses 0
    tipping_idx = np.where(ci_low < null_val)[0]
    if len(tipping_idx) > 0:
        t_idx = tipping_idx[0]
        delta_star = float(deltas[t_idx])
        ax.axvline(
            delta_star,
            color="#555555",
            linestyle=":",
            linewidth=1.5,
            label=f"Tipping Point $\\delta^* = {delta_star:.2f}$",
        )
        ax.scatter([delta_star], [ci_low[t_idx]], color="#d62728", zorder=5, s=60)
        ax.annotate(
            f"Tipping Point: $\\delta^* = {delta_star:.2f}\\sigma$\n(Null becomes plausible)",
            xy=(delta_star, ci_low[t_idx]),
            xytext=(delta_star + 0.3, ci_low[t_idx] - 0.35),
            arrowprops=dict(facecolor="#333333", shrink=0.08, width=1, headwidth=5),
            fontsize=9.5,
            backgroundcolor="#f8f9fa",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="#cccccc", facecolor="#f8f9fa"),
        )

    ax.set_xlabel(r"Sensitivity Parameter $\delta$ (Standard Deviations Shift in Missing Data)")
    ax.set_ylabel("Estimated Population Mean Outcome")
    ax.set_title("Figure 6: Sensitivity Analysis Tipping-Point Curve (Pattern Mixture Model)")
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left")

    save_plot(fig, "sensitivity_tipping_point_curve", output_dir)


def plot_runtime_scaling(output_dir: Path) -> None:
    """Figure 7: Computational runtime scaling vs sample size N and dimension p."""
    print("[7/7] Generating runtime_scaling_n_p...")
    sample_sizes = [500, 1000, 2500, 5000, 10000]
    df_n = benchmark_runtime_vs_n(sample_sizes=sample_sizes, random_state=42)

    feature_counts = [4, 8, 16, 32, 64]
    df_p = benchmark_runtime_vs_dimension(
        feature_counts=feature_counts, n_samples=2000, random_state=42
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    methods = [
        ("Umbra (Auto)", "-o", 2.2),
        ("MAR MICE (Ridge)", "--s", 1.6),
        ("MAR MICE (PMM)", "--d", 1.6),
        ("Heckman Selection", "--^", 1.6),
        ("Pattern Mixture", "--x", 1.6),
    ]

    # Panel A: Scaling vs N
    for m, style, lw in methods:
        if m in df_n.columns:
            ax1.plot(
                df_n["N"],
                df_n[m],
                style,
                label=m,
                color=COLORS.get(m, "#555555"),
                linewidth=lw,
                markersize=5.5,
            )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xticks(sample_sizes)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.set_xlabel("Sample Size N (Log Scale)")
    ax1.set_ylabel("Execution Time (seconds, Log Scale)")
    ax1.set_title("A: Runtime Scaling vs Sample Size N (p = 5)")
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

    # Panel B: Scaling vs p
    for m, style, lw in methods:
        if m in df_p.columns:
            ax2.plot(
                df_p["p"],
                df_p[m],
                style,
                label=m,
                color=COLORS.get(m, "#555555"),
                linewidth=lw,
                markersize=5.5,
            )

    ax2.set_xlabel("Feature Count p (N = 2,000)")
    ax2.set_ylabel("Execution Time (seconds)")
    ax2.set_title("B: Runtime Scaling vs Feature Dimension p")
    ax2.set_xticks(feature_counts)
    ax2.grid(True)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

    fig.tight_layout()
    save_plot(fig, "runtime_scaling_n_p", output_dir)


def generate_all_figures(output_dir: str = "benchmarks/figures"):
    """Generate all 7 publication figures programmatically."""
    out_dir = ensure_output_dir(output_dir)
    print(f"Generating 7 publication figures in: {out_dir.resolve()} (PNG @ 300 DPI + Vector PDF)")
    t0 = time.perf_counter()

    plot_bias_vs_missingness(out_dir)
    plot_rmse_vs_missingness(out_dir)
    plot_coverage_vs_sample_size(out_dir)
    plot_router_confusion_matrix(out_dir)
    plot_error_by_mechanism_strength(out_dir)
    plot_sensitivity_tipping_point(out_dir)
    plot_runtime_scaling(out_dir)

    print(f"\nAll 7 publication figures successfully generated in {time.perf_counter() - t0:.1f}s.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-grade benchmark figures for Umbra."
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/figures",
        help="Target directory for PNG and PDF figures (default: benchmarks/figures)",
    )
    args = parser.parse_args()
    generate_all_figures(args.output_dir)


if __name__ == "__main__":
    main()
