"""
Streamlit Web Application for Umbra.

Interactive exploration of MNAR missing data diagnostics,
honest imputation, and sensitivity grid analysis.
"""

from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from scripts.build_synthetic_benchmarks import generate_benchmark_battery
from umbra.api import UmbraImputer
from umbra.diagnostics.mcar_test import littles_mcar_test
from umbra.diagnostics.mnar_risk_score import diagnose_dataframe
from umbra.explain import diagnostics_to_markdown
from umbra.sensitivity.grid_analysis import run_sensitivity_grid

# Streamlit Page Setup
st.set_page_config(
    page_title="Umbra | MNAR Missing Data Diagnostics & Sensitivity",
    page_icon="??",
    layout="wide",
)

st.title("?? Umbra: MNAR-Aware Missing Data Imputer")
st.markdown(
    """
    **Diagnose when missing data is Not-Missing-At-Random (MNAR), and refuse to pretend a single
    confident point estimate is safe when it isn't.**
    """
)

# Sidebar: Data Source Selection
st.sidebar.header("?? Data Source")
data_source = st.sidebar.radio(
    "Choose Data Input:",
    [
        "Preset: MNAR High Severity (Income Self-Censoring)",
        "Preset: MNAR Medium Severity",
        "Preset: MAR (Missing at Random)",
        "Preset: MCAR (Missing Completely at Random)",
        "Preset: Real-World CPS Income Survey",
        "Upload Custom CSV",
    ],
)


@st.cache_data
def get_presets():
    return generate_benchmark_battery(n_samples=1500, random_state=42)


presets = get_presets()

df = None
ground_truth_df = None
selected_target = None

if data_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file with missing values", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    if "MNAR High" in data_source:
        bench = presets["MNAR_HIGH"]
    elif "MNAR Medium" in data_source:
        bench = presets["MNAR_MEDIUM"]
    elif "MAR" in data_source:
        bench = presets["MAR"]
    elif "MCAR" in data_source:
        bench = presets["MCAR"]
    else:
        cps_path = Path("data/processed/cps_income_survey_ground_truth.csv")
        if cps_path.exists():
            df_full = pd.read_csv(cps_path)
            # Create MNAR dropout in annual_income
            rng = np.random.RandomState(42)
            y = df_full["annual_income"].values
            log_y = np.log(y)
            prob_miss = 1 / (
                1
                + np.exp(
                    -(
                        1.2 * (log_y - np.mean(log_y))
                        + 0.4 * (df_full["contact_attempts"].values - 3)
                    )
                )
            )
            mask = rng.uniform(0, 1, size=len(df_full)) < prob_miss
            df = df_full.copy()
            df.loc[mask, "annual_income"] = np.nan
            ground_truth_df = df_full
            selected_target = "annual_income"
        else:
            bench = presets["MNAR_MEDIUM"]

    if df is None:
        df = bench.data_observed.copy()
        ground_truth_df = bench.data_complete.copy()
        selected_target = bench.target_col

if df is None:
    st.info("?? Please upload a CSV file or select a preset benchmark from the sidebar to begin.")
    st.stop()

# Identify missing columns
missing_cols = [c for c in df.columns if df[c].isna().any()]
if not selected_target or selected_target not in missing_cols:
    selected_target = missing_cols[0] if missing_cols else None

tabs = st.tabs(
    [
        "1. Data Overview",
        "2. MNAR Diagnostics",
        "3. Sensitivity Analysis & Imputation",
        "4. Methodology",
    ]
)

# TAB 1: Data Overview
with tabs[0]:
    st.subheader("Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Columns with Missing Data", len(missing_cols))

    st.dataframe(df.head(10), use_container_width=True)

    if missing_cols:
        st.subheader("Missingness Proportions")
        miss_summary = pd.DataFrame(
            {
                "Column": missing_cols,
                "Missing Rows": [df[c].isna().sum() for c in missing_cols],
                "Missing Rate": [f"{df[c].isna().mean():.1%}" for c in missing_cols],
                "Data Type": [str(df[c].dtype) for c in missing_cols],
            }
        )
        st.dataframe(miss_summary, use_container_width=True)
    else:
        st.success("No missing values detected in this dataset.")

# TAB 2: MNAR Diagnostics
with tabs[1]:
    st.subheader("Missingness Mechanism Screening")
    st.markdown(
        "> ?? **Fundamental Limit**: Observed data alone cannot prove whether missingness is MAR vs. MNAR. "
        "Umbra synthesizes multiple converging heuristics into an empirical **MNAR Risk Score**."
    )

    if not missing_cols:
        st.info("No incomplete columns to diagnose.")
    else:
        with st.spinner(
            "Running Little's MCAR test, covariate shift analysis, and self-censoring checks..."
        ):
            numeric_df = df.select_dtypes(include=[np.number])
            little_res = littles_mcar_test(numeric_df) if numeric_df.shape[1] > 1 else None
            reports = diagnose_dataframe(df)

        # Little's MCAR Test Card
        if little_res:
            st.markdown("#### Global MCAR Assessment (Little's Test)")
            col_l1, col_l2, col_l3, col_l4 = st.columns(4)
            col_l1.metric("Chi-Squared Stat", f"{little_res.statistic:.2f}")
            col_l2.metric("Degrees of Freedom", little_res.degrees_of_freedom)
            col_l3.metric("p-value", f"{little_res.p_value:.2e}")
            verdict = (
                "? REJECT MCAR (MAR or MNAR)"
                if little_res.is_rejected
                else "? Consistent with MCAR"
            )
            col_l4.metric("Little's Test Verdict", verdict)
            st.caption(little_res.note)

        st.markdown("---")
        st.markdown("#### Per-Column Risk Assessments")

        for col_name, rep in reports.items():
            if rep.risk_level == "HIGH":
                badge = "?? **HIGH MNAR RISK**"
            elif rep.risk_level == "MEDIUM":
                badge = "?? **MEDIUM MNAR RISK**"
            else:
                badge = "?? **LOW MNAR RISK**"

            with st.expander(
                f"{col_name} ? {badge} (Score: {rep.composite_score:.2f})", expanded=True
            ):
                st.write(f"**Explanation:** {rep.explanation}")
                st.write(f"**Recommended Strategy:** `{rep.recommended_strategy}`")
                if rep.shadow_candidate:
                    st.write(
                        f"**Candidate Instrument / Shadow Variable:** `{rep.shadow_candidate}`"
                    )
                if rep.citation:
                    st.info(f"?? **Literature Reference:** {rep.citation}")

                # Signals table
                sig_data = []
                for s in rep.signals:
                    sig_data.append(
                        {
                            "Diagnostic Signal": s.name,
                            "Status": "TRIGGERED" if s.is_triggered else "PASS",
                            "Score": f"{s.score:.2f}",
                            "Weight": f"{s.weight:.2f}",
                            "Description": s.description,
                        }
                    )
                st.dataframe(pd.DataFrame(sig_data), use_container_width=True)

        # Download diagnostic report
        md_report = diagnostics_to_markdown(reports)
        st.download_button(
            label="?? Download Diagnostic Audit Report (Markdown)",
            data=md_report,
            file_name="umbra_diagnostics_report.md",
            mime="text/markdown",
        )

# TAB 3: Sensitivity Analysis & Imputation
with tabs[2]:
    st.subheader("Honest Imputation & Sensitivity Analysis")
    st.markdown(
        "Instead of pretending a single imputed number is safe, explore how downstream estimates "
        "shift across a grid of plausible unobserved non-response departures (delta)."
    )

    if not missing_cols:
        st.info("No missing columns to impute.")
    else:
        target_to_impute = st.selectbox(
            "Select Target Variable to Impute & Analyze:", missing_cols, index=0
        )

        col_s1, col_s2 = st.columns(2)
        strategy = col_s1.selectbox(
            "Imputation Strategy:",
            ["auto", "mar", "heckman", "pattern_mixture"],
            index=0,
            help="'auto' applies Heckman if shadow variable exists, otherwise pattern mixture sensitivity.",
        )
        delta_val = col_s2.slider(
            "Pattern-Mixture Shift (Delta in Std Deviations):",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="Delta=0 is standard MAR. Positive delta assumes missing units have higher true values.",
        )

        with st.spinner("Computing sensitivity grid and generating imputations..."):
            sens_report = run_sensitivity_grid(df, target_column=target_to_impute)
            imputer = UmbraImputer(strategy=strategy, delta=delta_val, run_sensitivity=False)
            df_imputed = imputer.fit_transform(df)

        # Plot sensitivity curve
        fig, ax = plt.subplots(figsize=(10, 4.5))
        grid_df = sens_report.grid_df
        deltas = grid_df["delta"].values
        means = grid_df["target_mean"].values

        ax.plot(
            deltas,
            means,
            marker="o",
            color="#1f77b4",
            linewidth=2.5,
            label="Imputed Mean vs MNAR Delta",
        )
        ax.axhline(
            sens_report.mar_baseline_estimate,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Naive MAR Estimate ({sens_report.mar_baseline_estimate:.2f})",
        )
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

        if ground_truth_df is not None and target_to_impute in ground_truth_df:
            true_mean = ground_truth_df[target_to_impute].mean()
            ax.axhline(
                true_mean,
                color="green",
                linestyle="-",
                linewidth=2,
                label=f"True Ground Truth ({true_mean:.2f})",
            )

        ax.set_xlabel("MNAR Sensitivity Shift (delta in residual std devs)")
        ax.set_ylabel(f"Imputed Mean of {target_to_impute}")
        ax.set_title(
            f"Sensitivity Analysis: Downstream Shift Across MNAR Assumptions for '{target_to_impute}'"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Naive MAR Point Estimate", f"{sens_report.mar_baseline_estimate:.2f}")
        col_m2.metric("Plausible Range Min", f"{sens_report.estimate_min:.2f}")
        col_m3.metric("Plausible Range Max", f"{sens_report.estimate_max:.2f}")
        col_m4.metric("Sensitivity Spread", f"{sens_report.uncertainty_spread:.2f}")

        st.info(f"?? **Interpretation:** {sens_report.interpretation}")

        # Download imputed data
        csv_buffer = StringIO()
        df_imputed.to_csv(csv_buffer, index=False)
        st.download_button(
            label="?? Download Imputed CSV",
            data=csv_buffer.getvalue(),
            file_name=f"imputed_{target_to_impute}.csv",
            mime="text/csv",
        )

# TAB 4: Methodology
with tabs[3]:
    st.subheader("Methodological Foundation")
    st.markdown(
        """
        ### The Missingness Regimes
        - **MCAR (Missing Completely At Random)**: Missingness is purely random noise ($P(R|X, Y) = P(R)$).
        - **MAR (Missing At Random)**: Missingness depends only on observed covariates ($P(R|X, Y) = P(R|X)$).
          This is what MICE, MissForest, and scikit-learn's `IterativeImputer` assume.
        - **MNAR (Missing Not At Random)**: Missingness depends directly on the unobserved value itself ($P(R|X, Y)$ depends on $Y$).

        ### Why Most Tools Hide the Truth
        Under MNAR, the missingness mechanism is **mathematically unidentifiable** from observed data alone.
        Standard tools pretend MAR holds, yielding false confidence and biased conclusions.

        Umbra's contribution is:
        1. Multi-signal empirical screening for MNAR risk.
        2. Heckman selection modeling with instrumental shadow variables.
        3. Automated sensitivity tipping-point analysis across plausible assumption grids.
        """
    )
