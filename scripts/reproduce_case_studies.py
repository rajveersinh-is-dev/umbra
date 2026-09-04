"""
Reproduce End-to-End Semi-Synthetic Case Studies across 4 Realistic Domains.

Evaluates:
1. CPS Labor Economics Income Survey (Survey nonresponse)
2. NHANES Health Examination & Biomarkers (Clinical / lab nonresponse)
3. California Housing Reference (Spatial economics)
4. Longitudinal Clinical Trial Attrition (Patient dropout / pharmacovigilance)

Usage:
  python scripts/reproduce_case_studies.py
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import umbra  # noqa: E402
from umbra.api import UmbraImputer  # noqa: E402
from umbra.imputers.heckman_selection import HeckmanSelectionImputer  # noqa: E402
from umbra.imputers.mar_chained_equations import MARChainedEquationsImputer  # noqa: E402
from umbra.imputers.pattern_mixture import PatternMixtureImputer  # noqa: E402


class MeanBaseline:
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for c in df_out.columns:
            if df_out[c].isna().any():
                df_out[c] = df_out[c].fillna(df_out[c].mean())
        return df_out


def run_case_study(
    name: str,
    target_col: str,
    shadow_col: str | None,
    regressors: List[str],
    data_dir: Path,
) -> Dict[str, Any]:
    obs_file = data_dir / f"{name}_observed.csv"
    comp_file = data_dir / f"{name}_complete.csv"

    if not obs_file.exists() or not comp_file.exists():
        print(f"Data files for '{name}' not found in {data_dir}. Generating benchmark datasets...")
        from scripts.prepare_empirical_datasets import run_all_preparations

        run_all_preparations()

    if not obs_file.exists() or not comp_file.exists():
        raise FileNotFoundError(f"Missing processed data for {name} in {data_dir}")

    df_obs = pd.read_csv(obs_file)
    df_true = pd.read_csv(comp_file)

    mis_mask = df_obs[target_col].isna().to_numpy()
    missing_rate = float(np.mean(mis_mask))

    y_true_all = df_true[target_col].to_numpy()
    y_true_mis = y_true_all[mis_mask]

    # 1. Run Umbra Diagnostic Audit
    report = umbra.diagnose(df_obs)

    # 2. Fit true regression on complete data
    X_reg_true = df_true[regressors].to_numpy()
    reg_true = LinearRegression().fit(X_reg_true, y_true_all)
    beta_true = reg_true.coef_

    # 3. Benchmark Imputers
    methods: Dict[str, Any] = {
        "Naive Mean": MeanBaseline(),
        "MAR MICE (PMM)": MARChainedEquationsImputer(imputation_method="pmm", random_state=42),
        "Pattern Mixture (delta=0)": PatternMixtureImputer(delta=0.0, random_state=42),
    }

    if shadow_col and shadow_col in df_obs.columns:
        methods["Heckman Selection"] = HeckmanSelectionImputer(
            shadow_cols={target_col: shadow_col}, random_state=42
        )
        methods["Umbra Auto"] = UmbraImputer(
            strategy="auto",
            shadow_cols={target_col: shadow_col},
            run_sensitivity=True,
            random_state=42,
        )
    else:
        methods["Umbra Auto"] = UmbraImputer(
            strategy="auto",
            run_sensitivity=True,
            random_state=42,
        )

    results = []
    tipping_point = None

    for m_name, imputer in methods.items():
        df_imp = imputer.fit_transform(df_obs.copy())
        if isinstance(df_imp, np.ndarray):
            col_idx = list(df_obs.columns).index(target_col)
            y_imp_all = df_imp[:, col_idx]
            reg_indices = [list(df_obs.columns).index(r) for r in regressors]
            X_reg_imp = df_imp[:, reg_indices]
        else:
            y_imp_all = df_imp[target_col].to_numpy()
            X_reg_imp = df_imp[regressors].to_numpy()

        y_imp_mis = y_imp_all[mis_mask]

        bias_cell = float(np.mean(y_imp_mis) - np.mean(y_true_mis))
        rmse_cell = float(root_mean_squared_error(y_true_mis, y_imp_mis))
        mae_cell = float(mean_absolute_error(y_true_mis, y_imp_mis))

        reg_imp = LinearRegression().fit(X_reg_imp, y_imp_all)
        beta_imp = reg_imp.coef_
        beta_error = float(np.sum(np.abs(beta_imp - beta_true)))

        if m_name == "Umbra Auto" and hasattr(imputer, "sensitivity_reports_"):
            sens = imputer.sensitivity_reports_.get(target_col)
            if sens and sens.tipping_points:
                tipping_point = sens.tipping_points[0].tipping_delta

        results.append(
            {
                "Method": m_name,
                "Cell Bias": bias_cell,
                "Cell RMSE": rmse_cell,
                "Cell MAE": mae_cell,
                "Beta Error": beta_error,
            }
        )

    return {
        "name": name,
        "missing_rate": missing_rate,
        "diagnostics": report,
        "benchmark_table": pd.DataFrame(results),
        "tipping_point": tipping_point,
    }


def main():
    data_dir = root_dir / "data" / "processed"
    print("=" * 75)
    print("UMBRA REPRODUCIBLE REAL-WORLD CASE STUDIES")
    print("=" * 75)

    cases = [
        {
            "name": "cps_income",
            "target_col": "annual_income",
            "shadow_col": "contact_attempts",
            "regressors": ["age", "education_years", "hours_per_week"],
            "title": "Case 1: CPS Labor Economics (Wage Nonresponse)",
        },
        {
            "name": "nhanes_health",
            "target_col": "fasting_glucose",
            "shadow_col": "phlebotomy_difficulty",
            "regressors": ["age", "bmi", "systolic_bp"],
            "title": "Case 2: NHANES Clinical Biomarkers (Lab Nonresponse)",
        },
        {
            "name": "california_housing",
            "target_col": "median_income",
            "shadow_col": None,
            "regressors": ["housing_age", "ave_rooms", "population"],
            "title": "Case 3: California Housing Census (Spatial Missingness)",
        },
        {
            "name": "clinical_trial_attrition",
            "target_col": "endpoint_score",
            "shadow_col": "travel_distance",
            "regressors": ["baseline_score", "treatment_arm"],
            "title": "Case 4: Longitudinal Clinical Trial (Patient Dropout)",
        },
    ]

    all_summaries = []

    for c in cases:
        print(f"\nEvaluating {c['title']}...")
        res = run_case_study(
            name=c["name"],
            target_col=c["target_col"],
            shadow_col=c["shadow_col"],
            regressors=c["regressors"],
            data_dir=data_dir,
        )

        diag = res["diagnostics"]
        print(f"  Missing rate: {res['missing_rate']:.1%}")
        print(
            f"  Little's MCAR: stat={diag.mcar.statistic:.2f}, df={diag.mcar.degrees_of_freedom}, p={diag.mcar.p_value:.4e} (MCAR Rejected: {diag.mcar.is_rejected})"
        )
        cand_list = [
            f"{rep.best_candidate.variable_name} (F={rep.best_candidate.first_stage_f_stat:.1f})"
            for rep in diag.shadow_variables.values()
            if rep.best_candidate and rep.best_candidate.is_statistically_viable_candidate
        ]
        print(f"  Identified Auxiliary Variables: {cand_list if cand_list else 'None'}")
        if res["tipping_point"] is not None:
            print(f"  Tipping Point Delta: {res['tipping_point']:.3f} SD")

        print("\nImputer Performance Leaderboard:")
        print(res["benchmark_table"].to_string(index=False))
        all_summaries.append(res)

    print("\n" + "=" * 75)
    print("All 4 semi-synthetic case studies successfully executed and verified.")
    print("=" * 75)


if __name__ == "__main__":
    main()
