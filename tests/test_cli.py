"""
Unit tests for Umbra CLI.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from umbra.cli import main


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "age": rng.randint(20, 70, size=100),
            "education": rng.randint(8, 20, size=100),
            "income": rng.normal(50000, 15000, size=100),
            "contact": rng.normal(5, 2, size=100),
        }
    )
    # Introduce missingness in income
    df.loc[0:25, "income"] = np.nan
    csv_file = tmp_path / "test_data.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.2.0" in result.output


def test_cli_diagnose(sample_csv: Path, tmp_path: Path):
    runner = CliRunner()
    out_md = tmp_path / "report.md"
    result = runner.invoke(main, ["diagnose", str(sample_csv), "-m", str(out_md)])
    assert result.exit_code == 0
    assert "income" in result.output
    assert out_md.exists()
    content = out_md.read_text(encoding="utf-8")
    assert "income" in content


def test_cli_impute_auto(sample_csv: Path, tmp_path: Path):
    runner = CliRunner()
    out_csv = tmp_path / "imputed.csv"
    sens_csv = tmp_path / "sensitivity.csv"
    result = runner.invoke(
        main,
        [
            "impute",
            str(sample_csv),
            "-o",
            str(out_csv),
            "--sensitivity-output",
            str(sens_csv),
            "-sc",
            "income:contact",
        ],
    )
    assert result.exit_code == 0
    assert out_csv.exists()
    df_imp = pd.read_csv(out_csv)
    assert not df_imp["income"].isna().any()
