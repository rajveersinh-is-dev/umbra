"""
Command Line Interface (CLI) for Umbra.

Commands:
- `umbra diagnose <file.csv>`: Screens columns for MNAR risk and prints explanation.
- `umbra impute <file.csv>`: Performs honest imputation with sensitivity ranges.
"""

from pathlib import Path
from typing import Optional

import click
import pandas as pd
from rich.console import Console

from umbra import __version__ as umbra_version
from umbra.api import UmbraImputer
from umbra.diagnostics.mnar_risk_score import diagnose_dataframe
from umbra.explain import diagnostics_to_markdown, explain_diagnostics, explain_sensitivity


@click.group()
@click.version_option(version=umbra_version, prog_name="umbra")
def main():
    """Umbra: MNAR-Aware Missing Data Diagnostic and Imputation CLI."""
    pass


@main.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output-markdown",
    "-m",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to export diagnostic report in GitHub-flavored Markdown.",
)
def diagnose(data_path: Path, output_markdown: Optional[Path]):
    """
    Diagnose missingness mechanisms in DATA_PATH and output MNAR risk assessment.
    """
    console = Console()
    console.print(f"[bold green]Loading data from {data_path}...[/bold green]")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        console.print(f"[bold red]Error reading file: {e}[/bold red]")
        raise click.Abort()

    missing_cols = [c for c in df.columns if df[c].isna().any()]
    if not missing_cols:
        console.print("[bold green]No missing values found in the dataset![/bold green]")
        return

    console.print(f"Found {len(missing_cols)} column(s) with missing data: {missing_cols}")
    console.print(
        "[dim]Evaluating Little's MCAR test, covariate shifts, and self-censoring heuristics...[/dim]"
    )

    reports = diagnose_dataframe(df)
    explain_diagnostics(reports, console=console)

    if output_markdown:
        md_text = diagnostics_to_markdown(reports)
        output_markdown.write_text(md_text, encoding="utf-8")
        console.print(
            f"[bold green]Diagnostic markdown report saved to: {output_markdown}[/bold green]"
        )


@main.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output path for imputed CSV (defaults to imputed_<input>.csv).",
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["auto", "mar", "heckman", "pattern_mixture"]),
    default="auto",
    help="Imputation strategy.",
)
@click.option(
    "--delta",
    type=float,
    default=0.0,
    help="Sensitivity shift delta in std devs for pattern-mixture models.",
)
@click.option(
    "--sensitivity/--no-sensitivity",
    default=True,
    help="Whether to run sensitivity grid analysis on high/medium MNAR risk columns.",
)
@click.option(
    "--sensitivity-output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output path for sensitivity grid results CSV.",
)
@click.option(
    "--shadow-col",
    "-sc",
    multiple=True,
    help="Specify shadow variable pair in format target:shadow_var (e.g. income:contact_attempts).",
)
def impute(
    data_path: Path,
    output: Optional[Path],
    strategy: str,
    delta: float,
    sensitivity: bool,
    sensitivity_output: Optional[Path],
    shadow_col: tuple,
):
    """
    Impute missing values in DATA_PATH with honest MNAR handling and sensitivity intervals.
    """
    console = Console()
    console.print(f"[bold green]Loading data from {data_path}...[/bold green]")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        console.print(f"[bold red]Error reading file: {e}[/bold red]")
        raise click.Abort()

    # Parse shadow cols
    shadow_dict = {}
    for item in shadow_col:
        if ":" in item:
            t, s = item.split(":", 1)
            shadow_dict[t.strip()] = s.strip()

    if output is None:
        output = data_path.parent / f"imputed_{data_path.name}"

    console.print(f"[bold]Fitting UmbraImputer (strategy={strategy}, delta={delta})...[/bold]")
    imputer = UmbraImputer(
        strategy=strategy,
        delta=delta,
        shadow_cols=shadow_dict if shadow_dict else None,
        run_sensitivity=sensitivity,
        verbose=True,
    )
    df_imputed = imputer.fit_transform(df)

    df_imputed.to_csv(output, index=False)
    console.print(f"[bold green]Imputed dataset saved to: {output}[/bold green]")

    if sensitivity and imputer.sensitivity_reports_:
        console.rule("[bold yellow]MNAR Sensitivity Analysis[/bold yellow]")
        for col, rep in imputer.sensitivity_reports_.items():
            console.print(explain_sensitivity(rep))
            if sensitivity_output:
                col_safe = col.replace(" ", "_").replace("/", "_")
                out_path = (
                    sensitivity_output
                    if len(imputer.sensitivity_reports_) == 1
                    else sensitivity_output.with_stem(f"{sensitivity_output.stem}_{col_safe}")
                )
                rep.grid_df.to_csv(out_path, index=False)
                console.print(
                    f"[bold green]Sensitivity grid for '{col}' saved to: {out_path}[/bold green]"
                )


if __name__ == "__main__":
    main()
