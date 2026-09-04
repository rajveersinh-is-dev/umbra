"""
Human-Readable Explanation and Reporting Engine for Umbra.

Translates complex statistical tests, covariate shifts, and sensitivity analyses
into actionable, transparent prose and formatted tables for working data scientists.
"""

from typing import Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from umbra.diagnostics.mnar_risk_score import MNARRiskReport
from umbra.sensitivity.grid_analysis import SensitivityReport


def format_risk_badge(risk_level: str) -> str:
    """Format risk level as colored markdown or text badge."""
    if risk_level == "HIGH":
        return "[red]HIGH RISK[/red]"
    elif risk_level == "MEDIUM":
        return "[yellow]MEDIUM RISK[/yellow]"
    else:
        return "[green]LOW RISK[/green]"


def explain_diagnostics(
    reports: Dict[str, MNARRiskReport],
    console: Optional[Console] = None,
) -> str:
    """
    Generate an interactive Rich console display or formatted text summary of diagnostic reports.
    """
    if console is None:
        console = Console(record=True, width=110)

    console.print()
    console.rule("[bold cyan]Umbra Missingness Mechanism Diagnostics Report[/bold cyan]")
    console.print(
        "[dim]Note: MNAR is fundamentally unidentifiable from observed data alone. "
        "Umbra evaluates converging heuristic evidence to guide honest modeling.[/dim]\n"
    )

    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Variable", style="cyan", width=18)
    table.add_column("Missing Rate", justify="right", width=14)
    table.add_column("MNAR Risk", justify="center", width=14)
    table.add_column("Risk Score", justify="right", width=12)
    table.add_column("Recommended Strategy", style="italic", width=30)
    table.add_column("Shadow Variable", style="yellow", width=18)

    for col_name, rep in reports.items():
        if rep.risk_level == "HIGH":
            risk_styled = "[bold white on red] HIGH [/bold white on red]"
        elif rep.risk_level == "MEDIUM":
            risk_styled = "[bold black on yellow] MEDIUM [/bold black on yellow]"
        else:
            risk_styled = "[bold white on green] LOW [/bold white on green]"

        table.add_row(
            col_name,
            f"{rep.missing_rate:.1%}",
            risk_styled,
            f"{rep.composite_score:.2f}",
            rep.recommended_strategy,
            rep.shadow_candidate or "[dim]None[/dim]",
        )

    console.print(table)
    console.print()

    # Detailed variable cards
    for col_name, rep in reports.items():
        color = (
            "red"
            if rep.risk_level == "HIGH"
            else ("yellow" if rep.risk_level == "MEDIUM" else "green")
        )
        title = f"Variable Assessment: {col_name} ({rep.risk_level} MNAR Risk - Score {rep.composite_score:.2f})"

        details_text = Text()
        details_text.append(f"? Summary Explanation: {rep.explanation}\n\n", style="bold")
        details_text.append("? Triggered Diagnostic Signals:\n", style="underline")

        for sig in rep.signals:
            if sig.is_triggered:
                details_text.append(
                    f"  [+] {sig.name} (weight={sig.weight:.2f}, score={sig.score:.2f}):\n",
                    style="bold yellow",
                )
                details_text.append(f"      {sig.description}\n", style="dim")
            else:
                details_text.append(
                    f"  [-] {sig.name}: Not triggered ({sig.description})\n", style="dim"
                )

        if rep.domain_heuristic_matched and rep.citation:
            details_text.append(f"\n? Literature Citation: {rep.citation}\n", style="cyan italic")

        panel = Panel(details_text, title=title, border_style=color, padding=(1, 2))
        console.print(panel)
        console.print()

    return console.export_text() if console.record else ""


def diagnostics_to_markdown(reports: Dict[str, MNARRiskReport]) -> str:
    """Export diagnostic reports as GitHub-flavored Markdown."""
    lines = [
        "# Umbra Missingness Diagnostics Audit",
        "",
        "> [!NOTE]",
        "> **Identifiability Limit**: True Missing-Not-At-Random (MNAR) cannot be mathematically distinguished ",
        "> from unmeasured confounding using observed data alone. Umbra evaluates converging heuristics ",
        "> to quantify risk and guide defensible sensitivity analysis rather than claiming certainty.",
        "",
        "## Summary of Evaluated Variables",
        "",
        "| Variable | Missing Rate | MNAR Risk Level | Risk Score | Recommended Imputation Strategy | Candidate Instrument |",
        "| :--- | :---: | :---: | :---: | :--- | :--- |",
    ]

    for col_name, rep in reports.items():
        badge = f"**{rep.risk_level}**"
        shadow = f"`{rep.shadow_candidate}`" if rep.shadow_candidate else "?"
        lines.append(
            f"| `{col_name}` | {rep.missing_rate:.1%} | {badge} | {rep.composite_score:.2f} | `{rep.recommended_strategy}` | {shadow} |"
        )

    lines.append("")
    lines.append("## Detailed Evidence Breakdown")
    lines.append("")

    for col_name, rep in reports.items():
        lines.append(f"### Variable `{col_name}` ({rep.risk_level} Risk)")
        lines.append("")
        lines.append(f"**Explanation**: {rep.explanation}")
        lines.append("")
        lines.append("**Diagnostic Signals**:")
        for sig in rep.signals:
            status = "TRIGGERED" if sig.is_triggered else "PASS"
            lines.append(
                f"- `[{status}]` **{sig.name}** (Score: {sig.score:.2f}): {sig.description}"
            )

        if rep.citation:
            lines.append("")
            lines.append(f"> [!TIP]\n> **Literature Reference**: {rep.citation}")
        lines.append("")

    return "\n".join(lines)


def explain_sensitivity(report: SensitivityReport) -> str:
    """Generate human-readable explanation of sensitivity grid results."""
    lines = [
        f"Sensitivity Audit for '{report.target_column}':",
        f"  - Baseline MAR Estimate (delta=0) : {report.mar_baseline_estimate:.3f}",
        f"  - Plausible Grid Range            : [{report.estimate_min:.3f}, {report.estimate_max:.3f}]",
        f"  - Total Uncertainty Spread        : {report.uncertainty_spread:.3f}",
    ]
    if report.is_fragile:
        lines.append("  - Status: FRAGILE! The conclusion flips sign under plausible MNAR shifts.")
    else:
        lines.append("  - Status: ROBUST. The finding retains its sign across plausible shifts.")
    lines.append(f"  - Guidance: {report.interpretation}")
    return "\n".join(lines)
