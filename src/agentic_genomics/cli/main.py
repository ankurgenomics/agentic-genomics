"""`agentic-genomics` command-line interface.

Run::

    agentic-genomics interpret --vcf data/samples/proband_demo.vcf \\
        --hpo HP:0001250,HP:0001263 --out reports/proband_demo.md
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from agentic_genomics import __version__
from agentic_genomics.agents.variant_interpreter import run_variant_interpreter

app = typer.Typer(
    add_completion=False,
    help="Agentic AI systems for genomics. Start with `interpret`.",
    rich_markup_mode="markdown",
)


@app.callback(invoke_without_command=True)
def _root(
    version: bool = typer.Option(False, "--version", help="Print version and exit."),
) -> None:
    if version:
        rprint(f"agentic-genomics {__version__}")
        raise typer.Exit()


@app.command()
def interpret(
    vcf: Path = typer.Option(..., exists=True, readable=True, help="Input VCF (bgzipped OK)."),
    hpo: str = typer.Option(
        "",
        help="Comma-separated HPO term IDs for the proband, e.g. HP:0001250,HP:0001263.",
    ),
    out: Path = typer.Option(
        Path("reports/variant_report.md"),
        help="Path to write the markdown report.",
    ),
    max_variants: int = typer.Option(
        50,
        min=1,
        max=500,
        help="Safety cap on variants pulled from the VCF.",
    ),
    trace_out: Path | None = typer.Option(
        None,
        help="Optional path to write the JSON reasoning trace.",
    ),
) -> None:
    """Run GenomicsCopilot on a VCF and write a ranked markdown report."""
    hpo_terms = [t.strip() for t in hpo.split(",") if t.strip()]
    rprint(Panel.fit(
        f"[bold]GenomicsCopilot[/bold]\n"
        f"VCF: {vcf}\n"
        f"HPO: {', '.join(hpo_terms) or '(none)'}\n"
        f"Max variants: {max_variants}",
        title="Run",
    ))

    state = run_variant_interpreter(
        vcf_path=str(vcf),
        hpo_terms=hpo_terms,
        max_variants=max_variants,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(state.report_markdown)

    if trace_out:
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        trace_out.write_text(json.dumps(state.reasoning_trace, indent=2))

    table = Table(title="Reasoning trace")
    table.add_column("Node", style="bold")
    table.add_column("Summary")
    for step in state.reasoning_trace:
        table.add_row(step["node"], step["summary"])
    rprint(table)

    rprint(f"\n[green]Report written to[/green] {out}")
    if trace_out:
        rprint(f"[green]Trace written to[/green] {trace_out}")


if __name__ == "__main__":
    app()
