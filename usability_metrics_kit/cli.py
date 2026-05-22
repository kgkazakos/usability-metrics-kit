"""Command-line interface for usability-metrics-kit (``umk``)."""

from __future__ import annotations

import json
import sys
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import llm, loaders
from .models import MetricResult, Scorecard
from .registry import REGISTRY, by_family, family_label, get

console = Console()


def _load_for(spec_kind: str, path: str) -> Any:
    if spec_kind == "matrix":
        return loaders.load_matrix(path)
    if spec_kind == "values":
        return loaders.load_values(path)
    return loaders.load_int_values(path)


def _score_one(metric: str, path: str) -> MetricResult:
    spec = get(metric)
    return spec.scorer(_load_for(spec.input_kind, path))


def _maybe_enrich(result: MetricResult, use_llm: bool, backend: str | None) -> dict:
    payload = result.to_dict()
    if use_llm and llm.is_available(backend):
        llm.enrich(payload, backend)
    return payload


def _render(result: dict) -> None:
    direction = "higher is better" if "higher" in result["direction"] else "lower is better"
    header = f"[bold]{result['name']}[/bold]  ·  {result['scale']}  ·  {direction}"
    body = Table.grid(padding=(0, 2))
    body.add_column(justify="right", style="dim")
    body.add_column()
    interval = result.get("interval")
    score_line = f"[bold cyan]{result['score']}[/bold cyan]"
    if interval:
        score_line += f"  [dim](95% CI {interval['low']}–{interval['high']}, {interval['method']})[/dim]"
    body.add_row("score", score_line)
    body.add_row("n", str(result["n"]))
    body.add_row("reading", result["interpretation"])
    if result.get("components"):
        body.add_row("detail", json.dumps(result["components"]))
    if result.get("benchmark"):
        body.add_row("benchmark", json.dumps(result["benchmark"]))
    if result.get("llm_summary"):
        body.add_row("[green]narrative[/green]", result["llm_summary"])
    for w in result.get("warnings", []):
        body.add_row("[yellow]note[/yellow]", w)
    console.print(Panel(body, title=header, border_style="cyan"))


@click.group(help="Calculate, benchmark and narrate usability metrics.")
@click.version_option()
def cli() -> None:  # pragma: no cover - entrypoint
    pass


@cli.command("list", help="List every metric the kit can score.")
def list_metrics() -> None:
    for family, specs in by_family().items():
        table = Table(title=family_label(family), title_justify="left", box=None)
        table.add_column("key", style="cyan")
        table.add_column("name")
        table.add_column("input", style="dim")
        table.add_column("what it does")
        for spec in specs:
            inp = f"{spec.input_kind} ({spec.item_scale})"
            table.add_row(spec.key, spec.name, inp, spec.blurb)
        console.print(table)
        console.print()


@cli.command("score", help="Score one metric from a CSV file.")
@click.option("--metric", "-m", required=True, type=click.Choice(sorted(REGISTRY)))
@click.argument("path", type=click.Path(exists=True))
@click.option("--json-output", is_flag=True, help="Emit JSON only (for CI).")
@click.option("--llm/--no-llm", "use_llm", default=True, help="LLM narrative (on by default).")
@click.option("--backend", type=click.Choice(["anthropic", "gemini", "openai"]), default=None)
def score(metric: str, path: str, json_output: bool, use_llm: bool, backend: str | None) -> None:
    try:
        result = _score_one(metric, path)
    except (ValueError, KeyError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        sys.exit(1)
    payload = _maybe_enrich(result, use_llm, backend)
    if json_output:
        click.echo(json.dumps(payload, indent=2))
    else:
        _render(payload)


@cli.command("scorecard", help="Score many metrics from a JSON spec: {metric: csv_path}.")
@click.argument("spec", type=click.Path(exists=True))
@click.option("--study", default="study", help="Study label.")
@click.option("--json-output", is_flag=True)
@click.option("--llm/--no-llm", "use_llm", default=True)
@click.option("--backend", type=click.Choice(["anthropic", "gemini", "openai"]), default=None)
def scorecard(spec: str, study: str, json_output: bool, use_llm: bool, backend: str | None) -> None:
    with open(spec, encoding="utf-8") as fh:
        mapping = json.load(fh)
    card = Scorecard(study=study)
    payloads = []
    for metric, path in mapping.items():
        try:
            result = _score_one(metric, path)
        except (ValueError, KeyError) as exc:
            console.print(f"[red]error scoring {metric}:[/red] {exc}")
            sys.exit(1)
        card.results.append(result)
        payloads.append(_maybe_enrich(result, use_llm, backend))
    if json_output:
        click.echo(json.dumps({"study": study, "results": payloads}, indent=2))
    else:
        console.print(f"[bold]Usability scorecard — {study}[/bold]\n")
        for p in payloads:
            _render(p)


def main() -> None:  # pragma: no cover - entrypoint
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
