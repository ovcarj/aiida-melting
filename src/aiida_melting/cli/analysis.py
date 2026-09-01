"""Read-only analysis commands shared by ``verdi`` and ``aiida-melting``."""

from __future__ import annotations

import json
from pathlib import Path

import click

from ..analysis.calphy.diagnostics import calphy_diagnostics
from ..analysis.calphy.reader import read_calphy_process
from ..analysis.query import query_results


def _filters(function):
    for decorator in (
        click.option("--element", "elements", multiple=True, help="Require an element."),
        click.option("--pressure-gpa", type=float),
        click.option("--method"),
        click.option("--calculator"),
        click.option("--model-hash"),
        click.option("--status"),
    ):
        function = decorator(function)
    return function


def _records(**kwargs):
    return query_results(**{key: value for key, value in kwargs.items() if value not in (None, ())})


@click.command("results")
@_filters
@click.option(
    "--format", "output_format", type=click.Choice(("table", "csv", "json")), default="table"
)
def results_command(output_format: str, **filters) -> None:
    """List successful public melting results."""
    records = _records(**filters)
    if output_format == "json":
        click.echo(json.dumps([record.as_dict() for record in records], indent=2, sort_keys=True))
        return
    from ..analysis.tables import results_table

    table = results_table(records)
    if output_format == "csv":
        click.echo(table.to_csv(index=False), nl=False)
    else:
        click.echo(table.to_string(index=False) if not table.empty else "No matching results.")


@click.command("inspect")
@click.argument("process")
def inspect_command(process: str) -> None:
    """Print retrieved Calphy availability and diagnostic metrics for PROCESS."""
    try:
        diagnostics = calphy_diagnostics(read_calphy_process(process))
    except Exception as exception:
        raise click.ClickException(str(exception)) from exception
    click.echo(
        json.dumps(
            {
                "metrics": diagnostics.metrics,
                "messages": diagnostics.messages,
                "missing_data": diagnostics.missing_data,
                "warnings": diagnostics.warnings,
                "uncertainty_available": diagnostics.uncertainty_available,
            },
            indent=2,
            sort_keys=True,
        )
    )


_CALPHY_PLOT_KINDS = (
    "equilibration",
    "reference-switching",
    "temperature-sweeps",
    "free-energy-crossing",
    "attempt-history",
    "overview",
)


def _calphy_plotter(kind: str):
    from ..analysis.calphy import plots

    return {
        "equilibration": plots.plot_equilibration,
        "reference-switching": plots.plot_reference_switching,
        "temperature-sweeps": plots.plot_temperature_sweeps,
        "free-energy-crossing": plots.plot_free_energy_crossing,
        "attempt-history": plots.plot_attempt_history,
        "overview": plots.plot_calphy_overview,
    }[kind]


@click.group("plot")
def plot_group() -> None:
    """Create a non-interactive analysis figure."""


@plot_group.command("calphy")
@click.argument("process")
@click.option("--kind", "plot_kind", type=click.Choice(_CALPHY_PLOT_KINDS), default="overview")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
def plot_calphy_command(process: str, plot_kind: str, output: Path) -> None:
    """Save a Calphy figure for PROCESS to OUTPUT."""
    try:
        figure = _calphy_plotter(plot_kind)(read_calphy_process(process))
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160, bbox_inches="tight")
    except Exception as exception:
        raise click.ClickException(str(exception)) from exception
    click.echo(str(output))


def _comparison_command(name: str, plotter_name: str, help_text: str):
    @plot_group.command(name)
    @_filters
    @click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
    def command(output: Path, **filters) -> None:
        """Save a general melting-results figure."""
        from ..analysis import plots

        figure = getattr(plots, plotter_name)(_records(**filters))
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160, bbox_inches="tight")
        click.echo(str(output))

    command.help = help_text


_comparison_command("comparison", "plot_comparison", "Compare melting results.")
_comparison_command(
    "size-convergence", "plot_size_convergence", "Plot temperature against atom count."
)


def add_analysis_commands(group) -> None:
    """Attach analysis commands to an existing Click root group."""
    group.add_command(results_command)
    group.add_command(inspect_command)
    group.add_command(plot_group)
