"""Shared CLI commands."""

import json

import click

from ..api import get_common_inputs, get_method_inputs, list_melting_methods


@click.command("methods")
def methods_command() -> None:
    """List installed melting methods."""
    for identifier in list_melting_methods():
        click.echo(identifier)


@click.command("inputs")
@click.argument("method", required=False)
def inputs_command(method: str | None) -> None:
    """Show common inputs and, optionally, inputs for METHOD."""
    payload = {"common": get_common_inputs()}
    if method:
        try:
            payload["method"] = get_method_inputs(method)
        except Exception as exception:
            raise click.ClickException(str(exception)) from exception
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


def add_commands(group) -> None:
    """Attach the shared command set to a Click group."""
    group.add_command(methods_command)
    group.add_command(inputs_command)
