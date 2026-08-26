"""Standalone command-line entry point."""

import click

from .common import add_commands


@click.group()
def main() -> None:
    """Inspect installed aiida-melting methods and their inputs."""


add_commands(main)
