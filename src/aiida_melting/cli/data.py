"""The ``verdi data melting`` command group."""

import click

from .common import add_commands


@click.group("melting")
def melting() -> None:
    """Inspect installed melting workflows."""


add_commands(melting)
