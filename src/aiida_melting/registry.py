"""Dynamic melting-method discovery."""

from __future__ import annotations

from importlib.metadata import entry_points

from aiida.common.exceptions import EntryPointError

from .contracts import BaseMeltingWorkChain

PREFIX = "melting."
DISPATCHER = "melting.calculate"


def list_melting_methods() -> list[str]:
    """Return canonical identifiers for installed compatible method workflows."""
    methods: list[str] = []
    for entry_point in entry_points(group="aiida.workflows"):
        if not entry_point.name.startswith(PREFIX) or entry_point.name == DISPATCHER:
            continue
        try:
            workflow = entry_point.load()
        except Exception:
            continue
        if isinstance(workflow, type) and issubclass(workflow, BaseMeltingWorkChain):
            methods.append(entry_point.name)
    return sorted(set(methods))


def canonicalize_identifier(identifier: str) -> str:
    """Expand a short method alias."""
    return identifier if identifier.startswith(PREFIX) else f"{PREFIX}{identifier}"


def get_melting_workflow(identifier: str) -> type[BaseMeltingWorkChain]:
    """Load a compatible method workflow by canonical name or short alias."""
    canonical = canonicalize_identifier(identifier)
    matches = [
        entry_point
        for entry_point in entry_points(group="aiida.workflows")
        if entry_point.name == canonical
    ]
    if not matches:
        raise EntryPointError(f"unknown melting method: {identifier!r}")
    workflow = matches[0].load()
    if canonical == DISPATCHER or not (
        isinstance(workflow, type) and issubclass(workflow, BaseMeltingWorkChain)
    ):
        raise EntryPointError(f"entry point {canonical!r} is not a melting method workchain")
    return workflow
