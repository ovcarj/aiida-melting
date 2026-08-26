"""Public discovery and process-input introspection API."""

from __future__ import annotations

from typing import Any

from .registry import get_melting_workflow, list_melting_methods
from .workflows.dispatcher import MeltingWorkChain


def _type_names(valid_type) -> list[str]:
    if valid_type is None:
        return []
    types = valid_type if isinstance(valid_type, tuple) else (valid_type,)
    return [item.__name__ for item in types]


def _describe(namespace, *, skip: set[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, port in namespace.items():
        if skip and name in skip:
            continue
        item: dict[str, Any] = {
            "required": port.required,
            "help": port.help or "",
        }
        if hasattr(port, "valid_type"):
            item["types"] = _type_names(port.valid_type)
        if hasattr(port, "items"):
            item["dynamic"] = port.dynamic
            item["children"] = _describe(port)
        result[name] = item
    return result


def get_common_inputs() -> dict[str, Any]:
    """Describe the stable dispatcher inputs, excluding method-specific parameters."""
    return _describe(MeltingWorkChain.spec().inputs, skip={"metadata", "method_parameters"})


def get_method_inputs(identifier: str) -> dict[str, Any]:
    """Describe the method-specific input namespace for a workflow."""
    workflow = get_melting_workflow(identifier)
    namespace = workflow.spec().inputs.get("method_parameters")
    return _describe(namespace) if namespace is not None else {}


__all__ = ("get_common_inputs", "get_melting_workflow", "get_method_inputs", "list_melting_methods")
