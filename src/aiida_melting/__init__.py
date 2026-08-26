"""Extensible AiiDA melting-temperature workflows."""

from .api import get_common_inputs, get_melting_workflow, get_method_inputs, list_melting_methods

__all__ = (
    "get_common_inputs",
    "get_melting_workflow",
    "get_method_inputs",
    "list_melting_methods",
)
__version__ = "0.1.0"
