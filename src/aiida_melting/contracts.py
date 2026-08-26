"""Shared process contracts and semantic validators."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from aiida import orm
from aiida.common.exceptions import ValidationError
from aiida.engine import WorkChain

ALLOWED_STATUSES = frozenset({"success", "warning", "not_converged", "failed"})
REQUIRED_CAPABILITIES = frozenset({"energy", "forces", "stress"})


def define_common_inputs(spec, *, include_method: bool = False) -> None:
    """Define the common input ports on a process specification."""
    spec.input("composition", valid_type=orm.Dict, help="Chemical composition mapping.")
    spec.input("calculator", valid_type=orm.Dict, help="Calculator capability description.")
    spec.input(
        "structure",
        valid_type=(orm.StructureData, orm.Dict),
        required=False,
        help="Structure or a structure-source specification.",
    )
    if include_method:
        spec.input("method", valid_type=orm.Str, help="Melting method identifier.")
        spec.input_namespace("method_parameters", dynamic=True, help="Selected method inputs.")


def define_common_outputs(spec) -> None:
    """Define the common output ports on a process specification."""
    spec.output("melting_temperature", valid_type=orm.Float, help="Temperature in kelvin.")
    spec.output("status", valid_type=orm.Str, help="Scientific result status.")
    spec.output("report", valid_type=orm.Dict, help="Machine-readable method report.")


def _plain(value: orm.Dict | Mapping[str, Any]) -> dict[str, Any]:
    return value.get_dict() if isinstance(value, orm.Dict) else dict(value)


def normalize_composition(value: orm.Dict | Mapping[str, Any]) -> dict[str, float]:
    """Validate and normalize a composition, sorting element symbols alphabetically."""
    from aiida.common.constants import elements

    raw = _plain(value)
    if not raw:
        raise ValidationError("composition must not be empty")
    known = {record["symbol"] for record in elements.values()}
    normalized: dict[str, float] = {}
    for symbol, amount in raw.items():
        if symbol not in known:
            raise ValidationError(f"unknown chemical element: {symbol!r}")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise ValidationError(f"amount for {symbol!r} must be numeric")
        amount = float(amount)
        if not math.isfinite(amount) or amount <= 0:
            raise ValidationError(f"amount for {symbol!r} must be finite and strictly positive")
        normalized[symbol] = amount
    total = sum(normalized.values())
    return {symbol: normalized[symbol] / total for symbol in sorted(normalized)}


def validate_calculator(value: orm.Dict | Mapping[str, Any]) -> dict[str, Any]:
    """Validate a calculator description while retaining extension fields."""
    raw = _plain(value)
    missing = {"name", "provides", "metadata"} - raw.keys()
    if missing:
        raise ValidationError(
            f"calculator is missing required fields: {', '.join(sorted(missing))}"
        )
    if not isinstance(raw["name"], str) or not raw["name"]:
        raise ValidationError("calculator.name must be a non-empty string")
    if not isinstance(raw["metadata"], dict):
        raise ValidationError("calculator.metadata must be a mapping")
    provides = raw["provides"]
    if not isinstance(provides, list) or not all(isinstance(item, str) for item in provides):
        raise ValidationError("calculator.provides must be a list of strings")
    missing_capabilities = REQUIRED_CAPABILITIES - set(provides)
    if missing_capabilities:
        raise ValidationError(
            "calculator is missing required capabilities: "
            + ", ".join(sorted(missing_capabilities))
        )
    return raw


def structure_composition(structure: orm.StructureData) -> dict[str, float]:
    """Return normalized composition, including weighted/mixed-occupancy kinds."""
    amounts: dict[str, float] = {}
    kinds = {kind.name: kind for kind in structure.kinds}
    for site in structure.sites:
        kind = kinds[site.kind_name]
        for symbol, weight in zip(kind.symbols, kind.weights, strict=True):
            amounts[symbol] = amounts.get(symbol, 0.0) + float(weight)
    return normalize_composition(amounts)


def validate_structure_composition(
    structure: orm.StructureData, composition: Mapping[str, float], tolerance: float = 1e-8
) -> None:
    """Require a structure to match an already normalized composition."""
    actual = structure_composition(structure)
    if set(actual) != set(composition) or any(
        abs(actual[symbol] - composition[symbol]) > tolerance for symbol in composition
    ):
        raise ValidationError(
            f"structure composition {actual} does not match requested "
            f"composition {dict(composition)}"
        )


def validate_source_specification(value: orm.Dict | Mapping[str, Any]) -> dict[str, Any]:
    """Validate the closed structure-source schema."""
    raw = _plain(value)
    unknown = set(raw) - {"source", "parameters"}
    if unknown:
        raise ValidationError(f"unknown structure source fields: {', '.join(sorted(unknown))}")
    if set(raw) != {"source", "parameters"}:
        raise ValidationError("structure source requires exactly 'source' and 'parameters'")
    if raw["source"] != "materials_project":
        raise ValidationError(f"unknown structure source: {raw['source']!r}")
    if not isinstance(raw["parameters"], dict):
        raise ValidationError("structure source parameters must be a mapping")
    return raw


def validate_outputs(outputs: Mapping[str, orm.Data]) -> str | None:
    """Return an error message when common workflow outputs are malformed."""
    required = {"melting_temperature", "status", "report"}
    if missing := required - outputs.keys():
        return f"child is missing outputs: {', '.join(sorted(missing))}"
    temperature, status, report = (
        outputs["melting_temperature"],
        outputs["status"],
        outputs["report"],
    )
    if (
        not isinstance(temperature, orm.Float)
        or not math.isfinite(temperature.value)
        or temperature.value <= 0
    ):
        return "child melting_temperature must be a finite, positive Float"
    if not isinstance(status, orm.Str) or status.value not in ALLOWED_STATUSES:
        return f"child status must be one of: {', '.join(sorted(ALLOWED_STATUSES))}"
    if not isinstance(report, orm.Dict):
        return "child report must be a Dict"
    return None


class BaseMeltingWorkChain(WorkChain):
    """Base class identifying compatible melting method workchains."""

    @classmethod
    def define(cls, spec) -> None:
        super().define(spec)
        define_common_inputs(spec)
        define_common_outputs(spec)
