"""Shared process contracts and semantic validators."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from aiida import orm
from aiida.common.exceptions import ValidationError
from aiida.engine import WorkChain

ALLOWED_STATUSES = frozenset({"success", "unconverged", "ambiguous"})
REQUIRED_CAPABILITIES = frozenset({"energy", "forces", "stress"})


def define_common_inputs(spec, *, include_method: bool = False) -> None:
    """Define the common input ports on a process specification."""
    spec.input("composition", valid_type=orm.Dict, help="Chemical composition mapping.")
    spec.input("pressure", valid_type=orm.Float, help="Applied pressure in GPa.")
    spec.input(
        "description",
        valid_type=orm.Str,
        required=False,
        help="Human-readable calculation description.",
    )
    spec.input_namespace("calculator", help="Calculator description and artifacts.")
    spec.input(
        "calculator.metadata",
        valid_type=orm.Dict,
        help="Calculator identity, capabilities, and implementation metadata.",
    )
    spec.input_namespace(
        "calculator.files",
        valid_type=orm.SinglefileData,
        dynamic=True,
        required=False,
        help="Named provenance-tracked calculator artifacts.",
    )
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


def validate_pressure(value: orm.Float | float) -> float:
    """Return a finite pressure in GPa."""
    pressure = value.value if isinstance(value, orm.Float) else float(value)
    if not math.isfinite(pressure):
        raise ValidationError("pressure must be finite")
    return pressure


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


def validate_report(
    report: orm.Dict,
    status: str,
    *,
    composition: Mapping[str, float] | None = None,
    pressure: float | None = None,
    calculator_name: str | None = None,
    method: str | None = None,
) -> str | None:
    """Validate the minimal common report schema and optional expected values."""
    raw = report.get_dict()
    required = {"method", "units", "composition", "pressure", "calculator", "convergence_status"}
    if missing := required - raw.keys():
        return f"report is missing fields: {', '.join(sorted(missing))}"
    if not isinstance(raw["method"], str) or not raw["method"]:
        return "report.method must be a non-empty string"
    if method is not None and raw["method"] != method:
        return f"report.method must be {method!r}"
    units = raw["units"]
    if (
        not isinstance(units, dict)
        or units.get("melting_temperature") != "K"
        or units.get("pressure") != "GPa"
    ):
        return "report.units must define melting_temperature='K' and pressure='GPa'"
    try:
        reported_composition = normalize_composition(raw["composition"])
    except (TypeError, ValidationError) as exception:
        return f"report.composition is invalid: {exception}"
    if composition is not None and (
        set(reported_composition) != set(composition)
        or any(abs(reported_composition[key] - composition[key]) > 1e-8 for key in composition)
    ):
        return "report.composition does not match the requested composition"
    try:
        reported_pressure = validate_pressure(float(raw["pressure"]))
    except (TypeError, ValueError, ValidationError) as exception:
        return f"report.pressure is invalid: {exception}"
    if pressure is not None and abs(reported_pressure - pressure) > 1e-12:
        return "report.pressure does not match the requested pressure"
    calculator = raw["calculator"]
    if not isinstance(calculator, dict) or not isinstance(calculator.get("name"), str):
        return "report.calculator must contain a string name"
    if calculator_name is not None and calculator["name"] != calculator_name:
        return "report.calculator.name does not match calculator metadata"
    if raw["convergence_status"] != status:
        return "report.convergence_status must match status"
    return None


def validate_outputs(
    outputs: Mapping[str, orm.Data],
    *,
    composition: Mapping[str, float] | None = None,
    pressure: float | None = None,
    calculator_name: str | None = None,
    method: str | None = None,
) -> str | None:
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
    return validate_report(
        report,
        status.value,
        composition=composition,
        pressure=pressure,
        calculator_name=calculator_name,
        method=method,
    )


class BaseMeltingWorkChain(WorkChain):
    """Base class identifying compatible melting method workchains."""

    @classmethod
    def define(cls, spec) -> None:
        super().define(spec)
        define_common_inputs(spec)
        define_common_outputs(spec)
