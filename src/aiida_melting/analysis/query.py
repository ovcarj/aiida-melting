"""AiiDA QueryBuilder-backed discovery of public melting results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from math import gcd
from typing import Any

from aiida import orm

from .records import ResultRecord

DISPATCHER_PROCESS_TYPE = "aiida.workflows:melting.calculate"


def _formula(composition: dict[str, int]) -> str | None:
    if not composition:
        return None
    divisor = 0
    for amount in composition.values():
        divisor = amount if divisor == 0 else gcd(divisor, amount)
    return "".join(
        element + ("" if amount // divisor == 1 else str(amount // divisor))
        for element, amount in sorted(composition.items())
    )


def _structure_composition(structure: orm.StructureData | None) -> dict[str, int]:
    if structure is None:
        return {}
    return dict(sorted(Counter(site.kind_name for site in structure.sites).items()))


def _output(node: orm.ProcessNode, name: str) -> orm.Node | None:
    return node.outputs[name] if name in node.outputs else None


def _string_value(node: orm.Node | None) -> str | None:
    return node.value if isinstance(node, orm.Str) else None


def _float_value(node: orm.Node | None) -> float | None:
    return float(node.value) if isinstance(node, orm.Float) else None


def _record(node: orm.ProcessNode) -> ResultRecord | None:
    temperature = _float_value(_output(node, "melting_temperature"))
    if temperature is None:
        return None
    report_node = _output(node, "report")
    report: dict[str, Any] = report_node.get_dict() if isinstance(report_node, orm.Dict) else {}
    input_structure = node.inputs.structure if "structure" in node.inputs else None
    if not isinstance(input_structure, orm.StructureData):
        input_structure = None
    composition = _structure_composition(input_structure)
    calculator = report.get("calculator", {})
    if not isinstance(calculator, dict):
        calculator = {}
    prepared = report.get("prepared_structure_uuid")
    material_id = None
    structure_hash = None
    label = None
    if input_structure is not None:
        structure_hash = input_structure.base.caching.get_hash()
        label = input_structure.label or None
        extras = input_structure.base.extras.all
        material_id = extras.get("materials_project_id") or extras.get("material_id")
    uncertainty = report.get("diagnostics", {}).get("uncertainty_K")
    pressure = report.get("pressure")
    pressure_gpa = pressure.get("GPa") if isinstance(pressure, dict) else pressure
    atom_count = report.get("atom_count")
    if atom_count is None and isinstance(prepared, str):
        try:
            prepared_node = orm.load_node(prepared)
            atom_count = (
                len(prepared_node.sites) if isinstance(prepared_node, orm.StructureData) else None
            )
        except Exception:
            atom_count = None
    return ResultRecord(
        process_pk=node.pk,
        process_uuid=str(node.uuid),
        ctime=node.ctime.isoformat(),
        formula=_formula(composition),
        composition=composition,
        elements=tuple(sorted(composition)),
        pressure_gpa=float(pressure_gpa) if pressure_gpa is not None else None,
        method=report.get("method"),
        calculator=calculator.get("name"),
        artifact_filename=calculator.get("artifact_filename"),
        artifact_sha256=calculator.get("artifact_sha256"),
        structure_uuid=str(input_structure.uuid) if input_structure else None,
        structure_hash=structure_hash,
        structure_label=label,
        material_id=material_id,
        prepared_structure_uuid=prepared,
        atom_count=int(atom_count) if atom_count is not None else None,
        melting_temperature_k=temperature,
        uncertainty_k=float(uncertainty) if uncertainty is not None else None,
        scientific_status=_string_value(_output(node, "status")),
        report_uuid=str(report_node.uuid) if report_node else None,
    )


def query_results(
    *,
    elements: Iterable[str] | None = None,
    pressure_gpa: float | None = None,
    method: str | None = None,
    calculator: str | None = None,
    model_hash: str | None = None,
    status: str | None = None,
    date_from: datetime | None = None,
    date_until: datetime | None = None,
) -> list[ResultRecord]:
    """Return successful public dispatcher results matching optional filters.

    Filters not represented in AiiDA columns are applied after querying, keeping
    this API stable as workflow reports evolve.
    """
    filters: dict[str, Any] = {
        "process_type": {"==": DISPATCHER_PROCESS_TYPE},
    }
    if date_from is not None:
        filters["ctime"] = {">=": date_from}
    if date_until is not None:
        filters.setdefault("ctime", {}).update({"<=": date_until})
    builder = orm.QueryBuilder().append(orm.ProcessNode, filters=filters)
    records = [
        record
        for row in builder.iterall()
        if row[0].is_finished_ok and (record := _record(row[0])) is not None
    ]
    requested_elements = set(elements or ())
    return [
        record
        for record in records
        if (not requested_elements or requested_elements.issubset(record.elements))
        and (pressure_gpa is None or record.pressure_gpa == pressure_gpa)
        and (method is None or record.method == method)
        and (calculator is None or record.calculator == calculator)
        and (model_hash is None or record.artifact_sha256 == model_hash)
        and (status is None or record.scientific_status == status)
    ]
