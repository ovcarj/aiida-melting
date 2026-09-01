"""Normalized, serializable records for completed melting workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ResultRecord:
    """A portable summary of one public melting dispatcher process."""

    process_pk: int
    process_uuid: str
    ctime: str
    formula: str | None
    composition: dict[str, float]
    elements: tuple[str, ...]
    pressure_gpa: float | None
    method: str | None
    calculator: str | None
    artifact_filename: str | None
    artifact_sha256: str | None
    structure_uuid: str | None
    structure_hash: str | None
    structure_label: str | None
    material_id: str | None
    prepared_structure_uuid: str | None
    atom_count: int | None
    melting_temperature_k: float
    uncertainty_k: float | None
    scientific_status: str | None
    report_uuid: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        payload = asdict(self)
        payload["elements"] = list(self.elements)
        return payload
