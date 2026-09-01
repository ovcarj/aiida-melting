"""Tabular presentation of normalized melting result records."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .records import ResultRecord


def results_table(records: Iterable[ResultRecord]) -> pd.DataFrame:
    """Make a stable DataFrame suitable for terminal, CSV, or notebook output."""
    rows = []
    for record in records:
        rows.append(
            {
                "process_pk": record.process_pk,
                "formula": record.formula,
                "material_id": record.material_id,
                "pressure_gpa": record.pressure_gpa,
                "method": record.method,
                "calculator": record.calculator,
                "artifact_sha256": record.artifact_sha256,
                "atom_count": record.atom_count,
                "melting_temperature_k": record.melting_temperature_k,
                "uncertainty_k": record.uncertainty_k,
                "status": record.scientific_status,
                "ctime": record.ctime,
            }
        )
    return pd.DataFrame(rows)
