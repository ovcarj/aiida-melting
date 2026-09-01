"""Typed data objects independent of AiiDA and plotting libraries."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class TableData:
    """Numeric file content with its original column labels and source path."""

    columns: tuple[str, ...]
    values: NDArray[np.float64]
    source: str


@dataclass(frozen=True, slots=True)
class SwitchingData:
    """One forward or backward reference-switching replica."""

    direction: str
    replica: int
    lambda_values: NDArray[np.float64]
    integrand: NDArray[np.float64] | None
    raw: TableData


@dataclass(frozen=True, slots=True)
class PhaseData:
    """Retrieved data for one solid or liquid Calphy phase."""

    name: str
    directory: str
    report: dict
    equilibration: TableData | None
    switching: tuple[SwitchingData, ...]
    temperature_scaling: tuple[SwitchingData, ...]
    free_energy: TableData | None
    files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AttemptData:
    """One adaptive Calphy temperature interval."""

    key: str
    temperature_hint_k: float | None
    solid: PhaseData | None
    liquid: PhaseData | None


@dataclass(frozen=True, slots=True)
class CalphyAnalysis:
    """All analysis-relevant data from a retrieved Calphy calculation."""

    root: str
    input_parameters: dict
    attempts: tuple[AttemptData, ...]
    melting_temperature_k: float | None
    uncertainty_k: float | None
    log_records: tuple[str, ...]
    files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CalphyDiagnostics:
    """Measured diagnostics and availability notes without an aggregate grade."""

    metrics: dict[str, float | int | None] = field(default_factory=dict)
    messages: tuple[str, ...] = ()
    missing_data: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    uncertainty_available: bool = False
