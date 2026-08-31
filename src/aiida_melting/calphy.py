"""Pure helpers shared by the Calphy processes."""

from __future__ import annotations

import math
import re
import shlex
from typing import Any

from aiida.common.exceptions import ValidationError

CPU_LAMMPS_CMDARGS: list[str] = []
GPU_LAMMPS_CMDARGS = ["-k", "on", "g", "1", "-sf", "kk"]
MLIAP_GPU_LAMMPS_CMDARGS = [
    "-k",
    "on",
    "g",
    "1",
    "-sf",
    "kk",
    "-pk",
    "kokkos",
    "newton",
    "on",
    "neigh",
    "half",
]
SAFE_LAMMPS_CMDARGS = (
    CPU_LAMMPS_CMDARGS,
    GPU_LAMMPS_CMDARGS,
    MLIAP_GPU_LAMMPS_CMDARGS,
)
SHELL_METACHARACTERS = frozenset(";&|<>`$(){}[]*?!\\\"'")
TM_PATTERN = re.compile(
    r"STATE:\s*Tm\s*=\s*(?P<temperature>\S+)\s*K\s*\+/-\s*(?P<uncertainty>\S+)\s*K"
)
RELIABILITY_PATTERNS = (
    "STATE: Tm unreliable",
    "treat the reported Tm as unreliable",
)


def pressure_gpa_to_bar(pressure: float) -> float:
    """Convert GPa to the metal-unit pressure used by Calphy/LAMMPS."""
    value = float(pressure)
    if not math.isfinite(value):
        raise ValidationError("pressure must be finite")
    return value * 10_000.0


def validate_lammps_cmdargs(arguments: list[Any]) -> str:
    """Validate the closed argument allowlist and return deterministic shell-like text.

    Calphy parses this string back into subprocess tokens; AiiDA never evaluates it
    through a shell.
    """
    if not isinstance(arguments, list):
        raise ValidationError("lammps_cmdargs must be a list")
    for token in arguments:
        if not isinstance(token, str) or not token:
            raise ValidationError("every LAMMPS argument must be a non-empty string")
        if any(character.isspace() for character in token):
            raise ValidationError("LAMMPS argument tokens must not contain whitespace")
        if any(character in SHELL_METACHARACTERS for character in token):
            raise ValidationError("LAMMPS argument tokens must not contain shell metacharacters")
    if arguments not in SAFE_LAMMPS_CMDARGS:
        raise ValidationError(
            "supported lammps_cmdargs are [], "
            "['-k', 'on', 'g', '1', '-sf', 'kk'], or the one-GPU ML-IAP "
            "Kokkos argument list"
        )
    return shlex.join(arguments)


CALPHY_FAILURE_PATTERNS = (
    (
        "lammps_style_unavailable",
        (
            "unrecognized pair style",
            "unknown pair style",
            "invalid pair style",
            "is missing styles this calculation needs",
        ),
    ),
    (
        "melting_attempts_exhausted",
        ("maximum number of tries reached", "failed to extrapolate melting temperature"),
    ),
    (
        "calphy_input_rejected",
        (
            "validation error for calculation",
            "pydantic_core._pydantic_core.validationerror",
            "uses the legacy calphy input format",
            "has no 'calculations:' list",
            "unknown top-level key",
            "input file input.yaml not found",
        ),
    ),
    (
        "lammps_runtime_failed",
        (
            "lammps segment failed:",
            "lammps execution error",
            "lammps reported an error",
            "error: lost atoms",
            "cuda error",
            "segmentation fault",
        ),
    ),
)


def classify_calphy_failure(text: str) -> str | None:
    """Classify known terminal Calphy/LAMMPS failures from retrieved diagnostics."""
    lowered = text.lower()
    for classification, patterns in CALPHY_FAILURE_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return classification
    if "traceback (most recent call last):" in lowered:
        return "calphy_execution_failed"
    return None


def parse_temperature_log(text: str, n_iterations: int) -> dict[str, Any]:
    """Parse and normalize the final Calphy melting-temperature state record."""
    matches = list(TM_PATTERN.finditer(text))
    if not matches:
        raise ValueError("final Calphy melting-temperature record is missing or malformed")
    match = matches[-1]
    try:
        temperature = float(match.group("temperature"))
        uncertainty = float(match.group("uncertainty"))
    except ValueError as exception:
        raise ValueError(
            "Calphy melting-temperature record contains a malformed number"
        ) from exception
    if not math.isfinite(temperature) or temperature <= 0:
        raise ArithmeticError("Calphy melting temperature must be finite and positive")
    uncertainty_available = n_iterations > 1 and math.isfinite(uncertainty) and uncertainty > 0
    reliability_warnings = [marker for marker in RELIABILITY_PATTERNS if marker in text]
    return {
        "temperature": temperature,
        "uncertainty": uncertainty if uncertainty_available else None,
        "uncertainty_available": uncertainty_available,
        "status": "ambiguous" if reliability_warnings else "success",
        "reliability_warnings": reliability_warnings,
    }


def is_transient_calphy_failure(exit_status: int | None) -> bool:
    """Classify only scheduler/resource and incomplete-retrieval failures as retryable."""
    return exit_status in {100, 110, 120, 140, 301}


def is_transient_transport_exception(exception: str | None) -> bool:
    """Classify only recognizable transport/connectivity exceptions as retryable."""
    text = exception or ""
    return any(
        marker in text
        for marker in (
            "TransportTaskException",
            "TransportInternalError",
            "ConnectionError",
            "TimeoutError",
        )
    )
