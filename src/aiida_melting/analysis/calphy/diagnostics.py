"""Transparent Calphy diagnostic metrics; no inferred scientific grade."""

from __future__ import annotations

from .data import CalphyAnalysis, CalphyDiagnostics, PhaseData


def _phase_metrics(
    phase: PhaseData | None, metrics: dict[str, float | int | None], missing: list[str]
) -> None:
    if phase is None:
        missing.append("phase result")
        return
    prefix = phase.name
    if phase.equilibration is None:
        missing.append(f"{prefix} avg.dat")
    else:
        values = phase.equilibration.values
        metrics[f"{prefix}_equilibration_blocks"] = len(values)
        if len(values) > 1:
            metrics[f"{prefix}_energy_drift_eV_atom"] = float(values[-1, 5] - values[0, 5])
            metrics[f"{prefix}_pressure_drift_bar"] = float(values[-1, 4] - values[0, 4])
    metrics[f"{prefix}_switching_replicas"] = len(phase.switching)
    metrics[f"{prefix}_temperature_scaling_replicas"] = len(phase.temperature_scaling)
    dissipation = phase.report.get("results", {}).get("dissipation")
    if dissipation is not None:
        metrics[f"{prefix}_dissipation_eV_atom"] = float(dissipation)


def calphy_diagnostics(data: CalphyAnalysis) -> CalphyDiagnostics:
    """Summarize availability, drift, replica count, and explicit log warnings."""
    metrics: dict[str, float | int | None] = {
        "attempt_count": len(data.attempts),
        "melting_temperature_k": data.melting_temperature_k,
        "uncertainty_k": data.uncertainty_k,
    }
    missing: list[str] = []
    for attempt in data.attempts:
        _phase_metrics(attempt.solid, metrics, missing)
        _phase_metrics(attempt.liquid, metrics, missing)
    warnings = tuple(
        line
        for line in data.log_records
        if any(word in line.lower() for word in ("warning", "unstable", "extrapolat", "restart"))
    )
    messages = []
    if data.melting_temperature_k is None:
        messages.append("No finite positive melting temperature was found in the retrieved log.")
    if data.uncertainty_k is None:
        messages.append("A positive finite melting-temperature uncertainty is unavailable.")
    return CalphyDiagnostics(
        metrics=metrics,
        messages=tuple(messages),
        missing_data=tuple(sorted(set(missing))),
        warnings=warnings,
        uncertainty_available=data.uncertainty_k is not None,
    )
