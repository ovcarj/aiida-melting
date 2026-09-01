"""Matplotlib figures for retrieved Calphy data."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from .data import AttemptData, CalphyAnalysis, PhaseData, SwitchingData


def _last_attempt(data: CalphyAnalysis) -> AttemptData:
    if not data.attempts:
        raise ValueError("No complete Calphy phase directories were retrieved.")
    return data.attempts[-1]


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_equilibration(data: CalphyAnalysis, *, rolling_window: int = 10):
    """Plot block-averaged temperature, energy, volume, and pressure."""
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    labels = (
        (7, "Temperature (K)"),
        (5, "Energy (eV/atom)"),
        (1, "Cell length (A)"),
        (4, "Pressure (bar)"),
    )
    for phase in (_last_attempt(data).solid, _last_attempt(data).liquid):
        if phase is None or phase.equilibration is None:
            continue
        values = phase.equilibration.values
        x = np.arange(len(values))
        for axis, (column, ylabel) in zip(axes.flat, labels, strict=True):
            if column >= values.shape[1]:
                continue
            axis.plot(x, values[:, column], alpha=0.35, label=f"{phase.name} blocks")
            mean = _rolling(values[:, column], rolling_window)
            axis.plot(
                np.arange(len(mean)) + len(values) - len(mean), mean, label=f"{phase.name} rolling"
            )
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Averaging block")
    axes[0, 0].legend(fontsize="small")
    figure.suptitle("Calphy equilibration block averages")
    figure.tight_layout()
    return figure


def _plot_switches(axis, records: tuple[SwitchingData, ...], title: str) -> None:
    for record in records:
        if record.integrand is None:
            continue
        axis.plot(
            record.lambda_values,
            record.integrand,
            alpha=0.65,
            label=f"{record.direction} {record.replica}",
        )
    axis.set_title(title)
    axis.set_xlabel("lambda")
    axis.set_ylabel("dU system - reference (eV/atom)")
    axis.grid(alpha=0.25)
    if axis.lines:
        axis.legend(fontsize="small")
    else:
        axis.text(
            0.5,
            0.5,
            "No single-reference integrand available",
            ha="center",
            transform=axis.transAxes,
        )


def plot_reference_switching(data: CalphyAnalysis):
    """Plot forward/backward reference-switching integrands by phase and replica."""
    attempt = _last_attempt(data)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    _plot_switches(
        axes[0], attempt.solid.switching if attempt.solid else (), "Solid reference switching"
    )
    _plot_switches(
        axes[1], attempt.liquid.switching if attempt.liquid else (), "Liquid reference switching"
    )
    figure.tight_layout()
    return figure


def plot_temperature_sweeps(data: CalphyAnalysis):
    """Plot forward/backward reversible-scaling traces as dU/lambda."""
    attempt = _last_attempt(data)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, phase in zip(axes, (attempt.solid, attempt.liquid), strict=True):
        records = phase.temperature_scaling if phase else ()
        for record in records:
            raw = record.raw.values
            denominator = record.lambda_values
            axis.plot(
                denominator, raw[:, 0] / denominator, label=f"{record.direction} {record.replica}"
            )
        axis.set_title(f"{phase.name.title() if phase else 'Missing'} temperature scaling")
        axis.set_xlabel("lambda")
        axis.set_ylabel("dU/lambda (eV/atom)")
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize="small")
    figure.tight_layout()
    return figure


def _curve(phase: PhaseData | None):
    if phase is None or phase.free_energy is None or phase.free_energy.values.shape[1] < 3:
        return None
    return phase.free_energy.values[:, :3]


def plot_free_energy_crossing(data: CalphyAnalysis):
    """Plot phase free energies, their uncertainties, and delta-G over overlap."""
    attempt = _last_attempt(data)
    solid, liquid = _curve(attempt.solid), _curve(attempt.liquid)
    if solid is None or liquid is None:
        raise ValueError("Both phase temperature_sweep.dat files are required.")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for curve, name in ((solid, "solid"), (liquid, "liquid")):
        axes[0].plot(curve[:, 0], curve[:, 1], label=name)
        axes[0].fill_between(
            curve[:, 0], curve[:, 1] - curve[:, 2], curve[:, 1] + curve[:, 2], alpha=0.2
        )
    axes[0].set(
        xlabel="Temperature (K)", ylabel="Free energy (eV/atom)", title="Free-energy curves"
    )
    axes[0].legend()
    shared = np.intersect1d(solid[:, 0], liquid[:, 0])
    if len(shared):
        solid_values = np.array([solid[solid[:, 0] == value][0] for value in shared])
        liquid_values = np.array([liquid[liquid[:, 0] == value][0] for value in shared])
        delta = liquid_values[:, 1] - solid_values[:, 1]
        error = np.hypot(liquid_values[:, 2], solid_values[:, 2])
        axes[1].plot(shared, delta, label="G liquid - G solid")
        axes[1].fill_between(shared, delta - error, delta + error, alpha=0.2)
    if data.melting_temperature_k is not None:
        for axis in axes:
            axis.axvline(
                data.melting_temperature_k, color="black", linestyle="--", label="reported Tm"
            )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(
        xlabel="Temperature (K)", ylabel="Delta G (eV/atom)", title="Free-energy difference"
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")
    figure.tight_layout()
    return figure


def plot_attempt_history(data: CalphyAnalysis):
    """Show adaptive temperature candidates and log warnings by attempt."""
    figure, axis = plt.subplots(figsize=(8, 4))
    x = np.arange(1, len(data.attempts) + 1)
    temperatures = [attempt.temperature_hint_k for attempt in data.attempts]
    axis.plot(x, temperatures, marker="o", label="candidate temperature")
    if data.melting_temperature_k is not None:
        axis.axhline(data.melting_temperature_k, color="black", linestyle="--", label="reported Tm")
    axis.set(xlabel="Calphy attempt", ylabel="Temperature (K)", title="Adaptive attempt history")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_calphy_overview(data: CalphyAnalysis):
    """Produce a compact overview of equilibration and free-energy outputs."""
    attempt = _last_attempt(data)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for phase in (attempt.solid, attempt.liquid):
        if phase is not None and phase.equilibration is not None:
            values = phase.equilibration.values
            axes[0, 0].plot(values[:, 7], label=phase.name)
            axes[0, 1].plot(values[:, 5], label=phase.name)
    axes[0, 0].set_title("Equilibration temperature")
    axes[0, 1].set_title("Equilibration energy")
    for axis, ylabel in zip(axes[0], ("K", "eV/atom"), strict=True):
        axis.set_ylabel(ylabel)
        axis.legend()
        axis.grid(alpha=0.25)
    for phase in (attempt.solid, attempt.liquid):
        curve = _curve(phase)
        if curve is not None:
            axes[1, 0].plot(curve[:, 0], curve[:, 1], label=phase.name)
    axes[1, 0].set(xlabel="Temperature (K)", ylabel="Free energy (eV/atom)", title="Free energy")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.03,
        0.85,
        f"Reported Tm: {data.melting_temperature_k!s} K\nUncertainty: {data.uncertainty_k!s} K\nAttempts: {len(data.attempts)}",
        va="top",
    )
    figure.tight_layout()
    return figure
