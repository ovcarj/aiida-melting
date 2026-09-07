"""General result-comparison figures."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from matplotlib import pyplot as plt

from .records import ResultRecord


def _label(record: ResultRecord) -> str:
    artifact = record.artifact_filename or record.artifact_sha256
    if record.artifact_filename and record.artifact_sha256:
        artifact = f"{record.artifact_filename} ({record.artifact_sha256[:8]})"
    return " / ".join(
        value for value in (record.formula, record.calculator, artifact, record.method) if value
    )


def _uncertainties(records: Iterable[ResultRecord]) -> list[float]:
    """Convert unavailable Calphy uncertainties to Matplotlib's missing value."""
    return [
        record.uncertainty_k if record.uncertainty_k is not None else np.nan for record in records
    ]


def plot_comparison(records: Iterable[ResultRecord]):
    """Plot melting temperatures grouped by material, calculator, and method."""
    records = list(records)
    figure, axis = plt.subplots(figsize=(max(6, len(records) * 1.1), 4.5))
    for index, record in enumerate(records):
        uncertainty = record.uncertainty_k if record.uncertainty_k is not None else np.nan
        axis.errorbar(index, record.melting_temperature_k, yerr=uncertainty, fmt="o")
    axis.set_xticks(
        range(len(records)), [_label(record) for record in records], rotation=35, ha="right"
    )
    axis.set_ylabel("Melting temperature (K)")
    axis.set_title("Melting-result comparison")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def plot_size_convergence(records: Iterable[ResultRecord]):
    """Plot cell-size histories, expanding embedded convergence wrapper reports."""
    grouped: dict[tuple[str | None, str | None, str | None], list[ResultRecord]] = defaultdict(list)
    for record in records:
        if record.atom_count is not None:
            grouped[(record.formula, record.calculator, record.artifact_sha256)].append(record)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for key, group in grouped.items():
        group.sort(key=lambda record: record.atom_count or 0)
        artifact = group[0].artifact_filename or group[0].artifact_sha256
        if group[0].artifact_filename and group[0].artifact_sha256:
            artifact = f"{group[0].artifact_filename} ({group[0].artifact_sha256[:8]})"
        label = " / ".join(value for value in (*key[:2], artifact) if value) or "result"
        regular = [record for record in group if not record.convergence]
        if regular:
            axis.errorbar(
                [record.atom_count for record in regular],
                [record.melting_temperature_k for record in regular],
                yerr=_uncertainties(regular),
                marker="o",
                label=label,
            )
        for record in group:
            history = (record.convergence or {}).get("tested", [])
            if not isinstance(history, list):
                continue
            valid = [point for point in history if point.get("status") != "unconverged"]
            excluded = [point for point in history if point.get("status") == "unconverged"]
            if valid:
                axis.errorbar(
                    [point.get("atom_count") for point in valid],
                    [point.get("temperature") for point in valid],
                    yerr=[point.get("uncertainty_K") or np.nan for point in valid],
                    marker="o",
                    label=label,
                )
            if excluded:
                axis.errorbar(
                    [point.get("atom_count") for point in excluded],
                    [point.get("temperature") for point in excluded],
                    yerr=[point.get("uncertainty_K") or np.nan for point in excluded],
                    fmt="x",
                    label=f"{label} (excluded)",
                )
    axis.set(
        xlabel="Prepared atom count",
        ylabel="Melting temperature (K)",
        title="Cell-size convergence",
    )
    axis.grid(alpha=0.25)
    if grouped:
        axis.legend()
    figure.tight_layout()
    return figure
