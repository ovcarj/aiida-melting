"""Regression tests for general analysis figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from aiida_melting.analysis.plots import plot_comparison, plot_size_convergence
from aiida_melting.analysis.records import ResultRecord


def _record(atom_count: int, temperature: float, uncertainty: float | None) -> ResultRecord:
    return ResultRecord(
        process_pk=atom_count,
        process_uuid=f"uuid-{atom_count}",
        ctime="2026-09-01T00:00:00+00:00",
        formula="Cu",
        composition={"Cu": 1},
        elements=("Cu",),
        pressure_gpa=0.0,
        method="melting.calphy",
        calculator="eam",
        artifact_filename="Cu.eam.alloy",
        artifact_sha256="potential-hash",
        structure_uuid="structure-uuid",
        structure_hash="structure-hash",
        structure_label=None,
        material_id=None,
        prepared_structure_uuid="prepared-uuid",
        atom_count=atom_count,
        melting_temperature_k=temperature,
        uncertainty_k=uncertainty,
        scientific_status="success",
        report_uuid="report-uuid",
    )


def test_general_plots_allow_unavailable_uncertainty(tmp_path) -> None:
    """One-iteration Calphy results have no uncertainty and must still plot."""
    records = [_record(500, 1355.0, None), _record(2048, 1332.0, 3.0)]
    for name, plotter in (("comparison", plot_comparison), ("size", plot_size_convergence)):
        output = tmp_path / f"{name}.png"
        figure = plotter(records)
        figure.savefig(output)
        assert output.stat().st_size > 0
