"""Tests for read-only Calphy analysis objects."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiida_melting.analysis.calphy.diagnostics import calphy_diagnostics
from aiida_melting.analysis.calphy.reader import read_calphy_directory


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_reader_and_diagnostics_for_single_attempt(tmp_path: Path) -> None:
    _write(tmp_path / "melting.log", "STATE: Tm = 1354.5 K +/- 4.2 K\n")
    _write(tmp_path / "input.yaml", "n_iterations: 2\n")
    for phase in ("solid", "liquid"):
        directory = tmp_path / f"ts-structure.data-{phase}-1350-0"
        _write(
            directory / "avg.dat",
            "# TimeStep lx ly lz press pe etotal temp\n"
            "10 3 3 3 2 -3.1 -3.0 1340\n"
            "20 3 3 3 4 -3.0 -2.9 1360\n",
        )
        _write(
            directory / "forward_1.dat",
            "# dU_sys dU_ref1 lambda\n-3 1 0\n-2 2 1\n",
        )
        _write(
            directory / "ts.forward_1.dat",
            "# dU press vol lambda\n-3 1 2 0.5\n",
        )
        _write(
            directory / "temperature_sweep.dat",
            "# temperature free_energy error\n1300 -3.0 0.01\n1400 -2.9 0.02\n",
        )
        _write(directory / "report.yaml", "results:\n  dissipation: 0.02\n")

    result = read_calphy_directory(tmp_path)
    diagnostics = calphy_diagnostics(result)

    assert result.melting_temperature_k == 1354.5
    assert result.uncertainty_k == 4.2
    assert len(result.attempts) == 1
    assert len(result.attempts[0].solid.switching) == 1
    assert len(result.attempts[0].solid.temperature_scaling) == 1
    assert diagnostics.uncertainty_available
    assert diagnostics.metrics["solid_energy_drift_eV_atom"] == pytest.approx(0.1)


def test_reader_treats_nonpositive_uncertainty_as_unavailable(tmp_path: Path) -> None:
    _write(tmp_path / "melting.log", "STATE: Tm = 1354.5 K +/- 0 K\n")
    result = read_calphy_directory(tmp_path)
    assert result.melting_temperature_k == 1354.5
    assert result.uncertainty_k is None
    assert not calphy_diagnostics(result).uncertainty_available
