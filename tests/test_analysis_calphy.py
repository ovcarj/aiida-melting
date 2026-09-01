"""Tests for read-only Calphy analysis objects."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from aiida_melting.analysis.calphy.diagnostics import calphy_diagnostics
from aiida_melting.analysis.calphy.plots import (
    plot_attempt_history,
    plot_calphy_overview,
    plot_equilibration,
    plot_free_energy_crossing,
    plot_reference_switching,
    plot_temperature_sweeps,
)
from aiida_melting.analysis.calphy.reader import _table, read_calphy_directory


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


def test_reader_uses_native_calphy_column_header() -> None:
    table = _table(Path(__file__).parent / "fixtures" / "calphy_avg.dat")
    assert table is not None
    assert table.columns == (
        "TimeStep",
        "lx[A]",
        "ly[A]",
        "lz[A]",
        "press[bar]",
        "pe[eV/atom]",
        "etotal[eV/atom]",
        "temp[K]",
    )


def test_reader_orders_attempts_from_generated_input_files(tmp_path: Path) -> None:
    _write(
        tmp_path / "melting.log",
        "STATE: Temperature range of 1000-1200 K\nSTATE: Temperature range of 1400-1600 K\n",
    )
    for index, temperature in enumerate((1000, 1400)):
        for phase in ("solid", "liquid"):
            directory = tmp_path / f"ts-structure.data-{phase}-{temperature}--10"
            _write(
                directory / "input_file.yaml",
                "calculations:\n"
                f"- inputfile: attempt.{index}.yaml\n"
                f"  temperature: [{temperature}, {temperature + 200}]\n",
            )

    result = read_calphy_directory(tmp_path)

    assert [attempt.key for attempt in result.attempts] == ["attempt.0.yaml", "attempt.1.yaml"]
    assert [attempt.temperature_range_k for attempt in result.attempts] == [
        (1000.0, 1200.0),
        (1400.0, 1600.0),
    ]
    assert result.attempts[-1].solid is not None
    assert result.attempts[-1].solid.directory.endswith("solid-1400--10")


def test_retrieved_style_sources_are_relative_to_analysis_root(tmp_path: Path) -> None:
    directory = tmp_path / "ts-structure.data-solid-1000-0"
    _write(directory / "avg.dat", "# columns\n1 2\n")
    result = read_calphy_directory(tmp_path)
    assert result.attempts[0].solid is not None
    assert result.attempts[0].solid.equilibration is not None
    assert result.attempts[0].solid.equilibration.source == "ts-structure.data-solid-1000-0/avg.dat"


def test_all_calphy_plots_render_for_complete_data(tmp_path: Path) -> None:
    _write(tmp_path / "melting.log", "STATE: Tm = 1354.5 K +/- 4.2 K\n")
    for phase in ("solid", "liquid"):
        directory = tmp_path / f"ts-structure.data-{phase}-1300-0"
        _write(directory / "avg.dat", "# a b c d e f g h\n1 2 3 4 5 6 7 8\n")
        _write(directory / "forward_1.dat", "# dU_sys dU_ref lambda\n-3 1 0\n-2 2 1\n")
        _write(directory / "ts.forward_1.dat", "# dU press vol lambda\n-3 1 2 0.5\n")
        _write(
            directory / "temperature_sweep.dat",
            "# temperature free_energy error\n1300 -3.0 0.01\n1400 -2.9 0.02\n",
        )
    result = read_calphy_directory(tmp_path)
    for name, plotter in (
        ("equilibration", plot_equilibration),
        ("switching", plot_reference_switching),
        ("sweeps", plot_temperature_sweeps),
        ("crossing", plot_free_energy_crossing),
        ("attempts", plot_attempt_history),
        ("overview", plot_calphy_overview),
    ):
        output = tmp_path / f"{name}.png"
        figure = plotter(result)
        figure.savefig(output)
        assert output.stat().st_size > 0
