"""Unit tests for Calphy adapters, validation, and structure preparation."""

import math

import pytest
from aiida import orm
from aiida.common.exceptions import ValidationError

from aiida_melting.calculators import EamCalculatorAdapter, MaceCalculatorAdapter
from aiida_melting.calphy import (
    GPU_LAMMPS_CMDARGS,
    is_transient_calphy_failure,
    is_transient_transport_exception,
    parse_temperature_log,
    pressure_gpa_to_bar,
    validate_lammps_cmdargs,
)
from aiida_melting.structures_calphy import prepare_supercell, validate_calphy_structure


def calculator(name, **implementation):
    return orm.Dict(
        dict={
            "name": name,
            "provides": ["energy", "forces", "stress"],
            "metadata": implementation,
        }
    )


@pytest.mark.usefixtures("aiida_profile_clean")
def test_eam_adapter():
    artifact = orm.SinglefileData.from_bytes(b"eam", filename="Cu.eam.alloy")
    spec, returned = EamCalculatorAdapter.translate(
        calculator("eam", pair_style="eam/alloy", elements=["Cu"]),
        {"potential": artifact},
        ["Cu"],
    )
    assert returned is artifact
    assert spec.pair_style == "eam/alloy"
    assert spec.pair_coeff == "* * Cu.eam.alloy Cu"


@pytest.mark.usefixtures("aiida_profile_clean")
@pytest.mark.parametrize(
    ("metadata", "files", "message"),
    [
        ({"pair_style": "eam/alloy", "elements": ["Cu"]}, {}, "potential"),
        ({"pair_style": "lj/cut", "elements": ["Cu"]}, {"potential": "file"}, "pair_style"),
        (
            {"pair_style": "eam/alloy", "elements": ["Al"]},
            {"potential": "file"},
            "element mapping",
        ),
    ],
)
def test_eam_adapter_rejections(metadata, files, message):
    converted = {
        key: orm.SinglefileData.from_bytes(b"eam", filename="potential.eam") for key in files
    }
    with pytest.raises(ValidationError, match=message):
        EamCalculatorAdapter.translate(calculator("eam", **metadata), converted, ["Cu"])


@pytest.mark.usefixtures("aiida_profile_clean")
def test_mace_adapter():
    artifact = orm.SinglefileData.from_bytes(b"model", filename="mace-medium-lammps.pt")
    spec, returned = MaceCalculatorAdapter.translate(
        calculator("mace", model_format="mace-lammps", elements=["Cu"]),
        {"model": artifact},
        ["Cu"],
    )
    assert returned is artifact
    assert spec.pair_style == "mliap unified mace-medium-lammps.pt 0"
    assert spec.pair_coeff == "* * Cu"


@pytest.mark.usefixtures("aiida_profile_clean")
@pytest.mark.parametrize(
    ("filename", "model_format"),
    [("raw.model", "mace-lammps"), ("model-lammps.pt", "torchscript")],
)
def test_mace_adapter_rejects_unready_model(filename, model_format):
    artifact = orm.SinglefileData.from_bytes(b"model", filename=filename)
    with pytest.raises(ValidationError):
        MaceCalculatorAdapter.translate(
            calculator("mace", model_format=model_format, elements=["Cu"]),
            {"model": artifact},
            ["Cu"],
        )


def test_pressure_and_arguments():
    assert pressure_gpa_to_bar(1.25) == 12500
    assert validate_lammps_cmdargs([]) == ""
    assert validate_lammps_cmdargs(GPU_LAMMPS_CMDARGS) == "-k on g 1 -sf kk"
    for invalid in (["-in", "job"], ["-k", "on", "g", "2", "-sf", "kk"], ["x y"], [";"]):
        with pytest.raises(ValidationError):
            validate_lammps_cmdargs(invalid)


def test_temperature_parser_semantics():
    parsed = parse_temperature_log("STATE: Tm = 1350.5 K +/- 2.5 K", 2)
    assert parsed["temperature"] == 1350.5
    assert parsed["uncertainty"] == 2.5
    assert parsed["status"] == "success"
    one_iteration = parse_temperature_log("STATE: Tm = 1350 K +/- 0 K", 1)
    assert one_iteration["uncertainty"] is None
    assert one_iteration["status"] == "success"
    warning = parse_temperature_log(
        "STATE: Tm unreliable, sweep dissipation too high\nSTATE: Tm = 1350 K +/- 3 K", 2
    )
    assert warning["status"] == "ambiguous"
    for value in ("nan", "inf", "-1", "0"):
        with pytest.raises(ArithmeticError):
            parse_temperature_log(f"STATE: Tm = {value} K +/- 2 K", 2)
    with pytest.raises(ValueError):
        parse_temperature_log("calculation finished", 2)


def test_restart_classification_is_closed():
    assert all(is_transient_calphy_failure(status) for status in (100, 110, 120, 140, 301))
    assert not any(is_transient_calphy_failure(status) for status in (None, 0, 131, 302, 303, 304))
    assert is_transient_transport_exception("TransportTaskException: connection lost")
    assert not is_transient_transport_exception("TypeError: plugin bug")


def cu_structure():
    structure = orm.StructureData(cell=[[3.6, 0, 0], [0, 3.6, 0], [0, 0, 3.6]])
    structure.append_atom(position=(0, 0, 0), symbols="Cu")
    return structure


@pytest.mark.usefixtures("aiida_profile_clean")
def test_prepare_supercell_is_deterministic():
    first = prepare_supercell(cu_structure(), orm.List(list=[2, 1, 1]))
    second = prepare_supercell(cu_structure(), orm.List(list=[2, 1, 1]))
    assert len(first["structure"].sites) == 2
    assert first["data"].get_content() == second["data"].get_content()
    assert "2 atoms" in first["data"].get_content()


@pytest.mark.usefixtures("aiida_profile_clean")
def test_invalid_structures():
    nonperiodic = cu_structure()
    nonperiodic.pbc = (True, True, False)
    with pytest.raises(ValueError, match="periodic"):
        validate_calphy_structure(nonperiodic)
    singular = orm.StructureData(cell=[[1, 0, 0], [0, 1, 0], [0, 0, 0]], pbc=True)
    singular.append_atom(position=(0, 0, 0), symbols="Cu")
    with pytest.raises(ValueError, match="nonsingular"):
        validate_calphy_structure(singular)
    mixed = orm.StructureData(cell=[[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    mixed.append_atom(position=(0, 0, 0), symbols=("Cu", "Al"), weights=(0.5, 0.5))
    with pytest.raises(ValueError, match="mixed"):
        validate_calphy_structure(mixed)
    assert math.isfinite(abs(__import__("numpy").linalg.det(cu_structure().cell)))
