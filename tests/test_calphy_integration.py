"""Opt-in tests for configured real Calphy infrastructure."""

import json
import os
from pathlib import Path

import pytest
from aiida import orm
from aiida.engine import run_get_node

from aiida_melting.workflows.dispatcher import MeltingWorkChain

pytestmark = pytest.mark.integration


def _configuration(artifact_variable):
    required = {
        "calphy_code": os.getenv("AIIDA_MELTING_CALPHY_CODE"),
        "lammps_code": os.getenv("AIIDA_MELTING_LAMMPS_CODE"),
        "artifact": os.getenv(artifact_variable),
    }
    if missing := [key for key, value in required.items() if not value]:
        pytest.skip("real integration variables are unset: " + ", ".join(missing))
    return required


def _structure():
    structure = orm.StructureData(cell=[[3.615, 0, 0], [0, 3.615, 0], [0, 0, 3.615]])
    for position in ((0, 0, 0), (0, 1.8075, 1.8075), (1.8075, 0, 1.8075), (1.8075, 1.8075, 0)):
        structure.append_atom(position=position, symbols="Cu")
    return structure


def _run(configuration, calculator, cmdargs):
    scheduler_options = json.loads(os.getenv("AIIDA_MELTING_SCHEDULER_OPTIONS", "{}"))
    results, node = run_get_node(
        MeltingWorkChain,
        composition=orm.Dict(dict={"Cu": 1}),
        pressure=orm.Float(0),
        calculator=calculator,
        structure=_structure(),
        method=orm.Str("calphy"),
        method_parameters={
            "calphy_code": orm.load_code(configuration["calphy_code"]),
            "lammps_code": orm.load_code(configuration["lammps_code"]),
            "temperature_guess": orm.Float(1300),
            "seed": orm.Int(24681357),
            "lammps_cmdargs": orm.List(list=cmdargs),
            "scheduler_options": orm.Dict(dict=scheduler_options),
        },
    )
    assert node.is_finished_ok, node.exit_message
    assert results["melting_temperature"].value > 0
    assert results["status"].value in {"success", "ambiguous"}
    report = results["report"].get_dict()
    assert report["child_processes"]["calcjobs"]
    assert report["retrieved_files"]
    return results


def test_real_cu_eam_twice():
    configuration = _configuration("AIIDA_MELTING_EAM_POTENTIAL")
    artifact = orm.SinglefileData(file=Path(configuration["artifact"]))
    calculator = {
        "metadata": orm.Dict(
            dict={
                "name": "eam",
                "provides": ["energy", "forces", "stress"],
                "metadata": {"pair_style": "eam/alloy", "elements": ["Cu"]},
            }
        ),
        "files": {"potential": artifact},
    }
    first = _run(configuration, calculator, [])
    second = _run(configuration, calculator, [])
    first_report = first["report"].get_dict()
    second_report = second["report"].get_dict()
    uncertainties = [
        first_report["diagnostics"]["uncertainty_K"],
        second_report["diagnostics"]["uncertainty_K"],
    ]
    tolerance = float(os.getenv("AIIDA_MELTING_REPRODUCIBILITY_TOLERANCE_K", "10"))
    if all(value is not None for value in uncertainties):
        tolerance = max(tolerance, sum(uncertainties))
    assert (
        abs(first["melting_temperature"].value - second["melting_temperature"].value) <= tolerance
    )


def test_real_cu_mace_mpa_0_medium_one_gpu():
    configuration = _configuration("AIIDA_MELTING_MACE_MPA_0_MEDIUM_MODEL")
    artifact = orm.SinglefileData(file=Path(configuration["artifact"]))
    assert artifact.filename.endswith("lammps.pt")
    calculator = {
        "metadata": orm.Dict(
            dict={
                "name": "mace",
                "provides": ["energy", "forces", "stress"],
                "metadata": {"model_format": "mace-lammps", "elements": ["Cu"]},
            }
        ),
        "files": {"model": artifact},
    }
    _run(configuration, calculator, ["-k", "on", "g", "1", "-sf", "kk"])
