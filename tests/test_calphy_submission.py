"""Dry-run level tests for exact Calphy submission files."""

import yaml
from aiida import orm
from aiida.common.folders import SandboxFolder
from aiida.engine.utils import instantiate_process

from aiida_melting.calculations.calphy import CalphyCalculation


def make_code(computer, label, executable):
    return orm.InstalledCode(computer=computer, label=label, filepath_executable=executable).store()


def parameters(pair_style="eam/alloy", pair_coeff="* * Cu.eam Cu"):
    return {
        "calculations": [
            {
                "mode": "melting_temperature",
                "lattice": "structure.data",
                "file_format": "lammps-data",
                "element": ["Cu"],
                "mass": [63.546],
                "repeat": [1, 1, 1],
                "pressure": 0.0,
                "pair_style": pair_style,
                "pair_coeff": pair_coeff,
                "execution_mode": "executable",
                "queue": {"scheduler": "local", "cores": 1, "commands": []},
                "md": {"timestep": 0.001, "seed": 42, "cmdargs": ""},
                "n_equilibration_steps": 10000,
                "n_switching_steps": 25000,
                "n_iterations": 2,
                "equilibration_control": "berendsen",
                "melting_temperature": {"guess": 1350.0, "step": 400.0, "attempts": 5},
            }
        ]
    }


def prepare(aiida_manager, aiida_localhost, cmdargs=None, mace=False):
    calphy = make_code(aiida_localhost, "calphy", "/opt/calphy/bin/calphy_kernel")
    lammps = make_code(aiida_localhost, "lammps", "/opt/lammps/bin/lmp")
    structure = orm.SinglefileData.from_bytes(b"LAMMPS data", filename="structure.data")
    if mace:
        potential = orm.SinglefileData.from_bytes(b"model", filename="mace-medium-lammps.pt")
        raw = parameters("mliap unified mace-medium-lammps.pt 0", "* * Cu")
    else:
        potential = orm.SinglefileData.from_bytes(b"eam", filename="Cu.eam")
        raw = parameters()
    process = instantiate_process(
        aiida_manager.get_runner(),
        CalphyCalculation,
        code=calphy,
        lammps_code=lammps,
        structure_data=structure,
        potential=potential,
        parameters=orm.Dict(dict=raw),
        lammps_cmdargs=orm.List(list=cmdargs or []),
        metadata={"dry_run": True},
    )
    folder = SandboxFolder()
    calcinfo = process.prepare_for_submission(folder)
    return folder, calcinfo, calphy, lammps, structure, potential


def test_cpu_submission(aiida_profile_clean, aiida_manager, aiida_localhost):
    folder, calcinfo, calphy, lammps, structure, potential = prepare(aiida_manager, aiida_localhost)
    with folder.open("input.yaml") as handle:
        data = yaml.safe_load(handle)
    calculation = data["calculations"][0]
    assert calculation["lammps_executable"] == str(lammps.filepath_executable)
    assert calculation["queue"] == {"scheduler": "local", "cores": 1, "commands": []}
    assert calculation["md"]["cmdargs"] == ""
    codeinfo = calcinfo.codes_info[0]
    assert codeinfo.code_uuid == calphy.uuid
    assert codeinfo.cmdline_params == ["-i", "input.yaml", "-k", "0"]
    assert codeinfo.withmpi is False
    assert (structure.uuid, "structure.data", "structure.data") in calcinfo.local_copy_list
    assert (potential.uuid, "Cu.eam", "Cu.eam") in calcinfo.local_copy_list
    assert "*.log" in calcinfo.retrieve_list
    assert not any(
        "restart" in str(item) or "trajectory" in str(item) for item in calcinfo.retrieve_list
    )


def test_one_gpu_mace_submission(aiida_profile_clean, aiida_manager, aiida_localhost):
    folder, calcinfo, _calphy, _lammps, _structure, potential = prepare(
        aiida_manager,
        aiida_localhost,
        cmdargs=["-k", "on", "g", "1", "-sf", "kk"],
        mace=True,
    )
    with folder.open("input.yaml") as handle:
        calculation = yaml.safe_load(handle)["calculations"][0]
    assert calculation["md"]["cmdargs"] == "-k on g 1 -sf kk"
    assert calculation["pair_style"] == "mliap unified mace-medium-lammps.pt 0"
    assert calculation["pair_coeff"] == "* * Cu"
    assert (potential.uuid, potential.filename, potential.filename) in calcinfo.local_copy_list
