"""Direct executable Calphy CalcJob integration."""

from __future__ import annotations

from copy import deepcopy
from typing import ClassVar

import yaml
from aiida import orm
from aiida.common import CalcInfo, CodeInfo
from aiida.common.exceptions import InputValidationError
from aiida.engine import CalcJob

from ..calculators import RESERVED_ARTIFACT_FILENAMES
from ..calphy import validate_lammps_cmdargs


class CalphyCalculation(CalcJob):
    """Run one complete Calphy adaptive melting calculation."""

    INPUT_FILE = "input.yaml"
    STRUCTURE_FILE = "structure.data"
    STDOUT_FILE = "calphy.stdout"
    STDERR_FILE = "calphy.stderr"
    RETRIEVE_LIST: ClassVar[list] = [
        "calphy.stdout",
        "calphy.stderr",
        "*.log",
        "*.yaml",
        "*.sub",
        "*.out",
        "*.err",
        ("*/report.yaml", ".", 2),
        ("*/input*.yaml", ".", 2),
        ("*/*.log", ".", 2),
        ("*/*.dat", ".", 2),
        ("*/*.sub", ".", 2),
    ]

    @classmethod
    def define(cls, spec) -> None:
        super().define(spec)
        spec.input("code", valid_type=orm.InstalledCode)
        spec.input("lammps_code", valid_type=orm.InstalledCode)
        spec.input("structure_data", valid_type=orm.SinglefileData)
        spec.input("potential", valid_type=orm.SinglefileData)
        spec.input("parameters", valid_type=orm.Dict)
        spec.input("lammps_cmdargs", valid_type=orm.List, default=lambda: orm.List(list=[]))
        spec.input(
            "metadata.options.resources",
            valid_type=dict,
            default=lambda: {"num_machines": 1, "num_mpiprocs_per_machine": 1},
        )
        spec.input("metadata.options.withmpi", valid_type=bool, default=False)
        spec.input("metadata.options.parser_name", valid_type=str, default="melting.calphy")
        spec.output("melting_temperature", valid_type=orm.Float)
        spec.output("status", valid_type=orm.Str)
        spec.output("uncertainty", valid_type=orm.Float, required=False)
        spec.output("calculation_metadata", valid_type=orm.Dict)
        spec.output("diagnostics", valid_type=orm.Dict)
        spec.output("retrieved_files", valid_type=orm.List)
        spec.exit_code(300, "ERROR_CALPHY_EXECUTION_FAILED", message="Calphy execution failed")
        spec.exit_code(
            301, "ERROR_INCOMPLETE_RETRIEVAL", message="Required Calphy output was not retrieved"
        )
        spec.exit_code(
            302, "ERROR_MALFORMED_TEMPERATURE", message="Malformed Calphy temperature output"
        )
        spec.exit_code(
            303, "ERROR_INVALID_TEMPERATURE", message="Invalid Calphy temperature output"
        )
        spec.exit_code(
            304, "ERROR_PARSER_CORRUPTION", message="Calphy parser could not read retrieved output"
        )
        spec.exit_code(305, "ERROR_CALPHY_INPUT_REJECTED", message="Calphy rejected its input")
        spec.exit_code(
            306,
            "ERROR_LAMMPS_STYLE_UNAVAILABLE",
            message="The configured LAMMPS executable lacks a required style",
        )
        spec.exit_code(307, "ERROR_LAMMPS_RUNTIME_FAILED", message="LAMMPS execution failed")
        spec.exit_code(
            308,
            "ERROR_MELTING_ATTEMPTS_EXHAUSTED",
            message="Calphy exhausted its melting-temperature attempts",
        )
        spec.inputs.validator = cls.validate_inputs

    @staticmethod
    def validate_inputs(inputs, _ctx=None):
        calphy_code = inputs.get("code")
        lammps_code = inputs.get("lammps_code")
        if calphy_code is not None and lammps_code is not None:
            if not isinstance(calphy_code, orm.InstalledCode) or not isinstance(
                lammps_code, orm.InstalledCode
            ):
                return "calphy_code and lammps_code must be InstalledCode nodes"
            if calphy_code.computer.uuid != lammps_code.computer.uuid:
                return "calphy_code and lammps_code must be installed on the same Computer"
            if not calphy_code.filepath_executable:
                return "calphy_code must define an executable path"
            if not lammps_code.filepath_executable:
                return "lammps_code must define an executable path"
        if (
            inputs.get("structure_data") is not None
            and inputs["structure_data"].filename != "structure.data"
        ):
            return "structure_data must be named structure.data"
        if (
            inputs.get("potential") is not None
            and inputs["potential"].filename in RESERVED_ARTIFACT_FILENAMES
        ):
            return f"potential filename {inputs['potential'].filename!r} is reserved"
        try:
            if inputs.get("lammps_cmdargs") is not None:
                validate_lammps_cmdargs(inputs["lammps_cmdargs"].get_list())
        except Exception as exception:
            return str(exception)
        return None

    def prepare_for_submission(self, folder) -> CalcInfo:
        validation_error = self.validate_inputs(self.inputs)
        if validation_error:
            raise InputValidationError(validation_error)

        parameters = deepcopy(self.inputs.parameters.get_dict())
        calculation = parameters["calculations"][0]
        calculation["lammps_executable"] = str(self.inputs.lammps_code.filepath_executable)
        calculation.setdefault("md", {})["cmdargs"] = validate_lammps_cmdargs(
            self.inputs.lammps_cmdargs.get_list()
        )
        with folder.open(self.INPUT_FILE, "w", encoding="utf8") as handle:
            yaml.safe_dump(parameters, handle, sort_keys=True)

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = ["-i", self.INPUT_FILE, "-k", "0"]
        codeinfo.withmpi = False
        codeinfo.stdout_name = self.STDOUT_FILE
        codeinfo.stderr_name = self.STDERR_FILE

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = [
            (
                self.inputs.structure_data.uuid,
                self.inputs.structure_data.filename,
                self.STRUCTURE_FILE,
            ),
            (
                self.inputs.potential.uuid,
                self.inputs.potential.filename,
                self.inputs.potential.filename,
            ),
        ]
        calcinfo.retrieve_list = list(self.RETRIEVE_LIST)
        return calcinfo
