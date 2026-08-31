"""Public Calphy melting workflow and internal restart wrapper."""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from typing import ClassVar

from aiida import orm
from aiida.common.exceptions import ValidationError
from aiida.engine import (
    BaseRestartWorkChain,
    ProcessHandlerReport,
    ToContext,
    calcfunction,
    process_handler,
    while_,
)

from ..calculations.calphy import CalphyCalculation
from ..calculators import MaceCalculatorAdapter, get_calculator_adapter
from ..calphy import (
    MLIAP_GPU_LAMMPS_CMDARGS,
    is_transient_calphy_failure,
    is_transient_transport_exception,
    pressure_gpa_to_bar,
    validate_lammps_cmdargs,
)
from ..contracts import (
    BaseMeltingWorkChain,
    normalize_composition,
    validate_calculator,
    validate_pressure,
    validate_structure_composition,
)
from ..structures_calphy import prepare_supercell, validate_calphy_structure

METHOD_VERSION = "1"
TARGET_CALPHY_VERSION = "2.0.1"


def _positive(value, name: str) -> float:
    result = float(value.value)
    if not math.isfinite(result) or result <= 0:
        raise ValidationError(f"method_parameters.{name} must be finite and positive")
    return result


def _positive_int(value, name: str, *, allow_zero: bool = False) -> int:
    result = int(value.value)
    minimum = 0 if allow_zero else 1
    if result < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValidationError(f"method_parameters.{name} must be {qualifier}")
    return result


def _artifact_sha256(artifact: orm.SinglefileData) -> str:
    digest = hashlib.sha256()
    with artifact.open(mode="rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@calcfunction
def create_calphy_report(
    composition: orm.Dict,
    pressure: orm.Float,
    calculator_metadata: orm.Dict,
    input_structure: orm.StructureData,
    prepared_structure: orm.StructureData,
    artifact: orm.SinglefileData,
    calphy_code: orm.InstalledCode,
    lammps_code: orm.InstalledCode,
    protocol: orm.Dict,
    calculation_metadata: orm.Dict,
    diagnostics: orm.Dict,
    retrieved_files: orm.List,
    process_identifiers: orm.Dict,
) -> orm.Dict:
    """Build the extended common report with explicit data/code dependencies."""
    calculator = calculator_metadata.get_dict()
    implementation = calculator["metadata"]
    diag = diagnostics.get_dict()
    return orm.Dict(
        dict={
            "method": "melting.calphy",
            "method_version": METHOD_VERSION,
            "target_calphy_version": TARGET_CALPHY_VERSION,
            "units": {"melting_temperature": "K", "pressure": "GPa"},
            "composition": normalize_composition(composition),
            "pressure": pressure.value,
            "pressure_bar": pressure_gpa_to_bar(pressure.value),
            "input_structure_uuid": input_structure.uuid,
            "prepared_structure_uuid": prepared_structure.uuid,
            "supercell": protocol["supercell"],
            "seed": protocol["seed"],
            "protocol": protocol.get_dict(),
            "calculation_metadata": calculation_metadata.get_dict(),
            "calculator": {
                "name": calculator["name"],
                "metadata": implementation,
                "artifact_key": protocol["artifact_key"],
                "artifact_uuid": artifact.uuid,
                "artifact_filename": artifact.filename,
                "artifact_sha256": _artifact_sha256(artifact),
                "elements": protocol["elements"],
            },
            "codes": {
                "calphy": {
                    "uuid": calphy_code.uuid,
                    "executable": str(calphy_code.filepath_executable),
                },
                "lammps": {
                    "uuid": lammps_code.uuid,
                    "executable": str(lammps_code.filepath_executable),
                    "prepend_append_scripts_applied": False,
                },
            },
            "diagnostics": diag,
            "uncertainty_available": diag["uncertainty_available"],
            "child_processes": process_identifiers.get_dict(),
            "retrieved_files": retrieved_files.get_list(),
        }
    )


class CalphyCalculationWorkChain(BaseRestartWorkChain):
    """Retry a complete Calphy job only after a classified transient failure."""

    _process_class = CalphyCalculation
    _terminal_exit_labels: ClassVar[dict[int, str]] = {
        300: "ERROR_CALPHY_EXECUTION_FAILED",
        302: "ERROR_MALFORMED_TEMPERATURE",
        303: "ERROR_INVALID_TEMPERATURE",
        304: "ERROR_PARSER_CORRUPTION",
        305: "ERROR_CALPHY_INPUT_REJECTED",
        306: "ERROR_LAMMPS_STYLE_UNAVAILABLE",
        307: "ERROR_LAMMPS_RUNTIME_FAILED",
        308: "ERROR_MELTING_ATTEMPTS_EXHAUSTED",
    }

    @classmethod
    def define(cls, spec) -> None:
        super().define(spec)
        spec.expose_inputs(CalphyCalculation, namespace="calculation")
        spec.expose_outputs(CalphyCalculation)
        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(cls.run_process, cls.inspect_process),
            cls.results,
        )
        spec.exit_code(300, "ERROR_CALPHY_EXECUTION_FAILED", message="Calphy execution failed")
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

    def setup(self) -> None:
        super().setup()
        self.ctx.inputs = self.exposed_inputs(CalphyCalculation, namespace="calculation")

    @process_handler(priority=600)
    def handle_transient_failure(self, node):
        """Repeat from input.yaml with the identical seed; never continue remote state."""
        if node.is_failed and is_transient_calphy_failure(node.exit_status):
            return ProcessHandlerReport(do_break=True)
        return None

    def inspect_process(self):
        """Retry recognizable transport exceptions; leave all other exceptions terminal."""
        node = self.ctx.children[self.ctx.iteration - 1]
        if node.is_excepted and is_transient_transport_exception(node.exception):
            if self.ctx.iteration >= self.inputs.max_iterations.value:
                return self.exit_codes.ERROR_MAXIMUM_ITERATIONS_EXCEEDED
            self.report(f"retrying transient transport failure from {node.pk}")
            return None
        if node.is_failed and node.exit_status in self._terminal_exit_labels:
            label = self._terminal_exit_labels[node.exit_status]
            return getattr(self.exit_codes, label)
        return super().inspect_process()


class CalphyMeltingWorkChain(BaseMeltingWorkChain):
    """Prepare, execute, parse, and report a Calphy melting calculation."""

    @classmethod
    def define(cls, spec) -> None:
        super().define(spec)
        namespace = "method_parameters."
        spec.input(namespace + "calphy_code", valid_type=orm.InstalledCode)
        spec.input(namespace + "lammps_code", valid_type=orm.InstalledCode)
        spec.input(namespace + "temperature_guess", valid_type=orm.Float)
        spec.input(namespace + "seed", valid_type=orm.Int)
        spec.input(
            namespace + "supercell", valid_type=orm.List, default=lambda: orm.List(list=[1, 1, 1])
        )
        spec.input(
            namespace + "temperature_step", valid_type=orm.Float, default=lambda: orm.Float(400.0)
        )
        spec.input(namespace + "max_attempts", valid_type=orm.Int, default=lambda: orm.Int(5))
        spec.input(namespace + "n_iterations", valid_type=orm.Int, default=lambda: orm.Int(2))
        spec.input(
            namespace + "n_equilibration_steps", valid_type=orm.Int, default=lambda: orm.Int(10000)
        )
        spec.input(
            namespace + "n_switching_steps", valid_type=orm.Int, default=lambda: orm.Int(25000)
        )
        spec.input(namespace + "timestep", valid_type=orm.Float, default=lambda: orm.Float(0.001))
        spec.input(
            namespace + "equilibration_control",
            valid_type=orm.Str,
            default=lambda: orm.Str("berendsen"),
        )
        spec.input(namespace + "md", valid_type=orm.Dict, required=False)
        spec.input(namespace + "tolerance", valid_type=orm.Dict, required=False)
        spec.input(namespace + "scheduler_options", valid_type=orm.Dict, required=False)
        spec.input(
            namespace + "lammps_cmdargs", valid_type=orm.List, default=lambda: orm.List(list=[])
        )
        spec.input(namespace + "max_restarts", valid_type=orm.Int, default=lambda: orm.Int(0))
        spec.outline(
            cls.validate_inputs,
            cls.prepare_structure,
            cls.submit_calculation,
            cls.inspect_calculation,
            cls.finalize,
        )
        spec.exit_code(201, "ERROR_INVALID_INPUT", message="Invalid Calphy input: {reason}")
        spec.exit_code(
            301, "ERROR_CALCULATION_FAILED", message="Calphy calculation failed: {reason}"
        )
        spec.exit_code(302, "ERROR_MALFORMED_RESULTS", message="Malformed Calphy results: {reason}")

    def validate_inputs(self):
        try:
            if "structure" not in self.inputs or not isinstance(
                self.inputs.structure, orm.StructureData
            ):
                raise ValidationError("an explicit resolved StructureData is required")
            self.ctx.composition = normalize_composition(self.inputs.composition)
            self.ctx.pressure = validate_pressure(self.inputs.pressure)
            calculator = validate_calculator(self.inputs.calculator.metadata)
            validate_structure_composition(self.inputs.structure, self.ctx.composition)
            validate_calphy_structure(self.inputs.structure)
            params = self.inputs.method_parameters
            if params.calphy_code.computer.uuid != params.lammps_code.computer.uuid:
                raise ValidationError("calphy_code and lammps_code must be on the same Computer")
            if not params.calphy_code.filepath_executable:
                raise ValidationError("calphy_code must define an executable path")
            if not params.lammps_code.filepath_executable:
                raise ValidationError("lammps_code must define an executable path")
            supercell = params.supercell.get_list()
            if len(supercell) != 3 or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in supercell
            ):
                raise ValidationError("supercell must contain exactly three positive integers")
            for name in ("temperature_guess", "temperature_step", "timestep"):
                _positive(params[name], name)
            for name in (
                "seed",
                "max_attempts",
                "n_iterations",
                "n_equilibration_steps",
                "n_switching_steps",
            ):
                _positive_int(params[name], name)
            _positive_int(params.max_restarts, "max_restarts", allow_zero=True)
            if params.equilibration_control.value not in {"berendsen", "nose-hoover"}:
                raise ValidationError("equilibration_control must be 'berendsen' or 'nose-hoover'")
            self.ctx.cmdargs = validate_lammps_cmdargs(params.lammps_cmdargs.get_list())
            self.ctx.elements = sorted(self.ctx.composition)
            adapter = get_calculator_adapter(calculator["name"])
            potential, artifact = adapter.translate(
                self.inputs.calculator.metadata,
                dict(self.inputs.calculator.files),
                self.ctx.elements,
            )
            self.ctx.potential = potential
            self.ctx.artifact = artifact
            if (
                potential.model_format == MaceCalculatorAdapter.MODEL_FORMAT
                and params.lammps_cmdargs.get_list() != MLIAP_GPU_LAMMPS_CMDARGS
            ):
                raise ValidationError(
                    "mace-mliap requires the supported one-GPU ML-IAP Kokkos arguments"
                )
        except Exception as exception:
            return self.exit_codes.ERROR_INVALID_INPUT.format(reason=str(exception))
        return None

    def prepare_structure(self):
        prepared = prepare_supercell(self.inputs.structure, self.inputs.method_parameters.supercell)
        self.ctx.prepared_structure = prepared["structure"]
        self.ctx.structure_data = prepared["data"]

    def _protocol(self) -> dict:
        params = self.inputs.method_parameters
        return {
            "supercell": params.supercell.get_list(),
            "temperature_guess": params.temperature_guess.value,
            "temperature_step": params.temperature_step.value,
            "max_attempts": params.max_attempts.value,
            "n_iterations": params.n_iterations.value,
            "n_equilibration_steps": params.n_equilibration_steps.value,
            "n_switching_steps": params.n_switching_steps.value,
            "timestep": params.timestep.value,
            "equilibration_control": params.equilibration_control.value,
            "seed": params.seed.value,
            "lammps_cmdargs": params.lammps_cmdargs.get_list(),
            "lammps_cmdargs_serialized": self.ctx.cmdargs,
            "elements": self.ctx.elements,
            "artifact_key": self.ctx.potential.artifact_key,
        }

    def _calphy_input(self) -> dict:
        params = self.inputs.method_parameters
        from ase.data import atomic_masses, atomic_numbers

        md = deepcopy(params.md.get_dict()) if "md" in params else {}
        md.update(
            {
                "timestep": params.timestep.value,
                "seed": params.seed.value,
                "cmdargs": self.ctx.cmdargs,
            }
        )
        calculation = {
            "mode": "melting_temperature",
            "lattice": CalphyCalculation.STRUCTURE_FILE,
            "file_format": "lammps-data",
            "element": self.ctx.elements,
            "mass": [
                float(atomic_masses[atomic_numbers[element]]) for element in self.ctx.elements
            ],
            "repeat": [1, 1, 1],
            "pressure": pressure_gpa_to_bar(self.ctx.pressure),
            "pair_style": self.ctx.potential.pair_style,
            "pair_coeff": self.ctx.potential.pair_coeff,
            "execution_mode": "executable",
            "queue": {"scheduler": "local", "cores": 1, "commands": []},
            "md": md,
            "n_equilibration_steps": params.n_equilibration_steps.value,
            "n_switching_steps": params.n_switching_steps.value,
            "n_iterations": params.n_iterations.value,
            "equilibration_control": params.equilibration_control.value,
            "melting_temperature": {
                "guess": params.temperature_guess.value,
                "step": params.temperature_step.value,
                "attempts": params.max_attempts.value,
            },
        }
        if "tolerance" in params:
            calculation["tolerance"] = params.tolerance.get_dict()
        return {"calculations": [calculation]}

    def submit_calculation(self):
        params = self.inputs.method_parameters
        metadata = {"options": {"withmpi": False}}
        if "scheduler_options" in params:
            metadata["options"].update(params.scheduler_options.get_dict())
            metadata["options"]["withmpi"] = False
        inputs = {
            "calculation": {
                "code": params.calphy_code,
                "lammps_code": params.lammps_code,
                "structure_data": self.ctx.structure_data,
                "potential": self.ctx.artifact,
                "parameters": orm.Dict(dict=self._calphy_input()),
                "lammps_cmdargs": params.lammps_cmdargs,
                "metadata": metadata,
            },
            "max_iterations": orm.Int(params.max_restarts.value + 1),
        }
        return ToContext(calculation=self.submit(CalphyCalculationWorkChain, **inputs))

    def inspect_calculation(self):
        child = self.ctx.calculation
        if not child.is_finished_ok:
            return self.exit_codes.ERROR_CALCULATION_FAILED.format(
                reason=f"exit status {child.exit_status}: {child.exit_message or 'no message'}"
            )
        required = {
            "melting_temperature",
            "status",
            "calculation_metadata",
            "diagnostics",
            "retrieved_files",
        }
        if missing := required - set(child.outputs):
            return self.exit_codes.ERROR_MALFORMED_RESULTS.format(
                reason=f"missing outputs: {', '.join(sorted(missing))}"
            )
        temperature = child.outputs.melting_temperature.value
        if not math.isfinite(temperature) or temperature <= 0:
            return self.exit_codes.ERROR_MALFORMED_RESULTS.format(
                reason="temperature must be finite and positive"
            )
        return None

    def finalize(self):
        child = self.ctx.calculation
        calcjobs = [node for node in child.called_descendants if isinstance(node, orm.CalcJobNode)]
        identifiers = orm.Dict(
            dict={
                "restart_workchain": {"pk": child.pk, "uuid": child.uuid},
                "calcjobs": [{"pk": node.pk, "uuid": node.uuid} for node in calcjobs],
            }
        )
        protocol = orm.Dict(dict=self._protocol())
        report = create_calphy_report(
            self.inputs.composition,
            self.inputs.pressure,
            self.inputs.calculator.metadata,
            self.inputs.structure,
            self.ctx.prepared_structure,
            self.ctx.artifact,
            self.inputs.method_parameters.calphy_code,
            self.inputs.method_parameters.lammps_code,
            protocol,
            child.outputs.calculation_metadata,
            child.outputs.diagnostics,
            child.outputs.retrieved_files,
            identifiers,
        )
        self.out("melting_temperature", child.outputs.melting_temperature)
        self.out("status", child.outputs.status)
        self.out("report", report)
