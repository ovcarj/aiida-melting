"""Supercell-size convergence wrapper for direct Calphy melting calculations."""

from __future__ import annotations

import math

from aiida import orm
from aiida.common.exceptions import ValidationError
from aiida.engine import ToContext, calcfunction, if_, while_

from ..contracts import (
    BaseMeltingWorkChain,
    normalize_composition,
    validate_calculator,
    validate_outputs,
    validate_pressure,
    validate_structure_composition,
)
from ..convergence.calphy import get_calphy_adapter, inject_supercell


def relative_difference_percent(first: float, second: float) -> float:
    """Symmetric relative temperature difference, expressed as percent."""
    return 100.0 * 2.0 * abs(second - first) / (second + first)


@calcfunction
def create_convergence_outputs(
    selected_status: orm.Str,
    selected_report: orm.Dict,
    configuration: orm.Dict,
    **child_outputs: orm.Data,
) -> dict[str, orm.Data]:
    """Create derived public nodes while retaining links to every child output.

    ``child_outputs`` is intentionally dynamic: the number of evaluated cells is
    only known after the workflow has run.  Its values are otherwise not decoded
    here; receiving them is what makes the complete history explicit provenance.
    """
    config = configuration.get_dict()
    report = dict(selected_report.get_dict())
    report["method"] = "melting.supercell_convergence"
    report["convergence"] = config["convergence"]
    # The wrapper status means "no size pair met the requested tolerance" when
    # convergence was not demonstrated, independently of a child provisional
    # status.
    status = selected_status.value if config["convergence"]["converged"] else "unconverged"
    return {"status": orm.Str(status), "report": orm.Dict(dict=report)}


class SupercellConvergenceWorkChain(BaseMeltingWorkChain):
    """Run isotropic direct-Calphy supercells until adjacent valid points agree."""

    @classmethod
    def define(cls, spec) -> None:
        super().define(spec)
        namespace = "method_parameters."
        spec.input(namespace + "inner_method", valid_type=orm.Str)
        spec.input_namespace(namespace + "inner_method_parameters", dynamic=True, required=True)
        spec.input(
            namespace + "initial_sizes", valid_type=orm.List, default=lambda: orm.List(list=[6, 7])
        )
        spec.input(namespace + "maximum_size", valid_type=orm.Int, default=lambda: orm.Int(9))
        spec.input(
            namespace + "relative_tolerance_percent",
            valid_type=orm.Float,
            default=lambda: orm.Float(2.0),
        )
        spec.input(
            namespace + "parallel_initial", valid_type=orm.Bool, default=lambda: orm.Bool(True)
        )
        spec.outline(
            cls.validate_inputs,
            cls.setup,
            if_(cls.parallel_initial)(cls.submit_parallel_initial, cls.submit_first_initial),
            if_(cls.parallel_initial)(cls.inspect_parallel_initial, cls.inspect_first_initial),
            if_(cls.not_parallel_initial)(cls.submit_second_initial, cls.inspect_second_initial),
            while_(cls.should_continue)(cls.submit_next, cls.inspect_next),
            cls.finalize,
        )
        spec.exit_code(201, "ERROR_INVALID_INPUT", message="Invalid convergence input: {reason}")
        spec.exit_code(301, "ERROR_CHILD_PROCESS_FAILED", message="Child process failed: {reason}")
        spec.exit_code(302, "ERROR_MALFORMED_OUTPUTS", message="Malformed child outputs: {reason}")

    def validate_inputs(self):
        try:
            if not isinstance(self.inputs.structure, orm.StructureData):
                raise ValidationError("an explicit resolved StructureData is required")
            self.ctx.composition = normalize_composition(self.inputs.composition)
            self.ctx.pressure = validate_pressure(self.inputs.pressure)
            self.ctx.calculator = validate_calculator(self.inputs.calculator.metadata)
            validate_structure_composition(self.inputs.structure, self.ctx.composition)
            params = self.inputs.method_parameters
            self.ctx.inner_workchain = get_calphy_adapter(params.inner_method.value)
            supplied = dict(params.inner_method_parameters)
            # Validate ownership before any child is submitted.
            inject_supercell(supplied, 1)
            sizes = params.initial_sizes.get_list()
            if (
                len(sizes) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in sizes
                )
                or sizes[0] >= sizes[1]
            ):
                raise ValidationError(
                    "initial_sizes must contain two strictly increasing positive integers"
                )
            maximum = params.maximum_size.value
            if isinstance(maximum, bool) or maximum < sizes[1]:
                raise ValidationError("maximum_size must be at least the larger initial size")
            tolerance = params.relative_tolerance_percent.value
            if not math.isfinite(tolerance) or tolerance <= 0:
                raise ValidationError("relative_tolerance_percent must be finite and positive")
            self.ctx.initial_sizes = sizes
            self.ctx.maximum_size = maximum
            self.ctx.tolerance = tolerance
            self.ctx.parallel = params.parallel_initial.value
            self.ctx.inner_parameters = supplied
        except Exception as exception:
            return self.exit_codes.ERROR_INVALID_INPUT.format(reason=str(exception))
        return None

    def setup(self):
        self.ctx.records = []
        self.ctx.next_size = self.ctx.initial_sizes[1] + 1
        self.ctx.converged = False
        self.ctx.selected = None

    def _child_inputs(self, size: int) -> dict:
        inputs = {
            "composition": self.inputs.composition,
            "pressure": self.inputs.pressure,
            "calculator": dict(self.inputs.calculator),
            "structure": self.inputs.structure,
            "method_parameters": inject_supercell(self.ctx.inner_parameters, size),
        }
        if "description" in self.inputs:
            inputs["description"] = self.inputs.description
        return inputs

    def _submit_size(self, size: int):
        return self.submit(self.ctx.inner_workchain, **self._child_inputs(size))

    def submit_parallel_initial(self):
        first, second = self.ctx.initial_sizes
        return ToContext(
            initial_first=self._submit_size(first), initial_second=self._submit_size(second)
        )

    def submit_first_initial(self):
        return ToContext(initial_first=self._submit_size(self.ctx.initial_sizes[0]))

    def submit_second_initial(self):
        return ToContext(initial_second=self._submit_size(self.ctx.initial_sizes[1]))

    def _inspect_child(self, child, size: int):
        if not child.is_finished_ok:
            reason = (
                f"supercell {size}: exit status {child.exit_status}: "
                f"{child.exit_message or 'no message'}"
            )
            return self.exit_codes.ERROR_CHILD_PROCESS_FAILED.format(
                reason=reason
            )
        outputs = {
            name: getattr(child.outputs, name, None)
            for name in ("melting_temperature", "status", "report")
        }
        error = validate_outputs(
            outputs,
            composition=self.ctx.composition,
            pressure=self.ctx.pressure,
            calculator_name=self.ctx.calculator["name"],
            method="melting.calphy",
        )
        if error:
            return self.exit_codes.ERROR_MALFORMED_OUTPUTS.format(
                reason=f"supercell {size}: {error}"
            )
        self.ctx.records.append(
            {"size": size, "child": child, "valid": child.outputs.status.value != "unconverged"}
        )
        return None

    def _compare_latest(self):
        valid = [record for record in self.ctx.records if record["valid"]]
        if len(valid) < 2:
            return
        first, second = valid[-2:]
        difference = relative_difference_percent(
            first["child"].outputs.melting_temperature.value,
            second["child"].outputs.melting_temperature.value,
        )
        if difference <= self.ctx.tolerance:
            self.ctx.converged = True
            self.ctx.selected = second

    def inspect_parallel_initial(self):
        for size, child in (
            (self.ctx.initial_sizes[0], self.ctx.initial_first),
            (self.ctx.initial_sizes[1], self.ctx.initial_second),
        ):
            result = self._inspect_child(child, size)
            if result:
                return result
        self._compare_latest()

    def inspect_first_initial(self):
        return self._inspect_child(self.ctx.initial_first, self.ctx.initial_sizes[0])

    def inspect_second_initial(self):
        result = self._inspect_child(self.ctx.initial_second, self.ctx.initial_sizes[1])
        if result:
            return result
        self._compare_latest()

    def not_parallel_initial(self):
        return not self.ctx.parallel

    def parallel_initial(self):
        return self.ctx.parallel

    def should_continue(self):
        return not self.ctx.converged and self.ctx.next_size <= self.ctx.maximum_size

    def submit_next(self):
        size = self.ctx.next_size
        self.ctx.pending_size = size
        self.ctx.next_size += 1
        return ToContext(next_child=self._submit_size(size))

    def inspect_next(self):
        result = self._inspect_child(self.ctx.next_child, self.ctx.pending_size)
        if result:
            return result
        self._compare_latest()

    def _configuration(self, selected: dict) -> orm.Dict:
        tested, exclusions, comparisons = [], [], []
        previous = None
        for record in self.ctx.records:
            child = record["child"]
            temperature = child.outputs.melting_temperature.value
            report = child.outputs.report.get_dict()
            entry = {
                "size": record["size"],
                "supercell": [record["size"]] * 3,
                "atom_count": len(self.inputs.structure.sites) * record["size"] ** 3,
                "temperature": temperature,
                "status": child.outputs.status.value,
                "uncertainty_K": report.get("diagnostics", {}).get("uncertainty_K"),
                "child_pk": child.pk,
                "child_uuid": child.uuid,
            }
            tested.append(entry)
            if not record["valid"]:
                exclusions.append({"size": record["size"], "reason": "child status is unconverged"})
                continue
            if previous is not None:
                comparisons.append(
                    {
                        "sizes": [previous["size"], record["size"]],
                        "relative_difference_percent": relative_difference_percent(
                            previous["child"].outputs.melting_temperature.value, temperature
                        ),
                    }
                )
            previous = record
        return orm.Dict(
            dict={
                "convergence": {
                    "variable": "isotropic_supercell_size",
                    "initial_sizes": self.ctx.initial_sizes,
                    "maximum_size": self.ctx.maximum_size,
                    "relative_tolerance_percent": self.ctx.tolerance,
                    "parallel_initial": self.ctx.parallel,
                    "seed": self.ctx.inner_parameters.get("seed").value
                    if "seed" in self.ctx.inner_parameters
                    else None,
                    "tested": tested,
                    "excluded": exclusions,
                    "comparisons": comparisons,
                    "converged": self.ctx.converged,
                    "selected_size": selected["size"],
                    "selected_supercell": [selected["size"]] * 3,
                    "selected_child_pk": selected["child"].pk,
                    "selected_child_uuid": selected["child"].uuid,
                    "selected_process": {
                        "pk": selected["child"].pk,
                        "uuid": selected["child"].uuid,
                    },
                }
            }
        )

    def finalize(self):
        valid = [record for record in self.ctx.records if record["valid"]]
        selected = self.ctx.selected or (valid[-1] if valid else self.ctx.records[-1])
        child = selected["child"]
        configuration = self._configuration(selected)
        child_outputs = {}
        for record in self.ctx.records:
            prefix = f"size_{record['size']}"
            candidate = record["child"].outputs
            child_outputs[f"{prefix}_temperature"] = candidate.melting_temperature
            child_outputs[f"{prefix}_status"] = candidate.status
            child_outputs[f"{prefix}_report"] = candidate.report
        outputs = create_convergence_outputs(
            child.outputs.status, child.outputs.report, configuration, **child_outputs
        )
        self.out("melting_temperature", child.outputs.melting_temperature)
        self.out_many(outputs)
