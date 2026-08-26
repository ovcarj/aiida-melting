"""Provenance-preserving dispatcher for installed melting workflows."""

from __future__ import annotations

from aiida.engine import ToContext, WorkChain

from ..contracts import (
    define_common_inputs,
    define_common_outputs,
    normalize_composition,
    validate_calculator,
    validate_outputs,
    validate_structure_composition,
)
from ..registry import get_melting_workflow
from ..structures import resolve_structure


class MeltingWorkChain(WorkChain):
    """Validate common inputs, select a method, and forward its outputs unchanged."""

    @classmethod
    def define(cls, spec) -> None:
        super().define(spec)
        define_common_inputs(spec, include_method=True)
        define_common_outputs(spec)
        spec.outline(cls.validate_and_prepare, cls.submit_method, cls.inspect_method)
        spec.exit_code(201, "ERROR_INVALID_INPUT", message="Invalid semantic input: {reason}")
        spec.exit_code(202, "ERROR_STRUCTURE_SOURCE_NOT_IMPLEMENTED", message="{reason}")
        spec.exit_code(203, "ERROR_UNKNOWN_METHOD", message="{reason}")
        spec.exit_code(204, "ERROR_INVALID_METHOD_PARAMETERS", message="{reason}")
        spec.exit_code(301, "ERROR_CHILD_PROCESS_FAILED", message="Child process failed: {reason}")
        spec.exit_code(302, "ERROR_MALFORMED_OUTPUTS", message="Malformed child outputs: {reason}")

    def validate_and_prepare(self):
        try:
            self.ctx.composition = normalize_composition(self.inputs.composition)
            validate_calculator(self.inputs.calculator)
        except Exception as exception:
            return self.exit_codes.ERROR_INVALID_INPUT.format(reason=str(exception))

        if "structure" in self.inputs:
            try:
                self.ctx.structure = resolve_structure(self.inputs.structure)
                validate_structure_composition(self.ctx.structure, self.ctx.composition)
            except NotImplementedError as exception:
                return self.exit_codes.ERROR_STRUCTURE_SOURCE_NOT_IMPLEMENTED.format(
                    reason=str(exception)
                )
            except Exception as exception:
                return self.exit_codes.ERROR_INVALID_INPUT.format(reason=str(exception))

        try:
            self.ctx.workflow_class = get_melting_workflow(self.inputs.method.value)
        except Exception as exception:
            return self.exit_codes.ERROR_UNKNOWN_METHOD.format(reason=str(exception))

        parameters = dict(self.inputs.method_parameters)
        namespace = self.ctx.workflow_class.spec().inputs["method_parameters"]
        validation_error = namespace.validate(parameters)
        if validation_error:
            return self.exit_codes.ERROR_INVALID_METHOD_PARAMETERS.format(
                reason=str(validation_error)
            )
        self.ctx.parameters = parameters
        return None

    def submit_method(self):
        inputs = {
            "composition": self.inputs.composition,
            "calculator": self.inputs.calculator,
            "method_parameters": self.ctx.parameters,
        }
        if "structure" in self.ctx:
            inputs["structure"] = self.ctx.structure
        return ToContext(child=self.submit(self.ctx.workflow_class, **inputs))

    def inspect_method(self):
        child = self.ctx.child
        if not child.is_finished_ok:
            return self.exit_codes.ERROR_CHILD_PROCESS_FAILED.format(
                reason=f"exit status {child.exit_status}: {child.exit_message or 'no message'}"
            )
        output_names = ("melting_temperature", "status", "report")
        outputs = {name: getattr(child.outputs, name, None) for name in output_names}
        error = validate_outputs(outputs)
        if error:
            return self.exit_codes.ERROR_MALFORMED_OUTPUTS.format(reason=error)
        for name in output_names:
            self.out(name, outputs[name])
        return None
