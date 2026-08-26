"""Deterministic non-scientific mock melting workflow."""

import math

from aiida import orm
from aiida.engine import calcfunction, if_

from ..contracts import BaseMeltingWorkChain, normalize_composition, validate_calculator


@calcfunction
def create_mock_outputs(
    composition: orm.Dict, pressure: orm.Float, calculator_metadata: orm.Dict
) -> dict[str, orm.Data]:
    """Create mock output nodes with explicit calculation provenance."""
    return {
        "status": orm.Str("success"),
        "report": orm.Dict(
            dict={
                "method": "melting.mock",
                "composition": normalize_composition(composition),
                "units": {"melting_temperature": "K", "pressure": "GPa"},
                "pressure": pressure.value,
                "calculator": {"name": calculator_metadata.get_dict()["name"]},
                "warnings": ["Mock result only: no scientific melting calculation was performed."],
            }
        ),
    }


class MockMeltingWorkChain(BaseMeltingWorkChain):
    """Return the requested temperature for integration testing and examples."""

    @classmethod
    def define(cls, spec) -> None:
        super().define(spec)
        spec.input("method_parameters.temperature", valid_type=orm.Float)
        spec.outline(
            cls.validate_inputs, if_(cls.has_structure)(cls.validate_structure), cls.run_mock
        )
        spec.exit_code(201, "ERROR_INVALID_INPUT", message="Invalid semantic input: {reason}")

    def validate_inputs(self):
        try:
            self.ctx.composition = normalize_composition(self.inputs.composition)
            validate_calculator(self.inputs.calculator.metadata)
            temperature = self.inputs.method_parameters.temperature.value
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("method_parameters.temperature must be finite and positive")
        except Exception as exception:
            # AiiDA ValidationError and numerical errors become provenance-bearing failures.
            return self.exit_codes.ERROR_INVALID_INPUT.format(reason=str(exception))
        return None

    def has_structure(self):
        return "structure" in self.inputs

    def validate_structure(self):
        from ..contracts import validate_structure_composition

        if not isinstance(self.inputs.structure, orm.StructureData):
            return self.exit_codes.ERROR_INVALID_INPUT.format(
                reason="method workflows require a resolved StructureData"
            )
        try:
            validate_structure_composition(self.inputs.structure, self.ctx.composition)
        except Exception as exception:
            return self.exit_codes.ERROR_INVALID_INPUT.format(reason=str(exception))
        return None

    def run_mock(self):
        temperature = self.inputs.method_parameters.temperature
        self.out("melting_temperature", temperature)
        self.out_many(
            create_mock_outputs(
                self.inputs.composition, self.inputs.pressure, self.inputs.calculator.metadata
            )
        )
