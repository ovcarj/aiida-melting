"""Tests for semantic contracts."""

import math

import pytest
from aiida import orm
from aiida.common.exceptions import ValidationError

from aiida_melting.contracts import (
    normalize_composition,
    structure_composition,
    validate_calculator,
    validate_outputs,
    validate_pressure,
    validate_source_specification,
    validate_structure_composition,
)


@pytest.mark.usefixtures("aiida_profile_clean")
class TestContracts:
    def test_normalize_composition(self):
        assert normalize_composition({"O": 2, "Al": 4}) == {"Al": 2 / 3, "O": 1 / 3}

    @pytest.mark.parametrize(
        "composition",
        [{}, {"Xx": 1}, {"Al": 0}, {"Al": -1}, {"Al": math.inf}, {"Al": True}],
    )
    def test_invalid_composition(self, composition):
        with pytest.raises(ValidationError):
            normalize_composition(composition)

    def test_calculator_extensions_are_allowed(self):
        value = {
            "name": "test",
            "provides": ["energy", "forces", "stress", "charges"],
            "metadata": {"version": "1"},
            "extension": 42,
        }
        assert validate_calculator(value) == value

    @pytest.mark.parametrize(
        "calculator",
        [
            {},
            {"name": "x", "provides": ["energy", "forces"], "metadata": {}},
            {"name": "x", "provides": ["Energy", "forces", "stress"], "metadata": {}},
        ],
    )
    def test_invalid_calculator(self, calculator):
        with pytest.raises(ValidationError):
            validate_calculator(calculator)

    def test_mixed_occupancy_structure(self):
        structure = orm.StructureData(cell=[[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        structure.append_atom(position=(0, 0, 0), symbols=("Al", "Mg"), weights=(0.25, 0.75))
        assert structure_composition(structure) == {"Al": 0.25, "Mg": 0.75}
        validate_structure_composition(structure, {"Al": 0.25, "Mg": 0.75})
        with pytest.raises(ValidationError):
            validate_structure_composition(structure, {"Al": 0.5, "Mg": 0.5})

    def test_source_schema(self):
        valid = {"source": "materials_project", "parameters": {"material_id": "mp-1"}}
        assert validate_source_specification(valid) == valid
        for invalid in (
            {"source": "other", "parameters": {}},
            {"source": "materials_project", "parameters": {}, "extra": 1},
            {"source": "materials_project"},
        ):
            with pytest.raises(ValidationError):
                validate_source_specification(invalid)

    def test_output_validation(self):
        valid = {
            "melting_temperature": orm.Float(1000),
            "status": orm.Str("ambiguous"),
            "report": orm.Dict(
                dict={
                    "method": "melting.mock",
                    "units": {"melting_temperature": "K", "pressure": "GPa"},
                    "composition": {"Al": 1.0},
                    "pressure": 0.0,
                    "calculator": {"name": "test"},
                    "convergence": {
                        "variable": "atom_count",
                        "tested_values": [256, 500],
                        "tolerance_K": 20,
                        "converged": True,
                    },
                }
            ),
        }
        assert validate_outputs(valid) is None
        assert validate_outputs({**valid, "melting_temperature": orm.Float(math.nan)})
        assert validate_outputs({**valid, "status": orm.Str("unknown")})

    def test_pressure(self):
        assert validate_pressure(orm.Float(-2.5)) == -2.5
        with pytest.raises(ValidationError):
            validate_pressure(math.inf)
