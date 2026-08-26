"""End-to-end tests using an in-memory AiiDA runner."""

import pytest
from aiida import orm
from aiida.engine import calcfunction, run_get_node

from aiida_melting.contracts import BaseMeltingWorkChain
from aiida_melting.workflows.dispatcher import MeltingWorkChain
from aiida_melting.workflows.mock import MockMeltingWorkChain


@calcfunction
def make_test_outputs() -> dict[str, orm.Data]:
    return {"status": orm.Str("warning"), "report": orm.Dict(dict={"test": True})}


class InvalidOutputWorkChain(BaseMeltingWorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("method_parameters.temperature", valid_type=orm.Float)
        spec.outline(cls.run_invalid)

    def run_invalid(self):
        self.out("melting_temperature", self.inputs.method_parameters.temperature)
        self.out_many(make_test_outputs())


class FailedWorkChain(BaseMeltingWorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("method_parameters.temperature", valid_type=orm.Float)
        spec.outline(cls.fail)
        spec.exit_code(450, "ERROR_TEST_FAILURE", message="intentional test failure")

    def fail(self):
        return self.exit_codes.ERROR_TEST_FAILURE


@pytest.fixture
def inputs():
    return {
        "composition": orm.Dict(dict={"Al": 2, "O": 3}),
        "calculator": orm.Dict(
            dict={
                "name": "mock-calculator",
                "provides": ["energy", "forces", "stress"],
                "metadata": {"version": "none"},
            }
        ),
        "method_parameters": {"temperature": orm.Float(1234.5)},
    }


@pytest.mark.usefixtures("aiida_profile_clean")
def test_mock_workchain(inputs):
    results, node = run_get_node(MockMeltingWorkChain, **inputs)
    assert node.is_finished_ok
    assert results["melting_temperature"].value == 1234.5
    assert results["status"].value == "success"
    assert results["report"].get_dict()["units"] == {"melting_temperature": "K"}
    assert "no scientific" in results["report"].get_dict()["warnings"][0].lower()


@pytest.mark.usefixtures("aiida_profile_clean")
def test_dispatcher_forwards_nodes_and_records_call(inputs):
    results, node = run_get_node(MeltingWorkChain, method=orm.Str("mock"), **inputs)
    assert node.is_finished_ok
    called = node.called
    assert len(called) == 1
    child = called[0]
    assert child.process_type == "aiida.workflows:melting.mock"
    assert results["melting_temperature"].uuid == child.outputs.melting_temperature.uuid
    assert results["status"].uuid == child.outputs.status.uuid
    assert results["report"].uuid == child.outputs.report.uuid


@pytest.mark.usefixtures("aiida_profile_clean")
@pytest.mark.parametrize(
    ("changes", "exit_status"),
    [
        ({"method": "absent"}, 203),
        ({"method_parameters": {}}, 204),
        ({"composition": {"Xx": 1}}, 201),
    ],
)
def test_stable_dispatcher_failures(inputs, changes, exit_status):
    if isinstance(changes.get("method"), str):
        changes["method"] = orm.Str(changes["method"])
    if isinstance(changes.get("composition"), dict):
        changes["composition"] = orm.Dict(dict=changes["composition"])
    supplied = {"method": orm.Str("mock"), **inputs, **changes}
    _, node = run_get_node(MeltingWorkChain, **supplied)
    assert node.exit_status == exit_status


@pytest.mark.usefixtures("aiida_profile_clean")
def test_structure_source_not_implemented(inputs):
    source = orm.Dict(dict={"source": "materials_project", "parameters": {"id": "mp-1"}})
    _, node = run_get_node(MeltingWorkChain, method=orm.Str("mock"), structure=source, **inputs)
    assert node.exit_status == 202


@pytest.mark.usefixtures("aiida_profile_clean")
def test_structure_composition_failure(inputs):
    structure = orm.StructureData(cell=[[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    structure.append_atom(position=(0, 0, 0), symbols="Al")
    _, node = run_get_node(MeltingWorkChain, method=orm.Str("mock"), structure=structure, **inputs)
    assert node.exit_status == 201


@pytest.mark.usefixtures("aiida_profile_clean")
def test_malformed_child_output(monkeypatch, inputs):
    monkeypatch.setattr(
        "aiida_melting.workflows.dispatcher.get_melting_workflow",
        lambda identifier: InvalidOutputWorkChain,
    )
    inputs["method_parameters"]["temperature"] = orm.Float(-1)
    _, node = run_get_node(MeltingWorkChain, method=orm.Str("invalid-test"), **inputs)
    assert node.exit_status == 302


@pytest.mark.usefixtures("aiida_profile_clean")
def test_failed_child(monkeypatch, inputs):
    monkeypatch.setattr(
        "aiida_melting.workflows.dispatcher.get_melting_workflow",
        lambda identifier: FailedWorkChain,
    )
    _, node = run_get_node(MeltingWorkChain, method=orm.Str("failed-test"), **inputs)
    assert node.exit_status == 301
