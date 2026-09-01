"""End-to-end tests using an in-memory AiiDA runner."""

from pathlib import Path

import pytest
from aiida import orm
from aiida.engine import calcfunction, run_get_node
from click.testing import CliRunner

from aiida_melting.analysis.calphy.reader import read_calphy_retrieved
from aiida_melting.analysis.query import query_results
from aiida_melting.cli.main import main
from aiida_melting.contracts import BaseMeltingWorkChain
from aiida_melting.workflows.dispatcher import MeltingWorkChain
from aiida_melting.workflows.mock import MockMeltingWorkChain

FAKE_CALPHY = Path(__file__).parent / "fixtures" / "calphy_kernel"


@calcfunction
def make_test_outputs() -> dict[str, orm.Data]:
    return {"status": orm.Str("ambiguous"), "report": orm.Dict(dict={"test": True})}


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
        "pressure": orm.Float(0.0),
        "description": orm.Str("contract stabilization test"),
        "calculator": {
            "metadata": orm.Dict(
                dict={
                    "name": "mock-calculator",
                    "provides": ["energy", "forces", "stress"],
                    "metadata": {"version": "none"},
                }
            ),
        },
        "method_parameters": {"temperature": orm.Float(1234.5)},
    }


@pytest.mark.usefixtures("aiida_profile_clean")
def test_mock_workchain(inputs):
    results, node = run_get_node(MockMeltingWorkChain, **inputs)
    assert node.is_finished_ok
    assert results["melting_temperature"].value == 1234.5
    assert results["status"].value == "success"
    assert results["report"].get_dict()["units"] == {
        "melting_temperature": "K",
        "pressure": "GPa",
    }
    assert results["report"].get_dict()["pressure"] == 0.0
    assert results["report"].get_dict()["calculator"] == {"name": "mock-calculator"}
    assert "convergence_status" not in results["report"].get_dict()
    assert "no scientific" in results["report"].get_dict()["warnings"][0].lower()


@pytest.mark.usefixtures("aiida_profile_clean")
def test_dispatcher_forwards_nodes_and_records_call(inputs):
    inputs["calculator"]["files"] = {
        "potential": orm.SinglefileData.from_bytes(b"mock potential", filename="potential.eam")
    }
    results, node = run_get_node(MeltingWorkChain, method=orm.Str("mock"), **inputs)
    assert node.is_finished_ok
    called = node.called
    assert len(called) == 1
    child = called[0]
    assert child.process_type == "aiida.workflows:melting.mock"
    assert results["melting_temperature"].uuid == child.outputs.melting_temperature.uuid
    assert results["status"].uuid == child.outputs.status.uuid
    assert results["report"].uuid == child.outputs.report.uuid
    assert child.inputs.pressure.uuid == inputs["pressure"].uuid
    assert child.inputs.description.uuid == inputs["description"].uuid
    assert (
        child.inputs.calculator.files.potential.uuid
        == inputs["calculator"]["files"]["potential"].uuid
    )


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


@pytest.mark.usefixtures("aiida_profile_clean")
def test_calphy_dispatcher_end_to_end(aiida_localhost, tmp_path):
    calphy_code = orm.InstalledCode(
        computer=aiida_localhost,
        label="fake-calphy",
        filepath_executable=str(FAKE_CALPHY),
    ).store()
    lammps_code = orm.InstalledCode(
        computer=aiida_localhost,
        label="fake-lammps",
        filepath_executable="/bin/true",
    ).store()
    structure = orm.StructureData(cell=[[3.6, 0, 0], [0, 3.6, 0], [0, 0, 3.6]])
    structure.append_atom(position=(0, 0, 0), symbols="Cu")
    artifact = orm.SinglefileData.from_bytes(b"fake eam", filename="Cu.eam.alloy")
    supplied = {
        "composition": orm.Dict(dict={"Cu": 1}),
        "pressure": orm.Float(0.1),
        "calculator": {
            "metadata": orm.Dict(
                dict={
                    "name": "eam",
                    "provides": ["energy", "forces", "stress"],
                    "metadata": {"pair_style": "eam/alloy", "elements": ["Cu"]},
                }
            ),
            "files": {"potential": artifact},
        },
        "structure": structure,
        "method": orm.Str("calphy"),
        "method_parameters": {
            "calphy_code": calphy_code,
            "lammps_code": lammps_code,
            "temperature_guess": orm.Float(1300),
            "seed": orm.Int(12345),
        },
    }
    results, node = run_get_node(MeltingWorkChain, **supplied)
    assert node.is_finished_ok, node.exit_message
    method = node.called[0]
    restart = method.called[1]
    calcjob = restart.called[0]
    assert method.process_type == "aiida.workflows:melting.calphy"
    assert calcjob.process_type == "aiida.calculations:melting.calphy"
    assert results["melting_temperature"].value == 1325.5
    assert results["melting_temperature"].uuid == calcjob.outputs.melting_temperature.uuid
    assert results["status"].value == "success"
    report = results["report"].get_dict()
    assert report["method"] == "melting.calphy"
    assert report["pressure_bar"] == 1000.0
    assert report["calculator"]["artifact_uuid"] == artifact.uuid
    assert report["codes"]["calphy"]["uuid"] == calphy_code.uuid
    assert report["codes"]["lammps"]["uuid"] == lammps_code.uuid
    assert report["codes"]["lammps"]["prepend_append_scripts_applied"] is False
    assert report["uncertainty_available"] is True
    assert "melting_temperature.log" in report["retrieved_files"]

    # Exercise the public AiiDA-query and CLI paths against a real dispatcher
    # process and its retrieved FolderData, not mocked analysis records.
    records = query_results(elements=("Cu",), calculator="eam", status="success")
    assert len(records) == 1
    assert records[0].composition == {"Cu": 1}
    assert records[0].artifact_filename == "Cu.eam.alloy"
    analysis = read_calphy_retrieved(calcjob.outputs.retrieved)
    assert analysis.attempts
    assert analysis.attempts[0].solid is not None
    assert analysis.attempts[0].solid.equilibration is not None
    assert "aiida-melting-calphy-" not in analysis.root
    assert analysis.root.startswith("aiida://")
    assert analysis.attempts[0].solid.directory.startswith(analysis.root)
    assert "aiida-melting-calphy-" not in analysis.attempts[0].solid.equilibration.source

    runner = CliRunner()
    result = runner.invoke(main, ["results", "--element", "Cu", "--format", "json"])
    assert result.exit_code == 0
    assert str(node.pk) in result.output
    result = runner.invoke(main, ["inspect", str(node.pk)])
    assert result.exit_code == 0
    assert '"uncertainty_available": true' in result.output
    output = tmp_path / "overview.png"
    result = runner.invoke(
        main, ["plot", "calphy", str(node.pk), "--kind", "overview", "--output", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert output.stat().st_size > 0
