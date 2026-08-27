"""Tests for discovery, introspection, and command-line interfaces."""

import json

import pytest
from aiida.common.exceptions import EntryPointError
from click.testing import CliRunner

from aiida_melting.api import get_common_inputs, get_melting_workflow, get_method_inputs
from aiida_melting.cli.main import main
from aiida_melting.registry import list_melting_methods
from aiida_melting.workflows.calphy import CalphyMeltingWorkChain
from aiida_melting.workflows.mock import MockMeltingWorkChain


def test_registry_aliases():
    assert "melting.calphy" in list_melting_methods()
    assert "melting.mock" in list_melting_methods()
    assert get_melting_workflow("calphy") is CalphyMeltingWorkChain
    assert get_melting_workflow("mock") is MockMeltingWorkChain
    assert get_melting_workflow("melting.mock") is MockMeltingWorkChain
    with pytest.raises(EntryPointError):
        get_melting_workflow("absent")
    with pytest.raises(EntryPointError):
        get_melting_workflow("melting.calculate")


def test_introspection():
    common = get_common_inputs()
    assert set(common) == {
        "calculator",
        "composition",
        "description",
        "method",
        "pressure",
        "structure",
    }
    assert common["pressure"]["required"] is True
    assert common["description"]["required"] is False
    assert common["calculator"]["children"]["metadata"]["types"] == ["Dict"]
    assert common["calculator"]["children"]["files"]["dynamic"] is True
    method = get_method_inputs("mock")
    assert method["temperature"]["types"] == ["Float"]
    assert method["temperature"]["required"] is True
    calphy = get_method_inputs("calphy")
    assert calphy["calphy_code"]["types"] == ["InstalledCode"]
    assert calphy["supercell"]["required"] is False
    assert calphy["temperature_guess"]["required"] is True


def test_standalone_cli():
    runner = CliRunner()
    result = runner.invoke(main, ["methods"])
    assert result.exit_code == 0
    assert "melting.mock" in result.output
    result = runner.invoke(main, ["inputs", "mock"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "common" in payload
    assert "temperature" in payload["method"]


def test_cli_unknown_method():
    result = CliRunner().invoke(main, ["inputs", "absent"])
    assert result.exit_code != 0
    assert "unknown melting method" in result.output
