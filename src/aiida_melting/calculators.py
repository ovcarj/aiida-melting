"""Calculator contract helpers and Calphy potential adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import PurePath
from typing import Any

from aiida import orm
from aiida.common.exceptions import ValidationError

from .contracts import REQUIRED_CAPABILITIES, validate_calculator


@dataclass(frozen=True)
class PotentialSpec:
    """Normalized, shell-free LAMMPS potential description."""

    name: str
    artifact_key: str
    artifact_filename: str
    elements: tuple[str, ...]
    pair_style: str
    pair_coeff: str
    model_format: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["elements"] = list(self.elements)
        return result


def _implementation(metadata: orm.Dict | dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = validate_calculator(metadata)
    implementation = raw["metadata"]
    return raw, implementation


def _elements(implementation: dict[str, Any], expected: list[str]) -> tuple[str, ...]:
    elements = implementation.get("elements")
    if (
        not isinstance(elements, list)
        or not elements
        or not all(isinstance(item, str) and item for item in elements)
    ):
        raise ValidationError("calculator.metadata.elements must be a non-empty list of symbols")
    if elements != expected:
        raise ValidationError(
            f"calculator element mapping {elements} does not match structure type order {expected}"
        )
    return tuple(elements)


def _artifact(files: dict[str, orm.SinglefileData], key: str) -> orm.SinglefileData:
    try:
        artifact = files[key]
    except KeyError as exception:
        raise ValidationError(f"calculator.files.{key} is required") from exception
    if not isinstance(artifact, orm.SinglefileData):
        raise ValidationError(f"calculator.files.{key} must be SinglefileData")
    filename = artifact.filename
    if PurePath(filename).name != filename or any(character.isspace() for character in filename):
        raise ValidationError("calculator artifact filename must be a whitespace-free basename")
    return artifact


class EamCalculatorAdapter:
    """Translate the closed EAM calculator schema to Calphy commands."""

    NAME = "eam"
    PAIR_STYLES = frozenset({"eam", "eam/alloy", "eam/fs"})

    @classmethod
    def translate(
        cls,
        metadata: orm.Dict | dict[str, Any],
        files: dict[str, orm.SinglefileData],
        expected_elements: list[str],
    ) -> tuple[PotentialSpec, orm.SinglefileData]:
        raw, implementation = _implementation(metadata)
        if raw["name"] != cls.NAME:
            raise ValidationError("EAM adapter requires calculator.name='eam'")
        artifact = _artifact(files, "potential")
        elements = _elements(implementation, expected_elements)
        pair_style = implementation.get("pair_style")
        if pair_style not in cls.PAIR_STYLES:
            allowed = ", ".join(sorted(cls.PAIR_STYLES))
            raise ValidationError(f"calculator.metadata.pair_style must be one of: {allowed}")
        if pair_style == "eam" and len(elements) != 1:
            raise ValidationError("pair_style 'eam' is supported only for one element")
        suffix = " " + " ".join(elements) if pair_style != "eam" else ""
        spec = PotentialSpec(
            name=cls.NAME,
            artifact_key="potential",
            artifact_filename=artifact.filename,
            elements=elements,
            pair_style=pair_style,
            pair_coeff=f"* * {artifact.filename}{suffix}",
        )
        return spec, artifact


class MaceCalculatorAdapter:
    """Translate a pre-converted MACE ML-IAP model to Calphy commands."""

    NAME = "mace"
    MODEL_FORMAT = "mace-lammps"

    @classmethod
    def translate(
        cls,
        metadata: orm.Dict | dict[str, Any],
        files: dict[str, orm.SinglefileData],
        expected_elements: list[str],
    ) -> tuple[PotentialSpec, orm.SinglefileData]:
        raw, implementation = _implementation(metadata)
        if raw["name"] != cls.NAME:
            raise ValidationError("MACE adapter requires calculator.name='mace'")
        artifact = _artifact(files, "model")
        elements = _elements(implementation, expected_elements)
        model_format = implementation.get("model_format")
        if model_format != cls.MODEL_FORMAT:
            raise ValidationError("calculator.metadata.model_format must be 'mace-lammps'")
        if not artifact.filename.endswith("lammps.pt"):
            raise ValidationError("MACE model must be an already converted *lammps.pt file")
        pair_style = f"mliap unified {artifact.filename} 0"
        spec = PotentialSpec(
            name=cls.NAME,
            artifact_key="model",
            artifact_filename=artifact.filename,
            elements=elements,
            pair_style=pair_style,
            pair_coeff="* * " + " ".join(elements),
            model_format=model_format,
        )
        return spec, artifact


def get_calculator_adapter(name: str) -> type[EamCalculatorAdapter] | type[MaceCalculatorAdapter]:
    """Return the adapter for a supported calculator name."""
    adapters = {
        EamCalculatorAdapter.NAME: EamCalculatorAdapter,
        MaceCalculatorAdapter.NAME: MaceCalculatorAdapter,
    }
    try:
        return adapters[name]
    except KeyError as exception:
        raise ValidationError(f"unsupported Calphy calculator: {name!r}") from exception


__all__ = (
    "REQUIRED_CAPABILITIES",
    "EamCalculatorAdapter",
    "MaceCalculatorAdapter",
    "PotentialSpec",
    "get_calculator_adapter",
    "validate_calculator",
)
