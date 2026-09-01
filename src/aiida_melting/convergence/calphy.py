"""The deliberately small adapter for Calphy size-convergence runs."""

from __future__ import annotations

from aiida import orm
from aiida.common.exceptions import ValidationError

from ..workflows.calphy import CalphyMeltingWorkChain


def get_calphy_adapter(identifier: str) -> type[CalphyMeltingWorkChain]:
    """Return the supported direct melting workflow for an inner identifier."""
    if identifier not in {"calphy", "melting.calphy"}:
        raise ValidationError("inner_method must be 'calphy' or 'melting.calphy'")
    return CalphyMeltingWorkChain


def inject_supercell(parameters: dict, size: int) -> dict:
    """Copy child parameters and add the isotropic supercell owned by the wrapper."""
    if "supercell" in parameters:
        raise ValidationError("inner_method_parameters.supercell is owned by the wrapper")
    result = dict(parameters)
    result["supercell"] = orm.List(list=[size, size, size])
    return result
