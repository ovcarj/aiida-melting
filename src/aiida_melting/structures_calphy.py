"""Provenance-producing structure preparation for Calphy."""

from __future__ import annotations

import io
import math

import numpy as np
from aiida import orm
from aiida.engine import calcfunction


def validate_calphy_structure(structure: orm.StructureData) -> None:
    """Validate the deliberately restricted initial structure contract."""
    if tuple(structure.pbc) != (True, True, True):
        raise ValueError("Calphy requires three-dimensional periodic boundary conditions")
    cell = np.asarray(structure.cell, dtype=float)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)) or abs(np.linalg.det(cell)) < 1e-12:
        raise ValueError("structure cell must be finite and nonsingular")
    if not structure.sites:
        raise ValueError("structure must contain at least one site")
    for kind in structure.kinds:
        if (
            len(kind.symbols) != 1
            or len(kind.weights) != 1
            or not math.isclose(float(kind.weights[0]), 1.0, abs_tol=1e-12)
        ):
            raise ValueError("mixed, vacant, or partially occupied kinds are not supported")


@calcfunction
def prepare_supercell(structure: orm.StructureData, supercell: orm.List) -> dict[str, orm.Data]:
    """Replicate a structure and emit deterministic LAMMPS data."""
    from ase.io import write

    validate_calphy_structure(structure)
    repetitions = supercell.get_list()
    if len(repetitions) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in repetitions
    ):
        raise ValueError("supercell must contain exactly three positive integers")
    atoms = structure.get_ase() * tuple(repetitions)
    elements = sorted(set(atoms.get_chemical_symbols()))
    order = sorted(
        range(len(atoms)), key=lambda index: (elements.index(atoms[index].symbol), index)
    )
    atoms = atoms[order]
    stream = io.StringIO()
    write(
        stream,
        atoms,
        format="lammps-data",
        atom_style="atomic",
        masses=True,
        specorder=elements,
    )
    data = orm.SinglefileData.from_bytes(stream.getvalue().encode(), filename="structure.data")
    return {"structure": orm.StructureData(ase=atoms), "data": data}
