# aiida-melting

`aiida-melting` is an extensible AiiDA framework for reproducible
melting-temperature workflows. It provides a common interface for submitting
calculations, retaining their scientific inputs and outputs in provenance, and
comparing completed results.

The package includes a deterministic mock workflow and a direct Calphy 2.0.1
integration. The Calphy workflow supports EAM potentials and the ML-IAP unified
MACE interface with externally converted `*-mliap_lammps.pt` models. The
original LAMMPS `pair_style mace` interface, Materials Project retrieval, and
implicit model download or conversion are not implemented. Python 3.11 or newer
and `aiida-core>=2.9,<3` are required.

## Installation

```console
python -m pip install -e '.[dev]'
```

For read-only result tables and figures in a regular installation, install the
optional analysis dependencies:

```console
python -m pip install -e '.[analysis]'
```

The package registers `melting.calculate`, `melting.mock`, and `melting.calphy`
in the `aiida.workflows` entry-point group. It also registers the
`melting.calphy` calculation and parser entry points. Inspect the installation
with either CLI:

```console
aiida-melting methods
aiida-melting inputs mock
verdi data melting methods
verdi data melting inputs mock
```

## Input contract

The dispatcher `MeltingWorkChain` has these common inputs:

- `composition` (`Dict`, required): element symbols mapped to finite, strictly
  positive amounts. Values are normalized to atomic fractions and keys sorted.
- `pressure` (`Float`, required): finite applied pressure in GPa. Negative
  values are permitted for tensile conditions.
- `description` (`Str`, optional): a human-readable calculation description.
- `calculator` (namespace, required): `calculator.metadata` is a required
  `Dict` containing `name`, `metadata`, and `provides`. `provides` must contain
  the case-sensitive capabilities `energy`, `forces`, and `stress`. Additional
  capabilities and fields are retained. `calculator.files` is an optional,
  dynamic namespace of named `SinglefileData` artifacts such as EAM potential
  files or MACE checkpoints. These nodes retain their provenance. Executable
  `Code` inputs remain method-specific.
- `structure` (`StructureData | Dict`, optional): an explicit structure or a
  source specification. Explicit compositions, including weighted mixed kinds,
  must match within `1e-8`.
- `method` (`Str`, required): a canonical identifier such as `melting.mock`, or
  its short alias `mock`.
- `method_parameters` (dynamic namespace, required): validated against the
  selected method's process specification.

A source specification is a closed schema:

```python
{"source": "materials_project", "parameters": {...}}
```

Unknown sources and fields are rejected. `materials_project` is recognized but
deliberately returns the dispatcher's not-implemented exit code; it performs no
network access.

Incorrect AiiDA node types are rejected during process submission. Semantic
errors are checked inside the dispatcher so the failed process and its stable
exit code are stored in provenance:

| Code | Meaning |
| ---: | --- |
| 201 | Invalid semantic common input |
| 202 | Recognized structure source is not implemented |
| 203 | Unknown or incompatible method |
| 204 | Invalid method parameters |
| 301 | Child process failed |
| 302 | Child returned malformed outputs |

## Output contract

Every method returns `melting_temperature` (`Float`, kelvin), `status` (`Str`),
and `report` (`Dict`). Temperature must be finite and positive. Allowed statuses
are `success`, `unconverged`, and `ambiguous`. An `ambiguous` result still carries
a positive provisional melting-temperature estimate; methods should explain its
interpretation in the report. Execution failures are represented only by non-zero
AiiDA exit codes, not successful-process statuses. Child process failure or
malformed output fails dispatch. Valid child output nodes are forwarded unchanged,
and the dispatcher records a `CALL_WORK` link to the child.

The report permits extension fields but requires `method`, `units`,
`composition`, `pressure`, and `calculator`. Units must identify melting
temperature as `K` and pressure as `GPa`, and the calculator field must contain
its name. The dispatcher verifies the report's composition, pressure, calculator
identity, and canonical method identifier against its inputs.

Convergence is distinct from the overall scientific status and is method-specific.
Methods performing convergence studies should add a structured extension such as:

```python
"convergence": {
    "variable": "atom_count",
    "tested_values": [256, 500, 864],
    "tolerance_K": 20,
    "converged": True,
}
```

A result may therefore be cell-size converged while its overall status remains
`ambiguous` for another scientific reason.

## Mock example

```python
from aiida import orm
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory

dispatcher = WorkflowFactory("melting.calculate")
results, process = run_get_node(
    dispatcher,
    composition=orm.Dict(dict={"Al": 2, "O": 3}),
    pressure=orm.Float(0.0),
    description=orm.Str("Al2O3 mock example"),
    calculator={
        "metadata": orm.Dict(
            dict={
                "name": "example",
                "provides": ["energy", "forces", "stress"],
                "metadata": {},
            }
        ),
        # Optional artifacts are provenance-tracked SinglefileData nodes:
        # "files": {"potential": orm.SinglefileData("potential.eam")},
    },
    method=orm.Str("mock"),
    method_parameters={"temperature": orm.Float(2300.0)},
)
assert process.is_finished_ok
assert results["melting_temperature"].value == 2300.0
```

The mock simply echoes its temperature parameter, normalizes composition in its
report, reports kelvin units, and adds an explicit non-scientific warning.

## Analysis

The read-only analysis tools work with completed melting workflows: they do not
submit calculations, alter provenance, or retrieve trajectories that the
calculation did not retain.

The command groups expose result queries, diagnostics, and figures. `results`
emits a terminal table, CSV, or JSON and filters by element, pressure, method,
calculator, artifact SHA-256, and scientific status. The artifact hash is the
reliable potential or model identity. A result record also keeps the input
structure caching hash, which identifies node content but is not a
crystallographic-equivalence fingerprint.

The Calphy tools read the retrieved `FolderData` attached to a process and can
also read an exported retrieved directory. Available figures cover
block-averaged equilibration, reference switching, reversible temperature
scaling, free-energy curves, adaptive attempts, and a compact overview. Their
diagnostics report measured values and explicit Calphy log warnings; they do not
manufacture a pass/fail scientific grade.

For switching files with more than one reference term, the raw traces remain
available but the reader does not infer a combined integrand without explicit
reference weights. This avoids silently misrepresenting multicomponent free
energy paths.

## Adding a method

Subclass `aiida_melting.contracts.BaseMeltingWorkChain`, add inputs below its
`method_parameters` namespace, and implement the common outputs. Register the
class under a canonical `melting.<name>` entry point in `aiida.workflows`.
Discovery is dynamic: no dispatcher or registry edit is needed. The included
Calphy implementation follows this pattern and is documented in
[docs/calphy.md](docs/calphy.md).

The direct-Calphy isotropic supercell wrapper is documented in [docs/convergence.md](docs/convergence.md).

The public helpers `list_melting_methods()`, `get_melting_workflow(identifier)`,
`get_common_inputs()`, and `get_method_inputs(identifier)` support programmatic
discovery and introspection.
