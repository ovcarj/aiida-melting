# aiida-melting

`aiida-melting` 0.2.0 is an MIT-licensed, extensible AiiDA framework for
melting-temperature workflows. It requires Python 3.11 or newer and
`aiida-core>=2.9,<3`.

This initial release provides the framework and a deterministic mock workflow.
It does **not** implement scientific melting calculations, Calphy integration,
Materials Project access, structure generation, or calculator execution.

## Installation

```console
python -m pip install -e '.[dev]'
```

The package registers `melting.calculate` and `melting.mock` in the
`aiida.workflows` entry-point group. Inspect the installation with either CLI:

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
are `success`, `unconverged`, and `ambiguous`. Execution failures are represented
only by non-zero AiiDA exit codes, not successful-process statuses. Child process
failure or malformed output fails dispatch. Valid child output nodes are
forwarded unchanged, and the dispatcher records a `CALL_WORK` link to the child.

The report permits extension fields but requires `method`, `units`,
`composition`, `pressure`, `calculator`, and `convergence_status`. Units must
identify melting temperature as `K` and pressure as `GPa`; the calculator field
must contain its name; and convergence status must equal the top-level status.
The dispatcher verifies the report's composition, pressure, calculator identity,
and canonical method identifier against its inputs.

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

## Adding a method

Subclass `aiida_melting.contracts.BaseMeltingWorkChain`, add inputs below its
`method_parameters` namespace, and implement the common outputs. Register the
class under a canonical `melting.<name>` entry point in `aiida.workflows`.
Discovery is dynamic: no dispatcher or registry edit is needed. A future Calphy
integration should therefore register `melting.calphy` and independently supply
its calculation and validation implementation.

The public helpers `list_melting_methods()`, `get_melting_workflow(identifier)`,
`get_common_inputs()`, and `get_method_inputs(identifier)` support programmatic
discovery and introspection.
