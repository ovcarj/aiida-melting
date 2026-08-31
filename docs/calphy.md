# Calphy melting workflow

`melting.calphy` targets Calphy 2.0.1 and runs Calphy directly as one AiiDA
`CalcJob`. Calphy owns its adaptive melting-temperature loop and starts LAMMPS
subprocesses itself. Splitting those subprocesses into separate AiiDA
calculations would break the runtime continuity that Calphy manages.

The report uses `target_calphy_version` because the configured executable's
version is not probed at runtime. Its `InstalledCode` UUID and executable path
remain part of the provenance.

## Codes and runtime environment

`method_parameters.calphy_code` and `method_parameters.lammps_code` must both be
stored `InstalledCode` nodes on the same `Computer`. AiiDA launches only
`calphy_code`, as `calphy_kernel -i input.yaml -k 0`, without MPI. Calphy
receives the absolute `lammps_code` executable path and starts one LAMMPS rank.
The LAMMPS code node is nevertheless an explicit provenance input.

The complete nested runtime is a contract of `calphy_code.prepend_text`. It must
establish all modules, the Calphy Python environment, shared-library paths,
CUDA/Kokkos state, and dependencies needed by the chosen LAMMPS binary. For
example:

```sh
module purge
module load cuda/12
source /path/to/calphy/bin/activate
export LD_LIBRARY_PATH=/path/to/lammps/lib:${LD_LIBRARY_PATH:-}
```

The workflow deliberately does not apply `lammps_code.prepend_text` or
`append_text`: AiiDA does not launch that code. Put every required nested-process
setup command on `calphy_code`. Static validation confirms node type, computer
identity, and executable paths, but cannot prove that modules, libraries,
drivers, or binaries are compatible. Verify that runtime contract on the target
machine before production use.

## Method inputs and units

Required method inputs are `calphy_code`, `lammps_code`,
`temperature_guess` (K), and a positive deterministic `seed`. Defaults are:

| Input | Default | Meaning |
| --- | ---: | --- |
| `supercell` | `[1, 1, 1]` | deterministic replication before submission |
| `temperature_step` | `400.0` K | half-width of Calphy's search interval |
| `max_attempts` | `5` | Calphy adaptive attempts |
| `n_iterations` | `2` | switching repetitions |
| `n_equilibration_steps` | `10000` | MD equilibration steps |
| `n_switching_steps` | `25000` | thermodynamic switching steps |
| `timestep` | `0.001` ps | LAMMPS metal-unit timestep |
| `equilibration_control` | `berendsen` | `berendsen` or `nose-hoover` |
| `lammps_cmdargs` | `[]` | validated LAMMPS subprocess arguments |
| `max_restarts` | `0` | whole-job retries after transient failures |

Optional `md` and `tolerance` dictionaries extend the matching Calphy blocks.
The explicit timestep, seed, and safe command arguments take precedence in
`md`. `scheduler_options` is merged into AiiDA CalcJob options; `withmpi` is
always false. The default resources request one machine and one process. GPU
resources must be requested with options appropriate to the configured AiiDA
scheduler.

The common pressure is in GPa and is converted exactly to LAMMPS metal units as
`bar = GPa × 10,000`. The report retains both values. Time is in ps and
temperature in K.

Only these LAMMPS argument token lists are accepted:

```python
[]
["-k", "on", "g", "1", "-sf", "kk"]
["-k", "on", "g", "1", "-sf", "kk",
 "-pk", "kokkos", "newton", "on", "neigh", "half"]
```

The ML-IAP MACE schema requires the third list. It exposes exactly one GPU and
sets the Kokkos Newton and neighbor modes required by the unified interface.
The shorter GPU list remains available for non-ML-IAP LAMMPS use.

Every token is checked for whitespace and shell metacharacters before the
closed allowlist comparison. The list is serialized with `shlex.join` to
Calphy's `md.cmdargs`; Calphy converts it back to subprocess tokens. It is never
evaluated through a shell. Multi-rank and multi-GPU execution are outside this
prototype.

## Calculator schemas

The explicit structure must be fully occupied, nonsingular, periodic in all
three dimensions, and compositionally consistent. Mixed occupancies and
vacancies are rejected. Species types are ordered alphabetically in
deterministic LAMMPS data.

EAM example:

```python
calculator={
    "metadata": orm.Dict(dict={
        "name": "eam",
        "provides": ["energy", "forces", "stress"],
        "metadata": {
            "pair_style": "eam/alloy",  # eam, eam/alloy, or eam/fs
            "elements": ["Cu"],
        },
    }),
    "files": {"potential": orm.SinglefileData("Cu.eam.alloy")},
}
```

MACE example:

```python
calculator={
    "metadata": orm.Dict(dict={
        "name": "mace",
        "provides": ["energy", "forces", "stress"],
        "metadata": {
            "model_format": "mace-mliap",
            "elements": ["Cu"],
        },
    }),
    "files": {
        "model": orm.SinglefileData("mace-mpa-0-medium-mliap_lammps.pt")
    },
}
```

This is the only supported MACE format. The artifact must already be a
LAMMPS-ready `*-mliap_lammps.pt` model. The workflow emits `pair_style mliap
unified ../<model> 0` and the corresponding element-mapped `pair_coeff`. Calphy
runs each phase one directory below the CalcJob root, so `..` resolves the model
staged by AiiDA. It never downloads or converts a model. Prepare the model
externally with the MACE version and target GPU architecture used by the LAMMPS
build, then import it as `SinglefileData`. The model UUID, filename, SHA-256
hash, format, and mapping are reported.

Artifact filenames must be whitespace-free basenames and must not collide with
generated AiiDA/Calphy inputs, scheduler scripts, or stdout/stderr files. Names
such as `input.yaml`, `structure.data`, and `_aiidasubmit.sh` are rejected.

## Parsing, retrieval, and retries

The parser reads the final `STATE: Tm = ... K +/- ... K` record and requires a
finite positive temperature. A completed calculation is `success` unless
Calphy explicitly emits a reliability diagnostic; a valid provisional estimate
with such a warning is `ambiguous`. Missing uncertainty alone is not ambiguous.

Known terminal failures are classified before temperature parsing:

| Exit code | Classification |
| ---: | --- |
| 305 | Calphy rejected its input |
| 306 | required LAMMPS style is unavailable |
| 307 | LAMMPS failed during execution |
| 308 | Calphy exhausted its melting attempts |

These failures are terminal and retain their code through the internal restart
wrapper. Only explicitly transient scheduler, transport, and incomplete-
retrieval failures are retried.

Uncertainty is unavailable when `n_iterations == 1`, or when the reported value
is absent, zero, negative, NaN, or infinite. No `uncertainty` output is created
in those cases and status is unchanged. The report always records
`uncertainty_available`.

Permanent retrieval includes stdout/stderr, top-level logs and effective YAML
inputs, phase reports, temperature/switching data, segment logs, and generated
scripts. Restart files and trajectory dumps remain remote. For debugging,
inspect or copy them from the CalcJob's `remote_folder` before cleaning it;
permanently retrieving them should be a deliberate storage-aware customization.

The internal restart workchain reruns the whole job with identical inputs and
seed after recognized transient transport, node, walltime, memory, or
incomplete-retrieval failures, up to `max_restarts`. It does not claim to resume
Calphy state. Invalid input, unsupported calculators, malformed output,
executable incompatibility, invalid accounts, and scientific reliability
warnings are not retried.

The common report contains method/protocol versions, input and prepared
structure UUIDs, composition and pressure, protocol values, calculator and
artifact provenance, code UUIDs and paths, diagnostics, uncertainty
availability, child identifiers, and retrieved-file inventory. It makes no
experimental accuracy claim. Cell-size convergence remains future work and is
independent of `status`.
