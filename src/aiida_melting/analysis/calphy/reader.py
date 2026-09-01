"""Read Calphy's retrieved text files into analysis data objects."""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import yaml
from aiida import orm

from .data import AttemptData, CalphyAnalysis, PhaseData, SwitchingData, TableData

_STATE = re.compile(r"STATE:\s*Tm\s*=\s*([-+0-9.eE]+)\s*K(?:\s*\+/-\s*([-+0-9.eE]+)\s*K)?")
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_PHASE = re.compile(
    rf"^(?P<prefix>ts-.+)-(?P<phase>solid|liquid)-(?P<temperature>{_NUMBER})-(?P<pressure>{_NUMBER})$"
)
_REPLICA = re.compile(
    r"(?:ts\.)?(?P<direction>forward|backward)(?:_(?P<leg>leg\d+))?_(?P<replica>\d+)\.dat$"
)
_RANGE = re.compile(
    r"(?:STATE:\s*)?Temperature range of\s+([-+0-9.eE]+)\s*-\s*([-+0-9.eE]+)\s*K?"
)
_ATTEMPT_FILE = re.compile(r"\.(\d+)\.yaml$")


def _yaml(path: Path) -> dict:
    try:
        payload = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _table(path: Path, *, source: str | None = None) -> TableData | None:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    # Native Calphy tables begin with a descriptive comment followed by the
    # machine-readable column header. Select the last comment before data.
    comments = []
    for line in lines:
        if line.startswith("#"):
            comments.append(line[1:].strip())
        elif line.strip():
            break
    header = comments[-1] if comments else ""
    columns = tuple(header.split())
    try:
        values = np.loadtxt(path, comments="#", ndmin=2)
    except (OSError, ValueError):
        return None
    if values.size == 0:
        return None
    return TableData(
        columns=columns, values=np.asarray(values, dtype=float), source=source or str(path)
    )


def _integrand(table: TableData) -> tuple[np.ndarray, np.ndarray | None]:
    columns = [column.lower() for column in table.columns]
    lambda_index = next((i for i, column in enumerate(columns) if "lambda" in column), None)
    if lambda_index is None:
        lambda_index = table.values.shape[1] - 1
    lambdas = table.values[:, lambda_index]
    system = next((i for i, column in enumerate(columns) if "du_sys" in column), None)
    references = [i for i, column in enumerate(columns) if "du_ref" in column]
    if system is None or not references:
        return lambdas, None
    # A Calphy leg switches between the system Hamiltonian and the complete
    # reference Hamiltonian. When the latter is decomposed into several
    # ``dU_ref*`` terms, its energy is their sum.
    return lambdas, table.values[:, system] - table.values[:, references].sum(axis=1)


def _switches(directory: Path, root: Path, prefix: str = "") -> tuple[SwitchingData, ...]:
    records: list[SwitchingData] = []
    for path in sorted(directory.glob(f"{prefix}*.dat")):
        if not prefix and path.name.startswith("ts."):
            continue
        match = _REPLICA.search(path.name)
        if match is None:
            continue
        table = _table(path, source=str(path.relative_to(root)))
        if table is None:
            continue
        lambdas, integrand = _integrand(table)
        records.append(
            SwitchingData(
                match["direction"],
                match["leg"],
                int(match["replica"]),
                lambdas,
                integrand,
                table,
            )
        )
    return tuple(records)


def _phase(directory: Path, root: Path, name: str, source_root: str | None = None) -> PhaseData:
    files = tuple(
        str(path.relative_to(directory)) for path in sorted(directory.rglob("*")) if path.is_file()
    )
    return PhaseData(
        name=name,
        directory=(
            f"{source_root}/{directory.relative_to(root)}" if source_root is not None else str(directory)
        ),
        report=_yaml(directory / "report.yaml"),
        equilibration=_table(directory / "avg.dat", source=str((directory / "avg.dat").relative_to(root))),
        switching=_switches(directory, root),
        temperature_scaling=_switches(directory, root, "ts."),
        free_energy=_table(
            directory / "temperature_sweep.dat",
            source=str((directory / "temperature_sweep.dat").relative_to(root)),
        ),
        files=files,
    )


def read_calphy_directory(path: str | Path, *, source_root: str | None = None) -> CalphyAnalysis:
    """Read a local directory containing the retrieved output of one CalcJob."""
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"Calphy result directory does not exist: {root}")
    files = tuple(str(item.relative_to(root)) for item in sorted(root.rglob("*")) if item.is_file())
    logs = [item for item in root.glob("*.log") if item.is_file()]
    log_records: list[str] = []
    temperature = uncertainty = None
    for log in logs:
        content = log.read_text(errors="replace")
        log_records.extend(
            line for line in content.splitlines() if "STATE:" in line or "WARNING" in line
        )
        for match in _STATE.finditer(content):
            candidate = float(match[1])
            if np.isfinite(candidate) and candidate > 0:
                temperature = candidate
                uncertainty = float(match[2]) if match[2] is not None else None
    ranges: list[tuple[float, float]] = []
    for log in logs:
        for match in _RANGE.finditer(log.read_text(errors="replace")):
            candidate = (float(match[1]), float(match[2]))
            if not ranges or ranges[-1] != candidate:
                ranges.append(candidate)

    # ``ts-...-<temperature>-<pressure>`` uses the final component for
    # pressure. The generated YAML records the actual attempt file and bracket.
    grouped: dict[str, dict[str, object]] = {}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        match = _PHASE.search(directory.name)
        if match is None:
            continue
        parameters = _yaml(directory / "input_file.yaml")
        calculation = (parameters.get("calculations") or [{}])[0]
        inputfile = calculation.get("inputfile")
        temperatures = calculation.get("temperature")
        bracket = (
            (float(temperatures[0]), float(temperatures[1]))
            if isinstance(temperatures, list) and len(temperatures) == 2
            else None
        )
        # The input-file name is Calphy's attempt identity. Fall back to a
        # temperature/pressure identity only for incomplete legacy retrievals.
        key = str(inputfile) if inputfile else f"{match['temperature']}@{match['pressure']}"
        group = grouped.setdefault(
            key,
            {
                "phases": {},
                "bracket": bracket,
                "hint": float(match["temperature"]),
                "index": int(_ATTEMPT_FILE.search(str(inputfile)).group(1))
                if inputfile and _ATTEMPT_FILE.search(str(inputfile))
                else None,
            },
        )
        group["phases"][match["phase"]] = _phase(
            directory, root, match["phase"], source_root=source_root
        )

    def attempt_order(item: tuple[str, dict[str, object]]) -> tuple[int, float, str]:
        key, values = item
        index = values["index"]
        bracket = values["bracket"]
        if isinstance(index, int):
            return (0, index, key)
        if isinstance(bracket, tuple) and bracket in ranges:
            return (1, float(ranges.index(bracket)), key)
        return (2, float(values["hint"]), key)

    attempts = []
    for key, values in sorted(grouped.items(), key=attempt_order):
        phases = values["phases"]
        bracket = values["bracket"]
        attempts.append(
            AttemptData(
                key=key,
                temperature_hint_k=(sum(bracket) / 2 if isinstance(bracket, tuple) else values["hint"]),
                temperature_range_k=bracket if isinstance(bracket, tuple) else None,
                solid=phases.get("solid"),
                liquid=phases.get("liquid"),
            )
        )
    return CalphyAnalysis(
        root=source_root or str(root),
        input_parameters=_yaml(root / "input.yaml"),
        attempts=tuple(attempts),
        melting_temperature_k=temperature,
        uncertainty_k=uncertainty
        if uncertainty and np.isfinite(uncertainty) and uncertainty > 0
        else None,
        log_records=tuple(log_records),
        files=files,
    )


def read_calphy_retrieved(folder: orm.FolderData) -> CalphyAnalysis:
    """Read an AiiDA retrieved FolderData without requiring manual export."""
    with tempfile.TemporaryDirectory(prefix="aiida-melting-calphy-") as temporary:
        root = Path(temporary)
        for directory, _, object_names in folder.base.repository.walk():
            for object_name in object_names:
                relative = Path(directory) / object_name
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with folder.base.repository.open(str(relative), "rb") as source:
                    with target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
        return read_calphy_directory(root, source_root=f"aiida://{folder.uuid}")


def read_calphy_process(identifier: int | str | orm.ProcessNode) -> CalphyAnalysis:
    """Read the retrieved folder of a CalcJob, method workflow, or dispatcher."""
    node = orm.load_node(identifier) if not isinstance(identifier, orm.ProcessNode) else identifier
    if isinstance(node, orm.CalcJobNode) and "retrieved" in node.outputs:
        return read_calphy_retrieved(node.outputs.retrieved)
    descendants = node.called_descendants if isinstance(node, orm.ProcessNode) else []
    for descendant in reversed(descendants):
        if isinstance(descendant, orm.CalcJobNode) and "retrieved" in descendant.outputs:
            return read_calphy_retrieved(descendant.outputs.retrieved)
    raise ValueError(f"No retrieved Calphy CalcJob was found below process {node.pk}.")
