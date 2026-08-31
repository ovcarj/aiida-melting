"""Parser for direct Calphy melting calculations."""

from __future__ import annotations

from aiida import orm
from aiida.parsers import Parser

from ..calphy import classify_calphy_failure, parse_temperature_log

FAILURE_EXIT_CODES = {
    "calphy_execution_failed": "ERROR_CALPHY_EXECUTION_FAILED",
    "calphy_input_rejected": "ERROR_CALPHY_INPUT_REJECTED",
    "lammps_style_unavailable": "ERROR_LAMMPS_STYLE_UNAVAILABLE",
    "lammps_runtime_failed": "ERROR_LAMMPS_RUNTIME_FAILED",
    "melting_attempts_exhausted": "ERROR_MELTING_ATTEMPTS_EXHAUSTED",
}


class CalphyParser(Parser):
    """Parse Calphy's final state record and retain compact diagnostics."""

    def parse(self, **kwargs):
        try:
            repository = self.retrieved.base.repository
            inventory = sorted(
                str(root / filename)
                for root, _directories, filenames in repository.walk()
                for filename in filenames
            )
        except Exception:
            if self.node.exit_status is not None and self.node.exit_status > 0:
                return self.node.exit_code
            return self.exit_codes.ERROR_INCOMPLETE_RETRIEVAL

        diagnostic_paths = [
            path
            for path in inventory
            if path.endswith((".log", ".out", ".err"))
            or path in {"calphy.stdout", "calphy.stderr"}
        ]
        diagnostics_by_path: list[tuple[str, str]] = []
        try:
            for path in diagnostic_paths:
                with repository.open(path, "r") as handle:
                    diagnostics_by_path.append((path, handle.read()))
        except (OSError, UnicodeError):
            return self.exit_codes.ERROR_PARSER_CORRUPTION

        diagnostic_text = "\n".join(text for _path, text in diagnostics_by_path)
        if classification := classify_calphy_failure(diagnostic_text):
            return getattr(self.exit_codes, FAILURE_EXIT_CODES[classification])
        if self.node.exit_status is not None and self.node.exit_status > 0:
            return self.node.exit_code
        if "input.yaml" not in inventory:
            return self.exit_codes.ERROR_INCOMPLETE_RETRIEVAL

        log_files = [path for path in inventory if path.endswith(".log")]
        if not log_files:
            if diagnostic_text.strip():
                return self.exit_codes.ERROR_CALPHY_EXECUTION_FAILED
            return self.exit_codes.ERROR_INCOMPLETE_RETRIEVAL
        contents = [(path, text) for path, text in diagnostics_by_path if path.endswith(".log")]
        candidates = [(path, text) for path, text in contents if "STATE: Tm =" in text]
        if not candidates:
            return self.exit_codes.ERROR_MALFORMED_TEMPERATURE
        path, text = candidates[-1]
        calculation = self.node.inputs.parameters.get_dict()["calculations"][0]
        n_iterations = calculation["n_iterations"]
        try:
            parsed = parse_temperature_log(text, n_iterations)
        except ArithmeticError:
            return self.exit_codes.ERROR_INVALID_TEMPERATURE
        except ValueError:
            return self.exit_codes.ERROR_MALFORMED_TEMPERATURE
        except Exception:
            return self.exit_codes.ERROR_PARSER_CORRUPTION

        warnings = [line.strip() for line in text.splitlines() if "WARNING" in line.upper()]
        diagnostics = {
            "source_log": path,
            "state_records": [line.strip() for line in text.splitlines() if "STATE:" in line],
            "warnings": warnings,
            "reliability_warnings": parsed["reliability_warnings"],
            "uncertainty_available": parsed["uncertainty_available"],
            "uncertainty_K": parsed["uncertainty"],
            "n_iterations": n_iterations,
        }
        self.out("melting_temperature", orm.Float(parsed["temperature"]))
        self.out("status", orm.Str(parsed["status"]))
        if parsed["uncertainty_available"]:
            self.out("uncertainty", orm.Float(parsed["uncertainty"]))
        self.out(
            "calculation_metadata",
            orm.Dict(
                dict={
                    "mode": calculation["mode"],
                    "execution_mode": calculation["execution_mode"],
                    "lammps_ranks": calculation["queue"]["cores"],
                    "elements": calculation["element"],
                    "pressure_bar": calculation["pressure"],
                    "seed": calculation["md"]["seed"],
                    "n_iterations": n_iterations,
                }
            ),
        )
        self.out("diagnostics", orm.Dict(dict=diagnostics))
        self.out("retrieved_files", orm.List(list=inventory))
        return None
