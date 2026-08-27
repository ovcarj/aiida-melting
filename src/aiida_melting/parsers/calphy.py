"""Parser for direct Calphy melting calculations."""

from __future__ import annotations

from aiida import orm
from aiida.parsers import Parser

from ..calphy import parse_temperature_log


class CalphyParser(Parser):
    """Parse Calphy's final state record and retain compact diagnostics."""

    def parse(self, **kwargs):
        if self.node.exit_status is not None and self.node.exit_status > 0:
            return self.node.exit_code
        try:
            repository = self.retrieved.base.repository
            inventory = sorted(
                str(root / filename)
                for root, _directories, filenames in repository.walk()
                for filename in filenames
            )
        except Exception:
            return self.exit_codes.ERROR_INCOMPLETE_RETRIEVAL

        log_files = [path for path in inventory if path.endswith(".log")]
        if not log_files:
            if "calphy.stderr" in inventory:
                try:
                    with repository.open("calphy.stderr", "r") as handle:
                        if handle.read().strip():
                            return self.exit_codes.ERROR_CALPHY_EXECUTION_FAILED
                except (OSError, UnicodeError):
                    return self.exit_codes.ERROR_PARSER_CORRUPTION
            return self.exit_codes.ERROR_INCOMPLETE_RETRIEVAL
        contents: list[tuple[str, str]] = []
        try:
            for path in log_files:
                with repository.open(path, "r") as handle:
                    contents.append((path, handle.read()))
        except (OSError, UnicodeError):
            return self.exit_codes.ERROR_PARSER_CORRUPTION
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
