from os import getenv

DEBUG = getenv("DEBUG", "0") == "1"


class BaseScorer:
    filter_value = -float("Inf")

    """Base class for scorers.

    A scorer is a function that takes a sampled code string, along with diagnostics from the LSP server, and returns a score.
    """

    def __call__(
        self, score: float, code_string: str, diagnostics: list[dict], adding: str
    ) -> float:
        # Calculate the last position in the code string
        lines = code_string[: -(len(adding) - 1)].rstrip().split("\n")
        last_position = {
            "line": len(lines) - 1,
            "character": len(lines[-1]),
        }

        # Filter out diagnostics relating to incomplete code
        filtered_diagnostics = []
        for diagnostic in diagnostics:
            end = diagnostic["range"]["end"]
            if end["line"] > last_position["line"] or (
                end["line"] == last_position["line"]
                and end["character"] >= last_position["character"]
            ):
                # Diagnostics attached to the last position (or after)
                # are considered to be "recoverable" errors once more code
                # is sampled, so we ignore them
                continue

            # Any other diagnostic is allowed to progress to the next stage
            filtered_diagnostics.append(diagnostic)

        if filtered_diagnostics:
            # On error, return the filter value
            error = next((f for f in filtered_diagnostics if f["severity"] == 1), None)
            if error:
                if DEBUG:
                    print("error", score)
                    print("------------------")
                    print(code_string)
                    print("------------------")
                    print(last_position)
                    print("------------------")
                    print(diagnostics)
                    print("------------------")
                    print(filtered_diagnostics)
                    print("------------------")
                return self.filter_value

            # On warning, reduce the score by the number of warnings
            n_warnings = len([f for f in filtered_diagnostics if f["severity"] == 2])

            return score * (1 + n_warnings)

        # Otherwise, return the original score
        return score
