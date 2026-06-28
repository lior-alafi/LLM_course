from __future__ import annotations

import subprocess

from doit_agent.debug import trace


@trace
def run_shell(
    command: str,
    shell: str = "/bin/bash",
    timeout: int = 20,
) -> dict[str, str | int]:
    """Run a command and capture stdout and stderr separately.

    Uses subprocess.run with capture_output=True so stdout and stderr are
    kept apart and both are available for history and output-awareness queries
    (e.g. "why did that command fail?").  Output is printed by the caller
    (agent.py) after the command completes, which avoids the double-print bug
    that arose when streaming and _emit() both wrote to the terminal.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            executable=shell,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "stdout": e.stdout or "",
            "stderr": ((e.stderr or "") + f"\nCommand timed out after {timeout} seconds."),
            "returncode": 124,
        }
