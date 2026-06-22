from __future__ import annotations

import subprocess


def run_shell(command: str, shell: str = "/bin/bash") -> dict[str, str | int]:
    try:
        result = subprocess.run(
            command,
            shell=True,
            executable=shell,
            text=True,
            capture_output=True,
            timeout=20,
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Command timed out after 20 seconds.",
            "returncode": 124,
        }
