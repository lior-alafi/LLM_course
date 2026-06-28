from __future__ import annotations

import subprocess
import sys
import threading

from doit_agent.debug import trace


@trace
def run_shell(
    command: str,
    shell: str = "/bin/bash",
    timeout: int = 20,
    stream: bool = True,
) -> dict[str, str | int]:
    """Run a command, streaming its output live while also capturing it.

    subprocess.run(capture_output=True) blocks until the command finishes and
    only then hands back all output — so the user sees nothing until the end.
    Here we read the merged stdout/stderr line by line, print each line as it
    arrives (when stream=True), and accumulate the full text for history.

    stdout and stderr are merged so output appears in the real order it was
    produced. The combined text is returned under "stdout".
    # ponytail: merged stream; split into separate stdout/stderr only if a
    # feature needs them apart (two-pipe reader threads).
    """
    proc = subprocess.Popen(
        command,
        shell=True,
        executable=shell,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,  # line-buffered
    )

    timed_out = threading.Event()

    def _kill() -> None:
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout, _kill)
    timer.start()

    captured: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            captured.append(line)
            if stream:
                sys.stdout.write(line)
                sys.stdout.flush()
    finally:
        timer.cancel()
        proc.wait()

    out = "".join(captured)
    returncode = proc.returncode

    if timed_out.is_set():
        msg = f"Command timed out after {timeout} seconds.\n"
        if stream:
            sys.stdout.write(msg)
            sys.stdout.flush()
        out += msg
        returncode = 124

    return {"stdout": out, "stderr": "", "returncode": returncode}
