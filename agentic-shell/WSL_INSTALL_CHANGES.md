# WSL/Linux install fixes

Changed files:

- `README.md`
- `install.bash`
- `shell_setup.bash`
- `doit`
- `doit.cfg.example`

Main fixes:

1. `install.bash` now checks Python 3.10+, creates/reuses `.venv`, and prints a clear WSL fix if `python3-venv` is missing.
2. `install.bash` copies `doit.cfg.example` to `~/doit.cfg` and tries to auto-detect a reachable Ollama API base, including Windows Ollama from WSL2 through the default gateway IP.
3. `shell_setup.bash` always prefers `.venv/bin/python`, preventing `ModuleNotFoundError: litellm` when the system Python is used by mistake.
4. `shell_setup.bash` keeps `DOIT_SESSION_ID` stable per terminal and improves bash history flushing for user-awareness.
5. `doit` now loads `~/doit.cfg` through `ConfigLoader()` instead of reading `doit.cfg` from the project directory.
6. `doit.cfg.example` no longer contains a stale hard-coded WSL IP address.
