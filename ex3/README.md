# doit agent scaffold

`doit` is a command-line LLM shell agent. It receives a natural-language request,
asks an LLM for one shell command or a normal answer, validates the command,
optionally asks for confirmation in secure mode, executes it, and stores history,
outputs, memory, and prompt logs.

The project supports both local Ollama models and hosted providers through
LiteLLM. It was tested on Linux and WSL2.

---

## Repository layout

```text
doit                         # Python entrypoint used by the shell wrapper
doit_agent/                  # agent, config, safety, memory, state, logging
install.bash                 # one-shot installer for Linux / WSL
shell_setup.bash             # shell wrapper that exposes the doit function
doit.cfg.example             # template copied to ~/doit.cfg
acdl/prompt/                 # ACDL specs for actual LLM prompt calls
acdl/scenarios/              # end-to-end ACDL scenario traces for the report
```

---

## Requirements

- Linux or WSL2
- Bash
- Python 3.10+
- Python venv support
- Optional: Ollama, either inside WSL/Linux or running on Windows
- Optional: API key for a hosted provider such as Gemini/OpenAI

On Ubuntu/WSL, if `python3 -m venv` is missing, install it with:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

---

## Install on Linux or WSL

From the project root:

```bash
bash install.bash
```

The installer does the following:

1. Creates `.venv/` inside the project.
2. Installs Python dependencies from `requirements.txt`.
3. Makes `doit` executable.
4. Copies `doit.cfg.example` to `~/doit.cfg` if it does not already exist.
5. Detects a reachable Ollama API base when possible.
6. Adds `source "<project>/shell_setup.bash"` to `~/.bashrc`.
7. Sources the setup file in the current shell so the `doit` function becomes available immediately.

After installation, reload your shell or open a new terminal:

```bash
source ~/.bashrc
```

Then try:

```bash
doit "list files"
doit -vv "list files"
```

---

## Configuration

The runtime configuration is read from:

```text
~/doit.cfg
```

The project-local `doit.cfg` file is not required at runtime. It is safer to keep
machine-specific settings and API keys outside the submitted code directory.

### Local Ollama on Linux

If Ollama is running in the same Linux/WSL environment:

```ini
[model]
provider = ollama
model = ollama/qwen3:4b-instruct
api_base = http://localhost:11434
```

Check that Ollama is reachable:

```bash
curl http://localhost:11434/api/tags
```

### WSL2 with Ollama running on Windows

If Ollama runs on Windows and `doit` runs inside WSL2, `localhost` may not always
resolve to the Windows service. The installer tries to detect the Windows host
address automatically. To check manually:

```bash
WIN_HOST=$(ip route | awk '/default/ {print $3; exit}')
curl "http://$WIN_HOST:11434/api/tags"
```

If the curl command works, set this in `~/doit.cfg`:

```ini
[model]
provider = ollama
model = ollama/qwen3:4b-instruct
api_base = http://<WIN_HOST>:11434
```

For example:

```ini
api_base = http://172.26.160.1:11434
```

The exact IP can change between WSL sessions, so if Ollama suddenly stops being
reachable, re-run:

```bash
bash install.bash
```

or update `~/doit.cfg` manually.

### Hosted Gemini example

Set the environment variable in your shell:

```bash
export GOOGLE_API_KEY="your-key-here"
```

Then configure:

```ini
[model]
provider = gemini
model = gemini/gemini-2.5-flash
api_key_env = GOOGLE_API_KEY
```

Do not commit real API keys to the repository.

---

## Secure mode

In `~/doit.cfg`:

```ini
[agent]
secure = true
```

You can override it per shell session:

```bash
export DOIT_SECURE=false
```

or:

```bash
export DOIT_SECURE=true
```

When secure mode is enabled, read-only commands run directly, while filesystem
modifications require confirmation.

---

## Shell behavior

The installer adds a shell function named `doit`. The function is necessary
because commands like `cd`, `export`, `source`, and `alias` must run in the
current shell, not in a child Python process.

For example:

```bash
doit "move back a folder"
doit "set FOO to hello"
doit "load my .env file"
```

The Python agent prints a marker only when a command must be executed by the
parent shell. The shell wrapper intercepts that marker and evaluates the command
in the current shell.

---

## Verbose logging

```bash
doit -v "list files"
doit -vv "list files"
```

- `-v` prints a compact LLM debug trace.
- `-vv` prints the matched prompt-level ACDL file, full messages, raw response,
  and parsed response.

Logs are written under:

```text
.doit/logs/
```

---

## ACDL documentation

There are two ACDL levels:

```text
acdl/prompt/      # actual LLM prompt contexts used by PromptLogger
acdl/scenarios/   # full end-to-end runtime scenarios for report visuals
```

`PromptLogger` should point only to files under `acdl/prompt/`, because each log
entry corresponds to one LLM call. Scenario ACDL files describe complete `doit`
invocations and are used for documentation and visualizations.

---

## Troubleshooting

### `python3 -m venv` fails on WSL

Install venv support:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
```

Then re-run:

```bash
bash install.bash
```

### `ModuleNotFoundError: litellm`

This usually means the shell wrapper is not using the project virtual
environment. Re-run the installer and reload your shell:

```bash
bash install.bash
source ~/.bashrc
```

### Ollama cannot be reached from WSL

Try:

```bash
WIN_HOST=$(ip route | awk '/default/ {print $3; exit}')
curl "http://$WIN_HOST:11434/api/tags"
```

If that works, set:

```ini
api_base = http://<WIN_HOST>:11434
```

in `~/doit.cfg`.

### `doit: cannot locate the doit Python script`

Your shell is sourcing an old `shell_setup.bash` path. Re-run:

```bash
bash install.bash
source ~/.bashrc
```
