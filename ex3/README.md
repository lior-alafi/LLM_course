# doit agent scaffold v2

This is a modular scaffold for Assignment 3: Agents.

## Added in v2

- `-v` and `-vv` verbose flags.
- Interactive clarifications.
- User shell-history awareness.
- Memory store and memory delete.
- Single-command enforcement.
- Context compaction / summarization extension.
- ACDL files per version.

## Install

Run the one-shot installer — it handles everything: venv, dependencies, config, and shell setup.

```bash
bash install.bash
```

What it does:
1. Creates a `.venv` Python virtual environment
2. Installs dependencies (`litellm`, `pydantic`)
3. Makes the `doit` script executable
4. Copies `doit.cfg.example` → `~/doit.cfg` (skips if already exists)
5. Adds `shell_setup.bash` to `~/.bashrc` (removes any stale entries first)
6. Sources the shell function immediately so `doit` works in the current session

> **After install:** edit `~/doit.cfg` to set your model provider before using `doit`.

## Configure

Open `~/doit.cfg` and set at minimum:

```ini
[model]
provider = ollama          # or: gemini, openai, etc.
model = ollama/gemma4:e4b
api_base = http://<host>:11434   # Ollama only
# api_key_env = GOOGLE_API_KEY   # cloud providers
```

### WSL + Windows Ollama

Find your Windows host IP and point the config at it:

```bash
WIN_HOST=$(ip route | awk '/default/ {print $3}')
curl "http://$WIN_HOST:11434/api/tags"   # verify Ollama is reachable
```

Then in `~/doit.cfg`:

```ini
api_base = http://<WIN_HOST>:11434
```

## Run

```bash
doit "list files"
doit -vv "list files"
doit move back a folder
doit set FOO to hello
```
