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

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
chmod +x doit
cp doit.cfg.example ~/doit.cfg
```

## Run

```bash
./doit "list files"
./doit -vv "list files"
```

## WSL + Windows Ollama

```bash
WIN_HOST=$(ip route | awk '/default/ {print $3}')
curl "http://$WIN_HOST:11434/api/tags"
```

Then set in `~/doit.cfg`:

```ini
api_base = http://<WIN_HOST>:11434
```

## Shell setup

This enables the `doit` shell function, which runs commands in your **real shell**
so that `cd`, `export`, `source`, `alias`, etc. actually take effect in your terminal.

```bash
# Replace with the actual path to the scaffold folder
echo "source ~/path/to/doit_agent_scaffold_v2/shell_setup.bash" >> ~/.bashrc
source ~/.bashrc
```

Then use `doit` directly (without `./`):

```bash
doit list files
doit move back a folder
doit set FOO to hello
```
