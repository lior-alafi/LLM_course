# doit agent scaffold

This is a modular scaffold for Assignment 3: Agents.

## Features included

- `doit` CLI executable.
- `ConfigLoader` that loads `~/doit.cfg`.
- `DOIT_SECURE` environment override.
- `LLMFactory` using LiteLLM.
- Gemini / Ollama / OpenAI-style model switching.
- `StateStore` ABC.
- `FileStateStore` with JSONL and `fcntl.flock`.
- `RedisStateStore` implementation/skeleton.
- `MemoryStore` ABC.
- `FileMemoryStore` with JSON and lock.
- `RedisMemoryStore`.
- Rule-based safety classifier.
- Optional LLM safety classifier.
- Prompt logs under `~/.doit/logs`.
- ACDL files for the stateful agent and memory extraction.

## Install

```bash
pip install -r requirements.txt
```

Copy config:

```bash
cp doit.cfg.example ~/doit.cfg
```

Make CLI executable:

```bash
chmod +x doit
```

Add the project folder to PATH, or copy `doit` somewhere already in PATH.

## Shell session id

Add this to `~/.bashrc` or `~/.zshrc`:

```bash
if [ -z "$DOIT_SESSION_ID" ]; then
  export DOIT_SESSION_ID="$(uuidgen 2>/dev/null || python3 -c 'import uuid; print(uuid.uuid4())')"
fi
```

Without this hook, all invocations use the fallback session id `default`.
This preserves basic multi-turn behavior, but does not separate terminal windows.

## Example

```bash
doit "list files"
doit "sort them by date"
DOIT_SECURE=true doit "create a folder called data"
```
