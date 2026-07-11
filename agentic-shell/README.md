# doit — an LLM-powered agentic shell

`doit` turns a natural-language request into a shell command, runs it, and keeps memory and history across invocations — e.g. `doit "list files"`, `doit "move it back"` (resolves "it" from prior turns), `doit "sort them by date"`.

[Assignment spec](https://yoavg.github.io/llm-class-2025-2026/ass3-agentic-shell/) · report: [`acdl/`](acdl) (per-version agent specs, written alongside implementation, per [ACDL](https://acdlang26.github.io/acdlsite/syntax-reference.html))

## Implementation highlights

- **Command translation & execution** — natural language → single shell command; real shell integration via `shell_setup.bash`, which `eval`s a `DOIT_EXEC:<cmd>` marker so builtins like `cd`/`export` affect the live shell, not a subprocess.
- **Safety** — read-only commands run directly; anything that modifies the filesystem needs confirmation. `RuleBasedSafetyClassifier` (deterministic regex) runs first, with an LLM fallback (`SafetyService`) for ambiguous cases.
- **Single-command enforcement** — `SingleCommandPolicy` rejects chained commands (`;`, `&&`, `||`, backticks, `$(`) so the agent can't smuggle multiple actions into one call.
- **Multi-model support** — provider-agnostic via `litellm`; tested against an API model (Gemini) and local Ollama models (tool-calling and non-tool-calling), configured per-deployment in `doit.cfg`.
- **Multi-turn memory** — two separate stores: session/global **history** (`.doit/history.jsonl`, flock-protected) for "what did we just do," and persistent **memories** (`.doit/memories.json`) for facts that survive terminal restarts (e.g. "this folder is my LLM class project").
- **Clarifications** — the agent asks instead of guessing when a request is ambiguous (e.g. "which date type?"), up to `max_clarification_rounds`.
- **User & output awareness** — reads `~/.bash_history`/`~/.zsh_history` to see commands the user ran manually (not through the agent), and can answer follow-up questions about a prior command's stdout/stderr.
- **Multi-session handling** — tracks a session ID per terminal window so parallel `doit` sessions don't cross-contaminate context.
- **Context summarization** (extension) — compacts older interactions into a rolling summary instead of growing the prompt unbounded.

## Architecture

Request lifecycle (`doit_agent/agent.py:DoitAgent.run`):

`decide()` (LLM call with cwd + history + memories + shell history + context summary) → `clarify()` loop → `memory_service.extract_and_apply()` → safety check → `SingleCommandPolicy.validate()` → execute → save + summarize.

| Module | Role |
|---|---|
| `doit_agent/agent.py` | Orchestrates the full lifecycle |
| `doit_agent/prompts.py` | All LLM message builders |
| `doit_agent/schemas.py` | Pydantic models: `AgentDecision`, `SafetyDecision`, `MemoryExtractionResult`, `ContextSummaryResult` |
| `doit_agent/config.py` | Reads `doit.cfg`; `DOIT_SECURE` env var overrides `secure` |
| `doit_agent/llm_factory.py` | `LiteLLMClient` — provider abstraction over `litellm.completion` |
| `doit_agent/safety.py` | Rule-based classifier + LLM-fallback safety service |
| `doit_agent/memory_service.py` | Extracts memory store/delete requests from a query |
| `doit_agent/context_summary.py` | Rolling session summarization |
| `doit_agent/state/`, `doit_agent/memory/` | Pluggable storage backends (file / Redis) |
| `doit_agent/user_awareness/` | Reads external shell history for context |

## Install

```bash
bash install.bash
```

This creates a `.venv`, installs dependencies (`litellm`, `pydantic`), makes `doit` executable, copies `doit.cfg.example` → `~/doit.cfg`, and wires `shell_setup.bash` into `~/.bashrc`.

## Configure

Edit `~/doit.cfg`:

```ini
[model]
provider = ollama          # or: gemini, openai, anthropic, openrouter
model = ollama/gemma4:e4b
api_base = http://<host>:11434   # Ollama only
# api_key_env = GOOGLE_API_KEY   # cloud providers
```

### WSL + Windows Ollama

```bash
WIN_HOST=$(ip route | awk '/default/ {print $3}')
curl "http://$WIN_HOST:11434/api/tags"   # verify Ollama is reachable
```

## Run

```bash
doit "list files"
doit -v "show git log"      # short LLM debug output
doit -vv "move back"        # full ACDL + context + raw response
```
