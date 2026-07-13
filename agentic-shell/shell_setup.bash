# shell_setup.bash — shell integration for doit.
# Source this file from ~/.bashrc or ~/.zshrc.
# The installer adds it to ~/.bashrc automatically.

# One stable session id per terminal window. This allows multi-terminal state.
if [ -z "${DOIT_SESSION_ID:-}" ]; then
  if command -v uuidgen >/dev/null 2>&1; then
    export DOIT_SESSION_ID="$(uuidgen)"
  else
    export DOIT_SESSION_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
  fi
fi

# Optional secure-mode override:
# export DOIT_SECURE=true
# export DOIT_SECURE=false

# Bash: flush commands to ~/.bash_history sooner so doit can inspect recent
# manual user actions. Avoid adding the hook twice if this file is sourced more
# than once.
if [ -n "${BASH_VERSION:-}" ]; then
  shopt -s histappend 2>/dev/null || true
  case ";${PROMPT_COMMAND:-};" in
    *";history -a; history -n;"*) ;;
    *) export PROMPT_COMMAND="history -a; history -n${PROMPT_COMMAND:+; $PROMPT_COMMAND}" ;;
  esac
fi

# Resolve the project directory when this file is sourced.
# Bash provides BASH_SOURCE. Zsh provides ${(%):-%x}. The fallback handles
# manual export of DOIT_SCAFFOLD_DIR if a shell does not expose either.
if [ -n "${BASH_SOURCE[0]:-}" ]; then
  _DOIT_SETUP_FILE="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
  _DOIT_SETUP_FILE="${(%):-%x}"
else
  _DOIT_SETUP_FILE="$0"
fi

_DOIT_SCAFFOLD_DIR="${DOIT_SCAFFOLD_DIR:-$(cd "$(dirname "$_DOIT_SETUP_FILE")" 2>/dev/null && pwd)}"

# Prefer the project venv created by install.bash. This is critical on WSL,
# where calling plain python3 often misses litellm/pydantic.
_DOIT_PROJECT_PY="$_DOIT_SCAFFOLD_DIR/.venv/bin/python"
if [ -x "$_DOIT_PROJECT_PY" ]; then
  _DOIT_PY="$_DOIT_PROJECT_PY"
else
  _DOIT_PY="$(command -v python3 || true)"
fi

# Shell wrapper. It runs the Python agent and intercepts DOIT_EXEC:<command>
# markers so parent-shell commands such as cd/export/source affect the real
# interactive shell.
doit() {
  local _DOIT_BIN="$_DOIT_SCAFFOLD_DIR/doit"

  if [ ! -f "$_DOIT_BIN" ]; then
    echo "doit: cannot locate the doit Python script." >&2
    echo "      Expected: $_DOIT_BIN" >&2
    echo "      Re-run: bash install.bash" >&2
    return 1
  fi

  if [ -z "${_DOIT_PY:-}" ] || [ ! -x "$_DOIT_PY" ]; then
    echo "doit: cannot locate Python." >&2
    echo "      Re-run: bash install.bash" >&2
    return 1
  fi

  local _exec_cmd=""
  local _agent_status=0
  local _stdbuf=""

  if command -v stdbuf >/dev/null 2>&1; then
    _stdbuf="stdbuf -oL"
  fi

  # Keep stdin on /dev/tty for confirmations/clarifications, while streaming
  # stdout/stderr line-by-line through this wrapper.
  while IFS= read -r _line; do
    if [ "${_line#DOIT_EXEC:}" != "$_line" ]; then
      _exec_cmd="${_line#DOIT_EXEC:}"
    else
      printf '%s\n' "$_line"
    fi
  done < <($_stdbuf "$_DOIT_PY" "$_DOIT_BIN" "$@" < /dev/tty 2>&1)
  _agent_status=$?

  if [ -n "$_exec_cmd" ]; then
    printf '\033[0;36m▶ %s\033[0m\n' "$_exec_cmd"
    eval "$_exec_cmd"
    return $?
  fi

  return $_agent_status
}
