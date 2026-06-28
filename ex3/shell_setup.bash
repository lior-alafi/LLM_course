# Add this to ~/.bashrc or ~/.zshrc.
# It creates one stable session id per terminal window.

if [ -z "$DOIT_SESSION_ID" ]; then
  export DOIT_SESSION_ID="$(uuidgen 2>/dev/null || python3 -c 'import uuid; print(uuid.uuid4())')"
fi

# Optional secure-mode override:
# export DOIT_SECURE=true

# Bash only: make commands appear in ~/.bash_history sooner.
# This helps doit inspect recent manual user actions.
if [ -n "$BASH_VERSION" ]; then
  shopt -s histappend
  export PROMPT_COMMAND="history -a; history -n${PROMPT_COMMAND:+; $PROMPT_COMMAND}"
fi

# ---------------------------------------------------------------------------
# Resolve the scaffold directory ONCE when this file is sourced.
# BASH_SOURCE[0] is the path to THIS file at source time, so we resolve it
# to an absolute path immediately.  The doit() function then uses this
# variable so it always finds the script regardless of the caller's cwd.
# ---------------------------------------------------------------------------
_DOIT_SCAFFOLD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"

# ---------------------------------------------------------------------------
# doit shell wrapper
#
# This function is the KEY to making doit commands affect the REAL shell:
#
#   The Python agent never runs commands itself — it just decides what to run
#   and prints a "DOIT_EXEC:<command>" marker.  This wrapper captures that
#   marker and runs the command with `eval` directly inside the current shell
#   process.  That means:
#
#     doit move back a folder     → eval "cd .."         → directory actually changes
#     doit set FOO to bar         → eval "export FOO=bar" → variable actually set
#     doit load my env file       → eval "source .env"   → sourced in this shell
#     doit create alias ll        → eval "alias ll=..."  → alias actually registered
#     doit run git commit         → eval "git commit ..." → runs in real shell
#
#   Any regular output from the agent (explanations, safety messages, etc.)
#   is printed normally.  Only the DOIT_EXEC: marker line is intercepted.
#
# Usage:  source this file from ~/.bashrc or ~/.zshrc, then just type:
#
#     doit <your request in plain English>
# ---------------------------------------------------------------------------
doit() {
  # ── Find the real Python doit script ──────────────────────────────────────
  local _DOIT_BIN="$_DOIT_SCAFFOLD_DIR/doit"

  if [ ! -f "$_DOIT_BIN" ]; then
    echo "doit: cannot locate the doit Python script." >&2
    echo "      Expected: $_DOIT_BIN" >&2
    echo "      Re-source shell_setup.bash from the scaffold directory." >&2
    return 1
  fi

  # ── Stream output line-by-line ─────────────────────────────────────────────
  # We used to capture everything with $(...) so we could scan for DOIT_EXEC:.
  # The problem: $(...) buffers ALL output until Python exits, so nothing is
  # visible while the agent runs — clarification questions, debug logs, errors
  # all hang silently.
  #
  # Fix: pipe Python's output through a while-read loop.  Each line is printed
  # immediately as it arrives.  We intercept only the DOIT_EXEC: marker line.
  # stderr is sent directly to the terminal (not piped) so errors always show.
  #
  # stdbuf -oL forces Python's stdout to line-buffer mode so lines arrive
  # promptly even when the output is a pipe and not a TTY.

  local _PY="$_DOIT_SCAFFOLD_DIR/.venv/bin/python"
  if [ ! -x "$_PY" ]; then
    _PY="python3"
  fi

  local _exec_cmd=""
  local _agent_exit

  while IFS= read -r _line; do
    if [[ "$_line" == DOIT_EXEC:* ]]; then
      _exec_cmd="${_line#DOIT_EXEC:}"
    else
      printf '%s\n' "$_line"
    fi
  done < <(stdbuf -oL "$_PY" "$_DOIT_BIN" "$@" 2>&1)
  _agent_exit=$?

  # ── Execute the intercepted command in THIS shell ──────────────────────────
  if [ -n "$_exec_cmd" ]; then
    printf '\033[0;36m▶ %s\033[0m\n' "$_exec_cmd"
    eval "$_exec_cmd"
    return $?
  fi

  return $_agent_exit
}
