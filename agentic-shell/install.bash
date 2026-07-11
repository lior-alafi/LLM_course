#!/usr/bin/env bash
# install.bash — one-shot setup for the doit agent scaffold
# Run from the scaffold directory:  bash install.bash

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { printf "${CYAN}[install]${NC} %s\n" "$*"; }
success() { printf "${GREEN}[install]${NC} ✓ %s\n" "$*"; }
warn()    { printf "${YELLOW}[install]${NC} ⚠ %s\n" "$*"; }
die()     { printf "${RED}[install]${NC} ✗ %s\n" "$*" >&2; exit 1; }

# ── Resolve scaffold dir to an absolute path ──────────────────────────────────
SCAFFOLD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

printf "\n${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
printf "${BOLD}  doit agent scaffold — installer${NC}\n"
printf "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n\n"
info "Scaffold directory: $SCAFFOLD_DIR"

# ── 1. Python venv ────────────────────────────────────────────────────────────
printf "\n${BOLD}[1/5] Setting up Python environment${NC}\n"
info "Creating Python virtual environment (.venv) …"
python3 -m venv "$SCAFFOLD_DIR/.venv"
success "venv created at $SCAFFOLD_DIR/.venv"

# ── 2. Install dependencies ───────────────────────────────────────────────────
printf "\n${BOLD}[2/5] Installing Python dependencies${NC}\n"
info "Installing from requirements.txt …"
"$SCAFFOLD_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$SCAFFOLD_DIR/.venv/bin/pip" install --quiet -r "$SCAFFOLD_DIR/requirements.txt"
success "Dependencies installed (litellm, pydantic)"

# ── 3. Make the doit script executable ───────────────────────────────────────
printf "\n${BOLD}[3/5] Making doit executable${NC}\n"
info "chmod +x doit …"
chmod +x "$SCAFFOLD_DIR/doit"
success "doit is now executable"

# ── 4. Copy doit.cfg.example → ~/doit.cfg (only if it doesn't exist) ──────────
printf "\n${BOLD}[4/5] Configuring doit.cfg${NC}\n"
CFG_DEST="$HOME/doit.cfg"
if [ -f "$CFG_DEST" ]; then
  warn "~/doit.cfg already exists — skipping copy."
  info "If you need to reset it, run: cp $SCAFFOLD_DIR/doit.cfg.example ~/doit.cfg"
else
  cp "$SCAFFOLD_DIR/doit.cfg.example" "$CFG_DEST"
  success "Created ~/doit.cfg from example template"
fi

printf "\n${YELLOW}┌─────────────────────────────────────────────────────┐${NC}\n"
printf "${YELLOW}│  ACTION REQUIRED: edit your config file             │${NC}\n"
printf "${YELLOW}│                                                     │${NC}\n"
printf "${YELLOW}│  nano ~/doit.cfg                                    │${NC}\n"
printf "${YELLOW}│                                                     │${NC}\n"
printf "${YELLOW}│  Set at minimum:                                    │${NC}\n"
printf "${YELLOW}│    provider  = ollama | gemini | openai | ...       │${NC}\n"
printf "${YELLOW}│    model     = e.g. ollama/gemma4:e4b               │${NC}\n"
printf "${YELLOW}│    api_base  = (if using Ollama — your host URL)    │${NC}\n"
printf "${YELLOW}│    api_key_env = (if using a cloud provider)        │${NC}\n"
printf "${YELLOW}└─────────────────────────────────────────────────────┘${NC}\n"

# ── 5. Wire shell_setup.bash into ~/.bashrc ───────────────────────────────────
printf "\n${BOLD}[5/5] Wiring shell function into ~/.bashrc${NC}\n"
SHELL_SETUP="$SCAFFOLD_DIR/shell_setup.bash"
SOURCE_LINE="source \"$SHELL_SETUP\""
BASHRC="$HOME/.bashrc"

# Remove any stale/incorrect lines that reference shell_setup.bash
if grep -q "shell_setup.bash" "$BASHRC" 2>/dev/null; then
  warn "Removing existing shell_setup.bash entries from $BASHRC …"
  tmp="$(mktemp)"
  grep -v "shell_setup.bash" "$BASHRC" > "$tmp"
  mv "$tmp" "$BASHRC"
fi

# Append the correct line
printf '\n# doit agent scaffold — added by install.bash\n%s\n' "$SOURCE_LINE" >> "$BASHRC"
success "Added to $BASHRC: source \"$SHELL_SETUP\""

# ── 6. Source it now for the current session ──────────────────────────────────
# shellcheck disable=SC1090
source "$SHELL_SETUP"
success "Sourced shell_setup.bash — doit is active in this session"

# ── Done ──────────────────────────────────────────────────────────────────────
printf "\n${BOLD}${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
printf "${BOLD}${GREEN}  Installation complete! 🎉${NC}\n"
printf "${BOLD}${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n\n"

printf "${BOLD}Next steps:${NC}\n"
printf "  ${YELLOW}1.${NC} Edit your config:     ${CYAN}nano ~/doit.cfg${NC}\n"
printf "       Set provider, model, api_base / api_key_env\n"
printf "  ${YELLOW}2.${NC} Reload your shell:    ${CYAN}source ~/.bashrc${NC}   (or open a new terminal)\n"
printf "  ${YELLOW}3.${NC} Try it out:\n"
printf "       ${CYAN}doit list files${NC}\n"
printf "       ${CYAN}doit move back a folder${NC}\n"
printf "       ${CYAN}doit -vv \"set FOO to hello\"${NC}\n"
printf "\n${YELLOW}Tip:${NC} For WSL + Windows Ollama, set in ~/doit.cfg:\n"
printf "     api_base = http://\$(ip route | awk '/default/ {print \$3}'):11434\n"
printf "\n"
