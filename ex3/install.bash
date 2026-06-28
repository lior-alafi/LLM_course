#!/usr/bin/env bash
# install.bash — one-shot setup for the doit agent scaffold.
# Works on Linux and WSL2.
# Run from the project root: bash install.bash

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { printf "${CYAN}[install]${NC} %s\n" "$*"; }
success() { printf "${GREEN}[install]${NC} ✓ %s\n" "$*"; }
warn()    { printf "${YELLOW}[install]${NC} ⚠ %s\n" "$*"; }
die()     { printf "${RED}[install]${NC} ✗ %s\n" "$*" >&2; exit 1; }

SCAFFOLD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCAFFOLD_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CFG_DEST="$HOME/doit.cfg"
BASHRC="$HOME/.bashrc"
SHELL_SETUP="$SCAFFOLD_DIR/shell_setup.bash"

printf "\n${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
printf "${BOLD}  doit agent scaffold — installer${NC}\n"
printf "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n\n"
info "Project directory: $SCAFFOLD_DIR"

is_wsl() {
  grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null
}

ensure_python() {
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python3 was not found. Install Python 3.10+ and rerun install.bash."
  local version
  version="$($PYTHON_BIN - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
)"
  info "Using Python: $PYTHON_BIN ($version)"
  "$PYTHON_BIN" - <<'PY' || die "Python 3.10+ is required."
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

create_venv() {
  if [ -x "$VENV_DIR/bin/python" ]; then
    success "Existing virtual environment found at $VENV_DIR"
    return 0
  fi

  info "Creating Python virtual environment at $VENV_DIR …"
  if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    warn "Could not create a virtual environment."
    if is_wsl; then
      cat >&2 <<'MSG'

On Ubuntu/WSL this usually means python3-venv is missing.
Run:

  sudo apt update
  sudo apt install -y python3-venv python3-pip
  bash install.bash

MSG
    fi
    exit 1
  fi
  success "venv created"
}

install_dependencies() {
  info "Installing dependencies from requirements.txt …"
  "$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip
  "$VENV_DIR/bin/python" -m pip install --quiet -r "$SCAFFOLD_DIR/requirements.txt"
  success "Dependencies installed"
}

probe_url() {
  local url="$1"
  "$VENV_DIR/bin/python" - "$url" <<'PY' >/dev/null 2>&1
import sys, urllib.request
url = sys.argv[1].rstrip('/') + '/api/tags'
try:
    with urllib.request.urlopen(url, timeout=1.2) as r:
        raise SystemExit(0 if 200 <= r.status < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
}

detect_ollama_api_base() {
  local candidates=()
  candidates+=("http://localhost:11434")
  candidates+=("http://127.0.0.1:11434")

  if is_wsl && command -v ip >/dev/null 2>&1; then
    local win_host
    win_host="$(ip route | awk '/default/ {print $3; exit}')"
    if [ -n "${win_host:-}" ]; then
      candidates+=("http://$win_host:11434")
    fi
  fi

  local url
  for url in "${candidates[@]}"; do
    if probe_url "$url"; then
      printf '%s\n' "$url"
      return 0
    fi
  done
  return 1
}

set_ini_value() {
  local file="$1" section="$2" key="$3" value="$4"
  "$VENV_DIR/bin/python" - "$file" "$section" "$key" "$value" <<'PY'
from pathlib import Path
import sys
file, section, key, value = sys.argv[1:]
path = Path(file)
lines = path.read_text(encoding='utf-8').splitlines()
out = []
in_section = False
seen_section = False
set_key = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith('[') and stripped.endswith(']'):
        if in_section and not set_key:
            out.append(f'{key} = {value}')
            set_key = True
        in_section = stripped == f'[{section}]'
        seen_section = seen_section or in_section
        out.append(line)
        continue
    if in_section and stripped.startswith(f'{key}') and '=' in stripped and not stripped.startswith('#'):
        out.append(f'{key} = {value}')
        set_key = True
    else:
        out.append(line)
if not seen_section:
    out.extend(['', f'[{section}]', f'{key} = {value}'])
elif in_section and not set_key:
    out.append(f'{key} = {value}')
path.write_text('\n'.join(out) + '\n', encoding='utf-8')
PY
}

configure_cfg() {
  if [ -f "$CFG_DEST" ]; then
    warn "~/doit.cfg already exists — not overwriting it."
    info "Current config: $CFG_DEST"
    return 0
  fi

  cp "$SCAFFOLD_DIR/doit.cfg.example" "$CFG_DEST"
  success "Created ~/doit.cfg from doit.cfg.example"

  local api_base
  if api_base="$(detect_ollama_api_base 2>/dev/null)"; then
    set_ini_value "$CFG_DEST" "model" "api_base" "$api_base"
    success "Detected Ollama API base: $api_base"
  else
    warn "Could not auto-detect Ollama. Edit ~/doit.cfg if you use Ollama."
    if is_wsl; then
      info "WSL hint: WIN_HOST=\$(ip route | awk '/default/ {print \$3; exit}')"
      info "Then test: curl http://\$WIN_HOST:11434/api/tags"
    fi
  fi
}

wire_shell_setup() {
  touch "$BASHRC"

  if grep -q "shell_setup.bash" "$BASHRC" 2>/dev/null; then
    warn "Removing old shell_setup.bash entries from $BASHRC …"
    local tmp
    tmp="$(mktemp)"
    grep -v "shell_setup.bash" "$BASHRC" > "$tmp"
    mv "$tmp" "$BASHRC"
  fi

  printf '\n# doit agent scaffold — added by install.bash\nsource "%s"\n' "$SHELL_SETUP" >> "$BASHRC"
  success "Added shell setup to $BASHRC"
}

printf "\n${BOLD}[1/6] Checking Python${NC}\n"
ensure_python

printf "\n${BOLD}[2/6] Setting up virtual environment${NC}\n"
create_venv

printf "\n${BOLD}[3/6] Installing dependencies${NC}\n"
install_dependencies

printf "\n${BOLD}[4/6] Making scripts executable${NC}\n"
chmod +x "$SCAFFOLD_DIR/doit" "$SHELL_SETUP"
success "doit and shell_setup.bash are executable"

printf "\n${BOLD}[5/6] Configuring ~/doit.cfg${NC}\n"
configure_cfg

printf "\n${BOLD}[6/6] Wiring shell wrapper${NC}\n"
wire_shell_setup

# shellcheck disable=SC1090
source "$SHELL_SETUP"
success "Sourced shell_setup.bash in this installer shell"

printf "\n${BOLD}${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
printf "${BOLD}${GREEN}  Installation complete${NC}\n"
printf "${BOLD}${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n\n"

printf "${BOLD}Next steps:${NC}\n"
printf "  ${YELLOW}1.${NC} Check/edit config:  ${CYAN}nano ~/doit.cfg${NC}\n"
printf "  ${YELLOW}2.${NC} Reload shell:       ${CYAN}source ~/.bashrc${NC}  or open a new terminal\n"
printf "  ${YELLOW}3.${NC} Test:               ${CYAN}doit \"list files\"${NC}\n"
printf "  ${YELLOW}4.${NC} Debug:              ${CYAN}doit -vv \"list files\"${NC}\n\n"

if is_wsl; then
  printf "${YELLOW}WSL note:${NC} If Ollama runs on Windows and doit cannot connect, run:\n"
  printf "  ${CYAN}WIN_HOST=\$(ip route | awk '/default/ {print \$3; exit}')${NC}\n"
  printf "  ${CYAN}curl http://\$WIN_HOST:11434/api/tags${NC}\n"
  printf "Then set ${CYAN}api_base = http://\$WIN_HOST:11434${NC} in ~/doit.cfg.\n\n"
fi
