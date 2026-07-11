from __future__ import annotations
from dataclasses import dataclass
import re

# Commands that mutate the shell itself (cwd, env, aliases, sourced files).
# These MUST run in the parent shell via the DOIT_EXEC eval path; a subprocess
# would change only its own throwaway environment. Everything else can run
# in-process via run_shell so we can capture stdout/stderr/returncode.
PARENT_SHELL_BUILTINS = frozenset({
    "cd", "export", "source", ".", "alias", "unalias", "set", "unset",
    "pushd", "popd", "umask", "eval", "exec",
})


def needs_parent_shell(command: str) -> bool:
    # ponytail: first-token check; upgrade to a real tokenizer only if a builtin slips through.
    first = command.strip().split(maxsplit=1)[0] if command.strip() else ""
    return first in PARENT_SHELL_BUILTINS


@dataclass(frozen=True)
class CommandPolicyResult:
    allowed: bool; reason: str|None=None
class SingleCommandPolicy:
    # `;` separates statements, but an escaped `\;` (e.g. find -exec ... \;) is
    # part of a single command — only reject unescaped semicolons.
    FORBIDDEN_PATTERNS=[r'\n', r'(?<!\\);', r'&&', r'\|\|', r'`', r'\$\(']
    def validate(self, command: str) -> CommandPolicyResult:
        c = command.strip()
        if not c:
            return CommandPolicyResult(False, 'Empty command.')
        
        # Mask characters inside quotes to avoid false positives
        masked = []
        state = "normal"
        escape = False
        for char in c:
            if escape:
                escape = False
                masked.append(char if state == "normal" else ' ')
                continue
            if char == '\\' and state != "sq":
                escape = True
                masked.append(char if state == "normal" else ' ')
                continue
            
            if state == "normal":
                if char == "'":
                    state = "sq"
                    masked.append(' ')
                elif char == '"':
                    state = "dq"
                    masked.append(' ')
                else:
                    masked.append(char)
            elif state == "sq":
                if char == "'":
                    state = "normal"
                masked.append(' ')
            elif state == "dq":
                if char == '"':
                    state = "normal"
                masked.append(' ')
                
        masked_command = "".join(masked)

        for pat in self.FORBIDDEN_PATTERNS:
            if re.search(pat, masked_command):
                return CommandPolicyResult(False, 'The proposed command appears to contain multiple shell operations. This assignment allows only one command at a time.')
        return CommandPolicyResult(True)
