from __future__ import annotations
from dataclasses import dataclass
import re
@dataclass(frozen=True)
class CommandPolicyResult:
    allowed: bool; reason: str|None=None
class SingleCommandPolicy:
    FORBIDDEN_PATTERNS=[r'\n', r';', r'&&', r'\|\|', r'`', r'\$\(']
    def validate(self, command:str)->CommandPolicyResult:
        c=command.strip()
        if not c: return CommandPolicyResult(False,'Empty command.')
        for pat in self.FORBIDDEN_PATTERNS:
            if re.search(pat,c):
                return CommandPolicyResult(False,'The proposed command appears to contain multiple shell operations. This assignment allows only one command at a time.')
        return CommandPolicyResult(True)
