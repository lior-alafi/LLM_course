from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
@dataclass
class ShellHistoryEntry:
    source: str; command: str
class ShellHistoryProvider:
    def __init__(self, home:Path|None=None): self.home=home or Path.home()
    def get_recent_commands(self, limit:int=20)->list[ShellHistoryEntry]:
        entries=[]
        for fn in ['.bash_history','.zsh_history']:
            path=self.home/fn
            if not path.exists(): continue
            try: lines=path.read_text(encoding='utf-8',errors='ignore').splitlines()
            except Exception: continue
            for line in lines:
                cmd=self._parse(line)
                if cmd and not self._is_doit(cmd): entries.append(ShellHistoryEntry(str(path),cmd))
        return entries[-limit:]
    def _parse(self,line:str)->str:
        line=line.strip()
        if not line: return ''
        m=re.match(r'^:\s*\d+:\d+;(.*)$', line)
        return m.group(1).strip() if m else line
    def _is_doit(self,cmd:str)->bool:
        first=cmd.strip().split(maxsplit=1)[0] if cmd.strip() else ''
        return first=='doit' or first.endswith('/doit')
def format_shell_history(entries:list[ShellHistoryEntry])->str:
    if not entries: return 'No recent external shell commands were found.'
    return '\n'.join(f'- {e.command}' for e in entries)
