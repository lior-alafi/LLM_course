from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json, uuid
import re
from typing import Any
class PromptLogger:
    def __init__(self, base_dir:Path|None=None, verbose_level:int=0):
        self.base_dir=base_dir or (Path(__file__).resolve().parent.parent/'.doit'/'logs'); self.base_dir.mkdir(parents=True,exist_ok=True); self.verbose_level=verbose_level
    def log(self, *, acdl_spec:str, model:str, messages:list[dict[str,str]], raw_response:str, parsed_response:dict[str,Any]|None=None)->None:
        rec={'id':str(uuid.uuid4()), 'timestamp':datetime.now(timezone.utc).isoformat(), 'acdl_spec':acdl_spec, 'model':model, 'messages':messages, 'raw_response':raw_response, 'parsed_response':parsed_response}
        spec_dir=self.base_dir / _safe_log_dir_name(acdl_spec)
        spec_dir.mkdir(parents=True,exist_ok=True)
        path=spec_dir / f"{rec['timestamp'].replace(':','-')}_{rec['id']}.json"
        path.write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
        self._maybe_print(rec,path)
    def _maybe_print(self, rec:dict[str,Any], path:Path)->None:
        if self.verbose_level<=0: return
        print('\n'+'='*80); print('LLM CALL'); print('='*80)
        print(f"ACDL spec: {rec['acdl_spec']}"); print(f"Model: {rec['model']}"); print(f'Log file: {path}'); print('-'*80)
        if self.verbose_level>=2:
            acdl=_load_acdl_text(rec['acdl_spec'])
            if acdl:
                print('ACDL file content:'); print(acdl); print('-'*80)
        print('Messages sent to LLM:')
        for i,msg in enumerate(rec['messages'],1):
            print(f"\n[{i}] role={msg.get('role','unknown')}")
            content=msg.get('content','')
            print(_shorten(content,800) if self.verbose_level==1 else content)
        print('\nRaw response:'); print(_shorten(rec['raw_response'],1200) if self.verbose_level==1 else rec['raw_response'])
        if rec['parsed_response'] is not None:
            print('\nParsed response:'); print(json.dumps(rec['parsed_response'],ensure_ascii=False,indent=2))
        print('='*80+'\n')
def _shorten(text:str, max_chars:int)->str:
    return text if len(text)<=max_chars else text[:max_chars]+'\n...[truncated]...'
def _safe_log_dir_name(acdl_spec:str)->str:
    name=re.sub(r'[^A-Za-z0-9_.-]+','_',acdl_spec.strip())
    return name.strip('._') or 'unknown'
def _load_acdl_text(acdl_spec:str)->str|None:
    mapping = {
        "DoitAgentStateful": "acdl/prompt/new_doit_agent_stateful.acdl",
        "DoitMemoryExtraction": "acdl/prompt/new_doit_memory_extraction.acdl",
        "DoitSafetyCheck": "acdl/prompt/new_doit_safety_check.acdl",
        "DoitClarification": "acdl/prompt/new_doit_clarification.acdl",
        "DoitContextSummary": "acdl/prompt/new_doit_context_summary.acdl",
    }
    
    rel=mapping.get(acdl_spec)
    if not rel: return None
    for path in [Path.cwd()/rel, Path(__file__).resolve().parent.parent/rel]:
        if path.exists(): return path.read_text(encoding='utf-8')
    return None
