from __future__ import annotations
from pathlib import Path
import json, fcntl
from doit_agent.debug import trace
from doit_agent.json_utils import extract_json_object
from doit_agent.llm_client import LLMClient
from doit_agent.prompt_log import PromptLogger
from doit_agent.prompts import build_context_summary_messages
from doit_agent.schemas import ContextSummaryResult
from doit_agent.state.models import InteractionRecord
class ContextSummaryStore:
    def __init__(self, base_dir:Path|None=None):
        self.base_dir=base_dir or (Path(__file__).resolve().parent.parent/'.doit'); self.base_dir.mkdir(parents=True,exist_ok=True)
        self.path=self.base_dir/'session_summaries.json'; self.lock_path=self.base_dir/'session_summaries.lock'
        if not self.path.exists(): self.path.write_text('{}',encoding='utf-8')
        self.lock_path.touch(exist_ok=True)
        
    @trace
    def get_summary(self, session_id:str)->str|None:
        with self.lock_path.open('r+') as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_SH)
            try: data=self._read()
            finally: fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        item=data.get(session_id); return item.get('summary') if item else None
    @trace
    def get_summarized_count(self, session_id:str)->int:
        with self.lock_path.open('r+') as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_SH)
            try: data=self._read()
            finally: fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        item=data.get(session_id); return item.get('summarized_count', 0) if item else 0
    @trace
    def set_summary(self, session_id:str, summary:str, summarized_count:int)->None:
        with self.lock_path.open('r+') as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                data=self._read(); data[session_id]={'summary':summary,'summarized_count':summarized_count}; self._write(data)
            finally: fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    def _read(self)->dict:
        try:
            text=self.path.read_text(encoding='utf-8').strip(); return json.loads(text) if text else {}
        except Exception as e:
            print(f"[context_summary] Failed to read summaries file: {e}")
            return {}
    def _write(self,data:dict)->None:
        tmp=self.path.with_suffix('.json.tmp'); tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(self.path)
class ContextSummaryService:
    def __init__(self, *, llm:LLMClient, model_name:str, prompt_logger:PromptLogger|None=None, store:ContextSummaryStore|None=None):
        self.llm=llm; self.model_name=model_name; self.prompt_logger=prompt_logger; self.store=store or ContextSummaryStore()
    @trace
    def maybe_update_summary(self, *, session_id:str, records:list[InteractionRecord], threshold:int, recent_keep:int)->None:
        if len(records)<threshold: return
        older=records[:-recent_keep] if recent_keep>0 else records
        if not older: return
        # Skip if no new records have entered the "older" slice since last summarization
        last_count=self.store.get_summarized_count(session_id)
        if len(older)<=last_count: return
        messages=build_context_summary_messages(old_summary=self.store.get_summary(session_id), records=older)
        try:
            raw=self.llm.complete_text(messages)
            data=extract_json_object(raw)
            result=ContextSummaryResult.model_validate(data)
            if self.prompt_logger: self.prompt_logger.log(acdl_spec='DoitContextSummary',model=self.model_name,messages=messages,raw_response=raw,parsed_response=result.model_dump())
            self.store.set_summary(session_id,result.summary,summarized_count=len(older))
        except Exception as e:
            print(f"[context_summary] Failed to update summary: {e}")
            return
    @trace
    def get_summary(self, session_id:str)->str|None: return self.store.get_summary(session_id)
