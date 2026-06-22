from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, field_validator
Intent=Literal['execute_command','conversation','answer','clarification','error']
class AgentDecision(BaseModel):
    intent: Intent
    command: str|None=None; answer: str|None=None; error: str|None=None
    clarification_question: str|None=None
    clarification_options: list[str]=Field(default_factory=list)
    explanation: str|None=None
    @field_validator('clarification_options', mode='before')
    @classmethod
    def none_to_empty_list(cls,v): return [] if v is None else v
    @field_validator('clarification_question','command','answer','error','explanation', mode='before')
    @classmethod
    def coerce_to_str_or_none(cls,v):
        # Local models sometimes output [] or {} for optional string fields.
        if v is None or v == [] or v == {}: return None
        if isinstance(v, list): return ' '.join(str(i) for i in v) if v else None
        return v
SafetyLevel=Literal['read_only','modifies_filesystem','dangerous','unsupported']
class SafetyDecision(BaseModel):
    safety_level: SafetyLevel; requires_confirmation: bool; allowed: bool; explanation: str
    source: Literal['rule_based','llm','fallback']='fallback'
MemoryAction=Literal['store','delete','none']
class MemoryCandidate(BaseModel):
    action: MemoryAction='none'; key: str|None=None; value: str|None=None; reason: str|None=None
    should_store: bool|None=None
    def normalized_action(self)->MemoryAction:
        if self.action!='none': return self.action
        if self.should_store: return 'store'
        return 'none'
class MemoryExtractionResult(BaseModel):
    memories: list[MemoryCandidate]=Field(default_factory=list)
    @field_validator('memories', mode='before')
    @classmethod
    def none_to_empty_memories(cls,v): return [] if v is None else v
class ContextSummaryResult(BaseModel):
    summary: str
