from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator


Intent = Literal[
    "execute_command",
    "conversation",
    "answer",
    "clarification",
    "error",
]


class AgentDecision(BaseModel):
    intent: Intent

    command: str | None = None
    answer: str | None = None
    error: str | None = None

    clarification_question: str | None = None
    clarification_options: list[str] = Field(default_factory=list)

    explanation: str | None = None

    @field_validator("clarification_options", mode="before")
    @classmethod
    def none_to_empty_list(cls, value):
        if value is None:
            return []
        return value


SafetyLevel = Literal[
    "read_only",
    "modifies_filesystem",
    "dangerous",
    "unsupported",
]


class SafetyDecision(BaseModel):
    safety_level: SafetyLevel
    requires_confirmation: bool
    allowed: bool
    explanation: str
    source: Literal["rule_based", "llm", "fallback"] = "fallback"


class MemoryCandidate(BaseModel):
    should_store: bool
    key: str | None = None
    value: str | None = None
    reason: str | None = None


class MemoryExtractionResult(BaseModel):
    memories: list[MemoryCandidate] = Field(default_factory=list)

    @field_validator("memories", mode="before")
    @classmethod
    def none_to_empty_memories(cls, value):
        if value is None:
            return []
        return value