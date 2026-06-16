from __future__ import annotations

from pydantic import ValidationError

from doit_agent.json_utils import extract_json_object
from doit_agent.llm_client import LLMClient
from doit_agent.memory.store import MemoryStore
from doit_agent.prompt_log import PromptLogger
from doit_agent.prompts import build_memory_extraction_messages
from doit_agent.schemas import MemoryExtractionResult


class MemoryService:
    def __init__(
        self,
        *,
        llm: LLMClient,
        memory_store: MemoryStore,
        model_name: str,
        prompt_logger: PromptLogger | None = None,
    ):
        self.llm = llm
        self.memory_store = memory_store
        self.model_name = model_name
        self.prompt_logger = prompt_logger

    def extract_and_store(self, user_query: str) -> list[str]:
        existing = self.memory_store.list_memories()
        messages = build_memory_extraction_messages(user_query, existing)
        raw = self.llm.complete_text(messages)

        try:
            data = extract_json_object(raw)
            result = MemoryExtractionResult.model_validate(data)

            if self.prompt_logger:
                self.prompt_logger.log(
                    acdl_spec="DoitMemoryExtraction",
                    model=self.model_name,
                    messages=messages,
                    raw_response=raw,
                    parsed_response=result.model_dump(),
                )

        except Exception:
            return []

        stored_keys: list[str] = []

        for candidate in result.memories:
            if (
                candidate.should_store
                and candidate.key
                and candidate.value
            ):
                record = self.memory_store.upsert_memory(
                    key=candidate.key,
                    value=candidate.value,
                    source_query=user_query,
                )
                stored_keys.append(record.key)

        return stored_keys
