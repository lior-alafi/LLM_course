from __future__ import annotations

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

    def extract_and_apply(
        self,
        user_query: str,
        cwd: str | None = None,
        command: str | None = None,
    ) -> list[str]:
        existing = self.memory_store.list_memories()
        messages = build_memory_extraction_messages(
            user_query, existing, cwd=cwd, command=command
        )

        try:
            raw = self.llm.complete_text(messages)
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

        actions: list[str] = []

        for candidate in result.memories:
            action = candidate.normalized_action()

            if action == "store" and candidate.key and candidate.value:
                record = self.memory_store.upsert_memory(
                    key=candidate.key,
                    value=candidate.value,
                    source_query=user_query,
                )
                actions.append(f"stored:{record.key}")

            elif action == "delete" and candidate.key:
                deleted = self.memory_store.delete_memory(candidate.key)
                if deleted:
                    actions.append(f"deleted:{candidate.key}")
                else:
                    actions.append(f"delete_missed:{candidate.key}")

        return actions
