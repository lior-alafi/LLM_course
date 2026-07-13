from __future__ import annotations

from doit_agent.debug import trace
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

    @trace
    def extract(
        self,
        user_query: str,
        cwd: str | None = None,
        command: str | None = None,
    ) -> MemoryExtractionResult | None:
        """
        Ask the LLM what memory actions this turn implies, WITHOUT writing
        them yet. Callers that gate execution (policy/safety/confirmation)
        should only call `apply()` once a turn is actually going through --
        otherwise a rejected or cancelled command still leaves behind a
        memory of an action that never happened.
        """
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
            return result
        except Exception as e:
            print(f"[memory_service] Failed to extract memories: {e}")
            return None

    @trace
    def apply(
        self,
        result: MemoryExtractionResult | None,
        user_query: str,
    ) -> list[str]:
        if result is None:
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

    @trace
    def extract_and_apply(
        self,
        user_query: str,
        cwd: str | None = None,
        command: str | None = None,
    ) -> list[str]:
        """Convenience wrapper for callers with no execution gate to wait for."""
        return self.apply(self.extract(user_query, cwd=cwd, command=command), user_query)
