from __future__ import annotations

from abc import ABC, abstractmethod

from doit_agent.memory.models import MemoryRecord


class MemoryStore(ABC):
    @abstractmethod
    def upsert_memory(self, key: str, value: str, source_query: str | None = None) -> MemoryRecord:
        """
        Insert or update a memory by key.
        """
        raise NotImplementedError

    @abstractmethod
    def get_memory(self, key: str) -> MemoryRecord | None:
        raise NotImplementedError

    @abstractmethod
    def list_memories(self) -> list[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def delete_memory(self, key: str) -> bool:
        raise NotImplementedError
