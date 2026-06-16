from __future__ import annotations

from doit_agent.config import DoitConfig
from doit_agent.memory.store import MemoryStore
from doit_agent.memory.file_store import FileMemoryStore
from doit_agent.memory.redis_store import RedisMemoryStore


class MemoryStoreFactory:
    @staticmethod
    def create(config: DoitConfig) -> MemoryStore:
        backend = config.memory.backend

        if backend == "file":
            return FileMemoryStore()

        if backend == "redis":
            if not config.memory.redis_url:
                raise RuntimeError("memory.backend=redis requires memory.redis_url")
            return RedisMemoryStore(config.memory.redis_url)

        raise ValueError(f"Unsupported memory backend: {backend}")
