from __future__ import annotations

from doit_agent.config import DoitConfig
from doit_agent.state.store import StateStore
from doit_agent.state.file_store import FileStateStore
from doit_agent.state.redis_store import RedisStateStore


class StateStoreFactory:
    @staticmethod
    def create(config: DoitConfig) -> StateStore:
        backend = config.state.backend

        if backend == "file":
            return FileStateStore()

        if backend == "redis":
            if not config.state.redis_url:
                raise RuntimeError("state.backend=redis requires state.redis_url")
            return RedisStateStore(config.state.redis_url)

        raise ValueError(f"Unsupported state backend: {backend}")
