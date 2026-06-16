from __future__ import annotations

import json

from doit_agent.memory.models import MemoryRecord
from doit_agent.memory.store import MemoryStore


class RedisMemoryStore(MemoryStore):
    """
    Redis-backed memory store.

    Key design:
    - Hash:
        doit:memories
    - Each field is a memory key.
    - Each value is a JSON MemoryRecord.
    """

    def __init__(self, redis_url: str):
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "RedisMemoryStore requires the redis package. "
                "Install with: pip install redis"
            ) from exc

        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.key = "doit:memories"

    def upsert_memory(
        self,
        key: str,
        value: str,
        source_query: str | None = None,
    ) -> MemoryRecord:
        normalized_key = self._normalize_key(key)
        existing = self.redis.hget(self.key, normalized_key)

        if existing:
            record = MemoryRecord.from_dict(json.loads(existing))
            record.update(value=value, source_query=source_query)
        else:
            record = MemoryRecord.create(
                key=normalized_key,
                value=value,
                source_query=source_query,
            )

        self.redis.hset(
            self.key,
            normalized_key,
            json.dumps(record.to_dict(), ensure_ascii=False),
        )

        return record

    def get_memory(self, key: str) -> MemoryRecord | None:
        normalized_key = self._normalize_key(key)
        raw = self.redis.hget(self.key, normalized_key)

        if not raw:
            return None

        return MemoryRecord.from_dict(json.loads(raw))

    def list_memories(self) -> list[MemoryRecord]:
        values = self.redis.hvals(self.key)
        return [MemoryRecord.from_dict(json.loads(value)) for value in values]

    def delete_memory(self, key: str) -> bool:
        normalized_key = self._normalize_key(key)
        return bool(self.redis.hdel(self.key, normalized_key))

    @staticmethod
    def _normalize_key(key: str) -> str:
        return key.strip().lower().replace(" ", "_")
