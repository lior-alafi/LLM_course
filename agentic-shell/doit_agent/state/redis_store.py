from __future__ import annotations

import json

from doit_agent.state.models import InteractionRecord
from doit_agent.state.store import StateStore


class RedisStateStore(StateStore):
    """
    Future implementation.

    Expected Redis design:
    - Global list:
        doit:history
    - Per-session list:
        doit:session:{session_id}:history
    - Session index:
        doit:sessions

    Redis list pushes are atomic, so this is naturally safe for many doit processes.
    """

    def __init__(self, redis_url: str):
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "RedisStateStore requires the redis package. "
                "Install with: pip install redis"
            ) from exc

        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)

    def append_interaction(self, record: InteractionRecord) -> None:
        raw = json.dumps(record.to_dict(), ensure_ascii=False)

        pipe = self.redis.pipeline()
        pipe.rpush("doit:history", raw)
        pipe.rpush(f"doit:session:{record.session_id}:history", raw)
        pipe.sadd("doit:sessions", record.session_id)
        pipe.execute()

    def get_recent_interactions(
        self,
        *,
        limit: int = 10,
        session_id: str | None = None,
    ) -> list[InteractionRecord]:
        key = "doit:history"

        if session_id is not None:
            key = f"doit:session:{session_id}:history"

        rows = self.redis.lrange(key, -limit, -1)
        return [InteractionRecord.from_dict(json.loads(row)) for row in rows]

    def get_recent_sessions(
        self,
        *,
        limit: int = 5,
    ) -> dict[str, list[InteractionRecord]]:
        sessions: dict[str, list[InteractionRecord]] = {}

        for session_id in self.redis.smembers("doit:sessions"):
            rows = self.redis.lrange(f"doit:session:{session_id}:history", -limit, -1)
            sessions[session_id] = [
                InteractionRecord.from_dict(json.loads(row)) for row in rows
            ]

        return sessions
