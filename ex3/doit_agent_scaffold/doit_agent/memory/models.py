from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any
import uuid


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryRecord:
    id: str
    key: str
    value: str
    created_at: str
    updated_at: str
    source_query: str | None = None

    @staticmethod
    def create(
        *,
        key: str,
        value: str,
        source_query: str | None = None,
    ) -> "MemoryRecord":
        now = utc_now_iso()
        return MemoryRecord(
            id=str(uuid.uuid4()),
            key=key,
            value=value,
            created_at=now,
            updated_at=now,
            source_query=source_query,
        )

    def update(self, value: str, source_query: str | None = None) -> None:
        self.value = value
        self.updated_at = utc_now_iso()
        if source_query:
            self.source_query = source_query

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "MemoryRecord":
        return MemoryRecord(**data)
