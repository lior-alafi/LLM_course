from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any
import os
import uuid


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_session_id() -> str:
    """
    Same terminal window should reuse the same DOIT_SESSION_ID.

    Important design choice:
    If DOIT_SESSION_ID is missing, we use "default" rather than a new UUID.
    This preserves basic multi-turn behavior even without shell integration.
    """
    return os.environ.get("DOIT_SESSION_ID", "default")


@dataclass
class InteractionRecord:
    id: str
    timestamp: str
    session_id: str
    cwd: str

    user_query: str

    intent: str
    command: str | None = None
    answer: str | None = None
    explanation: str | None = None

    stdout: str | None = None
    stderr: str | None = None
    returncode: int | None = None

    safety_level: str | None = None
    safety_requires_confirmation: bool | None = None
    user_confirmed: bool | None = None

    model: str | None = None

    @staticmethod
    def create(
        *,
        session_id: str,
        cwd: str,
        user_query: str,
        intent: str,
        command: str | None = None,
        answer: str | None = None,
        explanation: str | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        returncode: int | None = None,
        safety_level: str | None = None,
        safety_requires_confirmation: bool | None = None,
        user_confirmed: bool | None = None,
        model: str | None = None,
    ) -> "InteractionRecord":
        return InteractionRecord(
            id=str(uuid.uuid4()),
            timestamp=utc_now_iso(),
            session_id=session_id,
            cwd=cwd,
            user_query=user_query,
            intent=intent,
            command=command,
            answer=answer,
            explanation=explanation,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            safety_level=safety_level,
            safety_requires_confirmation=safety_requires_confirmation,
            user_confirmed=user_confirmed,
            model=model,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "InteractionRecord":
        return InteractionRecord(**data)
