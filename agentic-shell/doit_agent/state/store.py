from __future__ import annotations

from abc import ABC, abstractmethod

from doit_agent.state.models import InteractionRecord


class StateStore(ABC):
    @abstractmethod
    def append_interaction(self, record: InteractionRecord) -> None:
        """
        Append one interaction atomically.
        Must be safe when two doit processes write at the same time.
        """
        raise NotImplementedError

    @abstractmethod
    def get_recent_interactions(
        self,
        *,
        limit: int = 10,
        session_id: str | None = None,
    ) -> list[InteractionRecord]:
        """
        Return recent interactions.
        If session_id is provided, return only records from that session.
        """
        raise NotImplementedError

    @abstractmethod
    def get_recent_sessions(
        self,
        *,
        limit: int = 5,
        max_sessions: int = 5,
    ) -> dict[str, list[InteractionRecord]]:
        """
        Return recent interactions grouped by session: up to `limit` records
        from each of at most `max_sessions` distinct sessions.
        Useful for multi-terminal awareness.
        """
        raise NotImplementedError
