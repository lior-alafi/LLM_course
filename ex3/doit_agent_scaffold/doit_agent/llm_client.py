from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def complete_text(self, messages: list[dict[str, str]]) -> str:
        """
        Returns the raw assistant text.
        The caller is responsible for parsing it.
        """
        raise NotImplementedError
