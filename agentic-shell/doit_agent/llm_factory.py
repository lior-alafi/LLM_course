from __future__ import annotations

from litellm import completion

from doit_agent.config import ModelConfig
from doit_agent.llm_client import LLMClient


class LiteLLMClient(LLMClient):
    def __init__(self, config: ModelConfig):
        self.config = config

    def complete_text(self, messages: list[dict[str, str]]) -> str:
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0,
        }
        if self.config.api_key:
            kwargs['api_key'] = self.config.api_key
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base

        try:
            response = completion(**kwargs)
            return response.choices[0].message.content
        except Exception as exc:
            raise RuntimeError(
                f"Could not call LLM model={self.config.model}. "
                f"If you are using Ollama, make sure Ollama is running and api_base is reachable. "
                f"Original error: {exc}"
            ) from exc


class LLMFactory:
    @staticmethod
    def create(config: ModelConfig) -> LLMClient:
        supported = {"gemini", "ollama", "openai", "anthropic", "openrouter"}

        if config.provider not in supported:
            raise ValueError(
                f"Unsupported provider: {config.provider}. "
                f"Supported providers: {sorted(supported)}"
            )

        return LiteLLMClient(config)
