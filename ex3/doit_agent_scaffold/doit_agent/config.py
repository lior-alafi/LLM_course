from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import configparser
import os


DEFAULT_CONFIG_PATH = Path.home() / "doit.cfg"


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    api_base: str | None = None
    api_key_env: str | None = None


@dataclass(frozen=True)
class AgentConfig:
    shell: str = "/bin/bash"
    secure: bool = False


@dataclass(frozen=True)
class StateConfig:
    backend: str = "file"
    redis_url: str | None = None


@dataclass(frozen=True)
class MemoryConfig:
    backend: str = "file"
    redis_url: str | None = None


@dataclass(frozen=True)
class DoitConfig:
    model: ModelConfig
    agent: AgentConfig
    state: StateConfig
    memory: MemoryConfig


def _parse_bool_env(value: str | None) -> bool | None:
    if value is None:
        return None

    normalized = value.strip().lower()

    if normalized in {"1", "true", "yes", "y", "on"}:
        return True

    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(
        f"Invalid boolean value for DOIT_SECURE: {value}. "
        "Use true/false, yes/no, 1/0, on/off."
    )


class ConfigLoader:
    def __init__(self, path: Path | None = None):
        self.path = path or DEFAULT_CONFIG_PATH

    def load(self) -> DoitConfig:
        parser = configparser.ConfigParser()

        if self.path.exists():
            parser.read(self.path)

        provider = parser.get("model", "provider", fallback="ollama")
        model = parser.get("model", "model", fallback="ollama/gemma3:4b")
        api_base = parser.get("model", "api_base", fallback=None)
        api_key_env = parser.get("model", "api_key_env", fallback=None)

        shell = parser.get("agent", "shell", fallback="/bin/bash")
        secure_from_cfg = parser.getboolean("agent", "secure", fallback=False)

        secure_from_env = _parse_bool_env(os.getenv("DOIT_SECURE"))
        secure = secure_from_env if secure_from_env is not None else secure_from_cfg

        state_backend = parser.get("state", "backend", fallback="file")
        state_redis_url = parser.get("state", "redis_url", fallback=None)

        memory_backend = parser.get("memory", "backend", fallback="file")
        memory_redis_url = parser.get("memory", "redis_url", fallback=None)

        if api_key_env:
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Config requested api_key_env={api_key_env}, "
                    f"but environment variable is missing."
                )

        return DoitConfig(
            model=ModelConfig(
                provider=provider,
                model=model,
                api_base=api_base,
                api_key_env=api_key_env,
            ),
            agent=AgentConfig(
                shell=shell,
                secure=secure,
            ),
            state=StateConfig(
                backend=state_backend,
                redis_url=state_redis_url,
            ),
            memory=MemoryConfig(
                backend=memory_backend,
                redis_url=memory_redis_url,
            ),
        )
