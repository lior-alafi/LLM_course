from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import uuid
from typing import Any
import os

class PromptLogger:
    def __init__(
        self,
        base_dir: Path | None = None,
        verbose_level: int = 0,
    ):
        self.base_dir = base_dir or (Path.home() / ".doit" / "logs")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.verbose_level = verbose_level

    def log(
        self,
        *,
        acdl_spec: str,
        model: str,
        messages: list[dict[str, str]],
        raw_response: str,
        parsed_response: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "acdl_spec": acdl_spec,
            "model": model,
            "messages": messages,
            "raw_response": raw_response,
            "parsed_response": parsed_response,
        }

        filename = f"{record['timestamp'].replace(':', '-')}_{record['id']}.json"
        path = self.base_dir / filename

        path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self._maybe_print(record)

    def _maybe_print(self, record: dict[str, Any]) -> None:
        if self.verbose_level <= 0:
            return

        print("\n" + "=" * 80)
        print("LLM CALL")
        print("=" * 80)
        print(f"ACDL spec: {record['acdl_spec']}")
        print(f"Model: {record['model']}")
        print(f"Log file: {self.base_dir}")
        print("-" * 80)

        if self.verbose_level >= 2:
            acdl_text = _load_acdl_text(record["acdl_spec"])
            if acdl_text:
                print("ACDL file content:")
                print(acdl_text)
                print("-" * 80)
                
        if self.verbose_level == 1:
            print("Messages sent to LLM:")
            for i, msg in enumerate(record["messages"], start=1):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                print(f"\n[{i}] role={role}")
                print(_shorten(content, 800))

            print("\nRaw response:")
            print(_shorten(record["raw_response"], 1200))

            if record["parsed_response"] is not None:
                print("\nParsed response:")
                print(json.dumps(record["parsed_response"], ensure_ascii=False, indent=2))

            print("=" * 80 + "\n")
            return

        # -vv and above
        print("Messages sent to LLM:")
        for i, msg in enumerate(record["messages"], start=1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            print(f"\n[{i}] role={role}")
            print(content)

        print("\nRaw response:")
        print(record["raw_response"])

        if record["parsed_response"] is not None:
            print("\nParsed response:")
            print(json.dumps(record["parsed_response"], ensure_ascii=False, indent=2))

        print("=" * 80 + "\n")


def _shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n...[truncated]..."


def _load_acdl_text(acdl_spec: str) -> str | None:
    mapping = {
        "DoitAgentStateful": "acdl/doit_agent_stateful.acdl",
        "DoitMemoryExtraction": "acdl/doit_memory_extraction.acdl",
        "DoitSafetyCheck": "acdl/doit_agent_stateful.acdl",
    }

    rel_path = mapping.get(acdl_spec)
    if not rel_path:
        return None

    cwd_path = Path.cwd() / rel_path
    if cwd_path.exists():
        return cwd_path.read_text(encoding="utf-8")

    # Also try relative to the package root / project root.
    project_path = Path(__file__).resolve().parent.parent / rel_path
    if project_path.exists():
        return project_path.read_text(encoding="utf-8")

    return None