from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import uuid
from typing import Any


class PromptLogger:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or (Path.home() / ".doit" / "logs")
        self.base_dir.mkdir(parents=True, exist_ok=True)

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
