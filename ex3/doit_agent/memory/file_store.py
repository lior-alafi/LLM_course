from __future__ import annotations

from pathlib import Path
import json
import fcntl

from doit_agent.debug import trace
from doit_agent.memory.models import MemoryRecord
from doit_agent.memory.store import MemoryStore


class FileMemoryStore(MemoryStore):
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or (Path(__file__).resolve().parent.parent.parent / ".doit")
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.memory_path = self.base_dir / "memories.json"
        self.lock_path = self.base_dir / "memories.lock"

        if not self.memory_path.exists():
            self.memory_path.write_text("{}", encoding="utf-8")

        self.lock_path.touch(exist_ok=True)

    @trace
    def upsert_memory(
        self,
        key: str,
        value: str,
        source_query: str | None = None,
    ) -> MemoryRecord:
        normalized_key = self._normalize_key(key)

        with self.lock_path.open("r+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            try:
                data = self._read_unlocked()

                if normalized_key in data:
                    record = MemoryRecord.from_dict(data[normalized_key])
                    record.update(value=value, source_query=source_query)
                else:
                    record = MemoryRecord.create(
                        key=normalized_key,
                        value=value,
                        source_query=source_query,
                    )

                data[normalized_key] = record.to_dict()
                self._write_unlocked(data)

                return record
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @trace
    def get_memory(self, key: str) -> MemoryRecord | None:
        normalized_key = self._normalize_key(key)

        with self.lock_path.open("r+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)

            try:
                data = self._read_unlocked()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        if normalized_key not in data:
            return None

        return MemoryRecord.from_dict(data[normalized_key])

    @trace
    def list_memories(self) -> list[MemoryRecord]:
        with self.lock_path.open("r+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)

            try:
                data = self._read_unlocked()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        return [MemoryRecord.from_dict(item) for item in data.values()]

    @trace
    def delete_memory(self, key: str) -> bool:
        normalized_key = self._normalize_key(key)

        with self.lock_path.open("r+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            try:
                data = self._read_unlocked()
                existed = normalized_key in data

                if existed:
                    del data[normalized_key]
                    self._write_unlocked(data)

                return existed
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> dict:
        try:
            text = self.memory_path.read_text(encoding="utf-8").strip()
            if not text:
                return {}
            return json.loads(text)
        except Exception as e:
            print(f"[memory/file_store] Failed to read memories: {e}")
            return {}

    def _write_unlocked(self, data: dict) -> None:
        tmp_path = self.memory_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self.memory_path)

    @staticmethod
    def _normalize_key(key: str) -> str:
        return key.strip().lower().replace(" ", "_")
