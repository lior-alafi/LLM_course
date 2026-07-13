from __future__ import annotations

from pathlib import Path
import json
import fcntl

from doit_agent.debug import trace
from doit_agent.state.models import InteractionRecord
from doit_agent.state.store import StateStore


class FileStateStore(StateStore):
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or (Path(__file__).resolve().parent.parent.parent / ".doit")
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.history_path = self.base_dir / "history.jsonl"
        self.lock_path = self.base_dir / "history.lock"

        self.history_path.touch(exist_ok=True)
        self.lock_path.touch(exist_ok=True)

    @trace
    def append_interaction(self, record: InteractionRecord) -> None:
        line = json.dumps(record.to_dict(), ensure_ascii=False)

        with self.lock_path.open("r+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            try:
                with self.history_path.open("a", encoding="utf-8") as history_file:
                    history_file.write(line + "\n")
                    history_file.flush()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @trace
    def get_recent_interactions(
        self,
        *,
        limit: int = 10,
        session_id: str | None = None,
    ) -> list[InteractionRecord]:
        with self.lock_path.open("r+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)

            try:
                records = self._read_all_unlocked()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        if session_id is not None:
            records = [r for r in records if r.session_id == session_id]

        return records[-limit:]

    @trace
    def get_recent_sessions(
        self,
        *,
        limit: int = 5,
        max_sessions: int = 5,
    ) -> dict[str, list[InteractionRecord]]:
        with self.lock_path.open("r+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)

            try:
                records = self._read_all_unlocked()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        sessions: dict[str, list[InteractionRecord]] = {}

        for record in reversed(records):
            if record.session_id not in sessions:
                # Every distinct session_id ever seen would otherwise
                # accumulate here forever (bloating every future prompt) --
                # cap how many distinct (most-recent) sessions we track.
                if len(sessions) >= max_sessions:
                    continue
                sessions[record.session_id] = []

            if len(sessions[record.session_id]) < limit:
                sessions[record.session_id].append(record)

        for sid in sessions:
            sessions[sid] = list(reversed(sessions[sid]))

        return sessions

    def _read_all_unlocked(self) -> list[InteractionRecord]:
        records: list[InteractionRecord] = []

        with self.history_path.open("r", encoding="utf-8") as history_file:
            for line in history_file:
                line = line.strip()

                if not line:
                    continue

                try:
                    data = json.loads(line)
                    records.append(InteractionRecord.from_dict(data))
                except Exception as e:
                    print(f"[state/file_store] Skipping corrupted history line: {e}")
                    continue

        return records
