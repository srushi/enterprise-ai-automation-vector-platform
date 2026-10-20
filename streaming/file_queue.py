"""File-backed message queue with a Kafka-like produce/consume interface.

Used as the local fallback when no Kafka broker is available. Each topic is
an append-only JSONL file; consumers track their offset in a sidecar file so
restarts resume where they left off — the same at-least-once semantics the
Kafka consumer in ``consumer.py`` provides.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import config


class FileQueue:
    """Append-only JSONL topic store with per-consumer offsets."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else config.STREAM_QUEUE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _topic_path(self, topic: str) -> Path:
        return self.base_dir / f"{topic}.jsonl"

    def _offset_path(self, topic: str, consumer: str) -> Path:
        return self.base_dir / f".offset-{topic}-{consumer}"

    def produce(self, topic: str, message: dict) -> None:
        with self._topic_path(topic).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(message) + "\n")

    def poll(
        self, topic: str, consumer: str = "default", max_messages: int = 1000
    ) -> list[dict]:
        """Return new messages for ``consumer`` and advance its offset."""
        path = self._topic_path(topic)
        if not path.exists():
            return []
        offset_file = self._offset_path(topic, consumer)
        offset = (
            int(offset_file.read_text(encoding="utf-8").strip())
            if offset_file.exists()
            else 0
        )
        lines = path.read_text(encoding="utf-8").splitlines()
        new_lines = lines[offset : offset + max_messages]
        if new_lines:
            offset_file.write_text(str(offset + len(new_lines)), encoding="utf-8")
        return [json.loads(line) for line in new_lines if line.strip()]

    def pending(self, topic: str, consumer: str = "default") -> int:
        path = self._topic_path(topic)
        if not path.exists():
            return 0
        offset_file = self._offset_path(topic, consumer)
        offset = (
            int(offset_file.read_text(encoding="utf-8").strip())
            if offset_file.exists()
            else 0
        )
        total = sum(1 for _ in path.open(encoding="utf-8"))
        return max(0, total - offset)
