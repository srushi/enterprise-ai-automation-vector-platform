"""Streaming telemetry consumer: topic -> landing zone.

Consumes telemetry events (from Kafka when configured, otherwise the local
file queue) and appends them as JSONL into the landing zone's stream area,
where the warehouse loader / vector indexer can pick them up — closing the
streaming loop with the same interface as the batch ELT path.

Usage:
    python -m streaming.consumer [--topic telemetry] [--max-messages 500]
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config
from streaming.file_queue import FileQueue
from streaming.producer import DEFAULT_TOPIC

CONSUMER_NAME = "landing-writer"


class TelemetryConsumer:
    """Reads telemetry events and lands them in the S3-simulated landing zone."""

    def __init__(self, topic: str = DEFAULT_TOPIC) -> None:
        self.topic = topic
        self._kafka = None
        self._queue: Optional[FileQueue] = None
        bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
        if bootstrap:
            try:
                from kafka import KafkaConsumer  # optional dependency

                self._kafka = KafkaConsumer(
                    topic,
                    bootstrap_servers=bootstrap.split(","),
                    group_id="landing-writer",
                    auto_offset_reset="earliest",
                    value_deserializer=lambda b: json.loads(b.decode("utf-8")),
                    consumer_timeout_ms=3000,
                )
            except Exception:
                self._kafka = None
        if self._kafka is None:
            self._queue = FileQueue()

    @property
    def transport(self) -> str:
        return "kafka" if self._kafka is not None else "file-queue"

    def poll(self, max_messages: int = 500) -> list[dict]:
        if self._kafka is not None:
            events = []
            for message in self._kafka:
                events.append(message.value)
                if len(events) >= max_messages:
                    break
            return events
        assert self._queue is not None
        return self._queue.poll(
            self.topic, consumer=CONSUMER_NAME, max_messages=max_messages
        )

    def close(self) -> None:
        if self._kafka is not None:
            self._kafka.close()


def land_events(events: list[dict], landing_dir: Optional[Path] = None) -> Path:
    """Append events to a timestamped JSONL file in the stream landing area."""
    target_dir = Path(landing_dir) if landing_dir else config.LANDING_STREAM_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = target_dir / f"telemetry_{stamp}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Consume telemetry -> landing zone.")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--max-messages", type=int, default=500)
    args = parser.parse_args(argv)

    consumer = TelemetryConsumer(topic=args.topic)
    try:
        events = consumer.poll(max_messages=args.max_messages)
    finally:
        consumer.close()
    if events:
        path = land_events(events)
        print(
            f"Consumed {len(events)} events via {consumer.transport} -> {path}"
        )
    else:
        print(f"No new events on topic '{args.topic}' via {consumer.transport}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
