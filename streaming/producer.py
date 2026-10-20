"""Streaming telemetry producer.

Emits integration-platform telemetry events (atom heartbeat, queue depth,
execution activity). Transport selection:

* If ``kafka-python`` is installed AND ``KAFKA_BOOTSTRAP_SERVERS`` is set,
  events go to a real Kafka topic.
* Otherwise events go to the local file queue (same produce interface).

Usage:
    python -m streaming.producer --count 50 [--topic telemetry]
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Optional

from ingestion.log_generator import generate_telemetry_event
from streaming.file_queue import FileQueue

DEFAULT_TOPIC = "integration-telemetry"


class TelemetryProducer:
    """Produces telemetry events to Kafka or the local file queue."""

    def __init__(self, topic: str = DEFAULT_TOPIC) -> None:
        self.topic = topic
        self._kafka = None
        self._queue: Optional[FileQueue] = None
        bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
        if bootstrap:
            try:
                from kafka import KafkaProducer  # optional dependency

                self._kafka = KafkaProducer(
                    bootstrap_servers=bootstrap.split(","),
                    value_serializer=lambda v: __import__("json")
                    .dumps(v)
                    .encode("utf-8"),
                )
            except Exception:
                self._kafka = None
        if self._kafka is None:
            self._queue = FileQueue()

    @property
    def transport(self) -> str:
        return "kafka" if self._kafka is not None else "file-queue"

    def send(self, event: dict) -> None:
        if self._kafka is not None:
            self._kafka.send(self.topic, value=event)
        else:
            assert self._queue is not None
            self._queue.produce(self.topic, event)

    def flush(self) -> None:
        if self._kafka is not None:
            self._kafka.flush()

    def close(self) -> None:
        if self._kafka is not None:
            self._kafka.close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Produce telemetry events.")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    producer = TelemetryProducer(topic=args.topic)
    try:
        for _ in range(args.count):
            producer.send(generate_telemetry_event(rng))
        producer.flush()
    finally:
        producer.close()
    print(
        f"Produced {args.count} telemetry events to topic "
        f"'{args.topic}' via {producer.transport}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
