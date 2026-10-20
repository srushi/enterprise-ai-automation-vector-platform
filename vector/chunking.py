"""Chunking: turn warehouse / landing-zone data into searchable documents.

Incident-style chunking strategy: one document per non-successful execution
(failures and warnings are what operators search for), plus one document per
error-catalog entry from the transformed data products. Each document is a
self-contained natural-language paragraph so both embedding search and an
LLM prompt can use it without further joins.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class Document:
    """A single searchable chunk."""

    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "text": self.text, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        return cls(
            doc_id=data["doc_id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
        )


def chunk_execution_records(records: Iterable[dict]) -> list[Document]:
    """Create incident documents from FAILED / WARNING execution records."""
    docs: list[Document] = []
    for rec in records:
        if rec.get("status") == "SUCCESS":
            continue
        error_part = (
            f"Error {rec.get('error_code')}: {rec.get('error_message')}. "
            if rec.get("error_code")
            else ""
        )
        text = (
            f"Integration execution {rec.get('execution_id')} for partner "
            f"{rec.get('partner_name')} ({rec.get('partner_id')}) ran process "
            f"'{rec.get('process_name')}' (transaction {rec.get('transaction_type')}, "
            f"{rec.get('direction')}) from {rec.get('source_system')} to "
            f"{rec.get('target_system')} and finished with status "
            f"{rec.get('status')} at {rec.get('start_time')}. "
            f"{error_part}"
            f"Records processed: {rec.get('records_processed')}, records failed: "
            f"{rec.get('records_failed')}, duration {rec.get('duration_ms')} ms, "
            f"environment {rec.get('environment')}."
        )
        docs.append(
            Document(
                doc_id=f"exec-{rec.get('execution_id')}",
                text=text,
                metadata={
                    "type": "execution_incident",
                    "partner_id": rec.get("partner_id"),
                    "partner_name": rec.get("partner_name"),
                    "status": rec.get("status"),
                    "error_code": rec.get("error_code"),
                    "start_time": rec.get("start_time"),
                },
            )
        )
    return docs


def chunk_error_catalog(rows: Iterable[dict]) -> list[Document]:
    """Create one summary document per error code from the error catalog."""
    docs: list[Document] = []
    for row in rows:
        text = (
            f"Error {row.get('error_code')} — {row.get('error_message')} — occurred "
            f"{row.get('occurrences')} times across {row.get('partners_affected')} "
            f"trading partners between {row.get('first_seen')} and "
            f"{row.get('last_seen')}."
        )
        docs.append(
            Document(
                doc_id=f"error-{row.get('error_code')}",
                text=text,
                metadata={
                    "type": "error_summary",
                    "error_code": row.get("error_code"),
                    "occurrences": row.get("occurrences"),
                },
            )
        )
    return docs


def load_records_from_landing(landing_dir: Path) -> list[dict]:
    """Read every JSONL record under a landing-zone directory."""
    records: list[dict] = []
    for path in sorted(Path(landing_dir).rglob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_error_catalog(products_dir: Path) -> list[dict]:
    """Read the transformed error catalog product, if it exists."""
    path = Path(products_dir) / "error_catalog.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def build_corpus(
    landing_dir: Optional[Path] = None,
    products_dir: Optional[Path] = None,
    warehouse_records: Optional[list[dict]] = None,
) -> list[Document]:
    """Assemble the full document corpus from the available sources."""
    import config

    records = (
        warehouse_records
        if warehouse_records is not None
        else load_records_from_landing(landing_dir or config.LANDING_DIR)
    )
    docs = chunk_execution_records(records)
    docs.extend(
        chunk_error_catalog(load_error_catalog(products_dir or config.PRODUCTS_DIR))
    )
    return docs
