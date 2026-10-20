"""Synthetic industrial / B2B integration execution-log generator.

Produces realistic Dell-Boomi-style execution logs for semiconductor supply
chain integrations (purchase orders, ASNs, invoices, inventory feeds) across a
set of trading partners. The output mimics what an integration platform lands
in an S3 raw zone: one JSON object per execution, newline-delimited.

Usage:
    python -m ingestion.log_generator --count 500 --out data/landing/raw
    python -m ingestion.log_generator --count 60 --out data/sample --sample
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

PARTNERS = [
    ("PTNR-1001", "GlobalFoundries"),
    ("PTNR-1002", "TSMC Logistics"),
    ("PTNR-1003", "ASE Group"),
    ("PTNR-1004", "Amkor Technology"),
    ("PTNR-1005", "Samsung Semiconductor"),
    ("PTNR-1006", "Applied Materials"),
    ("PTNR-1007", "Lam Research"),
    ("PTNR-1008", "KLA Corporation"),
    ("PTNR-1009", "ASML Supply Ops"),
    ("PTNR-1010", "Broadcom Distribution"),
]

PROCESSES = [
    ("EDI 850 Purchase Order Inbound", "850", "INBOUND"),
    ("EDI 856 ASN Outbound", "856", "OUTBOUND"),
    ("EDI 810 Invoice Outbound", "810", "OUTBOUND"),
    ("EDI 846 Inventory Inquiry", "846", "INBOUND"),
    ("EDI 855 PO Acknowledgment", "855", "OUTBOUND"),
    ("RosettaNet PIP3A4 Distribute PO", "RN-3A4", "OUTBOUND"),
    ("SAP IDoc ORDERS05 Sync", "IDOC-ORDERS", "INBOUND"),
    ("SFTP Shipment Status Feed", "214", "INBOUND"),
]

ERROR_CATALOG = [
    ("E_MAP_204", "Map failed: segment ISA13 missing in source profile"),
    ("E_CONN_117", "AS2 connection timeout after 30000 ms to partner endpoint"),
    ("E_CERT_042", "SSL certificate expired for AS2 partner channel"),
    ("E_VAL_318", "997 functional acknowledgment reported rejected transaction set"),
    ("E_SFTP_209", "SFTP authentication failure: permission denied (publickey)"),
    ("E_SCHEMA_077", "XML schema validation error: element 'DeliveryDate' not expected"),
    ("E_DUP_512", "Duplicate interchange control number detected; execution skipped"),
    ("E_SAP_661", "SAP IDoc status 51: application document not posted"),
]

ENVIRONMENTS = ["PROD", "PROD", "PROD", "TEST"]


@dataclass
class ExecutionLog:
    """One integration platform execution record."""

    execution_id: str
    partner_id: str
    partner_name: str
    process_name: str
    transaction_type: str
    direction: str
    status: str  # SUCCESS | FAILED | WARNING
    start_time: str
    end_time: str
    duration_ms: int
    records_processed: int
    records_failed: int
    bytes_transferred: int
    error_code: Optional[str]
    error_message: Optional[str]
    source_system: str
    target_system: str
    environment: str
    ingested_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def _random_execution(rng: random.Random, start: datetime) -> ExecutionLog:
    partner_id, partner_name = rng.choice(PARTNERS)
    process_name, txn_type, direction = rng.choice(PROCESSES)
    # ~82% success, ~6% warning, ~12% failure — typical production mix.
    roll = rng.random()
    if roll < 0.82:
        status = "SUCCESS"
    elif roll < 0.88:
        status = "WARNING"
    else:
        status = "FAILED"

    duration_ms = int(rng.lognormvariate(7.2, 0.9))  # median ~1.3 s
    records_processed = rng.randint(1, 5000)
    if status == "FAILED":
        error_code, error_message = rng.choice(ERROR_CATALOG)
        records_failed = rng.randint(1, max(1, records_processed // 3))
        duration_ms = int(duration_ms * rng.uniform(1.2, 3.0))
    elif status == "WARNING":
        error_code, error_message = rng.choice(ERROR_CATALOG)
        records_failed = rng.randint(0, 5)
    else:
        error_code, error_message = None, None
        records_failed = 0

    end = start + timedelta(milliseconds=duration_ms)
    source, target = (
        ("SAP ECC", "Dell Boomi") if direction == "INBOUND" else ("Dell Boomi", "SAP S/4HANA")
    )
    return ExecutionLog(
        execution_id=str(uuid.UUID(int=rng.getrandbits(128))),
        partner_id=partner_id,
        partner_name=partner_name,
        process_name=process_name,
        transaction_type=txn_type,
        direction=direction,
        status=status,
        start_time=start.isoformat(),
        end_time=end.isoformat(),
        duration_ms=duration_ms,
        records_processed=records_processed,
        records_failed=records_failed,
        bytes_transferred=records_processed * rng.randint(180, 2400),
        error_code=error_code,
        error_message=error_message,
        source_system=source,
        target_system=target,
        environment=rng.choice(ENVIRONMENTS),
    )


def generate_logs(
    count: int,
    seed: int = 42,
    days: int = 7,
    end_time: Optional[datetime] = None,
) -> Iterator[ExecutionLog]:
    """Yield ``count`` execution logs spread over the last ``days`` days."""
    rng = random.Random(seed)
    end = end_time or datetime.now(timezone.utc)
    start_window = end - timedelta(days=days)
    span_seconds = (end - start_window).total_seconds()
    offsets = sorted(rng.random() * span_seconds for _ in range(count))
    for offset in offsets:
        yield _random_execution(rng, start_window + timedelta(seconds=offset))


def generate_telemetry_event(rng: random.Random) -> dict:
    """Generate one streaming telemetry event (queue depth / atom heartbeat)."""
    partner_id, partner_name = rng.choice(PARTNERS)
    return {
        "event_id": str(uuid.UUID(int=rng.getrandbits(128))),
        "event_time": datetime.now(timezone.utc).isoformat(),
        "partner_id": partner_id,
        "partner_name": partner_name,
        "atom_id": f"atom-{rng.randint(1, 4):02d}",
        "queue_depth": rng.randint(0, 950),
        "active_executions": rng.randint(0, 120),
        "cpu_percent": round(rng.uniform(4.0, 96.0), 2),
        "memory_percent": round(rng.uniform(22.0, 91.0), 2),
        "status": rng.choices(
            ["HEALTHY", "DEGRADED", "DOWN"], weights=[0.9, 0.08, 0.02]
        )[0],
    }


def write_logs(logs: Iterator[ExecutionLog], out_dir: Path, tag: str) -> Path:
    """Write logs as a JSONL file in ``out_dir`` and return the file path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"execution_logs_{tag}_{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for log in logs:
            fh.write(log.to_json() + "\n")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate execution logs.")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Write a fixed-name sample file (execution_logs_sample.jsonl).",
    )
    args = parser.parse_args(argv)

    logs = list(generate_logs(args.count, seed=args.seed, days=args.days))
    if args.sample:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "execution_logs_sample.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for log in logs:
                fh.write(log.to_json() + "\n")
    else:
        path = write_logs(iter(logs), args.out, tag="raw")
    print(f"Wrote {len(logs)} execution logs to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
