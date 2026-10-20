"""Warehouse loader: landing zone (S3) -> cloud warehouse.

The production target is Snowflake. When the ``SNOWFLAKE_*`` environment
variables are set, this module loads through the Snowflake Python connector;
otherwise it falls back to a local SQLite warehouse with identical table
semantics so the full pipeline runs offline on a laptop.

Table: ``raw_execution_logs`` — one row per integration execution, keyed by
``execution_id``. Loads are idempotent: re-loading a file upserts by
``execution_id`` instead of duplicating rows.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import config

TABLE = "raw_execution_logs"

COLUMNS = [
    "execution_id",
    "partner_id",
    "partner_name",
    "process_name",
    "transaction_type",
    "direction",
    "status",
    "start_time",
    "end_time",
    "duration_ms",
    "records_processed",
    "records_failed",
    "bytes_transferred",
    "error_code",
    "error_message",
    "source_system",
    "target_system",
    "environment",
    "ingested_at",
]

REQUIRED_FIELDS = {
    "execution_id",
    "partner_id",
    "process_name",
    "status",
    "start_time",
    "records_processed",
}

VALID_STATUSES = {"SUCCESS", "FAILED", "WARNING"}
VALID_DIRECTIONS = {"INBOUND", "OUTBOUND"}

_SQLITE_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    execution_id      TEXT PRIMARY KEY,
    partner_id        TEXT NOT NULL,
    partner_name      TEXT,
    process_name      TEXT NOT NULL,
    transaction_type  TEXT,
    direction         TEXT,
    status            TEXT NOT NULL,
    start_time        TEXT NOT NULL,
    end_time          TEXT,
    duration_ms       INTEGER,
    records_processed INTEGER NOT NULL,
    records_failed    INTEGER DEFAULT 0,
    bytes_transferred INTEGER DEFAULT 0,
    error_code        TEXT,
    error_message     TEXT,
    source_system     TEXT,
    target_system     TEXT,
    environment       TEXT,
    ingested_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_raw_logs_partner ON {TABLE} (partner_id);
CREATE INDEX IF NOT EXISTS idx_raw_logs_start ON {TABLE} (start_time);
"""

_SNOWFLAKE_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    execution_id      VARCHAR PRIMARY KEY,
    partner_id        VARCHAR NOT NULL,
    partner_name      VARCHAR,
    process_name      VARCHAR NOT NULL,
    transaction_type  VARCHAR,
    direction         VARCHAR,
    status            VARCHAR NOT NULL,
    start_time        TIMESTAMP_NTZ NOT NULL,
    end_time          TIMESTAMP_NTZ,
    duration_ms       NUMBER,
    records_processed NUMBER NOT NULL,
    records_failed    NUMBER DEFAULT 0,
    bytes_transferred NUMBER DEFAULT 0,
    error_code        VARCHAR,
    error_message     VARCHAR,
    source_system     VARCHAR,
    target_system     VARCHAR,
    environment       VARCHAR,
    ingested_at       TIMESTAMP_NTZ
)
"""


def validate_record(record: dict) -> list[str]:
    """Validate one execution-log record. Returns a list of error strings."""
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["record is not a JSON object"]
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        errors.append(f"missing required fields: {sorted(missing)}")
    status = record.get("status")
    if status is not None and status not in VALID_STATUSES:
        errors.append(f"invalid status: {status!r}")
    direction = record.get("direction")
    if direction is not None and direction not in VALID_DIRECTIONS:
        errors.append(f"invalid direction: {direction!r}")
    processed = record.get("records_processed")
    if processed is not None and (
        not isinstance(processed, int) or processed < 0
    ):
        errors.append("records_processed must be a non-negative integer")
    failed = record.get("records_failed")
    if failed is not None and processed is not None and isinstance(failed, int):
        if failed < 0 or (isinstance(processed, int) and failed > processed):
            errors.append("records_failed out of range")
    return errors


@dataclass
class LoadReport:
    """Outcome of a load operation."""

    loaded: int = 0
    rejected: int = 0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"loaded={self.loaded} rejected={self.rejected}"


class WarehouseLoader:
    """Loads execution logs into Snowflake (prod) or SQLite (local fallback)."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        backend: Optional[str] = None,
    ) -> None:
        self.backend = backend or (
            "snowflake" if config.snowflake_configured() else "sqlite"
        )
        self.db_path = Path(db_path) if db_path else config.WAREHOUSE_DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
        self._sf_conn = None

    # -- connections ------------------------------------------------------
    def _sqlite(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _snowflake(self):
        if self._sf_conn is None:
            import snowflake.connector  # optional production dependency

            self._sf_conn = snowflake.connector.connect(
                account=_env("SNOWFLAKE_ACCOUNT"),
                user=_env("SNOWFLAKE_USER"),
                password=_env("SNOWFLAKE_PASSWORD"),
                warehouse=_env("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
                database=_env("SNOWFLAKE_DATABASE", "INTEGRATION_DB"),
                schema=_env("SNOWFLAKE_SCHEMA", "RAW"),
            )
        return self._sf_conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._sf_conn is not None:
            self._sf_conn.close()
            self._sf_conn = None

    def __enter__(self) -> "WarehouseLoader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- schema ------------------------------------------------------------
    def ensure_schema(self) -> None:
        if self.backend == "snowflake":
            cur = self._snowflake().cursor()
            cur.execute(_SNOWFLAKE_DDL)
            cur.close()
        else:
            self._sqlite().executescript(_SQLITE_DDL)
            self._sqlite().commit()

    # -- loading -----------------------------------------------------------
    def load_records(self, records: Iterable[dict]) -> LoadReport:
        """Validate and upsert records. Invalid records are rejected."""
        report = LoadReport()
        valid: list[dict] = []
        for record in records:
            errors = validate_record(record)
            if errors:
                report.rejected += 1
                report.errors.extend(
                    f"{record.get('execution_id', '<no id>')}: {e}" for e in errors
                )
            else:
                valid.append({col: record.get(col) for col in COLUMNS})
        if not valid:
            return report
        if self.backend == "snowflake":
            self._load_snowflake(valid)
        else:
            self._load_sqlite(valid)
        report.loaded = len(valid)
        return report

    def _load_sqlite(self, rows: list[dict]) -> None:
        conn = self._sqlite()
        placeholders = ", ".join("?" for _ in COLUMNS)
        sql = (
            f"INSERT OR REPLACE INTO {TABLE} ({', '.join(COLUMNS)}) "
            f"VALUES ({placeholders})"
        )
        conn.executemany(sql, [tuple(r[c] for c in COLUMNS) for r in rows])
        conn.commit()

    def _load_snowflake(self, rows: list[dict]) -> None:
        conn = self._snowflake()
        cur = conn.cursor()
        cols = ", ".join(COLUMNS)
        binds = ", ".join(f"%({c})s" for c in COLUMNS)
        updates = ", ".join(
            f"target.{c} = source.{c}" for c in COLUMNS if c != "execution_id"
        )
        merge = (
            f"MERGE INTO {TABLE} AS target USING (SELECT {binds}) AS source "
            f"ON target.execution_id = source.execution_id "
            f"WHEN MATCHED THEN UPDATE SET {updates} "
            f"WHEN NOT MATCHED THEN INSERT ({cols}) VALUES ({binds})"
        )
        # Snowflake MERGE with a SELECT source needs column aliases; execute
        # row-by-row for clarity in this reference implementation.
        for row in rows:
            cur.execute(merge, row)
        conn.commit()
        cur.close()

    def load_jsonl(self, path: Path) -> LoadReport:
        """Load one JSONL landing-zone file."""
        records = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return self.load_records(records)

    def load_directory(self, directory: Path) -> LoadReport:
        """Load every ``*.jsonl`` file in a landing-zone directory."""
        total = LoadReport()
        for path in sorted(Path(directory).glob("*.jsonl")):
            rep = self.load_jsonl(path)
            total.loaded += rep.loaded
            total.rejected += rep.rejected
            total.errors.extend(rep.errors)
        return total

    # -- reads ---------------------------------------------------------------
    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        if self.backend == "snowflake":
            cur = self._snowflake().cursor()
            cur.execute(sql)
            cols = [d[0].lower() for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
            cur.close()
            return rows
        cur = self._sqlite().execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def count(self) -> int:
        rows = self.query(f"SELECT COUNT(*) AS n FROM {TABLE}")
        return int(rows[0]["n"]) if rows else 0

    def max_start_time(self) -> Optional[str]:
        rows = self.query(f"SELECT MAX(start_time) AS mx FROM {TABLE}")
        return rows[0]["mx"] if rows else None


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    import os

    return os.environ.get(name, default)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Load landing zone -> warehouse.")
    parser.add_argument("--load-dir", type=Path, default=config.LANDING_RAW_DIR)
    parser.add_argument("--load-file", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None)
    args = parser.parse_args(argv)

    with WarehouseLoader(db_path=args.db) as loader:
        loader.ensure_schema()
        if args.load_file:
            report = loader.load_jsonl(args.load_file)
        else:
            report = loader.load_directory(args.load_dir)
        print(f"Backend: {loader.backend} | {report} | table rows: {loader.count()}")
        for err in report.errors[:10]:
            print(f"  rejected: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
