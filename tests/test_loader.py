"""Tests for the warehouse loader (SQLite fallback path)."""

from __future__ import annotations

import json
from pathlib import Path

from warehouse.snowflake_loader import WarehouseLoader, validate_record


def test_load_and_count(loader, sample_records):
    report = loader.load_records(sample_records)
    assert report.rejected == 0
    assert report.loaded == len(sample_records)
    assert loader.count() == len(sample_records)


def test_load_is_idempotent(loader, sample_records):
    loader.load_records(sample_records)
    first = loader.count()
    report = loader.load_records(sample_records)  # reload same file
    assert report.loaded == len(sample_records)
    assert loader.count() == first  # upsert by execution_id, no duplicates


def test_invalid_records_are_rejected(loader, sample_records):
    good = dict(sample_records[0])
    bad_status = dict(good, execution_id="bad-1", status="EXPLODED")
    missing_field = dict(good, execution_id="bad-2")
    del missing_field["partner_id"]
    report = loader.load_records([good, bad_status, missing_field])
    assert report.loaded == 1
    assert report.rejected == 2
    assert loader.count() == 1


def test_load_jsonl_file(loader, sample_records, tmp_path: Path):
    path = tmp_path / "landing.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in sample_records[:10]:
            fh.write(json.dumps(rec) + "\n")
    report = loader.load_jsonl(path)
    assert report.loaded == 10
    assert loader.count() == 10


def test_max_start_time(loader, sample_records):
    loader.load_records(sample_records)
    latest = loader.max_start_time()
    assert latest == max(r["start_time"] for r in sample_records)


def test_validate_record_accepts_generated_records(sample_records):
    for rec in sample_records:
        assert validate_record(rec) == []
