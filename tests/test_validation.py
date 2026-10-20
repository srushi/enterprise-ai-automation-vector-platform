"""Tests for record validation and health-monitoring logic."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from monitoring.health_monitor import (
    dataset_freshness,
    detect_payload_drops,
    run_health_checks,
)
from warehouse.snowflake_loader import validate_record

BASE = {
    "execution_id": "abc-123",
    "partner_id": "PTNR-1001",
    "process_name": "EDI 850 Purchase Order Inbound",
    "status": "SUCCESS",
    "start_time": "2026-10-01T00:00:00+00:00",
    "records_processed": 100,
    "records_failed": 0,
    "direction": "INBOUND",
}


def test_validate_ok():
    assert validate_record(BASE) == []


def test_validate_missing_required_field():
    rec = {k: v for k, v in BASE.items() if k != "partner_id"}
    errors = validate_record(rec)
    assert any("missing required fields" in e for e in errors)


def test_validate_bad_status_and_direction():
    errors = validate_record(dict(BASE, status="MAYBE", direction="SIDEWAYS"))
    assert any("invalid status" in e for e in errors)
    assert any("invalid direction" in e for e in errors)


def test_validate_failed_exceeds_processed():
    errors = validate_record(dict(BASE, records_failed=101))
    assert any("records_failed out of range" in e for e in errors)


def test_validate_negative_records():
    errors = validate_record(dict(BASE, records_processed=-5))
    assert errors


def _volumes(totals: list[int]) -> list[dict]:
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    return [
        {
            "day": (start + timedelta(days=i)).strftime("%Y-%m-%d"),
            "executions": 10,
            "records_processed": total,
            "failures": 0,
        }
        for i, total in enumerate(totals)
    ]


def test_payload_drop_detected():
    volumes = _volumes([1000, 1050, 980, 120])  # last day collapses
    alerts = detect_payload_drops(volumes, drop_threshold=0.4)
    assert len(alerts) == 1
    assert alerts[0]["type"] == "PAYLOAD_DROP"
    assert alerts[0]["severity"] == "CRITICAL"  # >70% drop


def test_payload_drop_not_triggered_on_normal_variation():
    volumes = _volumes([1000, 950, 1020, 990])
    assert detect_payload_drops(volumes, drop_threshold=0.4) == []


def test_payload_drop_needs_history():
    assert detect_payload_drops(_volumes([100])) == []


def test_freshness_and_health_report(loader, sample_records):
    loader.load_records(sample_records)
    now = datetime.now(timezone.utc) + timedelta(days=365)  # far future -> stale
    freshness = dataset_freshness(loader, now=now)
    assert freshness["staleness_hours"] > 24 * 300
    report = run_health_checks(loader, max_staleness_hours=26.0, now=now)
    assert report["total_executions"] == len(sample_records)
    assert any(a["type"] == "STALE_FEED" for a in report["alerts"])


def test_empty_warehouse_is_unhealthy(loader):
    report = run_health_checks(loader)
    assert report["healthy"] is False
    assert any(a["type"] == "EMPTY_DATASET" for a in report["alerts"])
