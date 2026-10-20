"""Dataset health monitoring: freshness + payload-drop alerting.

Reads the warehouse table and answers two operational questions:

1. **Freshness** — how old is the newest execution record? (A stale feed
   usually means a broken upstream schedule, not a quiet day.)
2. **Payload drops** — did the most recent day's record volume fall sharply
   versus the trailing baseline? Sudden drops are the classic silent-failure
   signature in B2B integrations: the job "succeeds" but partners stop
   sending, or a mapping change quietly discards documents.

Alerts are returned as structured dicts and printed by the CLI; the Airflow
DAG can raise on CRITICAL alerts to fail the run visibly.

Usage:
    python -m monitoring.health_monitor [--max-staleness-hours 26] [--drop-threshold 0.4]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from warehouse.snowflake_loader import WarehouseLoader


def _parse_ts(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def dataset_freshness(loader: WarehouseLoader, now: Optional[datetime] = None) -> dict:
    """Hours between 'now' and the newest record in the warehouse."""
    latest = loader.max_start_time()
    if latest is None:
        return {"latest_record": None, "staleness_hours": None}
    now = now or datetime.now(timezone.utc)
    staleness = (now - _parse_ts(latest)).total_seconds() / 3600.0
    return {"latest_record": latest, "staleness_hours": round(staleness, 2)}


def daily_volumes(loader: WarehouseLoader) -> list[dict]:
    """Records processed per day, oldest first."""
    return loader.query(
        """
        SELECT substr(start_time, 1, 10) AS day,
               COUNT(*) AS executions,
               SUM(records_processed) AS records_processed,
               SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) AS failures
        FROM raw_execution_logs
        GROUP BY day ORDER BY day
        """
    )


def detect_payload_drops(
    volumes: list[dict],
    drop_threshold: float = 0.4,
    baseline_days: int = 3,
) -> list[dict]:
    """Flag days whose record volume drops >= threshold vs trailing baseline."""
    alerts: list[dict] = []
    if len(volumes) < 2:
        return alerts
    totals = [int(v["records_processed"] or 0) for v in volumes]
    for i in range(1, len(volumes)):
        window = totals[max(0, i - baseline_days) : i]
        baseline = sum(window) / len(window)
        if baseline <= 0:
            continue
        change = (totals[i] - baseline) / baseline
        if change <= -drop_threshold:
            alerts.append(
                {
                    "type": "PAYLOAD_DROP",
                    "severity": "CRITICAL" if change <= -0.7 else "WARNING",
                    "day": volumes[i]["day"],
                    "records_processed": totals[i],
                    "baseline_avg": round(baseline, 1),
                    "change_percent": round(change * 100, 1),
                    "message": (
                        f"Record volume on {volumes[i]['day']} dropped "
                        f"{abs(change) * 100:.1f}% vs trailing baseline "
                        f"({totals[i]} vs ~{baseline:.0f} records)."
                    ),
                }
            )
    return alerts


def run_health_checks(
    loader: WarehouseLoader,
    max_staleness_hours: float = 26.0,
    drop_threshold: float = 0.4,
    now: Optional[datetime] = None,
) -> dict:
    """Run all checks; return a structured health report."""
    freshness = dataset_freshness(loader, now=now)
    volumes = daily_volumes(loader)
    alerts: list[dict] = []

    if freshness["staleness_hours"] is None:
        alerts.append(
            {
                "type": "EMPTY_DATASET",
                "severity": "CRITICAL",
                "message": "Warehouse contains no execution records.",
            }
        )
    elif freshness["staleness_hours"] > max_staleness_hours:
        alerts.append(
            {
                "type": "STALE_FEED",
                "severity": "WARNING",
                "message": (
                    f"Newest record is {freshness['staleness_hours']}h old "
                    f"(limit {max_staleness_hours}h)."
                ),
            }
        )
    alerts.extend(detect_payload_drops(volumes, drop_threshold=drop_threshold))

    total_exec = sum(int(v["executions"]) for v in volumes)
    total_failed = sum(int(v["failures"] or 0) for v in volumes)
    return {
        "freshness": freshness,
        "days_observed": len(volumes),
        "total_executions": total_exec,
        "total_failures": total_failed,
        "failure_rate": round(total_failed / total_exec, 4) if total_exec else None,
        "alerts": alerts,
        "healthy": not any(a["severity"] == "CRITICAL" for a in alerts),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Warehouse health checks.")
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--max-staleness-hours", type=float, default=26.0)
    parser.add_argument("--drop-threshold", type=float, default=0.4)
    args = parser.parse_args(argv)

    with WarehouseLoader(db_path=args.db) as loader:
        loader.ensure_schema()
        report = run_health_checks(
            loader,
            max_staleness_hours=args.max_staleness_hours,
            drop_threshold=args.drop_threshold,
        )
    print(f"Executions: {report['total_executions']} over "
          f"{report['days_observed']} day(s); failure rate: {report['failure_rate']}")
    print(f"Freshness: {report['freshness']}")
    if report["alerts"]:
        for alert in report["alerts"]:
            print(f"[{alert['severity']}] {alert['type']}: {alert['message']}")
    else:
        print("No alerts. Dataset healthy.")
    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
