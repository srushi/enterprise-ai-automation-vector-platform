"""Pandas local runner — the same transformations as ``transform_job.py``.

Implements identical aggregation logic with pandas so the data products can
be built and verified without a Spark installation. Used by the Airflow DAG's
local mode, the demo, and CI.

Usage:
    python -m spark.local_runner --input data/landing/raw --output data/products
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import config


def load_raw(input_path: Path) -> pd.DataFrame:
    """Read all JSONL execution logs under ``input_path`` into a DataFrame."""
    path = Path(input_path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    frames = [pd.read_json(f, lines=True) for f in files if f.stat().st_size > 0]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["execution_date"] = df["start_time"].dt.date.astype(str)
    return df


def build_execution_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Daily rollup per partner/process — mirrors the Spark job exactly."""
    if df.empty:
        return df
    grouped = df.groupby(
        ["execution_date", "partner_id", "partner_name", "process_name"]
    )
    summary = grouped.agg(
        executions=("execution_id", "count"),
        successes=("status", lambda s: int((s == "SUCCESS").sum())),
        failures=("status", lambda s: int((s == "FAILED").sum())),
        warnings=("status", lambda s: int((s == "WARNING").sum())),
        total_records_processed=("records_processed", "sum"),
        total_records_failed=("records_failed", "sum"),
        avg_duration_ms=("duration_ms", "mean"),
        total_bytes=("bytes_transferred", "sum"),
    ).reset_index()
    summary["avg_duration_ms"] = summary["avg_duration_ms"].round(2)
    summary["success_rate"] = (
        summary["successes"] / summary["executions"]
    ).round(4)
    return summary.sort_values(["execution_date", "partner_id"]).reset_index(
        drop=True
    )


def build_error_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Failure rollup per error code — mirrors the Spark job exactly."""
    if df.empty:
        return df
    errors = df[(df["status"] != "SUCCESS") & df["error_code"].notna()]
    if errors.empty:
        return pd.DataFrame(
            columns=[
                "error_code",
                "error_message",
                "occurrences",
                "partners_affected",
                "first_seen",
                "last_seen",
            ]
        )
    catalog = (
        errors.groupby(["error_code", "error_message"])
        .agg(
            occurrences=("execution_id", "count"),
            partners_affected=("partner_id", "nunique"),
            first_seen=("start_time", "min"),
            last_seen=("start_time", "max"),
        )
        .reset_index()
        .sort_values("occurrences", ascending=False)
        .reset_index(drop=True)
    )
    catalog["first_seen"] = catalog["first_seen"].astype(str)
    catalog["last_seen"] = catalog["last_seen"].astype(str)
    return catalog


def run(input_path: Path, output_dir: Path) -> dict:
    """Build all data products and write them as JSON + CSV."""
    df = load_raw(Path(input_path))
    summary = build_execution_summary(df)
    catalog = build_error_catalog(df)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_json(output_dir / "execution_summary.json", orient="records", indent=2)
    summary.to_csv(output_dir / "execution_summary.csv", index=False)
    catalog.to_json(output_dir / "error_catalog.json", orient="records", indent=2)
    catalog.to_csv(output_dir / "error_catalog.csv", index=False)
    counts = {
        "execution_summary": int(len(summary)),
        "error_catalog": int(len(catalog)),
    }
    # Also expose a run manifest so downstream steps (vector ingest,
    # monitoring) can discover what was produced.
    (output_dir / "manifest.json").write_text(
        json.dumps({"source_rows": int(len(df)), "products": counts}, indent=2),
        encoding="utf-8",
    )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local (pandas) transform runner.")
    parser.add_argument("--input", type=Path, default=config.LANDING_RAW_DIR)
    parser.add_argument("--output", type=Path, default=config.PRODUCTS_DIR)
    args = parser.parse_args(argv)
    counts = run(args.input, args.output)
    print(f"Data products written to {args.output}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
