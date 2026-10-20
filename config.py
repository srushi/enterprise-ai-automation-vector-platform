"""Shared configuration for the Enterprise AI Automation & Vector Data Platform.

All paths are resolved relative to the project root and can be overridden with
environment variables so the same code runs locally, in Airflow, or in CI.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("PLATFORM_DATA_DIR", PROJECT_ROOT / "data"))
LANDING_DIR = Path(os.environ.get("PLATFORM_LANDING_DIR", DATA_DIR / "landing"))
LANDING_RAW_DIR = LANDING_DIR / "raw"
LANDING_STREAM_DIR = LANDING_DIR / "stream"
PRODUCTS_DIR = Path(os.environ.get("PLATFORM_PRODUCTS_DIR", DATA_DIR / "products"))
VECTOR_INDEX_DIR = Path(
    os.environ.get("PLATFORM_VECTOR_INDEX_DIR", DATA_DIR / "vector_index")
)
STREAM_QUEUE_DIR = Path(
    os.environ.get("PLATFORM_STREAM_QUEUE_DIR", DATA_DIR / "stream_queue")
)
WAREHOUSE_DB_PATH = Path(
    os.environ.get("PLATFORM_WAREHOUSE_DB", DATA_DIR / "warehouse.db")
)


def snowflake_configured() -> bool:
    """Return True when the SNOWFLAKE_* environment variables are present."""
    required = ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD")
    return all(os.environ.get(name) for name in required)


def ensure_dirs() -> None:
    """Create the local runtime directories (landing zone, products, index)."""
    for path in (
        LANDING_RAW_DIR,
        LANDING_STREAM_DIR,
        PRODUCTS_DIR,
        VECTOR_INDEX_DIR,
        STREAM_QUEUE_DIR,
        DATA_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
