"""Pytest bootstrap: make the project root importable and provide fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.log_generator import generate_logs  # noqa: E402
from warehouse.snowflake_loader import WarehouseLoader  # noqa: E402


@pytest.fixture()
def sample_records() -> list[dict]:
    """Forty deterministic execution-log records as plain dicts."""
    return [
        json.loads(log.to_json())
        for log in generate_logs(40, seed=123, days=3)
    ]


@pytest.fixture()
def loader(tmp_path: Path):
    """A SQLite-backed WarehouseLoader on a temp database."""
    with WarehouseLoader(
        db_path=tmp_path / "test_warehouse.db", backend="sqlite"
    ) as ldr:
        ldr.ensure_schema()
        yield ldr
