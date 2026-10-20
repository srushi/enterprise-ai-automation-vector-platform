"""Airflow DAG: end-to-end ELT + AI data product orchestration.

Pipeline: generate -> land -> load -> transform -> vector ingest -> monitor

* ``generate_logs``      — synthetic execution logs into the landing zone
                           (in production this is the platform export / S3 sync).
* ``load_warehouse``     — landing zone -> Snowflake (SQLite fallback locally).
* ``transform_products`` — pandas local runner; swap for a SparkSubmitOperator
                           running ``spark/transform_job.py`` in production.
* ``build_vector_index`` — chunk incidents, embed, persist the vector index.
* ``health_check``       — freshness + payload-drop alerts; CRITICAL alerts
                           fail the task so the run shows red in the UI.

The DAG module degrades gracefully when Airflow is not installed (the task
callables remain importable for tests and local runs).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402


def generate_logs(**context) -> str:
    from ingestion.log_generator import generate_logs as gen
    from ingestion.log_generator import write_logs

    config.ensure_dirs()
    logs = gen(count=500, seed=int(datetime.utcnow().timestamp()) % 100000)
    path = write_logs(logs, config.LANDING_RAW_DIR, tag="raw")
    return str(path)


def load_warehouse(**context) -> str:
    from warehouse.snowflake_loader import WarehouseLoader

    with WarehouseLoader() as loader:
        loader.ensure_schema()
        report = loader.load_directory(config.LANDING_RAW_DIR)
    if report.rejected:
        raise ValueError(f"{report.rejected} records failed validation")
    return f"{report} backend={loader.backend}"


def transform_products(**context) -> str:
    from spark.local_runner import run

    counts = run(config.LANDING_RAW_DIR, config.PRODUCTS_DIR)
    return str(counts)


def build_vector_index(**context) -> str:
    from vector.build_index import build_index

    index = build_index(config.VECTOR_INDEX_DIR)
    return f"indexed {len(index)} documents"


def health_check(**context) -> str:
    from monitoring.health_monitor import run_health_checks
    from warehouse.snowflake_loader import WarehouseLoader

    with WarehouseLoader() as loader:
        loader.ensure_schema()
        report = run_health_checks(loader)
    critical = [a for a in report["alerts"] if a["severity"] == "CRITICAL"]
    if critical:
        raise RuntimeError(f"CRITICAL health alerts: {critical}")
    return f"healthy={report['healthy']} alerts={len(report['alerts'])}"


try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    default_args = {
        "owner": "data-platform",
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "email_on_failure": False,
    }

    with DAG(
        dag_id="enterprise_ai_automation_elt",
        description="ELT -> transform -> vector ingest -> health monitoring",
        default_args=default_args,
        start_date=datetime(2026, 1, 1),
        schedule="@hourly",
        catchup=False,
        tags=["elt", "snowflake", "dbt", "vector", "ai"],
    ) as dag:
        t_generate = PythonOperator(
            task_id="generate_logs", python_callable=generate_logs
        )
        t_load = PythonOperator(
            task_id="load_warehouse", python_callable=load_warehouse
        )
        t_transform = PythonOperator(
            task_id="transform_products", python_callable=transform_products
        )
        t_vector = PythonOperator(
            task_id="build_vector_index", python_callable=build_vector_index
        )
        t_health = PythonOperator(
            task_id="health_check", python_callable=health_check
        )

        t_generate >> t_load >> t_transform >> t_vector >> t_health

except ImportError:  # Airflow not installed — callables remain usable.
    dag = None
