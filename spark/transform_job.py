"""PySpark transformation job: raw execution logs -> AI data products.

Reads newline-delimited execution logs from the landing zone and produces
structured, analytics/AI-ready data products:

* ``execution_summary`` — daily rollup per partner and process with success
  rates, record volumes, and latency statistics.
* ``error_catalog`` — failure counts per error code, used to seed the vector
  index with searchable incident documents.

In production this runs on a Spark cluster (Databricks / EMR) against the S3
landing zone and writes Parquet back to S3. For a dependency-free local run
with identical logic, see ``spark/local_runner.py``.

Usage:
    spark-submit spark/transform_job.py --input data/landing/raw --output data/products
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_data_products(spark, input_path: str, output_path: str) -> dict:
    """Run the Spark transformations. Returns row counts per data product."""
    from pyspark.sql import functions as F

    raw = spark.read.json(input_path)
    raw = raw.withColumn("execution_date", F.to_date("start_time"))

    execution_summary = (
        raw.groupBy("execution_date", "partner_id", "partner_name", "process_name")
        .agg(
            F.count("*").alias("executions"),
            F.sum(F.when(F.col("status") == "SUCCESS", 1).otherwise(0)).alias(
                "successes"
            ),
            F.sum(F.when(F.col("status") == "FAILED", 1).otherwise(0)).alias(
                "failures"
            ),
            F.sum(F.when(F.col("status") == "WARNING", 1).otherwise(0)).alias(
                "warnings"
            ),
            F.sum("records_processed").alias("total_records_processed"),
            F.sum("records_failed").alias("total_records_failed"),
            F.round(F.avg("duration_ms"), 2).alias("avg_duration_ms"),
            F.sum("bytes_transferred").alias("total_bytes"),
        )
        .withColumn(
            "success_rate",
            F.round(F.col("successes") / F.col("executions"), 4),
        )
        .orderBy("execution_date", "partner_id")
    )

    error_catalog = (
        raw.filter(F.col("status") != "SUCCESS")
        .filter(F.col("error_code").isNotNull())
        .groupBy("error_code", "error_message")
        .agg(
            F.count("*").alias("occurrences"),
            F.countDistinct("partner_id").alias("partners_affected"),
            F.min("start_time").alias("first_seen"),
            F.max("start_time").alias("last_seen"),
        )
        .orderBy(F.desc("occurrences"))
    )

    out = Path(output_path)
    (execution_summary.coalesce(1).write.mode("overwrite")
        .parquet(str(out / "execution_summary")))
    (error_catalog.coalesce(1).write.mode("overwrite")
        .parquet(str(out / "error_catalog")))

    return {
        "execution_summary": execution_summary.count(),
        "error_catalog": error_catalog.count(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PySpark log transformation job.")
    parser.add_argument("--input", required=True, help="Landing zone path (JSONL).")
    parser.add_argument("--output", required=True, help="Data products output path.")
    args = parser.parse_args(argv)

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName("ai-automation-execution-transform")
        .master("local[*]")
        .getOrCreate()
    )
    try:
        counts = build_data_products(spark, args.input, args.output)
    finally:
        spark.stop()
    print(f"Data products written to {args.output}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
