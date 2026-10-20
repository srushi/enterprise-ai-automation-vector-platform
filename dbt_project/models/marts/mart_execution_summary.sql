-- Mart: daily execution summary per partner + process.
-- Mirrors the Spark/pandas `execution_summary` data product so BI and the
-- AI data products read consistent numbers regardless of build path.
select
    execution_date,
    partner_id,
    partner_name,
    process_name,
    count(*) as executions,
    sum(case when status = 'SUCCESS' then 1 else 0 end) as successes,
    sum(case when status = 'FAILED' then 1 else 0 end) as failures,
    sum(case when status = 'WARNING' then 1 else 0 end) as warnings,
    round(
        sum(case when status = 'SUCCESS' then 1 else 0 end) * 1.0 / count(*),
        4
    ) as success_rate,
    sum(records_processed) as total_records_processed,
    sum(records_failed) as total_records_failed,
    round(avg(duration_ms), 2) as avg_duration_ms,
    sum(bytes_transferred) as total_bytes
from {{ ref('stg_execution_logs') }}
group by 1, 2, 3, 4
