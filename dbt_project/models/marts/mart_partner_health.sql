-- Mart: partner-level health rollup across the full history.
-- Feeds partner scorecards and the health-monitor payload-drop checks.
select
    partner_id,
    partner_name,
    count(*) as total_executions,
    sum(case when status = 'FAILED' then 1 else 0 end) as total_failures,
    round(
        sum(case when status = 'SUCCESS' then 1 else 0 end) * 1.0 / count(*),
        4
    ) as success_rate,
    sum(records_processed) as total_records_processed,
    sum(records_failed) as total_records_failed,
    round(avg(duration_ms), 2) as avg_duration_ms,
    min(started_at) as first_execution_at,
    max(started_at) as last_execution_at,
    count(distinct error_code) as distinct_error_codes
from {{ ref('stg_execution_logs') }}
group by 1, 2
