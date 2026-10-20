-- Staging: typed, lightly cleaned view over the raw landed execution logs.
-- One row per execution; column names normalized for downstream marts.
with source as (
    select * from {{ source('warehouse_raw', 'raw_execution_logs') }}
),

cleaned as (
    select
        execution_id,
        partner_id,
        partner_name,
        process_name,
        transaction_type,
        direction,
        upper(status) as status,
        cast(start_time as timestamp) as started_at,
        cast(end_time as timestamp) as ended_at,
        cast(substr(cast(start_time as varchar), 1, 10) as date) as execution_date,
        duration_ms,
        records_processed,
        coalesce(records_failed, 0) as records_failed,
        coalesce(bytes_transferred, 0) as bytes_transferred,
        records_processed - coalesce(records_failed, 0) as records_succeeded,
        error_code,
        error_message,
        source_system,
        target_system,
        environment
    from source
    where execution_id is not null
      and environment = 'PROD'
)

select * from cleaned
