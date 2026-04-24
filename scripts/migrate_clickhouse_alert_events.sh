#!/usr/bin/env bash
set -euo pipefail

service_name="${CLICKHOUSE_SERVICE:-clickhouse}"
database_name="${CLICKHOUSE_DATABASE:-default}"
clickhouse_user="${CLICKHOUSE_USER:-llm_router}"
clickhouse_password="${CLICKHOUSE_PASSWORD:-llm_router_pass}"

run_query() {
  docker compose exec -T "${service_name}" clickhouse-client \
    --user "${clickhouse_user}" \
    --password "${clickhouse_password}" \
    --query "$1"
}

create_alert_events_table() {
  run_query "
    CREATE TABLE IF NOT EXISTS ${database_name}.alert_events (
        timestamp DateTime64(3),
        alert_type String,
        severity String,
        description String,
        anomaly_type String,
        source_event_type String,
        request_id String,
        query_id String,
        user_id String,
        model_name String,
        provider String,
        window_start_ms UInt64,
        window_end_ms UInt64,
        payload_json String
    ) ENGINE = MergeTree()
    PARTITION BY toYYYYMMDD(timestamp)
    ORDER BY (
        timestamp,
        source_event_type,
        severity,
        alert_type,
        request_id,
        query_id,
        model_name,
        provider
    )
    TTL toDateTime(timestamp) + INTERVAL 30 DAY
  "
}

table_exists="$(run_query "EXISTS TABLE ${database_name}.alert_events" | tr -d '\r\n')"

if [[ "${table_exists}" == "0" ]]; then
  create_alert_events_table
  echo "Created ${database_name}.alert_events with the current MergeTree schema."
  exit 0
fi

show_create_output="$(run_query "SHOW CREATE TABLE ${database_name}.alert_events")"
if grep -Fq "ENGINE = MergeTree()" <<<"${show_create_output}" \
  && grep -Fq "request_id" <<<"${show_create_output}" \
  && grep -Fq "query_id" <<<"${show_create_output}" \
  && grep -Fq "model_name" <<<"${show_create_output}" \
  && grep -Fq "provider" <<<"${show_create_output}"; then
  echo "${database_name}.alert_events already uses the current schema. No migration needed."
  exit 0
fi

backup_table_name="alert_events_backup_$(date -u +%Y%m%d%H%M%S)"
source_row_count="$(run_query "SELECT count() FROM ${database_name}.alert_events" | tr -d '\r\n')"

run_query "RENAME TABLE ${database_name}.alert_events TO ${database_name}.${backup_table_name}"
create_alert_events_table
run_query "INSERT INTO ${database_name}.alert_events SELECT * FROM ${database_name}.${backup_table_name}"

migrated_row_count="$(run_query "SELECT count() FROM ${database_name}.alert_events" | tr -d '\r\n')"

if [[ "${source_row_count}" != "${migrated_row_count}" ]]; then
  echo "Row count mismatch after migration: source=${source_row_count}, migrated=${migrated_row_count}." >&2
  echo "Backup table retained at ${database_name}.${backup_table_name}." >&2
  exit 1
fi

echo "Migrated ${database_name}.alert_events to the current schema."
echo "Copied ${migrated_row_count} rows and retained backup table ${database_name}.${backup_table_name}."
