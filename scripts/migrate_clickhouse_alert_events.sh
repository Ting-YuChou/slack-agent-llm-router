#!/usr/bin/env bash
set -euo pipefail

service_name="${CLICKHOUSE_SERVICE:-clickhouse}"
database_name="${CLICKHOUSE_DATABASE:-default}"
python_bin="${PYTHON:-python3}"
script_dir="$(dirname "${BASH_SOURCE[0]}")"

# Runs inside the container and keeps the password off every command line.
client_script="$(cat "${script_dir}/clickhouse_client_in_container.sh")"

run_query() {
  docker compose exec -T "${service_name}" sh -c "${client_script}" sh --query "$1"
}

# The router's own definition (src/analytics_schema.py), never a copy.
alert_events_sql() {
  "${python_bin}" "${script_dir}/alert_events_sql.py" "$@"
}

create_alert_events_table() {
  run_query "$(alert_events_sql create "${database_name}")"
}

table_exists="$(run_query "EXISTS TABLE ${database_name}.alert_events" | tr -d '\r\n')"

if [[ "${table_exists}" == "0" ]]; then
  create_alert_events_table
  echo "Created ${database_name}.alert_events with the current ReplacingMergeTree schema."
  exit 0
fi

show_create_output="$(run_query "SHOW CREATE TABLE ${database_name}.alert_events")"
if grep -Fq "ReplacingMergeTree" <<<"${show_create_output}"; then
  echo "${database_name}.alert_events already uses the current schema (ReplacingMergeTree). No migration needed."
  exit 0
fi

if grep -Fq "ENGINE = MergeTree" <<<"${show_create_output}" \
  && grep -Fq "request_id" <<<"${show_create_output}" \
  && grep -Fq "query_id" <<<"${show_create_output}" \
  && grep -Fq "model_name" <<<"${show_create_output}" \
  && grep -Fq "provider" <<<"${show_create_output}"; then
  # The router adds the newer columns itself; moving the engine to
  # ReplacingMergeTree is scripts/migrate_clickhouse_analytics_v2.py's job.
  echo "${database_name}.alert_events already uses the current schema columns on a MergeTree engine. No migration needed here."
  echo "Run scripts/migrate_clickhouse_analytics_v2.py --apply to move it to ReplacingMergeTree."
  exit 0
fi

backup_table_name="alert_events_backup_$(date -u +%Y%m%d%H%M%S)"
source_row_count="$(run_query "SELECT count() FROM ${database_name}.alert_events" | tr -d '\r\n')"

run_query "RENAME TABLE ${database_name}.alert_events TO ${database_name}.${backup_table_name}"
create_alert_events_table
backup_columns="$(run_query "SELECT name FROM system.columns WHERE database = '${database_name}' AND table = '${backup_table_name}' ORDER BY position" | tr -d '\r' | paste -sd, -)"
run_query "$(alert_events_sql copy "${database_name}" "${backup_table_name}" "${backup_columns}")"

migrated_row_count="$(run_query "SELECT count() FROM ${database_name}.alert_events" | tr -d '\r\n')"

if [[ "${source_row_count}" != "${migrated_row_count}" ]]; then
  echo "Row count mismatch after migration: source=${source_row_count}, migrated=${migrated_row_count}." >&2
  echo "Backup table retained at ${database_name}.${backup_table_name}." >&2
  exit 1
fi

echo "Migrated ${database_name}.alert_events to the current schema."
echo "Copied ${migrated_row_count} rows and retained backup table ${database_name}.${backup_table_name}."
