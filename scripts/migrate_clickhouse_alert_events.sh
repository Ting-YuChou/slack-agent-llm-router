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

# SQL is generated into a variable first: a plain assignment stops the
# script under set -e when the generator fails, before anything changes.
require_sql() {
  if [[ -z "$2" ]]; then
    echo "Could not generate the $1 SQL for ${database_name}.alert_events; nothing was changed." >&2
    exit 1
  fi
}

table_exists="$(run_query "EXISTS TABLE ${database_name}.alert_events" | tr -d '\r\n')"

if [[ "${table_exists}" == "0" ]]; then
  create_sql="$(alert_events_sql create "${database_name}")"
  require_sql create "${create_sql}"
  run_query "${create_sql}"
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
# The backup keeps these columns; read them while the table has its name.
source_columns="$(run_query "SELECT name FROM system.columns WHERE database = '${database_name}' AND table = 'alert_events' ORDER BY position" | tr -d '\r' | paste -sd, -)"
require_sql "column list" "${source_columns}"

# All SQL is generated before the table is renamed.
create_sql="$(alert_events_sql create "${database_name}")"
require_sql create "${create_sql}"
copy_sql="$(alert_events_sql copy "${database_name}" "${backup_table_name}" "${source_columns}")"
require_sql copy "${copy_sql}"
expected_count_sql="$(alert_events_sql expected-count "${database_name}" "${backup_table_name}" "${source_columns}")"
require_sql expected-count "${expected_count_sql}"

run_query "RENAME TABLE ${database_name}.alert_events TO ${database_name}.${backup_table_name}"
run_query "${create_sql}"
run_query "${copy_sql}"

# ReplacingMergeTree collapses copied rows that share a sorting key, so
# compare the rows it keeps (FINAL) with the distinct keys copied.
expected_row_count="$(run_query "${expected_count_sql}" | tr -d '\r\n')"
migrated_row_count="$(run_query "SELECT count() FROM ${database_name}.alert_events FINAL" | tr -d '\r\n')"

if [[ "${expected_row_count}" != "${migrated_row_count}" ]]; then
  echo "Row count mismatch after migration: expected=${expected_row_count} distinct rows, migrated=${migrated_row_count}." >&2
  echo "Backup table retained at ${database_name}.${backup_table_name}." >&2
  exit 1
fi

echo "Migrated ${database_name}.alert_events to the current schema."
echo "Copied ${source_row_count} rows and retained backup table ${database_name}.${backup_table_name}."
if [[ "${source_row_count}" != "${expected_row_count}" ]]; then
  echo "The new table keeps ${migrated_row_count}: $((source_row_count - expected_row_count)) duplicate rows share a sorting key and collapse."
fi
