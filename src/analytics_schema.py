"""ClickHouse analytics tables, defined once.

Each table's definition generates its ``CREATE TABLE IF NOT EXISTS`` and the
``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` statements that bring an older
table up to date. Adding a column means adding it here and to its row builder.

Stdlib only: the router imports this without the ClickHouse client.
"""

import re
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    # DEFAULT expression, in the CREATE and in the ALTER.
    default: Optional[str] = None
    # DEFAULT for the ALTER only: fills rows written before the column existed.
    # A sorting-key column is only ALTERed when it has one (legacy tables that
    # predate the column and the current key).
    backfill: Optional[str] = None

    def definition(self, default: Optional[str] = None) -> str:
        default = default if default is not None else self.default
        if default is None:
            return f"{self.name} {self.type}"
        return f"{self.name} {self.type} DEFAULT {default}"


@dataclass(frozen=True)
class Table:
    name: str
    columns: Tuple[Column, ...]
    engine: str
    partition_by: str
    order_by: str
    ttl: Optional[str] = None
    settings: Optional[str] = None

    @property
    def key_columns(self) -> FrozenSet[str]:
        """Columns referenced by the sorting or partition key."""
        names = {column.name for column in self.columns}
        referenced = set(_IDENTIFIER.findall(f"{self.order_by} {self.partition_by}"))
        return frozenset(names & referenced)

    def create_sql(self, database: str) -> str:
        columns = ",\n".join(f"    {column.definition()}" for column in self.columns)
        lines = [
            f"CREATE TABLE IF NOT EXISTS {database}.{self.name} (",
            columns,
            f") ENGINE = {self.engine}",
            f"PARTITION BY {self.partition_by}",
            f"ORDER BY {self.order_by}",
        ]
        if self.ttl:
            lines.append(f"TTL {self.ttl}")
        if self.settings:
            lines.append(f"SETTINGS {self.settings}")
        return "\n".join(lines)

    def alter_sql(self, database: str) -> List[str]:
        keys = self.key_columns
        return [
            f"ALTER TABLE {database}.{self.name} ADD COLUMN IF NOT EXISTS "
            + column.definition(column.backfill)
            for column in self.columns
            if column.name not in keys or column.backfill is not None
        ]


def _kafka_columns() -> Tuple[Column, ...]:
    return (
        Column("kafka_topic", "String", default="''"),
        Column("kafka_partition", "Int32", default="-1"),
        Column("kafka_offset", "Int64", default="-1"),
    )


QUERY_LOGS = Table(
    name="query_logs",
    columns=(
        Column("event_id", "String", backfill="query_id"),
        Column("query_id", "String"),
        Column("timestamp", "DateTime64(3)"),
        Column("user_id", "String"),
        Column("user_tier", "Enum8('free' = 1, 'premium' = 2, 'enterprise' = 3)"),
        Column("query_text", "String"),
        Column("query_type", "String"),
        Column("selected_model", "String"),
        Column("token_count_input", "UInt32"),
        Column("token_count_output", "UInt32"),
        Column("latency_ms", "UInt32"),
        Column("cost_usd", "Float64"),
        Column("status", "Enum8('success' = 1, 'error' = 2, 'timeout' = 3)"),
        Column("error_message", "String"),
        Column("context_compressed", "Boolean"),
        Column("compression_ratio", "Float32"),
        Column("cached_response", "Boolean"),
        *_kafka_columns(),
    ),
    engine="ReplacingMergeTree()",
    partition_by="toYYYYMM(timestamp)",
    order_by="(timestamp, event_id)",
    ttl="toDateTime(timestamp) + INTERVAL 90 DAY",
)

SYSTEM_METRICS = Table(
    name="system_metrics",
    columns=(
        Column(
            "event_id",
            "String",
            backfill="concat(service, ':', metric_name, ':', toString(timestamp))",
        ),
        Column("timestamp", "DateTime64(3)"),
        Column("service", "String"),
        Column("metric_name", "String"),
        Column("metric_value", "Float64"),
        Column("labels", "Map(String, String)"),
        *_kafka_columns(),
    ),
    engine="ReplacingMergeTree()",
    partition_by="toYYYYMMDD(timestamp)",
    order_by="(timestamp, service, metric_name, event_id)",
    ttl="toDateTime(timestamp) + INTERVAL 30 DAY",
)

MODEL_PERFORMANCE = Table(
    name="model_performance",
    columns=(
        Column(
            "event_id",
            "String",
            backfill="concat(model_name, ':', toString(timestamp))",
        ),
        Column("timestamp", "DateTime64(3)"),
        Column("model_name", "String"),
        # A sorting-key column that older tables lack.
        Column("provider", "String", default="''", backfill="''"),
        Column("window_start_ms", "UInt64", default="0"),
        Column("window_end_ms", "UInt64", default="0"),
        Column("window_size_seconds", "UInt32", default="60"),
        Column("requests_count", "UInt32"),
        Column("success_count", "UInt32", default="0"),
        Column("success_rate", "Float32"),
        Column("avg_latency_ms", "Float32"),
        Column("avg_tokens_per_second", "Float32"),
        Column("error_count", "UInt32"),
        Column("total_tokens", "UInt32", default="0"),
        Column("total_cost_usd", "Float64"),
        Column("queries_per_second", "Float32", default="0"),
        Column("cache_hit_rate", "Float32", default="0"),
        Column("cached_count", "UInt32", default="0"),
        Column("gpu_utilization", "Float32"),
        Column("memory_usage_gb", "Float32"),
        *_kafka_columns(),
    ),
    engine="ReplacingMergeTree()",
    partition_by="toYYYYMMDD(timestamp)",
    order_by="(timestamp, model_name, provider, event_id)",
    ttl="toDateTime(timestamp) + INTERVAL 60 DAY",
)

USER_ANALYTICS = Table(
    name="user_analytics",
    columns=(
        Column("user_id", "String"),
        Column("date", "Date"),
        Column("total_queries", "UInt32"),
        Column("total_tokens", "UInt32"),
        Column("total_cost_usd", "Float64"),
        Column("avg_query_length", "Float32"),
        Column("most_used_model", "String"),
        Column("query_types", "Array(String)"),
        Column("satisfaction_score", "Float32"),
    ),
    engine="SummingMergeTree()",
    partition_by="toYYYYMM(date)",
    order_by="(user_id, date)",
    ttl="date + INTERVAL 180 DAY",
)

ALERT_EVENTS = Table(
    name="alert_events",
    columns=(
        Column(
            "event_id",
            "String",
            backfill=(
                "concat(source_event_type, ':', alert_type, ':', request_id, "
                "':', query_id, ':', model_name, ':', provider, ':', "
                "toString(timestamp))"
            ),
        ),
        Column("timestamp", "DateTime64(3)"),
        Column("alert_type", "String"),
        Column("severity", "String"),
        Column("description", "String"),
        Column("anomaly_type", "String"),
        Column("source_event_type", "String"),
        Column("request_id", "String"),
        Column("query_id", "String"),
        Column("user_id", "String"),
        Column("model_name", "String"),
        Column("provider", "String"),
        Column("window_start_ms", "UInt64"),
        Column("window_end_ms", "UInt64"),
        Column("payload_json", "String"),
        # Orders rows that share a timestamp; older rows take their timestamp.
        Column("emitted_at", "DateTime64(3)", backfill="timestamp"),
        *_kafka_columns(),
    ),
    engine="ReplacingMergeTree()",
    partition_by="toYYYYMMDD(timestamp)",
    order_by="(source_event_type, timestamp, event_id)",
    ttl="toDateTime(timestamp) + INTERVAL 30 DAY",
)

ROUTING_POLICY_STATE_EVENTS = Table(
    name="routing_policy_state_events",
    columns=(
        Column(
            "event_id",
            "String",
            backfill="concat(scope_type, ':', scope_key, ':', toString(timestamp))",
        ),
        Column("timestamp", "DateTime64(3)"),
        Column("scope_type", "String"),
        Column("scope_key", "String"),
        Column("user_id", "String"),
        Column("session_id", "String"),
        Column("user_tier", "String"),
        Column("hint_reason", "String"),
        Column("recent_request_count", "UInt32"),
        Column("recent_error_rate", "Float32"),
        Column("avg_latency_ms", "Float32"),
        Column("fast_lane_hit_rate", "Float32"),
        Column("dominant_query_type", "String"),
        Column("query_complexity", "String"),
        Column("requires_low_latency", "Boolean"),
        Column("requires_high_reasoning", "Boolean"),
        Column("route_to_fast_lane", "Boolean"),
        Column("burst_protection_active", "Boolean"),
        Column("enterprise_priority_active", "Boolean"),
        Column("preferred_models", "Array(String)"),
        Column("avoid_models", "Array(String)"),
        Column("avoid_providers", "Array(String)"),
        Column("payload_json", "String"),
        # Orders rows that share a timestamp; older rows take their timestamp.
        Column("emitted_at", "DateTime64(3)", backfill="timestamp"),
        *_kafka_columns(),
    ),
    engine="ReplacingMergeTree()",
    partition_by="toYYYYMMDD(timestamp)",
    order_by="(timestamp, scope_type, scope_key, event_id)",
    ttl="toDateTime(timestamp) + INTERVAL 30 DAY",
)

# One row per chat request (inference.completed): no query text, no context.
# A replay of one request (same id and ts) collapses to its latest version;
# distinct calls that reuse an id keep their rows.
CHAT_REQUESTS = Table(
    name="chat_requests",
    columns=(
        Column("request_id", "String"),
        Column("ts", "DateTime64(3)"),
        Column("completed_at", "DateTime64(3)"),
        Column("version", "UInt64"),
        Column("user_id", "String"),
        Column("tier", "LowCardinality(String)"),
        Column("source", "LowCardinality(String)"),
        Column("surface", "LowCardinality(String)"),
        Column("thread_key_hash", "String"),
        Column("query_type", "LowCardinality(String)"),
        Column("model", "LowCardinality(String)"),
        Column("provider", "LowCardinality(String)"),
        Column("status", "LowCardinality(String)"),
        Column("error_type", "LowCardinality(String)"),
        Column("error_code", "LowCardinality(String)"),
        Column("finish_reason", "LowCardinality(String)"),
        Column("fallback_from_model", "LowCardinality(String)"),
        Column("attempts", "UInt8"),
        Column("cached", "Boolean"),
        Column("routing_strategy", "LowCardinality(String)"),
        Column("policy_source", "LowCardinality(String)"),
        Column("reasoning_effort", "LowCardinality(String)"),
        Column("latency_ms", "UInt32"),
        Column("routing_ms", "Nullable(UInt32)"),
        Column("queue_wait_ms", "Nullable(UInt32)"),
        Column("ttft_ms", "Nullable(UInt32)"),
        Column("token_count_input", "UInt32"),
        Column("token_count_output", "UInt32"),
        Column("reasoning_tokens", "UInt32"),
        Column("provider_cached_input_tokens", "UInt32"),
        Column("cost_usd", "Float64"),
        Column("event_id", "String"),
        *_kafka_columns(),
    ),
    engine="ReplacingMergeTree(version)",
    partition_by="toYYYYMMDD(ts)",
    order_by="(toStartOfHour(ts), request_id, ts)",
    ttl="toDateTime(ts) + INTERVAL 90 DAY",
)

# A rating of a chat answer by the user who clicked it; each click is one
# row, the latest per user and request is their rating.
CHAT_FEEDBACK = Table(
    name="chat_feedback",
    columns=(
        Column("request_id", "String"),
        Column("user_id", "String"),
        Column("rating", "LowCardinality(String)"),
        Column("click_id", "String"),
        Column("source", "LowCardinality(String)"),
        Column("clicked_at", "DateTime64(3)"),
        Column("version", "UInt64"),
        Column("event_id", "String"),
        *_kafka_columns(),
    ),
    engine="ReplacingMergeTree(version)",
    partition_by="toYYYYMM(clicked_at)",
    order_by="(request_id, user_id, click_id)",
    ttl="toDateTime(clicked_at) + INTERVAL 90 DAY",
)

ROUTING_DECISIONS = Table(
    name="routing_decisions",
    columns=(
        Column("event_id", "String"),
        Column("timestamp", "DateTime64(3)"),
        Column("request_id", "String"),
        Column("selected_model", "String"),
        Column("served_model", "String"),
        Column("fallback_from_model", "String", backfill="''"),
        Column("user_tier", "String"),
        Column("query_type", "String"),
        Column("routing_strategy", "String"),
        Column("reason_code", "String"),
        Column("routing_reason", "String"),
        Column("policy_source", "String"),
        Column("token_count", "UInt32"),
        Column("estimated_cost", "Float64"),
        Column("confidence", "Float32"),
        Column("classifier_confidence", "Float32"),
        Column("route_to_fast_lane", "Boolean"),
        Column("actual_fast_lane_hit", "Boolean"),
        Column("fallback_models", "Array(String)"),
        Column("shadow_selected_model", "String"),
        Column("shadow_policy_source", "String"),
        Column("candidates_json", "String"),
        *_kafka_columns(),
    ),
    engine="ReplacingMergeTree()",
    partition_by="toYYYYMMDD(timestamp)",
    order_by="(timestamp, event_id)",
    ttl="toDateTime(timestamp) + INTERVAL 30 DAY",
)

# Creation order; new tables go at the end.
TABLES: Tuple[Table, ...] = (
    QUERY_LOGS,
    SYSTEM_METRICS,
    MODEL_PERFORMANCE,
    USER_ANALYTICS,
    ALERT_EVENTS,
    ROUTING_POLICY_STATE_EVENTS,
    CHAT_REQUESTS,
    CHAT_FEEDBACK,
    ROUTING_DECISIONS,
)
