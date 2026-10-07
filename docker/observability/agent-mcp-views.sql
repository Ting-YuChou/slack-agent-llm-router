CREATE DATABASE IF NOT EXISTS agent_mcp;
CREATE USER IF NOT EXISTS agent_mcp_view_owner HOST NONE NOT IDENTIFIED;
GRANT SELECT ON default.agent_runs TO agent_mcp_view_owner;
GRANT SELECT ON default.agent_usage TO agent_mcp_view_owner;
GRANT SELECT ON default.agent_tool_calls TO agent_mcp_view_owner;
GRANT SELECT ON default.agent_test_results TO agent_mcp_view_owner;
GRANT SELECT ON default.otel_traces TO agent_mcp_view_owner;

CREATE OR REPLACE VIEW agent_mcp.run_metrics_hourly
DEFINER = agent_mcp_view_owner SQL SECURITY DEFINER AS
SELECT toStartOfHour(collected_at) AS hour, model, reasoning_effort, status,
       count() AS runs,
       countIf(status NOT IN ('completed', 'succeeded')) AS errors,
       quantile(0.5)(JSONExtractFloat(payload_json, 'duration_ms')) AS p50_latency_ms,
       quantile(0.95)(JSONExtractFloat(payload_json, 'duration_ms')) AS p95_latency_ms
FROM default.agent_runs_latest
GROUP BY hour, model, reasoning_effort, status;

CREATE OR REPLACE VIEW agent_mcp.model_metrics_daily
DEFINER = agent_mcp_view_owner SQL SECURITY DEFINER AS
SELECT toStartOfDay(r.collected_at) AS day, r.model, r.reasoning_effort, r.status,
       countDistinct(r.run_id) AS runs, sumOrNull(u.total_tokens) AS total_tokens,
       sumOrNull(u.cost_total) AS cost_usd
FROM default.agent_runs_latest AS r
LEFT JOIN default.agent_usage_latest AS u ON r.run_id = u.run_id
GROUP BY day, r.model, r.reasoning_effort, r.status;

CREATE OR REPLACE VIEW agent_mcp.tool_metrics_hourly
DEFINER = agent_mcp_view_owner SQL SECURITY DEFINER AS
SELECT toStartOfHour(collected_at) AS hour, tool, phase, count() AS calls,
       countIf(ifNull(is_error, 0) = 1) AS errors
FROM default.agent_tool_calls_latest
GROUP BY hour, tool, phase;

CREATE OR REPLACE VIEW agent_mcp.test_metrics_hourly
DEFINER = agent_mcp_view_owner SQL SECURITY DEFINER AS
SELECT toStartOfHour(collected_at) AS hour, status, parser, count() AS test_runs,
       sumOrNull(passed) AS passed, sumOrNull(failed) AS failed
FROM default.agent_test_results_latest
GROUP BY hour, status, parser;

CREATE OR REPLACE VIEW agent_mcp.trace_metrics_hourly
DEFINER = agent_mcp_view_owner SQL SECURITY DEFINER AS
SELECT toStartOfHour(Timestamp) AS hour, SpanName AS span, StatusCode AS status,
       count() AS spans, quantile(0.5)(Duration / 1000000) AS p50_latency_ms,
       quantile(0.95)(Duration / 1000000) AS p95_latency_ms,
       countIf(StatusCode = 'Error') AS errors
FROM default.otel_traces
GROUP BY hour, span, status;

GRANT SELECT ON agent_mcp.run_metrics_hourly TO agent_mcp_reader;
GRANT SELECT ON agent_mcp.model_metrics_daily TO agent_mcp_reader;
GRANT SELECT ON agent_mcp.tool_metrics_hourly TO agent_mcp_reader;
GRANT SELECT ON agent_mcp.test_metrics_hourly TO agent_mcp_reader;
GRANT SELECT ON agent_mcp.trace_metrics_hourly TO agent_mcp_reader;
