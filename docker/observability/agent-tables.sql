CREATE TABLE IF NOT EXISTS agent_events (
  sequence UInt64,
  schema_version UInt16,
  pi_version String,
  trace_id Nullable(String),
  source_time Nullable(DateTime64(6, 'UTC')),
  event_id String,
  run_id String,
  session_id String,
  kind String,
  collected_at DateTime64(6, 'UTC'),
  version UInt64,
  payload_json String CODEC(ZSTD(3))
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL collected_at + INTERVAL 30 DAY;

CREATE VIEW IF NOT EXISTS agent_events_latest AS SELECT * FROM agent_events FINAL;

CREATE TABLE IF NOT EXISTS agent_runs (
  sequence UInt64,
  schema_version UInt16,
  pi_version String,
  trace_id Nullable(String),
  source_time Nullable(DateTime64(6, 'UTC')),
  event_id String,
  run_id String,
  session_id String,
  kind String,
  collected_at DateTime64(6, 'UTC'),
  version UInt64,
  payload_json String CODEC(ZSTD(3)),
  status String,
  model String,
  reasoning_effort String,
  routing_json String,
  capture_complete Nullable(UInt8)
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY run_id TTL collected_at + INTERVAL 90 DAY;

CREATE VIEW IF NOT EXISTS agent_runs_latest AS SELECT * FROM agent_runs FINAL;

CREATE TABLE IF NOT EXISTS agent_usage (
  sequence UInt64,
  schema_version UInt16,
  pi_version String,
  trace_id Nullable(String),
  source_time Nullable(DateTime64(6, 'UTC')),
  event_id String,
  run_id String,
  session_id String,
  kind String,
  collected_at DateTime64(6, 'UTC'),
  version UInt64,
  payload_json String CODEC(ZSTD(3)),
  pi_session_id String,
  entry_id String,
  parent_id Nullable(String),
  source String,
  input Nullable(UInt64),
  output Nullable(UInt64),
  cache_read Nullable(UInt64),
  cache_write Nullable(UInt64),
  cache_write_1h Nullable(UInt64),
  reasoning Nullable(UInt64),
  total_tokens Nullable(UInt64),
  cost_input Nullable(Float64),
  cost_output Nullable(Float64),
  cost_cache_read Nullable(Float64),
  cost_cache_write Nullable(Float64),
  cost_total Nullable(Float64)
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL collected_at + INTERVAL 90 DAY;

CREATE VIEW IF NOT EXISTS agent_usage_latest AS SELECT * FROM agent_usage FINAL;

CREATE TABLE IF NOT EXISTS agent_tool_calls (
  sequence UInt64,
  schema_version UInt16,
  pi_version String,
  trace_id Nullable(String),
  source_time Nullable(DateTime64(6, 'UTC')),
  event_id String,
  run_id String,
  session_id String,
  kind String,
  collected_at DateTime64(6, 'UTC'),
  version UInt64,
  payload_json String CODEC(ZSTD(3)),
  tool_call_id String,
  tool String,
  phase String,
  is_error Nullable(UInt8)
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL collected_at + INTERVAL 90 DAY;

CREATE VIEW IF NOT EXISTS agent_tool_calls_latest AS SELECT * FROM agent_tool_calls FINAL;

CREATE TABLE IF NOT EXISTS agent_session_stats (
  sequence UInt64,
  schema_version UInt16,
  pi_version String,
  trace_id Nullable(String),
  source_time Nullable(DateTime64(6, 'UTC')),
  event_id String,
  run_id String,
  session_id String,
  kind String,
  collected_at DateTime64(6, 'UTC'),
  version UInt64,
  payload_json String CODEC(ZSTD(3)),
  phase String,
  pi_session_id String
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL collected_at + INTERVAL 90 DAY;

CREATE VIEW IF NOT EXISTS agent_session_stats_latest AS SELECT * FROM agent_session_stats FINAL;

CREATE TABLE IF NOT EXISTS agent_content_chunks (
  sequence UInt64,
  schema_version UInt16,
  pi_version String,
  trace_id Nullable(String),
  source_time Nullable(DateTime64(6, 'UTC')),
  event_id String,
  run_id String,
  session_id String,
  kind String,
  collected_at DateTime64(6, 'UTC'),
  version UInt64,
  payload_json String CODEC(ZSTD(3)),
  content_id String,
  chunk_index UInt32,
  chunk_count UInt32,
  sha256 String,
  data String CODEC(ZSTD(3))
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL collected_at + INTERVAL 30 DAY;

CREATE VIEW IF NOT EXISTS agent_content_chunks_latest AS SELECT * FROM agent_content_chunks FINAL;

CREATE TABLE IF NOT EXISTS agent_test_results (
  sequence UInt64,
  schema_version UInt16,
  pi_version String,
  trace_id Nullable(String),
  source_time Nullable(DateTime64(6, 'UTC')),
  event_id String,
  run_id String,
  session_id String,
  kind String,
  collected_at DateTime64(6, 'UTC'),
  version UInt64,
  payload_json String CODEC(ZSTD(3)),
  tool_call_id String,
  status String,
  parser String,
  passed Nullable(UInt64),
  failed Nullable(UInt64)
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL collected_at + INTERVAL 90 DAY;

CREATE VIEW IF NOT EXISTS agent_test_results_latest AS SELECT * FROM agent_test_results FINAL;

CREATE TABLE IF NOT EXISTS agent_feedback (
  sequence UInt64,
  schema_version UInt16,
  pi_version String,
  trace_id Nullable(String),
  source_time Nullable(DateTime64(6, 'UTC')),
  event_id String,
  run_id String,
  session_id String,
  kind String,
  collected_at DateTime64(6, 'UTC'),
  version UInt64,
  payload_json String CODEC(ZSTD(3)),
  user_id String,
  verdict String,
  feedback_id String
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL collected_at + INTERVAL 90 DAY;

CREATE VIEW IF NOT EXISTS agent_feedback_latest AS SELECT * FROM agent_feedback FINAL;

CREATE VIEW IF NOT EXISTS agent_feedback_current AS SELECT run_id, user_id, argMax(verdict, version) AS verdict FROM agent_feedback_latest GROUP BY run_id, user_id;

CREATE VIEW IF NOT EXISTS agent_usage_reconciliation AS
SELECT r.run_id, r.capture_complete, s.start_snapshots, s.end_snapshots,
  if(s.start_snapshots>0 AND s.end_snapshots>0, s.end_tokens-s.start_tokens, NULL) AS session_token_delta,
  u.ledger_tokens, u.ledger_cost,
  if(s.start_snapshots>0 AND s.end_snapshots>0, s.end_cost-s.start_cost, NULL) AS session_cost_delta,
  multiIf(r.capture_complete IS NULL OR r.capture_complete=0 OR s.start_snapshots=0 OR s.end_snapshots=0 OR u.reported_costs<u.ledger_rows OR u.reported_tokens<u.ledger_rows, 'unknown',
    abs((s.end_cost-s.start_cost)-u.ledger_cost)<0.00000001 AND s.end_tokens-s.start_tokens=u.ledger_tokens, 'matched', 'mismatch') AS reconciliation
FROM agent_runs_latest r LEFT JOIN
  (SELECT run_id, countIf(phase='start') AS start_snapshots, countIf(phase='end') AS end_snapshots,
    argMaxIf(JSONExtractInt(payload_json,'stats','tokens','total'),version,phase='start') AS start_tokens,
    argMaxIf(JSONExtractInt(payload_json,'stats','tokens','total'),version,phase='end') AS end_tokens,
    argMaxIf(JSONExtractFloat(payload_json,'stats','cost'),version,phase='start') AS start_cost,
    argMaxIf(JSONExtractFloat(payload_json,'stats','cost'),version,phase='end') AS end_cost
  FROM agent_session_stats_latest GROUP BY run_id) s ON r.run_id=s.run_id LEFT JOIN
  (SELECT run_id, count() AS ledger_rows, count(total_tokens) AS reported_tokens, count(cost_total) AS reported_costs, sum(total_tokens) AS ledger_tokens, sum(cost_total) AS ledger_cost
   FROM agent_usage_latest GROUP BY run_id) u ON r.run_id=u.run_id;
