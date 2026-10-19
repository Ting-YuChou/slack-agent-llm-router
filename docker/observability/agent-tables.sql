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
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL toDateTime(collected_at) + INTERVAL 30 DAY;

CREATE VIEW IF NOT EXISTS agent_events_latest AS SELECT * FROM agent_events FINAL;

CREATE TABLE IF NOT EXISTS agent_workflow_nodes (
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
  workflow_run_id String,
  node_id String,
  execution_id String,
  role String,
  wave UInt32,
  attempt UInt32,
  state String,
  duration_ms UInt64,
  queue_ms UInt64,
  result_ref String
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL toDateTime(collected_at) + INTERVAL 90 DAY;

CREATE VIEW IF NOT EXISTS agent_workflow_nodes_latest AS SELECT * FROM agent_workflow_nodes FINAL;

CREATE TABLE IF NOT EXISTS agent_workflow_revisions (
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
  workflow_run_id String,
  revision UInt32,
  task_count UInt32
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL toDateTime(collected_at) + INTERVAL 90 DAY;

CREATE VIEW IF NOT EXISTS agent_workflow_revisions_latest AS SELECT * FROM agent_workflow_revisions FINAL;

CREATE TABLE IF NOT EXISTS agent_workflow_dependencies (
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
  workflow_run_id String,
  node_id String,
  depends_on String
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL toDateTime(collected_at) + INTERVAL 90 DAY;

CREATE VIEW IF NOT EXISTS agent_workflow_dependencies_latest AS SELECT * FROM agent_workflow_dependencies FINAL;

CREATE TABLE IF NOT EXISTS agent_executions (
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
  execution_id String,
  workflow_run_id String,
  role String,
  attempt UInt32,
  phase String,
  prompts_version String
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL toDateTime(collected_at) + INTERVAL 90 DAY;

CREATE VIEW IF NOT EXISTS agent_executions_latest AS SELECT * FROM agent_executions FINAL;

CREATE TABLE IF NOT EXISTS agent_verifications (
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
  candidate_hash String,
  attempt UInt32,
  status String
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL toDateTime(collected_at) + INTERVAL 90 DAY;

CREATE VIEW IF NOT EXISTS agent_verifications_latest AS SELECT * FROM agent_verifications FINAL;

CREATE TABLE IF NOT EXISTS agent_grades (
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
  candidate_hash String,
  attempt UInt32,
  status String,
  prompts_version String
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL toDateTime(collected_at) + INTERVAL 90 DAY;

CREATE VIEW IF NOT EXISTS agent_grades_latest AS SELECT * FROM agent_grades FINAL;

CREATE TABLE IF NOT EXISTS agent_deliveries (
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
  candidate_hash String,
  status String
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL toDateTime(collected_at) + INTERVAL 90 DAY;

CREATE VIEW IF NOT EXISTS agent_deliveries_latest AS SELECT * FROM agent_deliveries FINAL;

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
  capture_complete Nullable(UInt8),
  prompts_version String
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY run_id TTL toDateTime(collected_at) + INTERVAL 90 DAY;

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
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL toDateTime(collected_at) + INTERVAL 90 DAY;

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
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL toDateTime(collected_at) + INTERVAL 90 DAY;

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
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL toDateTime(collected_at) + INTERVAL 90 DAY;

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
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL toDateTime(collected_at) + INTERVAL 30 DAY;

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
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL toDateTime(collected_at) + INTERVAL 90 DAY;

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
  feedback_id String,
  reason_category String,
  comment String
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL toDateTime(collected_at) + INTERVAL 90 DAY;

CREATE VIEW IF NOT EXISTS agent_feedback_latest AS SELECT * FROM agent_feedback FINAL;

CREATE TABLE IF NOT EXISTS agent_policy_decisions (
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
  tool String,
  decision String,
  reason String,
  approved Nullable(UInt8),
  argv0 String,
  cmd_hash String,
  latency_ms Nullable(UInt64)
) ENGINE=ReplacingMergeTree(version) PARTITION BY toYYYYMM(collected_at) ORDER BY event_id TTL toDateTime(collected_at) + INTERVAL 30 DAY;

CREATE VIEW IF NOT EXISTS agent_policy_decisions_latest AS SELECT * FROM agent_policy_decisions FINAL;

CREATE VIEW IF NOT EXISTS agent_feedback_current AS SELECT run_id, user_id, argMax(verdict, version) AS verdict FROM agent_feedback_latest GROUP BY run_id, user_id;

CREATE VIEW IF NOT EXISTS agent_grade_feedback_agreement AS
SELECT f.run_id AS run_id, f.user_id AS user_id, g.grade_status AS grade_status,
  g.candidate_hash AS candidate_hash, g.prompts_version AS prompts_version, f.verdict AS feedback_verdict,
  multiIf(g.grade_status = 'passed', f.verdict = 'accepted',
    g.grade_status = 'needs_changes', f.verdict = 'needs_changes', NULL) AS agrees
FROM agent_feedback_current f INNER JOIN
  (SELECT run_id, argMax(status, (attempt, version)) AS grade_status,
    argMax(candidate_hash, (attempt, version)) AS candidate_hash,
    argMax(prompts_version, (attempt, version)) AS prompts_version
  FROM agent_grades_latest GROUP BY run_id) g ON f.run_id = g.run_id;

CREATE VIEW IF NOT EXISTS agent_usage_reconciliation AS
SELECT r.run_id AS run_id, r.capture_complete, s.start_snapshots, s.end_snapshots,
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

CREATE VIEW IF NOT EXISTS agent_harness_metrics AS
SELECT run_id, status, model, collected_at,
  JSONExtractString(payload_json,'harness_metrics','outcome') AS outcome,
  JSONExtract(payload_json,'harness_metrics','turn_util','Nullable(Float64)') AS turn_util,
  JSONExtract(payload_json,'harness_metrics','tool_util','Nullable(Float64)') AS tool_util,
  JSONExtractUInt(payload_json,'harness_metrics','attempts') AS attempts,
  JSONExtract(payload_json,'harness_metrics','first_attempt_pass','Nullable(Bool)') AS first_attempt_pass,
  JSONExtract(payload_json,'harness_metrics','repair_success','Nullable(Bool)') AS repair_success,
  JSONExtractUInt(payload_json,'harness_metrics','writer_checks_calls') AS writer_checks_calls,
  JSONExtractUInt(payload_json,'harness_metrics','approvals') AS approvals,
  JSONExtractUInt(payload_json,'harness_metrics','policy_blocks') AS policy_blocks,
  JSONExtract(payload_json,'harness_metrics','cost_usd','Nullable(Float64)') AS cost_usd,
  JSONExtractFloat(payload_json,'harness_metrics','wall_s') AS wall_s,
  JSONExtract(payload_json,'harness_metrics','active_s','Nullable(Float64)') AS active_s,
  JSONExtract(payload_json,'harness_metrics','human_wait_s','Nullable(Float64)') AS human_wait_s,
  JSONExtract(payload_json,'harness_metrics','time_to_first_edit_s','Nullable(Float64)') AS time_to_first_edit_s
FROM agent_runs_latest WHERE JSONHas(payload_json,'harness_metrics');

CREATE VIEW IF NOT EXISTS agent_workflow_execution_durations AS
SELECT workflow_run_id, execution_id, any(role) AS role,
  if(countIf(phase='end' AND JSONHas(payload_json,'duration_ms'))>0,
    maxIf(JSONExtractUInt(payload_json,'duration_ms'),phase='end'),NULL) AS duration_ms,
  if(countIf(phase='end' AND JSONHas(payload_json,'queue_ms'))>0,
    maxIf(JSONExtractUInt(payload_json,'queue_ms'),phase='end'),NULL) AS queue_ms,
  greatest(toInt64(duration_ms)-toInt64(queue_ms),0) AS active_agent_ms
FROM agent_executions_latest GROUP BY workflow_run_id,execution_id;

CREATE VIEW IF NOT EXISTS agent_workflow_usage AS
SELECT a.*, if(JSONHas(r.routing_json,'jevCostUsd'), JSONExtractFloat(r.routing_json,'jevCostUsd'),
  if(JSONHas(r.routing_json,'jevLatencyMs'), NULL, 0)) AS routing_cost_usd,
  if(a.incomplete_executions=0 AND routing_cost_usd IS NOT NULL,
    a.known_execution_cost_usd+routing_cost_usd, NULL) AS total_cost_usd
FROM (
  SELECT e.workflow_run_id, count() AS executions, sum(u.ledger_tokens) AS total_tokens,
    sum(u.ledger_cost) AS known_execution_cost_usd,
    countIf(u.reconciliation='matched') AS matched_executions,
    countIf(u.reconciliation!='matched') AS incomplete_executions
  FROM (SELECT execution_id, any(workflow_run_id) AS workflow_run_id FROM agent_executions_latest GROUP BY execution_id) e
  LEFT JOIN agent_usage_reconciliation u ON e.execution_id=u.run_id GROUP BY e.workflow_run_id
) a LEFT JOIN agent_runs_latest r ON a.workflow_run_id=r.run_id;
