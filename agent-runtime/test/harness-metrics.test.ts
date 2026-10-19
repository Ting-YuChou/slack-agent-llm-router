import assert from "node:assert/strict";
import { test } from "node:test";

import { computeHarnessMetrics } from "../src/harness-metrics.js";
import type { RunRecord } from "../src/orchestrator.js";

const created = "2026-10-19T10:00:00.000Z";
function run(extra: Partial<RunRecord>): RunRecord {
  return { run_id: "r", session_id: "s", status: "completed", answer: "", owner_user_id: "U", created_at: created, updated_at: "2026-10-19T10:02:00.000Z", tool_count: 0, turn_count: 0, events: [], ...extra } as RunRecord;
}
const now = Date.parse("2026-10-19T10:02:00.000Z");

test("a no-profile run reports utilization and timing but no attempt verdicts", () => {
  const metrics = computeHarnessMetrics(run({ execution_outcome: "completed", turn_count: 5, tool_count: 10, limits: { maxTurns: 20, maxToolCalls: 40 }, first_edit_at: "2026-10-19T10:00:30.000Z" }), now);
  assert.deepEqual(metrics, {
    outcome: "completed", turn_util: 0.25, tool_util: 0.25, attempts: 0, first_attempt_pass: null, repair_success: null,
    writer_checks_calls: 0, approvals: 0, policy_blocks: 0, fixed_files: 0, cost_usd: null, wall_s: 120, active_s: null, human_wait_s: null, time_to_first_edit_s: 30,
  });
});

test("a run repaired once and then delivered counts as a repair success, not a first-attempt pass", () => {
  const metrics = computeHarnessMetrics(run({
    attempts: [
      { attempt: 1, execution_id: "e1", verification_status: "failed", grader_status: "unknown" },
      { attempt: 2, execution_id: "e2", verification_status: "passed", grader_status: "passed" },
    ],
    writer_checks_calls: 2, estimated_cost_usd: 0.04, approvals: { requested: 1, approved: 1, rejected: 0, expired: 0, wait_ms: 4000 },
    policy_summary: { allow: 3, block: 2, approval: 1, approved: 1, by_reason: {} },
    timing: { wall_ms: 90_000, active_ms: 60_000, human_wait_ms: 30_000 },
  }), now);
  assert.equal(metrics.attempts, 2);
  assert.equal(metrics.first_attempt_pass, false);
  assert.equal(metrics.repair_success, true);
  assert.equal(metrics.writer_checks_calls, 2);
  assert.equal(metrics.approvals, 1);
  assert.equal(metrics.policy_blocks, 2);
  assert.equal(metrics.cost_usd, 0.04);
  assert.deepEqual([metrics.wall_s, metrics.active_s, metrics.human_wait_s], [90, 60, 30]);
});

test("a run that hit its tool limit on the first attempt fails both verdicts", () => {
  const metrics = computeHarnessMetrics(run({ status: "failed", execution_outcome: "limit_reached", tool_count: 41, limits: { maxTurns: 20, maxToolCalls: 40 },
    attempts: [{ attempt: 1, execution_id: "e1" }] }), now);
  assert.equal(metrics.outcome, "limit_reached");
  assert.equal(metrics.tool_util, 1.025);
  assert.equal(metrics.first_attempt_pass, false);
  assert.equal(metrics.repair_success, null);
});

test("a first attempt that passes both gates is a first-attempt pass", () => {
  const metrics = computeHarnessMetrics(run({ attempts: [{ attempt: 1, execution_id: "e1", verification_status: "passed", grader_status: "passed" }] }), now);
  assert.equal(metrics.first_attempt_pass, true);
  assert.equal(metrics.repair_success, null);
});

test("files rewritten by fixers are summed over attempts", () => {
  const metrics = computeHarnessMetrics(run({ attempts: [
    { attempt: 1, execution_id: "e1", verification_status: "failed", grader_status: "unknown", fixes: [{ name: "black", files: ["a.py", "b.py"] }] },
    { attempt: 2, execution_id: "e2", verification_status: "passed", grader_status: "passed", fixes: [{ name: "black", files: ["a.py"] }, { name: "ruff", files: ["a.py"] }] },
  ] }), now);
  assert.equal(metrics.fixed_files, 4);
});
