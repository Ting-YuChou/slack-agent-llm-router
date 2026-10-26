import assert from "node:assert/strict";
import { test } from "node:test";

import type { DiskQuota } from "../src/disk-usage.js";
import { CodingAgentOrchestrator } from "../src/orchestrator.js";
import type { PublicRunEvent } from "../src/pi-rpc.js";
import { waitFor } from "./support/scenario.js";

const MIB = 1024 ** 2;
const WORKTREE = "/managed/w1";
const SESSION_DIR = "/runtime/sessions/S1";

class FakeProcess {
  readonly sessionStatePath = SESSION_DIR;
  handler: (event: PublicRunEvent) => void = () => undefined;
  aborted: string[] = [];
  prompt() {}
  decide() {}
  respondUi() {}
  async abort(runId: string) { this.aborted.push(runId); }
  async close() {}
}

/** Stands in for du: the size of each directory, and every directory measured. */
function disk(initial: Record<string, number>) {
  const bytes = { ...initial };
  const measured = new Set<string>();
  const quota = (limit: number): DiskQuota => ({ bytes: limit, intervalMs: 5, usage: async (directory) => { measured.add(directory); return bytes[directory] ?? 0; } });
  return { bytes, measured, quota };
}

function fixture(quotas: { diskQuota?: DiskQuota; sessionQuota?: DiskQuota }) {
  const process = new FakeProcess();
  const calls = { rollback: 0, commits: 0 };
  const orchestrator = new CodingAgentOrchestrator({
    worktrees: {
      create: async () => ({ path: WORKTREE, branch: "pi-agent/run", baselineCommit: "a".repeat(40) }),
      inspectDiff: async () => ({ files: ["src/a.ts"], bytes: 10, stat: "1 file changed" }),
      commit: async () => { calls.commits += 1; return "b".repeat(40); },
      rollback: async () => { calls.rollback += 1; },
      remove: async () => undefined,
    },
    startProcess: async (_session, onEvent) => { process.handler = onEvent; return process; },
    deadlineMs: 10_000,
    ...quotas,
  });
  return { orchestrator, process, calls };
}

test("a session directory over its quota fails the run as session_quota_exceeded and rolls back", async () => {
  // The worktree is over the session quota but under its own: each quota measures only its directory.
  const sizes = disk({ [WORKTREE]: 3 * MIB, [SESSION_DIR]: MIB });
  const { orchestrator, process, calls } = fixture({ diskQuota: sizes.quota(4 * MIB), sessionQuota: sizes.quota(2 * MIB) });
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "session-quota", user_id: "U", prompt: "fix" });
  await waitFor(() => sizes.measured.size, count => count === 2, () => `measured ${[...sizes.measured]}`);
  await new Promise(resolve => setTimeout(resolve, 30));
  assert.equal(orchestrator.getRun(accepted.run_id).status, "running", "under both quotas the run continues");
  sizes.bytes[SESSION_DIR] = 2 * MIB + 1;
  const run = await waitFor(() => orchestrator.getRun(accepted.run_id), current => current.status === "failed" && calls.rollback > 0, current => `status ${current.status}`);
  assert.equal(run.error?.code, "session_quota_exceeded");
  assert.match(run.error?.message ?? "", /session grew past its 2 MiB quota/);
  assert.deepEqual(process.aborted, [accepted.run_id], "the container is stopped before the rollback");
  assert.equal(calls.commits, 0);
  assert.deepEqual([...sizes.measured].sort(), [WORKTREE, SESSION_DIR].sort());
});

test("a discussion's read-only worktree is not measured, but its session directory is", async () => {
  const sizes = disk({ [WORKTREE]: 10_000, [SESSION_DIR]: 101 });
  const { orchestrator } = fixture({ diskQuota: sizes.quota(100), sessionQuota: sizes.quota(100) });
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "session-quota-discuss", user_id: "U", prompt: "why?", task_mode: "discuss" });
  const run = await waitFor(() => orchestrator.getRun(accepted.run_id), current => current.status === "failed", current => `status ${current.status}`);
  assert.equal(run.error?.code, "session_quota_exceeded");
  assert.deepEqual([...sizes.measured], [SESSION_DIR]);
});

test("without a session quota only the worktree is measured, as before", async () => {
  const sizes = disk({ [WORKTREE]: 10, [SESSION_DIR]: 10 ** 12 });
  const { orchestrator } = fixture({ diskQuota: sizes.quota(100) });
  const accepted = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "session-quota-off", user_id: "U", prompt: "fix" });
  await waitFor(() => sizes.measured.size, count => count > 0, () => "nothing measured");
  await new Promise(resolve => setTimeout(resolve, 30));
  assert.equal(orchestrator.getRun(accepted.run_id).status, "running");
  assert.deepEqual([...sizes.measured], [WORKTREE]);
});
