import assert from "node:assert/strict";
import { test } from "node:test";

import type { PublicRunEvent } from "../src/pi-rpc.js";
import { CodingAgentOrchestrator, type QualityServices } from "../src/orchestrator.js";

const MINUTE = 60_000;

class FakeProcess {
  handler: (event: PublicRunEvent) => void = () => undefined;
  responses: Array<{ id: string; response: { value?: string; cancelled?: true } }> = [];
  prompt() {}
  decide() {}
  respondUi(id: string, response: { value?: string; cancelled?: true }) { this.responses.push({ id, response }); }
  async abort() {}
  async close() {}
  emit(event: PublicRunEvent) { this.handler(event); }
  /** Asks budget_status as the policy extension does and returns the host's parsed reply. */
  ask(id: string): unknown {
    this.emit({ type: "budget_request", request_id: id });
    const found = this.responses.find(item => item.id === id);
    assert.ok(found?.response.value, `budget_status ${id} was not answered`);
    return JSON.parse(found.response.value);
  }
}

type BudgetStatus = ConstructorParameters<typeof CodingAgentOrchestrator>[0]["budgetStatus"];

function fixture(options: { quality?: Partial<QualityServices>; budgetStatus?: BudgetStatus } = {}) {
  const process = new FakeProcess();
  const quality: QualityServices | undefined = options.quality && {
    evaluate: async () => { throw new Error("not reached"); }, assertUnchanged: async () => undefined, execution: () => undefined,
    ...options.quality,
  };
  const orchestrator = new CodingAgentOrchestrator({
    worktrees: {
      create: async () => ({ path: "/managed/w1", branch: "pi-agent/run", baselineCommit: "a".repeat(40) }),
      inspectDiff: async () => ({ files: [], bytes: 0, stat: "" }), commit: async () => "b".repeat(40),
      rollback: async () => undefined, remove: async () => undefined,
    },
    startProcess: async (_session, onEvent) => { process.handler = onEvent; return process; },
    deadlineMs: 20 * MINUTE,
    ...(quality ? { quality } : {}),
    ...(options.budgetStatus ? { budgetStatus: options.budgetStatus } : {}),
  });
  return { orchestrator, process };
}

test("a run without a profile answers budget_status from the host's ledger, and a failed read answers unavailable", async () => {
  const asked: string[] = [];
  let fail = false;
  const { orchestrator, process } = fixture({ budgetStatus: runId => {
    asked.push(runId);
    if (fail) throw new Error("database is closed");
    return { used_usd: 1.25, limit_usd: 5 };
  } });
  const created = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "ledger", user_id: "U", prompt: "fix" });
  assert.deepEqual(process.ask("ui-1"), { used_usd: 1.25, limit_usd: 5 });
  assert.deepEqual(asked, [created.run_id], "the ledger is read under the run's own id, as its gateway token spends it");
  fail = true;
  assert.deepEqual(process.ask("ui-2"), { unavailable: true }, "the dialog is still answered, so the writer never waits out its timeout");
  await orchestrator.shutdown();
});

test("a run without a profile adds its deadline's time once less than a quarter of it is left", async t => {
  t.mock.timers.enable({ apis: ["Date", "setTimeout"] });
  const { orchestrator, process } = fixture({ budgetStatus: () => ({ used_usd: 0, limit_usd: 5 }) });
  await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "deadline", user_id: "U", prompt: "fix" });
  t.mock.timers.tick(15 * MINUTE);
  assert.deepEqual(process.ask("ui-1"), { used_usd: 0, limit_usd: 5 }, "exactly a quarter left is not yet short");
  t.mock.timers.tick(MINUTE);
  assert.deepEqual(process.ask("ui-2"), { used_usd: 0, limit_usd: 5, used_ms: 16 * MINUTE, limit_ms: 20 * MINUTE });
  t.mock.timers.reset();
  await orchestrator.shutdown();
});

test("a run with a profile answers from its workflow budget as before, and adds its active time once less than a quarter is left", async t => {
  t.mock.timers.enable({ apis: ["Date", "setTimeout"] });
  let ledgerReads = 0;
  const { orchestrator, process } = fixture({
    quality: { maxCostUsd: 4, activeTimeoutMs: 10 * MINUTE },
    budgetStatus: () => { ledgerReads++; return { used_usd: 0, limit_usd: 5 }; },
  });
  await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "profile", user_id: "U", prompt: "fix" });
  process.emit({ type: "usage", cost_usd: 3 });
  t.mock.timers.tick(5 * MINUTE);
  assert.deepEqual(process.ask("ui-1"), { used_usd: 3, limit_usd: 4 });
  t.mock.timers.tick(3 * MINUTE);
  assert.deepEqual(process.ask("ui-2"), { used_usd: 3, limit_usd: 4, used_ms: 8 * MINUTE, limit_ms: 10 * MINUTE });
  assert.equal(ledgerReads, 0, "a profile run's dollars come from its workflow budget, never the default ledger budget");
  t.mock.timers.reset();
  await orchestrator.shutdown();
});
