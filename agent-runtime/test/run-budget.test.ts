import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

import { resolveAgentModel } from "../src/agent-model.js";
import { BudgetLedger } from "../src/budget-ledger.js";
import { budgetIdFor, OUTPUT_TOKEN_CEILING, outputTokenLimitFor, parseDefaultRunBudget, priceFor, releaseRunBudget, runBudgetStatus } from "../src/run-budget.js";
import { validateProfile } from "../src/verification.js";

test("each harness role has its own output-token cap", () => {
  assert.equal(outputTokenLimitFor("writer"), 32768);
  assert.equal(outputTokenLimitFor("grader"), 16384);
  assert.equal(outputTokenLimitFor("investigator"), 8192);
});

const profile = { id: "p", repository: "o/r", image: `sha256:${"0".repeat(64)}`, rubric: ["ok"], allowedPaths: ["src"], checks: [{ name: "t", argv: ["true"], cwd: ".", timeoutMs: 1000 }] };

test("a profile without limits keeps every role's built-in output cap", () => {
  const validated = validateProfile(profile);
  assert.equal(validated.limits, undefined);
  for (const role of ["writer", "grader", "investigator"] as const) assert.equal(outputTokenLimitFor(role, validated.limits), outputTokenLimitFor(role));
});

test("a profile's limits.outputTokens sets the cap of the roles it names, up to the ceiling", () => {
  const validated = validateProfile({ ...profile, limits: { outputTokens: { writer: 16384, grader: OUTPUT_TOKEN_CEILING } } });
  assert.deepEqual(validated.limits, { outputTokens: { writer: 16384, grader: OUTPUT_TOKEN_CEILING } });
  assert.equal(outputTokenLimitFor("writer", validated.limits), 16384);
  assert.equal(outputTokenLimitFor("grader", validated.limits), OUTPUT_TOKEN_CEILING);
  assert.equal(outputTokenLimitFor("investigator", validated.limits), 8192, "an unnamed role keeps its built-in cap");
});

test("a profile output cap above the ceiling, or otherwise malformed, is rejected", () => {
  assert.equal(OUTPUT_TOKEN_CEILING, 32768);
  for (const limits of [
    { outputTokens: { writer: OUTPUT_TOKEN_CEILING + 1 } }, { outputTokens: { grader: 1_000_000 } }, { outputTokens: { writer: 1023 } },
    { outputTokens: { writer: 2048.5 } }, { outputTokens: { writer: "4096" } }, { outputTokens: { planner: 4096 } }, { outputTokens: 4096 },
    { outputTokens: null }, { outputTokens: [] }, { maxTurns: 10 }, null, [], 1,
  ]) assert.throws(() => validateProfile({ ...profile, limits }), /Invalid profile limits/, JSON.stringify(limits));
});

test("an output cap never exceeds the ceiling, even from an unvalidated profile", () => {
  assert.equal(outputTokenLimitFor("writer", { outputTokens: { writer: 1_000_000 } }), OUTPUT_TOKEN_CEILING);
});

test("workflow executions spend their parent's budget; other runs get a default budget of their own", () => {
  const opened: Array<[string, number]> = [];
  const book = { open: (id: string, usd: number) => { opened.push([id, usd]); } };
  assert.equal(budgetIdFor("exec-1", new Map([["exec-1", "workflow-1"]]), book, 2), "workflow-1");
  assert.deepEqual(opened, []);
  assert.equal(budgetIdFor("run-1", new Map(), book, 2), "run-1");
  assert.deepEqual(opened, [["run-1", 2]]);
});

test("the default run budget is configurable and must be positive", () => {
  assert.equal(parseDefaultRunBudget({}), 5);
  assert.equal(parseDefaultRunBudget({ PI_AGENT_DEFAULT_RUN_BUDGET_USD: "0.5" }), 0.5);
  assert.throws(() => parseDefaultRunBudget({ PI_AGENT_DEFAULT_RUN_BUDGET_USD: "0" }), /PI_AGENT_DEFAULT_RUN_BUDGET_USD/);
  assert.throws(() => parseDefaultRunBudget({ PI_AGENT_DEFAULT_RUN_BUDGET_USD: "free" }), /PI_AGENT_DEFAULT_RUN_BUDGET_USD/);
});

test("configured prices win; otherwise the model catalog price is used", () => {
  const luna = resolveAgentModel("openai/gpt-5.6-luna");
  assert.deepEqual(priceFor(luna.ref, { [luna.ref]: { input: 9, output: 99 } }), { input: 9, output: 99 });
  assert.deepEqual(priceFor(luna.ref, {}), { input: luna.cost.input, output: luna.cost.output, cacheRead: luna.cost.cacheRead });
  assert.equal(priceFor("openai/unknown", {}), undefined);
});

test("an execution recorded under a workflow keeps that workflow's budget after a restart lost the in-memory map", () => {
  const opened: string[] = [];
  const book = { open: (id: string) => { opened.push(id); } };
  const recorded = (id: string) => id === "exec-1" ? "workflow-1" : undefined;
  assert.equal(budgetIdFor("exec-1", new Map(), book, 2, recorded), "workflow-1");
  assert.deepEqual(opened, [], "no default budget is opened for a workflow execution");
  assert.equal(budgetIdFor("run-2", new Map(), book, 2, recorded), "run-2");
});

test("a run's own default budget stops when the run ends, so its token can spend nothing more", () => {
  const stopped: string[] = [];
  const book = { stop: (id: string) => { stopped.push(id); } };
  releaseRunBudget({ run_id: "run-1", status: "running" }, new Map(), book);
  releaseRunBudget({ run_id: "run-1", status: "interrupted" }, new Map(), book);
  releaseRunBudget({ run_id: "run-1", status: "cancelled" }, new Map(), book);
  releaseRunBudget({ run_id: "exec-1", status: "failed" }, new Map([["exec-1", "workflow-1"]]), book);
  assert.deepEqual(stopped, ["run-1"], "only terminal runs on their own default budget are stopped");
});

test("every way a run can end stops its own budget, including a rejected approval", () => {
  for (const status of ["completed", "failed", "cancelled", "timed_out", "rejected"]) {
    const stopped: string[] = [];
    releaseRunBudget({ run_id: "run-1", status }, new Map(), { stop: (id: string) => { stopped.push(id); } });
    assert.deepEqual(stopped, ["run-1"], status);
  }
});

test("budget_status reads the ledger budget a run spends: its own default one, or its workflow's", () => {
  const dir = mkdtempSync(path.join(tmpdir(), "run-budget-status-"));
  const ledger = new BudgetLedger(path.join(dir, "budget.sqlite"));
  try {
    const defaultUsd = parseDefaultRunBudget({});
    assert.deepEqual(runBudgetStatus("run-1", new Map(), ledger, defaultUsd), { used_usd: 0, limit_usd: 5 });
    ledger.reserve("run-1", "r1", 0.5); ledger.settle("r1", 1.25);
    ledger.reserve("run-1", "r2", 0.25);
    assert.deepEqual(runBudgetStatus("run-1", new Map(), ledger, defaultUsd), { used_usd: 1.5, limit_usd: 5 }, "settled spend plus open reservations");
    ledger.open("workflow-1", 8, 0.5);
    assert.deepEqual(runBudgetStatus("exec-1", new Map([["exec-1", "workflow-1"]]), ledger, defaultUsd), { used_usd: 0.5, limit_usd: 8 });
    assert.deepEqual(runBudgetStatus("exec-2", new Map(), ledger, defaultUsd, id => id === "exec-2" ? "workflow-1" : undefined), { used_usd: 0.5, limit_usd: 8 });
    assert.equal(runBudgetStatus("exec-3", new Map([["exec-3", "workflow-3"]]), ledger, defaultUsd), undefined, "a workflow budget never opened has no figures");
  } finally {
    ledger.close();
    rmSync(dir, { recursive: true, force: true });
  }
});
