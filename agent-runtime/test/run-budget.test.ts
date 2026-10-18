import assert from "node:assert/strict";
import { test } from "node:test";

import { resolveAgentModel } from "../src/agent-model.js";
import { budgetIdFor, outputTokenLimitFor, parseDefaultRunBudget, priceFor } from "../src/run-budget.js";

test("each harness role has its own output-token cap", () => {
  assert.equal(outputTokenLimitFor("writer"), 32768);
  assert.equal(outputTokenLimitFor("grader"), 16384);
  assert.equal(outputTokenLimitFor("investigator"), 8192);
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
  assert.deepEqual(priceFor(luna.ref, {}), { input: luna.cost.input, output: luna.cost.output });
  assert.equal(priceFor("openai/unknown", {}), undefined);
});
