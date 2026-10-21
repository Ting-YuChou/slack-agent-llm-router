import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { test } from "node:test";

import { AgentScheduler, DEFAULT_AGENT_CONTAINER_LIMIT, agentContainerLimit } from "../src/agent-scheduler.js";

const run = promisify(execFile);
const schedulerModule = fileURLToPath(new URL("../src/agent-scheduler.js", import.meta.url));

test("the container limit comes from PI_AGENT_CONTAINER_LIMIT and defaults to 4", () => {
  assert.equal(DEFAULT_AGENT_CONTAINER_LIMIT, 4);
  assert.equal(agentContainerLimit({}), 4);
  assert.equal(agentContainerLimit({ PI_AGENT_CONTAINER_LIMIT: "" }), 4);
  assert.equal(agentContainerLimit({ PI_AGENT_CONTAINER_LIMIT: "7" }), 7);
  for (const bad of ["0", "-1", "2.5", "33", "many"]) {
    assert.throws(() => agentContainerLimit({ PI_AGENT_CONTAINER_LIMIT: bad }), /PI_AGENT_CONTAINER_LIMIT/);
  }
  assert.equal(new AgentScheduler(agentContainerLimit({ PI_AGENT_CONTAINER_LIMIT: "7" })).limit, 7);
});

test("the shared scheduler, which health reports, reads the same env var", async () => {
  const script = `import(${JSON.stringify(schedulerModule)}).then(m => process.stdout.write(String(m.agentScheduler.limit)))`;
  const configured = await run(process.execPath, ["-e", script], { env: { ...process.env, PI_AGENT_CONTAINER_LIMIT: "6" } });
  assert.equal(configured.stdout, "6");
  const { PI_AGENT_CONTAINER_LIMIT: _unset, ...env } = process.env;
  const defaulted = await run(process.execPath, ["-e", script], { env });
  assert.equal(defaulted.stdout, "4");
});

test("the scheduler admits at most its limit at once", async () => {
  const scheduler = new AgentScheduler(2);
  const signal = new AbortController().signal;
  const first = await scheduler.acquire("a", signal);
  await scheduler.acquire("b", signal);
  let third = false;
  const pending = scheduler.acquire("c", signal).then((release) => { third = true; return release; });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(third, false);
  first();
  (await pending)();
  assert.equal(third, true);
});
