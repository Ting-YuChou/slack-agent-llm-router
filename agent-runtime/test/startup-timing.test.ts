import assert from "node:assert/strict";
import { test } from "node:test";

import { StartupTiming } from "../src/startup-timing.js";

test("startup timing summarises each stage of a cold start in milliseconds", () => {
  let now = 1000;
  const timing = new StartupTiming(() => {}, () => now);
  timing.start(); now += 5; timing.cliSpawned();
  now += 795; timing.containerRunning();
  now += 300; timing.rpcReady();
  now += 100; timing.configured(); timing.promptSent();
  assert.deepEqual(timing.summary(), { container_ms: 800, rpc_ready_ms: 300, configure_ms: 100, dispatch_to_prompt_ms: 1200 });
});

test("an unfinished start reports only the stages it completed", () => {
  let now = 0;
  const timing = new StartupTiming(() => {}, () => now);
  timing.start(); now += 50; timing.containerRunning();
  assert.deepEqual(timing.summary(), { container_ms: 50 });
});
