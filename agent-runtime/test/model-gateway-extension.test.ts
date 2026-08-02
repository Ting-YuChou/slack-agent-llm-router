import assert from "node:assert/strict";
import { test } from "node:test";

import { AGENT_MODEL_CONFIG } from "../src/agent-model.js";

test("agent model catalog defines Luna with max reasoning support", () => {
  assert.equal(AGENT_MODEL_CONFIG.id, "gpt-5.6-luna");
  assert.equal(AGENT_MODEL_CONFIG.thinkingLevelMap.max, "max");
  assert.equal(AGENT_MODEL_CONFIG.contextWindow, 1_050_000);
});
