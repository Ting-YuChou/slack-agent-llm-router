import assert from "node:assert/strict";
import { test } from "node:test";

import { validateAgentModelRequest } from "../src/model-gateway.js";

test("gateway accepts only Luna Responses requests at max reasoning", () => {
  assert.equal(validateAgentModelRequest({
    model: "gpt-5.6-luna",
    reasoning: { effort: "max" },
  }), true);
  assert.equal(validateAgentModelRequest({
    model: "gpt-5.6-luna",
    reasoning: { effort: "high" },
  }), false);
  assert.equal(validateAgentModelRequest({
    model: "gpt-5",
    reasoning: { effort: "max" },
  }), false);
});
