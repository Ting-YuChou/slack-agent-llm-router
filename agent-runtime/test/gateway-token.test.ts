import assert from "node:assert/strict";
import { test } from "node:test";

import { issueGatewayToken, verifyGatewayToken } from "../src/gateway-token.js";

test("gateway token is short-lived and bound to run and gpt-5.6-luna", () => {
  const token = issueGatewayToken({ runId: "run-1", model: "gpt-5.6-luna", expiresAt: 2_000 }, "secret");
  assert.deepEqual(verifyGatewayToken(token, "secret", 1_000), {
    runId: "run-1",
    model: "gpt-5.6-luna",
    expiresAt: 2_000,
  });
  assert.equal(verifyGatewayToken(token, "secret", 2_001), null);
  assert.equal(verifyGatewayToken(`${token}x`, "secret", 1_000), null);
});

test("gateway refuses tokens for any model except gpt-5.6-luna", () => {
  const token = issueGatewayToken({ runId: "run-1", model: "other", expiresAt: 2_000 }, "secret");
  assert.equal(verifyGatewayToken(token, "secret", 1_000), null);
});
