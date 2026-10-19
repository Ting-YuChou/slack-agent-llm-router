import assert from "node:assert/strict";
import { test } from "node:test";

import { DEFAULT_ROUTE_LIMITS, limitsForRoute, parseAgentLimitConfig, parseAgentLimits, parseRouteLimits } from "../src/agent-limits.js";
import type { RouteDecision } from "../src/jev-router.js";

const jev = (label: string): RouteDecision => ({ modelRef: "openai/gpt-5.6-luna", effort: "low", source: "jev", label });

test("base limits default to 20 turns and 40 tool calls and validate overrides", () => {
  assert.deepEqual(parseAgentLimits({}), { maxTurns: 20, maxToolCalls: 40 });
  assert.deepEqual(parseAgentLimits({ PI_AGENT_MAX_TURNS: "3", PI_AGENT_MAX_TOOL_CALLS: "7" }), { maxTurns: 3, maxToolCalls: 7 });
  assert.throws(() => parseAgentLimits({ PI_AGENT_MAX_TURNS: "0" }), /PI_AGENT_MAX_TURNS/);
  assert.throws(() => parseAgentLimits({ PI_AGENT_MAX_TOOL_CALLS: "1001" }), /PI_AGENT_MAX_TOOL_CALLS/);
});

test("route defaults scale the budget with the Jev task label", () => {
  assert.deepEqual(DEFAULT_ROUTE_LIMITS, {
    read_only: { maxTurns: 12, maxToolCalls: 24 },
    small_patch: { maxTurns: 20, maxToolCalls: 40 },
    complex: { maxTurns: 30, maxToolCalls: 60 },
  });
  const config = parseAgentLimitConfig({});
  assert.deepEqual(limitsForRoute(jev("read_only"), config), { maxTurns: 12, maxToolCalls: 24 });
  assert.deepEqual(limitsForRoute(jev("small_patch"), config), { maxTurns: 20, maxToolCalls: 40 });
  assert.deepEqual(limitsForRoute(jev("complex"), config), { maxTurns: 30, maxToolCalls: 60 });
});

test("fallback, shadow, explicit and unlabelled routes keep the base limits", () => {
  const config = parseAgentLimitConfig({ PI_AGENT_MAX_TURNS: "5", PI_AGENT_MAX_TOOL_CALLS: "9" });
  const base = { maxTurns: 5, maxToolCalls: 9 };
  assert.deepEqual(limitsForRoute({ ...jev("complex"), source: "fallback" }, config), base);
  assert.deepEqual(limitsForRoute({ ...jev("complex"), source: "shadow" }, config), base);
  assert.deepEqual(limitsForRoute({ modelRef: "openai/gpt-5.6-sol", effort: "max", source: "explicit" }, config), base);
  assert.deepEqual(limitsForRoute({ ...jev("unclear") }, config), base);
  assert.deepEqual(limitsForRoute({ modelRef: "openai/gpt-5.6-luna", effort: "max", source: "jev" }, config), base);
  assert.deepEqual(limitsForRoute(undefined, config), base);
});

test("PI_AGENT_ROUTE_LIMITS_JSON overrides individual routes and fields", () => {
  const routes = parseRouteLimits({ PI_AGENT_ROUTE_LIMITS_JSON: JSON.stringify({ complex: { maxToolCalls: 80 }, read_only: { maxTurns: 6, maxToolCalls: 10 } }) });
  assert.deepEqual(routes.complex, { maxTurns: 30, maxToolCalls: 80 });
  assert.deepEqual(routes.read_only, { maxTurns: 6, maxToolCalls: 10 });
  assert.deepEqual(routes.small_patch, DEFAULT_ROUTE_LIMITS.small_patch);
  assert.deepEqual(parseRouteLimits({ PI_AGENT_ROUTE_LIMITS_JSON: "" }), DEFAULT_ROUTE_LIMITS);
});

test("invalid PI_AGENT_ROUTE_LIMITS_JSON fails startup with the variable name", () => {
  for (const value of [
    "not json",
    "[]",
    JSON.stringify({ unclear: { maxTurns: 5 } }),
    JSON.stringify({ complex: 5 }),
    JSON.stringify({ complex: { maxTurns: 0 } }),
    JSON.stringify({ complex: { maxToolCalls: 1.5 } }),
    JSON.stringify({ complex: { maxToolCalls: "60" } }),
    JSON.stringify({ complex: { maxTokens: 60 } }),
  ]) assert.throws(() => parseRouteLimits({ PI_AGENT_ROUTE_LIMITS_JSON: value }), /PI_AGENT_ROUTE_LIMITS_JSON/, value);
});
