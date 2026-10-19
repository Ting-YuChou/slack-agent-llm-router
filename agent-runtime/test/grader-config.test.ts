import assert from "node:assert/strict";
import { test } from "node:test";

import { listAgentModels } from "../src/agent-model.js";
import { parseGraderConfig } from "../src/grader-config.js";

const all = listAgentModels();
const openaiOnly = all.filter(model => model.provider === "openai");

test("defaults to Sol high with a two-minute limit", () => {
  const config = parseGraderConfig({}, all);
  assert.equal(config.route.modelRef, "openai/gpt-5.6-sol");
  assert.equal(config.route.effort, "high");
  assert.equal(config.route.source, "grader");
  assert.equal(config.timeoutMs, 120_000);
});

test("model, effort and timeout come from the environment", () => {
  const other = all.find(model => model.ref !== "openai/gpt-5.6-sol" && ["low", "medium", "high", "max"].some(e => model.thinkingLevelMap[e] === e))!;
  const effort = (["high", "medium", "low", "max"] as const).find(e => other.thinkingLevelMap[e] === e)!;
  const config = parseGraderConfig({ PI_AGENT_GRADER_MODEL: other.ref, PI_AGENT_GRADER_EFFORT: effort, PI_AGENT_GRADER_TIMEOUT_MS: "300000" }, all);
  assert.equal(config.route.modelRef, other.ref);
  assert.equal(config.route.effort, effort);
  assert.equal(config.timeoutMs, 300_000);
});

test("a grader model whose provider is not configured fails at startup", () => {
  const foreign = all.find(model => model.provider !== "openai");
  if (!foreign) return;
  assert.throws(() => parseGraderConfig({ PI_AGENT_GRADER_MODEL: foreign.ref }, openaiOnly), /PI_AGENT_GRADER_MODEL/);
});

test("unknown models, unsupported efforts and bad timeouts are rejected", () => {
  assert.throws(() => parseGraderConfig({ PI_AGENT_GRADER_MODEL: "openai/not-a-model" }, all), /PI_AGENT_GRADER_MODEL/);
  assert.throws(() => parseGraderConfig({ PI_AGENT_GRADER_EFFORT: "extreme" }, all), /PI_AGENT_GRADER_EFFORT/);
  assert.throws(() => parseGraderConfig({ PI_AGENT_GRADER_TIMEOUT_MS: "5" }, all), /PI_AGENT_GRADER_TIMEOUT_MS/);
  assert.throws(() => parseGraderConfig({ PI_AGENT_GRADER_TIMEOUT_MS: "soon" }, all), /PI_AGENT_GRADER_TIMEOUT_MS/);
});
