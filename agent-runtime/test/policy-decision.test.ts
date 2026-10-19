import assert from "node:assert/strict";
import { test } from "node:test";

import { parsePolicyDecision, policyDecisionEntry, summarizePolicy, type PolicyDecision } from "../src/policy-decision.js";

test("an entry keeps the command's basename and a hash, never its text", () => {
  const entry = policyDecisionEntry({ tool: "bash", decision: "block", reason: "network_command", command: "/usr/bin/curl -H 'Authorization: Bearer sk-live-1' https://x", latencyMs: 3.7 });
  assert.equal(entry.argv0, "curl");
  assert.match(entry.cmd_hash ?? "", /^[0-9a-f]{16}$/);
  assert.equal(entry.latency_ms, 4);
  assert.doesNotMatch(JSON.stringify(entry), /sk-live|Authorization|https/);
});

test("an entry normalizes free-form reasons into a bounded code", () => {
  assert.equal(policyDecisionEntry({ tool: "write", decision: "block", reason: "Path blocked: Symlink Escape!" }).reason, "path_blocked_symlink_escape");
  assert.equal(policyDecisionEntry({ tool: "bash", decision: "allow", reason: "" }).reason, "unspecified");
  assert.ok(policyDecisionEntry({ tool: "bash", decision: "allow", reason: "x".repeat(200) }).reason.length <= 64);
  assert.equal(policyDecisionEntry({ tool: "bash", decision: "allow", reason: "ok", command: "   " }).argv0, undefined);
});

test("parsing accepts a well-formed entry and drops unknown fields", () => {
  const parsed = parsePolicyDecision({ tool: "bash", decision: "approval", reason: "unlisted_bash", approved: true, argv0: "python", cmd_hash: "0123456789abcdef", latency_ms: 12, command: "python -c 'leak'" });
  assert.deepEqual(parsed, { tool: "bash", decision: "approval", reason: "unlisted_bash", approved: true, argv0: "python", cmd_hash: "0123456789abcdef", latency_ms: 12 });
});

test("parsing rejects forged or malformed entries", () => {
  for (const data of [
    null, "block", {}, { tool: "bash", decision: "maybe", reason: "x" },
    { tool: "Bash Tool", decision: "block", reason: "x" },
    { tool: "bash", decision: "block", reason: "has spaces" },
    { tool: "bash", decision: "block", reason: "r".repeat(65) },
    { tool: "bash", decision: "block", reason: "ok", argv0: "a/b" },
    { tool: "bash", decision: "block", reason: "ok", cmd_hash: "not-hex" },
    { tool: "bash", decision: "block", reason: "ok", latency_ms: -1 },
    { tool: "bash", decision: "allow", reason: "ok", approved: true },
  ]) assert.equal(parsePolicyDecision(data), undefined, JSON.stringify(data));
});

test("the summary counts verdicts and caps distinct reasons", () => {
  let summary: ReturnType<typeof summarizePolicy> | undefined;
  const decisions: PolicyDecision[] = [
    { tool: "bash", decision: "block", reason: "network_command" },
    { tool: "bash", decision: "block", reason: "network_command" },
    { tool: "write", decision: "approval", reason: "write_approval", approved: true },
    { tool: "bash", decision: "approval", reason: "unlisted_bash", approved: false },
    { tool: "bash", decision: "allow", reason: "read_only_command" },
  ];
  for (const decision of decisions) summary = summarizePolicy(summary, decision);
  assert.deepEqual(summary, { allow: 1, block: 2, approval: 2, approved: 1, by_reason: { network_command: 2, write_approval: 1, unlisted_bash: 1, read_only_command: 1 } });
  for (let i = 0; i < 40; i++) summary = summarizePolicy(summary, { tool: "bash", decision: "block", reason: `r${i}` });
  assert.ok(Object.keys(summary!.by_reason).length <= 33);
  assert.ok((summary!.by_reason as Record<string, number>).other! > 0);
});
