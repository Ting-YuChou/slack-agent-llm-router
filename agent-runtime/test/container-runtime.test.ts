import assert from "node:assert/strict";
import { test } from "node:test";

import { buildAgentDockerArgs } from "../src/container-runtime.js";

test("agent container is non-root, resource-limited, read-only, and internal-network only", () => {
  const args = buildAgentDockerArgs({
    name: "pi-session-1",
    image: "pi-agent@sha256:abc",
    network: "pi-model-only",
    worktreePath: "/repo-worktrees/w1",
    gitMetadataPath: "/repo/.git",
    sessionStatePath: "/runtime/s1",
    gatewayUrl: "http://model-gateway:8080/v1",
    gatewayToken: "short-token",
    modelRef: "anthropic/claude-sonnet-4-6",
    extensionPaths: ["/opt/pi/extensions/policy.ts", "/opt/pi/extensions/model-gateway.ts"],
    pluginPaths: ["/opt/pi/plugins/workspace-summary.ts"],
    skillPaths: ["/opt/pi/skills/test-gap/SKILL.md"],
    toolNames: ["read", "write", "edit", "bash", "grep", "find", "ls", "workspace_summary"],
    piSessionId: "11111111-1111-4111-8111-111111111111",
  });
  const rendered = args.join(" ");

  for (const expected of [
    "--read-only", "--user 10001:10001", "--cap-drop ALL",
    "--security-opt no-new-privileges", "--cpus 2", "--memory 2g",
    "--pids-limit 256", "--network pi-model-only",
    "--tmpfs /tmp:rw,noexec,nosuid,size=536870912",
    "--no-extensions", "--no-skills", "--no-prompt-templates",
    "--mode rpc", "--provider anthropic", "--model claude-sonnet-4-6", "--thinking max",
  ]) assert.match(rendered, new RegExp(expected.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  assert.doesNotMatch(rendered, /docker\.sock|OPENAI_API_KEY=[A-Za-z0-9_-]{30,}/);
  assert.match(rendered, /ANTHROPIC_API_KEY=short-token/);
  assert.doesNotMatch(rendered, /OPENAI_API_KEY=short-token|OPENCODE_API_KEY=short-token/);
  assert.match(rendered, /PI_AGENT_SKILL_PATHS_JSON=\["\/opt\/pi\/skills\/test-gap\/SKILL\.md"\]/);
  assert.match(rendered, /type=bind,src=\/repo-worktrees\/w1,dst=\/repo-worktrees\/w1/);
  assert.match(rendered, /\/repo\/\.git.*\/repo\/\.git.*readonly/);
  assert.match(rendered, /\/repo-worktrees\/w1\/\.git.*readonly/);
  assert.match(rendered, /--no-skills.*--skill \/opt\/pi\/skills\/test-gap\/SKILL\.md/);
  assert.match(rendered, /--session-id 11111111-1111-4111-8111-111111111111/);
  assert.doesNotMatch(rendered, /--continue(?:\s|$)/);

  const hostUserArgs = buildAgentDockerArgs({
    name: "pi-session-host-user",
    image: "pi-agent@sha256:abc",
    network: "pi-model-only",
    worktreePath: "/repo-worktrees/w2",
    gitMetadataPath: "/repo/.git",
    sessionStatePath: "/runtime/s2",
    gatewayUrl: "http://model-gateway:8080/v1",
    gatewayToken: "short-token",
    modelRef: "opencode-go/deepseek-v4-pro",
    extensionPaths: [],
    pluginPaths: [],
    skillPaths: [],
    toolNames: ["read"],
    piSessionId: "22222222-2222-4222-8222-222222222222",
    piSessionFile: "2026-08-08T00-00-00-000Z_22222222-2222-4222-8222-222222222222.jsonl",
    user: "501:20",
  });
  assert.match(hostUserArgs.join(" "), /--user 501:20/);
  assert.match(hostUserArgs.join(" "), /OPENCODE_API_KEY=short-token/);
  assert.match(hostUserArgs.join(" "), /--provider opencode-go --model deepseek-v4-pro --thinking max/);
  assert.match(hostUserArgs.join(" "), /--session \/var\/lib\/pi-session\/2026-08-08T00-00-00-000Z_22222222-2222-4222-8222-222222222222\.jsonl/);
  assert.doesNotMatch(hostUserArgs.join(" "), /--session-id|--continue(?:\s|$)/);
});

test("agent container rejects a session basename that can escape its session directory", () => {
  assert.throws(() => buildAgentDockerArgs({
    name: "pi-session-invalid",
    image: "pi-agent@sha256:abc",
    network: "pi-model-only",
    worktreePath: "/repo-worktrees/w3",
    gitMetadataPath: "/repo/.git",
    sessionStatePath: "/runtime/s3",
    gatewayUrl: "http://model-gateway:8080/v1",
    gatewayToken: "short-token",
    modelRef: "openai/gpt-5.6-luna",
    extensionPaths: [],
    pluginPaths: [],
    skillPaths: [],
    toolNames: ["read"],
    piSessionId: "33333333-3333-4333-8333-333333333333",
    piSessionFile: "../outside.jsonl",
  }), /session file/i);
});
