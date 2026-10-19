import assert from "node:assert/strict";
import { test } from "node:test";

import { buildAgentDockerArgs, buildAgentSpawnEnv } from "../src/container-runtime.js";

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
  assert.match(rendered, /--env ANTHROPIC_API_KEY(?: |$)/);
  assert.doesNotMatch(rendered, /short-token/);
  assert.match(rendered, /PI_AGENT_SKILL_PATHS_JSON=\["\/opt\/pi\/skills\/test-gap\/SKILL\.md"\]/);
  assert.match(rendered, /type=bind,src=\/repo-worktrees\/w1,dst=\/repo-worktrees\/w1/);
  assert.match(rendered, /\/repo\/\.git.*\/repo\/\.git.*readonly/);
  assert.match(rendered, /\/repo-worktrees\/w1\/\.git.*readonly/);
  assert.match(rendered, /--no-skills.*--skill \/opt\/pi\/skills\/test-gap\/SKILL\.md/);

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
    user: "501:20",
  });
  assert.match(hostUserArgs.join(" "), /--user 501:20/);
  assert.match(hostUserArgs.join(" "), /--env OPENCODE_API_KEY(?: |$)/);
  assert.match(hostUserArgs.join(" "), /--provider opencode-go --model deepseek-v4-pro --thinking max/);
});

test("selected Sol run starts Pi with Sol high while retaining the internal gateway", () => {
  const args = buildAgentDockerArgs({
    name: "pi-sol", image: "pi-agent:1.0.1", network: "pi-model-only",
    worktreePath: "/worktree", gitMetadataPath: "/repo/.git", sessionStatePath: "/state",
    gatewayUrl: "http://model-gateway:8080", gatewayToken: "run-token",
    modelRef: "openai/gpt-5.6-sol", reasoningEffort: "high",
    extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"],
  });
  assert.match(args.join(" "), /--provider openai --model gpt-5\.6-sol --thinking high/);
  assert.match(args.join(" "), /--env OPENAI_API_KEY(?: |$)/);
});

test("MCP is absent when off and explicitly loads only the trusted builtin path when enabled", () => {
  const base = {
    name: "pi-mcp", image: "pi-agent:1.0.1", network: "pi-model-only",
    worktreePath: "/worktree", gitMetadataPath: "/repo/.git", sessionStatePath: "/state",
    gatewayUrl: "http://model-gateway:8080", gatewayToken: "model-token",
    modelRef: "openai/gpt-5.6-luna", extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"],
  };
  const off = buildAgentDockerArgs(base);
  assert.doesNotMatch(off.join(" "), /builtin:mcp|mcp-bootstrap|PI_AGENT_MCP/);
  assert.match(off.join(" "), /--no-approve/);
  assert.doesNotMatch(off.join(" "), /--approve(?:\s|$)/);

  const enabled = buildAgentDockerArgs({
    ...base,
    mcp: {
      mode: "github_read_only",
      servers: [{ server: "github", gatewayUrl: "http://mcp-github:8090/mcp", token: "run-bound-mcp-token" }],
    },
  });
  const rendered = enabled.join(" ");
  assert.match(rendered, /PI_AGENT_MCP_MODE=github_read_only/);
  assert.match(rendered, /--env PI_AGENT_MCP_SERVERS_JSON(?: |$)/);
  assert.doesNotMatch(rendered, /run-bound-mcp-token/);
  assert.match(rendered, /--no-extensions.*-e builtin:mcp.*-e \/opt\/pi\/extensions\/mcp-bootstrap\.ts/);
  assert.match(rendered, /--tools read,codemode,mcp__github__get_file_contents/);
  assert.equal((rendered.match(/builtin:codemode/g) ?? []).length, 1);
});

test("run-time Jev exposes codemode with only a short-lived classifier gateway token", () => {
  const base = {
    name: "pi-jev", image: "pi-agent:1.0.1", network: "pi-model-only",
    worktreePath: "/worktree", gitMetadataPath: "/repo/.git", sessionStatePath: "/state",
    gatewayUrl: "http://model-gateway:8080", gatewayToken: "chat-run-token",
    modelRef: "openai/gpt-5.6-luna", extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"],
  };
  const off = buildAgentDockerArgs(base);
  assert.doesNotMatch(off.join(" "), /builtin:codemode|OPENROUTER_API_KEY|PI_AGENT_JEV_CLASSIFIER_MODE/);

  const enabled = buildAgentDockerArgs({
    ...base,
    classifierGatewayToken: "classifier-run-token",
  });
  const rendered = enabled.join(" ");
  assert.match(rendered, /--env OPENROUTER_API_KEY(?: |$)/);
  assert.doesNotMatch(rendered, /classifier-run-token/);
  assert.match(rendered, /PI_AGENT_JEV_CLASSIFIER_MODE=on/);
  assert.match(rendered, /--no-extensions.*-e builtin:codemode/);
  assert.match(rendered, /--tools read,codemode/);
  assert.doesNotMatch(rendered, /OPENROUTER_API_KEY=sk-or-/);
});


test('discussion mode overrides writer capabilities and mounts only read-only workspace', () => {
  const args = buildAgentDockerArgs({ taskMode: 'discuss', name: 'discuss', image: 'pi', network: 'models',
    worktreePath: '/worktree', gitMetadataPath: '/host/.git', sessionStatePath: '/state', gatewayUrl: 'http://models', gatewayToken: 'short', modelRef: 'openai/gpt-5.6-luna',
    extensionPaths: ['/untrusted/extension.ts'], pluginPaths: ['/writer.ts'], skillPaths: ['/skill.md'], toolNames: ['bash','write','edit','codemode'], classifierGatewayToken: 'classifier-secret',
    mcp: { mode: 'read_only', servers: [{ server: 'github', gatewayUrl: 'http://github', token: 'mcp-secret' }] },
  } as any);
  assert.ok(args.includes('type=bind,src=/worktree,dst=/worktree,readonly'));
  assert.ok(!args.some(a => a.includes('/host/.git')));
  assert.equal(args[args.indexOf('--tools')+1],'read,grep,find,ls');
  assert.ok(args.includes('--no-context-files'));
  assert.ok(args.includes('/opt/pi/extensions/discussion.ts'));
  assert.doesNotMatch(args.join(' '), /untrusted|writer.ts|skill.md|classifier-secret|mcp-secret|builtin:mcp|builtin:codemode/);
});

test("writer turn and tool-call limits come from host configuration", () => {
  const base = {
    name: "pi-session-1", image: "pi-agent@sha256:abc", network: "pi-model-only", worktreePath: "/w", gitMetadataPath: "/repo/.git",
    sessionStatePath: "/s", gatewayUrl: "http://model-gateway:8080/v1", gatewayToken: "t", modelRef: "openai/gpt-5.6-luna",
    extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"],
  };
  assert.match(buildAgentDockerArgs(base).join(" "), /PI_AGENT_MAX_TURNS=20 .*PI_AGENT_MAX_TOOL_CALLS=40/);
  const rendered = buildAgentDockerArgs({ ...base, limits: { maxTurns: 3, maxToolCalls: 5 } }).join(" ");
  assert.match(rendered, /PI_AGENT_MAX_TURNS=3 /);
  assert.match(rendered, /PI_AGENT_MAX_TOOL_CALLS=5 /);
});

test("writer receives trusted-profile constraints; discussion mode drops them", () => {
  const base = {
    name: "pi-session-1", image: "pi-agent@sha256:abc", network: "pi-model-only", worktreePath: "/w", gitMetadataPath: "/repo/.git",
    sessionStatePath: "/s", gatewayUrl: "http://model-gateway:8080/v1", gatewayToken: "t", modelRef: "openai/gpt-5.6-luna",
    extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"],
  };
  const writerContext = { allowedPaths: ["src"], checks: ["pytest"], rubric: ["tests_cover_change"], diffLimits: { maxFiles: 50, maxBytes: 1048576, maxSingleFileBytes: 262144 } };
  const args = buildAgentDockerArgs({ ...base, writerContext });
  const env = args[args.findIndex((arg) => arg.startsWith("PI_AGENT_WRITER_JSON="))];
  assert.deepEqual(JSON.parse(env.slice("PI_AGENT_WRITER_JSON=".length)), writerContext);
  assert.doesNotMatch(buildAgentDockerArgs({ ...base, writerContext, taskMode: "discuss" }).join(" "), /PI_AGENT_WRITER_JSON/);
  assert.doesNotMatch(buildAgentDockerArgs(base).join(" "), /PI_AGENT_WRITER_JSON/);
});

test("credentials travel in the docker client's environment, never on its argv", () => {
  const options = {
    name: "pi-session-1", image: "pi-agent@sha256:abc", network: "pi-model-only", worktreePath: "/w", gitMetadataPath: "/repo/.git",
    sessionStatePath: "/s", gatewayUrl: "http://model-gateway:8080", gatewayToken: "gateway-secret-token", modelRef: "openai/gpt-5.6-luna",
    classifierGatewayToken: "classifier-secret-token",
    mcp: { mode: "read_only" as const, servers: [{ server: "github" as const, gatewayUrl: "http://mcp-github:8090/mcp", token: "mcp-secret-token" }] },
    extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read", "bash"],
  };
  const rendered = buildAgentDockerArgs(options).join(" ");
  assert.doesNotMatch(rendered, /secret-token/);
  const env = buildAgentSpawnEnv(options, { PATH: "/usr/bin", OPENAI_API_KEY: "host-real-key" });
  assert.equal(env.PATH, "/usr/bin");
  assert.equal(env.OPENAI_API_KEY, "gateway-secret-token");
  assert.equal(env.OPENROUTER_API_KEY, "classifier-secret-token");
  assert.match(env.PI_AGENT_MCP_SERVERS_JSON ?? "", /mcp-secret-token/);
  assert.deepEqual(Object.keys(env).sort(), ["OPENAI_API_KEY", "OPENROUTER_API_KEY", "PATH", "PI_AGENT_MCP_SERVERS_JSON"]);
  const discuss = buildAgentSpawnEnv({ ...options, taskMode: "discuss" }, { PATH: "/usr/bin" });
  assert.deepEqual(Object.keys(discuss).sort(), ["OPENAI_API_KEY", "PATH"]);
});

test("the write-approval mode reaches the writer only when the host sets it", () => {
  const base = {
    name: "pi-session-1", image: "pi-agent@sha256:abc", network: "pi-model-only", worktreePath: "/w", gitMetadataPath: "/repo/.git",
    sessionStatePath: "/s", gatewayUrl: "http://model-gateway:8080/v1", gatewayToken: "t", modelRef: "openai/gpt-5.6-luna",
    extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"],
  };
  assert.ok(buildAgentDockerArgs({ ...base, writeApproval: "none" }).includes("PI_AGENT_WRITE_APPROVAL=none"));
  assert.ok(buildAgentDockerArgs({ ...base, writeApproval: "per_prompt" }).includes("PI_AGENT_WRITE_APPROVAL=per_prompt"));
  assert.doesNotMatch(buildAgentDockerArgs(base).join(" "), /PI_AGENT_WRITE_APPROVAL/);
});
