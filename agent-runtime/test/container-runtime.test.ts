import assert from "node:assert/strict";
import { test } from "node:test";

import { execFileSync } from "node:child_process";
import { chmod, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { parseAgentLimitConfig } from "../src/agent-limits.js";
import { buildAgentDockerArgs, buildAgentSpawnEnv, DockerPiProcess } from "../src/container-runtime.js";
import { CREDENTIAL_PLACEHOLDER } from "../src/run-credentials.js";
import { WorktreeManager, writerGitDir } from "../src/worktree-manager.js";
import { FAKE_DOCKER } from "./support/fake-docker.js";

test("agent container is non-root, resource-limited, read-only, and internal-network only", () => {
  const args = buildAgentDockerArgs({
    name: "pi-session-1",
    image: "pi-agent@sha256:abc",
    network: "pi-model-only",
    worktreePath: "/repo-worktrees/w1",
    gitMetadataPath: "/repo/.git",
    sessionStatePath: "/runtime/s1",
    gatewayUrl: "http://model-gateway:8080/v1",
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
  assert.match(rendered, new RegExp(`--env ANTHROPIC_API_KEY=${CREDENTIAL_PLACEHOLDER} `));
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
    modelRef: "opencode-go/deepseek-v4-pro",
    extensionPaths: [],
    pluginPaths: [],
    skillPaths: [],
    toolNames: ["read"],
    user: "501:20",
  });
  assert.match(hostUserArgs.join(" "), /--user 501:20/);
  assert.match(hostUserArgs.join(" "), new RegExp(`--env OPENCODE_API_KEY=${CREDENTIAL_PLACEHOLDER} `));
  assert.match(hostUserArgs.join(" "), /--provider opencode-go --model deepseek-v4-pro --thinking max/);
});

test("selected Sol run starts Pi with Sol high while retaining the internal gateway", () => {
  const args = buildAgentDockerArgs({
    name: "pi-sol", image: "pi-agent:1.0.1", network: "pi-model-only",
    worktreePath: "/worktree", gitMetadataPath: "/repo/.git", sessionStatePath: "/state",
    gatewayUrl: "http://model-gateway:8080",
    modelRef: "openai/gpt-5.6-sol", reasoningEffort: "high",
    extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"],
  });
  assert.match(args.join(" "), /--provider openai --model gpt-5\.6-sol --thinking high/);
  assert.match(args.join(" "), new RegExp(`--env OPENAI_API_KEY=${CREDENTIAL_PLACEHOLDER} `));
});

test("MCP is absent when off and explicitly loads only the trusted builtin path when enabled", () => {
  const base = {
    name: "pi-mcp", image: "pi-agent:1.0.1", network: "pi-model-only",
    worktreePath: "/worktree", gitMetadataPath: "/repo/.git", sessionStatePath: "/state",
    gatewayUrl: "http://model-gateway:8080",
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
    gatewayUrl: "http://model-gateway:8080",
    modelRef: "openai/gpt-5.6-luna", extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"],
  };
  const off = buildAgentDockerArgs(base);
  assert.doesNotMatch(off.join(" "), /builtin:codemode|OPENROUTER_API_KEY|PI_AGENT_JEV_CLASSIFIER_MODE/);

  const enabled = buildAgentDockerArgs({
    ...base,
    classifier: true,
  });
  const rendered = enabled.join(" ");
  assert.match(rendered, new RegExp(`--env OPENROUTER_API_KEY=${CREDENTIAL_PLACEHOLDER} `));
  assert.doesNotMatch(rendered, /classifier-run-token/);
  assert.match(rendered, /PI_AGENT_JEV_CLASSIFIER_MODE=on/);
  assert.match(rendered, /--no-extensions.*-e builtin:codemode/);
  assert.match(rendered, /--tools read,codemode/);
  assert.doesNotMatch(rendered, /OPENROUTER_API_KEY=sk-or-/);
});


test('discussion mode overrides writer capabilities and mounts only read-only workspace', () => {
  const args = buildAgentDockerArgs({ taskMode: 'discuss', name: 'discuss', image: 'pi', network: 'models',
    worktreePath: '/worktree', gitMetadataPath: '/host/.git', sessionStatePath: '/state', gatewayUrl: 'http://models', modelRef: 'openai/gpt-5.6-luna',
    extensionPaths: ['/untrusted/extension.ts'], pluginPaths: ['/writer.ts'], skillPaths: ['/skill.md'], toolNames: ['bash','write','edit','codemode'], classifier: true,
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
    sessionStatePath: "/s", gatewayUrl: "http://model-gateway:8080/v1", modelRef: "openai/gpt-5.6-luna",
    extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"],
  };
  assert.match(buildAgentDockerArgs(base).join(" "), /PI_AGENT_MAX_TURNS=20 .*PI_AGENT_MAX_TOOL_CALLS=40/);
  const rendered = buildAgentDockerArgs({ ...base, limits: { maxTurns: 3, maxToolCalls: 5 } }).join(" ");
  assert.match(rendered, /PI_AGENT_MAX_TURNS=3 /);
  assert.match(rendered, /PI_AGENT_MAX_TOOL_CALLS=5 /);
});

test("a complex Jev route starts the writer container with the complex route limits", async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "fake-docker-"));
  const argsFile = path.join(dir, "run-args");
  await writeFile(path.join(dir, "docker"), `#!/bin/sh\nif [ "$1" = run ]; then echo "$@" > "${argsFile}"; exit 0; fi\nexit 1\n`);
  await chmod(path.join(dir, "docker"), 0o755);
  const previousPath = process.env.PATH;
  process.env.PATH = `${dir}:${previousPath}`;
  const routeLimits = parseAgentLimitConfig({});
  const piProcess = new DockerPiProcess({
    name: "pi-session-limits", image: "pi-agent@sha256:abc", network: "pi-model-only", worktreePath: "/w", gitMetadataPath: "/repo/.git",
    sessionStatePath: dir, gatewayUrl: "http://model-gateway:8080/v1", modelRef: "openai/gpt-5.6-sol",
    extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"], limits: routeLimits.base, routeLimits,
  }, () => "token", () => undefined);
  const route = { modelRef: "openai/gpt-5.6-sol", effort: "high", source: "jev", label: "complex" } as const;
  try {
    assert.deepEqual(piProcess.limitsFor(route), { maxTurns: 30, maxToolCalls: 60 });
    piProcess.prompt("run-limits-1", "refactor", route);
    let rendered = "";
    for (let i = 0; i < 200 && !rendered; i++) {
      rendered = await readFile(argsFile, "utf8").catch(() => "");
      if (!rendered) await new Promise((resolve) => setTimeout(resolve, 10));
    }
    assert.match(rendered, /PI_AGENT_MAX_TURNS=30 /);
    assert.match(rendered, /PI_AGENT_MAX_TOOL_CALLS=60 /);
  } finally {
    await piProcess.close();
    process.env.PATH = previousPath;
    await rm(dir, { recursive: true, force: true });
  }
});

test("writer receives trusted-profile constraints; discussion mode drops them", () => {
  const base = {
    name: "pi-session-1", image: "pi-agent@sha256:abc", network: "pi-model-only", worktreePath: "/w", gitMetadataPath: "/repo/.git",
    sessionStatePath: "/s", gatewayUrl: "http://model-gateway:8080/v1", modelRef: "openai/gpt-5.6-luna",
    extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"],
  };
  const writerContext = { allowedPaths: ["src"], checks: ["pytest"], rubric: ["tests_cover_change"], diffLimits: { maxFiles: 50, maxBytes: 1048576, maxSingleFileBytes: 262144 } };
  const args = buildAgentDockerArgs({ ...base, writerContext });
  const env = args[args.findIndex((arg) => arg.startsWith("PI_AGENT_WRITER_JSON="))];
  assert.deepEqual(JSON.parse(env.slice("PI_AGENT_WRITER_JSON=".length)), writerContext);
  assert.doesNotMatch(buildAgentDockerArgs({ ...base, writerContext, taskMode: "discuss" }).join(" "), /PI_AGENT_WRITER_JSON/);
  assert.doesNotMatch(buildAgentDockerArgs(base).join(" "), /PI_AGENT_WRITER_JSON/);
});

test("gateway credentials are neither on argv nor in the docker client's environment; MCP config travels in its environment", () => {
  const options = {
    name: "pi-session-1", image: "pi-agent@sha256:abc", network: "pi-model-only", worktreePath: "/w", gitMetadataPath: "/repo/.git",
    sessionStatePath: "/s", gatewayUrl: "http://model-gateway:8080", modelRef: "openai/gpt-5.6-luna", classifier: true,
    mcp: { mode: "read_only" as const, servers: [{ server: "github" as const, gatewayUrl: "http://mcp-github:8090/mcp", token: "mcp-secret-token" }] },
    extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read", "bash"],
  };
  const rendered = buildAgentDockerArgs(options).join(" ");
  assert.doesNotMatch(rendered, /secret-token/);
  assert.match(rendered, new RegExp(`--env OPENAI_API_KEY=${CREDENTIAL_PLACEHOLDER} `));
  assert.match(rendered, new RegExp(`--env OPENROUTER_API_KEY=${CREDENTIAL_PLACEHOLDER} `));
  const env = buildAgentSpawnEnv(options, { PATH: "/usr/bin", OPENAI_API_KEY: "host-real-key" });
  assert.equal(env.PATH, "/usr/bin");
  assert.match(env.PI_AGENT_MCP_SERVERS_JSON ?? "", /mcp-secret-token/);
  assert.deepEqual(Object.keys(env).sort(), ["PATH", "PI_AGENT_MCP_SERVERS_JSON"]);
  const discuss = buildAgentSpawnEnv({ ...options, taskMode: "discuss" }, { PATH: "/usr/bin" });
  assert.deepEqual(Object.keys(discuss).sort(), ["PATH"]);
});

test("the write-approval mode reaches the writer only when the host sets it", () => {
  const base = {
    name: "pi-session-1", image: "pi-agent@sha256:abc", network: "pi-model-only", worktreePath: "/w", gitMetadataPath: "/repo/.git",
    sessionStatePath: "/s", gatewayUrl: "http://model-gateway:8080/v1", modelRef: "openai/gpt-5.6-luna",
    extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"],
  };
  assert.ok(buildAgentDockerArgs({ ...base, writeApproval: "none" }).includes("PI_AGENT_WRITE_APPROVAL=none"));
  assert.ok(buildAgentDockerArgs({ ...base, writeApproval: "per_prompt" }).includes("PI_AGENT_WRITE_APPROVAL=per_prompt"));
  assert.doesNotMatch(buildAgentDockerArgs(base).join(" "), /PI_AGENT_WRITE_APPROVAL/);
});

test("the writer mounts its session's git mirror read-only and never the host git-common-dir", async () => {
  const repo = await mkdtemp(path.join(os.tmpdir(), "pi-mount-repo-"));
  const worktreeRoot = await mkdtemp(path.join(os.tmpdir(), "pi-mount-worktrees-"));
  const git = (...args: string[]) => execFileSync("git", args, { cwd: repo, encoding: "utf8" }).trim();
  git("init", "-q", "-b", "main");
  await writeFile(path.join(repo, "README.md"), "base\n");
  git("add", "README.md");
  git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-q", "-m", "base");
  const shared = path.resolve(repo, git("rev-parse", "--git-common-dir"));
  const created = await new WorktreeManager({ repoPath: repo, worktreeRoot, baseRef: "HEAD" }).create("mounts");
  const options = {
    name: "pi-session-mirror", image: "pi-agent@sha256:abc", network: "pi-model-only", sessionStatePath: "/runtime/s",
    gatewayUrl: "http://model-gateway:8080/v1", modelRef: "openai/gpt-5.6-luna", extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"],
  };
  const mounts = (args: string[]) => args.flatMap((arg, index) => args[index - 1] === "--mount" ? [arg] : []);

  const writer = mounts(buildAgentDockerArgs({ ...options, worktreePath: created.path, gitMetadataPath: writerGitDir(created, shared) }));
  assert.ok(writer.includes(`type=bind,src=${created.gitDir},dst=${created.gitDir},readonly`), writer.join("\n"));
  assert.ok(writer.includes(`type=bind,src=${created.path}/.git,dst=${created.path}/.git,readonly`), writer.join("\n"));
  const realShared = await realpath(shared);
  assert.ok(!writer.some((mount) => mount.includes(`src=${shared}`) || mount.includes(`src=${realShared}`)), writer.join("\n"));

  // A session restored from before mirrors keeps the shared metadata its worktree points at.
  const legacy = mounts(buildAgentDockerArgs({ ...options, worktreePath: "/legacy/w", gitMetadataPath: writerGitDir({}, shared) }));
  assert.ok(legacy.includes(`type=bind,src=${shared},dst=${shared},readonly`), legacy.join("\n"));
  await rm(repo, { recursive: true, force: true });
  await rm(worktreeRoot, { recursive: true, force: true });
});

test("closing a process runs its cleanup hook once, and a failing hook does not fail the close", async () => {
  let runs = 0;
  const piProcess = new DockerPiProcess({
    name: "pi-close-hook", image: "pi-agent@sha256:abc", network: "pi-run-x", worktreePath: "/w", gitMetadataPath: "/repo/.git",
    sessionStatePath: "/s", gatewayUrl: "http://model-gateway:8080/v1", modelRef: "openai/gpt-5.6-luna",
    extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"],
    onClosed: async () => { runs++; throw new Error("network still in use"); },
  }, () => "token", () => {});
  await piProcess.close();
  await piProcess.close();
  assert.equal(runs, 1);
});

test("aborting a run keeps the session's network: the next prompt in the session still starts on it", { timeout: 30_000 }, async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "fake-docker-abort-"));
  await writeFile(path.join(dir, "docker"), FAKE_DOCKER);
  await chmod(path.join(dir, "docker"), 0o755);
  const previousPath = process.env.PATH;
  process.env.PATH = `${dir}:${previousPath}`;
  let closedHooks = 0;
  let settle: () => void = () => undefined;
  const piProcess = new DockerPiProcess({
    name: "pi-abort-network", image: "pi-agent@sha256:abc", network: "pi-run-abort", worktreePath: "/w", gitMetadataPath: "/repo/.git",
    sessionStatePath: dir, gatewayUrl: "http://model-gateway:8080", modelRef: "openai/gpt-5.6-luna",
    extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"],
    onClosed: async () => { closedHooks++; },
  }, (runId) => `token-${runId}`, (event) => { if (event.type === "settled") settle(); });
  const settled = () => new Promise<void>((resolve) => { settle = resolve; });
  try {
    let done = settled(); piProcess.prompt("exec-1", "fix"); await done;
    // A failed, cancelled or rejected run is aborted; its session stays open for the owner's next message.
    await piProcess.abort("exec-1");
    assert.equal(closedHooks, 0, "abort must not tear down what the session still needs");
    done = settled(); piProcess.prompt("exec-2", "try again"); await done;
    const runs = (await readFile(path.join(dir, "run-args"), "utf8")).trim().split("\n");
    assert.equal(runs.length, 2);
    for (const args of runs) assert.match(args, /--network pi-run-abort /);
    await piProcess.close();
    assert.equal(closedHooks, 1, "closing the session removes it");
  } finally {
    await piProcess.close();
    process.env.PATH = previousPath;
    await rm(dir, { recursive: true, force: true });
  }
});
