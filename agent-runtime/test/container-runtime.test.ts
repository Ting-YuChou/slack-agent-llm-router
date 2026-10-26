import assert from "node:assert/strict";
import { test } from "node:test";

import { execFileSync } from "node:child_process";
import { chmod, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";

import { parseAgentLimitConfig } from "../src/agent-limits.js";
import { buildAgentDockerArgs, buildAgentSpawnEnv, DockerPiProcess } from "../src/container-runtime.js";
import type { PublicRunEvent } from "../src/pi-rpc.js";
import { STDERR_TAIL_BYTES } from "../src/stderr-tail.js";
import { AgentTracing } from "../src/tracing.js";
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

/**
 * FAKE_DOCKER's Pi, writing to stderr as it goes; the `mode` file next to it picks how it ends:
 * `configure` dies on set_model, `crash` dies holding its credentials, `flood` writes 10 MiB to stderr
 * and dies, and `clean` warns on stderr but settles normally.
 */
const STDERR_DOCKER = `#!/usr/bin/env node
const fs = require("fs"), path = require("path");
const dir = __dirname, [cmd] = process.argv.slice(2);
const mode = fs.readFileSync(path.join(dir, "mode"), "utf8").trim();
if (cmd === "run") {
  fs.writeFileSync(path.join(dir, "run.pid"), String(process.pid));
  let model = {}, level = "", buf = "";
  const out = (record) => process.stdout.write(JSON.stringify(record) + "\\n");
  const die = (text, code) => process.stderr.write(text, () => process.exit(code));
  process.stderr.write("pi: loading extensions\\n");
  process.stdin.on("data", (chunk) => {
    buf += chunk; let i;
    while ((i = buf.indexOf("\\n")) >= 0) {
      const r = JSON.parse(buf.slice(0, i)); buf = buf.slice(i + 1);
      if (r.type === "set_model" && mode === "configure") return die("Error: model registry is unreadable\\n", 2);
      if (r.type === "set_model") { model = { provider: r.provider, id: r.modelId }; out({ id: r.id, type: "response", command: r.type, success: true }); }
      else if (r.type === "set_thinking_level") { level = r.level; out({ id: r.id, type: "response", command: r.type, success: true }); }
      else if (r.type === "get_state") out({ id: r.id, type: "response", command: r.type, success: true, data: { model, thinkingLevel: level } });
      else if (r.type === "prompt") { out({ id: r.id, type: "response", command: "prompt", success: true }); out({ type: "extension_ui_request", id: "cred-" + r.id, method: "input", title: "pi-agent/credential", placeholder: "" }); }
      else if (r.type === "extension_ui_response") {
        const key = JSON.parse(r.value).OPENAI_API_KEY, mcp = JSON.parse(process.env.PI_AGENT_MCP_SERVERS_JSON)[0].token;
        if (mode === "crash") return die("provider refused " + key + "\\nmcp gateway refused " + mcp + "\\nError: worker crashed\\n    at run (/opt/pi/dist/main.js:1:1)\\n", 3);
        if (mode === "flood") return die("FIRST LINE\\n" + ("x".repeat(1023) + "\\n").repeat(10 * 1024) + "LAST LINE\\n", 1);
        process.stderr.write("warning: provider is slow\\n");
        out({ type: "message_end", message: { role: "assistant", content: [{ type: "text", text: "done" }], stopReason: "stop" } });
        out({ type: "agent_settled" });
      }
    }
  });
  process.stdin.on("end", () => process.exit(0));
  process.on("SIGTERM", () => process.exit(0));
} else if (cmd === "inspect") process.stdout.write("true\\n");
else if (cmd === "stop") { try { process.kill(Number(fs.readFileSync(path.join(dir, "run.pid"), "utf8")), "SIGTERM"); } catch {} }
`;

/** A Pi process on STDERR_DOCKER, given a model token and an MCP token it must never let out. */
async function withStderrPi(mode: string, body: (pi: {
  process: DockerPiProcess; events: PublicRunEvent[]; tracing: AgentTracing; exporter: InMemorySpanExporter; dir: string;
  next(match: (event: PublicRunEvent) => boolean): Promise<PublicRunEvent>;
}) => Promise<void>): Promise<void> {
  const dir = await mkdtemp(path.join(os.tmpdir(), "fake-docker-stderr-"));
  await writeFile(path.join(dir, "docker"), STDERR_DOCKER);
  await chmod(path.join(dir, "docker"), 0o755);
  await writeFile(path.join(dir, "mode"), mode);
  const previousPath = process.env.PATH;
  process.env.PATH = `${dir}:${previousPath}`;
  const exporter = new InMemorySpanExporter();
  const tracing = new AgentTracing([new SimpleSpanProcessor(exporter)]);
  const events: PublicRunEvent[] = [];
  const waiters: Array<{ match: (event: PublicRunEvent) => boolean; resolve: (event: PublicRunEvent) => void }> = [];
  const piProcess = new DockerPiProcess({
    name: "pi-stderr", image: "pi-agent@sha256:abc", network: "pi-run-stderr", worktreePath: "/w", gitMetadataPath: "/repo/.git",
    sessionStatePath: dir, gatewayUrl: "http://model-gateway:8080", modelRef: "openai/gpt-5.6-luna",
    extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"],
    mcp: { mode: "github_read_only", servers: [{ server: "github", gatewayUrl: "http://mcp-github:8090/mcp", token: "" }] },
  }, (runId) => `gw-l1-secret-${runId}`, (event) => {
    events.push(event);
    for (const waiter of waiters.splice(0)) if (waiter.match(event)) waiter.resolve(event); else waiters.push(waiter);
  }, (runId) => ({ github: `mcp-l1-secret-${runId}` }), undefined, undefined, "pi-stderr", tracing);
  const next = (match: (event: PublicRunEvent) => boolean) => new Promise<PublicRunEvent>((resolve) => {
    const seen = events.find(match);
    if (seen) resolve(seen); else waiters.push({ match, resolve });
  });
  try {
    await body({ process: piProcess, events, tracing, exporter, dir, next });
  } finally {
    await piProcess.close();
    await tracing.close();
    process.env.PATH = previousPath;
    await rm(dir, { recursive: true, force: true });
  }
}

const runSpan = (exporter: InMemorySpanExporter, runId: string) =>
  exporter.getFinishedSpans().find((span) => span.name === "agent.run" && span.attributes["agent.run_id"] === runId)!;

test("a container exiting non-zero reports its redacted stderr tail on the failure and the run span", { timeout: 30_000 }, async () => {
  await withStderrPi("crash", async ({ process: pi, events, tracing, exporter, next }) => {
    tracing.startRun("exec-crash-1", "pi-stderr");
    pi.prompt("exec-crash-1", "fix");
    const error = await next((event) => event.type === "error") as Extract<PublicRunEvent, { type: "error" }>;
    assert.equal(error.code, "container_exited");
    assert.equal(error.stderr_tail, [
      "pi: loading extensions", "provider refused [REDACTED]", "mcp gateway refused [REDACTED]",
      "Error: worker crashed", "    at run (/opt/pi/dist/main.js:1:1)", "",
    ].join("\n"));
    assert.doesNotMatch(JSON.stringify(events), /l1-secret/, "no token handed to the container comes back out");
    tracing.endRun("exec-crash-1", "failed");
    const span = runSpan(exporter, "exec-crash-1");
    assert.equal(span.attributes["agent.container.stderr_tail"], error.stderr_tail);
    assert.equal(span.attributes["agent.container.exit_code"], 3);
    assert.equal(span.attributes["agent.error.code"], "container_exited");
  });
});

test("a container dying while Pi is configured reports container_exited with its stderr, not a bare configuration failure", { timeout: 30_000 }, async () => {
  await withStderrPi("configure", async ({ process: pi, events, next }) => {
    pi.prompt("exec-configure-1", "fix");
    const error = await next((event) => event.type === "error") as Extract<PublicRunEvent, { type: "error" }>;
    assert.equal(error.code, "container_exited");
    assert.equal(error.stderr_tail, "pi: loading extensions\nError: model registry is unreadable\n");
    await new Promise((resolve) => setTimeout(resolve, 100));
    assert.deepEqual(events.filter((event) => event.type === "error").map((event) => (event as { code: string }).code), ["container_exited"]);
  });
});

test("a container writing 10 MiB to stderr leaves only its last 16 KiB on the failure", { timeout: 30_000 }, async () => {
  await withStderrPi("flood", async ({ process: pi, next }) => {
    pi.prompt("exec-flood-1", "fix");
    const error = await next((event) => event.type === "error") as Extract<PublicRunEvent, { type: "error" }>;
    const tail = error.stderr_tail ?? "";
    assert.ok(Buffer.byteLength(tail) <= STDERR_TAIL_BYTES);
    assert.ok(Buffer.byteLength(tail) > STDERR_TAIL_BYTES - 1024 - 10);
    assert.ok(tail.endsWith("\nLAST LINE\n"));
    assert.match(tail, /^x{1023}\n/, "the tail starts at a whole line");
    assert.doesNotMatch(tail, /FIRST LINE|pi: loading/);
  });
});

test("a clean exit attaches no stderr, and the next container's crash carries only its own", { timeout: 30_000 }, async () => {
  await withStderrPi("clean", async ({ process: pi, events, tracing, exporter, dir, next }) => {
    tracing.startRun("exec-clean-1", "pi-stderr");
    pi.prompt("exec-clean-1", "fix");
    await next((event) => event.type === "settled");
    tracing.endRun("exec-clean-1", "completed");
    assert.deepEqual(events.filter((event) => event.type === "error"), []);
    assert.doesNotMatch(JSON.stringify(events), /stderr|warning: provider is slow/);
    const clean = runSpan(exporter, "exec-clean-1");
    assert.equal(clean.attributes["agent.container.stderr_tail"], undefined);
    assert.equal(clean.attributes["agent.container.exit_code"], undefined);

    await writeFile(path.join(dir, "mode"), "crash");
    pi.prompt("exec-crash-2", "again");
    const error = await next((event) => event.type === "error") as Extract<PublicRunEvent, { type: "error" }>;
    assert.match(error.stderr_tail ?? "", /^pi: loading extensions\nprovider refused \[REDACTED\]\n/);
    assert.doesNotMatch(error.stderr_tail ?? "", /warning: provider is slow/);
  });
});
