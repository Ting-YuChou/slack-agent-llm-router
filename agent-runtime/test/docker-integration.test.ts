import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import { promisify } from "node:util";

import { buildAgentDockerArgs, DockerPiProcess } from "../src/container-runtime.js";
import type { PublicRunEvent } from "../src/pi-rpc.js";
import { WorktreeManager } from "../src/worktree-manager.js";
import { fakeGatewayScript, scriptedTurns, waitForContainerHealthz } from "./support/fake-provider.js";
import { AGENT_TEST_IMAGE } from "./support/docker-images.js";

const exec = promisify(execFile);
const enabled = process.env.PI_AGENT_DOCKER_INTEGRATION === "1";

async function command(command: string, args: string[], cwd?: string): Promise<string> {
  return (await exec(command, args, { cwd, encoding: "utf8" })).stdout.trim();
}

test("real Pi preserves its session while switching from Luna max to Sol high", { skip: !enabled, timeout: 60_000 }, async () => {
  const suffix = Math.random().toString(16).slice(2, 10);
  const network = `pi-integration-${suffix}`;
  const gateway = `model-gateway-${suffix}`;
  const repo = await mkdtemp(path.join(tmpdir(), "pi-integration-repo-"));
  const worktreeRoot = await mkdtemp(path.join(tmpdir(), "pi-integration-worktrees-"));
  const sessionStatePath = await mkdtemp(path.join(tmpdir(), "pi-integration-state-"));
  await command("git", ["init", "-b", "main"], repo);
  await writeFile(path.join(repo, "README.md"), "base\n");
  await command("git", ["add", "README.md"], repo);
  await command("git", ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "base"], repo);
  const worktrees = new WorktreeManager({ repoPath: repo, worktreeRoot, baseRef: "HEAD" });
  const isolated = await worktrees.create(`integration-${suffix}`);
  const hostUid = process.getuid?.();
  const hostGid = process.getgid?.();
  assert.notEqual(hostUid, undefined);
  assert.notEqual(hostGid, undefined);
  assert.notEqual(hostUid, 0, "integration test must run as a non-root host user");

  const gatewayScript = fakeGatewayScript(scriptedTurns([
    {
      kind: "function_call", name: "write", arguments: { path: "agent.txt", content: "written by pi\n" },
      expect: { model: "gpt-5.6-luna", effort: "max", includes: ["Close one concrete coverage gap"] },
    },
    { kind: "text", text: "Implemented the requested file.", expect: { model: "gpt-5.6-luna", effort: "max" } },
    { kind: "text", text: "agent.txt contains written by pi.", expect: { model: "gpt-5.6-sol", effort: "high", includes: ["written by pi"] } },
  ]));

  let piProcess: DockerPiProcess | undefined;
  try {
    await command("docker", ["network", "create", "--internal", network]);
    await command("docker", ["run", "--detach", "--rm", "--name", gateway, "--network", network, "--network-alias", "model-gateway", "node:22.22.0-bookworm-slim", "node", "-e", gatewayScript]);
    await waitForContainerHealthz(gateway);
    const events: PublicRunEvent[] = [];
    let resolveSettled!: () => void;
    let settled = new Promise<void>((resolve) => { resolveSettled = resolve; });
    piProcess = new DockerPiProcess(
      {
        name: `pi-integration-${suffix}`,
        image: AGENT_TEST_IMAGE,
        network,
        worktreePath: isolated.path,
        gitMetadataPath: isolated.gitDir,
        sessionStatePath,
        gatewayUrl: "http://model-gateway:8080",
        modelRef: "openai/gpt-5.6-luna",
        extensionPaths: ["/opt/pi/extensions/policy.ts", "/opt/pi/extensions/model-gateway.ts"],
        pluginPaths: ["/opt/pi/plugins/workspace-summary.ts"],
        skillPaths: ["/opt/pi/skills/test-gap/SKILL.md"],
        toolNames: ["read", "write", "edit", "bash", "grep", "find", "ls", "workspace_summary"],
        user: `${hostUid}:${hostGid}`,
      },
      () => "fake-run-token",
      (event) => {
        events.push(event);
        if (event.type === "approval") piProcess!.decide(event.approval_id, true);
        if (event.type === "settled") resolveSettled();
        if (event.type === "error") resolveSettled();
      },
    );
    piProcess.prompt("run-1", "/skill:test-gap Create agent.txt with the requested content.");
    await settled;

    assert.equal(await readFile(path.join(isolated.path, "agent.txt"), "utf8"), "written by pi\n");
    assert.ok(events.some((event) => event.type === "approval"));
    assert.ok(events.some((event) => event.type === "tool" && event.tool === "write"));
    assert.doesNotMatch(JSON.stringify(events), /thinking|chain.of.thought/i);

    settled = new Promise<void>((resolve) => { resolveSettled = resolve; });
    piProcess.prompt("run-2", "Read agent.txt and confirm its content.", {
      modelRef: "openai/gpt-5.6-sol",
      effort: "high",
      source: "jev",
    });
    await settled;

    assert.ok(
      events.some((event) => event.type === "answer" && /written by pi/i.test(event.text)),
      JSON.stringify(events),
    );
    assert.ok(!events.some((event) => event.type === "error"), JSON.stringify(events));
  } finally {
    await piProcess?.close();
    await command("docker", ["stop", gateway]).catch(() => "");
    await command("docker", ["network", "rm", network]).catch(() => "");
    await worktrees.remove(isolated.path).catch(() => undefined);
  }
});

test("Pi 1.0.1 resumes a session persisted by Pi 0.83", { skip: !enabled, timeout: 60_000 }, async () => {
  const suffix = Math.random().toString(16).slice(2, 10);
  const network = `pi-legacy-${suffix}`;
  const gateway = `model-gateway-legacy-${suffix}`;
  const repo = await mkdtemp(path.join(tmpdir(), "pi-legacy-repo-"));
  const worktreeRoot = await mkdtemp(path.join(tmpdir(), "pi-legacy-worktrees-"));
  const sessionStatePath = await mkdtemp(path.join(tmpdir(), "pi-legacy-state-"));
  await command("git", ["init", "-b", "main"], repo);
  await writeFile(path.join(repo, "README.md"), "base\n");
  await command("git", ["add", "README.md"], repo);
  await command("git", ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "base"], repo);
  const worktrees = new WorktreeManager({ repoPath: repo, worktreeRoot, baseRef: "HEAD" });
  const isolated = await worktrees.create(`legacy-${suffix}`);
  const fixture = await readFile(new URL("../../test/fixtures/pi-0.83-session.jsonl", import.meta.url), "utf8");
  await writeFile(
    path.join(sessionStatePath, "legacy-session.jsonl"),
    fixture.replace("__WORKTREE__", isolated.path),
  );
  const hostUid = process.getuid?.();
  const hostGid = process.getgid?.();
  assert.notEqual(hostUid, undefined);
  assert.notEqual(hostGid, undefined);
  assert.notEqual(hostUid, 0, "integration test must run as a non-root host user");

  const gatewayScript = fakeGatewayScript(scriptedTurns([
    { kind: "text", text: "Restored legacy-session-marker-083.", expect: { includes: ["legacy-session-marker-083"] } },
  ]));

  let piProcess: DockerPiProcess | undefined;
  try {
    await command("docker", ["network", "create", "--internal", network]);
    await command("docker", ["run", "--detach", "--rm", "--name", gateway, "--network", network, "--network-alias", "model-gateway", "node:22.22.0-bookworm-slim", "node", "-e", gatewayScript]);
    await waitForContainerHealthz(gateway);
    const events: PublicRunEvent[] = [];
    let resolveSettled!: () => void;
    const settled = new Promise<void>((resolve) => { resolveSettled = resolve; });
    piProcess = new DockerPiProcess(
      {
        name: `pi-legacy-${suffix}`,
        image: AGENT_TEST_IMAGE,
        network,
        worktreePath: isolated.path,
        gitMetadataPath: isolated.gitDir,
        sessionStatePath,
        gatewayUrl: "http://model-gateway:8080",
        modelRef: "openai/gpt-5.6-luna",
        extensionPaths: ["/opt/pi/extensions/model-gateway.ts"],
        pluginPaths: [],
        skillPaths: [],
        toolNames: ["read"],
        continueSession: true,
        user: `${hostUid}:${hostGid}`,
      },
      () => "fake-run-token",
      (event) => {
        events.push(event);
        if (event.type === "settled" || event.type === "error") resolveSettled();
      },
    );
    piProcess.prompt("legacy-follow-up", "Repeat the marker from the previous turn.");
    await settled;
    assert.ok(
      events.some((event) => event.type === "answer" && /legacy-session-marker-083/.test(event.text)),
      JSON.stringify(events),
    );
    assert.ok(!events.some((event) => event.type === "error"), JSON.stringify(events));
  } finally {
    await piProcess?.close();
    await command("docker", ["stop", gateway]).catch(() => "");
    await command("docker", ["network", "rm", network]).catch(() => "");
    await worktrees.remove(isolated.path).catch(() => undefined);
  }
});

test("Pi builtin MCP uses only the trusted GitHub registration and ignores project mcp.json", { skip: !enabled, timeout: 60_000 }, async () => {
  const suffix = Math.random().toString(16).slice(2, 10);
  const network = `pi-mcp-${suffix}`;
  const modelGateway = `model-gateway-mcp-${suffix}`;
  const mcpGateway = `mcp-gateway-${suffix}`;
  const repo = await mkdtemp(path.join(tmpdir(), "pi-mcp-repo-"));
  const worktreeRoot = await mkdtemp(path.join(tmpdir(), "pi-mcp-worktrees-"));
  const sessionStatePath = await mkdtemp(path.join(tmpdir(), "pi-mcp-state-"));
  await command("git", ["init", "-b", "main"], repo);
  await mkdir(path.join(repo, ".pi"));
  await writeFile(path.join(repo, ".pi", "mcp.json"), JSON.stringify({ mcpServers: { evil: { url: "http://evil.invalid/mcp", exposure: "direct" } } }));
  await writeFile(path.join(repo, "README.md"), "base\n");
  await command("git", ["add", "."], repo);
  await command("git", ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "base"], repo);
  const worktrees = new WorktreeManager({ repoPath: repo, worktreeRoot, baseRef: "HEAD" });
  const isolated = await worktrees.create(`mcp-${suffix}`);
  const hostUid = process.getuid?.();
  const hostGid = process.getgid?.();
  assert.notEqual(hostUid, undefined);
  assert.notEqual(hostGid, undefined);
  assert.notEqual(hostUid, 0, "integration test must run as a non-root host user");

const fakeMcpScript = String.raw`
const http=require("node:http");
http.createServer((req,res)=>{let body="";req.on("data",c=>body+=c);req.on("end",()=>{
  if(req.url==="/healthz"){res.writeHead(200);res.end("ok");return;}
  if(req.method==="GET"){res.writeHead(405);res.end();return;}
  if(req.method==="DELETE"){res.writeHead(200);res.end();return;}
  const msg=JSON.parse(body||"{}");
  if(msg.method==="notifications/initialized"){res.writeHead(202);res.end();return;}
  let result;
  if(msg.method==="initialize") result={protocolVersion:"2025-06-18",capabilities:{tools:{}},serverInfo:{name:"github",version:"test"}};
  else if(msg.method==="tools/list") result={tools:[{name:"get_file_contents",description:"Read a repository file",inputSchema:{type:"object",properties:{owner:{type:"string"},repo:{type:"string"},path:{type:"string"}},required:["owner","repo","path"]}}]};
  else if(msg.method==="tools/call"&&msg.params?.name==="get_file_contents"&&msg.params?.arguments?.owner==="acme"&&msg.params?.arguments?.repo==="widgets") result={content:[{type:"text",text:"trusted-mcp-marker"}]};
  else {res.writeHead(400);res.end("unexpected MCP request");return;}
  res.writeHead(200,{"content-type":"application/json","mcp-session-id":"fake-session"});res.end(JSON.stringify({jsonrpc:"2.0",id:msg.id,result}));
});}).listen(8090,"0.0.0.0");`;
  const fakeModelScript = fakeGatewayScript(scriptedTurns([
    {
      kind: "function_call", name: "mcp__github__get_file_contents", arguments: { owner: "acme", repo: "widgets", path: "README.md" },
      expect: { includes: ["mcp__github__get_file_contents"], excludes: ["mcp__evil__"] },
    },
    { kind: "text", text: "Read trusted-mcp-marker through GitHub MCP.", expect: { includes: ["trusted-mcp-marker"], excludes: ["mcp__evil__"] } },
  ]));

  let piProcess: DockerPiProcess | undefined;
  try {
    await command("docker", ["network", "create", "--internal", network]);
    await command("docker", ["run", "--detach", "--rm", "--name", modelGateway, "--network", network, "--network-alias", "model-gateway", "node:22.22.0-bookworm-slim", "node", "-e", fakeModelScript]);
    await command("docker", ["run", "--detach", "--rm", "--name", mcpGateway, "--network", network, "--network-alias", "mcp-gateway", "node:22.22.0-bookworm-slim", "node", "-e", fakeMcpScript]);
    await waitForContainerHealthz(modelGateway);
    await waitForContainerHealthz(mcpGateway, 8090);
    const events: PublicRunEvent[] = [];
    let resolveSettled!: () => void;
    const settled = new Promise<void>((resolve) => { resolveSettled = resolve; });
    piProcess = new DockerPiProcess({
      name: `pi-mcp-${suffix}`, image: AGENT_TEST_IMAGE, network,
      worktreePath: isolated.path, gitMetadataPath: isolated.gitDir, sessionStatePath,
      gatewayUrl: "http://model-gateway:8080", modelRef: "openai/gpt-5.6-luna",
      extensionPaths: ["/opt/pi/extensions/model-gateway.ts"], pluginPaths: [], skillPaths: [], toolNames: ["read"],
      user: `${hostUid}:${hostGid}`,
      mcp: { mode: "github_read_only", servers: [{ server: "github", gatewayUrl: "http://mcp-gateway:8090/mcp", token: "" }] },
    }, () => "fake-model-token", (event) => {
      events.push(event);
      if (event.type === "settled" || event.type === "error") resolveSettled();
    }, () => ({ github: "fake-mcp-token" }));
    piProcess.prompt("mcp-run", "Read README.md from acme/widgets with GitHub MCP.");
    await settled;
    const diagnostic = JSON.stringify({ events, model: await command("docker", ["logs", modelGateway]), mcp: await command("docker", ["logs", mcpGateway]) });
    assert.ok(events.some((event) => event.type === "tool" && event.tool === "mcp__github__get_file_contents"), diagnostic);
    assert.ok(events.some((event) => event.type === "answer" && event.text.includes("trusted-mcp-marker")), diagnostic);
    assert.ok(!events.some((event) => event.type === "error"), JSON.stringify(events));
  } finally {
    await piProcess?.close();
    await command("docker", ["stop", modelGateway]).catch(() => "");
    await command("docker", ["stop", mcpGateway]).catch(() => "");
    await command("docker", ["network", "rm", network]).catch(() => "");
    await worktrees.remove(isolated.path).catch(() => undefined);
  }
});

test("real Pi requests pass the gateway allowlist and bash children cannot read the run token", { skip: !enabled, timeout: 120_000 }, async () => {
  const { allowedAnthropicBeta, sanitizeAgentRequest } = await import("../src/gateway-request-policy.js");
  const suffix = Math.random().toString(16).slice(2, 10);
  const network = `pi-allowlist-${suffix}`;
  const gateway = `model-gateway-${suffix}`;
  const repo = await mkdtemp(path.join(tmpdir(), "pi-allowlist-repo-"));
  const worktreeRoot = await mkdtemp(path.join(tmpdir(), "pi-allowlist-worktrees-"));
  const sessionStatePath = await mkdtemp(path.join(tmpdir(), "pi-allowlist-state-"));
  const recordDir = await mkdtemp(path.join(tmpdir(), "pi-allowlist-record-"));
  await command("chmod", ["777", recordDir]);
  await command("git", ["init", "-b", "main"], repo);
  await writeFile(path.join(repo, "README.md"), "base\n");
  await command("git", ["add", "README.md"], repo);
  await command("git", ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "base"], repo);
  const worktrees = new WorktreeManager({ repoPath: repo, worktreeRoot, baseRef: "HEAD" });
  const isolated = await worktrees.create(`allowlist-${suffix}`);
  const hostUid = process.getuid?.();
  const hostGid = process.getgid?.();
  // Every process's initial environment in the container, PID 1 included, must be free of the run's token.
  const environ = "const fs=require('fs'),path=require('path');let leak=0;for(const p of fs.readdirSync('/proc')){if(!/^[0-9]+$/.test(p))continue;try{if(fs.readFileSync(path.join('/proc',p,'environ'),'latin1').includes('fake-run-token'))leak++}catch{}}";
  const probe = `node -e "${environ}process.stdout.write('TOKEN='+(process.env.OPENAI_API_KEY||'none')+';MCP='+(process.env.PI_AGENT_MCP_SERVERS_JSON?'set':'none')+';ENVIRON='+(leak?'leak':'clean'))"`;
  const gatewayScript = fakeGatewayScript(scriptedTurns([
    { kind: "function_call", name: "bash", arguments: { command: probe } },
    { kind: "text", text: "probe done" },
    { kind: "anthropic_text", text: "anthropic-ok" },
  ]), { recordPath: "/record/requests.jsonl" });

  let piProcess: DockerPiProcess | undefined;
  try {
    await command("docker", ["network", "create", "--internal", network]);
    await command("docker", ["run", "--detach", "--rm", "--name", gateway, "--network", network, "--network-alias", "model-gateway", "--mount", `type=bind,src=${recordDir},dst=/record`, "node:22.22.0-bookworm-slim", "node", "-e", gatewayScript]);
    await waitForContainerHealthz(gateway);
    const events: PublicRunEvent[] = [];
    let resolveSettled!: () => void;
    let settled = new Promise<void>((resolve) => { resolveSettled = resolve; });
    piProcess = new DockerPiProcess(
      {
        name: `pi-allowlist-${suffix}`, image: AGENT_TEST_IMAGE, network, worktreePath: isolated.path,
        gitMetadataPath: isolated.gitDir, sessionStatePath, gatewayUrl: "http://model-gateway:8080",
        modelRef: "openai/gpt-5.6-luna",
        extensionPaths: ["/opt/pi/extensions/policy.ts", "/opt/pi/extensions/model-gateway.ts", "/opt/pi/extensions/secret-env.ts"],
        pluginPaths: [], skillPaths: [], toolNames: ["read", "write", "edit", "bash", "grep", "find", "ls"],
        user: `${hostUid}:${hostGid}`,
      },
      () => "fake-run-token-value",
      (event) => {
        events.push(event);
        if (event.type === "approval") piProcess!.decide(event.approval_id, true);
        if (event.type === "settled" || event.type === "error") resolveSettled();
      },
    );
    piProcess.prompt("run-1", "Run the environment probe.");
    await settled;
    assert.ok(!events.some((event) => event.type === "error"), JSON.stringify(events));

    settled = new Promise<void>((resolve) => { resolveSettled = resolve; });
    piProcess.prompt("run-2", "Say hello.", { modelRef: "anthropic/claude-sonnet-4-6", effort: "max", source: "jev" });
    await settled;
    assert.ok(events.some((event) => event.type === "answer" && event.text.includes("anthropic-ok")), JSON.stringify(events));

    const requests = (await readFile(path.join(recordDir, "requests.jsonl"), "utf8")).trim().split("\n").map((line) => JSON.parse(line));
    const openai = requests.filter((request) => request.url === "/openai/v1/responses");
    const claude = requests.filter((request) => new URL(request.url, "http://x").pathname === "/anthropic/v1/messages");
    assert.ok(openai.length >= 2, JSON.stringify(requests.map((request) => request.url)));
    assert.ok(claude.length >= 1, JSON.stringify(requests.map((request) => request.url)));
    for (const request of openai) assert.deepEqual(sanitizeAgentRequest("openai-responses", request.body).ok, true, JSON.stringify(sanitizeAgentRequest("openai-responses", request.body)));
    for (const request of claude) {
      assert.deepEqual(sanitizeAgentRequest("anthropic-messages", request.body).ok, true, JSON.stringify(sanitizeAgentRequest("anthropic-messages", request.body)));
      const beta = request.headers["anthropic-beta"];
      if (beta) assert.equal(allowedAnthropicBeta(beta), beta.split(",").map((item: string) => item.trim()).join(","));
    }
    const toolOutput = JSON.stringify(openai[1].body.input);
    assert.match(toolOutput, /TOKEN=none;MCP=none;ENVIRON=clean/);
    // The token reached Pi through the credential dialog: every model request carries it.
    for (const request of [...openai, ...claude]) {
      assert.ok([request.headers.authorization, request.headers["x-api-key"]].some((value) => value === "Bearer fake-run-token-value" || value === "fake-run-token-value"), JSON.stringify(request.headers));
    }
    assert.doesNotMatch(toolOutput, /fake-run-token-value/);
  } finally {
    await piProcess?.close();
    await command("docker", ["stop", gateway]).catch(() => "");
    await command("docker", ["network", "rm", network]).catch(() => "");
    await worktrees.remove(isolated.path).catch(() => undefined);
  }
});

test("inside the writer container git sees only the baseline history and the session's own branch", { skip: !enabled, timeout: 60_000 }, async () => {
  const repo = await mkdtemp(path.join(tmpdir(), "pi-mirror-repo-"));
  const worktreeRoot = await mkdtemp(path.join(tmpdir(), "pi-mirror-worktrees-"));
  const sessionStatePath = await mkdtemp(path.join(tmpdir(), "pi-mirror-state-"));
  const commit = (message: string) => command("git", ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-q", "-m", message], repo);
  await command("git", ["init", "-b", "main"], repo);
  await writeFile(path.join(repo, "README.md"), "base\n");
  await command("git", ["add", "README.md"], repo);
  await commit("base");
  await command("git", ["checkout", "-q", "-b", "unrelated"], repo);
  await writeFile(path.join(repo, "unrelated.txt"), "not for agents\n");
  await command("git", ["add", "unrelated.txt"], repo);
  await commit("unrelated");
  const unrelated = await command("git", ["rev-parse", "HEAD"], repo);
  await command("git", ["checkout", "-q", "main"], repo);
  await command("git", ["remote", "add", "origin", "https://example.invalid/secret.git"], repo);
  const worktrees = new WorktreeManager({ repoPath: repo, worktreeRoot, baseRef: "main" });
  const other = await worktrees.create("other-session");
  await writeFile(path.join(other.path, "other.txt"), "other session\n");
  const otherCommit = await worktrees.commit(other.path, "other session work");
  const own = await worktrees.create("own-session");
  await writeFile(path.join(own.path, "own.txt"), "own session\n");
  const ownCommit = await worktrees.commit(own.path, "own session work");
  const hostUid = process.getuid?.();
  const hostGid = process.getgid?.();
  assert.notEqual(hostUid, 0, "integration test must run as a non-root host user");

  try {
    const args = buildAgentDockerArgs({
      name: "unused", image: AGENT_TEST_IMAGE, network: "none", worktreePath: own.path, gitMetadataPath: own.gitDir, sessionStatePath,
      gatewayUrl: "http://model-gateway:8080", modelRef: "openai/gpt-5.6-luna", extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"],
      user: `${hostUid}:${hostGid}`,
    });
    const mounts = args.flatMap((arg, index) => args[index - 1] === "--mount" ? ["--mount", arg] : []);
    const inContainer = (...gitArgs: string[]) => command("docker", ["run", "--rm", "--read-only", "--network", "none", "--user", `${hostUid}:${hostGid}`,
      "--tmpfs", "/tmp", "--env", "HOME=/tmp", ...mounts, "--workdir", own.path, AGENT_TEST_IMAGE, "git", ...gitArgs]);

    const expected = (await command("git", ["rev-list", ownCommit], repo)).split("\n").sort();
    assert.deepEqual((await inContainer("log", "--all", "--format=%H")).split("\n").sort(), expected);
    assert.ok(expected.includes(own.baselineCommit) && !expected.includes(otherCommit) && !expected.includes(unrelated));
    assert.equal(await inContainer("for-each-ref", "--format=%(refname)"), `refs/heads/${own.branch}`);
    await assert.rejects(inContainer("cat-file", "-e", `${otherCommit}^{commit}`));
    await assert.rejects(inContainer("config", "--get", "remote.origin.url"));
  } finally {
    await worktrees.remove(own.path).catch(() => undefined);
    await worktrees.remove(other.path).catch(() => undefined);
  }
});

test("session networks reach only their gateways: no external DNS, no egress, no other session", { skip: !enabled, timeout: 120_000 }, async () => {
  const { RunNetworks } = await import("../src/run-network.js");
  const suffix = Math.random().toString(16).slice(2, 10);
  const gateway = `run-net-gateway-${suffix}`;
  const shared = `run-net-shared-${suffix}`;
  const image = "node:22.22.0-bookworm-slim";
  const networks = new RunNetworks({ attachments: [{ container: gateway, alias: "model-gateway" }] });
  const sessions = [`A-${suffix}`, `B-${suffix}`];
  const containers = sessions.map(id => `run-net-${id}`);
  try {
    await command("docker", ["network", "create", "--internal", shared]);
    await command("docker", ["run", "--detach", "--rm", "--name", gateway, "--network", shared, image, "node", "-e", "require('http').createServer((_, r) => r.end('ok')).listen(8080)"]);
    for (const [index, id] of sessions.entries()) {
      const network = await networks.create(id);
      await command("docker", ["run", "--detach", "--rm", "--name", containers[index]!, "--network", network, "--dns", "127.0.0.1", image, "sleep", "300"]);
    }
    const inA = (script: string) => exec("docker", ["exec", containers[0]!, "node", "-e", script], { encoding: "utf8", timeout: 20_000 });
    const resolves = (host: string) => `require('dns').lookup('${host}', e => process.exit(e ? 1 : 0))`;
    const connects = (host: string, port: number) => `const s=require('net').connect({host:'${host}',port:${port},timeout:3000},()=>process.exit(0));s.on('error',()=>process.exit(1));s.on('timeout',()=>process.exit(1))`;
    await inA(connects("model-gateway", 8080));
    await assert.rejects(inA(resolves("example.com")), "external names must not resolve");
    await assert.rejects(inA(resolves("host.docker.internal")), "the host must not resolve");
    await assert.rejects(inA(connects("1.1.1.1", 443)), "no egress to the internet");
    const ipB = await command("docker", ["inspect", "--format", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", containers[1]!]);
    assert.ok(ipB, "session B has an address");
    await assert.rejects(inA(connects(ipB, 22)), "session A cannot reach session B");
    await assert.rejects(inA(resolves(containers[1]!)), "session A cannot even resolve session B");
  } finally {
    for (const container of [...containers, gateway]) await exec("docker", ["rm", "-f", container]).catch(() => undefined);
    for (const id of sessions) await networks.remove(id).catch(() => undefined);
    await exec("docker", ["network", "rm", shared]).catch(() => undefined);
  }
});
