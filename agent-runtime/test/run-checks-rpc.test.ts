import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { chmod, mkdtemp, copyFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

import { PiRpcBridge, RpcJsonlDecoder, type PublicRunEvent } from "../src/pi-rpc.js";
import { fakeGatewayScript, scriptedTurns, waitForContainerHealthz } from "./support/fake-provider.js";

const enabled = process.env.PI_AGENT_DOCKER_INTEGRATION === "1";

/** B2-2 transport risk check: a 16 KiB request and a 16 KiB response cross a real `pi --mode rpc` intact. */
test("real Pi RPC carries a 16 KiB extension UI input request and response intact", { skip: !enabled, timeout: 60_000 }, async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "pi-ui-probe-"));
  await copyFile(fileURLToPath(new URL("../../test/fixtures/extensions/ui-roundtrip-probe.ts", import.meta.url)), path.join(dir, "probe.ts"));
  await chmod(dir, 0o755); await chmod(path.join(dir, "probe.ts"), 0o644);
  const child = spawn("docker", ["run", "--rm", "-i", "--network", "none", "--read-only", "--cap-drop", "ALL", "--tmpfs", "/tmp:rw,nosuid,size=67108864", "--env", "HOME=/tmp",
    "--mount", `type=bind,src=${dir},dst=/probe,readonly`, "slack-pi-agent:1.0.1",
    "pi", "--mode", "rpc", "--no-session", "--no-extensions", "--no-skills", "--no-prompt-templates", "--no-themes", "-e", "/probe/probe.ts"], { stdio: ["pipe", "pipe", "pipe"] });
  const decoder = new RpcJsonlDecoder(); let stderr = "";
  child.stderr.on("data", chunk => { stderr += chunk; });
  const value = "V".repeat(16 * 1024 - 400) + "é".repeat(200);
  try {
    const notified = await new Promise<string>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`no probe result; stderr: ${stderr.slice(-2000)}`)), 50_000);
      child.stdout.on("data", chunk => {
        for (const record of decoder.push(chunk)) {
          if (record.type === "extension_ui_request" && record.method === "input") {
            assert.equal(record.title, "pi-agent/run_checks");
            assert.equal(record.placeholder, "P".repeat(16 * 1024));
            child.stdin.write(`${JSON.stringify({ type: "extension_ui_response", id: record.id, value })}\n`);
          }
          if (record.type === "extension_ui_request" && record.method === "notify") { clearTimeout(timer); resolve(String(record.message)); }
        }
      });
      child.once("exit", code => { clearTimeout(timer); reject(new Error(`pi exited ${code}: ${stderr.slice(-2000)}`)); });
      child.stdin.write(`${JSON.stringify({ id: "probe-1", type: "prompt", message: "/probe" })}\n`);
    });
    assert.equal(notified, `len=${value.length};sha=${createHash("sha256").update(value).digest("hex")}`);
  } finally { child.kill(); await rm(dir, { recursive: true, force: true }); }
});


function bridge() {
  const events: PublicRunEvent[] = []; const sent: Array<Record<string, unknown>> = [];
  const rpc = new PiRpcBridge({ writeLine: line => { sent.push(JSON.parse(line)); }, onEvent: event => { events.push(event); } });
  const feed = (...records: Array<Record<string, unknown>>) => rpc.feed(records.map(record => `${JSON.stringify(record)}\n`).join(""));
  return { rpc, events, sent, feed };
}
const start = (id: string, tool = "run_checks") => ({ type: "tool_execution_start", toolCallId: id, toolName: tool, args: {} });
const ask = (id: string, toolCallId: string, extra: Record<string, unknown> = {}) => ({ type: "extension_ui_request", id, method: "input", title: "pi-agent/run_checks", placeholder: JSON.stringify({ tool_call_id: toolCallId, ...extra }) });

test("bridge turns a run_checks dialog into checks_request only for the in-flight run_checks call", () => {
  const { events, sent, feed, rpc } = bridge();
  feed(start("tc1"), ask("ui-1", "tc1", { checks: ["lint"] }));
  assert.deepEqual(events.filter(event => event.type === "checks_request"), [{ type: "checks_request", request_id: "ui-1", tool_call_id: "tc1", checks: ["lint"] }]);
  assert.deepEqual(sent, []);
  rpc.respondUi("ui-1", { value: "{\"status\":\"passed\"}" });
  assert.deepEqual(sent.at(-1), { type: "extension_ui_response", id: "ui-1", value: "{\"status\":\"passed\"}" });
});

test("bridge cancels forged, repeated and unknown dialogs instead of leaving Pi waiting", () => {
  const { events, sent, feed } = bridge();
  feed(ask("forged", "tc9"));
  feed(start("tc2", "bash"), ask("wrong-tool", "tc2"));
  feed(start("tc3"), ask("first", "tc3"), ask("again", "tc3"));
  feed({ type: "tool_execution_end", toolCallId: "tc3", toolName: "run_checks", result: {} }, ask("after-end", "tc3"));
  feed({ type: "extension_ui_request", id: "pick", method: "select", title: "Pick", options: ["a"] });
  feed({ type: "extension_ui_request", id: "text", method: "editor", title: "Edit" });
  feed({ type: "extension_ui_request", id: "other-input", method: "input", title: "Name?" });
  feed(start("tc4"), { type: "extension_ui_request", id: "bad-json", method: "input", title: "pi-agent/run_checks", placeholder: "{" });
  feed({ type: "extension_ui_request", id: "note", method: "notify", message: "hi" });
  assert.deepEqual(events.filter(event => event.type === "checks_request").map(event => (event as { request_id: string }).request_id), ["first"]);
  assert.deepEqual(sent.map(record => record.id), ["forged", "wrong-tool", "again", "after-end", "pick", "text", "other-input", "bad-json"]);
  assert.ok(sent.every(record => record.type === "extension_ui_response" && record.cancelled === true));
});

test("bridge still surfaces confirm dialogs as approvals", () => {
  const { events, sent, feed } = bridge();
  feed({ type: "extension_ui_request", id: "c1", method: "confirm", title: "Run?", message: "npm test" });
  assert.equal(events[0]?.type, "approval");
  assert.deepEqual(sent, []);
});

/** Phase 2 acceptance: a real Pi writer calls run_checks, the host answers the dialog, and the model sees the report. */
test("real Pi writer completes a run_checks round trip through the dialog channel", { skip: !enabled, timeout: 180_000 }, async () => {
  const { execFile } = await import("node:child_process");
  const { promisify } = await import("node:util");
  const { mkdir, writeFile } = await import("node:fs/promises");
  const { DockerPiProcess } = await import("../src/container-runtime.js");
  const { WorktreeManager } = await import("../src/worktree-manager.js");
  const run = promisify(execFile);
  const command = async (file: string, args: string[], cwd?: string, input?: string) => {
    const child = run(file, args, { cwd, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 });
    if (input !== undefined) { child.child.stdin?.end(input); }
    return (await child).stdout.trim();
  };
  const suffix = Math.random().toString(16).slice(2, 10);
  const network = `pi-checks-${suffix}`, gateway = `model-gateway-checks-${suffix}`, image = `slack-pi-agent:run-checks-${suffix}`;
  const repo = await mkdtemp(path.join(tmpdir(), "pi-checks-repo-"));
  const worktreeRoot = await mkdtemp(path.join(tmpdir(), "pi-checks-worktrees-"));
  const sessionStatePath = await mkdtemp(path.join(tmpdir(), "pi-checks-state-"));
  await mkdir(path.join(repo, "src"));
  await writeFile(path.join(repo, "src", "a.js"), "export const a = 1;\n");
  await command("git", ["init", "-q", "-b", "main"], repo);
  await command("git", ["add", "."], repo);
  await command("git", ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base"], repo);
  const worktrees = new WorktreeManager({ repoPath: repo, worktreeRoot, baseRef: "HEAD" });
  const isolated = await worktrees.create(`checks-${suffix}`);
  const marker = "lint-report-marker-7f3a";
  const gatewayScript = fakeGatewayScript(scriptedTurns([
    { kind: "function_call", name: "run_checks", arguments: { checks: ["lint"] }, expect: { includes: ["run_checks"] } },
    { kind: "text", text: "Checks report received.", expect: { includes: [marker] } },
  ]));
  let piProcess: InstanceType<typeof DockerPiProcess> | undefined;
  try {
    // The released agent image plus this branch's run_checks extension.
    await command("docker", ["build", "-q", "-t", image, "-f", "-", fileURLToPath(new URL("../../extensions", import.meta.url))], undefined,
      "FROM slack-pi-agent:1.0.1\nCOPY --chown=10001:10001 checks.ts /opt/pi/extensions/checks.ts\n");
    await command("docker", ["network", "create", "--internal", network]);
    await command("docker", ["run", "--detach", "--rm", "--name", gateway, "--network", network, "--network-alias", "model-gateway", "node:22.22.0-bookworm-slim", "node", "-e", gatewayScript]);
    await waitForContainerHealthz(gateway);
    const events: PublicRunEvent[] = [];
    let resolveSettled!: () => void;
    const settled = new Promise<void>(resolve => { resolveSettled = resolve; });
    const report = { status: "failed", checks: [{ name: "lint", status: "failed" }], report: `src/a.js:1:1 E1 ${marker}` };
    piProcess = new DockerPiProcess({
      name: `pi-checks-${suffix}`, image, network, worktreePath: isolated.path, gitMetadataPath: path.join(repo, ".git"), sessionStatePath,
      gatewayUrl: "http://model-gateway:8080", modelRef: "openai/gpt-5.6-luna",
      extensionPaths: ["/opt/pi/extensions/model-gateway.ts", "/opt/pi/extensions/checks.ts"], pluginPaths: [], skillPaths: [],
      toolNames: ["read", "run_checks"], checksContext: { checks: ["lint"], defaultChecks: ["lint"], maxCalls: 2 },
      user: `${process.getuid?.()}:${process.getgid?.()}`,
    }, () => "fake-run-token", event => {
      events.push(event);
      if (event.type === "checks_request") piProcess!.respondUi(event.request_id, { value: JSON.stringify(report) });
      if (event.type === "settled" || event.type === "error") resolveSettled();
    });
    piProcess.prompt("run-1", "Run the lint check.");
    await settled;
    const request = events.find(event => event.type === "checks_request");
    assert.deepEqual(request && { ...request, request_id: "" }, { type: "checks_request", request_id: "", tool_call_id: "call_1|fc_1", checks: ["lint"] }, JSON.stringify(events));
    assert.ok(events.some(event => event.type === "settled" && event.outcome === "completed"), JSON.stringify(events));
    assert.ok(events.some(event => event.type === "answer" && /Checks report received/.test(event.text)), JSON.stringify(events));
  } finally {
    await piProcess?.close();
    await command("docker", ["stop", gateway]).catch(() => "");
    await command("docker", ["network", "rm", network]).catch(() => "");
    await command("docker", ["image", "rm", image]).catch(() => "");
    await worktrees.remove(isolated.path).catch(() => undefined);
    await rm(repo, { recursive: true, force: true }); await rm(worktreeRoot, { recursive: true, force: true }); await rm(sessionStatePath, { recursive: true, force: true });
  }
});
