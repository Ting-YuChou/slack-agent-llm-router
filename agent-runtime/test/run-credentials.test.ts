import assert from "node:assert/strict";
import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { test } from "node:test";

import { DockerPiProcess } from "../src/container-runtime.js";
import { PiRpcBridge, type PublicRunEvent } from "../src/pi-rpc.js";
import { CREDENTIAL_DIALOG, CREDENTIAL_PLACEHOLDER, parseCredentials } from "../src/run-credentials.js";

const dialog = (id: string, title = CREDENTIAL_DIALOG) => `${JSON.stringify({ type: "extension_ui_request", id, method: "input", title, placeholder: "" })}\n`;

test("only known credential names with bounded string values are accepted from the host's reply", () => {
  assert.deepEqual(parseCredentials(JSON.stringify({ OPENAI_API_KEY: "a", OPENROUTER_API_KEY: "b" })), { OPENAI_API_KEY: "a", OPENROUTER_API_KEY: "b" });
  for (const reply of [undefined, "", "not json", "[]", "null", "{}", JSON.stringify({ PATH: "/evil" }), JSON.stringify({ OPENAI_API_KEY: "" }),
    JSON.stringify({ OPENAI_API_KEY: 7 }), JSON.stringify({ OPENAI_API_KEY: "a", NODE_OPTIONS: "--require /x" }), JSON.stringify({ OPENAI_API_KEY: "x".repeat(8193) })]) {
    assert.equal(parseCredentials(reply), undefined, String(reply));
  }
});

async function loadGatewayExtension() {
  const previous = process.env.PI_MODEL_GATEWAY_URL;
  process.env.PI_MODEL_GATEWAY_URL = "http://model-gateway:8080";
  try {
    const module = await import(`${pathToFileURL(path.resolve(import.meta.dirname, "../../extensions/model-gateway.ts")).href}?${Math.random()}`);
    const handlers = new Map<string, (...args: any[]) => any>();
    module.default({ registerProvider: () => undefined, on: (event: string, handler: (...args: any[]) => any) => handlers.set(event, handler) });
    return handlers;
  } finally { if (previous === undefined) delete process.env.PI_MODEL_GATEWAY_URL; else process.env.PI_MODEL_GATEWAY_URL = previous; }
}

test("the gateway extension installs each prompt's credential from the host before the agent starts", async () => {
  const previous = process.env.OPENAI_API_KEY;
  try {
    const handler = (await loadGatewayExtension()).get("before_agent_start");
    assert.ok(handler, "model-gateway must ask for credentials before each prompt");
    const asked: unknown[][] = [];
    const ctx = (reply: string | undefined) => ({ ui: { input: async (...args: unknown[]) => { asked.push(args); return reply; } } });
    process.env.OPENAI_API_KEY = CREDENTIAL_PLACEHOLDER;
    await handler!({}, ctx(JSON.stringify({ OPENAI_API_KEY: "exec-1-token" })));
    assert.equal(process.env.OPENAI_API_KEY, "exec-1-token");
    assert.equal(asked[0][0], CREDENTIAL_DIALOG);
    await handler!({}, ctx(JSON.stringify({ OPENAI_API_KEY: "exec-2-token" })));
    assert.equal(process.env.OPENAI_API_KEY, "exec-2-token");
    // A refused or malformed reply leaves no earlier execution's token behind and fails the prompt.
    await assert.rejects(handler!({}, ctx(undefined)), /credential/i);
    assert.equal(process.env.OPENAI_API_KEY, CREDENTIAL_PLACEHOLDER);
    process.env.OPENAI_API_KEY = "exec-3-token";
    await assert.rejects(handler!({}, ctx(JSON.stringify({ PATH: "/evil" }))), /credential/i);
    assert.equal(process.env.OPENAI_API_KEY, CREDENTIAL_PLACEHOLDER);
  } finally { if (previous === undefined) delete process.env.OPENAI_API_KEY; else process.env.OPENAI_API_KEY = previous; }
});

test("the bridge answers one credential dialog per prompt with that execution's tokens and never records them", () => {
  const lines: string[] = []; const commands: unknown[] = []; const events: PublicRunEvent[] = [];
  let issued = 0;
  const bridge = new PiRpcBridge({
    writeLine: (line) => lines.push(line), onEvent: (event) => events.push(event), onCommand: (record) => commands.push(record),
    credentials: () => ({ OPENAI_API_KEY: `token-${++issued}` }),
  });
  bridge.prompt("run-1", "first");
  bridge.feed(dialog("c1"));
  bridge.feed(dialog("c2"));
  bridge.prompt("run-2", "second");
  bridge.feed(dialog("c3"));
  const replies = lines.map((line) => JSON.parse(line)).filter((record) => record.type === "extension_ui_response");
  assert.deepEqual(replies, [
    { type: "extension_ui_response", id: "c1", value: JSON.stringify({ OPENAI_API_KEY: "token-1" }) },
    { type: "extension_ui_response", id: "c2", cancelled: true },
    { type: "extension_ui_response", id: "c3", value: JSON.stringify({ OPENAI_API_KEY: "token-2" }) },
  ]);
  assert.doesNotMatch(JSON.stringify(commands), /token-\d/);
  assert.doesNotMatch(JSON.stringify(events), /token-\d/);
});

test("without a credential source the bridge refuses the dialog", () => {
  const lines: string[] = [];
  const bridge = new PiRpcBridge({ writeLine: (line) => lines.push(line), onEvent: () => undefined });
  bridge.prompt("run-1", "first");
  bridge.feed(dialog("c1"));
  assert.deepEqual(JSON.parse(lines.at(-1)!), { type: "extension_ui_response", id: "c1", cancelled: true });
});

test("the container gets a placeholder; the execution's tokens reach Pi only through the credential dialog", async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "fake-docker-cred-"));
  const files = { args: path.join(dir, "args"), env: path.join(dir, "env"), reply: path.join(dir, "reply") };
  await writeFile(path.join(dir, "docker"), [
    "#!/bin/sh",
    `if [ "$1" = run ]; then echo "$@" > "${files.args}"; env > "${files.env}"`,
    `  printf '%s\\n' '${dialog("cred-1").trim()}'; read -r line; printf '%s' "$line" > "${files.reply}"; exit 0; fi`,
    "exit 1",
  ].join("\n"));
  await chmod(path.join(dir, "docker"), 0o755);
  const previousPath = process.env.PATH;
  process.env.PATH = `${dir}:${previousPath}`;
  const piProcess = new DockerPiProcess({
    name: "pi-session-cred", image: "pi-agent@sha256:abc", network: "pi-model-only", worktreePath: "/w", gitMetadataPath: "/repo/.git",
    sessionStatePath: dir, gatewayUrl: "http://model-gateway:8080", modelRef: "openai/gpt-5.6-luna",
    extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"],
  }, (runId) => `model-token-for-${runId}`, () => undefined, undefined, (runId) => `classifier-token-for-${runId}`);
  try {
    piProcess.prompt("run-cred-1", "fix it");
    let reply = "";
    for (let i = 0; i < 300 && !reply; i++) {
      reply = await readFile(files.reply, "utf8").catch(() => "");
      if (!reply) await new Promise((resolve) => setTimeout(resolve, 10));
    }
    assert.deepEqual(JSON.parse(reply), { type: "extension_ui_response", id: "cred-1",
      value: JSON.stringify({ OPENAI_API_KEY: "model-token-for-run-cred-1", OPENROUTER_API_KEY: "classifier-token-for-run-cred-1" }) });
    const args = await readFile(files.args, "utf8"); const env = await readFile(files.env, "utf8");
    assert.match(args, new RegExp(`--env OPENAI_API_KEY=${CREDENTIAL_PLACEHOLDER} `));
    assert.match(args, new RegExp(`--env OPENROUTER_API_KEY=${CREDENTIAL_PLACEHOLDER} `));
    assert.doesNotMatch(args + env, /token-for-/);
  } finally {
    await piProcess.close();
    process.env.PATH = previousPath;
    await rm(dir, { recursive: true, force: true });
  }
});
