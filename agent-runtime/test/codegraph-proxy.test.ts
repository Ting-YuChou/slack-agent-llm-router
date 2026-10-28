import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { execFile, spawn } from "node:child_process";
import { once } from "node:events";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { test } from "node:test";
import { createSanitizedMirror, TOOL_CATALOG } from "../src/codegraph-lazy-proxy.js";
import { MCP_SERVERS } from "../src/mcp-config.js";
const exec = promisify(execFile);
const proxyModule = fileURLToPath(new URL("../src/codegraph-lazy-proxy.js", import.meta.url));

/**
 * Runs the proxy over a one-file worktree with a fake cgc. The fake logs each index and server start, and any request its
 * server receives before the MCP handshake ends; its index takes 200 ms and its server answers initialize after 200 ms.
 */
async function proxyWithFakeCgc(options: { failFirstIndex?: boolean } = {}) {
  const root = await mkdtemp(path.join(tmpdir(), "cgc-proxy-"));
  const worktree = path.join(root, "worktree"); const log = path.join(root, "cgc.log"); const cgc = path.join(root, "cgc.cjs");
  await mkdir(worktree);
  await exec("git", ["init", "-q"], { cwd: worktree });
  await writeFile(path.join(worktree, "main.ts"), "export const answer = 42;\n");
  await writeFile(log, "");
  await writeFile(cgc, `#!${process.execPath}
const fs = require("node:fs");
const log = (line) => fs.appendFileSync(${JSON.stringify(log)}, line + "\\n");
if (process.argv[2] === "index") {
  const failed = ${options.failFirstIndex === true} && !fs.readFileSync(${JSON.stringify(log)}, "utf8").includes("index");
  log("index");
  setTimeout(() => process.exit(failed ? 1 : 0), 200);
} else {
  log("start");
  let buffer = ""; let initialized = false;
  const answer = (message) => process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id: message.id, result: { content: [{ type: "text", text: message.params?.name ?? message.method }] } }) + "\\n");
  process.stdin.on("data", (chunk) => {
    buffer += chunk;
    for (let index = buffer.indexOf("\\n"); index >= 0; index = buffer.indexOf("\\n")) {
      const message = JSON.parse(buffer.slice(0, index)); buffer = buffer.slice(index + 1);
      if (message.method === "notifications/initialized") initialized = true;
      else if (message.method === "initialize") setTimeout(() => answer(message), 200);
      else { if (!initialized) log(message.method + " before initialized"); answer(message); }
    }
  });
}
`, { mode: 0o755 });
  const child = spawn(process.execPath, [proxyModule], {
    env: { ...process.env, CGC_COMMAND: cgc, CGC_RUN_STATE: path.join(root, "state"), PI_AGENT_WORKTREE: worktree }, stdio: ["pipe", "pipe", "inherit"],
  });
  const waiting = new Map<number, (response: any) => void>();
  let output = ""; let nextId = 1;
  child.stdout.on("data", (chunk: Buffer) => {
    output += chunk.toString("utf8");
    for (let index = output.indexOf("\n"); index >= 0; index = output.indexOf("\n")) {
      const response = JSON.parse(output.slice(0, index)); output = output.slice(index + 1);
      waiting.get(response.id)?.(response);
    }
  });
  return {
    call: (name: string) => new Promise<any>((resolve) => {
      const id = nextId++; waiting.set(id, resolve);
      child.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method: "tools/call", params: { name, arguments: {} } }) + "\n");
    }),
    log: async () => (await readFile(log, "utf8")).split("\n").filter(Boolean),
    close: async () => {
      if (child.exitCode === null) { child.kill("SIGTERM"); await once(child, "exit"); }
      await rm(root, { recursive: true, force: true });
    },
  };
}

test("sanitized mirror includes current source and excludes secrets, binary, symlink and ignored files", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "cgc-source-"));
  const state = await mkdtemp(path.join(tmpdir(), "cgc-state-"));
  try {
    await exec("git", ["init", "-q"], { cwd: root });
    await writeFile(path.join(root, ".gitignore"), "ignored.ts\n");
    await writeFile(path.join(root, "main.ts"), "export const current = 2;\n");
    await writeFile(path.join(root, ".env.local"), "SECRET=x\n");
    await writeFile(path.join(root, "private.pem"), "PRIVATE KEY\n");
    await writeFile(path.join(root, "binary.ts"), Buffer.from([0, 1, 2]));
    await writeFile(path.join(root, "ignored.ts"), "ignored\n");
    await symlink(path.join(root, "main.ts"), path.join(root, "linked.ts"));
    await exec("git", ["add", ".gitignore", "main.ts"], { cwd: root });
    const mirror = await createSanitizedMirror(root, state);
    assert.equal(await readFile(path.join(mirror, "main.ts"), "utf8"), "export const current = 2;\n");
    for (const name of [".env.local", "private.pem", "binary.ts", "ignored.ts", "linked.ts"]) {
      await assert.rejects(readFile(path.join(mirror, name)));
    }
  } finally {
    await rm(root, { recursive: true, force: true }); await rm(state, { recursive: true, force: true });
  }
});

test("sanitized mirror enforces total limits", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "cgc-limit-"));
  const state = await mkdtemp(path.join(tmpdir(), "cgc-limit-state-"));
  try {
    await exec("git", ["init", "-q"], { cwd: root });
    await writeFile(path.join(root, "large.ts"), "x".repeat(20));
    await exec("git", ["add", "large.ts"], { cwd: root });
    await assert.rejects(createSanitizedMirror(root, state, { maxFileBytes: 10, maxTotalBytes: 100, maxFiles: 10 }), /limit/);
  } finally { await rm(root, { recursive: true, force: true }); await rm(state, { recursive: true, force: true }); }
});

test("the CodeGraph child process gets a minimal environment without run credentials", async () => {
  const { codegraphEnv } = await import("../src/codegraph-lazy-proxy.js");
  const env = codegraphEnv({ PATH: "/usr/bin", OPENAI_API_KEY: "t", PI_AGENT_MCP_SERVERS_JSON: "[]", GITHUB_TOKEN: "t", LANG: "C.UTF-8" }, "/state", "/mirror");
  assert.deepEqual(env, { PATH: "/usr/bin", LANG: "C.UTF-8", HOME: "/state", CGC_ALLOWED_ROOTS: "/mirror", CGC_EMBEDDED_BUFFER_POOL_MB: "384" });
});

test("the CodeGraph child process keeps the image's baked-in tree-sitter parsers", async () => {
  const { codegraphEnv } = await import("../src/codegraph-lazy-proxy.js");
  // HOME is the run's empty state root, so without this the parsers resolve to a cache the offline container cannot fill.
  const env = codegraphEnv({ PATH: "/usr/bin", TREE_SITTER_LANGUAGE_PACK_CACHE_DIR: "/usr/local/share/tree-sitter-language-pack" }, "/state", "/mirror");
  assert.equal(env.TREE_SITTER_LANGUAGE_PACK_CACHE_DIR, "/usr/local/share/tree-sitter-language-pack");
});

test("the CodeGraph catalog offers cgc's own arguments, except the repo_path and graph_name the proxy owns", () => {
  assert.deepEqual(TOOL_CATALOG.map((tool) => tool.name), MCP_SERVERS.codegraph.codemodeTools);
  const schema = (name: string) => TOOL_CATALOG.find((tool) => tool.name === name)!.inputSchema as { properties?: Record<string, any>; required?: string[] };
  // Offered no arguments, the model guessed them: every analyze_code_relationships call failed and spent the call budget.
  assert.deepEqual(schema("analyze_code_relationships").required, ["query_type", "target"]);
  assert.ok(schema("analyze_code_relationships").properties?.query_type.enum.includes("find_callers"));
  assert.deepEqual(schema("find_code").required, ["query"]);
  assert.deepEqual(schema("calculate_cyclomatic_complexity").required, ["function_name"]);
  for (const tool of TOOL_CATALOG) {
    assert.ok(schema(tool.name).properties, tool.name);
    assert.ok(!("repo_path" in schema(tool.name).properties!) && !("graph_name" in schema(tool.name).properties!), tool.name);
  }
});

test("concurrent CodeGraph calls share one index and one cgc server, and wait for its handshake", { timeout: 30_000 }, async () => {
  const proxy = await proxyWithFakeCgc();
  try {
    // Each concurrent first call used to index the mirror and start its own server on the one graph database; all of them timed out.
    const first = [proxy.call("get_repository_stats"), proxy.call("find_dead_code")];
    while (!(await proxy.log()).includes("start")) await new Promise((resolve) => setTimeout(resolve, 10));
    // The server is up but has not answered initialize yet.
    const late = proxy.call("list_indexed_repositories");
    const responses = await Promise.all([...first, late]);
    assert.deepEqual(responses.map((response) => response.result?.content?.[0]?.text), ["get_repository_stats", "find_dead_code", "list_indexed_repositories"], JSON.stringify(responses));
    assert.deepEqual(await proxy.log(), ["index", "start"]);
  } finally { await proxy.close(); }
});

test("a failed CodeGraph start fails every call waiting on it, and the next call starts again", { timeout: 30_000 }, async () => {
  const proxy = await proxyWithFakeCgc({ failFirstIndex: true });
  try {
    const failed = await Promise.all([proxy.call("get_repository_stats"), proxy.call("find_dead_code")]);
    assert.ok(failed.every((response) => response.error && !response.result), JSON.stringify(failed));
    assert.deepEqual(await proxy.log(), ["index"]);
    const retried = await proxy.call("find_dead_code");
    assert.equal(retried.result?.content?.[0]?.text, "find_dead_code", JSON.stringify(retried));
    assert.deepEqual(await proxy.log(), ["index", "index", "start"]);
  } finally { await proxy.close(); }
});
