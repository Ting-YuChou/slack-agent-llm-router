import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { test } from "node:test";
import { createSanitizedMirror } from "../src/codegraph-lazy-proxy.js";
const exec = promisify(execFile);

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
