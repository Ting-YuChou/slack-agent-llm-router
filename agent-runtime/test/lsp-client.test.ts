import assert from "node:assert/strict";
import { access, mkdtemp, mkdir, realpath, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

import {
  LspClient,
  LspMessageParser,
  encodeLspMessage,
  languageServerForPath,
  resolveWorkspaceFile,
  typescriptServerInitializationOptions,
} from "../src/lsp-client.js";

test("LSP language routing is fixed to the bundled TypeScript and Python servers", () => {
  assert.deepEqual(languageServerForPath("src/a.ts"), {
    command: "typescript-language-server",
    args: ["--stdio"],
    languageId: "typescript",
  });
  assert.deepEqual(languageServerForPath("src/a.py"), {
    command: "pyright-langserver",
    args: ["--stdio"],
    languageId: "python",
  });
  assert.throws(() => languageServerForPath("src/a.rb"), /unsupported/i);
});

test("LSP file resolution rejects traversal and symlink escapes", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "pi-lsp-root-"));
  const outside = await mkdtemp(path.join(tmpdir(), "pi-lsp-outside-"));
  await mkdir(path.join(root, "src"));
  await writeFile(path.join(root, "src", "a.ts"), "export const a = 1;\n");
  await writeFile(path.join(outside, "secret.ts"), "secret\n");
  await writeFile(path.join(root, ".env"), "TOKEN=secret\n");
  await symlink(path.join(outside, "secret.ts"), path.join(root, "src", "escape.ts"));

  assert.equal(await resolveWorkspaceFile(root, "src/a.ts"), path.join(await realpath(root), "src", "a.ts"));
  await assert.rejects(resolveWorkspaceFile(root, "../outside.ts"), /outside/i);
  await assert.rejects(resolveWorkspaceFile(root, "src/escape.ts"), /outside/i);
  await assert.rejects(resolveWorkspaceFile(root, ".env"), /protected/i);
});

test("LSP framing parser handles fragmented Content-Length messages", () => {
  const payload = { jsonrpc: "2.0", id: 1, result: { ok: true } };
  const framed = encodeLspMessage(payload);
  const parser = new LspMessageParser();

  assert.deepEqual(parser.feed(framed.subarray(0, 12)), []);
  assert.deepEqual(parser.feed(framed.subarray(12)), [payload]);
});

test("TypeScript LSP pins tsserver outside the untrusted workspace", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "pi-lsp-untrusted-tsserver-"));
  const marker = path.join(root, "workspace-tsserver-was-loaded");
  const source = path.join(root, "sample.ts");
  const untrustedTsserver = path.join(root, "node_modules", "typescript", "lib", "tsserver.js");
  await mkdir(path.dirname(untrustedTsserver), { recursive: true });
  await writeFile(source, "export const answer = 42;\n");
  await writeFile(
    untrustedTsserver,
    `require("node:fs").writeFileSync(${JSON.stringify(marker)}, "loaded");\n`,
  );
  const options = typescriptServerInitializationOptions(root);

  assert.match(options.tsserver.path, /node_modules\/typescript\/lib\/tsserver\.js$/);
  assert.equal(path.relative(root, options.tsserver.path).startsWith(".."), true);
  const client = new LspClient(root, languageServerForPath(source));
  try {
    await client.start();
    await assert.rejects(access(marker));
  } finally {
    await client.close();
  }
});

for (const fixture of [
  {
    name: "TypeScript",
    file: "sample.ts",
    text: "export const answer = 42;\nconsole.log(answer);\n",
    position: { line: 1, character: 14 },
  },
  {
    name: "Python",
    file: "sample.py",
    text: "def answer():\n    return 42\n\nprint(answer())\n",
    position: { line: 3, character: 7 },
  },
]) {
  test(`${fixture.name} language server resolves a real definition`, async () => {
    const root = await mkdtemp(path.join(tmpdir(), "pi-lsp-live-"));
    const file = path.join(root, fixture.file);
    await writeFile(file, fixture.text);
    const spec = languageServerForPath(file);
    const client = new LspClient(root, spec);
    try {
      await client.start();
      const uri = new URL(`file://${file}`).href;
      client.notify("textDocument/didOpen", {
        textDocument: { uri, languageId: spec.languageId, version: 1, text: fixture.text },
      });
      const result = await client.request("textDocument/definition", {
        textDocument: { uri },
        position: fixture.position,
      });
      assert.ok(result, `${fixture.name} did not return a definition`);
      assert.match(JSON.stringify(result), /sample\.(?:ts|py)/);

      await client.close();
      assert.equal(client.isRunning(), false);
      await client.start();
      assert.equal(client.isRunning(), true);
    } finally {
      await client.close();
    }
  });
}
