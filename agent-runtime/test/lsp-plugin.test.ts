import assert from "node:assert/strict";
import { test } from "node:test";

import lspPlugin, { DocumentVersionTracker } from "../src/lsp-plugin.js";

test("LSP document versions reset only for the restarted language server", () => {
  const versions = new DocumentVersionTracker();
  assert.equal(versions.next("typescript-language-server", "file:///a.ts"), 1);
  assert.equal(versions.next("pyright-langserver", "file:///a.py"), 1);
  assert.equal(versions.next("pyright-langserver", "file:///a.py"), 2);

  versions.clearServer("typescript-language-server");

  assert.equal(versions.next("typescript-language-server", "file:///a.ts"), 1);
  assert.equal(versions.next("pyright-langserver", "file:///a.py"), 3);
});

test("LSP plugin registers one constrained read-only tool and shutdown cleanup", () => {
  const tools: Array<Record<string, unknown>> = [];
  const events: string[] = [];
  const pi = {
    registerTool: (tool: Record<string, unknown>) => tools.push(tool),
    on: (event: string) => events.push(event),
  };

  lspPlugin(pi as never);

  assert.deepEqual(tools.map((tool) => tool.name), ["lsp"]);
  assert.match(String(tools[0]?.description), /diagnostics.*definition.*references.*hover.*symbols/i);
  assert.match(String(tools[0]?.description), /read-only/i);
  assert.deepEqual(events, ["session_shutdown"]);
});
