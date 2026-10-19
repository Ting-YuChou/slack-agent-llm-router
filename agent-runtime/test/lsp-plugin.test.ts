import assert from "node:assert/strict";
import { test } from "node:test";

import lspPlugin, { DocumentVersionTracker, waitForDiagnostics } from "../src/lsp-plugin.js";

function fakeDiagnosticsClient() {
  const handlers = new Set<(value: unknown) => void>();
  return {
    handlers,
    onNotification: (_method: string, handler: (value: unknown) => void) => {
      handlers.add(handler);
      return () => handlers.delete(handler);
    },
    publish: (value: unknown) => [...handlers].forEach((handler) => handler(value)),
  };
}

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

test("LSP diagnostics timeout is explicit instead of an empty clean result", async () => {
  const client = fakeDiagnosticsClient();
  // waitForDiagnostics unrefs its timer; keep the event loop alive while awaiting it.
  const keepAlive = setTimeout(() => undefined, 1_000);
  try {
    const outcome = await waitForDiagnostics(client as never, "file:///w/a.ts", 10);
    assert.deepEqual(outcome, { status: "timeout", timeoutMs: 10 });
    assert.equal(client.handlers.size, 0);
  } finally {
    clearTimeout(keepAlive);
  }
});

test("LSP diagnostics resolve with the published list for the requested document only", async () => {
  const client = fakeDiagnosticsClient();
  const pending = waitForDiagnostics(client as never, "file:///w/a.ts", 1_000);
  client.publish({ uri: "file:///w/other.ts", diagnostics: [{ message: "other" }] });
  client.publish({ uri: "file:///w/a.ts", diagnostics: [] });
  assert.deepEqual(await pending, { status: "received", diagnostics: [] });
  assert.equal(client.handlers.size, 0);
});
