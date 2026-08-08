import { readFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import { LspClient, languageServerForPath, resolveWorkspaceFile } from "./lsp-client.js";

const MAX_DOCUMENT_BYTES = 256 * 1024;
const MAX_RESULT_CHARS = 60_000;
const ACTIONS = ["diagnostics", "definition", "references", "hover", "document_symbols", "workspace_symbols"] as const;

export class DocumentVersionTracker {
  private readonly byServer = new Map<string, Map<string, number>>();

  next(server: string, uri: string): number {
    const versions = this.byServer.get(server) ?? new Map<string, number>();
    const version = (versions.get(uri) ?? 0) + 1;
    versions.set(uri, version);
    this.byServer.set(server, versions);
    return version;
  }

  clearServer(server: string): void {
    this.byServer.delete(server);
  }

  clear(): void {
    this.byServer.clear();
  }
}

export default function lspPlugin(pi: ExtensionAPI) {
  const clients = new Map<string, LspClient>();
  const versions = new DocumentVersionTracker();

  pi.registerTool({
    name: "lsp",
    label: "Language Server",
    description: "Read-only LSP diagnostics, definition, references, hover, document symbols, and workspace symbols for TypeScript, JavaScript, and Python.",
    promptSnippet: "Use lsp for compiler-grade symbol navigation and diagnostics before editing code.",
    promptGuidelines: ["LSP is read-only; use edit/write separately for changes."],
    executionMode: "sequential",
    parameters: Type.Object({
      action: Type.Union(ACTIONS.map((action) => Type.Literal(action))),
      path: Type.String({ description: "Workspace-relative source file used to select and query the language server." }),
      line: Type.Optional(Type.Integer({ minimum: 1, description: "One-based line number." })),
      character: Type.Optional(Type.Integer({ minimum: 1, description: "One-based character offset." })),
      query: Type.Optional(Type.String({ maxLength: 200, description: "Symbol query for workspace_symbols." })),
    }),
    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      try {
        if (signal?.aborted) throw new Error("LSP request was cancelled");
        const absolutePath = await resolveWorkspaceFile(ctx.cwd, params.path);
        const text = await readFile(absolutePath, "utf8");
        if (Buffer.byteLength(text, "utf8") > MAX_DOCUMENT_BYTES) throw new Error("LSP document exceeds 256 KiB");
        const spec = languageServerForPath(absolutePath);
        const key = spec.command;
        let client = clients.get(key);
        const restarting = Boolean(client && !client.isRunning());
        if (!client) {
          client = new LspClient(ctx.cwd, spec);
          clients.set(key, client);
        }
        await client.start();
        if (restarting) versions.clearServer(key);
        const uri = pathToFileURL(absolutePath).href;
        const version = versions.next(key, uri);

        let diagnostics: Promise<unknown> | undefined;
        if (params.action === "diagnostics") diagnostics = waitForDiagnostics(client, uri);
        if (version === 1) {
          client.notify("textDocument/didOpen", {
            textDocument: { uri, languageId: spec.languageId, version, text },
          });
        } else {
          client.notify("textDocument/didChange", {
            textDocument: { uri, version },
            contentChanges: [{ text }],
          });
        }

        const position = {
          line: Math.max(0, (params.line ?? 1) - 1),
          character: Math.max(0, (params.character ?? 1) - 1),
        };
        let result: unknown;
        switch (params.action) {
          case "diagnostics":
            result = await diagnostics;
            break;
          case "definition":
            result = await client.request("textDocument/definition", { textDocument: { uri }, position });
            break;
          case "references":
            result = await client.request("textDocument/references", {
              textDocument: { uri }, position, context: { includeDeclaration: true },
            });
            break;
          case "hover":
            result = await client.request("textDocument/hover", { textDocument: { uri }, position });
            break;
          case "document_symbols":
            result = await client.request("textDocument/documentSymbol", { textDocument: { uri } });
            break;
          case "workspace_symbols":
            result = await client.request("workspace/symbol", { query: params.query ?? "" });
            break;
        }
        const safeResult = sanitizeLspResult(result, pathToFileURL(path.resolve(ctx.cwd)).href);
        const rendered = JSON.stringify(safeResult, null, 2);
        return {
          content: [{ type: "text", text: rendered.length > MAX_RESULT_CHARS ? `${rendered.slice(0, MAX_RESULT_CHARS)}\n…truncated` : rendered }],
          details: { action: params.action, path: params.path, language: spec.languageId },
        };
      } catch (error) {
        const message = error instanceof Error ? error.message : "LSP request failed";
        return {
          content: [{ type: "text", text: `LSP error: ${message}` }],
          details: { action: params.action, path: params.path },
          isError: true,
        };
      }
    },
  });

  pi.on("session_shutdown", async () => {
    await Promise.allSettled([...clients.values()].map((client) => client.close()));
    clients.clear();
    versions.clear();
  });
}

function waitForDiagnostics(client: LspClient, uri: string): Promise<unknown> {
  return new Promise((resolve) => {
    let settled = false;
    const dispose = client.onNotification("textDocument/publishDiagnostics", (value) => {
      if (!isRecord(value) || value.uri !== uri || settled) return;
      settled = true;
      clearTimeout(timeout);
      dispose();
      resolve(value.diagnostics ?? []);
    });
    const timeout = setTimeout(() => {
      if (settled) return;
      settled = true;
      dispose();
      resolve([]);
    }, 3_000);
    timeout.unref();
  });
}

function sanitizeLspResult(value: unknown, rootUri: string, depth = 0): unknown {
  if (depth > 8) return "[truncated]";
  if (typeof value === "string") {
    if (value.startsWith("file://") && value !== rootUri && !value.startsWith(`${rootUri}/`)) return "[outside-workspace]";
    return value.length > 4_000 ? `${value.slice(0, 4_000)}…` : value;
  }
  if (Array.isArray(value)) return value.slice(0, 100).map((item) => sanitizeLspResult(item, rootUri, depth + 1));
  if (isRecord(value)) {
    return Object.fromEntries(Object.entries(value).slice(0, 100).map(([key, item]) => [key, sanitizeLspResult(item, rootUri, depth + 1)]));
  }
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
