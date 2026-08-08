import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { realpath } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const MAX_LSP_MESSAGE_BYTES = 4 * 1024 * 1024;
const DEFAULT_REQUEST_TIMEOUT_MS = 10_000;

export interface LanguageServerSpec {
  command: "typescript-language-server" | "pyright-langserver";
  args: ["--stdio"];
  languageId: "typescript" | "typescriptreact" | "javascript" | "javascriptreact" | "python";
}

export function languageServerForPath(filePath: string): LanguageServerSpec {
  const extension = path.extname(filePath).toLowerCase();
  if (extension === ".ts") return { command: "typescript-language-server", args: ["--stdio"], languageId: "typescript" };
  if (extension === ".tsx") return { command: "typescript-language-server", args: ["--stdio"], languageId: "typescriptreact" };
  if (extension === ".js") return { command: "typescript-language-server", args: ["--stdio"], languageId: "javascript" };
  if (extension === ".jsx") return { command: "typescript-language-server", args: ["--stdio"], languageId: "javascriptreact" };
  if (extension === ".py") return { command: "pyright-langserver", args: ["--stdio"], languageId: "python" };
  throw new Error(`Unsupported LSP file type: ${extension || "none"}`);
}

export function typescriptServerInitializationOptions(_workspaceRoot: string): {
  tsserver: { path: string };
} {
  return {
    tsserver: {
      path: path.resolve(import.meta.dirname, "../../node_modules/typescript/lib/tsserver.js"),
    },
  };
}

export async function resolveWorkspaceFile(root: string, inputPath: string): Promise<string> {
  if (!inputPath.trim() || inputPath.includes("\0")) throw new Error("LSP path is invalid");
  const rootPath = await realpath(root);
  const requested = path.resolve(rootPath, inputPath);
  const relative = path.relative(rootPath, requested);
  if (relative.startsWith("..") || path.isAbsolute(relative)) throw new Error("LSP path is outside the workspace");
  const segments = relative.split(path.sep);
  const basename = segments.at(-1) ?? "";
  if (segments.includes(".git") || /^\.env(?:\.|$)/.test(basename) || /(?:id_rsa|id_ed25519|\.pem|\.key)$/i.test(basename)) {
    throw new Error("LSP path is protected");
  }
  const resolved = await realpath(requested);
  const resolvedRelative = path.relative(rootPath, resolved);
  if (resolvedRelative.startsWith("..") || path.isAbsolute(resolvedRelative)) {
    throw new Error("LSP path resolves outside the workspace");
  }
  return resolved;
}

export function encodeLspMessage(message: unknown): Buffer {
  const body = Buffer.from(JSON.stringify(message), "utf8");
  if (body.length > MAX_LSP_MESSAGE_BYTES) throw new Error("LSP message is too large");
  return Buffer.concat([Buffer.from(`Content-Length: ${body.length}\r\n\r\n`, "ascii"), body]);
}

export class LspMessageParser {
  private buffer = Buffer.alloc(0);

  feed(chunk: Buffer): unknown[] {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    if (this.buffer.length > MAX_LSP_MESSAGE_BYTES * 2) throw new Error("LSP stream buffer is too large");
    const messages: unknown[] = [];
    while (true) {
      const headerEnd = this.buffer.indexOf("\r\n\r\n");
      if (headerEnd < 0) break;
      const header = this.buffer.subarray(0, headerEnd).toString("ascii");
      const match = /(?:^|\r\n)Content-Length:\s*(\d+)(?:\r\n|$)/i.exec(header);
      if (!match) throw new Error("LSP message is missing Content-Length");
      const length = Number(match[1]);
      if (!Number.isSafeInteger(length) || length < 0 || length > MAX_LSP_MESSAGE_BYTES) {
        throw new Error("LSP message length is invalid");
      }
      const bodyStart = headerEnd + 4;
      if (this.buffer.length < bodyStart + length) break;
      const body = this.buffer.subarray(bodyStart, bodyStart + length).toString("utf8");
      this.buffer = this.buffer.subarray(bodyStart + length);
      messages.push(JSON.parse(body));
    }
    return messages;
  }
}

type NotificationHandler = (params: unknown) => void;

export class LspClient {
  private child?: ChildProcessWithoutNullStreams;
  private nextId = 1;
  private readonly pending = new Map<number, {
    resolve: (value: unknown) => void;
    reject: (error: Error) => void;
    timeout: ReturnType<typeof setTimeout>;
  }>();
  private readonly notificationHandlers = new Map<string, Set<NotificationHandler>>();

  constructor(
    private readonly root: string,
    private readonly spec: LanguageServerSpec,
  ) {}

  async start(): Promise<void> {
    if (this.child) return;
    const parser = new LspMessageParser();
    const child = spawn(this.spec.command, this.spec.args, {
      cwd: this.root,
      env: {
        PATH: `${path.resolve(import.meta.dirname, "../../node_modules/.bin")}${path.delimiter}${process.env.PATH ?? ""}`,
        HOME: process.env.HOME ?? "/tmp/pi-home",
      },
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.child = child;
    child.stderr.resume();
    child.stdout.on("data", (chunk: Buffer) => {
      if (this.child !== child) return;
      try {
        for (const message of parser.feed(chunk)) this.handleMessage(message);
      } catch (error) {
        if (this.child !== child) return;
        this.failPending(error instanceof Error ? error : new Error("LSP stream failed"));
        child.kill("SIGTERM");
      }
    });
    child.on("error", (error) => {
      if (this.child === child) this.failPending(error);
    });
    child.on("exit", () => {
      if (this.child !== child) return;
      this.child = undefined;
      this.failPending(new Error("Language server exited"));
    });
    await this.request("initialize", {
      processId: process.pid,
      rootUri: pathToFileURL(this.root).href,
      capabilities: {
        textDocument: { publishDiagnostics: {}, definition: {}, references: {}, hover: {}, documentSymbol: {} },
        workspace: { symbol: {} },
      },
      workspaceFolders: [{ uri: pathToFileURL(this.root).href, name: path.basename(this.root) }],
      ...(this.spec.command === "typescript-language-server"
        ? { initializationOptions: typescriptServerInitializationOptions(this.root) }
        : {}),
    });
    this.notify("initialized", {});
  }

  isRunning(): boolean {
    return Boolean(this.child && this.child.exitCode === null && !this.child.killed);
  }

  request(method: string, params: unknown, timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS): Promise<unknown> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`LSP request timed out: ${method}`));
      }, timeoutMs);
      timeout.unref();
      this.pending.set(id, { resolve, reject, timeout });
      try {
        this.write({ jsonrpc: "2.0", id, method, params });
      } catch (error) {
        clearTimeout(timeout);
        this.pending.delete(id);
        reject(error instanceof Error ? error : new Error("LSP request could not be sent"));
      }
    });
  }

  notify(method: string, params: unknown): void {
    this.write({ jsonrpc: "2.0", method, params });
  }

  onNotification(method: string, handler: NotificationHandler): () => void {
    const handlers = this.notificationHandlers.get(method) ?? new Set<NotificationHandler>();
    handlers.add(handler);
    this.notificationHandlers.set(method, handlers);
    return () => handlers.delete(handler);
  }

  async close(): Promise<void> {
    const child = this.child;
    if (!child) return;
    try { await this.request("shutdown", null, 2_000); } catch { /* terminate below */ }
    try { this.notify("exit", null); } catch { /* terminate below */ }
    child.kill("SIGTERM");
    this.child = undefined;
  }

  private write(message: unknown): void {
    if (!this.child || this.child.exitCode !== null) throw new Error("Language server is not running");
    this.child.stdin.write(encodeLspMessage(message));
  }

  private handleMessage(value: unknown): void {
    if (!isRecord(value)) return;
    if (typeof value.id === "number" && ("result" in value || "error" in value) && !value.method) {
      const pending = this.pending.get(value.id);
      if (!pending) return;
      this.pending.delete(value.id);
      clearTimeout(pending.timeout);
      if (value.error) pending.reject(new Error("Language server request failed"));
      else pending.resolve(value.result);
      return;
    }
    if (typeof value.method === "string" && typeof value.id === "number") {
      this.write({
        jsonrpc: "2.0",
        id: value.id,
        error: { code: -32601, message: "Server-initiated LSP requests are disabled" },
      });
      return;
    }
    if (typeof value.method === "string") {
      for (const handler of this.notificationHandlers.get(value.method) ?? []) handler(value.params);
    }
  }

  private failPending(error: Error): void {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timeout);
      pending.reject(error);
    }
    this.pending.clear();
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
