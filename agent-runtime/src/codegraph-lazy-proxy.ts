import { spawn, execFile, type ChildProcessWithoutNullStreams } from "node:child_process";
import { lstat, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { promisify } from "node:util";
import { MCP_SERVERS } from "./mcp-config.js";

const exec = promisify(execFile);
const EXCLUDED_PARTS = new Set([".git", ".codegraphcontext", "node_modules", "vendor", "dist", "build", ".next", "coverage", "__pycache__", ".venv", "venv"]);
const SECRET_FILE = /(^|\/)(\.env(?:\..*)?|.*\.(?:pem|key|p12|pfx)|id_(?:rsa|ed25519)|credentials(?:\..*)?)$/i;
const SOURCE_FILE = /\.(?:[cm]?[jt]sx?|py|go|rs|java|kt|kts|rb|php|cs|cpp|cc|cxx|c|h|hpp|swift|scala|sql|sh|bash|zsh|md|json|ya?ml|toml|xml|proto)$/i;

export interface MirrorLimits { maxFiles: number; maxFileBytes: number; maxTotalBytes: number }
const DEFAULT_LIMITS: MirrorLimits = { maxFiles: 20_000, maxFileBytes: 1_048_576, maxTotalBytes: 104_857_600 };

export async function createSanitizedMirror(worktree: string, stateRoot: string, limits: MirrorLimits = DEFAULT_LIMITS): Promise<string> {
  const mirror = path.join(stateRoot, "mirror");
  await rm(mirror, { recursive: true, force: true });
  await mkdir(mirror, { recursive: true, mode: 0o700 });
  const { stdout } = await exec("git", ["ls-files", "-co", "--exclude-standard", "-z"], { cwd: worktree, encoding: "buffer", maxBuffer: 16 * 1024 * 1024 });
  const entries = stdout.toString("utf8").split("\0").filter(Boolean);
  let count = 0; let total = 0;
  for (const relative of entries) {
    const normalized = relative.replaceAll("\\", "/");
    if (path.isAbsolute(relative) || normalized.split("/").some((part) => EXCLUDED_PARTS.has(part)) || SECRET_FILE.test(normalized) || !SOURCE_FILE.test(normalized)) continue;
    const source = path.join(worktree, relative);
    if (!path.resolve(source).startsWith(path.resolve(worktree) + path.sep)) continue;
    const stat = await lstat(source).catch(() => undefined);
    if (!stat?.isFile() || stat.isSymbolicLink()) continue;
    if (stat.size > limits.maxFileBytes) throw new Error("CodeGraph mirror file size limit exceeded");
    const bytes = await readFile(source);
    if (bytes.includes(0)) continue;
    count += 1; total += bytes.length;
    if (count > limits.maxFiles || total > limits.maxTotalBytes) throw new Error("CodeGraph mirror limit exceeded");
    const destination = path.join(mirror, relative);
    await mkdir(path.dirname(destination), { recursive: true, mode: 0o700 });
    await writeFile(destination, bytes, { mode: 0o600 });
  }
  return mirror;
}

const TOOL_CATALOG = MCP_SERVERS.codegraph.codemodeTools.map((name) => ({
  name, description: `Read-only CodeGraph analysis: ${name}`, inputSchema: { type: "object", additionalProperties: true },
}));

class CodeGraphProxy {
  private child?: ChildProcessWithoutNullStreams;
  private buffer = "";
  private nextId = 10_000;
  private pending = new Map<number, (value: any) => void>();
  private stateRoot = process.env.CGC_RUN_STATE ?? path.join("/tmp", `cgc-${process.pid}`);
  private calls = 0;
  private mirror?: string;
  private worktree = process.env.PI_AGENT_WORKTREE ?? process.cwd();

  async handle(message: any): Promise<any> {
    if (message.method === "initialize") return { jsonrpc: "2.0", id: message.id, result: { protocolVersion: "2025-06-18", capabilities: { tools: {} }, serverInfo: { name: "codegraph-lazy", version: "1" } } };
    if (message.method === "notifications/initialized" || message.method === "notifications/cancelled") return undefined;
    if (message.method === "ping") return { jsonrpc: "2.0", id: message.id, result: {} };
    if (message.method === "tools/list") return { jsonrpc: "2.0", id: message.id, result: { tools: TOOL_CATALOG } };
    if (message.method !== "tools/call" || !TOOL_CATALOG.some((tool) => tool.name === message.params?.name)) {
      return { jsonrpc: "2.0", id: message.id, error: { code: -32601, message: "Tool is not allowed" } };
    }
    if (this.calls >= MCP_SERVERS.codegraph.maxCalls) return { jsonrpc: "2.0", id: message.id, error: { code: -32001, message: "CodeGraph call budget is exhausted" } };
    this.calls += 1;
    await this.prepare();
    const params = structuredClone(message.params ?? {});
    if (params.arguments && typeof params.arguments === "object") {
      if ("repo_path" in params.arguments || ["find_code", "analyze_code_relationships", "find_dead_code", "calculate_cyclomatic_complexity", "find_most_complex_functions", "get_repository_stats"].includes(params.name)) {
        params.arguments.repo_path = this.mirror;
      }
      if (typeof params.arguments.path === "string" && params.arguments.path.startsWith(this.worktree)) {
        params.arguments.path = this.mirror + params.arguments.path.slice(this.worktree.length);
      }
    }
    return this.request(message.method, params, message.id);
  }

  private async prepare(): Promise<void> {
    if (this.child) return;
    const mirror = await createSanitizedMirror(this.worktree, this.stateRoot);
    this.mirror = mirror;
    const env = { ...process.env, HOME: this.stateRoot, CGC_ALLOWED_ROOTS: mirror, CGC_EMBEDDED_BUFFER_POOL_MB: "384" };
    const command = process.env.CGC_COMMAND ?? "cgc";
    await exec(command, ["index", "--no-progress", mirror], { cwd: this.stateRoot, env, timeout: 180_000, maxBuffer: 1_048_576 });
    this.child = spawn(command, ["mcp", "start"], { cwd: this.stateRoot, env, stdio: ["pipe", "pipe", "pipe"] });
    this.child.stdout.on("data", (chunk: Buffer) => this.onChildData(chunk));
    this.child.stderr.resume();
    await this.request("initialize", { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "pi-codegraph-proxy", version: "1" } });
    this.child.stdin.write(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }) + "\n");
  }

  private request(method: string, params: unknown, outwardId?: unknown): Promise<any> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error("CodeGraph request timed out")); }, method === "initialize" ? 180_000 : 60_000);
      this.pending.set(id, (value) => {
        clearTimeout(timer);
        if (outwardId !== undefined) value.id = outwardId;
        resolve(value);
      });
      this.child!.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
    });
  }

  private onChildData(chunk: Buffer): void {
    this.buffer += chunk.toString("utf8");
    while (this.buffer.includes("\n")) {
      const index = this.buffer.indexOf("\n");
      const line = this.buffer.slice(0, index); this.buffer = this.buffer.slice(index + 1);
      try { const value = JSON.parse(line); const callback = this.pending.get(value.id); if (callback) { this.pending.delete(value.id); callback(value); } } catch { /* ignore non-JSON diagnostics */ }
    }
  }

  async close(): Promise<void> {
    this.child?.kill("SIGTERM");
    await rm(this.stateRoot, { recursive: true, force: true });
  }
  terminate(): void { this.child?.kill("SIGTERM"); }
}

async function main(): Promise<void> {
  const proxy = new CodeGraphProxy();
  let input = "";
  process.stdin.on("data", (chunk) => {
    input += chunk.toString("utf8");
    while (input.includes("\n")) {
      const index = input.indexOf("\n"); const line = input.slice(0, index); input = input.slice(index + 1);
      if (!line.trim()) continue;
      void Promise.resolve().then(() => proxy.handle(JSON.parse(line))).then((response) => {
        if (response) process.stdout.write(JSON.stringify(response) + "\n");
      }).catch((error) => process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id: safeId(line), error: { code: -32000, message: error instanceof Error ? error.message : "CodeGraph failed" } }) + "\n"));
    }
  });
  for (const signal of ["SIGINT", "SIGTERM"] as const) process.on(signal, () => { void proxy.close().finally(() => process.exit(0)); });
  process.on("exit", () => proxy.terminate());
}
function safeId(line: string): unknown { try { return JSON.parse(line).id ?? null; } catch { return null; } }

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) void main();
