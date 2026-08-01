import { execFile } from "node:child_process";
import { mkdir } from "node:fs/promises";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { promisify } from "node:util";

import { DockerPiProcess } from "./container-runtime.js";
import { issueGatewayToken } from "./gateway-token.js";
import {
  CodingAgentOrchestrator,
  RuntimeError,
  type RunRecord,
} from "./orchestrator.js";
import { verifyPluginLock } from "./plugin-lock.js";
import { RuntimeStateStore } from "./runtime-state.js";
import { WorktreeManager } from "./worktree-manager.js";

const exec = promisify(execFile);
export const DEFAULT_AGENT_RUNTIME_PORT = 3001;
const DEFAULT_AGENT_RUNTIME_HOST = "127.0.0.1";
const MAX_REQUEST_BYTES = 64_000;

interface ServerOptions {
  orchestrator: CodingAgentOrchestrator;
  token: string;
  health: () => Record<string, unknown>;
  acceptRuns?: () => boolean;
}

export function createAgentHttpServer(options: ServerOptions) {
  return createServer(async (request, response) => {
    try {
      const url = new URL(request.url ?? "/", "http://runtime.local");
      if (request.method === "GET" && url.pathname === "/health") {
        sendJson(response, 200, options.health());
        return;
      }
      if (!authorized(request, options.token)) {
        sendError(response, 401, "unauthorized", "A valid Agent runtime token is required");
        return;
      }
      if (options.acceptRuns && !options.acceptRuns() && request.method === "POST") {
        sendError(response, 502, "runtime_unhealthy", "Agent runtime integrity checks failed");
        return;
      }

      if (request.method === "GET" && url.pathname === "/v1/sessions/lookup") {
        const session = options.orchestrator.lookupSession({
          team_id: requiredString(url.searchParams.get("team_id"), "team_id"),
          channel_id: requiredString(url.searchParams.get("channel_id"), "channel_id"),
          thread_ts: requiredString(url.searchParams.get("thread_ts"), "thread_ts"),
        });
        sendJson(response, 200, { found: Boolean(session), session });
        return;
      }

      if (request.method === "POST" && url.pathname === "/v1/sessions") {
        const body = await readJson(request);
        const result = await options.orchestrator.createSession(body as never);
        sendJson(response, 202, result);
        return;
      }
      const promptMatch = /^\/v1\/sessions\/([^/]+)\/prompts$/.exec(url.pathname);
      if (request.method === "POST" && promptMatch) {
        const body = asRecord(await readJson(request));
        const result = await options.orchestrator.prompt(
          decodeURIComponent(promptMatch[1]),
          requiredString(body.prompt, "prompt"),
          requiredString(body.user_id, "user_id"),
        );
        sendJson(response, 202, result);
        return;
      }
      const closeMatch = /^\/v1\/sessions\/([^/]+)\/close$/.exec(url.pathname);
      if (request.method === "POST" && closeMatch) {
        const body = asRecord(await readJson(request));
        await options.orchestrator.closeSession(decodeURIComponent(closeMatch[1]), requiredString(body.user_id, "user_id"));
        sendJson(response, 202, { status: "closed" });
        return;
      }
      const runMatch = /^\/v1\/runs\/([^/]+)$/.exec(url.pathname);
      if (request.method === "GET" && runMatch) {
        sendJson(response, 200, options.orchestrator.getRun(decodeURIComponent(runMatch[1])));
        return;
      }
      const eventsMatch = /^\/v1\/runs\/([^/]+)\/events$/.exec(url.pathname);
      if (request.method === "GET" && eventsMatch) {
        const runId = decodeURIComponent(eventsMatch[1]);
        options.orchestrator.getRun(runId);
        const after = Math.max(0, Number(url.searchParams.get("after") ?? "0") || 0);
        sendSse(response, options.orchestrator.listEvents(runId, after), after);
        return;
      }
      const decisionMatch = /^\/v1\/runs\/([^/]+)\/decisions$/.exec(url.pathname);
      if (request.method === "POST" && decisionMatch) {
        const body = asRecord(await readJson(request));
        const decision = requiredString(body.decision, "decision");
        if (decision !== "approve" && decision !== "reject") throw new RuntimeError("decision must be approve or reject", "invalid_request", 400);
        await options.orchestrator.decide(
          decodeURIComponent(decisionMatch[1]),
          requiredString(body.approval_id, "approval_id"),
          requiredString(body.user_id, "user_id"),
          decision,
        );
        sendJson(response, 202, { status: "accepted" });
        return;
      }
      const cancelMatch = /^\/v1\/runs\/([^/]+)\/cancel$/.exec(url.pathname);
      if (request.method === "POST" && cancelMatch) {
        const body = asRecord(await readJson(request));
        await options.orchestrator.cancel(decodeURIComponent(cancelMatch[1]), requiredString(body.user_id, "user_id"));
        sendJson(response, 202, { status: "cancelling" });
        return;
      }
      sendError(response, 404, "not_found", "Endpoint not found");
    } catch (error) {
      if (error instanceof RuntimeError) {
        sendError(response, error.statusCode, error.code, error.message);
      } else {
        logEvent("agent_runtime_request_failed", { error_code: "internal_error" });
        sendError(response, 502, "runtime_error", "Agent runtime request failed");
      }
    }
  });
}

export async function createProductionServer() {
  const token = process.env.AGENT_RUNTIME_TOKEN ?? "";
  const gatewaySecret = process.env.MODEL_GATEWAY_SIGNING_SECRET ?? "";
  if (!token || !gatewaySecret) throw new Error("AGENT_RUNTIME_TOKEN and MODEL_GATEWAY_SIGNING_SECRET are required");
  const repoPath = path.resolve(process.env.PI_AGENT_REPO_PATH ?? process.cwd());
  const runtimeRoot = path.resolve(process.env.PI_AGENT_STATE_ROOT ?? path.join(repoPath, ".pi-agent-runtime"));
  const worktreeRoot = path.resolve(process.env.PI_AGENT_WORKTREE_ROOT ?? path.join(repoPath, ".pi-agent-worktrees"));
  const image = process.env.PI_AGENT_IMAGE ?? "slack-pi-agent:0.83.0";
  const network = process.env.PI_AGENT_NETWORK ?? "pi-model-only";
  const gatewayUrl = process.env.PI_MODEL_GATEWAY_URL ?? "http://model-gateway:8080/v1";
  const hostUid = process.getuid?.();
  const hostGid = process.getgid?.();
  if (hostUid === undefined || hostGid === undefined || hostUid === 0) {
    throw new Error("Agent runtime must run as a non-root host user so isolated worktrees remain writable");
  }
  const containerUser = `${hostUid}:${hostGid}`;
  const lockPath = path.resolve(process.env.PI_AGENT_PLUGIN_LOCK ?? path.join(import.meta.dirname, "../../plugins.lock.json"));
  await mkdir(runtimeRoot, { recursive: true });
  await mkdir(worktreeRoot, { recursive: true });
  const actualImageDigest = process.env.PI_AGENT_IMAGE_DIGEST ?? await inspectImageDigest(image);
  const lock = await verifyPluginLock(lockPath, { rootDir: path.dirname(lockPath), actualImageDigest });
  const gitCommon = (await exec("git", ["rev-parse", "--git-common-dir"], { cwd: repoPath, encoding: "utf8" })).stdout.trim();
  const gitMetadataPath = path.resolve(repoPath, gitCommon);
  const worktrees = new WorktreeManager({
    repoPath,
    worktreeRoot,
    baseRef: process.env.PI_AGENT_BASE_REF ?? "HEAD",
  });
  await cleanupOrphanAgentContainers();
  const stateStore = new RuntimeStateStore(path.join(runtimeRoot, "runtime-state.json"));
  const orchestrator = new CodingAgentOrchestrator({
    worktrees,
    stateStore,
    maxActiveSessions: positiveInteger(process.env.PI_AGENT_MAX_CONCURRENCY, 2),
    deadlineMs: positiveInteger(process.env.PI_AGENT_DEADLINE_MS, 15 * 60_000),
    startProcess: async (session, onEvent) => {
      const sessionStatePath = path.join(runtimeRoot, "sessions", session.id);
      await mkdir(sessionStatePath, { recursive: true });
      return new DockerPiProcess(
        {
          name: `pi-session-${session.id.slice(0, 8)}`,
          image,
          network,
          worktreePath: session.worktreePath,
          gitMetadataPath,
          sessionStatePath,
          gatewayUrl,
          extensionPaths: ["/opt/pi/extensions/policy.ts", "/opt/pi/extensions/model-gateway.ts"],
          pluginPaths: lock.plugins.map((plugin) => plugin.container_path),
          toolNames: ["read", "write", "edit", "bash", "grep", "find", "ls", ...lock.plugins.flatMap((plugin) => plugin.enabled_tools)],
          continueSession: session.restored,
          safeCommands: parseSafeCommands(process.env.PI_AGENT_SAFE_COMMANDS),
          user: containerUser,
        },
        (runId) => issueGatewayToken({ runId, model: "gpt-5", expiresAt: Date.now() + 16 * 60_000 }, gatewaySecret),
        onEvent,
      );
    },
  });
  await orchestrator.restore();
  return createAgentHttpServer({
    orchestrator,
    token,
    acceptRuns: () => lock.healthy,
    health: () => ({
      status: lock.healthy ? "healthy" : "unhealthy",
      runtime: "pi-coding-agent",
      version: "0.83.0",
      provider: "openai",
      model: "gpt-5",
      tools: ["read", "write", "edit", "bash", "grep", "find", "ls", ...lock.plugins.flatMap((plugin) => plugin.enabled_tools)],
      plugin_integrity: lock.healthy ? "verified" : "failed",
      errors: lock.healthy ? [] : lock.errors,
    }),
  });
}

async function inspectImageDigest(image: string): Promise<string> {
  try {
    return (await exec("docker", ["image", "inspect", "--format={{.Id}}", image], { encoding: "utf8" })).stdout.trim();
  } catch {
    return "missing";
  }
}

async function cleanupOrphanAgentContainers(): Promise<void> {
  try {
    const output = (await exec("docker", ["ps", "-q", "--filter", "label=slack-pi-agent-session=true"], { encoding: "utf8" })).stdout;
    for (const containerId of output.split("\n").map((value) => value.trim()).filter(Boolean)) {
      await exec("docker", ["stop", "--time", "5", containerId], { encoding: "utf8" });
    }
  } catch {
    // Health/integrity checks will reject runs if Docker is unavailable.
  }
}

function authorized(request: IncomingMessage, token: string): boolean {
  return Boolean(token) && request.headers.authorization === `Bearer ${token}`;
}
async function readJson(request: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of request) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    total += buffer.length;
    if (total > MAX_REQUEST_BYTES) throw new RuntimeError("Request is too large", "invalid_request", 400);
    chunks.push(buffer);
  }
  try { return JSON.parse(Buffer.concat(chunks).toString("utf8")); }
  catch { throw new RuntimeError("Request body must be valid JSON", "invalid_request", 400); }
}
function asRecord(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new RuntimeError("Request body must be an object", "invalid_request", 400);
  return value as Record<string, unknown>;
}
function requiredString(value: unknown, name: string): string {
  if (typeof value !== "string" || !value.trim()) throw new RuntimeError(`${name} is required`, "invalid_request", 400);
  return value.trim();
}
function sendJson(response: ServerResponse, status: number, payload: unknown): void {
  response.writeHead(status, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
  response.end(JSON.stringify(payload));
}
function sendError(response: ServerResponse, status: number, code: string, message: string): void {
  sendJson(response, status, { error: { code, message } });
}
function sendSse(response: ServerResponse, events: Array<Record<string, unknown>>, offset: number): void {
  response.writeHead(200, { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-store", connection: "keep-alive" });
  events.forEach((event, index) => {
    response.write(`id: ${offset + index + 1}\n`);
    response.write(`event: ${String(event.type ?? "event")}\n`);
    response.write(`data: ${JSON.stringify(event)}\n\n`);
  });
  response.end();
}
function positiveInteger(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}
function parseSafeCommands(value: string | undefined): string[] {
  const defaults = ["npm test", "npm run test", "npm run lint", "npm run build", "pytest", "python -m pytest"];
  if (!value) return defaults;
  return value.split("||").map((command) => command.trim()).filter(Boolean);
}
function logEvent(event: string, fields: Record<string, unknown>): void {
  process.stdout.write(`${JSON.stringify({ event, ...fields })}\n`);
}

const entrypoint = process.argv[1] ? pathToFileURL(process.argv[1]).href : undefined;
if (entrypoint === import.meta.url) {
  const host = process.env.PI_AGENT_HOST ?? DEFAULT_AGENT_RUNTIME_HOST;
  const port = positiveInteger(process.env.PI_AGENT_PORT, DEFAULT_AGENT_RUNTIME_PORT);
  const server = await createProductionServer();
  server.listen(port, host, () => logEvent("agent_runtime_started", { host, port }));
}
