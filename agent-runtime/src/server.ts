import { execFile } from "node:child_process";
import { mkdir } from "node:fs/promises";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { promisify } from "node:util";

import { DockerPiProcess } from "./container-runtime.js";
import { AGENT_MODEL_ID, AGENT_REASONING_EFFORT, listAgentModels, resolveAgentModel } from "./agent-model.js";
import { JevRouter, type JevMode } from "./jev-router.js";
import { repositoryFromRemote, parseGitHubRepositories } from "./github-repository.js";
import { issueGatewayToken } from "./gateway-token.js";
import { GITHUB_READ_ONLY_TOOLS, parseMcpMode } from "./mcp-config.js";
import { issueMcpToken } from "./mcp-token.js";
import {
  CodingAgentOrchestrator,
  RuntimeError,
  type RunRecord,
} from "./orchestrator.js";
import { verifyPluginLock } from "./plugin-lock.js";
import { RuntimeStateStore } from "./runtime-state.js";
import { verifySkillLock } from "./skill-lock.js";
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
        if (body.routing_text !== undefined && typeof body.routing_text !== "string") {
          throw new RuntimeError("routing_text must be a string", "invalid_request", 400);
        }
        const result = await options.orchestrator.prompt(
          decodeURIComponent(promptMatch[1]),
          requiredString(body.prompt, "prompt"),
          requiredString(body.user_id, "user_id"),
          typeof body.routing_text === "string" ? body.routing_text : undefined,
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
  const image = process.env.PI_AGENT_IMAGE ?? "slack-pi-agent:1.0.1";
  const network = process.env.PI_AGENT_NETWORK ?? "pi-model-only";
  const gatewayUrl = process.env.PI_MODEL_GATEWAY_URL ?? "http://model-gateway:8080";
  const mcpMode = parseMcpMode(process.env.PI_AGENT_MCP_MODE);
  const mcpGatewayUrl = process.env.PI_AGENT_MCP_GATEWAY_URL ?? "http://mcp-gateway:8090/mcp";
  const mcpSigningSecret = process.env.MCP_GATEWAY_SIGNING_SECRET ?? "";
  const configuredProviders = new Set(
    (process.env.PI_AGENT_CONFIGURED_PROVIDERS ?? "openai")
      .split(",")
      .map((provider) => provider.trim())
      .filter(Boolean),
  );
  const configuredModels = listAgentModels().filter((model) => configuredProviders.has(model.provider));
  const jevMode = process.env.PI_AGENT_JEV_MODE ?? "off";
  if (!["off", "shadow", "on"].includes(jevMode)) throw new Error("PI_AGENT_JEV_MODE must be off, shadow, or on");
  const router = new JevRouter({ mode: jevMode as JevMode, apiKey: process.env.OPENROUTER_API_KEY });
  if (configuredModels.length === 0) throw new Error("At least one Agent model provider must be configured");
  const hostUid = process.getuid?.();
  const hostGid = process.getgid?.();
  if (hostUid === undefined || hostGid === undefined || hostUid === 0) {
    throw new Error("Agent runtime must run as a non-root host user so isolated worktrees remain writable");
  }
  const containerUser = `${hostUid}:${hostGid}`;
  const lockPath = path.resolve(process.env.PI_AGENT_PLUGIN_LOCK ?? path.join(import.meta.dirname, "../../plugins.lock.json"));
  const skillLockPath = path.resolve(process.env.PI_AGENT_SKILL_LOCK ?? path.join(import.meta.dirname, "../../skills.lock.json"));
  await mkdir(runtimeRoot, { recursive: true });
  await mkdir(worktreeRoot, { recursive: true });
  const actualImageDigest = process.env.PI_AGENT_IMAGE_DIGEST ?? await inspectImageDigest(image);
  const imageSkillLock = await readImageFile(image, "/opt/pi/skills.lock.json");
  const lock = await verifyPluginLock(lockPath, { rootDir: path.dirname(lockPath), actualImageDigest });
  const skillLock = await verifySkillLock(skillLockPath, {
    rootDir: path.dirname(skillLockPath),
    imageLockContent: imageSkillLock,
  });
  const integrityHealthy = lock.healthy && skillLock.healthy;
  const gitCommon = (await exec("git", ["rev-parse", "--git-common-dir"], { cwd: repoPath, encoding: "utf8" })).stdout.trim();
  const gitMetadataPath = path.resolve(repoPath, gitCommon);
  let mcpRepository: string | undefined;
  let mcpGatewayReachable = false;
  if (mcpMode === "github_read_only") {
    if (!mcpSigningSecret) throw new Error("MCP_GATEWAY_SIGNING_SECRET is required when MCP is enabled");
    const repositories = parseGitHubRepositories(process.env.PI_AGENT_GITHUB_REPOSITORIES);
    if (repositories.length === 0) throw new Error("PI_AGENT_GITHUB_REPOSITORIES is required when MCP is enabled");
    const remote = (await exec("git", ["remote", "get-url", "origin"], { cwd: repoPath, encoding: "utf8" })).stdout.trim();
    const currentRepository = repositoryFromRemote(remote);
    mcpRepository = repositories.find((repository) => repository.toLowerCase() === currentRepository?.toLowerCase());
    if (!mcpRepository) throw new Error("The Agent repository is not in PI_AGENT_GITHUB_REPOSITORIES");
    mcpGatewayReachable = await probeMcpGateway(mcpGatewayUrl);
  }
  if (mcpMode === "github_read_only") {
    const healthTimer = setInterval(() => {
      void probeMcpGateway(mcpGatewayUrl).then((reachable) => { mcpGatewayReachable = reachable; });
    }, positiveInteger(process.env.PI_AGENT_MCP_HEALTH_INTERVAL_MS, 5_000));
    healthTimer.unref();
  }
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
    availableModelRefs: configuredModels.map((model) => model.ref),
    router,
    mcpRepository,
    startProcess: async (session, onEvent) => {
      const model = resolveAgentModel(session.modelRef);
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
          modelRef: model.ref,
          extensionPaths: ["/opt/pi/extensions/policy.ts", "/opt/pi/extensions/model-gateway.ts"],
          pluginPaths: lock.plugins.map((plugin) => plugin.container_path),
          skillPaths: skillLock.skills.map((skill) => skill.container_path),
          toolNames: ["read", "write", "edit", "bash", "grep", "find", "ls", ...lock.plugins.flatMap((plugin) => plugin.enabled_tools)],
          continueSession: session.restored,
          safeCommands: parseSafeCommands(process.env.PI_AGENT_SAFE_COMMANDS),
          user: containerUser,
          ...(mcpMode === "github_read_only" ? {
            mcp: {
              mode: "github_read_only" as const,
              gatewayUrl: mcpGatewayUrl,
              tools: [...GITHUB_READ_ONLY_TOOLS],
            },
          } : {}),
        },
        (runId, route) => {
          const selectedModel = resolveAgentModel(route.modelRef);
          return issueGatewayToken({
            runId,
            provider: selectedModel.provider,
            model: selectedModel.id,
            api: selectedModel.api,
            reasoningEffort: route.effort,
            expiresAt: Date.now() + 16 * 60_000,
          }, gatewaySecret);
        },
        onEvent,
        mcpMode === "github_read_only" ? (runId) => issueMcpToken({
          run: runId,
          session: session.id,
          slackUser: session.ownerUserId,
          repository: mcpRepository!,
          server: "github",
          tools: [...GITHUB_READ_ONLY_TOOLS],
          mode: "github_read_only",
          expiry: Date.now() + 16 * 60_000,
        }, mcpSigningSecret) : undefined,
      );
    },
  });
  await orchestrator.restore();
  const runtimeHealthy = () => integrityHealthy && (mcpMode === "off" || mcpGatewayReachable);
  return createAgentHttpServer({
    orchestrator,
    token,
    acceptRuns: runtimeHealthy,
    health: () => ({
      status: runtimeHealthy() ? "healthy" : "unhealthy",
      runtime: "pi-coding-agent",
      version: "1.0.1",
      pi_version: "1.0.1",
      mcp_mode: mcpMode,
      mcp_gateway_reachable: mcpGatewayReachable,
      mcp_servers: mcpMode === "github_read_only" ? ["github"] : [],
      mcp_tool_count: mcpMode === "github_read_only" ? GITHUB_READ_ONLY_TOOLS.length : 0,
      provider: "openai",
      model: AGENT_MODEL_ID,
      reasoning_effort: AGENT_REASONING_EFFORT,
      models: listAgentModels().map((model) => ({
        ref: model.ref,
        provider: model.provider,
        model: model.id,
        reasoning_effort: model.reasoningEffort,
        configured: configuredProviders.has(model.provider),
      })),
      tools: ["read", "write", "edit", "bash", "grep", "find", "ls", ...lock.plugins.flatMap((plugin) => plugin.enabled_tools)],
      plugin_integrity: lock.healthy ? "verified" : "failed",
      skills: skillLock.skills.map((skill) => skill.name),
      skill_integrity: skillLock.healthy ? "verified" : "failed",
      errors: runtimeHealthy() ? [] : [
        ...lock.errors,
        ...skillLock.errors,
        ...(mcpMode !== "off" && !mcpGatewayReachable ? ["MCP gateway is unreachable"] : []),
      ],
    }),
  });
}

async function probeMcpGateway(gatewayUrl: string): Promise<boolean> {
  try {
    const url = new URL(gatewayUrl);
    const healthUrl = new URL("/healthz", url);
    if (url.hostname === "mcp-gateway") healthUrl.hostname = "127.0.0.1";
    const response = await fetch(healthUrl, { signal: AbortSignal.timeout(1_000) });
    return response.ok;
  } catch {
    return false;
  }
}

async function inspectImageDigest(image: string): Promise<string> {
  try {
    return (await exec("docker", ["image", "inspect", "--format={{.Id}}", image], { encoding: "utf8" })).stdout.trim();
  } catch {
    return "missing";
  }
}

async function readImageFile(image: string, filePath: string): Promise<string> {
  try {
    return (await exec("docker", [
      "run", "--rm", "--network", "none", "--read-only",
      "--entrypoint", "cat", image, filePath,
    ], { encoding: "utf8" })).stdout;
  } catch {
    return "";
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
