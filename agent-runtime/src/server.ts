import { execFile } from "node:child_process";
import { mkdir } from "node:fs/promises";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { guardedHandler } from "./http-handler.js";
import path from "node:path";
import os from "node:os";
import { pathToFileURL } from "node:url";
import { promisify } from "node:util";

import { randomUUID, createHash, timingSafeEqual } from "node:crypto";
import { PiRunCapture } from "./pi-capture.js";
import { AgentTelemetry, isCredentialKey, redact } from "./telemetry.js";
import { AgentTracing } from "./tracing.js";
import { DockerPiProcess, type ChecksContext, type WriterContext } from "./container-runtime.js";
import { ApprovalLedger } from "./approval-ledger.js";
import { approvalSettings } from "./approval-mode.js";
import { parseAgentLimitConfig } from "./agent-limits.js";

export { parseAgentLimits } from "./agent-limits.js";
import { budgetIdFor, outputTokenLimitFor, parseDefaultRunBudget } from "./run-budget.js";
import { AGENT_MODEL_ID, AGENT_REASONING_EFFORT, listAgentModels, resolveAgentModel } from "./agent-model.js";
import { JevRouter, type JevMode } from "./jev-router.js";
import { repositoryFromRemote, parseGitHubRepositories } from "./github-repository.js";
import { issueClassifierGatewayToken, issueGatewayToken } from "./gateway-token.js";
import { allServerTools, enabledMcpServers, MCP_SERVERS, parseMcpMode, parseMcpServers, type McpServerId, type McpRunServer } from "./mcp-config.js";
import { deriveMcpServerSecret, issueMcpToken } from "./mcp-token.js";
import { deriveBudgetRpcSecret } from "./budget-rpc.js";
import { IdempotencyCache, IdempotencyKeyReusedError } from "./idempotency.js";
import { diskQuotaFromEnv } from "./disk-usage.js";
import {
  CodingAgentOrchestrator,
  publicRun,
  RuntimeError,
  type RunRecord,
} from "./orchestrator.js";
import { archivePatch, undeliveredPatchPath } from "./patch-archive.js";
import { verifyPluginLock } from "./plugin-lock.js";
import { RuntimeStateStore } from "./runtime-state.js";
import { verifySkillLock } from "./skill-lock.js";
import { startEvidenceRetention } from "./evidence-retention.js";
import { ControlStore } from "./ci-control.js";
import { BudgetLedger } from "./budget-ledger.js";
import { createCiRuntime } from "./ci-runtime.js";
import { taskMode } from "./task-mode.js";
import { workflowMode } from "./structured-workflow.js";
import type { CiRepairs } from "./ci-repair.js";
import { loadProfile, type VerificationProfile } from "./verification.js";
import { createQualityServices } from "./workflow-runtime.js";
import { DEFAULT_DIFF_LIMITS, WorktreeManager } from "./worktree-manager.js";
import { parseGraderConfig } from "./grader-config.js";

const exec = promisify(execFile);
export const DEFAULT_AGENT_RUNTIME_PORT = 3001;
const DEFAULT_AGENT_RUNTIME_HOST = "127.0.0.1";
const MAX_REQUEST_BYTES = 64_000;

interface ServerOptions {
  /** Remembers POST /v1/sessions and /prompts results by Idempotency-Key so a client retry never starts a second run. */
  idempotency?: IdempotencyCache;
  budgetLedger?: BudgetLedger;
  budgetSecret?: string;
  orchestrator: CodingAgentOrchestrator;
  ci?: CiRepairs;
  token: string;
  health: () => Record<string, unknown>;
  acceptRuns?: () => boolean;
  feedback?: (run: RunRecord, payload: Record<string, unknown>, id: string) => Promise<boolean>;
}

const FEEDBACK_REASONS = ["wrong_approach", "incomplete", "broke_tests", "scope_creep", "style", "other"];
const FEEDBACK_COMMENT_MAX_CHARS = 2000;

function feedbackDetails(body: Record<string, unknown>): { reason_category?: string; comment?: string } {
  const details: { reason_category?: string; comment?: string } = {};
  if (body.reason_category !== undefined) {
    if (typeof body.reason_category !== "string" || !FEEDBACK_REASONS.includes(body.reason_category)) throw new RuntimeError("Invalid feedback reason_category", "invalid_request", 400);
    details.reason_category = body.reason_category;
  }
  if (body.comment !== undefined) {
    if (typeof body.comment !== "string" || [...body.comment].length > FEEDBACK_COMMENT_MAX_CHARS) throw new RuntimeError("Invalid feedback comment", "invalid_request", 400);
    // Redact here as well as in telemetry so no capture sink ever sees the raw comment.
    details.comment = redact(body.comment) as string;
  }
  return details;
}

export function createAgentHttpServer(options: ServerOptions) {
  const idempotent = <T>(request: IncomingMessage, route: string, body: unknown, work: () => Promise<T>) => idempotentRequest(options.idempotency, request, route, body, work);
  return createServer(guardedHandler("agent-runtime", async (request, response) => {
    try {
      const url = new URL(request.url ?? "/", "http://runtime.local");
      if (url.pathname.startsWith('/internal/budget/')) {
        if (!options.budgetSecret || !authorized(request,options.budgetSecret)) { sendError(response,401,'unauthorized','Gateway budget authentication required'); return; }
        if (request.method !== 'POST' || !options.budgetLedger) { sendError(response,503,'budget_unavailable','Host budget store unavailable'); return; }
        const body = asRecord(await readJson(request)); const ledger = options.budgetLedger;
        if (url.pathname === '/internal/budget/reserve' && options.acceptRuns && !options.acceptRuns()) { sendError(response,503,'runtime_unhealthy','New budget reservations are unavailable'); return; }
        const id = (value: unknown) => { const result = requiredString(value,'budget identity'); if (result.length > 240) throw new RuntimeError('Invalid budget identity','invalid_request',400); return result; };
        try {
          if (url.pathname === '/internal/budget/reserve') ledger.reserve(id(body.workflow),id(body.id),Number(body.amount),body.execution === undefined ? undefined : id(body.execution));
          else if (url.pathname === '/internal/budget/settle') ledger.settle(id(body.id),body.amount === null ? null : Number(body.amount));
          else if (url.pathname === '/internal/budget/first-model') { sendJson(response,200,{ result: ledger.firstModel(id(body.execution)) }); return; }
          else { sendError(response,404,'not_found','Unknown budget operation'); return; }
          sendJson(response,200,{});
        } catch { sendError(response,429,'budget_unavailable_or_exhausted','Budget admission refused'); }
        return;
      }
      if (request.method === "GET" && url.pathname === "/health") {
        sendJson(response, 200, options.health());
        return;
      }
      if (!authorized(request, options.token)) {
        sendError(response, 401, "unauthorized", "A valid Agent runtime token is required");
        return;
      }
      if (options.acceptRuns && !options.acceptRuns() && request.method === "POST" && !/\/(?:cancel|close)$/.test(url.pathname)) {
        sendError(response, 502, "runtime_unhealthy", "Agent runtime integrity or control state is unavailable");
        return;
      }

      if (request.method === "GET" && url.pathname === "/v1/ci-repairs") {
        sendJson(response, 200, options.ci ? await options.ci.notifications() : []); return;
      }
      if (request.method === "POST" && url.pathname === "/v1/ci-repairs") {
        if (!options.ci) throw new RuntimeError("CI repair requires trusted profile, owner mapping and separate publisher credentials", "ci_setup_required", 503);
        const body = asRecord(await readJson(request));
        if (parseTaskMode(body.task_mode) !== "implement") throw new RuntimeError("CI repair requires implement task_mode", "invalid_request", 400);
        options.ci.checkContext(body.team_id, body.channel_id, body.user_id);
        sendJson(response, 202, await options.ci.create(requiredString(body.target, "target"), requiredString(body.user_id, "user_id"), undefined, undefined, parseWorkflowMode(body.workflow_mode))); return;
      }
      const retryMatch = /^\/v1\/ci-repairs\/([^/]+)\/publication$/.exec(url.pathname);
      if (request.method === "POST" && retryMatch && options.ci) {
        const body = asRecord(await readJson(request)); options.ci.checkContext(body.team_id, body.channel_id, body.user_id); sendJson(response, 202, options.ci.retryPublication(decodeURIComponent(retryMatch[1]), requiredString(body.user_id, "user_id"))); return;
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
        const result = await idempotent(request, url.pathname, body, () => options.orchestrator.createSession(body as never));
        sendJson(response, 202, result);
        return;
      }
      const promptMatch = /^\/v1\/sessions\/([^/]+)\/prompts$/.exec(url.pathname);
      if (request.method === "POST" && promptMatch) {
        const body = asRecord(await readJson(request));
        if (body.routing_text !== undefined && typeof body.routing_text !== "string") {
          throw new RuntimeError("routing_text must be a string", "invalid_request", 400);
        }
        const result = await idempotent(request, url.pathname, body, () => options.orchestrator.prompt(
          decodeURIComponent(promptMatch[1]),
          requiredString(body.prompt, "prompt"),
          requiredString(body.user_id, "user_id"),
          typeof body.routing_text === "string" ? body.routing_text : undefined,
          body.task_id as string | undefined,
          parseWorkflowMode(body.workflow_mode),
          parseTaskMode(body.task_mode),
        ));
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
      const tasksMatch = /^\/v1\/runs\/([^/]+)\/tasks$/.exec(url.pathname);
      if (request.method === 'GET' && tasksMatch) {
        const id = decodeURIComponent(tasksMatch[1]);
        const child = options.ci?.has(id) ? options.ci.executionId(id) : id;
        sendJson(response,200,child ? options.orchestrator.tasks(child) : []); return;
      }
      const resumeMatch = /^\/v1\/runs\/([^/]+)\/resume$/.exec(url.pathname);
      if (request.method === 'POST' && resumeMatch) {
        const body = asRecord(await readJson(request)), id = decodeURIComponent(resumeMatch[1]);
        const user = requiredString(body.user_id,'user_id');
        sendJson(response,202,options.ci?.has(id) ? await options.ci.resume(id,user) : await options.orchestrator.resume(id,user)); return;
      }
      const runMatch = /^\/v1\/runs\/([^/]+)$/.exec(url.pathname);
      if (request.method === "GET" && runMatch) {
        const id = decodeURIComponent(runMatch[1]);
        sendJson(response, 200, options.ci?.has(id) ? await options.ci.get(id) : publicRun(options.orchestrator.getRun(id)));
        return;
      }
      const eventsMatch = /^\/v1\/runs\/([^/]+)\/events$/.exec(url.pathname);
      if (request.method === "GET" && eventsMatch) {
        const runId = decodeURIComponent(eventsMatch[1]);
        if (options.ci?.has(runId)) {
          const job = await options.ci.get(runId); sendSse(response, job.events ?? [], 0); return;
        }
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
        const id = decodeURIComponent(decisionMatch[1]);
        if (options.ci?.has(id)) {
          await options.ci.decide(id, requiredString(body.approval_id, "approval_id"), requiredString(body.user_id, "user_id"), decision, true);
          sendJson(response, 202, { status: "accepted" }); return;
        }
        await options.orchestrator.decide(
          decodeURIComponent(decisionMatch[1]),
          requiredString(body.approval_id, "approval_id"),
          requiredString(body.user_id, "user_id"),
          decision,
        );
        sendJson(response, 202, { status: "accepted" });
        return;
      }
      const feedbackMatch = /^\/v1\/runs\/([^/]+)\/feedback$/.exec(url.pathname);
      if (request.method === "POST" && feedbackMatch) {
        const run = options.orchestrator.getRun(decodeURIComponent(feedbackMatch[1]));
        const body = asRecord(await readJson(request));
        const user = requiredString(body.user_id, "user_id");
        if (run.owner_user_id !== user) throw new RuntimeError("Only the run owner can submit feedback", "feedback_forbidden", 401);
        if (!["completed", "failed", "cancelled", "rejected", "timed_out", "interrupted"].includes(run.status)) throw new RuntimeError("Run is still active", "feedback_conflict", 409);
        const verdict = requiredString(body.verdict, "verdict");
        const feedbackId = requiredString(body.feedback_id, "feedback_id");
        if (!["accepted", "needs_changes"].includes(verdict) || feedbackId.length > 200) throw new RuntimeError("Invalid feedback", "invalid_request", 400);
        const details = feedbackDetails(body);
        if (!options.feedback) throw new RuntimeError("Feedback capture is disabled", "feedback_unavailable", 503);
        const id = createHash("sha256").update(JSON.stringify([run.run_id, user, feedbackId, verdict])).digest("hex");
        if (!await options.feedback(run, {user_id: user, verdict, feedback_id: feedbackId, ...details}, id)) throw new RuntimeError("Feedback capture failed", "feedback_unavailable", 503);
        sendJson(response, 202, {status: "accepted"}); return;
      }
      const cancelMatch = /^\/v1\/runs\/([^/]+)\/cancel$/.exec(url.pathname);
      if (request.method === "POST" && cancelMatch) {
        const body = asRecord(await readJson(request));
        const id = decodeURIComponent(cancelMatch[1]);
        if (options.ci?.has(id)) await options.ci.cancel(id, requiredString(body.user_id, "user_id"));
        else await options.orchestrator.cancel(id, requiredString(body.user_id, "user_id"));
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
  }));
}

export async function settleCaptureWrite(runId: string, written: boolean, telemetry: Pick<AgentTelemetry, "releaseRun">, persistFailure: (runId: string, reason: string) => Promise<void>): Promise<void> {
  try {
    if (!written) await persistFailure(runId, "terminal_outbox_write_failed");
  } finally { telemetry.releaseRun(runId); }
}

export async function recoverPendingCaptures(runs: RunRecord[], recover: (run: RunRecord) => Promise<void>): Promise<void> {
  for (const run of runs) {
    if (run.status === "interrupted" && run.capture_complete === undefined) await recover(run);
  }
}

export async function createProductionServer() {
  const token = process.env.AGENT_RUNTIME_TOKEN ?? "";
  const gatewaySecret = process.env.MODEL_GATEWAY_SIGNING_SECRET ?? "";
  if (!token || !gatewaySecret) throw new Error("AGENT_RUNTIME_TOKEN and MODEL_GATEWAY_SIGNING_SECRET are required");
  const hostCoordinator = new ControlStore(path.join(os.tmpdir(), `slack-pi-${process.getuid?.() ?? 'user'}-coordinator.sqlite`));
  const releaseHostCoordinator = hostCoordinator.acquireCoordinator();
  try {
  const repoPath = path.resolve(process.env.PI_AGENT_REPO_PATH ?? process.cwd());
  const runtimeRoot = path.resolve(process.env.PI_AGENT_STATE_ROOT ?? path.join(repoPath, ".pi-agent-runtime"));
  const worktreeRoot = path.resolve(process.env.PI_AGENT_WORKTREE_ROOT ?? path.join(repoPath, ".pi-agent-worktrees"));
  const image = process.env.PI_AGENT_IMAGE ?? "slack-pi-agent:1.0.1";
  const network = process.env.PI_AGENT_NETWORK ?? "pi-model-only";
  const gatewayUrl = process.env.PI_MODEL_GATEWAY_URL ?? "http://model-gateway:8080";
  const mcpMode = parseMcpMode(process.env.PI_AGENT_MCP_MODE);
  const mcpServerIds = enabledMcpServers(mcpMode, parseMcpServers(process.env.PI_AGENT_MCP_SERVERS));
  const gatewayUrls: Partial<Record<McpServerId, string>> = {
    github: process.env.PI_AGENT_MCP_GITHUB_GATEWAY_URL ?? process.env.PI_AGENT_MCP_GATEWAY_URL ?? "http://mcp-github:8090/mcp",
    clickhouse: process.env.PI_AGENT_MCP_CLICKHOUSE_GATEWAY_URL ?? "http://mcp-clickhouse-gateway:8090/mcp",
    context7: process.env.PI_AGENT_MCP_CONTEXT7_GATEWAY_URL ?? "http://mcp-context7:8090/mcp",
  };
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
  const jevClassifierMode = parseJevClassifierMode(process.env.PI_AGENT_JEV_CLASSIFIER_MODE);
  if (jevClassifierMode === "on" && !process.env.OPENROUTER_API_KEY) {
    throw new Error("OPENROUTER_API_KEY is required when run-time Jev classification is enabled");
  }
  const jevClassifierMaxCalls = positiveInteger(process.env.PI_AGENT_JEV_CLASSIFIER_MAX_CALLS, 8);
  const router = new JevRouter({ mode: jevMode as JevMode, apiKey: process.env.OPENROUTER_API_KEY,
    admit: (id, text) => {
      if (!budgetLedger || !trustedProfile) return { settle: () => {} };
      const price = JSON.parse(process.env.PI_AGENT_MODEL_PRICES_JSON ?? "{}")["openrouter/typesafe/jev-1.13"];
      if (!price || !Number.isFinite(price.input) || price.input <= 0 || !Number.isFinite(price.output) || price.output < 0) throw new Error("Jev budget prices missing");
      budgetLedger.open(id, trustedProfile.profile.maxCostUsd);
      const reservation = randomUUID(); budgetLedger.reserve(id, reservation, ((Buffer.byteLength(text) + 4096) * price.input + 4096 * price.output) / 1_000_000);
      return { settle: cost => budgetLedger.settle(reservation, cost) };
    },
  });
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
  if (!await readImageFile(image, "/opt/pi/extensions/secret-env.ts")) throw new Error("Agent image must be rebuilt with the secret-env extension");
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
  const mcpServerReachable: Partial<Record<McpServerId, boolean>> = {};
  if (mcpServerIds.length > 0) {
    if (mcpServerIds.some((id) => id !== "codegraph") && !mcpSigningSecret) throw new Error("MCP_GATEWAY_SIGNING_SECRET is required when HTTP MCP is enabled");
    if (mcpServerIds.includes("github")) {
      const repositories = parseGitHubRepositories(process.env.PI_AGENT_GITHUB_REPOSITORIES);
      if (repositories.length === 0) throw new Error("PI_AGENT_GITHUB_REPOSITORIES is required when GitHub MCP is enabled");
      const remote = (await exec("git", ["remote", "get-url", "origin"], { cwd: repoPath, encoding: "utf8" })).stdout.trim();
      const currentRepository = repositoryFromRemote(remote);
      mcpRepository = repositories.find((repository) => repository.toLowerCase() === currentRepository?.toLowerCase());
      if (!mcpRepository) throw new Error("The Agent repository is not in PI_AGENT_GITHUB_REPOSITORIES");
    }
    await Promise.all(mcpServerIds.filter((id) => id !== "codegraph").map(async (id) => {
      mcpServerReachable[id] = await probeMcpGateway(gatewayUrls[id]!);
    }));
    if (mcpServerIds.includes("codegraph")) mcpServerReachable.codegraph = true;
  }
  if (mcpServerIds.length > 0) {
    const healthTimer = setInterval(() => {
      for (const id of mcpServerIds.filter((entry) => entry !== "codegraph")) {
        probeMcpGateway(gatewayUrls[id]!).then((reachable) => { mcpServerReachable[id] = reachable; }, () => { mcpServerReachable[id] = false; });
      }
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
  const telemetryEnabled = process.env.PI_AGENT_TELEMETRY_ENABLED === "true";
  const telemetry = telemetryEnabled ? new AgentTelemetry({
    file: path.join(runtimeRoot, "analytics", "outbox.sqlite"),
    brokers: (process.env.AGENT_KAFKA_BROKERS ?? "localhost:9092").split(","),
    secrets: Object.entries(process.env).filter(([key]) => isCredentialKey(key)).map(([,value]) => value!).filter(Boolean),
  }) : undefined;
  const tracing = AgentTracing.fromEnvironment();
  const observed = new Map<string, string>();
  const profilePath = process.env.PI_AGENT_VERIFICATION_PROFILE;
  const trustedProfile = profilePath ? await loadProfile(profilePath, repoPath, worktreeRoot) : undefined;
  if (trustedProfile) {
    if (await inspectImageDigest(trustedProfile.profile.image) === "missing") throw new Error("Pinned verification image is missing; build Dockerfile.verification and configure its digest");
    if (!await readImageFile(trustedProfile.profile.image, "/opt/pi/dist/src/verification-entry.js")) throw new Error("Verification image is missing the trusted runner");
    if (trustedProfile.profile.isolation !== "per_check" && !await readImageFile(trustedProfile.profile.image, "/opt/pi/dist/src/verification-runner.js")) throw new Error("Verification image must be rebuilt with the per-mode runner (or set isolation: per_check)");
    if (trustedProfile.profile.structuredWorkflow?.enabled && !await readImageFile(image, "/opt/pi/extensions/workflow-agent.ts")) throw new Error("Agent image must be rebuilt with workflow-agent extension");
    if (!await readImageFile(image, "/opt/pi/extensions/grader.ts")) throw new Error("Agent image must be rebuilt with the isolated grader extension");
    if (!await readImageFile(image, "/opt/pi/extensions/checks.ts")) throw new Error("Agent image must be rebuilt with the run_checks extension");
    const prices = JSON.parse(process.env.PI_AGENT_MODEL_PRICES_JSON ?? "{}");
    const refs = [...configuredModels.map(model => model.ref), ...((jevMode !== "off" || jevClassifierMode === "on") ? ["openrouter/typesafe/jev-1.13"] : [])];
    if (refs.some(ref => !prices[ref] || !Number.isFinite(prices[ref].input) || prices[ref].input <= 0 || !Number.isFinite(prices[ref].output) || prices[ref].output < 0)) throw new Error("PI_AGENT_MODEL_PRICES_JSON must provide conservative prices for all enabled coding/grader/Jev models");
    const remote = (await exec("git", ["remote", "get-url", "origin"], { cwd: repoPath, encoding: "utf8" })).stdout.trim();
    if (repositoryFromRemote(remote)?.toLowerCase() !== trustedProfile.profile.repository.toLowerCase()) throw new Error("Verification profile repository does not match the checkout origin");
  }
  if (trustedProfile?.profile.structuredWorkflow?.enabled && !configuredModels.some(model => model.ref === "openai/gpt-5.6-luna")) throw new Error("Structured workflow requires configured Luna worker provider");
  const grader = trustedProfile ? parseGraderConfig(process.env, configuredModels) : undefined;
  const executionParents = new Map<string, string>();
  function executionExpiry(id: string) {
    const parent = executionParents.get(id);
    const structured = parent && orchestrator.getRun(parent).workflow_mode === 'structured';
    return Date.now() + (structured ? 31 : 16) * 60_000;
  }
  const priorActive = new Map<string, number>();
  const verificationBases = new Map<string, string>();
  // Every run is metered, with or without a trusted profile (no-profile runs get a default budget).
  const budgetLedger = new BudgetLedger(path.join(runtimeRoot, "budget", "budget.sqlite"));
  const defaultRunBudgetUsd = parseDefaultRunBudget(process.env);
  const workflowControl = trustedProfile ? new ControlStore(path.join(runtimeRoot, "control", "state.sqlite"), () => { void orchestrator.shutdown().catch(() => {}); }) : undefined;
  const releaseCoordinator = workflowControl?.acquireCoordinator();
  const quality = trustedProfile ? createQualityServices({
    ...trustedProfile, control: workflowControl!, session: id => orchestrator.getRun(id).session_id, repoPath, worktreeRoot, verificationBases, priorActive, ledger: budgetLedger, mapping: executionParents, root: runtimeRoot, image: actualImageDigest, network, gatewayUrl, telemetry, tracing,
    githubMcp: mcpServerIds.includes('github') ? { gatewayUrl: gatewayUrls.github!, token: (id,owner) => {
      const token = issueMcpToken({ version: 2, run: id, session: id, slackUser: owner, server: 'github', tools: [...MCP_SERVERS.github.directTools], mode: 'read_only', scope: { repository: mcpRepository! }, maxCalls: 4, expiry: Date.now() + 5 * 60_000 },deriveMcpServerSecret(mcpSigningSecret,'github'));
      telemetry?.addSecret(token,id); return token;
    } } : undefined,
    phase: (id, phase, cost) => orchestrator.qualityPhase(id, phase, cost),
    grader,
    token: (id, route, role) => {
      const model = resolveAgentModel(route.modelRef);
      const token = issueGatewayToken({ runId: id, workflowRunId: executionParents.get(id), role, outputTokenLimit: outputTokenLimitFor(role), provider: model.provider, model: model.id, api: model.api, reasoningEffort: route.effort, expiresAt: executionExpiry(id), traceparent: tracing?.traceparent(id) }, gatewaySecret);
      telemetry?.addSecret(token, id); return token;
    },
  }) : undefined;
  let ciRuntime: Awaited<ReturnType<typeof createCiRuntime>> | undefined;
  const agentLimits = parseAgentLimitConfig(process.env);
  const approval = approvalSettings(process.env);
  // Always on, profile or not: a pending delivery approval survives a restart bound to its patch hash.
  const approvalLedger = new ApprovalLedger(path.join(runtimeRoot, "approvals", "approvals.sqlite"));
  const credentialValues = () => Object.entries(process.env).filter(([key]) => isCredentialKey(key)).map(([, value]) => value!).filter(Boolean);
  const orchestrator: CodingAgentOrchestrator = new CodingAgentOrchestrator({
    quality,
    diskQuota: diskQuotaFromEnv(process.env),
    requireVerification: process.env.PI_AGENT_REQUIRE_VERIFICATION === "1",
    archivePatch: async input => (await archivePatch({
      worktree: input.worktree, baseline: input.baseline, file: undeliveredPatchPath(runtimeRoot, input.run_id),
      secrets: Object.entries(process.env).filter(([key]) => isCredentialKey(key)).map(([, value]) => value!).filter(Boolean),
    }))?.path,
    captureTraceContext: id => tracing?.traceparent(id),
    captureState: id => ({complete: telemetry ? telemetry.completeness(id).length === 0 : null, reasons: telemetry?.completeness(id) ?? []}),
    observeLifecycle: (id, name, phase, failed, sessionId) => {
      if (name === 'resume' && phase === 'start') { tracing?.startRun(id,sessionId ?? 'unknown',undefined,orchestrator.getRun(id).traceparent); telemetry?.setTraceparent(id,tracing?.traceparent(id)); }
      if (name === "routing" && phase === "start") { tracing?.startRun(id, sessionId ?? "unknown"); telemetry?.setTraceparent(id, tracing?.traceparent(id)); }
      tracing?.lifecycle(id, name, phase, failed);
      if (name === "preparation" && failed) tracing?.endRun(id, "failed");
    },
    observePolicy: (runId, sessionId, decision) => {
      telemetry?.record(runId, sessionId, "policy_decision", decision).catch(() => console.error(JSON.stringify({ event: "agent_policy_telemetry_failed", run_id: runId })));
    },
    observeRun: run => {
      const summary = {...publicRun(run), workflow_prompt: undefined, workflow_requirement: undefined, events: undefined, capture_complete: run.capture_complete ?? null};
      const signature = JSON.stringify(summary);
      if (observed.get(run.run_id) === signature) return;
      if (!observed.has(run.run_id) && ["running", "awaiting_approval"].includes(run.status)) tracing?.startRun(run.run_id, run.session_id);
      observed.set(run.run_id, signature);
      tracing?.describeRun(run.run_id,run.model,run.reasoning_effort);
      telemetry?.record(run.run_id, run.session_id, "run", {...summary, traceparent: tracing?.traceparent(run.run_id)}).then(async written => {
        if (["running", "awaiting_approval"].includes(run.status)) return;
        await settleCaptureWrite(run.run_id, written, telemetry, (id, reason) => orchestrator.markCaptureIncomplete(id, reason));
      }).catch(() => console.error(JSON.stringify({event:"agent_capture_state_persist_failed",run_id:run.run_id})));
      if (observed.size > 2000) observed.delete(observed.keys().next().value!);
      if (!["running", "awaiting_approval"].includes(run.status)) tracing?.log(run.run_id, "agent.run.finished", {status:run.status, capture_complete:summary.capture_complete ?? false});
      if (!["running", "awaiting_approval"].includes(run.status) && !ciRuntime?.ci.has(run.run_id)) tracing?.endRun(run.run_id, run.status);
    },
    worktrees,
    stateStore,
    maxActiveSessions: positiveInteger(process.env.PI_AGENT_MAX_CONCURRENCY, 2),
    deadlineMs: positiveInteger(process.env.PI_AGENT_DEADLINE_MS, 15 * 60_000),
    deliveryApproval: { ttlMs: approval.deliveryTtlMs, secrets: credentialValues },
    approvalLedger,
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
          image: actualImageDigest,
          network,
          worktreePath: session.worktreePath,
          gitMetadataPath,
          sessionStatePath,
          gatewayUrl,
          modelRef: model.ref,
          ...writerTools(trustedProfile?.profile, ["/opt/pi/extensions/policy.ts", "/opt/pi/extensions/model-gateway.ts", "/opt/pi/extensions/secret-env.ts"], ["read", "write", "edit", "bash", "grep", "find", "ls", ...lock.plugins.flatMap((plugin) => plugin.enabled_tools)]),
          pluginPaths: lock.plugins.map((plugin) => plugin.container_path),
          skillPaths: skillLock.skills.map((skill) => skill.container_path),
          continueSession: session.restored,
          safeCommands: parseSafeCommands(process.env.PI_AGENT_SAFE_COMMANDS, { profiled: Boolean(trustedProfile) }),
          limits: agentLimits.base,
          routeLimits: agentLimits,
          ...(trustedProfile ? { writerContext: writerContextFor(trustedProfile.profile) } : {}),
          // Runs that skip delivery approval keep per-write prompts, so no run loses both gates.
          writeApproval: session.skipDelivery ? "per_prompt" : approval.writeApproval,
          user: containerUser,
          ...(mcpServerIds.length > 0 ? {
            mcp: {
              mode: mcpMode as "github_read_only" | "read_only",
              servers: mcpServerIds.map((id): McpRunServer => id === "codegraph"
                ? { server: "codegraph", command: MCP_SERVERS.codegraph.command! }
                : { server: id, gatewayUrl: gatewayUrls[id]!, token: "" }),
            },
          } : {}),
        },
        (runId, route) => {
          const selectedModel = resolveAgentModel(route.modelRef);
          const token = issueGatewayToken({
            runId,
            workflowRunId: budgetIdFor(runId, executionParents, budgetLedger, defaultRunBudgetUsd),
            role: "writer",
            outputTokenLimit: outputTokenLimitFor("writer"),
            provider: selectedModel.provider,
            model: selectedModel.id,
            api: selectedModel.api,
            reasoningEffort: route.effort,
            expiresAt: executionExpiry(runId),
            traceparent: tracing?.traceparent(runId),
          }, gatewaySecret);
          telemetry?.addSecret(token, runId); return token;
        },
        onEvent,
        mcpServerIds.length > 0 ? (runId) => Object.fromEntries(mcpServerIds.filter((id) => id !== "codegraph").map((id) => {
          const scope: Record<string, string> = id === "github" ? { repository: mcpRepository! } : id === "clickhouse" ? { database: "agent_mcp" } : { data: "public_docs" };
          const token = issueMcpToken({
            version: 2, run: runId, session: session.id, slackUser: session.ownerUserId, server: id,
            tools: allServerTools(id), mode: "read_only", scope, maxCalls: MCP_SERVERS[id].maxCalls,
            expiry: executionExpiry(runId),
          }, deriveMcpServerSecret(mcpSigningSecret, id));
          telemetry?.addSecret(token, runId); return [id, token];
        })) : undefined,
        jevClassifierMode === "on" ? (runId) => { const token = issueClassifierGatewayToken({
          kind: "classifier",
          runId,
          workflowRunId: executionParents.get(runId),
          provider: "openrouter",
          model: "typesafe/jev-1.13",
          api: "typesafe-system-one",
          maxCalls: jevClassifierMaxCalls,
          traceparent: tracing?.traceparent(runId),
          expiresAt: executionExpiry(runId),
        }, gatewaySecret); telemetry?.addSecret(token, runId); return token; } : undefined,
        telemetry, session.id, tracing, undefined, id => budgetLedger.prompt(id),
      );
    },
  });
  if (workflowControl) {
    for (const execution of workflowControl.all<{ id: string; parent: string; session: string; statePath: string; status: string; traceparent?: string }>("execution")) {
      if (execution.status !== "running") continue;
      if (telemetry) {
        telemetry.setTraceparent(execution.id, execution.traceparent);
        await new PiRunCapture(telemetry, execution.id, execution.session, execution.statePath, "stopped").recover();
        telemetry.markIncomplete(execution.parent, "execution_interrupted_before_terminal_snapshot");
        await telemetry.record(execution.id, execution.session, "run", { status: "interrupted", capture_complete: false });
        telemetry.releaseRun(execution.id);
      }
      workflowControl.put("execution", execution.id, { ...execution, status: "interrupted" });
    }
  }
  if (telemetry) {
    const previous = await stateStore.load();
    await recoverPendingCaptures(previous.runs, async run => {
      if (run.attempts?.length) return;
      const capture = new PiRunCapture(telemetry, run.run_id, run.session_id, path.join(runtimeRoot, "sessions", run.session_id), "stopped");
      await capture.recover();
    });
  }
  workflowControl?.interruptWorkflowNodes();
  await orchestrator.restore();
  // Undelivered patches land under artifacts/ with or without a profile, so retention always runs.
  const stopRetention = startEvidenceRetention(runtimeRoot, workflowControl ? () => workflowControl.pruneWorkflowContent(Date.now() - 30 * 86400000) : undefined);
  if (process.env.PI_AGENT_CI_REPAIR_ENABLED === "true" && !trustedProfile) throw new Error("CI repair requires PI_AGENT_VERIFICATION_PROFILE");
  ciRuntime = process.env.PI_AGENT_CI_REPAIR_ENABLED === "true" ? await createCiRuntime({ profile: trustedProfile!.profile, root: runtimeRoot, repoPath, worktreeRoot, orchestrator, telemetry, tracing, verificationBases, priorActive }) : undefined;
  let processHealthy = true;
  guardProcess(process, () => { processHealthy = false; });
  const runtimeHealthy = () => processHealthy && integrityHealthy && stateStore.healthy && (workflowControl?.healthy ?? true) && (ciRuntime?.ci.healthy ?? true);
  const idempotency = new IdempotencyCache({ file: path.join(runtimeRoot, "idempotency.json") });
  await idempotency.load().catch(() => console.error(JSON.stringify({ event: "agent_idempotency_load_failed" })));
  const server = createAgentHttpServer({
    idempotency,
    budgetLedger, budgetSecret: deriveBudgetRpcSecret(gatewaySecret),
    ci: ciRuntime?.ci,
    feedback: telemetry ? (run, payload, id) => telemetry.record(run.run_id, run.session_id, "feedback", payload, id) : undefined,
    orchestrator,
    token,
    acceptRuns: runtimeHealthy,
    health: () => ({
      status: runtimeHealthy() ? "healthy" : "unhealthy",
      runtime: "pi-coding-agent",
      control_store: stateStore.healthy ? "healthy" : "failed",
      version: "1.0.1",
      pi_version: "1.0.1",
      structured_workflow: trustedProfile?.profile.structuredWorkflow ?? { enabled: false },
      agent_container_limit: 4,
      verification_profile: trustedProfile ? { id: trustedProfile.profile.id, checksum: trustedProfile.checksum } : null,
      telemetry: telemetry?.health() ?? {status: "disabled"},
      mcp_mode: mcpMode,
      mcp_gateway_reachable: mcpServerIds.every((id) => mcpServerReachable[id] !== false),
      mcp_servers: mcpServerIds,
      mcp_tool_count: mcpServerIds.reduce((sum, id) => sum + allServerTools(id).length, 0),
      mcp_server_health: Object.fromEntries(mcpServerIds.map((id) => [id, {
        enabled: true, transport: MCP_SERVERS[id].transport, reachable: mcpServerReachable[id] ?? false,
        tool_count: allServerTools(id).length,
        exposure: { direct: MCP_SERVERS[id].directTools.length, codemode: MCP_SERVERS[id].codemodeTools.length },
      }])),
      jev_classifier_mode: jevClassifierMode,
      jev_classifier_model: jevClassifierMode === "on" ? "openrouter/typesafe/jev-1.13" : null,
      jev_classifier_max_calls: jevClassifierMode === "on" ? jevClassifierMaxCalls : 0,
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
      tools: [
        "read", "write", "edit", "bash", "grep", "find", "ls",
        ...(jevClassifierMode === "on" ? ["codemode"] : []),
        ...lock.plugins.flatMap((plugin) => plugin.enabled_tools),
      ],
      plugin_integrity: lock.healthy ? "verified" : "failed",
      skills: skillLock.skills.map((skill) => skill.name),
      skill_integrity: skillLock.healthy ? "verified" : "failed",
      errors: runtimeHealthy() ? [] : [
        ...lock.errors,
        ...skillLock.errors,
        ...(!stateStore.healthy ? ["Runtime control store is unavailable"] : []),
        ...(!processHealthy ? ["An unhandled rejection occurred; restart the runtime"] : []),
        ...(mcpServerIds.some((id) => mcpServerReachable[id] === false) ? ["An optional MCP gateway is unreachable"] : []),
      ],
    }),
  });
  server.on("close", () => {
    orchestrator.shutdown().finally(async () => { await ciRuntime?.close(); await stopRetention?.(); releaseCoordinator?.(); releaseHostCoordinator(); hostCoordinator.close(); workflowControl?.close(); budgetLedger.close(); await telemetry?.close().catch(() => {}); await tracing?.close().catch(() => {}); })
      .catch(() => console.error(JSON.stringify({ event: "agent_runtime_shutdown_failed" })));
  });
  return server;
  } catch (error) { releaseHostCoordinator(); hostCoordinator.close(); throw error; }
}

export type JevClassifierMode = "off" | "on";

export function parseJevClassifierMode(value: string | undefined): JevClassifierMode {
  if (value === undefined || value === "" || value === "off") return "off";
  if (value === "on") return "on";
  throw new Error("PI_AGENT_JEV_CLASSIFIER_MODE must be off or on");
}

async function probeMcpGateway(gatewayUrl: string): Promise<boolean> {
  try {
    const url = new URL(gatewayUrl);
    const healthUrl = new URL("/healthz", url);
    const localPorts: Record<string, string> = { "mcp-github": "8090", "mcp-clickhouse-gateway": "8091", "mcp-context7": "8092", "mcp-gateway": "8090" };
    if (localPorts[url.hostname]) {
      healthUrl.hostname = "127.0.0.1";
      healthUrl.port = localPorts[url.hostname]!;
    }
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
    throw new Error("Cannot confirm all orphan writers/verifiers/graders stopped; refusing rollback/startup");
  }
}

export function authorized(request: Pick<IncomingMessage, "headers">, token: string): boolean {
  const header = request.headers.authorization;
  if (!token || typeof header !== "string") return false;
  const expected = Buffer.from(`Bearer ${token}`, "utf8");
  const actual = Buffer.from(header, "utf8");
  // timingSafeEqual requires equal lengths; the length itself is not secret.
  return actual.length === expected.length && timingSafeEqual(actual, expected);
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
/** The writer gets run_checks (tool, extension and its context) only when a trusted profile can verify. */
export function writerTools(profile: VerificationProfile | undefined, extensionPaths: string[], toolNames: string[]): { extensionPaths: string[]; toolNames: string[]; checksContext?: ChecksContext } {
  if (!profile) return { extensionPaths, toolNames };
  const checks = profile.checks.map((check) => check.name);
  return {
    extensionPaths: [...extensionPaths, "/opt/pi/extensions/checks.ts"],
    toolNames: [...toolNames, "run_checks"],
    checksContext: { checks, defaultChecks: profile.writerChecks?.defaultChecks ?? checks, maxCalls: profile.writerChecks?.maxCalls ?? 4 },
  };
}

/** Constraints the writer sees up front and the policy extension enforces at write time. */
export function writerContextFor(profile: VerificationProfile): WriterContext {
  return { allowedPaths: [...profile.allowedPaths], checks: profile.checks.map((check) => check.name), rubric: [...profile.rubric], diffLimits: { ...DEFAULT_DIFF_LIMITS } };
}

const TEST_RUNNERS = new Set(["npm test", "npm run test", "pytest", "python -m pytest"]);

/** With a trusted profile the writer tests through run_checks (no network, no token, baseline config), not bash. */
export function parseSafeCommands(value: string | undefined, options: { profiled: boolean } = { profiled: false }): string[] {
  const defaults = ["npm test", "npm run test", "npm run lint", "npm run build", "pytest", "python -m pytest"];
  if (!value) return options.profiled ? defaults.filter(command => !TEST_RUNNERS.has(command)) : defaults;
  return value.split("||").map((command) => command.trim()).filter(Boolean);
}
/**
 * Node ends the process on an unhandled rejection; the runtime instead logs it and goes unhealthy,
 * so it stops taking new runs while keeping worktrees and evidence for an operator.
 */
export function guardProcess(target: NodeJS.EventEmitter, onFailure: (error: unknown) => void, log: (line: string) => void = line => process.stderr.write(`${line}\n`)): () => void {
  const handler = (reason: unknown) => {
    log(JSON.stringify({ event: "agent_runtime_unhandled_rejection", error: reason instanceof Error ? reason.name : typeof reason }));
    onFailure(reason);
  };
  target.on("unhandledRejection", handler);
  return () => { target.off("unhandledRejection", handler); };
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
  for (const signal of ["SIGTERM", "SIGINT"] as const) process.once(signal, () => server.close());
}

function parseTaskMode(value: unknown) {
  try { return taskMode(value); } catch { throw new RuntimeError("Invalid task_mode; expected discuss or implement", "invalid_request", 400); }
}

function parseWorkflowMode(value: unknown) {
  try { return workflowMode(value); } catch { throw new RuntimeError('Invalid workflow_mode','invalid_request',400); }
}

/** Runs `work` once per Idempotency-Key and request body; without a key or a cache it just runs. */
async function idempotentRequest<T>(cache: IdempotencyCache | undefined, request: IncomingMessage, route: string, body: unknown, work: () => Promise<T>): Promise<T> {
  const header = request.headers["idempotency-key"];
  if (!cache || typeof header !== "string" || !header) return work();
  if (header.length > 200 || !/^[\x21-\x7e]+$/.test(header)) throw new RuntimeError("Idempotency-Key must be 1-200 visible ASCII characters", "invalid_request", 400);
  const fingerprint = createHash("sha256").update(JSON.stringify(body)).digest("hex");
  try { return await cache.run(`${route}\n${header}`, fingerprint, work); }
  catch (error) {
    if (error instanceof IdempotencyKeyReusedError) throw new RuntimeError("Idempotency-Key was already used for a different request", "idempotency_key_reused", 422);
    throw error;
  }
}
