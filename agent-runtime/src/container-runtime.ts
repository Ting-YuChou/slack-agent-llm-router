import { execFile, spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { promisify } from "node:util";

import { PiRpcBridge, RpcProtocolError, type PublicRunEvent, type UiResponse } from "./pi-rpc.js";
import type { AgentProcess } from "./orchestrator.js";
import { resolveAgentModel } from "./agent-model.js";
import type { AgentReasoningEffort } from "./agent-model.js";
import type { RouteDecision } from "./jev-router.js";
import { limitsForRoute, type AgentLimitConfig, type AgentLimits } from "./agent-limits.js";
import type { AgentTracing } from "./tracing.js";
import { PiRunCapture } from "./pi-capture.js";
import type { AgentTelemetry } from "./telemetry.js";
import type { McpRunConfig, McpServerId } from "./mcp-config.js";
import { MCP_SERVERS, namespacedMcpToolNames } from "./mcp-config.js";
import { StartupTiming } from "./startup-timing.js";
import { taskMode, type TaskMode } from "./task-mode.js";
import { agentScheduler } from "./agent-scheduler.js";
import type { DiffLimits } from "./worktree-manager.js";
import { RESOURCE_LIMIT_FLAGS } from "./container-flags.js";
import { CREDENTIAL_PLACEHOLDER, type RunCredentials } from "./run-credentials.js";
import type { SealStatus } from "./session-seal.js";

export interface AgentContainerOptions {
  taskMode?: TaskMode;
  /** Runs once the process is closed and its container stopped (e.g. removing the session's network). */
  onClosed?: () => Promise<void>;
  name: string;
  image: string;
  network: string;
  worktreePath: string;
  /** The session's own Git mirror (see writerGitDir); never the host repository's git-common-dir. */
  gitMetadataPath: string;
  sessionStatePath: string;
  gatewayUrl: string;
  /** Enables the run-time Jev classifier; its token, like the model's, reaches Pi only through the credential dialog. */
  classifier?: boolean;
  modelRef: string;
  reasoningEffort?: AgentReasoningEffort;
  extensionPaths: string[];
  pluginPaths: string[];
  skillPaths: string[];
  toolNames: string[];
  continueSession?: boolean;
  safeCommands?: string[];
  user?: string;
  mcp?: McpRunConfig;
  readOnlyWorkspace?: boolean;
  graderContext?: { candidate_hash: string; rubric: string[]; allowed_paths?: string[]; diff_limits?: DiffLimits; deleted_paths?: string[] };
  workflowContext?: { role: string; wave: number; paths: string[]; previousTasks: unknown[]; mcpTools?: string[] };
  directMcpOnly?: boolean;
  workflowKey?: string;
  limits?: AgentLimits;
  /** Per-route ceilings; when set, each run's route selects `limits` (see limitsForRoute). */
  routeLimits?: AgentLimitConfig;
  /** Trusted-profile constraints the writer is told about and checked against at write time. */
  writerContext?: WriterContext;
  /** The writer's run_checks tool (extensions/checks.ts): available checks and its call limit. */
  checksContext?: ChecksContext;
  /** `none` lets the writer edit its worktree without prompts; delivery approval must then gate the commit. */
  writeApproval?: WriteApprovalMode;
  /** Pause a settled writer and resume it for the next compatible execution (profile `writerReuse`). */
  reuse?: boolean;
  /** A paused writer older than this is stopped instead of resumed (its MCP tokens date from its start). */
  reuseMaxAgeMs?: number;
  /** Seals the session directory whenever the container stops and checks it before `--continue`. */
  sessionSeal?: { seal(): Promise<void>; verify(): Promise<SealStatus> };
}

export interface ChecksContext { checks: string[]; defaultChecks: string[]; maxCalls: number }

export type WriteApprovalMode = "none" | "per_prompt";

export interface WriterContext { allowedPaths: string[]; checks: string[]; rubric: string[]; diffLimits: { maxFiles: number; maxBytes: number; maxSingleFileBytes: number } }

const exec = promisify(execFile);

/**
 * MCP configuration reaches the container as `--env NAME` resolved from the docker client's own
 * environment, so it never appears on the host process list. Gateway tokens are not here at all: the
 * container starts with a placeholder and Pi asks for each execution's tokens (CREDENTIAL_DIALOG).
 * Discussion runs get no MCP; this mirrors buildAgentDockerArgs so the names match the argv.
 */
export function buildAgentSpawnEnv(options: AgentContainerOptions, host: NodeJS.ProcessEnv): Record<string, string> {
  const discuss = taskMode(options.taskMode) === "discuss";
  return {
    ...(host.PATH ? { PATH: host.PATH } : {}),
    ...(!discuss && options.mcp ? { PI_AGENT_MCP_SERVERS_JSON: JSON.stringify(options.mcp.servers) } : {}),
  };
}

export function buildAgentDockerArgs(options: AgentContainerOptions): string[] {
  if (taskMode(options.taskMode) === "discuss") options = {
    ...options, readOnlyWorkspace: true, toolNames: ["read", "grep", "find", "ls"],
    extensionPaths: ["/opt/pi/extensions/model-gateway.ts", "/opt/pi/extensions/discussion.ts"],
    pluginPaths: [], skillPaths: [], safeCommands: [], mcp: undefined, classifier: undefined,
    graderContext: undefined, workflowContext: undefined, writerContext: undefined, checksContext: undefined,
  };
  const model = resolveAgentModel(options.modelRef);
  const tools = [...options.toolNames];
  const hasCodemodeMcp = !options.directMcpOnly && (options.mcp?.servers.some((item) => MCP_SERVERS[item.server].codemodeTools.length > 0) ?? false);
  if ((options.classifier || hasCodemodeMcp) && !tools.includes("codemode")) tools.push("codemode");
  const directMcpTools = options.mcp?.servers.flatMap((item) =>
    namespacedMcpToolNames(item.server, MCP_SERVERS[item.server].directTools)) ?? [];
  const args = [
    "run", "--rm", "-i",
    "--name", options.name,
    "--label", "slack-pi-agent-session=true",
    "--network", options.network,
    // Names on the internal network still resolve; nothing else is forwarded (CVE-2024-29018 defence in depth).
    "--dns", "127.0.0.1",
    "--read-only",
    "--user", options.user ?? "10001:10001",
    "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges",
    ...RESOURCE_LIMIT_FLAGS,
    "--tmpfs", "/tmp:rw,noexec,nosuid,size=536870912",
    "--env", `HOME=/tmp/pi-home`,
    "--env", `${model.credentialEnv}=${CREDENTIAL_PLACEHOLDER}`,
    "--env", `PI_MODEL_GATEWAY_URL=${options.gatewayUrl}`,
    "--env", `PI_AGENT_MAX_TURNS=${options.limits?.maxTurns ?? 20}`,
    "--env", `PI_AGENT_MAX_TOOL_CALLS=${options.limits?.maxToolCalls ?? 40}`,
    "--env", "PI_AGENT_BASH_TIMEOUT_MS=300000",
    "--env", `PI_AGENT_SAFE_COMMANDS_JSON=${JSON.stringify(options.safeCommands ?? [])}`,
    "--env", `PI_AGENT_SKILL_PATHS_JSON=${JSON.stringify(options.skillPaths)}`,
    "--env", `PI_AGENT_WORKTREE=${options.worktreePath}`,
    "--mount", `type=bind,src=${options.worktreePath},dst=${options.worktreePath}${options.readOnlyWorkspace ? ",readonly" : ""}`,
    ...(!options.readOnlyWorkspace ? ["--mount", `type=bind,src=${options.worktreePath}/.git,dst=${options.worktreePath}/.git,readonly`, "--mount", `type=bind,src=${options.gitMetadataPath},dst=${options.gitMetadataPath},readonly`] : []),
    "--mount", `type=bind,src=${options.sessionStatePath},dst=/var/lib/pi-session`,
    "--workdir", options.readOnlyWorkspace ? "/var/lib/pi-session" : options.worktreePath,
    options.image,
    "pi",
    "--mode", "rpc",
    "--provider", model.provider,
    "--model", model.id,
    "--thinking", options.reasoningEffort ?? model.reasoningEffort,
    "--session-dir", "/var/lib/pi-session",
    "--no-approve",
    "--no-extensions",
    "--no-skills",
    "--no-prompt-templates",
    "--no-themes",
    "--tools", [...tools, ...directMcpTools].join(","),
  ];
  if (options.readOnlyWorkspace) args.push("--no-context-files");
  if (options.continueSession) args.push("--continue");
  if (options.graderContext) args.splice(args.indexOf("--mount"), 0, "--env", `PI_AGENT_GRADER_JSON=${JSON.stringify(options.graderContext)}`);
  if (options.workflowContext) args.splice(args.indexOf("--mount"), 0, "--env", `PI_AGENT_WORKFLOW_JSON=${JSON.stringify(options.workflowContext)}`);
  if (options.writerContext) args.splice(args.indexOf("--mount"), 0, "--env", `PI_AGENT_WRITER_JSON=${JSON.stringify(options.writerContext)}`);
  if (options.checksContext) args.splice(args.indexOf("--mount"), 0, "--env", `PI_AGENT_CHECKS_JSON=${JSON.stringify(options.checksContext)}`);
  if (options.writeApproval) args.splice(args.indexOf("--mount"), 0, "--env", `PI_AGENT_WRITE_APPROVAL=${options.writeApproval}`);
  if (options.classifier) {
    args.splice(args.indexOf("--mount"), 0,
      "--env", `OPENROUTER_API_KEY=${CREDENTIAL_PLACEHOLDER}`,
      "--env", "PI_AGENT_JEV_CLASSIFIER_MODE=on",
    );
  }
  if (options.mcp) {
    args.splice(args.indexOf("--mount"), 0,
      "--env", `PI_AGENT_MCP_MODE=${options.mcp.mode}`,
      "--env", "PI_AGENT_MCP_SERVERS_JSON",
    );
    args.push("-e", "builtin:mcp", "-e", "/opt/pi/extensions/mcp-bootstrap.ts");
  }
  if (options.classifier || hasCodemodeMcp) args.push("-e", "builtin:codemode");
  for (const extension of [...options.extensionPaths, ...options.pluginPaths]) args.push("-e", extension);
  for (const skill of options.skillPaths) args.push("--skill", skill);
  return args;
}

export class DockerPiProcess implements AgentProcess {
  queueMs = 0;
  private activeRunId?: string;
  private readonly tracing?: AgentTracing;
  private capture?: PiRunCapture;
  private readonly telemetry?: AgentTelemetry;
  private readonly sessionId: string;
  private child?: ChildProcessWithoutNullStreams;
  private bridge?: PiRpcBridge;
  private closing = false;
  private hasSession = false;
  private activeContainerName?: string;
  private stopping?: Promise<void>;
  private readiness?: Promise<void>;
  private startup?: StartupTiming;
  private startupAbort?: AbortController;
  private queueAbort?: AbortController;
  private queuedStart?: Promise<void>;
  private releasePermit?: () => void;
  private closedHookRan = false;
  /** The settled writer is frozen (docker pause) and still holds its execution slot. */
  private paused = false;
  /** Whether the current execution resumed a paused container rather than starting one. */
  private reused = false;
  private activeTask?: TaskMode;
  private containerKey?: string;
  private containerStartedAt = 0;
  private credentials?: RunCredentials;
  private sealedChild?: ChildProcessWithoutNullStreams;
  private readonly options: Omit<AgentContainerOptions, "classifier" | "mcp"> & { mcp?: Omit<McpRunConfig, "token"> };
  private readonly tokenForRun: (runId: string, route: RouteDecision) => string;
  private readonly onEvent: (event: PublicRunEvent) => void;
  private readonly mcpTokenForRun?: (runId: string) => Partial<Record<McpServerId, string>>;
  private readonly classifierTokenForRun?: (runId: string) => string;

  constructor(
    options: Omit<AgentContainerOptions, "classifier" | "mcp"> & { mcp?: Omit<McpRunConfig, "token"> },
    tokenForRun: (runId: string, route: RouteDecision) => string,
    onEvent: (event: PublicRunEvent) => void,
    mcpTokenForRun?: (runId: string) => Partial<Record<McpServerId, string>>,
    classifierTokenForRun?: (runId: string) => string,
    telemetry?: AgentTelemetry,
    sessionId = options.name,
    tracing?: AgentTracing,
    private readonly onRecord?: (record: Record<string, unknown>) => void,
    private readonly onPrompt?: (runId: string) => void,
  ) {
    this.options = options;
    this.telemetry = telemetry; this.sessionId = sessionId; this.tracing = tracing;
    this.tokenForRun = tokenForRun;
    this.onEvent = onEvent;
    this.mcpTokenForRun = mcpTokenForRun;
    this.classifierTokenForRun = classifierTokenForRun;
    this.hasSession = options.continueSession ?? false;
  }

  /** The turn and tool-call limits a run on this route is started with. */
  limitsFor(route: RouteDecision): AgentLimits | undefined {
    return this.options.routeLimits ? limitsForRoute(route, this.options.routeLimits) : this.options.limits;
  }

  private start(runId: string, route: RouteDecision, selectedTask: TaskMode): void {
    if (this.child && this.child.exitCode === null) throw new Error("Pi container is already active");
    this.closing = false;
    this.activeRunId = runId;
    this.startup = new StartupTiming((name, phase, failed) => this.tracing?.lifecycle(runId, name, phase, failed));
    this.startup.start();
    this.startupAbort = new AbortController();
    const containerName = `${this.options.name}-${runId.slice(0, 8)}`;
    this.activeContainerName = containerName;
    const capture = this.telemetry ? new PiRunCapture(this.telemetry, runId, this.sessionId, this.options.sessionStatePath, containerName) : undefined;
    this.capture = capture;
    const { mcp: configuredMcp, ...baseOptions } = this.options;
    const mcp = selectedTask === "discuss" ? undefined : configuredMcp;
    if (mcp && !this.mcpTokenForRun) throw new Error("MCP token issuer is required when MCP is enabled");
    const classifier = selectedTask !== "discuss" && this.classifierTokenForRun !== undefined;
    this.credentials = this.credentialsFor(runId, route, selectedTask);
    this.activeTask = selectedTask; this.reused = false; this.containerStartedAt = Date.now();
    this.containerKey = this.reuseKey(route, selectedTask);
    const containerOptions: AgentContainerOptions = {
      ...baseOptions,
      taskMode: selectedTask,
      name: containerName,
      modelRef: route.modelRef,
      reasoningEffort: route.effort,
      limits: this.limitsFor(route),
      ...(classifier ? { classifier } : {}),
      ...(mcp ? { mcp: {
        ...mcp,
        servers: mcp.servers.map((item) => item.server === "codegraph" ? item : ({
          ...item, token: this.mcpTokenForRun!(runId)[item.server] ?? "",
        })),
      } } : {}),
      continueSession: this.hasSession,
    };
    const child = spawn("docker", buildAgentDockerArgs(containerOptions), {
      stdio: ["pipe", "pipe", "pipe"],
      env: buildAgentSpawnEnv(containerOptions, process.env),
    });
    this.child = child;
    const bridge: PiRpcBridge = new PiRpcBridge({
      credentials: () => this.bridge === bridge ? this.credentials : undefined,
      onReady: () => this.startup?.rpcReady(),
      writeLine: (line) => child.stdin.write(line),
      // The bridge outlives one execution when the container is reused; always attribute to the current one.
      onCommand: record => {
        this.capture?.onCommand(record);
        const current = this.activeRunId ?? runId;
        if (record.type === "prompt") {
          this.startup?.configured(); this.startup?.promptSent();
          if (this.startup) this.onEvent({ type: "startup", startup: { reused: this.reused, queue_ms: this.queueMs, ...this.startup.summary() } });
          this.onPrompt?.(current);
          this.tracing?.log(current, "agent.prompt_sent");
        }
      },
      onRecord: record => { this.capture?.onRecord(record); this.tracing?.rpc(this.activeRunId ?? runId, record); this.onRecord?.(record); },
      onEvent: (event) => {
        if ((event.type === "answer" || event.type === "answer_delta") && event.text.trim()) this.startup?.firstOutput();
        if (event.type === "settled") {
          this.hasSession = true;
          // A completed writer may be repaired: freeze it (verification sees a still worktree) instead of stopping it.
          const keep = this.options.reuse === true && event.outcome === "completed" && this.activeTask !== "discuss";
          void this.finalizeCapture(bridge, this.capture).then(() => this.bridge !== bridge ? undefined : keep ? this.pauseActiveContainer() : this.stopActiveContainer()).then(
            () => { if (this.bridge === bridge) this.onEvent(event); },
            () => { if (this.bridge === bridge) this.onEvent({
              type: "error",
              code: "container_stop_failed",
              message: "Agent container could not be confirmed stopped",
            }); },
          );
          return;
        }
        this.onEvent(event);
      },
    });
    this.bridge = bridge;
    child.once("spawn", () => this.startup?.cliSpawned());
    this.readiness = this.waitForContainer(containerName, this.startupAbort.signal).then(() => this.startup?.containerRunning());
    // The prompt path observes this rejection; attach immediately to avoid a startup race.
    void this.readiness.catch(() => this.startup?.fail());
    child.stdout.on("data", (chunk: Buffer) => {
      try {
        bridge.feed(chunk);
      } catch (error) {
        this.onEvent({
          type: "error",
          code: error instanceof RpcProtocolError ? "rpc_protocol_error" : "runtime_error",
          message: "Pi RPC stream failed validation",
        });
        child.kill("SIGTERM");
      }
    });
    child.on("error", () => {
      bridge.failPending();
      this.onEvent({ type: "error", code: "container_start_failed", message: "Agent container could not start" });
    });
    child.on("exit", (code) => {
      bridge.failPending();
      if (this.child === child) { this.activeContainerName = undefined; this.paused = false; }
      if (!this.closing && code !== 0) this.onEvent({ type: "error", code: "container_exited", message: "Agent container exited unexpectedly" });
    });
    child.stderr.resume();
  }

  prompt(runId: string, prompt: string, route?: RouteDecision, requestedTask?: TaskMode): void {
    const selectedTask = taskMode(requestedTask ?? this.options.taskMode);
    const selected = route ?? { modelRef: this.options.modelRef, effort: resolveAgentModel(this.options.modelRef).reasoningEffort, source: "explicit" };
    this.queueAbort?.abort(); const controller = new AbortController(); this.queueAbort = controller;
    this.queuedStart = (async () => {
      if (await this.resume(runId, prompt, selected, selectedTask)) return;
      await this.stopActiveContainer(); controller.signal.throwIfAborted();
      const queuedAt = Date.now();
      this.tracing?.lifecycle(runId, "queue", "start");
      const release = await agentScheduler.acquire(this.options.workflowKey ?? this.sessionId, controller.signal);
      if (controller.signal.aborted) { release(); return; }
      this.queueMs = Date.now() - queuedAt;
      this.releasePermit = release; this.tracing?.lifecycle(runId, "queue", "end");
      if (!await this.sessionIntact()) {
        this.releasePermit?.(); this.releasePermit = undefined;
        this.onEvent({ type: "error", code: "session_integrity_failed", message: "The agent session was modified after it was saved; it cannot be continued" });
        return;
      }
      this.start(runId, selected, selectedTask);
      this.dispatch(runId, prompt, selected);
    })().catch(() => { if (!controller.signal.aborted) this.onEvent({ type: "error", code: "container_admission_failed", message: "Agent could not acquire/start an execution slot" }); });
  }

  /** A session Pi will `--continue` must still be the one sealed when its last container stopped. */
  private async sessionIntact(): Promise<boolean> {
    if (!this.hasSession || !this.options.sessionSeal) return true;
    try {
      const status = await this.options.sessionSeal.verify();
      if (status === "unsealed") console.error(JSON.stringify({ event: "agent_session_unsealed", session_id: this.sessionId }));
      return status !== "tampered";
    } catch { return false; }
  }

  /** Configures the ready container's model for this execution and sends the prompt. */
  private dispatch(runId: string, prompt: string, route: RouteDecision): void {
    const model = resolveAgentModel(route.modelRef);
    const bridge = this.bridge!; const capture = this.capture;
    void this.readiness!.then(() => bridge.configureAndPrompt(runId, prompt, model.provider, model.id, route.effort, async () => {
      await capture?.beforePrompt(bridge);
      if (this.bridge !== bridge || this.closing) throw new RpcProtocolError("Pi configuration was interrupted");
    })).catch(() => {
      if (this.bridge !== bridge || this.closing) return;
      this.onEvent({ type: "error", code: "pi_configuration_failed", message: "Pi model configuration could not be verified" });
      this.stopActiveContainer().catch(() => {
        if (this.bridge === bridge) this.onEvent({ type: "error", code: "container_stop_failed", message: "Agent container could not be confirmed stopped" });
      });
    });
  }

  /** Each execution's own tokens, issued now and handed to Pi only through the credential dialog. */
  private credentialsFor(runId: string, route: RouteDecision, task: TaskMode): RunCredentials {
    return {
      [resolveAgentModel(route.modelRef).credentialEnv]: this.tokenForRun(runId, route),
      ...(task !== "discuss" && this.classifierTokenForRun ? { OPENROUTER_API_KEY: this.classifierTokenForRun(runId) } : {}),
    };
  }

  /** What a container is fixed to at start (provider key name, limits, task); a reused one must match. */
  private reuseKey(route: RouteDecision, task: TaskMode): string {
    return JSON.stringify([route.modelRef, this.limitsFor(route) ?? null, task]);
  }

  /** Freezes a settled writer, keeping its execution slot, so a repair can resume it; stops it if it cannot. */
  private async pauseActiveContainer(): Promise<void> {
    const name = this.activeContainerName;
    if (!name || !this.child || this.child.exitCode !== null) return;
    try {
      await exec("docker", ["pause", name], { env: { PATH: process.env.PATH } });
      this.paused = true;
    } catch { await this.stopActiveContainer(); }
  }

  /**
   * Resumes the paused writer for a compatible execution with that execution's credentials; false when a
   * fresh container must start instead (none paused, a different model or limits, older than
   * reuseMaxAgeMs, or unpause failed). MCP tokens, unlike model tokens, stay those issued at start: they
   * belong to the same run, which only a run's own repairs reuse, and are revoked when it ends.
   */
  private async resume(runId: string, prompt: string, route: RouteDecision, task: TaskMode): Promise<boolean> {
    const name = this.activeContainerName;
    if (!this.paused || !name || !this.bridge || this.child?.exitCode !== null || this.containerKey !== this.reuseKey(route, task)) return false;
    if (this.options.reuseMaxAgeMs !== undefined && Date.now() - this.containerStartedAt > this.options.reuseMaxAgeMs) return false;
    const startup = new StartupTiming((stage, phase, failed) => this.tracing?.lifecycle(runId, stage, phase, failed));
    startup.start();
    try { await exec("docker", ["unpause", name], { env: { PATH: process.env.PATH } }); }
    catch {
      startup.fail();
      this.onEvent({ type: "container_reuse_failed" });
      return false;
    }
    this.paused = false;
    startup.cliSpawned(); startup.containerRunning(); startup.rpcReady();
    this.activeRunId = runId; this.startup = startup; this.queueMs = 0; this.reused = true;
    this.credentials = this.credentialsFor(runId, route, task);
    this.capture = this.telemetry ? new PiRunCapture(this.telemetry, runId, this.sessionId, this.options.sessionStatePath, name) : undefined;
    this.readiness = Promise.resolve();
    this.dispatch(runId, prompt, route);
    return true;
  }

  /** Stops a writer kept paused for reuse and frees its slot; nothing to do when none is paused. */
  async release(): Promise<void> {
    if (this.paused) await this.stopActiveContainer();
  }
  decide(rpcUiId: string, approved: boolean): void { if(this.activeRunId) this.tracing?.approval(this.activeRunId,rpcUiId,approved); this.bridge?.respondToUi(rpcUiId, approved); }
  respondUi(rpcUiId: string, response: UiResponse): void { this.bridge?.respondUi(rpcUiId, response); }
  /** Ends the run's execution; the session stays open, so what it owns (its network) is kept for the next prompt. */
  async abort(runId: string): Promise<void> {
    const capture = this.capture;
    capture?.cancel();
    this.bridge?.abort(`abort-${runId}`);
    await this.stopExecution();
    await capture?.recover();
  }

  /** Ends the session: stops its execution, then runs the session's cleanup once. */
  async close(): Promise<void> {
    await this.stopExecution();
    const closed = this.closedHookRan ? undefined : this.options.onClosed;
    this.closedHookRan = true;
    // Cleanup after the container is gone must not turn a clean close into a failure.
    await closed?.().catch(() => console.error(JSON.stringify({ event: "agent_session_cleanup_failed", session_id: this.sessionId })));
  }

  private async stopExecution(): Promise<void> {
    this.queueAbort?.abort();
    await this.queuedStart;
    await this.stopActiveContainer();
  }

  private async waitForContainer(name: string, signal: AbortSignal): Promise<void> {
    const until = Date.now() + 15_000;
    while (Date.now() < until) {
      signal.throwIfAborted();
      try {
        const result = await exec("docker", ["inspect", "--format={{.State.Running}}", name], { timeout: 1000, signal, env: { PATH: process.env.PATH } });
        if (result.stdout.trim() === "true") return;
      } catch { signal.throwIfAborted(); }
      if (this.child?.exitCode !== null) throw new Error("Container exited before readiness");
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    throw new Error("Container readiness timed out");
  }

  private async finalizeCapture(bridge: PiRpcBridge, capture?: PiRunCapture): Promise<void> {
    if (!capture) return;
    let timer: ReturnType<typeof setTimeout> | undefined;
    try { await Promise.race([capture.finalize(bridge), new Promise<void>(resolve => {
      timer = setTimeout(() => { capture.cancel(); this.telemetry?.markIncomplete(capture.runId, "finalization_timeout"); resolve(); }, 10_000);
    })]); } finally { if (timer) clearTimeout(timer); }
  }

  private stopActiveContainer(): Promise<void> {
    if (this.stopping) return this.stopping;
    this.stopping = this.stopActiveContainerOnce().then(() => { this.releasePermit?.(); this.releasePermit = undefined; }).finally(() => {
      this.stopping = undefined;
    });
    return this.stopping;
  }

  private async stopActiveContainerOnce(): Promise<void> {
    this.startupAbort?.abort();
    const child = this.child;
    if (!child) return;
    if (child.exitCode !== null) { await this.sealSession(child); return; }
    this.closing = true;
    if(this.activeRunId) this.tracing?.lifecycle(this.activeRunId, "container_stop", "start");
    const containerName = this.activeContainerName;
    // Older engines refuse to stop a paused container; a failed unpause still leaves stop to try.
    if (containerName && this.paused) await exec("docker", ["unpause", containerName], { env: { PATH: process.env.PATH } }).catch(() => undefined);
    this.paused = false;
    if (containerName) {
      try {
        await exec("docker", ["stop", "--time", "5", containerName], {
          encoding: "utf8",
          env: { PATH: process.env.PATH },
        });
      } catch (stopError) {
        let running: string;
        try {
          running = (await exec("docker", ["inspect", "--format={{.State.Running}}", containerName], {
            encoding: "utf8",
            env: { PATH: process.env.PATH },
          })).stdout.trim();
        } catch (inspectError) {
          const message = errorMessage(inspectError);
          if (!/No such (?:object|container)/i.test(message)) {
            throw new Error("Docker could not confirm that the Agent container stopped", { cause: stopError });
          }
          running = "false";
        }
        if (running === "true") {
          try {
            await exec("docker", ["kill", containerName], {
              encoding: "utf8",
              env: { PATH: process.env.PATH },
            });
          } catch (killError) {
            throw new Error("Docker could not stop the Agent container", { cause: killError });
          }
        } else if (running !== "false") {
          throw new Error("Docker returned an invalid Agent container state");
        }
      }
    } else {
      throw new Error("Agent container identity was lost before shutdown");
    }
    await new Promise<void>((resolve) => {
      if (child.exitCode !== null) { resolve(); return; }
      const kill = setTimeout(() => { child.kill("SIGKILL"); resolve(); }, 2_000);
      kill.unref();
      child.once("exit", () => {
        clearTimeout(kill);
        resolve();
      });
    });
    if(this.activeRunId) this.tracing?.lifecycle(this.activeRunId, "container_stop", "end");
    await this.sealSession(child);
  }

  /** Seals what Pi left in the session directory, once per container, after it has stopped. */
  private async sealSession(child: ChildProcessWithoutNullStreams): Promise<void> {
    if (!this.options.sessionSeal || this.sealedChild === child) return;
    this.sealedChild = child;
    // A failed seal leaves the previous one, against which the next --continue reads as tampered.
    await this.options.sessionSeal.seal().catch(() => console.error(JSON.stringify({ event: "agent_session_seal_failed", session_id: this.sessionId })));
  }
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return `${error.message} ${(error as Error & { stderr?: string }).stderr ?? ""}`;
  return String(error);
}
