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
    // Issued per execution and handed to Pi only through the credential dialog, never its environment.
    const credentials: RunCredentials = {
      [resolveAgentModel(route.modelRef).credentialEnv]: this.tokenForRun(runId, route),
      ...(classifier ? { OPENROUTER_API_KEY: this.classifierTokenForRun!(runId) } : {}),
    };
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
    const bridge = new PiRpcBridge({
      credentials: () => credentials,
      onReady: () => this.startup?.rpcReady(),
      writeLine: (line) => child.stdin.write(line),
      onCommand: record => {
        capture?.onCommand(record);
        if (record.type === "prompt") {
          this.startup?.configured(); this.startup?.promptSent();
          if (this.startup) this.onEvent({ type: "startup", startup: { reused: false, queue_ms: this.queueMs, ...this.startup.summary() } });
          this.onPrompt?.(runId);
          this.tracing?.log(runId, "agent.prompt_sent");
        }
      },
      onRecord: record => { capture?.onRecord(record); this.tracing?.rpc(runId, record); this.onRecord?.(record); },
      onEvent: (event) => {
        if ((event.type === "answer" || event.type === "answer_delta") && event.text.trim()) this.startup?.firstOutput();
        if (event.type === "settled") {
          this.hasSession = true;
          void this.finalizeCapture(bridge, capture).then(() => this.bridge === bridge ? this.stopActiveContainer() : undefined).then(
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
      if (this.child === child) this.activeContainerName = undefined;
      if (!this.closing && code !== 0) this.onEvent({ type: "error", code: "container_exited", message: "Agent container exited unexpectedly" });
    });
    child.stderr.resume();
  }

  prompt(runId: string, prompt: string, route?: RouteDecision, requestedTask?: TaskMode): void {
    const selectedTask = taskMode(requestedTask ?? this.options.taskMode);
    const selected = route ?? { modelRef: this.options.modelRef, effort: resolveAgentModel(this.options.modelRef).reasoningEffort, source: "explicit" };
    const startPrompt = () => {
      this.start(runId, selected, selectedTask);
      const model = resolveAgentModel(selected.modelRef);
      const bridge = this.bridge!; const capture = this.capture;
      void this.readiness!.then(() => bridge.configureAndPrompt(runId, prompt, model.provider, model.id, selected.effort, async () => {
        await capture?.beforePrompt(bridge);
        if (this.bridge !== bridge || this.closing) throw new RpcProtocolError("Pi configuration was interrupted");
      })).catch(() => {
        if (this.bridge !== bridge || this.closing) return;
        this.onEvent({ type: "error", code: "pi_configuration_failed", message: "Pi model configuration could not be verified" });
        this.stopActiveContainer().catch(() => {
          if (this.bridge === bridge) this.onEvent({ type: "error", code: "container_stop_failed", message: "Agent container could not be confirmed stopped" });
        });
      });
    };
    this.queueAbort?.abort(); const controller = new AbortController(); this.queueAbort = controller;
    this.queuedStart = (async () => {
      await this.stopActiveContainer(); controller.signal.throwIfAborted();
      const queuedAt = Date.now();
      this.tracing?.lifecycle(runId, "queue", "start");
      const release = await agentScheduler.acquire(this.options.workflowKey ?? this.sessionId, controller.signal);
      if (controller.signal.aborted) { release(); return; }
      this.queueMs = Date.now() - queuedAt;
      this.releasePermit = release; this.tracing?.lifecycle(runId, "queue", "end"); startPrompt();
    })().catch(() => { if (!controller.signal.aborted) this.onEvent({ type: "error", code: "container_admission_failed", message: "Agent could not acquire/start an execution slot" }); });
  }
  decide(rpcUiId: string, approved: boolean): void { if(this.activeRunId) this.tracing?.approval(this.activeRunId,rpcUiId,approved); this.bridge?.respondToUi(rpcUiId, approved); }
  respondUi(rpcUiId: string, response: UiResponse): void { this.bridge?.respondUi(rpcUiId, response); }
  async abort(runId: string): Promise<void> {
    const capture = this.capture;
    capture?.cancel();
    this.bridge?.abort(`abort-${runId}`);
    await this.close();
    await capture?.recover();
  }

  async close(): Promise<void> {
    this.queueAbort?.abort();
    await this.queuedStart;
    await this.stopActiveContainer();
    const closed = this.closedHookRan ? undefined : this.options.onClosed;
    this.closedHookRan = true;
    // Cleanup after the container is gone must not turn a clean close into a failure.
    await closed?.().catch(() => console.error(JSON.stringify({ event: "agent_session_cleanup_failed", session_id: this.sessionId })));
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
    if (!child || child.exitCode !== null) return;
    this.closing = true;
    if(this.activeRunId) this.tracing?.lifecycle(this.activeRunId, "container_stop", "start");
    const containerName = this.activeContainerName;
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
  }
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return `${error.message} ${(error as Error & { stderr?: string }).stderr ?? ""}`;
  return String(error);
}
