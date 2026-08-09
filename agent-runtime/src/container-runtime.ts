import { execFile, spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { promisify } from "node:util";

import {
  PiRpcBridge,
  RpcProtocolError,
  type PublicRunEvent,
  type RpcSessionState,
  type RpcSessionStats,
  type RpcTreeSnapshot,
} from "./pi-rpc.js";
import type { AgentProcess, ProcessCheckpoint } from "./orchestrator.js";
import { resolveAgentModel } from "./agent-model.js";

export interface AgentContainerOptions {
  name: string;
  image: string;
  network: string;
  worktreePath: string;
  gitMetadataPath: string;
  sessionStatePath: string;
  gatewayUrl: string;
  gatewayToken: string;
  modelRef: string;
  extensionPaths: string[];
  pluginPaths: string[];
  skillPaths: string[];
  toolNames: string[];
  piSessionId: string;
  piSessionFile?: string;
  safeCommands?: string[];
  user?: string;
}

const exec = promisify(execFile);

export function buildAgentDockerArgs(options: AgentContainerOptions): string[] {
  const model = resolveAgentModel(options.modelRef);
  const args = [
    "run", "--rm", "-i",
    "--name", options.name,
    "--label", "slack-pi-agent-session=true",
    "--network", options.network,
    "--read-only",
    "--user", options.user ?? "10001:10001",
    "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges",
    "--cpus", "2",
    "--memory", "2g",
    "--pids-limit", "256",
    "--tmpfs", "/tmp:rw,noexec,nosuid,size=536870912",
    "--env", `HOME=/tmp/pi-home`,
    "--env", `${model.credentialEnv}=${options.gatewayToken}`,
    "--env", `PI_MODEL_GATEWAY_URL=${options.gatewayUrl}`,
    "--env", "PI_AGENT_MAX_TURNS=20",
    "--env", "PI_AGENT_MAX_TOOL_CALLS=40",
    "--env", "PI_AGENT_BASH_TIMEOUT_MS=300000",
    "--env", `PI_AGENT_SAFE_COMMANDS_JSON=${JSON.stringify(options.safeCommands ?? [])}`,
    "--env", `PI_AGENT_SKILL_PATHS_JSON=${JSON.stringify(options.skillPaths)}`,
    "--mount", `type=bind,src=${options.worktreePath},dst=${options.worktreePath}`,
    "--mount", `type=bind,src=${options.worktreePath}/.git,dst=${options.worktreePath}/.git,readonly`,
    "--mount", `type=bind,src=${options.gitMetadataPath},dst=${options.gitMetadataPath},readonly`,
    "--mount", `type=bind,src=${options.sessionStatePath},dst=/var/lib/pi-session`,
    "--workdir", options.worktreePath,
    options.image,
    "pi",
    "--mode", "rpc",
    "--provider", model.provider,
    "--model", model.id,
    "--thinking", model.reasoningEffort,
    "--session-dir", "/var/lib/pi-session",
    "--approve",
    "--no-extensions",
    "--no-skills",
    "--no-prompt-templates",
    "--no-themes",
    "--tools", options.toolNames.join(","),
  ];
  if (options.piSessionFile) {
    args.push("--session", `/var/lib/pi-session/${validatedSessionBasename(options.piSessionFile)}`);
  } else {
    args.push("--session-id", options.piSessionId);
  }
  for (const extension of [...options.extensionPaths, ...options.pluginPaths]) args.push("-e", extension);
  for (const skill of options.skillPaths) args.push("--skill", skill);
  return args;
}

export class DockerPiProcess implements AgentProcess {
  private child?: ChildProcessWithoutNullStreams;
  private bridge?: PiRpcBridge;
  private closing = false;
  private activeContainerName?: string;
  private stopping?: Promise<void>;
  private ready?: Promise<void>;
  private latestCheckpoint?: ProcessCheckpoint;
  private operationCheckpoint?: ProcessCheckpoint;
  private readonly options: Omit<AgentContainerOptions, "gatewayToken">;
  private readonly tokenForRun: (runId: string) => string;
  private readonly onEvent: (event: PublicRunEvent) => void;

  constructor(
    options: Omit<AgentContainerOptions, "gatewayToken">,
    tokenForRun: (runId: string) => string,
    onEvent: (event: PublicRunEvent) => void,
  ) {
    this.options = options;
    this.tokenForRun = tokenForRun;
    this.onEvent = onEvent;
  }

  private start(runId: string): void {
    if (this.child && this.child.exitCode === null) throw new Error("Pi container is already active");
    this.closing = false;
    this.operationCheckpoint = undefined;
    const containerName = `${this.options.name}-${runId.slice(0, 8)}`;
    this.activeContainerName = containerName;
    const child = spawn("docker", buildAgentDockerArgs({
      ...this.options,
      name: containerName,
      gatewayToken: this.tokenForRun(runId),
    }), {
      stdio: ["pipe", "pipe", "pipe"],
      env: { PATH: process.env.PATH },
    });
    this.child = child;
    const bridge = new PiRpcBridge({
      writeLine: (line) => child.stdin.write(line),
      onEvent: (event) => {
        if (event.type === "settled") {
          void this.finalizeSettled(event);
          return;
        }
        this.onEvent(event);
      },
    });
    this.bridge = bridge;
    this.ready = bridge.setAutoCompaction(true);
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
    child.on("error", () => this.onEvent({ type: "error", code: "container_start_failed", message: "Agent container could not start" }));
    child.on("exit", (code) => {
      if (this.child === child) this.activeContainerName = undefined;
      if (!this.closing && code !== 0) this.onEvent({ type: "error", code: "container_exited", message: "Agent container exited unexpectedly" });
    });
    child.stderr.resume();
  }

  prompt(runId: string, prompt: string): void {
    if (this.child && this.child.exitCode === null) {
      const previous = this.child;
      previous.once("exit", () => {
        this.start(runId);
        void this.sendPromptWhenReady(runId, prompt);
      });
      if (!this.closing) {
        this.closing = true;
        previous.kill("SIGTERM");
      }
      return;
    }
    this.start(runId);
    void this.sendPromptWhenReady(runId, prompt);
  }
  decide(rpcUiId: string, approved: boolean): void { this.bridge?.respondToUi(rpcUiId, approved); }
  async abort(runId: string): Promise<ProcessCheckpoint | void> {
    if (!this.child || this.child.exitCode !== null) {
      return this.operationCheckpoint ? { ...this.operationCheckpoint } : undefined;
    }
    this.bridge?.abort(`abort-${runId}`);
    let checkpoint: ProcessCheckpoint | undefined;
    try { checkpoint = await this.captureCheckpoint(); } catch { /* caller fails closed if state is uncertain */ }
    await this.stopActiveContainer();
    return checkpoint ?? this.operationCheckpoint;
  }

  async close(): Promise<void> {
    await this.stopActiveContainer();
  }

  async getCheckpoint(): Promise<ProcessCheckpoint> {
    if (!this.latestCheckpoint) throw new Error("Pi session checkpoint is unavailable");
    return { ...this.latestCheckpoint };
  }

  async getState(): Promise<RpcSessionState> {
    return this.withIdleBridge("state", (bridge) => bridge.getState());
  }

  async getSessionStats(): Promise<RpcSessionStats> {
    return this.withIdleBridge("stats", (bridge) => bridge.getSessionStats());
  }

  async getTree(): Promise<RpcTreeSnapshot> {
    return this.withIdleBridge("tree", (bridge) => bridge.getTree());
  }

  async compact(runId: string, customInstructions?: string): Promise<ProcessCheckpoint> {
    if (this.child && this.child.exitCode === null) throw new Error("Pi container is already active");
    this.start(runId);
    try {
      await this.ready;
      await this.bridge!.compact(customInstructions);
      return await this.captureCheckpoint();
    } catch (error) {
      try { await this.captureCheckpoint(); } catch { /* orchestrator marks uncertain state */ }
      throw error;
    } finally {
      await this.stopActiveContainer();
    }
  }

  private async sendPromptWhenReady(runId: string, prompt: string): Promise<void> {
    try {
      await this.ready;
      this.bridge!.prompt(runId, prompt);
    } catch {
      this.onEvent({ type: "error", code: "rpc_initialization_failed", message: "Pi RPC could not enable session safeguards" });
      await this.stopActiveContainer().catch(() => undefined);
    }
  }

  private async finalizeSettled(event: PublicRunEvent): Promise<void> {
    try {
      await this.captureCheckpoint();
    } catch {
      await this.stopActiveContainer().catch(() => undefined);
      this.onEvent({
        type: "error",
        code: "checkpoint_failed",
        message: "Pi session checkpoint could not be verified",
      });
      return;
    }
    try {
      await this.stopActiveContainer();
      this.onEvent(event);
    } catch {
      this.onEvent({
        type: "error",
        code: "container_stop_failed",
        message: "Agent container could not be confirmed stopped",
      });
    }
  }

  private async withIdleBridge<T>(operation: string, callback: (bridge: PiRpcBridge) => Promise<T>): Promise<T> {
    if (this.child && this.child.exitCode === null) throw new Error("Pi session is busy");
    this.start(`${operation}-${Date.now()}`);
    try {
      await this.ready;
      return await callback(this.bridge!);
    } finally {
      await this.stopActiveContainer();
    }
  }

  private async captureCheckpoint(): Promise<ProcessCheckpoint> {
    if (!this.bridge) throw new Error("Pi RPC bridge is unavailable");
    await this.ready;
    const [state, snapshot] = await Promise.all([this.bridge.getState(), this.bridge.getEntries()]);
    if (state.sessionId !== this.options.piSessionId) {
      throw new Error("Pi RPC returned an unexpected session id");
    }
    const sessionFile = sessionBasenameFromState(state.sessionFile);
    const userEntry = [...snapshot.entries].reverse().find((entry) => {
      const message = isRecord(entry.message) ? entry.message : undefined;
      return entry.type === "message" && message?.role === "user";
    });
    this.options.piSessionFile = sessionFile;
    this.latestCheckpoint = {
      piSessionId: state.sessionId,
      sessionFile,
      leafId: snapshot.leafId,
      ...(typeof userEntry?.id === "string" ? { userEntryId: userEntry.id } : {}),
    };
    this.operationCheckpoint = { ...this.latestCheckpoint };
    return { ...this.latestCheckpoint };
  }

  private stopActiveContainer(): Promise<void> {
    if (this.stopping) return this.stopping;
    this.stopping = this.stopActiveContainerOnce().finally(() => {
      this.stopping = undefined;
    });
    return this.stopping;
  }

  private async stopActiveContainerOnce(): Promise<void> {
    const child = this.child;
    if (!child || child.exitCode !== null) return;
    this.closing = true;
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
  }
}

function validatedSessionBasename(value: string): string {
  if (value !== pathBasename(value) || !/^[A-Za-z0-9._-]+\.jsonl$/.test(value)) {
    throw new Error("Pi session file must be a validated JSONL basename");
  }
  return value;
}

function pathBasename(value: string): string {
  return value.replaceAll("\\", "/").split("/").at(-1) ?? "";
}

function sessionBasenameFromState(value: string | undefined): string {
  if (!value || !value.startsWith("/var/lib/pi-session/")) {
    throw new Error("Pi RPC returned an invalid session path");
  }
  return validatedSessionBasename(value.slice("/var/lib/pi-session/".length));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return `${error.message} ${(error as Error & { stderr?: string }).stderr ?? ""}`;
  return String(error);
}
