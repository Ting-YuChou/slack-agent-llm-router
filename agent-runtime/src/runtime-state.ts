import { mkdir, open, readFile, rename } from "node:fs/promises";
import path from "node:path";

import type { RunRecord } from "./orchestrator.js";
import { appendRunEvent } from "./run-events.js";

export interface PersistedSession {
  id: string;
  key: string;
  ownerUserId: string;
  modelRef?: string;
  autoRouting?: boolean;
  worktreePath: string;
  /** The session's own Git mirror; sessions persisted before mirrors lack it and keep the shared metadata. */
  gitDir?: string;
  branch: string;
  baselineCommit: string;
  closed: boolean;
  lastActivity: number;
  activeRunId?: string;
  /** Set for CI-repair sessions, which gate delivery through their own publish approval. */
  skipDelivery?: boolean;
  /**
   * Set on every session created since seals became required: its `--continue` is refused without a seal.
   * A session without it was saved before then (legacy) and may still continue unsealed.
   */
  sealRequired?: boolean;
}

export interface RuntimeState {
  schema_version: 1;
  sessions: PersistedSession[];
  runs: RunRecord[];
}

export class RuntimeStateStore {
  healthy = true;
  private writeChain: Promise<void> = Promise.resolve();
  private expiredSessions: PersistedSession[] = [];

  constructor(
    private readonly filePath: string,
    private readonly now: () => number = Date.now,
  ) {}

  async load(): Promise<RuntimeState> {
    let state: RuntimeState;
    try {
      const parsed = JSON.parse(await readFile(this.filePath, "utf8"));
      if (!isState(parsed)) throw new Error("invalid state");
      state = parsed;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return { schema_version: 1, sessions: [], runs: [] };
      this.healthy = false; throw new Error("Runtime control state is unreadable or corrupt", { cause: error });
    }
    const sessionCutoff = this.now() - 24 * 60 * 60 * 1000;
    const auditCutoff = this.now() - 7 * 24 * 60 * 60 * 1000;
    this.expiredSessions = state.sessions.filter((session) => session.lastActivity < sessionCutoff);
    state.sessions = state.sessions.filter((session) => session.lastActivity >= sessionCutoff);
    const retainedSessionIds = new Set(state.sessions.map((session) => session.id));
    state.runs = state.runs.filter((run) => {
      const timestamp = Date.parse(run.updated_at);
      return retainedSessionIds.has(run.session_id) || !Number.isFinite(timestamp) || timestamp >= auditCutoff;
    });
    for (const run of state.runs) {
      // A run waiting for delivery has no live process to lose; the orchestrator restores it from the
      // approval ledger, or fails it as delivery_interrupted when the ledger has no pending approval.
      if (run.status === "awaiting_approval" && run.phase === "awaiting_delivery") continue;
      if (["starting", "running", "awaiting_approval"].includes(run.status)) interruptRestartedRun(run, this.now());
    }
    for (const session of state.sessions) session.activeRunId = undefined;
    return state;
  }

  takeExpiredSessions(): PersistedSession[] {
    const expired = this.expiredSessions;
    this.expiredSessions = [];
    return expired;
  }

  async save(state: RuntimeState): Promise<void> {
    const snapshot = JSON.stringify(state);
    const write = this.writeChain.then(async () => {
      await mkdir(path.dirname(this.filePath), { recursive: true, mode: 0o700 });
      const temporary = `${this.filePath}.tmp`;
      const file = await open(temporary, "w", 0o600);
      try { await file.chmod(0o600); await file.writeFile(snapshot, "utf8"); await file.sync(); }
      finally { await file.close(); }
      await rename(temporary, this.filePath);
      const directory = await open(path.dirname(this.filePath), "r");
      try { await directory.sync(); } finally { await directory.close(); }
    });
    // Keep later saves runnable after a failure; health follows the latest write.
    this.writeChain = write.catch(() => undefined);
    return write.then(() => { this.healthy = true; }, error => { this.healthy = false; throw error; });
  }
}

/** Marks a run the restart cut off; a tool approval waits on a live Pi dialog, so it can never resume. */
export function interruptRestartedRun(run: RunRecord, now: number): void {
  if (run.workflow_mode === "structured") run.interrupted_at = run.updated_at;
  run.status = "interrupted";
  run.updated_at = new Date(now).toISOString();
  run.error = { code: "runtime_restarted", message: "Agent runtime restarted during this prompt" };
  appendRunEvent(run, { type: "status", status: "interrupted" });
}

function isState(value: unknown): value is RuntimeState {
  return typeof value === "object" && value !== null &&
    (value as RuntimeState).schema_version === 1 &&
    Array.isArray((value as RuntimeState).sessions) &&
    Array.isArray((value as RuntimeState).runs);
}
