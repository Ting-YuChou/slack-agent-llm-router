import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";

import type { RunRecord } from "./orchestrator.js";

export interface PersistedSession {
  id: string;
  key: string;
  ownerUserId: string;
  modelRef?: string;
  worktreePath: string;
  branch: string;
  baselineCommit: string;
  closed: boolean;
  lastActivity: number;
  activeRunId?: string;
  piSessionId?: string;
  piSessionFile?: string;
  piLeafId?: string | null;
  parentSessionId?: string;
  forkSourceRunId?: string;
  needsAttention?: boolean;
  activeCheckpointRunId?: string;
  cleanupPending?: boolean;
}

export interface RuntimeState {
  schema_version: 1;
  sessions: PersistedSession[];
  runs: RunRecord[];
}

export class RuntimeStateStore {
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
    } catch {
      return { schema_version: 1, sessions: [], runs: [] };
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
      if (["starting", "running", "awaiting_approval"].includes(run.status)) {
        run.status = "interrupted";
        run.updated_at = new Date(this.now()).toISOString();
        run.error = { code: "runtime_restarted", message: "Agent runtime restarted during this prompt" };
        run.events.push({ type: "status", status: "interrupted" });
      }
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
    const snapshot = JSON.stringify(state, null, 2);
    this.writeChain = this.writeChain.catch(() => undefined).then(async () => {
      await mkdir(path.dirname(this.filePath), { recursive: true });
      const temporary = `${this.filePath}.tmp`;
      await writeFile(temporary, snapshot, { encoding: "utf8", mode: 0o600 });
      await rename(temporary, this.filePath);
    });
    return this.writeChain;
  }
}

function isState(value: unknown): value is RuntimeState {
  return typeof value === "object" && value !== null &&
    (value as RuntimeState).schema_version === 1 &&
    Array.isArray((value as RuntimeState).sessions) &&
    Array.isArray((value as RuntimeState).runs);
}
