import { mkdir, readdir, rm } from "node:fs/promises";
import path from "node:path";

import { SessionManager, type SessionEntry } from "@earendil-works/pi-coding-agent";

export class PiSessionStoreError extends Error {}

export interface PiSessionSnapshot {
  piSessionId: string;
  sessionFile: string;
  cwd: string;
  leafId: string | null;
  entries: SessionEntry[];
}

export class PiSessionStore {
  private readonly root: string;

  constructor(root: string) {
    this.root = path.resolve(root);
  }

  async discover(runtimeSessionId: string, sessionFile?: string): Promise<{ piSessionId: string; sessionFile: string }> {
    const directory = this.sessionDirectory(runtimeSessionId);
    let basename = sessionFile ? validatedBasename(sessionFile) : undefined;
    if (!basename) {
      let candidates: string[];
      try {
        candidates = (await readdir(directory)).filter((entry) => entry.endsWith(".jsonl"));
      } catch {
        candidates = [];
      }
      if (candidates.length !== 1) {
        throw new PiSessionStoreError(candidates.length === 0
          ? "Pi session file was not found"
          : "Pi session file migration is ambiguous");
      }
      basename = validatedBasename(candidates[0]);
    }
    const manager = this.open(runtimeSessionId, basename);
    const header = manager.getHeader();
    if (!header?.id) throw new PiSessionStoreError("Pi session header is invalid");
    return { piSessionId: header.id, sessionFile: basename };
  }

  async read(runtimeSessionId: string, sessionFile: string): Promise<PiSessionSnapshot> {
    const manager = this.open(runtimeSessionId, sessionFile);
    const header = manager.getHeader();
    if (!header?.id) throw new PiSessionStoreError("Pi session header is invalid");
    return {
      piSessionId: header.id,
      sessionFile: validatedBasename(sessionFile),
      cwd: manager.getCwd(),
      leafId: manager.getLeafId(),
      entries: manager.getEntries(),
    };
  }

  async rollback(input: {
    runtimeSessionId: string;
    sessionFile: string;
    parentLeafId: string | null;
    runId: string;
  }): Promise<{ leafId: string }> {
    const manager = this.open(input.runtimeSessionId, input.sessionFile);
    const leafId = manager.branchWithSummary(
      input.parentLeafId,
      `Slack Agent run ${input.runId} did not complete. Continue from the checkpoint before that run.`,
      { rollback: true, runId: input.runId },
      true,
    );
    return { leafId };
  }

  async fork(input: {
    sourceRuntimeSessionId: string;
    sourceSessionFile: string;
    sourceLeafId: string;
    childRuntimeSessionId: string;
    childPiSessionId: string;
    childCwd: string;
  }): Promise<{ sessionFile: string; leafId: string | null }> {
    const sourcePath = this.sessionPath(input.sourceRuntimeSessionId, input.sourceSessionFile);
    const source = this.open(input.sourceRuntimeSessionId, input.sourceSessionFile);
    if (!source.getEntry(input.sourceLeafId)) throw new PiSessionStoreError("Fork source leaf was not found");
    const extractedPath = source.createBranchedSession(input.sourceLeafId);
    if (!extractedPath) throw new PiSessionStoreError("Pi could not extract the selected session branch");
    const childDirectory = this.sessionDirectory(input.childRuntimeSessionId);
    await mkdir(childDirectory, { recursive: true });
    try {
      const child = SessionManager.forkFrom(extractedPath, path.resolve(input.childCwd), childDirectory, {
        id: input.childPiSessionId,
        parentSession: sourcePath,
      });
      const childPath = child.getSessionFile();
      if (!childPath) throw new PiSessionStoreError("Pi child session file was not created");
      return { sessionFile: validatedBasename(path.basename(childPath)), leafId: child.getLeafId() };
    } finally {
      await rm(extractedPath, { force: true });
    }
  }

  async remove(runtimeSessionId: string): Promise<void> {
    await rm(this.sessionDirectory(runtimeSessionId), { recursive: true, force: true });
  }

  private open(runtimeSessionId: string, sessionFile: string): SessionManager {
    const filePath = this.sessionPath(runtimeSessionId, sessionFile);
    try {
      return SessionManager.open(filePath, path.dirname(filePath));
    } catch (error) {
      throw new PiSessionStoreError("Pi session file could not be opened", { cause: error });
    }
  }

  private sessionPath(runtimeSessionId: string, sessionFile: string): string {
    return path.join(this.sessionDirectory(runtimeSessionId), validatedBasename(sessionFile));
  }

  private sessionDirectory(runtimeSessionId: string): string {
    if (!/^[A-Za-z0-9._-]{1,128}$/.test(runtimeSessionId)) {
      throw new PiSessionStoreError("Runtime session id is invalid");
    }
    const candidate = path.resolve(this.root, runtimeSessionId);
    if (candidate === this.root || !candidate.startsWith(`${this.root}${path.sep}`)) {
      throw new PiSessionStoreError("Runtime session directory must stay below its managed root");
    }
    return candidate;
  }
}

function validatedBasename(value: string): string {
  const normalized = value.replaceAll("\\", "/");
  if (normalized.includes("/") || !/^[A-Za-z0-9._-]+\.jsonl$/.test(normalized)) {
    throw new PiSessionStoreError("Pi session file must be a JSONL basename");
  }
  return normalized;
}
