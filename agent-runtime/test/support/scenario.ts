import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";

import { assertCandidateScope } from "../../src/candidate-scope.js";
import { CodingAgentOrchestrator, type PatchArchiver, type QualityServices, type RunRecord } from "../../src/orchestrator.js";
import { applyFixers } from "../../src/fixers.js";
import { archivePatch, undeliveredPatchPath } from "../../src/patch-archive.js";
import { PiRpcBridge, type PublicRunEvent } from "../../src/pi-rpc.js";
import { QualityGate, type CheckStatus, type Grade, type VerificationResult } from "../../src/quality-gate.js";
import { WorktreeManager } from "../../src/worktree-manager.js";

const exec = promisify(execFile);

/** Playback pauses at this record until the host answers the process through `respondUi`. */
export const awaitUi = { awaitUi: true } as const;

/** One writer execution: files it leaves in the worktree (null deletes) and the raw Pi RPC records it emits. */
export interface ScriptedAttempt { files: Record<string, string | null>; records: Array<Record<string, unknown>> }

/** A grader step: a verdict, an error the grader throws, or a function building the raw submission for grader call `call` (1-based). */
export type ScriptedGrade = Grade["status"] | Error | ((call: number, candidateHash: string) => unknown);

export interface Scenario {
  name: string;
  /** Omit for the no-profile path. */
  profile?: { allowedPaths: string[]; maxRepairs: number };
  attempts: ScriptedAttempt[];
  /** Per attempt: a status for a single scripted check, the full verification result, or an error the verifier throws. */
  verify?: Array<CheckStatus | VerificationResult | Error>;
  /** Per grader call, not per attempt, so retries consume their own entries. */
  grade?: ScriptedGrade[];
  /** Overrides merged over the scenario's quality services. */
  quality?: Partial<QualityServices>;
  /** Replaces the real patch archiver (which writes under the scenario root's artifacts/). */
  archivePatch?: PatchArchiver;
  /** Turns on delivery approval; `delivery` is how the owner answers the card (`expire` lets its short TTL run out). */
  deliveryApproval?: boolean;
  delivery?: "approve" | "reject" | "expire";
  /** A trusted fixer the gate runs on a scratch copy before snapshotting, like a profile's `fixers`. */
  fixer?: (workspace: string) => Promise<void>;
  /** Runs against the worktree while the delivery card is pending, before the owner answers. */
  beforeDelivery?: (worktree: string) => Promise<void>;
  expect: {
    status: RunRecord["status"];
    errorCode?: string;
    causeCode?: string;
    committed: boolean;
    verificationStatus?: string;
    repairPromptIncludes?: string[];
  };
}

/** Pi RPC records for a normally completed assistant turn. */
export function piTurn(stopReason: string, text = "", extra: Record<string, unknown> = {}): Array<Record<string, unknown>> {
  return [
    { type: "turn_start" },
    { type: "message_end", message: { role: "assistant", content: text ? [{ type: "text", text }] : [], stopReason, ...extra } },
  ];
}

export const settledRecord = { type: "agent_settled" };

class ScriptedProcess {
  prompts: string[] = [];
  responses: Array<{ id: string; response: unknown }> = [];
  private attempt = 0;
  private resumeUi?: () => void;
  private readonly bridge: PiRpcBridge;
  constructor(private readonly worktree: string, private readonly scenario: Scenario, onEvent: (event: PublicRunEvent) => void) {
    this.bridge = new PiRpcBridge({ writeLine: () => undefined, onEvent });
  }
  prompt(runId: string, prompt: string): void {
    this.prompts.push(prompt);
    const script = this.scenario.attempts[this.attempt++];
    if (!script) throw new Error(`scenario ${this.scenario.name} has no attempt ${this.attempt}`);
    this.bridge.prompt(runId, prompt);
    // A broken script must fail the scenario, not escape as an unhandled rejection.
    setImmediate(() => { this.play(script).catch((error: unknown) => this.bridge.feed(`${JSON.stringify({ type: "extension_error", error: String(error) })}\n`)); });
  }
  decide(): void { /* scenarios do not request approvals */ }
  respondUi(id: string, response: unknown): void {
    this.responses.push({ id, response });
    const resume = this.resumeUi;
    this.resumeUi = undefined;
    resume?.();
  }
  async abort(): Promise<void> {}
  async close(): Promise<void> {}
  private async play(script: ScriptedAttempt): Promise<void> {
    for (const [file, content] of Object.entries(script.files)) {
      const target = path.join(this.worktree, file);
      if (content === null) await rm(target, { force: true });
      else { await mkdir(path.dirname(target), { recursive: true }); await writeFile(target, content); }
    }
    for (const record of script.records) {
      if (record === awaitUi) { await new Promise<void>(resolve => { this.resumeUi = resolve; }); continue; }
      this.bridge.feed(`${JSON.stringify(record)}\n`);
    }
  }
}

async function git(cwd: string, ...args: string[]): Promise<string> {
  return (await exec("git", args, { cwd })).stdout.trim();
}

async function changedFiles(worktree: string, baseline: string): Promise<string[]> {
  const tracked = (await exec("git", ["diff", "--name-only", "-z", baseline], { cwd: worktree })).stdout.split("\0").filter(Boolean);
  const untracked = (await exec("git", ["ls-files", "--others", "--exclude-standard", "-z"], { cwd: worktree })).stdout.split("\0").filter(Boolean);
  return [...tracked, ...untracked];
}

function scriptedVerification(step: CheckStatus | VerificationResult | Error | undefined): VerificationResult {
  if (step instanceof Error) throw step;
  if (step && typeof step === "object") return step;
  const status = step ?? "passed";
  return { status, checks: [{ name: "scripted", status, exit_code: status === "passed" ? 0 : 1 }] };
}

function qualityFor(scenario: Scenario, counters: { gradeCalls: number }): QualityServices | undefined {
  const profile = scenario.profile;
  if (!profile) return undefined;
  const rubric = ["meets_request"];
  const gate = new QualityGate({
    profile: { id: "scenario", rubric, maxRepairs: profile.maxRepairs },
    snapshot: async input => {
      assertCandidateScope(await changedFiles(input.worktree, input.baseline), profile.allowedPaths);
      const fixed = scenario.fixer ? await applyFixers({ fixers: [{ name: "fmt", argv: ["fmt"], cwd: ".", timeoutMs: 1000 }], worktree: input.worktree, baseline: input.baseline,
        scratch: path.join(path.dirname(input.worktree), "..", "fixer-scratch"), signal: input.signal,
        run: async (_fixer, workspace) => { await scenario.fixer!(workspace); return { exit: 0, timedOut: false }; } }) : undefined;
      const patch = (await exec("git", ["diff", "--binary", input.baseline], { cwd: input.worktree })).stdout;
      return { hash: createHash("sha256").update(patch).digest("hex"), path: input.worktree, patch, ...(fixed ? { fixes: fixed.fixes } : {}) };
    },
    verify: async (_candidate, input) => scriptedVerification(scenario.verify?.[input.attempt - 1]),
    grade: async candidate => {
      const call = ++counters.gradeCalls;
      const step = scenario.grade?.[call - 1] ?? "passed";
      if (step instanceof Error) throw step;
      if (typeof step === "function") return step(call, candidate.hash);
      return { status: step, candidate_hash: candidate.hash, criteria: rubric.map(id => ({ id, passed: step === "passed", evidence: "scripted" })), findings: [] };
    },
  });
  return {
    evaluate: input => gate.evaluate(input),
    assertUnchanged: async () => undefined,
    execution: () => undefined,
    maxCostUsd: 10,
    activeTimeoutMs: 60_000,
    ...scenario.quality,
  };
}

/** An undelivered patch archived during the scenario, read back before the scenario root is removed. */
export interface ArchivedPatchFile { path: string; content: string; mode: number }

export interface ScenarioResult { run: RunRecord; prompts: string[]; responses: ScriptedProcess["responses"]; gradeCalls: number; commits: number; dirty: string; patches: ArchivedPatchFile[]; deliveryCard?: Record<string, unknown> }

/** Runs a scenario through the real orchestrator, worktree manager, Pi RPC bridge and quality gate. */
export async function runScenario(scenario: Scenario): Promise<ScenarioResult> {
  const root = await mkdtemp(path.join(tmpdir(), "pi-scenario-"));
  try {
    const repo = path.join(root, "repo");
    await mkdir(path.join(repo, "src"), { recursive: true });
    await writeFile(path.join(repo, "src", "a.js"), "export const a = 1;\n");
    await git(repo, "init", "-q", "-b", "main");
    await git(repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "add", ".");
    await git(repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q", "-m", "baseline");
    const worktrees = new WorktreeManager({ repoPath: repo, worktreeRoot: path.join(root, "worktrees"), baseRef: "main" });
    let scripted!: ScriptedProcess;
    const counters = { gradeCalls: 0 };
    const archived: string[] = [];
    const archiver: PatchArchiver = scenario.archivePatch ?? (async input => {
      const result = await archivePatch({ worktree: input.worktree, baseline: input.baseline, file: undeliveredPatchPath(root, input.run_id) });
      if (result) archived.push(result.path);
      return result?.path;
    });
    // Runs turn terminal before their rollback finishes; wait for it before inspecting the worktree.
    const pending = new Set<Promise<unknown>>();
    const track = <T>(work: Promise<T>): Promise<T> => { pending.add(work); void work.finally(() => pending.delete(work)).catch(() => undefined); return work; };
    const orchestrator = new CodingAgentOrchestrator({
      worktrees: {
        create: (id: string, base?: string) => worktrees.create(id, base),
        inspectDiff: (p: string, limits: Parameters<WorktreeManager["inspectDiff"]>[1]) => worktrees.inspectDiff(p, limits),
        patch: (p: string) => worktrees.patch(p),
        commit: (p: string, message: string) => withIdentity(() => worktrees.commit(p, message)),
        rollback: (p: string, baseline: string) => track(worktrees.rollback(p, baseline)),
        remove: (p: string) => worktrees.remove(p),
      },
      startProcess: async (session: { worktreePath: string }, onEvent: (event: PublicRunEvent) => void) => {
        scripted = new ScriptedProcess(session.worktreePath, scenario, onEvent);
        return scripted;
      },
      deadlineMs: 30_000,
      quality: qualityFor(scenario, counters),
      archivePatch: archiver,
      ...(scenario.deliveryApproval ? { deliveryApproval: { ttlMs: scenario.delivery === "expire" ? 50 : 30_000 } } : {}),
    });
    const created = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: scenario.name, user_id: "U", prompt: "Fix the bug in src/a.js" });
    let deliveryCard: Record<string, unknown> | undefined;
    if (scenario.deliveryApproval) {
      const waiting = await waitFor(() => orchestrator.getRun(created.run_id), run => TERMINAL.has(run.status) || run.phase === "awaiting_delivery", run => `run never reached delivery; last status ${run.status}`);
      deliveryCard = [...waiting.events].reverse().find(event => event.type === "approval" && event.kind === "delivery");
      if (deliveryCard) {
        const worktree = (orchestrator as unknown as { sessions: Map<string, { worktreePath: string }> }).sessions.get(created.session_id)!.worktreePath;
        await scenario.beforeDelivery?.(worktree);
        if (scenario.delivery === "approve" || scenario.delivery === "reject") {
          await orchestrator.decide(created.run_id, String(deliveryCard.approval_id), "U", scenario.delivery);
        }
      }
    }
    const run = await waitForTerminal(() => orchestrator.getRun(created.run_id));
    while (pending.size) await Promise.allSettled([...pending]);
    const session = (orchestrator as unknown as { sessions: Map<string, { worktreePath: string; baselineCommit: string }> }).sessions.get(created.session_id)!;
    const commits = Number(await git(session.worktreePath, "rev-list", "--count", "HEAD")) - 1;
    const dirty = await git(session.worktreePath, "status", "--porcelain");
    await orchestrator.shutdown();
    const patches = await Promise.all(archived.map(async file => ({ path: file, content: await readFile(file, "utf8"), mode: (await stat(file)).mode & 0o777 })));
    return { run: structuredClone(run), prompts: scripted.prompts, responses: scripted.responses, gradeCalls: counters.gradeCalls, commits, dirty, patches, ...(deliveryCard ? { deliveryCard } : {}) };
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

async function withIdentity<T>(action: () => Promise<T>): Promise<T> {
  const saved = { name: process.env.GIT_AUTHOR_NAME, email: process.env.GIT_AUTHOR_EMAIL, cname: process.env.GIT_COMMITTER_NAME, cemail: process.env.GIT_COMMITTER_EMAIL };
  process.env.GIT_AUTHOR_NAME = process.env.GIT_COMMITTER_NAME = "scenario";
  process.env.GIT_AUTHOR_EMAIL = process.env.GIT_COMMITTER_EMAIL = "scenario@example.com";
  try { return await action(); } finally {
    restore("GIT_AUTHOR_NAME", saved.name); restore("GIT_AUTHOR_EMAIL", saved.email);
    restore("GIT_COMMITTER_NAME", saved.cname); restore("GIT_COMMITTER_EMAIL", saved.cemail);
  }
}

function restore(key: string, value: string | undefined): void {
  if (value === undefined) delete process.env[key]; else process.env[key] = value;
}

const TERMINAL = new Set(["completed", "rejected", "cancelled", "failed", "timed_out", "interrupted"]);

/** Polls `read` (sync or async) until `done` accepts its value. */
export async function waitFor<T>(read: () => T | Promise<T>, done: (value: T) => boolean, describe: (value: T) => string, timeoutMs = 10_000): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = await read();
    if (done(value)) return value;
    await new Promise(resolve => setTimeout(resolve, 5));
  }
  throw new Error(describe(await read()));
}

function waitForTerminal(read: () => RunRecord): Promise<RunRecord> {
  return waitFor(read, run => TERMINAL.has(run.status), run => `run did not reach a terminal status; last status ${run.status}`);
}
