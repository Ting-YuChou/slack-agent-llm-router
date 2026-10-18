import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";

import { assertCandidateScope } from "../../src/candidate-scope.js";
import { CodingAgentOrchestrator, type QualityServices, type RunRecord } from "../../src/orchestrator.js";
import { PiRpcBridge, type PublicRunEvent } from "../../src/pi-rpc.js";
import { QualityGate, type CheckStatus, type Grade } from "../../src/quality-gate.js";
import { WorktreeManager } from "../../src/worktree-manager.js";

const exec = promisify(execFile);

/** One writer execution: files it leaves in the worktree (null deletes) and the raw Pi RPC records it emits. */
export interface ScriptedAttempt { files: Record<string, string | null>; records: Array<Record<string, unknown>> }

export interface Scenario {
  name: string;
  /** Omit for the no-profile path. */
  profile?: { allowedPaths: string[]; maxRepairs: number };
  attempts: ScriptedAttempt[];
  verify?: CheckStatus[];
  grade?: Grade["status"][];
  expect: {
    status: RunRecord["status"];
    errorCode?: string;
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
  private attempt = 0;
  private readonly bridge: PiRpcBridge;
  constructor(private readonly worktree: string, private readonly scenario: Scenario, onEvent: (event: PublicRunEvent) => void) {
    this.bridge = new PiRpcBridge({ writeLine: () => undefined, onEvent });
  }
  prompt(runId: string, prompt: string): void {
    this.prompts.push(prompt);
    const script = this.scenario.attempts[this.attempt++];
    if (!script) throw new Error(`scenario ${this.scenario.name} has no attempt ${this.attempt}`);
    this.bridge.prompt(runId, prompt);
    setImmediate(() => void this.play(script));
  }
  decide(): void { /* scenarios do not request approvals */ }
  async abort(): Promise<void> {}
  async close(): Promise<void> {}
  private async play(script: ScriptedAttempt): Promise<void> {
    for (const [file, content] of Object.entries(script.files)) {
      const target = path.join(this.worktree, file);
      if (content === null) await rm(target, { force: true });
      else { await mkdir(path.dirname(target), { recursive: true }); await writeFile(target, content); }
    }
    for (const record of script.records) this.bridge.feed(`${JSON.stringify(record)}\n`);
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

function qualityFor(scenario: Scenario): QualityServices | undefined {
  const profile = scenario.profile;
  if (!profile) return undefined;
  const rubric = ["meets_request"];
  const gate = new QualityGate({
    profile: { id: "scenario", rubric, maxRepairs: profile.maxRepairs },
    snapshot: async input => {
      assertCandidateScope(await changedFiles(input.worktree, input.baseline), profile.allowedPaths);
      const patch = (await exec("git", ["diff", "--binary", input.baseline], { cwd: input.worktree })).stdout;
      return { hash: createHash("sha256").update(patch).digest("hex"), path: input.worktree, patch };
    },
    verify: async (_candidate, input) => {
      const status = scenario.verify?.[input.attempt - 1] ?? "passed";
      return { status, checks: [{ name: "scripted", status, exit_code: status === "passed" ? 0 : 1 }] };
    },
    grade: async (candidate, _verification, input) => {
      const status = scenario.grade?.[input.attempt - 1] ?? "passed";
      return { status, candidate_hash: candidate.hash, criteria: rubric.map(id => ({ id, passed: status === "passed", evidence: "scripted" })), findings: [] };
    },
  });
  return {
    evaluate: input => gate.evaluate(input),
    assertUnchanged: async () => undefined,
    execution: () => undefined,
    maxCostUsd: 10,
    activeTimeoutMs: 60_000,
  };
}

export interface ScenarioResult { run: RunRecord; prompts: string[]; commits: number; dirty: string }

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
    // Runs turn terminal before their rollback finishes; wait for it before inspecting the worktree.
    const pending = new Set<Promise<unknown>>();
    const track = <T>(work: Promise<T>): Promise<T> => { pending.add(work); void work.finally(() => pending.delete(work)).catch(() => undefined); return work; };
    const orchestrator = new CodingAgentOrchestrator({
      worktrees: {
        create: (id: string, base?: string) => worktrees.create(id, base),
        inspectDiff: (p: string, limits: Parameters<WorktreeManager["inspectDiff"]>[1]) => worktrees.inspectDiff(p, limits),
        commit: (p: string, message: string) => withIdentity(() => worktrees.commit(p, message)),
        rollback: (p: string, baseline: string) => track(worktrees.rollback(p, baseline)),
        remove: (p: string) => worktrees.remove(p),
      },
      startProcess: async (session: { worktreePath: string }, onEvent: (event: PublicRunEvent) => void) => {
        scripted = new ScriptedProcess(session.worktreePath, scenario, onEvent);
        return scripted;
      },
      deadlineMs: 30_000,
      quality: qualityFor(scenario),
    });
    const created = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: scenario.name, user_id: "U", prompt: "Fix the bug in src/a.js" });
    const run = await waitForTerminal(() => orchestrator.getRun(created.run_id));
    while (pending.size) await Promise.allSettled([...pending]);
    const session = (orchestrator as unknown as { sessions: Map<string, { worktreePath: string; baselineCommit: string }> }).sessions.get(created.session_id)!;
    const commits = Number(await git(session.worktreePath, "rev-list", "--count", "HEAD")) - 1;
    const dirty = await git(session.worktreePath, "status", "--porcelain");
    await orchestrator.shutdown();
    return { run: structuredClone(run), prompts: scripted.prompts, commits, dirty };
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

async function waitForTerminal(read: () => RunRecord, timeoutMs = 10_000): Promise<RunRecord> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const run = read();
    if (TERMINAL.has(run.status)) return run;
    await new Promise(resolve => setTimeout(resolve, 5));
  }
  throw new Error(`run did not reach a terminal status; last status ${read().status}`);
}
