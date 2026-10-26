import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync, mkdirSync } from "node:fs";
import { rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { after, test } from "node:test";

import { BudgetLedger } from "../src/budget-ledger.js";
import { ControlStore } from "../src/ci-control.js";
import type { GraderProcessFactory } from "../src/grader.js";
import type { InvestigationProcessFactory } from "../src/agent-execution-runner.js";
import type { PublicRunEvent } from "../src/pi-rpc.js";
import type { VerificationProfile } from "../src/verification.js";
import { createQualityServices } from "../src/workflow-runtime.js";

const roots: string[] = [];
after(async () => { for (const root of roots) await rm(root, { recursive: true, force: true }); });

type Step = { record?: Record<string, unknown>; event?: PublicRunEvent };
const settled: Step = { event: { type: "settled", outcome: "completed" } as PublicRunEvent };
const submit = (tool: string, args: unknown): Step[] => [
  { record: { type: "tool_execution_start", toolName: tool, toolCallId: "t1", args } },
  { record: { type: "tool_execution_end", toolName: tool, toolCallId: "t1", isError: false } },
];
const play = (steps: Step[], onEvent: (event: PublicRunEvent) => void, onRecord: (record: Record<string, unknown>) => void) =>
  setImmediate(() => { for (const step of steps) { if (step.record) onRecord(step.record); if (step.event) onEvent(step.event); } });

/** A repo whose working tree changes src/a.js against its only commit. */
function fixture() {
  const root = mkdtempSync(path.join(os.tmpdir(), "quality-services-"));
  roots.push(root);
  const repo = path.join(root, "repo");
  mkdirSync(path.join(repo, "src"), { recursive: true });
  const git = (...args: string[]) => execFileSync("git", args, { cwd: repo, encoding: "utf8" }).trim();
  git("init", "-q"); git("config", "user.email", "t@example.com"); git("config", "user.name", "t");
  writeFileSync(path.join(repo, "src", "a.js"), "export const a = 1;\n");
  git("add", "."); git("commit", "-qm", "base");
  const baseline = git("rev-parse", "HEAD");
  writeFileSync(path.join(repo, "src", "a.js"), "export const a = 2;\n");
  return { root, repo, baseline };
}

const profile: VerificationProfile = {
  id: "services-test", repository: "fixture/repo", image: "sha256:" + "0".repeat(64), rubric: ["correct"],
  allowedPaths: ["src/"], baselinePaths: ["test/"], checks: [], maxRepairs: 1, maxCostUsd: 1, activeTimeoutMs: 60_000,
  structuredWorkflow: { enabled: true, concurrency: 1, maxCostUsd: 1, activeTimeoutMs: 60_000, discoveryCostUsd: 0.5 },
};

function services(root: string, repo: string, overrides: Partial<Parameters<typeof createQualityServices>[0]> = {}) {
  const records: Array<{ kind: string; payload: Record<string, unknown> }> = [];
  const telemetry = {
    record: async (_run: string, _session: string, kind: string, payload: unknown) => { records.push({ kind, payload: payload as Record<string, unknown> }); return true; },
    markIncomplete: () => {}, completeness: () => [], releaseRun: () => {}, setTraceparent: () => {}, redactText: (text: string) => text,
  };
  const created = createQualityServices({
    control: new ControlStore(path.join(root, "control.db")), session: id => `session-${id}`, priorActive: new Map(),
    repoPath: repo, worktreeRoot: path.join(root, "worktrees"), verificationBases: new Map(), profile, checksum: "checksum-1",
    root: path.join(root, "runtime"), image: "img", network: "net", gatewayUrl: "http://gw", token: () => "token",
    ledger: new BudgetLedger(path.join(root, "budget.db")), mapping: new Map(), phase: async () => {},
    telemetry: telemetry as never, ...overrides,
  });
  return { services: created, records };
}

test("the real quality services grade through an injected verifier and grader process", async () => {
  const { root, repo, baseline } = fixture();
  const verified: string[] = [];
  const createGraderProcess: GraderProcessFactory = (_options, onEvent, onRecord) => ({
    prompt: (_id, prompt) => {
      const hash = /"candidate_hash":"([^"]+)"/.exec(prompt)?.[1];
      play([...submit("submit_grade", { status: "passed", candidate_hash: hash, criteria: [{ id: "correct", passed: true, evidence: "src/a.js:1 exports 2" }], findings: [] }), settled], onEvent, onRecord);
    },
    close: async () => {},
  });
  const { services: quality, records } = services(root, repo, {
    createVerifier: () => ({ verify: async candidate => { verified.push(candidate.hash); return { status: "passed", checks: [] }; }, stop: async () => {} }),
    createGraderProcess,
  });
  quality.prepare?.("run-1", "single");
  const result = await quality.evaluate({ run_id: "run-1", worktree: repo, baseline, requirement: "make a 2", attempt: 1, signal: new AbortController().signal });
  assert.equal(result.action, "pass");
  assert.equal(verified.length, 1);
  assert.equal(verified[0], result.candidate.hash);
  const grade = records.find(entry => entry.kind === "grade");
  assert.ok(grade, "a grade event is recorded");
  assert.equal(grade.payload.attempt, 1);
  assert.equal(grade.payload.profile_checksum, "checksum-1");
  assert.equal(grade.payload.status, "passed");
});

test("discovery runs planner and investigator through an injected execution process", async () => {
  const { root, repo, baseline } = fixture();
  const roles: string[] = [];
  const createExecutionProcess: InvestigationProcessFactory = (_options, onEvent, onRecord) => ({
    prompt: (_id, prompt) => {
      const role = /You are a restricted (\w+)/.exec(prompt)?.[1] ?? "unknown";
      roles.push(role);
      const wave = Number(/"wave":(\d+)/.exec(prompt)?.[1] ?? 1);
      const steps = role === "planner"
        ? submit("submit_plan", { tasks: wave === 1 ? [{ id: "explore", role: "code_explorer", purpose: "find a", expected_output: "location", paths: ["src"], dependencies: [] }] : [] })
        : submit("submit_investigation", { status: "succeeded", findings: [{ path: "src/a.js", line: 1, reason: "a is exported here" }], evidence: ["src/a.js:1"], unresolved: [], suggested_tasks: [] });
      play([...steps, settled], onEvent, onRecord);
    },
    close: async () => {},
  });
  const { services: quality } = services(root, repo, { createExecutionProcess });
  quality.prepare?.("run-2", "structured");
  const brief = await quality.discover!({ run_id: "run-2", worktree: repo, baseline, requirement: "make a 2", attempt: 1, owner: "U1", initial_active_ms: 0, signal: new AbortController().signal });
  assert.deepEqual(roles, ["planner", "code_explorer", "planner"]);
  assert.match(brief, /a is exported here/);
});

test("planner, investigator and grader spend reaches the run under each one's role", async () => {
  const { root, repo, baseline } = fixture();
  const spent: Array<[string, number, string | undefined]> = [];
  const usage = (cost_usd: number): Step => ({ event: { type: "usage", cost_usd } as PublicRunEvent });
  const { services: quality } = services(root, repo, {
    phase: async (_run, phase, cost, role) => { if (cost) spent.push([phase, cost, role]); },
    createExecutionProcess: (_options, onEvent, onRecord) => ({
      prompt: (_id, prompt) => {
        const role = /You are a restricted (\w+)/.exec(prompt)?.[1] ?? "unknown";
        const wave = Number(/"wave":(\d+)/.exec(prompt)?.[1] ?? 1);
        const steps = role === "planner"
          ? submit("submit_plan", { tasks: wave === 1 ? [{ id: "explore", role: "code_explorer", purpose: "find a", expected_output: "location", paths: ["src"], dependencies: [] }] : [] })
          : submit("submit_investigation", { status: "succeeded", findings: [{ path: "src/a.js", line: 1, reason: "a is exported here" }], evidence: ["src/a.js:1"], unresolved: [], suggested_tasks: [] });
        play([usage(role === "planner" ? 0.25 : 0.0625), ...steps, settled], onEvent, onRecord);
      },
      close: async () => {},
    }),
    createVerifier: () => ({ verify: async () => ({ status: "passed", checks: [] }), stop: async () => {} }),
    createGraderProcess: (_options, onEvent, onRecord) => ({
      prompt: (_id, prompt) => {
        const hash = /"candidate_hash":"([^"]+)"/.exec(prompt)?.[1];
        play([usage(0.125), ...submit("submit_grade", { status: "passed", candidate_hash: hash, criteria: [{ id: "correct", passed: true, evidence: "src/a.js:1 exports 2" }], findings: [] }), settled], onEvent, onRecord);
      },
      close: async () => {},
    }),
  });
  quality.prepare?.("run-roles", "structured");
  await quality.discover!({ run_id: "run-roles", worktree: repo, baseline, requirement: "make a 2", attempt: 1, owner: "U1", initial_active_ms: 0, signal: new AbortController().signal });
  await quality.evaluate({ run_id: "run-roles", worktree: repo, baseline, requirement: "make a 2", attempt: 1, signal: new AbortController().signal });
  assert.deepEqual(spent, [["investigating", 0.25, "planner"], ["investigating", 0.0625, "code_explorer"], ["investigating", 0.25, "planner"], ["grading", 0.125, "grader"]]);
});

test("profile fixers run before the snapshot and their files reach the candidate", async () => {
  const { root, repo, baseline } = fixture();
  const fixed = { ...profile, fixers: [{ name: "fmt", argv: ["fmt"], cwd: ".", timeoutMs: 1000 }] };
  const { readFile, writeFile } = await import("node:fs/promises");
  const snapshots: string[] = [];
  const { services: quality } = services(root, repo, {
    profile: fixed,
    createFixerRunner: () => async (_fixer, workspace) => {
      await writeFile(path.join(workspace, "src", "a.js"), "export const a = 2; // formatted\n");
      return { exit: 0, timedOut: false };
    },
    createVerifier: () => ({ verify: async candidate => { snapshots.push(await readFile(path.join(candidate.path, "src", "a.js"), "utf8")); return { status: "passed", checks: [] }; }, stop: async () => {} }),
    createGraderProcess: (_options, onEvent, onRecord) => ({
      prompt: (_id, prompt) => {
        const hash = /"candidate_hash":"([^"]+)"/.exec(prompt)?.[1];
        play([...submit("submit_grade", { status: "passed", candidate_hash: hash, criteria: [{ id: "correct", passed: true, evidence: "src/a.js:1 exports 2" }], findings: [] }), settled], onEvent, onRecord);
      },
      close: async () => {},
    }),
  });
  quality.prepare?.("run-fix", "single");
  const result = await quality.evaluate({ run_id: "run-fix", worktree: repo, baseline, requirement: "make a 2", attempt: 1, signal: new AbortController().signal });
  assert.equal(result.action, "pass");
  assert.deepEqual(result.candidate.fixes, [{ name: "fmt", files: ["src/a.js"] }]);
  assert.deepEqual(snapshots, ["export const a = 2; // formatted\n"], "the verifier sees the fixed file");
  assert.equal(await readFile(path.join(repo, "src", "a.js"), "utf8"), "export const a = 2; // formatted\n");
});
