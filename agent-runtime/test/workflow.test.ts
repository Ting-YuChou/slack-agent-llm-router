import assert from "node:assert/strict";
import { test } from "node:test";
import * as workflowBudgetModule from "../src/workflow-budget.js";
import * as verificationModule from "../src/verification.js";
import * as startupTimingModule from "../src/startup-timing.js";
import * as qualityGateModule from "../src/quality-gate.js";
import * as graderModule from "../src/grader.js";
test('verification counts preserve Node TAP and pytest results without inventing counts for typecheck', async () => {
  const module: any = await import('../src/verification.js');
  assert.equal(typeof module.parseVerificationCounts,'function');
  assert.deepEqual(module.parseVerificationCounts('# tests 6\n# pass 5\n# fail 1\n'),{ passed: 5,failed: 1 });
  assert.deepEqual(module.parseVerificationCounts('7 passed, 1 failed in 0.03s'),{ passed: 7,failed: 1 });
  assert.deepEqual(module.parseVerificationCounts('typecheck completed'),{ passed: null,failed: null });
});

test("startup readiness does not finish when only the Docker CLI spawned", async () => {
  const module = startupTimingModule;
  assert.ok(module, "startup timing implementation exists");
  const events: any[] = [];
  const timing = new module.StartupTiming((...args: any[]) => events.push(args));
  timing.start(); timing.cliSpawned();
  assert.ok(!events.some(e => e[0] === "container_start" && e[1] === "end"));
  timing.containerRunning(); timing.rpcReady(); timing.configured(); timing.promptSent();
  timing.firstOutput(); timing.firstOutput();
  assert.equal(events.filter(e => e[0] === "prompt_to_first_output" && e[1] === "end").length, 1);
});

test("quality gate never grades failed tests, bounds repairs and binds the candidate", async () => {
  const module = qualityGateModule;
  assert.ok(module, "quality gate implementation exists");
  let grades = 0;
  const gate = new module.QualityGate({
    profile: { id: "test", rubric: ["correct"], maxRepairs: 2 },
    snapshot: async () => ({ hash: "candidate", path: "/frozen" }),
    verify: async () => ({ status: "failed", checks: [{ name: "test", status: "failed", exit_code: 1 }] }),
    grade: async () => { grades++; return { status: "passed", candidate_hash: "candidate", criteria: [{ id: "correct", passed: true }], findings: [] }; },
  });
  const input = { run_id: "r", worktree: "/w", baseline: "a".repeat(40), requirement: "fix", attempt: 1, signal: new AbortController().signal };
  assert.equal((await gate.evaluate(input)).action, "repair");
  assert.equal((await gate.evaluate({ ...input, attempt: 3 })).action, "fail");
  assert.equal(grades, 0);
});

test("grader cannot approve a different candidate or omit a required criterion", async () => {
  const module = qualityGateModule;
  assert.ok(module, "quality gate implementation exists");
  assert.throws(() => module.validateGrade({ status: "passed", candidate_hash: "other", criteria: [], findings: [] }, "candidate", ["correct"]));
  assert.throws(() => module.validateGrade({ status: "passed", candidate_hash: "candidate", criteria: [], findings: [] }, "candidate", ["correct"]));
  assert.throws(() => module.validateGrade({ status: "passed", candidate_hash: "candidate", criteria: [{ id: "correct", passed: false }], findings: [] }, "candidate", ["correct"]));
});

test("budget retains unknown usage reservations and pauses only human waiting", async () => {
  const module = workflowBudgetModule;
  assert.ok(module, "workflow budget implementation exists");
  let now = 0;
  const budget = new module.WorkflowBudget(1, 100, () => now);
  budget.reserve("call1", 0.8); budget.settle("call1", null);
  assert.throws(() => budget.reserve("call2", 0.3));
  now = 40; budget.pause(); now = 400; budget.resume(); now = 450;
  assert.equal(budget.activeMs(), 90);
  now = 461; assert.throws(() => budget.check());
});

test("RPC readiness requires a successful matching response, not arbitrary stdout", async () => {
  const { PiRpcBridge } = await import("../src/pi-rpc.js");
  const writes: any[] = []; let ready = 0;
  const bridge = new PiRpcBridge({ writeLine: (line: string) => writes.push(JSON.parse(line)), onEvent: () => {}, onReady: () => ready++ } as any);
  const configured = bridge.configureAndPrompt("r", "fix", "openai", "m", "high");
  bridge.feed(JSON.stringify({ type: "notice" }) + "\n");
  assert.equal(ready, 0);
  bridge.feed(JSON.stringify({ type: "response", id: writes[0].id, command: "set_model", success: true }) + "\n");
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(ready, 1);
  bridge.failPending(); await assert.rejects(configured);
});

test("verification profiles reject shell strings, floating images and workspace-loaded profiles", async () => {
  const m = verificationModule;
  assert.ok(m, "verification implementation exists");
  const profile = { id: "p", repository: "o/r", image: "repo@sha256:" + "a".repeat(64), rubric: ["correct"], checks: [{ name: "unit", argv: ["python", "-m", "pytest"], cwd: ".", timeoutMs: 1000 }], allowedPaths: ["src/", "tests/"] };
  assert.equal(m.validateProfile(profile).checks.length, 1);
  assert.throws(() => m.validateProfile({ ...profile, image: "repo:latest" }));
  assert.throws(() => m.validateProfile({ ...profile, checks: [{ name: "bad", argv: "curl bad | sh" }] }));
  assert.throws(() => m.validateProfile({ ...profile, checks: [{ name: "bad", argv: ["pytest"], cwd: "../", timeoutMs: 1 }] }));
});

test("candidate snapshot includes untracked code and changes its identity on modification", async () => {
  const m = verificationModule;
  assert.ok(m, "verification implementation exists");
  const { mkdtemp, writeFile, readFile, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os"); const { execFileSync } = await import("node:child_process");
  const root = await mkdtemp(tmpdir() + "/candidate-");
  try {
    execFileSync("git", ["init", "-q", root]);
    await writeFile(root + "/a.txt", "baseline");
    execFileSync("git", ["-C", root, "add", "."]);
    execFileSync("git", ["-C", root, "-c", "user.name=Test", "-c", "user.email=t@example.com", "commit", "-qm", "baseline"]);
    const baseline = execFileSync("git", ["-C", root, "rev-parse", "HEAD"]).toString().trim();
    await writeFile(root + "/new.txt", "candidate");
    const store = new m.CandidateStore(root + "/artifacts");
    const a = await store.create(root, baseline, "r");
    assert.equal(await readFile(a.path + "/new.txt", "utf8"), "candidate");
    await writeFile(root + "/new.txt", "changed");
    const b = await store.create(root, baseline, "r"); assert.notEqual(a.hash, b.hash);
    assert.ok(a.artifact);
    assert.match(await readFile(a.artifact, "utf8"), /candidate/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("grader Docker invocation has read-only source, fresh context and no mutation tools", async () => {
  const m = graderModule;
  assert.ok(m, "grader implementation exists");
  const args = m.graderDockerOptions({ image: "agent", network: "pi-model-only", gatewayUrl: "http://model-gateway:8080", candidate: { hash: "h", path: "/snapshot" }, statePath: "/state/new", executionId: "e", rubric: ["correct"] });
  assert.equal(args.continueSession, false);
  assert.equal(args.readOnlyWorkspace, true);
  assert.deepEqual(args.toolNames, ["read", "grep", "find", "ls", "submit_grade"]);
  assert.equal(args.mcp, undefined);
  const { buildAgentDockerArgs } = await import("../src/container-runtime.js");
  const invocation = buildAgentDockerArgs({ ...args, gatewayToken: "fake-token" });
  assert.ok(invocation.includes("--no-context-files"));
  assert.equal(invocation[invocation.indexOf("--workdir") + 1], "/var/lib/pi-session");
  assert.ok(!invocation.includes("--continue"));
});

test("invalid or unsupported grader claims become unknown and cannot trigger repair or commit", async () => {
  const { QualityGate } = await import("../src/quality-gate.js");
  const gate = new QualityGate({ profile: { id: "p", rubric: ["correct"], maxRepairs: 2 },
    snapshot: async () => ({ hash: "hash", path: "/snapshot" }),
    verify: async () => ({ status: "passed", checks: [] }),
    grade: async () => ({ status: "passed", candidate_hash: "hash", criteria: [{ id: "correct", passed: true }], findings: [] }),
  });
  const result = await gate.evaluate({ run_id: "r", worktree: "/workspace", baseline: "a".repeat(40), requirement: "fix", attempt: 1, signal: new AbortController().signal });
  assert.equal(result.action, "fail"); assert.equal(result.grade?.status, "unknown");
});

test("quality gate turns a scope violation into a repair that lists the files, then a scope_violation failure", async () => {
  const { QualityGate } = await import("../src/quality-gate.js");
  const { ScopeViolationError } = await import("../src/candidate-scope.js");
  let verified = 0;
  const gate = new QualityGate({
    profile: { id: "test", rubric: ["correct"], maxRepairs: 1 },
    snapshot: async () => { throw new ScopeViolationError(["docs/notes.md"]); },
    verify: async () => { verified++; return { status: "passed", checks: [] }; },
    grade: async () => ({}),
  });
  const input = { run_id: "r", worktree: "/w", baseline: "a".repeat(40), requirement: "fix the parser", attempt: 1, signal: new AbortController().signal };
  const first = await gate.evaluate(input);
  assert.equal(first.action, "repair");
  assert.deepEqual(first.failure, { code: "scope_violation", files: ["docs/notes.md"] });
  assert.match(first.repair_prompt ?? "", /docs\/notes\.md/);
  assert.match(first.repair_prompt ?? "", /fix the parser/);
  const last = await gate.evaluate({ ...input, attempt: 2 });
  assert.equal(last.action, "fail");
  assert.equal(last.failure?.code, "scope_violation");
  assert.equal(verified, 0);
});

test("quality gate keeps why the grader produced no verdict", async () => {
  const { QualityGate } = await import("../src/quality-gate.js");
  const { HarnessError } = await import("../src/error-taxonomy.js");
  const recorded: any[] = [];
  const gate = new QualityGate({ profile: { id: "p", rubric: ["correct"], maxRepairs: 2 },
    snapshot: async () => ({ hash: "hash", path: "/snapshot" }),
    verify: async () => ({ status: "passed", checks: [] }),
    grade: async () => { throw new HarnessError("grader_timeout", "Grader did not finish within 120s"); },
    record: async (kind, payload) => { recorded.push({ kind, payload }); },
  });
  const result = await gate.evaluate({ run_id: "r", worktree: "/w", baseline: "a".repeat(40), requirement: "fix", attempt: 1, signal: new AbortController().signal });
  assert.equal(result.grade?.status, "unknown");
  assert.equal(result.grade?.cause?.code, "grader_timeout");
  assert.equal(recorded.find(r => r.kind === "grade")?.payload.cause.code, "grader_timeout");
});

test("quality gate reports an invalid grader result as grader_invalid_submission", async () => {
  const { QualityGate } = await import("../src/quality-gate.js");
  const gate = new QualityGate({ profile: { id: "p", rubric: ["correct"], maxRepairs: 2 },
    snapshot: async () => ({ hash: "hash", path: "/snapshot" }),
    verify: async () => ({ status: "passed", checks: [] }),
    grade: async () => ({ status: "passed", candidate_hash: "other", criteria: [], findings: [] }),
  });
  const result = await gate.evaluate({ run_id: "r", worktree: "/w", baseline: "a".repeat(40), requirement: "fix", attempt: 1, signal: new AbortController().signal });
  assert.equal(result.grade?.cause?.code, "grader_invalid_submission");
});

test("quality gate propagates aborts and budget exhaustion instead of grading unknown", async () => {
  const { QualityGate } = await import("../src/quality-gate.js");
  const controller = new AbortController();
  const make = (grade: () => Promise<unknown>) => new QualityGate({ profile: { id: "p", rubric: ["correct"], maxRepairs: 2 },
    snapshot: async () => ({ hash: "hash", path: "/snapshot" }),
    verify: async () => ({ status: "passed", checks: [] }), grade });
  const input = { run_id: "r", worktree: "/w", baseline: "a".repeat(40), requirement: "fix", attempt: 1, signal: controller.signal };
  await assert.rejects(make(async () => { throw new Error("workflow_budget_exhausted"); }).evaluate(input), /workflow_budget_exhausted/);
  await assert.rejects(make(async () => { controller.abort(); throw new Error("grader_aborted"); }).evaluate(input));
});

function retryGate(grade: (call: number) => Promise<unknown>) {
  return import("../src/quality-gate.js").then(({ QualityGate }) => {
    let calls = 0;
    const gate = new QualityGate({ profile: { id: "p", rubric: ["correct"], maxRepairs: 2 },
      snapshot: async () => ({ hash: "hash", path: "/snapshot" }),
      verify: async () => ({ status: "passed", checks: [] }),
      grade: async () => grade(++calls),
    });
    return { gate, calls: () => calls };
  });
}
const gateInput = () => ({ run_id: "r", worktree: "/w", baseline: "a".repeat(40), requirement: "fix", attempt: 1, signal: new AbortController().signal });
const passedGrade = { status: "passed", candidate_hash: "hash", criteria: [{ id: "correct", passed: true, evidence: "ok" }], findings: [] };

test("quality gate retries a grader infrastructure error once", async () => {
  const { HarnessError } = await import("../src/error-taxonomy.js");
  const { gate, calls } = await retryGate(async call => { if (call === 1) throw new HarnessError("grader_timeout", "slow"); return passedGrade; });
  const result = await gate.evaluate(gateInput());
  assert.equal(result.action, "pass");
  assert.equal(calls(), 2);
  assert.equal(result.grade?.grader_attempts, 2);
});

test("quality gate fails with the grader cause after a second infrastructure error", async () => {
  const { HarnessError } = await import("../src/error-taxonomy.js");
  const { gate, calls } = await retryGate(async () => { throw new HarnessError("grader_timeout", "slow"); });
  const result = await gate.evaluate(gateInput());
  assert.equal(result.action, "fail");
  assert.equal(calls(), 2);
  assert.deepEqual(result.failure, { code: "verification_failed", cause_code: "grader_timeout" });
});

test("a grader that submits unknown is not retried and fails as insufficient_evidence", async () => {
  const { gate, calls } = await retryGate(async () => ({ status: "unknown", candidate_hash: "hash", criteria: [{ id: "correct", passed: false }], findings: [] }));
  const result = await gate.evaluate(gateInput());
  assert.equal(result.action, "fail");
  assert.equal(calls(), 1);
  assert.equal(result.grade?.cause?.code, "insufficient_evidence");
  assert.deepEqual(result.failure, { code: "verification_failed", cause_code: "insufficient_evidence" });
});
