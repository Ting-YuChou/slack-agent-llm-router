import assert from "node:assert/strict";
import { test } from "node:test";
test('verification counts preserve Node TAP and pytest results without inventing counts for typecheck', async () => {
  const module: any = await import('../src/verification.js');
  assert.equal(typeof module.parseVerificationCounts,'function');
  assert.deepEqual(module.parseVerificationCounts('# tests 6\n# pass 5\n# fail 1\n'),{ passed: 5,failed: 1 });
  assert.deepEqual(module.parseVerificationCounts('7 passed, 1 failed in 0.03s'),{ passed: 7,failed: 1 });
  assert.deepEqual(module.parseVerificationCounts('typecheck completed'),{ passed: null,failed: null });
});

test("startup readiness does not finish when only the Docker CLI spawned", async () => {
  const module = await import("../src/startup-timing.js" as string).catch(() => null);
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
  const module = await import("../src/quality-gate.js" as string).catch(() => null);
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
  const module = await import("../src/quality-gate.js" as string).catch(() => null);
  assert.ok(module, "quality gate implementation exists");
  assert.throws(() => module.validateGrade({ status: "passed", candidate_hash: "other", criteria: [], findings: [] }, "candidate", ["correct"]));
  assert.throws(() => module.validateGrade({ status: "passed", candidate_hash: "candidate", criteria: [], findings: [] }, "candidate", ["correct"]));
  assert.throws(() => module.validateGrade({ status: "passed", candidate_hash: "candidate", criteria: [{ id: "correct", passed: false }], findings: [] }, "candidate", ["correct"]));
});

test("budget retains unknown usage reservations and pauses only human waiting", async () => {
  const module = await import("../src/workflow-budget.js" as string).catch(() => null);
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
  const m = await import("../src/verification.js" as string).catch(() => null);
  assert.ok(m, "verification implementation exists");
  const profile = { id: "p", repository: "o/r", image: "repo@sha256:" + "a".repeat(64), rubric: ["correct"], checks: [{ name: "unit", argv: ["python", "-m", "pytest"], cwd: ".", timeoutMs: 1000 }], allowedPaths: ["src/", "tests/"] };
  assert.equal(m.validateProfile(profile).checks.length, 1);
  assert.throws(() => m.validateProfile({ ...profile, image: "repo:latest" }));
  assert.throws(() => m.validateProfile({ ...profile, checks: [{ name: "bad", argv: "curl bad | sh" }] }));
  assert.throws(() => m.validateProfile({ ...profile, checks: [{ name: "bad", argv: ["pytest"], cwd: "../", timeoutMs: 1 }] }));
});

test("candidate snapshot includes untracked code and changes its identity on modification", async () => {
  const m = await import("../src/verification.js" as string).catch(() => null);
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
    assert.match(await readFile(a.artifact, "utf8"), /candidate/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("grader Docker invocation has read-only source, fresh context and no mutation tools", async () => {
  const m = await import("../src/grader.js" as string).catch(() => null);
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
