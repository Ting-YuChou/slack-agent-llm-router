import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

import { ScopeViolationError } from "../src/candidate-scope.js";
import { DockerVerifier, VerifierScheduler, validateProfile, type ContainerResult, type RunContainer } from "../src/verification.js";
import { MAX_WRITER_CHECKS_BYTES, WriterChecks, type WriterChecksRequest } from "../src/writer-checks.js";

const base = { id: "p", repository: "o/r", image: `sha256:${"0".repeat(64)}`, rubric: ["ok"], allowedPaths: ["src"], isolation: "per_check" as const };
const checks = [{ name: "lint", argv: ["lint"], cwd: ".", timeoutMs: 1000 }, { name: "unit", argv: ["unit"], cwd: ".", timeoutMs: 1000 }, { name: "slow", argv: ["slow"], cwd: ".", timeoutMs: 1000 }];

async function harness(options: { outcome?: (check: string) => ContainerResult | Promise<ContainerResult>; snapshotError?: Error; writerChecks?: object } = {}) {
  const root = await mkdtemp(path.join(tmpdir(), "writer-checks-"));
  await mkdir(path.join(root, "candidate"));
  const profile = validateProfile({ ...base, checks, ...(options.writerChecks ? { writerChecks: options.writerChecks } : {}) });
  const started: string[] = []; const removed: string[] = []; const signals: AbortSignal[] = [];
  const runContainer: RunContainer = async ({ name, args, signal }) => {
    const spec = JSON.parse(args.at(-1)!) as { mode: string; name: string };
    started.push(`${spec.mode}:${spec.name}`); signals.push(signal);
    return (await options.outcome?.(spec.name)) ?? { output: "", exit: 0, timedOut: false };
  };
  const service = new WriterChecks({ profile,
    snapshot: async () => { if (options.snapshotError) throw options.snapshotError; return { hash: "h", path: path.join(root, "candidate"), artifact: path.join(root, "candidate.patch") }; },
    verifier: () => new DockerVerifier(profile, { runContainer, removeContainer: async id => { removed.push(id); }, scheduler: new VerifierScheduler(2) }),
  });
  const request = (extra: Partial<WriterChecksRequest> = {}): WriterChecksRequest => ({ run_id: "run-1", worktree: "/w", baseline: "a".repeat(40), signal: new AbortController().signal, ...extra });
  return { root, service, started, removed, signals, request, cleanup: () => rm(root, { recursive: true, force: true }) };
}

test("run_checks runs only the candidate mode and only the requested profile checks", async () => {
  const h = await harness({ outcome: name => name === "unit" ? { output: `src/a.py:2:1: E999 bad\nlog at ${"/x"}`, exit: 1, timedOut: false } : { output: "", exit: 0, timedOut: false } });
  try {
    const response = await h.service.run(h.request({ checks: ["unit", "lint", "unit"] }));
    assert.deepEqual(h.started, ["candidate:lint", "candidate:unit"]);
    assert.equal(response.status, "failed");
    assert.deepEqual(response.checks, [{ name: "lint", status: "passed" }, { name: "unit", status: "failed" }]);
    assert.match(response.report, /src\/a\.py:2:1 E999 bad/);
    assert.ok(!JSON.stringify(response).includes(h.root), "no host paths reach the writer");
  } finally { await h.cleanup(); }
});

test("run_checks defaults to the profile's writer checks", async () => {
  const h = await harness({ writerChecks: { defaultChecks: ["lint"], maxCalls: 2 } });
  try {
    await h.service.run(h.request());
    assert.deepEqual(h.started, ["candidate:lint"]);
  } finally { await h.cleanup(); }
});

test("unknown checks and scope violations come back as results, not errors", async () => {
  const h = await harness({ snapshotError: new ScopeViolationError(["docs/x.md", "README.md"]) });
  try {
    const unknown = await h.service.run(h.request({ checks: ["lint", "deploy"] }));
    assert.equal(unknown.status, "rejected"); assert.equal(unknown.reason, "unknown_check");
    assert.match(unknown.report, /"deploy"/);
    const scope = await h.service.run(h.request({ checks: ["lint"] }));
    assert.equal(scope.status, "scope_violation");
    assert.deepEqual(scope.files, ["docs/x.md", "README.md"]);
    assert.deepEqual(h.started, []);
  } finally { await h.cleanup(); }
});

test("run_checks output is bounded to 16 KiB", async () => {
  const noisy = Array.from({ length: 3000 }, (_, i) => `src/f${i}.py:${i + 1}:1: E501 ${"x".repeat(60)} "quoted"`).join("\n");
  const h = await harness({ outcome: () => ({ output: noisy, exit: 1, timedOut: false }) });
  try {
    const response = await h.service.run(h.request());
    assert.ok(Buffer.byteLength(JSON.stringify(response)) <= MAX_WRITER_CHECKS_BYTES);
    assert.match(response.report, /more omitted/);
  } finally { await h.cleanup(); }
});

test("stop(run) aborts in-flight checks and removes their containers", async () => {
  let entered!: () => void; const running = new Promise<void>(resolve => { entered = resolve; });
  const h = await harness({ outcome: name => name === "slow" ? new Promise(() => { entered(); }) : { output: "", exit: 0, timedOut: false } });
  try {
    const pending = h.service.run(h.request({ checks: ["slow"] }));
    await running;
    await h.service.stop("run-1");
    assert.ok(h.signals.at(-1)?.aborted);
    assert.equal(h.removed.length, 1);
    const busy = await h.service.run(h.request({ checks: ["lint"] }));
    assert.equal(busy.reason, "busy", "a stopped call that has not unwound still holds the run");
    void pending.catch(() => undefined);
  } finally { await h.cleanup(); }
});

test("profile writerChecks: maxCalls defaults to 4 and is at most 10", () => {
  assert.deepEqual(validateProfile({ ...base, checks }).writerChecks, { defaultChecks: ["lint", "unit", "slow"], maxCalls: 4 });
  assert.throws(() => validateProfile({ ...base, checks, writerChecks: { maxCalls: 11 } }), /writer checks/);
  assert.throws(() => validateProfile({ ...base, checks, writerChecks: { defaultChecks: ["deploy"] } }), /writer checks/);
});
