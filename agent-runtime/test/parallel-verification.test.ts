import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

import { DockerVerifier, VerifierScheduler, validateProfile, verifyConcurrency, type ContainerResult, type RunContainer } from "../src/verification.js";

const base = { id: "p", repository: "o/r", image: `sha256:${"0".repeat(64)}`, rubric: ["ok"], allowedPaths: ["src"], isolation: "per_check" as const };
const spec = (name: string, kind?: "static" | "test") => ({ name, argv: ["run", name], cwd: ".", timeoutMs: 1000, ...(kind ? { kind } : {}) });

async function withCandidate<T>(work: (candidate: { hash: string; path: string }) => Promise<T>): Promise<T> {
  const root = await mkdtemp(path.join(tmpdir(), "parallel-verify-"));
  try { await mkdir(path.join(root, "candidate")); return await work({ hash: "h", path: path.join(root, "candidate") }); }
  finally { await rm(root, { recursive: true, force: true }); }
}

/** A scripted container: `outcome(mode, check)` decides the result; every start is recorded. */
function scripted(outcome: (mode: string, check: string) => ContainerResult | Promise<ContainerResult>) {
  const started: string[] = []; let inFlight = 0; let peak = 0;
  const run: RunContainer = async ({ args }) => {
    const spec = JSON.parse(args.at(-1)!) as { mode: string; name: string };
    started.push(`${spec.mode}:${spec.name}`); inFlight++; peak = Math.max(peak, inFlight);
    try { await new Promise(resolve => setTimeout(resolve, 5)); return await outcome(spec.mode, spec.name); } finally { inFlight--; }
  };
  return { run, started, peak: () => peak };
}
const ok: ContainerResult = { output: "", exit: 0, timedOut: false };
const fail: ContainerResult = { output: "src/a.py:1:1: E501 long", exit: 1, timedOut: false };
const lost: ContainerResult = { output: "", exit: null, timedOut: false };
const removeContainer = async () => undefined;

test("baseline and candidate run in parallel within the host scheduler limit", async () => {
  const profile = validateProfile({ ...base, checks: [spec("a"), spec("b")] });
  const container = scripted(() => ok);
  const verifier = new DockerVerifier(profile, { runContainer: container.run, removeContainer, scheduler: new VerifierScheduler(2) });
  const result = await withCandidate(candidate => verifier.verify(candidate, new AbortController().signal));
  assert.equal(result.status, "passed");
  assert.deepEqual(container.started.slice(0, 2).sort(), ["baseline:a", "candidate:a"]);
  assert.equal(container.peak(), 2);
  assert.deepEqual(result.checks.map(check => check.name), ["baseline:a", "baseline:b", "candidate:a", "candidate:b"]);
});

test("a failed check does not stop its mode; every failure is collected", async () => {
  const profile = validateProfile({ ...base, checks: [spec("a"), spec("b")] });
  const container = scripted(() => fail);
  const result = await withCandidate(candidate => new DockerVerifier(profile, { runContainer: container.run, removeContainer, scheduler: new VerifierScheduler(2) }).verify(candidate, new AbortController().signal));
  assert.equal(result.status, "failed");
  assert.equal(result.checks.length, 4);
  assert.ok(result.checks.every(check => check.failures?.[0]?.file === "src/a.py"));
});

test("an unknown check stops only its own mode", async () => {
  const profile = validateProfile({ ...base, checks: [spec("a"), spec("b")] });
  const container = scripted(mode => mode === "baseline" ? lost : ok);
  const result = await withCandidate(candidate => new DockerVerifier(profile, { runContainer: container.run, removeContainer, scheduler: new VerifierScheduler(2) }).verify(candidate, new AbortController().signal));
  assert.equal(result.status, "unknown");
  assert.deepEqual(result.checks.map(check => `${check.name}=${check.status}`), ["baseline:a=unknown", "candidate:a=passed", "candidate:b=passed"]);
});

test("static checks run first in each mode", async () => {
  const profile = validateProfile({ ...base, checks: [spec("tests", "test"), spec("lint", "static"), spec("types", "static")] });
  const container = scripted(() => ok);
  await withCandidate(candidate => new DockerVerifier(profile, { runContainer: container.run, removeContainer, scheduler: new VerifierScheduler(1) }).verify(candidate, new AbortController().signal, ["candidate"]));
  assert.deepEqual(container.started, ["candidate:lint", "candidate:types", "candidate:tests"]);
  assert.throws(() => validateProfile({ ...base, checks: [{ ...spec("x"), kind: "slow" }] }), /kind/);
});

test("the scheduler is shared: two verifiers never exceed the host limit", async () => {
  const profile = validateProfile({ ...base, checks: [spec("a"), spec("b")] });
  const container = scripted(() => ok);
  const scheduler = new VerifierScheduler(2);
  await withCandidate(candidate => Promise.all([0, 1].map(() => new DockerVerifier(profile, { runContainer: container.run, removeContainer, scheduler }).verify(candidate, new AbortController().signal))));
  assert.equal(container.started.length, 8);
  assert.equal(container.peak(), 2);
});

test("aborting a waiting check releases its queue slot", async () => {
  const scheduler = new VerifierScheduler(1);
  let release!: () => void;
  const first = scheduler.run(() => new Promise<void>(resolve => { release = resolve; }), new AbortController().signal);
  const controller = new AbortController();
  const waiting = scheduler.run(async () => "never", controller.signal);
  controller.abort(new Error("stopped"));
  await assert.rejects(waiting, /stopped/);
  release(); await first;
  assert.equal(await scheduler.run(async () => "next", new AbortController().signal), "next");
});

test("PI_AGENT_VERIFY_CONCURRENCY defaults to 2 and is validated", () => {
  assert.equal(verifyConcurrency({}), 2);
  assert.equal(verifyConcurrency({ PI_AGENT_VERIFY_CONCURRENCY: "3" }), 3);
  assert.throws(() => new VerifierScheduler(verifyConcurrency({ PI_AGENT_VERIFY_CONCURRENCY: "0" })));
  assert.throws(() => new VerifierScheduler(verifyConcurrency({ PI_AGENT_VERIFY_CONCURRENCY: "many" })));
});
