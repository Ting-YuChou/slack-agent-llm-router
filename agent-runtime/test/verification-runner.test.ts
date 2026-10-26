import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

import { DockerVerifier, VerifierScheduler, validateProfile, type RunContainer } from "../src/verification.js";
import { execute, parseRunnerRecords, runModeChecks, type RunnerDeps, type RunnerRecord, type RunnerSpec } from "../src/verification-runner.js";

const spec: RunnerSpec = { runner: "mode", mode: "candidate", baselinePaths: ["tests"], checks: [
  { name: "lint", file: "lint", argv: ["lint"], cwd: ".", timeoutMs: 1000 },
  { name: "unit tests", file: "unit_tests", argv: ["test"], cwd: "pkg", timeoutMs: 2000 },
] };

function fakeDeps(exits: Array<number | null | Error>) {
  const calls: string[] = []; const lines: string[] = []; const envs: Array<Record<string, string>> = []; const options: Array<{ logPath: string; timeoutMs: number; cwd: string }> = [];
  let clock = 0; let call = 0;
  const deps: RunnerDeps = {
    prepareWorkspace: (_spec, check) => { calls.push(`prepare:${check.name}`); return `/workspace/check/${check.cwd}`; },
    startServices: async seq => { calls.push(`services:${seq}`); return { HOME: `/tmp/check-${seq}/home`, TMPDIR: `/tmp/check-${seq}/tmp` }; },
    execute: async (check, opts) => {
      calls.push(`execute:${check.name}`); envs.push(opts.env); options.push(opts); clock += 5;
      const exit = exits[call++];
      if (exit instanceof Error) throw exit;
      return { exit, timedOut: false, logTruncated: false };
    },
    reap: async () => { calls.push("reap"); },
    emit: line => { lines.push(line); },
    now: () => clock,
  };
  return { deps, calls, lines, envs, options };
}

test("runModeChecks isolates each check and emits one ordered record per check", async () => {
  const { deps, calls, lines, envs, options } = fakeDeps([1, 0]);
  const records = await runModeChecks(spec, deps);
  assert.deepEqual(calls, ["prepare:lint", "services:1", "execute:lint", "reap", "prepare:unit tests", "services:2", "execute:unit tests", "reap"]);
  assert.deepEqual(records.map(record => [record.seq, record.name, record.exit]), [[1, "lint", 1], [2, "unit tests", 0]]);
  assert.deepEqual(lines.map(line => JSON.parse(line)), records);
  assert.ok(lines.every(line => line.endsWith("\n") && !line.slice(0, -1).includes("\n")));
  assert.ok(envs.every(env => env.PYTHONNOUSERSITE === "1"));
  assert.notEqual(envs[0].HOME, envs[1].HOME, "every check gets a fresh HOME");
  assert.deepEqual(options.map(item => [item.logPath, item.timeoutMs, item.cwd]), [["/out/lint.log", 1000, "/workspace/check/."], ["/out/unit_tests.log", 2000, "/workspace/check/pkg"]]);
});

test("runModeChecks stops the mode after a check without a trustworthy result and still reaps", async () => {
  const { deps, calls } = fakeDeps([new Error("redis down"), 0]);
  const records = await runModeChecks(spec, deps);
  assert.deepEqual(records.map(record => [record.name, record.exit, record.cause]), [["lint", null, "check_setup_failed"]]);
  assert.equal(calls.at(-1), "reap");
});

const record = (seq: number, name: string, exit: number | null = 0): RunnerRecord => ({ pi_verify: 1, seq, name, exit, timed_out: false, duration_ms: 1, log_truncated: false });
const lines = (...records: unknown[]) => records.map(item => typeof item === "string" ? item : JSON.stringify(item)).join("\n") + "\n";

test("parseRunnerRecords accepts only a strictly ordered, duplicate-free sequence", () => {
  const names = ["lint", "test"];
  assert.deepEqual(parseRunnerRecords(lines(record(1, "lint"), record(2, "test")), names), { records: [record(1, "lint"), record(2, "test")] });
  assert.equal(parseRunnerRecords(lines(record(1, "lint"), record(1, "lint")), names).violation, "runner_record_duplicate");
  assert.equal(parseRunnerRecords(lines(record(2, "test")), names).violation, "runner_record_out_of_order");
  assert.equal(parseRunnerRecords(lines(record(1, "test")), names).violation, "runner_record_out_of_order");
  assert.equal(parseRunnerRecords(lines("hello"), names).violation, "runner_output_invalid");
  assert.equal(parseRunnerRecords(lines(record(1, "lint", null), record(2, "test")), names).violation, "runner_record_after_stop");
  assert.deepEqual(parseRunnerRecords("", names), { records: [] });
});

test("execute caps the log, enforces its own timeout and kills the whole process group", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "runner-exec-"));
  try {
    const big = await execute({ name: "big", file: "big", argv: [process.execPath, "-e", "process.stdout.write('x'.repeat(3*1024*1024)); process.stderr.write(process.env.PYTHONNOUSERSITE)"], cwd: ".", timeoutMs: 10_000 },
      { cwd: root, env: { PATH: process.env.PATH ?? "", PYTHONNOUSERSITE: "1" }, logPath: path.join(root, "big.log"), timeoutMs: 10_000, maxLogBytes: 1024 * 1024 });
    assert.deepEqual(big, { exit: 0, timedOut: false, logTruncated: true });
    const log = await readFile(path.join(root, "big.log"), "utf8");
    assert.ok(log.length <= 1024 * 1024 + 32 && log.endsWith("[log truncated]\n"));

    const started = Date.now();
    const hung = await execute({ name: "hung", file: "hung", argv: [process.execPath, "-e", "require('child_process').spawn(process.execPath,['-e','setInterval(()=>{},1000)'],{stdio:'inherit'}); setInterval(()=>{},1000)"], cwd: ".", timeoutMs: 300 },
      { cwd: root, env: { PATH: process.env.PATH ?? "" }, logPath: path.join(root, "hung.log"), timeoutMs: 300, maxLogBytes: 1024 });
    assert.deepEqual(hung, { exit: null, timedOut: true, logTruncated: false, cause: "check_timeout" });
    assert.ok(Date.now() - started < 5000, "the grandchild holding the pipes does not keep the check alive");

    const failed = await execute({ name: "f", file: "f", argv: [process.execPath, "-e", "process.exit(3)"], cwd: ".", timeoutMs: 5000 },
      { cwd: root, env: { PATH: process.env.PATH ?? "" }, logPath: path.join(root, "f.log"), timeoutMs: 5000, maxLogBytes: 1024 });
    assert.equal(failed.exit, 3);
    const missing = await execute({ name: "m", file: "m", argv: ["/nonexistent/tool"], cwd: ".", timeoutMs: 5000 },
      { cwd: root, env: { PATH: "" }, logPath: path.join(root, "m.log"), timeoutMs: 5000, maxLogBytes: 1024 });
    assert.deepEqual(missing, { exit: null, timedOut: false, logTruncated: false, cause: "check_spawn_failed" });
  } finally { await rm(root, { recursive: true, force: true }); }
});

const base = { id: "p", repository: "o/r", image: `sha256:${"0".repeat(64)}`, rubric: ["ok"], allowedPaths: ["src"] };
const checks = [{ name: "lint", argv: ["lint"], cwd: ".", timeoutMs: 1000 }, { name: "test", argv: ["test"], cwd: ".", timeoutMs: 1000 }];

/** A per-mode container: writes each check's log into the mounted /out and prints `stdout(mode)`. */
function modeContainer(stdout: (mode: string) => string, logs: Record<string, string> = {}) {
  const runs: Array<{ mode: string; checks: string[] }> = [];
  const run: RunContainer = async ({ args }) => {
    const spec = JSON.parse(args.at(-1)!) as RunnerSpec;
    const out = args[args.findIndex(arg => arg.endsWith(",dst=/out")) ].split(",")[1].slice(4);
    runs.push({ mode: spec.mode, checks: spec.checks.map(check => check.name) });
    for (const check of spec.checks) await writeFile(path.join(out, `${check.file}.log`), logs[check.name] ?? "");
    return { output: "", stdout: stdout(spec.mode), exit: 0, timedOut: false };
  };
  return { run, runs };
}

async function verifyWith(container: RunContainer, profileChecks = checks) {
  const root = await mkdtemp(path.join(tmpdir(), "per-mode-"));
  try {
    await mkdir(path.join(root, "candidate"));
    const verifier = new DockerVerifier(validateProfile({ ...base, checks: profileChecks }), { runContainer: container, removeContainer: async () => undefined, scheduler: new VerifierScheduler(2) });
    return await verifier.verify({ hash: "h", path: path.join(root, "candidate") }, new AbortController().signal);
  } finally { await rm(root, { recursive: true, force: true }); }
}

test("per_mode is the default isolation: one container per mode, results from its records and logs", async () => {
  assert.equal(validateProfile({ ...base, checks }).isolation, "per_mode");
  assert.throws(() => validateProfile({ ...base, checks, isolation: "shared" }), /isolation/);
  const container = modeContainer(() => lines(record(1, "lint", 1), record(2, "test", 0)), { lint: "src/a.py:3:1: F401 'os' imported but unused\n", test: "5 passed in 0.1s\n" });
  const result = await verifyWith(container.run);
  assert.deepEqual(container.runs.map(run => run.mode).sort(), ["baseline", "candidate"]);
  assert.ok(container.runs.every(run => run.checks.join() === "lint,test"));
  assert.equal(result.status, "failed");
  const lint = result.checks.find(check => check.name === "candidate:lint")!;
  assert.equal(lint.failures?.[0]?.file, "src/a.py");
  assert.equal(result.checks.find(check => check.name === "candidate:test")?.passed, 5);
});

test("per_mode: each check records its kind, including the unknown one the runner never reached", async () => {
  const kinded = [{ ...checks[0], kind: "static" as const }, { ...checks[1], kind: "test" as const }];
  const done = await verifyWith(modeContainer(() => lines(record(1, "lint"), record(2, "test"))).run, kinded);
  assert.deepEqual(done.checks.filter(check => check.name.startsWith("candidate:")).map(check => `${check.name}=${check.kind}`), ["candidate:lint=static", "candidate:test=test"]);
  const missing = await verifyWith(modeContainer(() => lines(record(1, "lint"))).run, kinded);
  assert.equal(missing.checks.find(check => check.name === "candidate:test")?.kind, "test");
  const forged = await verifyWith(modeContainer(() => lines(record(1, "lint"), record(1, "lint"))).run, kinded);
  assert.equal(forged.checks.find(check => check.name === "candidate:lint")?.kind, "static");
});

test("per_mode: forged, duplicated or missing records make the mode unknown", async () => {
  const forged = await verifyWith(modeContainer(mode => mode === "candidate" ? lines(record(1, "lint"), record(2, "test"), record(2, "test")) : lines(record(1, "lint"), record(2, "test"))).run);
  assert.equal(forged.status, "unknown");
  assert.deepEqual(forged.checks.filter(check => check.name.startsWith("candidate:")).map(check => `${check.name}=${check.status}:${check.cause ?? ""}`), ["candidate:lint=passed:", "candidate:test=unknown:runner_record_duplicate"]);
  const missing = await verifyWith(modeContainer(() => lines(record(1, "lint"))).run);
  assert.equal(missing.status, "unknown");
  assert.equal(missing.checks.find(check => check.name === "candidate:test")?.cause, "runner_incomplete");
  const stopped = await verifyWith(modeContainer(() => lines(record(1, "lint", null))).run);
  assert.deepEqual(stopped.checks.map(check => check.name).sort(), ["baseline:lint", "candidate:lint"]);
  assert.equal(stopped.status, "unknown");
});
