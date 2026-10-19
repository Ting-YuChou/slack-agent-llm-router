import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile, lstat, copyFile, realpath, open } from "node:fs/promises";
import { constants } from "node:fs";
import path from "node:path";
import { validateWorkspacePath } from "./policy.js";
import { redact } from "./telemetry.js";
import { parseRunnerRecords, MAX_LOG_BYTES, type RunnerSpec } from "./verification-runner.js";
import { excerpt, MAX_REPORT_BYTES, parseDiagnostics, parseJUnit, type CheckFailure } from "./check-report.js";
import type { Candidate, CheckStatus, VerificationCheck, VerificationResult } from "./quality-gate.js";
const exec = promisify(execFile);
/** One trusted check. `{report}` in argv becomes the check's report path under /out. */
export interface VerificationCheckSpec { name: string; ciJob?: string; argv: string[]; cwd: string; timeoutMs: number; report?: { format: "junit" }; kind?: "static" | "test" }
export interface VerificationProfile {
  id: string; repository: string; image: string; rubric: string[]; allowedPaths: string[];
  baselinePaths: string[]; checks: VerificationCheckSpec[];
  maxRepairs: number; maxCostUsd: number; activeTimeoutMs: number;
  ciCheckout?: "head" | "merge";
  slack?: { team_id: string; channel_id: string; user_id: string }; workflows?: string[];
  structuredWorkflow?: { enabled: boolean; concurrency: number; maxCostUsd: number; activeTimeoutMs: number; discoveryCostUsd: number };
  /** per_mode (default): one container per mode, each check isolated inside it; per_check: one container per check. */
  isolation?: "per_mode" | "per_check";
}
export const REPORT_TOKEN = "{report}";
/** File-safe check name used for /out/<check>.report and /out/<check>.log. */
export const checkFileName = (name: string) => name.replace(/[^\w-]/g, "_");
export const reportPath = (name: string) => `/out/${checkFileName(name)}.report`;
/** argv with `{report}` replaced by the check's report path inside the verifier. */
export function checkArgv(check: VerificationCheckSpec): string[] { return check.argv.map(arg => arg.split(REPORT_TOKEN).join(reportPath(check.name))); }
const safeRelative = (p: string) => p === "." || (p.length > 0 && !path.isAbsolute(p) && !p.split(/[\\/]/).includes(".."));
export function validateProfile(raw: unknown): VerificationProfile {
  const p = raw as VerificationProfile;
  if (!p || typeof p.id !== "string" || !/^[\w.-]{1,80}$/.test(p.id) || !/^[\w.-]+\/[\w.-]+$/.test(p.repository) || typeof p.image !== "string" || !/^(?:[\w./:-]+@)?sha256:[a-f0-9]{64}$/.test(p.image)) throw new Error("Invalid verification profile identity/image");
  if (!Array.isArray(p.rubric) || !p.rubric.length || p.rubric.some(x => typeof x !== "string" || x.length > 2000) || new Set(p.rubric).size !== p.rubric.length) throw new Error("Invalid rubric");
  if (!Array.isArray(p.allowedPaths) || p.allowedPaths.some(x => typeof x !== "string" || !safeRelative(x))) throw new Error("Invalid allowed paths");
  if (!Array.isArray(p.checks) || !p.checks.length || p.checks.some(c => !c || !Array.isArray(c.argv) || !c.argv.length || c.argv.some(x => typeof x !== "string" || x.includes("\0")) || !safeRelative(c.cwd) || !Number.isInteger(c.timeoutMs) || c.timeoutMs <= 0 || typeof c.name !== "string")) throw new Error("Invalid verification command");
  if (p.isolation !== undefined && !["per_mode", "per_check"].includes(p.isolation)) throw new Error("Invalid verification isolation");
  if (p.checks.some(c => c.kind !== undefined && !["static", "test"].includes(c.kind))) throw new Error("Invalid check kind");
  if (p.checks.some(c => (c.report !== undefined && (!c.report || c.report.format !== "junit")) || (c.report === undefined && c.argv.some(x => x.includes(REPORT_TOKEN))))) throw new Error("Invalid check report");
  if (p.baselinePaths !== undefined && (!Array.isArray(p.baselinePaths) || p.baselinePaths.some(x => typeof x !== "string" || !safeRelative(x)))) throw new Error("Invalid baseline paths");
  if ((p.maxRepairs !== undefined && (!Number.isInteger(p.maxRepairs) || p.maxRepairs < 0 || p.maxRepairs > 2)) || (p.maxCostUsd !== undefined && (!Number.isFinite(p.maxCostUsd) || p.maxCostUsd <= 0 || p.maxCostUsd > 1)) || (p.activeTimeoutMs !== undefined && (!Number.isInteger(p.activeTimeoutMs) || p.activeTimeoutMs <= 0 || p.activeTimeoutMs > 900_000))) throw new Error("Invalid workflow limits");
  if (p.slack && [p.slack.team_id, p.slack.channel_id, p.slack.user_id].some(value => typeof value !== "string" || !value.trim())) throw new Error("Invalid Slack mapping");
  if (p.workflows !== undefined && (!Array.isArray(p.workflows) || !p.workflows.length || p.workflows.some(value => typeof value !== "string" || !value.trim()))) throw new Error("Invalid workflow allowlist");
  if (p.ciCheckout !== undefined && !["head", "merge"].includes(p.ciCheckout)) throw new Error("Invalid CI checkout mode");
  if (!p.allowedPaths.length || new Set(p.checks.map(c => c.name)).size !== p.checks.length) throw new Error("Empty scope or duplicate checks");
  if (p.structuredWorkflow !== undefined) {
    const s = p.structuredWorkflow;
    if (!s || typeof s.enabled !== 'boolean') throw new Error('Invalid structured workflow configuration');
    const config = { enabled: s.enabled, concurrency: s.concurrency ?? 2, maxCostUsd: s.maxCostUsd ?? 5, activeTimeoutMs: s.activeTimeoutMs ?? 1_800_000, discoveryCostUsd: s.discoveryCostUsd ?? 1.5 };
    if (!Number.isInteger(config.concurrency) || config.concurrency < 1 || config.concurrency > 4 || !Number.isFinite(config.maxCostUsd) || config.maxCostUsd <= 0 || config.maxCostUsd > 5 || !Number.isInteger(config.activeTimeoutMs) || config.activeTimeoutMs <= 0 || config.activeTimeoutMs > 1_800_000 || !Number.isFinite(config.discoveryCostUsd) || config.discoveryCostUsd <= 0 || config.discoveryCostUsd >= config.maxCostUsd) throw new Error('Invalid structured workflow limits');
    p.structuredWorkflow = config;
  }
  return { ...p, isolation: p.isolation ?? "per_mode", baselinePaths: p.baselinePaths ?? ["tests/", "agent-runtime/test/"], maxRepairs: Math.min(2, Math.max(0, p.maxRepairs ?? 2)), maxCostUsd: p.maxCostUsd ?? 1, activeTimeoutMs: p.activeTimeoutMs ?? 900_000 };
}
export async function loadProfile(file: string, repo: string, worktreeRoot: string) {
  const resolved = await realpath(file);
  for (const root of [repo, worktreeRoot]) {
    const relative = path.relative(await realpath(root).catch(() => path.resolve(root)), resolved);
    if (!relative.startsWith("..") && !path.isAbsolute(relative)) throw new Error("Verification profile must live outside the repository/worktrees");
  }
  const data = await readFile(resolved);
  return { profile: validateProfile(JSON.parse(data.toString())), checksum: createHash("sha256").update(data).digest("hex") };
}
export class CandidateStore {
  constructor(readonly root: string) {}
  async create(worktree: string, baseline: string, runId: string): Promise<Candidate> {
    if (!/^[a-f0-9]{40,64}$/.test(baseline)) throw new Error("Invalid baseline");
    await exec("git", ["add", "-N", "--", "."], { cwd: worktree });
    const patch = (await exec("git", ["diff", "--binary", baseline, "--"], { cwd: worktree, maxBuffer: 2 * 1024 * 1024 })).stdout;
    const paths = (await exec("git", ["ls-files", "-z", "--cached", "--others", "--exclude-standard"], { cwd: worktree, maxBuffer: 4 * 1024 * 1024 })).stdout.split("\0").filter(Boolean);
    const location = path.join(this.root, runId.replace(/[^\w-]/g, "_"), randomUUID());
    const snapshot = path.join(location, "candidate"); await mkdir(snapshot, { recursive: true, mode: 0o700 });
    const hash = createHash("sha256").update(baseline); let bytes = 0;
    for (const file of [...new Set(paths)].sort()) {
      if (path.resolve(worktree, file).startsWith(path.resolve(this.root) + path.sep)) continue;
      let stat; try { stat = await lstat(path.join(worktree, file)); } catch { continue; }
      if (!stat.isFile() || stat.isSymbolicLink()) throw new Error("Candidate contains an unsupported file");
      const checked = await validateWorkspacePath(worktree, file, "read");
      if (!checked.allowed) continue;
      bytes += stat.size; if (bytes > 64 * 1024 * 1024 || stat.size > 2 * 1024 * 1024) throw new Error("Candidate snapshot too large");
      const content = await readFile(path.join(worktree, file));
      hash.update(file).update("\0").update(String(stat.mode & 0o111)).update("\0").update(content);
      const target = path.join(snapshot, file); await mkdir(path.dirname(target), { recursive: true, mode: 0o700 });
      await writeFile(target, content, { mode: stat.mode & 0o111 ? 0o500 : 0o400 });
    }
    const artifact = path.join(location, "candidate.patch"); await writeFile(artifact, String(redact(patch)), { mode: 0o400 });
    // Baseline test/config copies cannot be changed by the candidate.
    const baselineDir = path.join(location, "baseline"); await mkdir(baselineDir, { mode: 0o700 });
    const originalFiles = (await exec("git", ["ls-tree", "-r", "-z", baseline], { cwd: worktree })).stdout.split("\0").filter(Boolean);
    for (const entry of originalFiles) {
      const split = entry.indexOf("\t"); const file = entry.slice(split + 1);
      const executable = entry.startsWith("100755 ");
      if (!(await validateWorkspacePath(worktree, file, "read")).allowed) continue;
      const data = (await exec("git", ["show", `${baseline}:${file}`], { cwd: worktree, encoding: "buffer", maxBuffer: 2 * 1024 * 1024 })).stdout;
      const dest = path.join(baselineDir, file); await mkdir(path.dirname(dest), { recursive: true }); await writeFile(dest, data, { mode: executable ? 0o500 : 0o400 });
    }
    return { hash: hash.digest("hex"), path: snapshot, patch, artifact };
  }
  async assertUnchanged(candidate: Candidate, worktree: string, baseline: string, runId: string) {
    const actual = await this.create(worktree, baseline, runId);
    if (candidate.hash !== actual.hash) throw new Error("candidate_changed_after_verification");
  }
}
/** What one verifier container produced. `exit` is null when the runner itself failed or was killed. */
export interface ContainerResult { output: string; exit: number | null; timedOut: boolean; stdout?: string }
export type RunContainer = (run: { name: string; args: string[]; timeoutMs: number; signal: AbortSignal }) => Promise<ContainerResult>;

export const runDockerContainer: RunContainer = async ({ name, args, timeoutMs, signal }) => {
  const stop = () => { void exec("docker", ["kill", name], { timeout: 5000 }).catch(() => {}); };
  signal.addEventListener("abort", stop, { once: true });
  try {
    const result = await exec("docker", args, { timeout: timeoutMs, signal, maxBuffer: 4 * 1024 * 1024, env: { PATH: process.env.PATH } });
    return { output: result.stdout + result.stderr, stdout: result.stdout, exit: 0, timedOut: false };
  } catch (error: any) {
    const output = String(error.stdout ?? "") + String(error.stderr ?? "");
    return { output, stdout: String(error.stdout ?? ""), exit: typeof error.code === "number" && ![125, 126, 127].includes(error.code) && !error.killed ? error.code : null, timedOut: error.killed === true && !signal.aborted };
  } finally { signal.removeEventListener("abort", stop); }
};

async function removeDockerContainer(id: string): Promise<void> {
  try { await exec("docker", ["rm", "-f", id], { timeout: 5000 }); }
  catch (error: any) { if (!/No such (?:container|object)/i.test(String(error.stderr))) throw new Error("verification_stop_unconfirmed"); }
}

/** Host-wide limit on concurrently running verifier containers. */
export class VerifierScheduler {
  private running = 0;
  private readonly waiting: Array<() => void> = [];
  constructor(readonly concurrency: number) {
    if (!Number.isInteger(concurrency) || concurrency < 1 || concurrency > 16) throw new Error("PI_AGENT_VERIFY_CONCURRENCY must be an integer from 1 to 16");
  }
  async run<T>(work: () => Promise<T>, signal: AbortSignal): Promise<T> {
    signal.throwIfAborted();
    if (this.running >= this.concurrency) {
      await new Promise<void>((resolve, reject) => {
        const ready = () => { signal.removeEventListener("abort", abort); resolve(); };
        const abort = () => { const index = this.waiting.indexOf(ready); if (index >= 0) this.waiting.splice(index, 1); reject(signal.reason); };
        this.waiting.push(ready); signal.addEventListener("abort", abort, { once: true });
      });
    } else this.running++;
    try { return await work(); }
    finally { const next = this.waiting.shift(); if (next) next(); else this.running--; }
  }
}
export function verifyConcurrency(env: NodeJS.ProcessEnv = process.env): number {
  const raw = env.PI_AGENT_VERIFY_CONCURRENCY;
  return raw === undefined || raw === "" ? 2 : Number(raw);
}
export const verifierScheduler = new VerifierScheduler(verifyConcurrency());

/** Static checks (lint, typecheck) first: they are cheap and their diagnostics are the most precise. */
export function orderChecks(checks: VerificationCheckSpec[]): VerificationCheckSpec[] {
  return [...checks.filter(check => check.kind === "static"), ...checks.filter(check => check.kind !== "static")];
}

export function combineStatus(checks: VerificationCheck[]): CheckStatus {
  return checks.some(c => c.status === "unknown") ? "unknown" : checks.every(c => c.status === "passed") ? "passed" : "failed";
}

type Mode = "baseline" | "candidate";
export class DockerVerifier {
  private readonly active = new Set<string>();
  private readonly runContainer: RunContainer;
  private readonly removeContainer: (id: string) => Promise<void>;
  private readonly scheduler: VerifierScheduler;
  async stop() { for (const id of this.active) await this.remove(id); }
  private async remove(id: string) { await this.removeContainer(id); this.active.delete(id); }
  constructor(private readonly profile: VerificationProfile, options: { runContainer?: RunContainer; removeContainer?: (id: string) => Promise<void>; scheduler?: VerifierScheduler } = {}) {
    this.runContainer = options.runContainer ?? runDockerContainer;
    this.removeContainer = options.removeContainer ?? removeDockerContainer;
    this.scheduler = options.scheduler ?? verifierScheduler;
  }
  /** Baseline and candidate run side by side; a failed check never hides the next one, an unknown one ends its mode. */
  async verify(candidate: Candidate, signal: AbortSignal, modes: Mode[] = ["baseline", "candidate"], checks = this.profile.checks): Promise<VerificationResult> {
    const ordered = orderChecks(checks);
    const settled = await Promise.allSettled(modes.map(mode => this.runMode(candidate, mode, ordered, signal)));
    signal.throwIfAborted();
    const failure = settled.find(result => result.status === "rejected");
    if (failure?.status === "rejected") throw failure.reason;
    const results = settled.flatMap(result => result.status === "fulfilled" ? result.value : []);
    return { status: combineStatus(results), checks: results };
  }
  private async runMode(candidate: Candidate, mode: Mode, checks: VerificationCheckSpec[], signal: AbortSignal): Promise<VerificationCheck[]> {
    const out = path.join(path.dirname(candidate.path), "out", mode); await mkdir(out, { recursive: true, mode: 0o700 });
    if (this.profile.isolation !== "per_check") return this.scheduler.run(() => this.runModeContainer(candidate, mode, checks, out, signal), signal);
    const results: VerificationCheck[] = [];
    for (const check of checks) {
      signal.throwIfAborted();
      const result = await this.scheduler.run(() => this.runCheck(candidate, mode, check, out, signal), signal);
      results.push(result);
      if (result.status === "unknown") break;
    }
    return results;
  }
  /** One container runs the whole mode; each check gets a fresh workspace, HOME, Redis and its own timeout inside it. */
  private async runModeContainer(candidate: Candidate, mode: Mode, checks: VerificationCheckSpec[], out: string, signal: AbortSignal): Promise<VerificationCheck[]> {
    const container = "pi-verify-" + randomUUID();
    const spec: RunnerSpec = { runner: "mode", mode, baselinePaths: this.profile.baselinePaths, checks: checks.map(check => ({ name: check.name, file: checkFileName(check.name), argv: checkArgv(check), cwd: check.cwd, timeoutMs: check.timeoutMs })) };
    const args = ["run", "--rm", "--name", container, ...this.isolationArgs(candidate, out), this.profile.image, "node", "/opt/pi/dist/src/verification-entry.js", JSON.stringify(spec)];
    // Each check's own timeout plus workspace copy and Redis start-up.
    const timeoutMs = checks.reduce((total, check) => total + check.timeoutMs + 15_000, 30_000);
    this.active.add(container);
    let result: ContainerResult;
    try { result = await this.runContainer({ name: container, args, timeoutMs, signal }); }
    finally { await this.remove(container); }
    signal.throwIfAborted();
    const { records, violation } = parseRunnerRecords(result.stdout ?? "", checks.map(check => check.name));
    const results: VerificationCheck[] = [];
    for (const record of records) {
      const check = checks[record.seq - 1];
      const output = (await readLog(out, check.name)) ?? "";
      const artifact = path.join(path.dirname(candidate.path), `${mode}-${checkFileName(check.name)}.log`);
      await writeFile(artifact, String(redact(output)), { mode: 0o600 });
      const exit = record.exit !== null && [125, 126, 127].includes(record.exit) ? null : record.exit;
      results.push({ name: `${mode}:${check.name}`, status: exit === 0 ? "passed" : exit === null ? "unknown" : "failed", exit_code: exit, duration_ms: record.duration_ms, ...parseVerificationCounts(output), artifact,
        ...(exit === 0 ? {} : { failures: await checkFailures(check, output, out), excerpt: excerpt(output) }),
        ...(record.timed_out ? { timed_out: true, cause: "check_timeout" } : exit === null ? { cause: record.cause ?? "check_runner_failed" } : {}) });
    }
    const unknown = (check: VerificationCheckSpec, cause: string, timedOut = false): VerificationCheck => ({ name: `${mode}:${check.name}`, status: "unknown", exit_code: null, cause, ...(timedOut ? { timed_out: true } : {}) });
    // A forged or duplicated record means the latest accepted one cannot be trusted either.
    if (violation) results.splice(Math.max(0, results.length - 1), 1, unknown(checks[Math.max(0, results.length - 1)], violation));
    else if (results.length < checks.length && results.at(-1)?.status !== "unknown") results.push(unknown(checks[results.length], result.timedOut ? "mode_timeout" : "runner_incomplete", result.timedOut));
    return results;
  }
  private async runCheck(candidate: Candidate, mode: Mode, check: VerificationCheckSpec, out: string, signal: AbortSignal): Promise<VerificationCheck> {
    const started = performance.now();
    const container = "pi-verify-" + randomUUID();
    const args = ["run", "--rm", "--name", container, ...this.isolationArgs(candidate, out), this.profile.image, "node", "/opt/pi/dist/src/verification-entry.js", JSON.stringify({ ...check, argv: checkArgv(check), mode, baselinePaths: this.profile.baselinePaths })];
    this.active.add(container);
    let result: ContainerResult;
    try { result = await this.runContainer({ name: container, args, timeoutMs: check.timeoutMs, signal }); }
    finally { await this.remove(container); }
    signal.throwIfAborted();
    const { output, exit, timedOut } = result;
    const artifact = path.join(path.dirname(candidate.path), `${mode}-${checkFileName(check.name)}.log`);
    await writeFile(artifact, String(redact(output)), { mode: 0o600 });
    return { name: `${mode}:${check.name}`, status: exit === 0 ? "passed" : exit === null ? "unknown" : "failed", exit_code: exit, duration_ms: performance.now() - started, ...parseVerificationCounts(output), artifact,
      ...(exit === 0 ? {} : { failures: await checkFailures(check, output, out), excerpt: excerpt(output) }),
      ...(timedOut ? { timed_out: true, cause: "check_timeout" } : exit === null ? { cause: "check_runner_failed" } : {}) };
  }
  private isolationArgs(candidate: Candidate, out: string): string[] {
    return ["--label", "slack-pi-agent-session=true", "--network", "none", "--read-only", "--user", `${process.getuid?.()}:${process.getgid?.()}`, "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--cpus", "2", "--memory", "2g", "--pids-limit", "256", "--tmpfs", "/tmp:rw,nosuid,size=536870912", "--tmpfs", "/workspace:rw,nosuid,size=1073741824,mode=1777", "--mount", `type=bind,src=${candidate.path},dst=/candidate,readonly`, "--mount", `type=bind,src=${path.join(path.dirname(candidate.path), "baseline")},dst=/baseline,readonly`, "--mount", `type=bind,src=${out},dst=/out`];
  }
}

export function parseVerificationCounts(output: string) {
  const passed = output.match(/(\d+) passed/) ?? output.match(/(?:^|\n)(?:#|ℹ)\s*pass\s+(\d+)\b/);
  const failed = output.match(/(\d+) failed/) ?? output.match(/(?:^|\n)(?:#|ℹ)\s*fail\s+(\d+)\b/);
  return { passed: passed ? Number(passed[1]) : null, failed: failed ? Number(failed[1]) : null };
}

/** A file the check wrote to /out; only a regular file within `max` bytes is read, never through a symlink. */
async function readOut(file: string, max: number): Promise<Buffer | undefined> {
  let handle;
  try { handle = await open(file, constants.O_RDONLY | constants.O_NOFOLLOW); }
  catch { return undefined; }
  try {
    const stat = await handle.stat();
    if (!stat.isFile() || stat.size > max) return undefined;
    return await handle.readFile();
  } finally { await handle.close(); }
}
export const readReport = (out: string, name: string) => readOut(path.join(out, `${checkFileName(name)}.report`), MAX_REPORT_BYTES);
/** The runner caps logs at 8 MiB plus its truncation marker. */
export const readLog = async (out: string, name: string) => (await readOut(path.join(out, `${checkFileName(name)}.log`), MAX_LOG_BYTES + 64))?.toString();

/** JUnit failures when the check has a parseable report, otherwise diagnostics parsed from its log. */
async function checkFailures(check: VerificationCheckSpec, output: string, out: string): Promise<CheckFailure[]> {
  if (check.report?.format === "junit") {
    const report = await readReport(out, check.name);
    try { if (report) { const failures = parseJUnit(report, check.cwd); if (failures.length) return failures; } } catch { /* fall back to the log */ }
  }
  return parseDiagnostics(output, check.cwd);
}
