// Per-mode check runner. It runs as PID 1 of the credential-free verification container,
// so checks (same uid) cannot kill or stop it; its stdout carries only check records.
import { spawn } from "node:child_process";
import { chmodSync, cpSync, createWriteStream, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, symlinkSync } from "node:fs";
import { createConnection } from "node:net";
import path from "node:path";

export const WORKSPACE = "/workspace/check";
export const MAX_LOG_BYTES = 8 * 1024 * 1024;

export interface RunnerCheck { name: string; file: string; argv: string[]; cwd: string; timeoutMs: number }
export interface RunnerSpec { runner: "mode"; mode: "baseline" | "candidate"; baselinePaths: string[]; checks: RunnerCheck[] }
/** One line on the runner's stdout per finished check, in order. */
export interface RunnerRecord { pi_verify: 1; seq: number; name: string; exit: number | null; timed_out: boolean; duration_ms: number; log_truncated: boolean; cause?: string }
export interface ExecuteResult { exit: number | null; timedOut: boolean; logTruncated: boolean; cause?: string }

export interface RunnerDeps {
  /** Recreates the workspace from the candidate (baseline paths restored in baseline mode); returns the check's cwd. */
  prepareWorkspace(spec: RunnerSpec, check: RunnerCheck): string;
  /** Fresh HOME and TMPDIR plus a task-local Redis; returns them as environment. */
  startServices(seq: number): Promise<Record<string, string>>;
  execute(check: RunnerCheck, options: { cwd: string; env: Record<string, string>; logPath: string; timeoutMs: number; maxLogBytes: number }): Promise<ExecuteResult>;
  /** Kills every process the check (or its services) left behind. */
  reap(): Promise<void>;
  emit(line: string): void;
  now(): number;
}

export function checkEnv(services: Record<string, string>): Record<string, string> {
  return { PATH: process.env.PATH ?? "/usr/local/bin:/usr/bin:/bin", PYTHONDONTWRITEBYTECODE: "1", PYTHONNOUSERSITE: "1", ...services };
}

/** Runs every check of one mode in order; a check without a trustworthy result ends the mode. */
export async function runModeChecks(spec: RunnerSpec, deps: RunnerDeps): Promise<RunnerRecord[]> {
  const records: RunnerRecord[] = [];
  for (const [index, check] of spec.checks.entries()) {
    const seq = index + 1; const started = deps.now();
    let result: ExecuteResult;
    try {
      const cwd = deps.prepareWorkspace(spec, check);
      const env = checkEnv(await deps.startServices(seq));
      result = await deps.execute(check, { cwd, env, logPath: path.join("/out", `${check.file}.log`), timeoutMs: check.timeoutMs, maxLogBytes: MAX_LOG_BYTES });
    } catch { result = { exit: null, timedOut: false, logTruncated: false, cause: "check_setup_failed" }; }
    finally { await deps.reap(); }
    const record: RunnerRecord = { pi_verify: 1, seq, name: check.name, exit: result.exit, timed_out: result.timedOut, duration_ms: Math.max(0, deps.now() - started), log_truncated: result.logTruncated, ...(result.cause ? { cause: result.cause } : {}) };
    records.push(record); deps.emit(`${JSON.stringify(record)}\n`);
    if (record.exit === null) break;
  }
  return records;
}

/**
 * Validates the runner's stdout against the checks the host asked for. Anything but a strictly
 * ordered, duplicate-free sequence of records is a protocol violation and the mode is unknown.
 */
export function parseRunnerRecords(stdout: string, expected: string[]): { records: RunnerRecord[]; violation?: string } {
  const records: RunnerRecord[] = [];
  for (const line of stdout.split("\n")) {
    if (!line.trim()) continue;
    let record: RunnerRecord;
    try { record = JSON.parse(line); } catch { return { records, violation: "runner_output_invalid" }; }
    const index = records.length;
    if (!record || record.pi_verify !== 1 || typeof record.seq !== "number") return { records, violation: "runner_output_invalid" };
    if (record.seq <= index) return { records, violation: "runner_record_duplicate" };
    if (record.seq !== index + 1 || record.name !== expected[index]) return { records, violation: "runner_record_out_of_order" };
    if ((record.exit !== null && !Number.isInteger(record.exit)) || typeof record.timed_out !== "boolean" || typeof record.duration_ms !== "number") return { records, violation: "runner_output_invalid" };
    if (index > 0 && records[index - 1].exit === null) return { records, violation: "runner_record_after_stop" };
    records.push(record);
  }
  return { records };
}

function writable(directory: string) {
  chmodSync(directory, 0o700);
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) writable(file);
    else if (entry.isFile()) chmodSync(file, statSync(file).mode & 0o111 ? 0o700 : 0o600);
    else throw new Error("Unsupported candidate file");
  }
}

/** Copies the read-only candidate into a fresh workspace; baseline mode restores the trusted test/config paths. */
export function prepareWorkspace(mode: RunnerSpec["mode"], baselinePaths: string[], checkCwd: string): string {
  rmSync(WORKSPACE, { recursive: true, force: true });
  mkdirSync(WORKSPACE, { mode: 0o700 });
  cpSync("/candidate", WORKSPACE, { recursive: true, force: true, preserveTimestamps: false, mode: 0 });
  writable(WORKSPACE);
  if (mode === "baseline") for (const name of baselinePaths) {
    const source = path.resolve("/baseline", name), target = path.resolve(WORKSPACE, name);
    if (!source.startsWith("/baseline/") || !target.startsWith(WORKSPACE + "/")) throw new Error("Invalid baseline path");
    rmSync(target, { recursive: true, force: true });
    if (existsSync(source)) cpSync(source, target, { recursive: true, force: true });
  }
  writable(WORKSPACE);
  mkdirSync(path.join(WORKSPACE, "agent-runtime"), { recursive: true });
  if (!existsSync(path.join(WORKSPACE, "agent-runtime/node_modules"))) symlinkSync("/opt/pi/node_modules", path.join(WORKSPACE, "agent-runtime/node_modules"));
  const cwd = path.resolve(WORKSPACE, checkCwd);
  if (cwd !== WORKSPACE && !cwd.startsWith(WORKSPACE + "/")) throw new Error("Invalid check cwd");
  return cwd;
}

async function redisReady(): Promise<boolean> {
  for (let attempt = 0; attempt < 50; attempt++) {
    const ready = await new Promise<boolean>(resolve => { const socket = createConnection({ host: "127.0.0.1", port: 6379 }); socket.once("connect", () => { socket.destroy(); resolve(true); }); socket.once("error", () => resolve(false)); socket.setTimeout(100, () => { socket.destroy(); resolve(false); }); });
    if (ready) return true;
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  return false;
}

/** Fresh HOME, TMPDIR and a task-local Redis that cannot reach any host or shared instance. */
export async function startServices(seq: number): Promise<Record<string, string>> {
  const root = `/tmp/check-${seq}`;
  rmSync(root, { recursive: true, force: true });
  for (const dir of ["home", "tmp", "redis"]) mkdirSync(path.join(root, dir), { recursive: true, mode: 0o700 });
  const redis = spawn("redis-server", ["--bind", "127.0.0.1", "--port", "6379", "--save", "", "--appendonly", "no", "--dir", path.join(root, "redis")], { stdio: "ignore", detached: true });
  const failed = new Promise<boolean>(resolve => redis.once("error", () => resolve(false)));
  if (!await Promise.race([redisReady(), failed])) throw new Error("redis_unavailable");
  return { HOME: path.join(root, "home"), TMPDIR: path.join(root, "tmp"), REDIS_HOST: "127.0.0.1", REDIS_PORT: "6379", REDIS_URL: "redis://127.0.0.1:6379/0" };
}

/** Runs one check in its own process group with a hard timeout; output goes only to its capped log. */
export function execute(check: RunnerCheck, options: { cwd: string; env: Record<string, string>; logPath: string; timeoutMs: number; maxLogBytes: number }): Promise<ExecuteResult> {
  return new Promise(resolve => {
    const log = createWriteStream(options.logPath, { mode: 0o600 });
    let bytes = 0; let logTruncated = false; let timedOut = false; let settled = false;
    const write = (chunk: Buffer) => {
      if (logTruncated) return;
      if (bytes + chunk.length > options.maxLogBytes) { log.write(chunk.subarray(0, options.maxLogBytes - bytes)); log.write("\n[log truncated]\n"); logTruncated = true; return; }
      bytes += chunk.length; log.write(chunk);
    };
    const child = spawn(check.argv[0], check.argv.slice(1), { cwd: options.cwd, env: options.env, stdio: ["ignore", "pipe", "pipe"], detached: true });
    child.stdout.on("data", write); child.stderr.on("data", write);
    // 'exit' may come before or after the pipes close; remember closes that already happened.
    const closed = [child.stdout, child.stderr].map(stream => new Promise(done => stream.once("close", done)));
    const killGroup = () => { try { if (child.pid) process.kill(-child.pid, "SIGKILL"); } catch { /* already gone */ } };
    const timer = setTimeout(() => { timedOut = true; killGroup(); }, options.timeoutMs);
    const finish = (result: ExecuteResult) => {
      if (settled) return; settled = true; clearTimeout(timer); killGroup();
      // Leftover processes may hold the pipes open; the reaper kills them, so do not wait long.
      const close = setTimeout(() => { child.stdout.destroy(); child.stderr.destroy(); }, 1000);
      const done = () => { clearTimeout(close); log.end(() => resolve({ ...result, logTruncated })); };
      Promise.all(closed).then(done, done);
    };
    child.once("error", () => finish({ exit: null, timedOut: false, logTruncated, cause: "check_spawn_failed" }));
    child.once("exit", code => finish(timedOut ? { exit: null, timedOut: true, logTruncated, cause: "check_timeout" } : { exit: code ?? null, timedOut: false, logTruncated, ...(code === null ? { cause: "check_killed" } : {}) }));
  });
}

/** SIGKILLs every live process except this runner. Zombies are already dead and are skipped. */
export async function reap(): Promise<void> {
  if (!existsSync("/proc/self/stat")) return;
  for (let round = 0; round < 50; round++) {
    const alive = readdirSync("/proc").filter(name => /^\d+$/.test(name) && Number(name) !== process.pid).filter(pid => {
      try { return !/^\d+ \(.*\) Z/.test(readFileSync(`/proc/${pid}/stat`, "utf8")); } catch { return false; }
    });
    if (!alive.length) return;
    for (const pid of alive) { try { process.kill(Number(pid), "SIGKILL"); } catch { /* exited */ } }
    await new Promise(resolve => setTimeout(resolve, 20));
  }
}

export const containerDeps: RunnerDeps = {
  prepareWorkspace: (spec, check) => prepareWorkspace(spec.mode, spec.baselinePaths, check.cwd),
  startServices, execute, reap,
  emit: line => { process.stdout.write(line); },
  now: () => performance.now(),
};
