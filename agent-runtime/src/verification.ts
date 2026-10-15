import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile, lstat, copyFile, realpath } from "node:fs/promises";
import path from "node:path";
import { validateWorkspacePath } from "./policy.js";
import { redact } from "./telemetry.js";
import type { Candidate, VerificationResult } from "./quality-gate.js";
const exec = promisify(execFile);
export interface VerificationProfile {
  id: string; repository: string; image: string; rubric: string[]; allowedPaths: string[];
  baselinePaths: string[]; checks: Array<{ name: string; ciJob?: string; argv: string[]; cwd: string; timeoutMs: number }>;
  maxRepairs: number; maxCostUsd: number; activeTimeoutMs: number;
  ciCheckout?: "head" | "merge";
  slack?: { team_id: string; channel_id: string; user_id: string }; workflows?: string[];
  structuredWorkflow?: { enabled: boolean; concurrency: number; maxCostUsd: number; activeTimeoutMs: number; discoveryCostUsd: number };
}
const safeRelative = (p: string) => p === "." || (p.length > 0 && !path.isAbsolute(p) && !p.split(/[\\/]/).includes(".."));
export function validateProfile(raw: unknown): VerificationProfile {
  const p = raw as VerificationProfile;
  if (!p || typeof p.id !== "string" || !/^[\w.-]{1,80}$/.test(p.id) || !/^[\w.-]+\/[\w.-]+$/.test(p.repository) || typeof p.image !== "string" || !/^(?:[\w./:-]+@)?sha256:[a-f0-9]{64}$/.test(p.image)) throw new Error("Invalid verification profile identity/image");
  if (!Array.isArray(p.rubric) || !p.rubric.length || p.rubric.some(x => typeof x !== "string" || x.length > 2000) || new Set(p.rubric).size !== p.rubric.length) throw new Error("Invalid rubric");
  if (!Array.isArray(p.allowedPaths) || p.allowedPaths.some(x => typeof x !== "string" || !safeRelative(x))) throw new Error("Invalid allowed paths");
  if (!Array.isArray(p.checks) || !p.checks.length || p.checks.some(c => !c || !Array.isArray(c.argv) || !c.argv.length || c.argv.some(x => typeof x !== "string" || x.includes("\0")) || !safeRelative(c.cwd) || !Number.isInteger(c.timeoutMs) || c.timeoutMs <= 0 || typeof c.name !== "string")) throw new Error("Invalid verification command");
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
  return { ...p, baselinePaths: p.baselinePaths ?? ["tests/", "agent-runtime/test/"], maxRepairs: Math.min(2, Math.max(0, p.maxRepairs ?? 2)), maxCostUsd: p.maxCostUsd ?? 1, activeTimeoutMs: p.activeTimeoutMs ?? 900_000 };
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
export class DockerVerifier {
  private readonly active = new Set<string>();
  async stop() { for (const id of this.active) await this.remove(id); }
  private async remove(id: string) {
    try { await exec("docker", ["rm", "-f", id], { timeout: 5000 }); }
    catch (error: any) { if (!/No such (?:container|object)/i.test(String(error.stderr))) throw new Error("verification_stop_unconfirmed"); }
    this.active.delete(id);
  }
  constructor(private readonly profile: VerificationProfile) {}
  async verify(candidate: Candidate, signal: AbortSignal): Promise<VerificationResult> {
    const checks: VerificationResult["checks"] = [];
    for (const mode of ["baseline", "candidate"] as const) for (const check of this.profile.checks) {
      signal.throwIfAborted(); const started = performance.now();
      const container = "pi-verify-" + randomUUID();
      const args = ["run", "--rm", "--name", container, "--label", "slack-pi-agent-session=true", "--network", "none", "--read-only", "--user", `${process.getuid?.()}:${process.getgid?.()}`, "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--cpus", "2", "--memory", "2g", "--pids-limit", "256", "--tmpfs", "/tmp:rw,nosuid,size=536870912", "--tmpfs", "/workspace:rw,nosuid,size=1073741824,mode=1777", "--mount", `type=bind,src=${candidate.path},dst=/candidate,readonly`, "--mount", `type=bind,src=${path.join(path.dirname(candidate.path), "baseline")},dst=/baseline,readonly`, this.profile.image, "node", "/opt/pi/dist/src/verification-entry.js", JSON.stringify({ ...check, mode, baselinePaths: this.profile.baselinePaths })];
      this.active.add(container);
      let output = ""; let exit: number | null = null;
      const stop = () => { void exec("docker", ["kill", container], { timeout: 5000 }).catch(() => {}); };
      signal.addEventListener("abort", stop, { once: true });
      try {
        const result = await exec("docker", args, { timeout: check.timeoutMs, signal, maxBuffer: 4 * 1024 * 1024, env: { PATH: process.env.PATH } }); output = result.stdout + result.stderr; exit = 0;
      } catch (error: any) { output = String(error.stdout ?? "") + String(error.stderr ?? ""); exit = typeof error.code === "number" && ![125, 126, 127].includes(error.code) && !error.killed ? error.code : null; }
      finally { signal.removeEventListener("abort", stop); await this.remove(container); }
      signal.throwIfAborted();
      const artifact = path.join(path.dirname(candidate.path), `${mode}-${check.name.replace(/[^\w-]/g, "_")}.log`);
      await writeFile(artifact, String(redact(output)), { mode: 0o600 });
      checks.push({ name: `${mode}:${check.name}`, status: exit === 0 ? "passed" : exit === null ? "unknown" : "failed", exit_code: exit, duration_ms: performance.now() - started, ...parseVerificationCounts(output), artifact });
      if (exit === null) return { status: "unknown", checks };
    }
    return { status: checks.every(c => c.status === "passed") ? "passed" : "failed", checks };
  }
}

export function parseVerificationCounts(output: string) {
  const passed = output.match(/(\d+) passed/) ?? output.match(/(?:^|\n)(?:#|ℹ)\s*pass\s+(\d+)\b/);
  const failed = output.match(/(\d+) failed/) ?? output.match(/(?:^|\n)(?:#|ℹ)\s*fail\s+(\d+)\b/);
  return { passed: passed ? Number(passed[1]) : null, failed: failed ? Number(failed[1]) : null };
}
