import { execFile } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { copyFile, lstat, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

import { runDockerContainer, type RunContainer } from "./verification.js";
import { RESOURCE_LIMIT_FLAGS } from "./container-flags.js";
import { hardenedGitEnv } from "./worktree-manager.js";

const exec = promisify(execFile);
/** Same ceiling as a single writer write: a fixer cannot grow a file past what the agent could write. */
const MAX_FIXED_BYTES = 256 * 1024;

/** A trusted deterministic formatter (black, ruff --fix…) run on the candidate before verification. */
export interface FixerSpec { name: string; argv: string[]; cwd: string; timeoutMs: number }
export interface FixResult { name: string; files: string[] }
export interface FixOutcome { fixes: FixResult[]; cause?: "fixer_failed" | "fixer_timeout" }
export type RunFixer = (fixer: FixerSpec, workspace: string, signal: AbortSignal) => Promise<{ exit: number | null; timedOut: boolean }>;

const lines = (out: string) => out.split("\0").filter(Boolean);
const digest = async (file: string) => { try { return createHash("sha256").update(await readFile(file)).digest("hex"); } catch { return undefined; } };
async function regularFile(file: string): Promise<boolean> { try { return (await lstat(file)).isFile(); } catch { return false; } }

/** Files the candidate added or modified, as regular files; deletions and symlinks are never fixed. */
async function changedFiles(worktree: string, baseline: string): Promise<string[]> {
  const tracked = lines((await exec("git", ["diff", "--name-only", "-z", baseline, "--"], { cwd: worktree, env: hardenedGitEnv(), maxBuffer: 4 * 1024 * 1024 })).stdout);
  const untracked = lines((await exec("git", ["ls-files", "--others", "--exclude-standard", "-z"], { cwd: worktree, env: hardenedGitEnv(), maxBuffer: 4 * 1024 * 1024 })).stdout);
  const files: string[] = [];
  for (const file of new Set([...tracked, ...untracked])) if (await regularFile(path.join(worktree, file))) files.push(file);
  return files.sort();
}

/**
 * Runs the profile's fixers, in order, on a scratch copy of the worktree, then copies their edits back only for
 * files the candidate itself changed. A fixer that crashes or times out discards every fixer's work, so the
 * worktree is either fully fixed or untouched; a non-zero exit (ruff with unfixable findings) still counts.
 */
export async function applyFixers(input: { fixers: FixerSpec[]; worktree: string; baseline: string; scratch: string; run: RunFixer; signal: AbortSignal }): Promise<FixOutcome> {
  if (!input.fixers.length) return { fixes: [] };
  const changed = await changedFiles(input.worktree, input.baseline);
  if (!changed.length) return { fixes: [] };
  const workspace = path.join(input.scratch, randomUUID());
  await mkdir(workspace, { recursive: true, mode: 0o700 });
  try {
    const all = lines((await exec("git", ["ls-files", "-z", "--cached", "--others", "--exclude-standard"], { cwd: input.worktree, env: hardenedGitEnv(), maxBuffer: 4 * 1024 * 1024 })).stdout);
    for (const file of all) {
      const source = path.join(input.worktree, file);
      if (!await regularFile(source)) continue;
      await mkdir(path.dirname(path.join(workspace, file)), { recursive: true });
      await copyFile(source, path.join(workspace, file));
    }
    const fixes: FixResult[] = [];
    for (const fixer of input.fixers) {
      input.signal.throwIfAborted();
      const before = await Promise.all(changed.map(file => digest(path.join(workspace, file))));
      let result: { exit: number | null; timedOut: boolean };
      try { result = await input.run(fixer, workspace, input.signal); }
      catch (error) { if (input.signal.aborted) throw error; return { fixes: [], cause: "fixer_failed" }; }
      if (result.timedOut) return { fixes: [], cause: "fixer_timeout" };
      if (result.exit === null) return { fixes: [], cause: "fixer_failed" };
      const after = await Promise.all(changed.map(file => digest(path.join(workspace, file))));
      const touched = changed.filter((_, index) => after[index] !== undefined && after[index] !== before[index]);
      if (touched.length) fixes.push({ name: fixer.name, files: touched });
    }
    for (const file of new Set(fixes.flatMap(fix => fix.files))) {
      const fixed = await readFile(path.join(workspace, file));
      const target = path.join(input.worktree, file);
      // Re-check: the worktree must still hold a regular file there, never a symlink planted meanwhile.
      if (fixed.length > MAX_FIXED_BYTES || !await regularFile(target)) continue;
      await writeFile(target, fixed);
    }
    return { fixes };
  } finally { await rm(workspace, { recursive: true, force: true }); }
}

/** Runs a fixer in the trusted verification image: no network, read-only root, only the scratch copy mounted. */
export function dockerFixerRunner(image: string, runContainer: RunContainer = runDockerContainer): RunFixer {
  return async (fixer, workspace, signal) => {
    const name = `pi-fixer-${randomUUID()}`;
    const args = ["run", "--rm", "--name", name, "--label", "slack-pi-agent-session=true", "--network", "none", "--read-only",
      "--user", `${process.getuid?.()}:${process.getgid?.()}`, "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
      ...RESOURCE_LIMIT_FLAGS, "--tmpfs", "/tmp:rw,nosuid,size=268435456", "--env", "HOME=/tmp",
      "--mount", `type=bind,src=${workspace},dst=/workspace`, "--workdir", path.posix.join("/workspace", fixer.cwd), image, ...fixer.argv];
    const result = await runContainer({ name, args, timeoutMs: fixer.timeoutMs, signal });
    return { exit: result.exit, timedOut: result.timedOut };
  };
}
