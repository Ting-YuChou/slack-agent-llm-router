import { execFile } from "node:child_process";
import { lstat, readFile } from "node:fs/promises";
import path from "node:path";

import { DEFAULT_DIFF_LIMITS, type DiffLimits } from "./worktree-manager.js";

export { DEFAULT_DIFF_LIMITS };

/** No single write or edit may carry more than this, whatever the profile allows. */
export const WRITE_CAP_BYTES = 256 * 1024;
export const DIFF_SUMMARY_TOOL = "diff_summary";
const MAX_LISTED_FILES = 200;
const GIT_MAX_BUFFER = 64 * 1024 * 1024;

/** Bytes each changed path adds to the patch the host checks against the diff limits. */
export type DiffUsage = Map<string, number>;

export type BudgetVerdict = { allowed: true } | { allowed: false; reason: "diff_file_limit" | "diff_byte_limit"; message: string };

/** The per-write cap: the profile's single-file limit when it is stricter, never anything looser than 256 KiB. */
export function writeCap(limits: DiffLimits): number {
  const single = limits.maxSingleFileBytes;
  return typeof single === "number" && Number.isFinite(single) && single > 0 ? Math.min(WRITE_CAP_BYTES, single) : WRITE_CAP_BYTES;
}

export function usedBytes(usage: DiffUsage): number {
  let total = 0;
  for (const bytes of usage.values()) total += bytes;
  return total;
}

/**
 * Checks a write of `bytes` to `target` against the diff limits. The target's own entry is replaced,
 * so rewriting a changed path never counts it twice.
 */
export function checkWrite(usage: DiffUsage, limits: DiffLimits, target: string, bytes: number): BudgetVerdict {
  if (!usage.has(target) && usage.size >= limits.maxFiles) {
    return { allowed: false, reason: "diff_file_limit", message: `diff_file_limit: ${target} would be a new changed file, and ${usage.size} of ${limits.maxFiles} files have changed already; change fewer files or revert one first` };
  }
  const total = usedBytes(usage) - (usage.get(target) ?? 0) + bytes;
  if (total > limits.maxBytes) {
    return { allowed: false, reason: "diff_byte_limit", message: `diff_byte_limit: ${target} would take the diff to about ${total} of ${limits.maxBytes} bytes; make a smaller change` };
  }
  return { allowed: true };
}

function git(cwd: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    // The git metadata is mounted read-only, so git must not try to refresh the index.
    execFile("git", args, { cwd, env: { ...process.env, GIT_OPTIONAL_LOCKS: "0" }, maxBuffer: GIT_MAX_BUFFER, timeout: 15_000, encoding: "buffer" },
      (error, stdout) => error ? reject(error) : resolve(stdout.toString("latin1")));
  });
}

const splitZ = (output: string) => output.split("\0").filter(Boolean).map((name) => Buffer.from(name, "latin1").toString("utf8"));

async function untrackedFiles(cwd: string): Promise<string[]> {
  return splitZ(await git(cwd, ["ls-files", "--others", "--exclude-standard", "-z"]));
}

async function fileSize(cwd: string, file: string): Promise<number> {
  try { return (await lstat(path.join(cwd, file))).size; } catch { return 0; }
}

/**
 * What the worktree's diff against the baseline costs per path, whoever made it (write, edit or bash).
 * Tracked paths cost their share of `git diff --binary HEAD`, the patch the host checks; untracked
 * paths cost their size, since the read-only index cannot take `git add -N`. Undefined when git cannot say.
 */
export async function measureDiff(cwd: string): Promise<DiffUsage | undefined> {
  try {
    const [names, patch, untracked] = await Promise.all([
      git(cwd, ["diff", "--name-only", "-z", "--no-renames", "HEAD", "--"]).then(splitZ),
      git(cwd, ["diff", "--binary", "--no-renames", "HEAD", "--"]),
      untrackedFiles(cwd),
    ]);
    // A patch line starting with "diff --git " is always a file header: content lines carry a prefix.
    const chunks = patch ? patch.split(/\n(?=diff --git )/) : [];
    if (chunks.length !== names.length) return undefined;
    const usage: DiffUsage = new Map(names.map((name, index) => [name, chunks[index]!.length + (index < chunks.length - 1 ? 1 : 0)]));
    for (const file of untracked) usage.set(file, await fileSize(cwd, file));
    return usage;
  } catch {
    return undefined;
  }
}

async function untrackedNumstat(cwd: string, file: string): Promise<string> {
  try {
    const content = await readFile(path.join(cwd, file));
    if (content.includes(0)) return `-\t-\t${file}`;
    const lines = content.length ? content.toString("utf8").split("\n").length - (content.at(-1) === 0x0a ? 1 : 0) : 0;
    return `${lines}\t0\t${file}`;
  } catch {
    return `-\t-\t${file}`;
  }
}

export interface DiffSummaryResult {
  text: string;
  details: { files: number; bytes: number; remainingFiles: number; remainingBytes: number; writeCapBytes: number; limits: DiffLimits };
}

/** Numstat of the worktree against the baseline, bash changes included, and what is left of the diff limits. */
export async function diffSummary(cwd: string, limits: DiffLimits): Promise<DiffSummaryResult | undefined> {
  const usage = await measureDiff(cwd);
  if (!usage) return undefined;
  let numstat: string[];
  try {
    numstat = (await git(cwd, ["diff", "--numstat", "-z", "--no-renames", "HEAD", "--"])).split("\0").filter(Boolean)
      .map((line) => Buffer.from(line, "latin1").toString("utf8"));
    for (const file of await untrackedFiles(cwd)) numstat.push(await untrackedNumstat(cwd, file));
  } catch {
    return undefined;
  }
  const bytes = usedBytes(usage);
  const details = {
    files: usage.size, bytes, remainingFiles: Math.max(0, limits.maxFiles - usage.size), remainingBytes: Math.max(0, limits.maxBytes - bytes),
    writeCapBytes: writeCap(limits), limits,
  };
  const listed = numstat.slice(0, MAX_LISTED_FILES);
  const lines = [
    "Changes against the baseline (added, deleted, path; untracked files included):",
    ...(listed.length ? listed : ["(no changes)"]),
    ...(numstat.length > listed.length ? [`… and ${numstat.length - listed.length} more paths`] : []),
    `Diff budget: ${details.files} of ${limits.maxFiles} files and about ${bytes} of ${limits.maxBytes} bytes used; ${details.remainingFiles} more files and ${details.remainingBytes} more bytes fit. A single write may carry at most ${details.writeCapBytes} bytes.`,
  ];
  return { text: lines.join("\n"), details };
}
