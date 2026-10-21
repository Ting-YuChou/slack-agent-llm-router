import { execFile, spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

import { redactBytes } from "./telemetry.js";
import { hardenedGitEnv } from "./worktree-manager.js";

const exec = promisify(execFile);

/** Undelivered patches larger than this are cut off and marked as truncated. */
export const PATCH_ARCHIVE_MAX_BYTES = 16 * 1024 * 1024;

/** Patches read back for the owner (download) are cut off here and marked as truncated. */
export const PATCH_DOWNLOAD_MAX_BYTES = 2 * 1024 * 1024;

const TRUNCATION_MARKER = "# [pi-agent] patch truncated at ";

export interface ArchivedPatch { path: string; bytes: number; truncated: boolean }

/** Why a retained patch could not be applied to a session's worktree; nothing was changed. */
export class PatchApplyError extends Error {
  constructor(readonly code: "worktree_not_clean" | "patch_conflict" | "patch_truncated", message: string) { super(message); this.name = "PatchApplyError"; }
}

/** Where production keeps the patch of a run that was rolled back instead of delivered. */
export function undeliveredPatchPath(root: string, runId: string, now = new Date()): string {
  return path.join(root, "artifacts", runId.replace(/[^\w-]/g, "_"), `undelivered-${now.toISOString().replace(/[:.]/g, "-")}.patch`);
}

/**
 * Writes the worktree's full change against `baseline` (untracked and binary files included) to `file`,
 * redacted and read-only. Returns undefined, writing nothing, when there is no change to keep.
 */
export async function archivePatch(options: { worktree: string; baseline: string; file: string; maxBytes?: number; secrets?: string[] }): Promise<ArchivedPatch | undefined> {
  if (!/^[a-f0-9]{40,64}$/.test(options.baseline)) throw new Error("Invalid baseline");
  const maxBytes = options.maxBytes ?? PATCH_ARCHIVE_MAX_BYTES;
  await exec("git", ["add", "-N", "--", "."], { cwd: options.worktree, env: hardenedGitEnv() });
  const { data, truncated } = await readDiff(options.worktree, options.baseline, maxBytes);
  if (!data.length) return undefined;
  const marker = truncated ? Buffer.from(`\n${TRUNCATION_MARKER}${maxBytes} bytes; the rest of the diff was not archived\n`) : Buffer.alloc(0);
  const content = Buffer.concat([redactBytes(data, options.secrets ?? []), marker]);
  await mkdir(path.dirname(options.file), { recursive: true, mode: 0o700 });
  await writeFile(options.file, content, { mode: 0o400, flag: "wx" });
  return { path: options.file, bytes: content.length, truncated };
}

/**
 * Reads an archived patch back, redacted again (archives are redacted when written; this also covers
 * secrets the host learned since) and cut at `maxBytes` with a marker. Undefined when the file is gone.
 */
export async function readArchivedPatch(file: string, options: { maxBytes?: number; secrets?: string[] } = {}): Promise<{ data: Buffer; bytes: number; truncated: boolean } | undefined> {
  const maxBytes = options.maxBytes ?? PATCH_DOWNLOAD_MAX_BYTES;
  let raw: Buffer;
  try { raw = await readFile(file); }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined; throw error; }
  const redacted = redactBytes(raw, options.secrets ?? []);
  if (redacted.length <= maxBytes) return { data: redacted, bytes: redacted.length, truncated: false };
  const marker = Buffer.from(`\n${TRUNCATION_MARKER}${maxBytes} of ${redacted.length} bytes for download\n`);
  return { data: Buffer.concat([redacted.subarray(0, maxBytes), marker]), bytes: redacted.length, truncated: true };
}

/**
 * Applies an archived patch to a clean worktree as uncommitted changes, all or nothing.
 * Refuses a dirty worktree, a patch that does not apply cleanly, and an archive cut off by the size cap.
 */
export async function applyArchivedPatch(worktree: string, file: string): Promise<void> {
  if ((await readFile(file)).includes(`\n${TRUNCATION_MARKER}`)) throw new PatchApplyError("patch_truncated", "The retained patch was truncated when archived and cannot be applied");
  const status = (await exec("git", ["status", "--porcelain=v1", "--untracked-files=all"], { cwd: worktree, env: hardenedGitEnv(), encoding: "utf8" })).stdout;
  if (status.trim()) throw new PatchApplyError("worktree_not_clean", "The session worktree has uncommitted changes");
  try { await exec("git", ["apply", "--check", "--binary", "--", file], { cwd: worktree, env: hardenedGitEnv() }); }
  catch { throw new PatchApplyError("patch_conflict", "The retained patch does not apply cleanly to the session worktree"); }
  await exec("git", ["apply", "--binary", "--", file], { cwd: worktree, env: hardenedGitEnv() });
}

/** Streams `git diff --binary` and stops reading once `maxBytes` have arrived. */
function readDiff(worktree: string, baseline: string, maxBytes: number): Promise<{ data: Buffer; truncated: boolean }> {
  return new Promise((resolve, reject) => {
    const child = spawn("git", ["diff", "--binary", baseline, "--"], { cwd: worktree, env: hardenedGitEnv(), stdio: ["ignore", "pipe", "pipe"] });
    const chunks: Buffer[] = [];
    let size = 0;
    let truncated = false;
    let stderr = "";
    child.stdout.on("data", (chunk: Buffer) => {
      if (truncated) return;
      const room = maxBytes - size;
      if (chunk.length > room) {
        chunks.push(chunk.subarray(0, room)); size = maxBytes; truncated = true;
        child.kill();
        return;
      }
      chunks.push(chunk); size += chunk.length;
    });
    child.stderr.on("data", (chunk: Buffer) => { if (stderr.length < 4096) stderr += chunk.toString(); });
    child.on("error", reject);
    child.on("close", code => {
      if (!truncated && code !== 0) { reject(new Error(`git diff failed (${code}): ${stderr.trim().slice(0, 200)}`)); return; }
      resolve({ data: Buffer.concat(chunks, size), truncated });
    });
  });
}
