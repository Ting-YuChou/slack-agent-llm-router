import { execFile, spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

import { redactBytes } from "./telemetry.js";
import { hardenedGitEnv } from "./worktree-manager.js";

const exec = promisify(execFile);

/** Undelivered patches larger than this are cut off and marked as truncated. */
export const PATCH_ARCHIVE_MAX_BYTES = 16 * 1024 * 1024;

export interface ArchivedPatch { path: string; bytes: number; truncated: boolean }

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
  const marker = truncated ? Buffer.from(`\n# [pi-agent] patch truncated at ${maxBytes} bytes; the rest of the diff was not archived\n`) : Buffer.alloc(0);
  const content = Buffer.concat([redactBytes(data, options.secrets ?? []), marker]);
  await mkdir(path.dirname(options.file), { recursive: true, mode: 0o700 });
  await writeFile(options.file, content, { mode: 0o400, flag: "wx" });
  return { path: options.file, bytes: content.length, truncated };
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
