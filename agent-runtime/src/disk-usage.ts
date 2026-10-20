import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

/** How much a run's worktree may hold, how often it is measured, and how. */
export interface DiskQuota { bytes: number; intervalMs: number; usage(directory: string): Promise<number> }

/** Allocated bytes under `directory` (`du -sk`, which every Linux and macOS host has). */
export async function diskUsageBytes(directory: string): Promise<number> {
  const { stdout } = await execFileAsync("du", ["-sk", directory], { timeout: 30_000, maxBuffer: 64 * 1024 });
  const kib = Number.parseInt(stdout.trim().split(/\s+/)[0] ?? "", 10);
  if (!Number.isFinite(kib) || kib < 0) throw new Error("du returned no size");
  return kib * 1024;
}

/** PI_AGENT_WORKTREE_QUOTA_BYTES (default 2 GiB), checked every 15 s. */
export function diskQuotaFromEnv(env: NodeJS.ProcessEnv): DiskQuota {
  const raw = env.PI_AGENT_WORKTREE_QUOTA_BYTES?.trim();
  const bytes = raw ? Number(raw) : 2 * 1024 ** 3;
  if (!Number.isSafeInteger(bytes) || bytes <= 0) throw new Error("PI_AGENT_WORKTREE_QUOTA_BYTES must be a positive integer");
  return { bytes, intervalMs: 15_000, usage: diskUsageBytes };
}
