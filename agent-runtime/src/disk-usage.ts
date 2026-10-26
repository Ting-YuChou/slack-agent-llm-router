import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

/** How much a directory a run writes (its worktree or Pi session) may hold, how often it is measured, and how. */
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
  return quotaFromEnv(env, "PI_AGENT_WORKTREE_QUOTA_BYTES", 2 * 1024 ** 3);
}

/** PI_AGENT_SESSION_QUOTA_BYTES (default 256 MiB) for a session's Pi directory, checked every 15 s. */
export function sessionQuotaFromEnv(env: NodeJS.ProcessEnv): DiskQuota {
  return quotaFromEnv(env, "PI_AGENT_SESSION_QUOTA_BYTES", 256 * 1024 ** 2);
}

function quotaFromEnv(env: NodeJS.ProcessEnv, name: string, defaultBytes: number): DiskQuota {
  const raw = env[name]?.trim();
  const bytes = raw ? Number(raw) : defaultBytes;
  if (!Number.isSafeInteger(bytes) || bytes <= 0) throw new Error(`${name} must be a positive integer`);
  return { bytes, intervalMs: 15_000, usage: diskUsageBytes };
}
