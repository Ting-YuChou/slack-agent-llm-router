import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { readdir, lstat, rm } from "node:fs/promises";
import path from "node:path";
const exec = promisify(execFile);

export async function pruneEvidence(root: string, mounts: string[], before: number) {
  const active = mounts.map(mount => path.resolve(mount));
  for (const category of ["artifacts", "graders", "reproductions", "sessions", "investigations"]) {
    const directory = path.join(root, category);
    const entries = await readdir(directory, { withFileTypes: true }).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return []; throw error; });
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const candidate = path.resolve(directory, entry.name);
      if (active.some(mount => candidate === mount || mount.startsWith(candidate + path.sep) || candidate.startsWith(mount + path.sep))) continue;
      // One unreadable entry keeps only itself; the rest of the pass still runs.
      try { await pruneEntry(candidate, before); }
      catch { console.error(JSON.stringify({ event: "agent_evidence_prune_failed", category })); }
    }
  }
}

/** Removes an expired evidence directory unless something inside it changed recently. lstat: links are never followed. */
async function pruneEntry(candidate: string, before: number): Promise<void> {
  if ((await lstat(candidate)).mtimeMs >= before) return;
  // Session JSONL files can be appended without updating their parent directory.
  let scanned = 0;
  async function recent(directory: string): Promise<boolean> {
    for (const child of await readdir(directory, { withFileTypes: true })) {
      if (++scanned > 10_000) return true;
      const location = path.join(directory, child.name);
      if ((await lstat(location)).mtimeMs >= before) return true;
      if (child.isDirectory() && await recent(location)) return true;
    }
    return false;
  }
  if (!await recent(candidate)) await rm(candidate, { recursive: true, force: true });
}
export function startEvidenceRetention(root: string, pruneControl?: () => void) {
  let pending: Promise<void> | undefined;
  async function clean() {
    try {
      const ids = (await exec("docker", ["ps", "-aq", "--filter", "label=slack-pi-agent-session=true"], { timeout: 5000, env: { PATH: process.env.PATH } })).stdout.split(/\s+/).filter(Boolean);
      const containers = ids.length ? JSON.parse((await exec("docker", ["inspect", ...ids], { timeout: 5000, maxBuffer: 4 * 1024 * 1024, env: { PATH: process.env.PATH } })).stdout) : [];
      const mounts = containers.flatMap((container: any) => container.Mounts.map((mount: any) => mount.Source));
      await pruneEvidence(root, mounts, Date.now() - 30 * 86400000);
      pruneControl?.();
    } catch { /* Unknown Docker state or filesystem errors retain evidence. */ }
  }
  const run = () => { if (!pending) pending = clean().finally(() => { pending = undefined; }); };
  const timer = setInterval(run, 86400000); timer.unref(); run();
  return async () => { clearInterval(timer); await pending; };
}
