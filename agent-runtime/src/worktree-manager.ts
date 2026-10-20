import { execFile } from "node:child_process";
import { mkdir, realpath, rm, stat } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

const exec = promisify(execFile);

export class WorktreeError extends Error {}
export class DiffLimitError extends WorktreeError {}

export interface WorktreeInfo {
  path: string;
  branch: string;
  baselineCommit: string;
  /** The session's own bare mirror: the only Git metadata its container may mount. */
  gitDir: string;
}

export class ProtectedFileError extends DiffLimitError {}

export interface DiffLimits {
  maxFiles: number;
  maxBytes: number;
  maxSingleFileBytes: number;
}

export const DEFAULT_DIFF_LIMITS: DiffLimits = { maxFiles: 50, maxBytes: 1024 * 1024, maxSingleFileBytes: 256 * 1024 };

export interface DiffSummary {
  files: string[];
  bytes: number;
  stat: string;
}

const MIRROR_DIR = "mirrors";
const BASELINE_KEY = "pi-agent.baseline";

export class WorktreeManager {
  private readonly repoPath: string;
  private readonly worktreeRoot: string;
  private readonly baseRef: string;

  constructor(options: { repoPath: string; worktreeRoot: string; baseRef: string }) {
    this.repoPath = path.resolve(options.repoPath);
    this.worktreeRoot = path.resolve(options.worktreeRoot);
    this.baseRef = options.baseRef;
  }

  /**
   * Gives the session its own bare mirror holding only the baseline's history, so the writer can
   * never read other branches, other sessions' commits or the host repository's config. The mirror
   * is a full copy of that history (no alternates, which would expose every object), so each
   * session costs the baseline's packed size on disk.
   */
  async create(runId: string, trustedBase?: string): Promise<WorktreeInfo> {
    await mkdir(path.join(this.worktreeRoot, MIRROR_DIR), { recursive: true });
    const safeId = runId.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 64) || "run";
    const date = new Date().toISOString().slice(0, 10).replaceAll("-", "");
    const branch = `pi-agent/${date}-${safeId}`;
    const worktreePath = path.join(this.worktreeRoot, `${date}-${safeId}`);
    const mirror = this.mirrorFor(worktreePath);
    const baselineCommit = await this.git(this.repoPath, "rev-parse", "--verify", `${trustedBase ?? this.baseRef}^{commit}`);
    try {
      await this.git(this.worktreeRoot, "init", "--quiet", "--bare", mirror);
      await this.git(this.repoPath, "-c", "core.hooksPath=/dev/null", "push", "--quiet", "--no-verify", "--no-follow-tags",
        "--recurse-submodules=no", mirror, `${baselineCommit}:refs/heads/${branch}`);
      await this.git(mirror, "config", BASELINE_KEY, baselineCommit);
      await this.git(mirror, "-c", "core.hooksPath=/dev/null", "worktree", "add", "--quiet", worktreePath, branch);
    } catch (error) {
      await rm(mirror, { recursive: true, force: true }).catch(() => undefined);
      throw new WorktreeError("Unable to create isolated Git worktree", { cause: error });
    }
    return { path: await realpath(worktreePath), branch, baselineCommit, gitDir: await realpath(mirror) };
  }

  async inspectDiff(worktreePath: string, limits: DiffLimits): Promise<DiffSummary> {
    await this.assertManagedPath(worktreePath);
    await this.git(worktreePath, "add", "-N", "--", ".");
    const status = await this.gitRaw(worktreePath, "status", "--porcelain=v1", "-z", "--untracked-files=all");
    const files = this.parseStatusPaths(status);
    const ignored = (await this.gitRaw(worktreePath, "ls-files", "--others", "--ignored", "--exclude-standard", "-z"))
      .split("\0").filter(Boolean);
    const protectedFile = [...files, ...ignored].find(isProtectedChangedPath);
    if (protectedFile) {
      throw new ProtectedFileError(`Protected file modification detected: ${protectedFile}`);
    }
    if (files.length > limits.maxFiles) {
      throw new DiffLimitError(`Changed file limit exceeded (${files.length}/${limits.maxFiles})`);
    }
    for (const file of files) {
      const fullPath = path.resolve(worktreePath, file);
      try {
        const metadata = await stat(fullPath);
        if (metadata.isFile() && metadata.size > limits.maxSingleFileBytes) {
          throw new DiffLimitError(`Single file limit exceeded: ${file}`);
        }
      } catch (error) {
        if (error instanceof DiffLimitError) throw error;
      }
    }
    const patch = await this.gitRaw(worktreePath, "diff", "--binary", "HEAD", "--");
    const bytes = Buffer.byteLength(patch);
    if (bytes > limits.maxBytes) {
      throw new DiffLimitError(`Total diff limit exceeded (${bytes}/${limits.maxBytes})`);
    }
    const statOutput = await this.gitRaw(worktreePath, "diff", "--stat", "HEAD", "--");
    return { files, bytes, stat: statOutput.trim() };
  }

  /** The full binary patch of the worktree against HEAD, untracked files included. */
  async patch(worktreePath: string): Promise<string> {
    await this.assertManagedPath(worktreePath);
    await this.git(worktreePath, "add", "-N", "--", ".");
    return this.gitRaw(worktreePath, "diff", "--binary", "HEAD", "--");
  }

  /** Commits the worktree and, for a mirrored session, publishes its branch to the main repository. */
  async commit(worktreePath: string, message: string): Promise<string> {
    await this.assertManagedPath(worktreePath);
    await this.git(worktreePath, "add", "--all", "--", ".");
    const pending = await this.gitRaw(worktreePath, "status", "--porcelain=v1");
    if (pending.trim()) {
      await this.git(
        worktreePath,
        "-c", "core.hooksPath=/dev/null",
        "-c", "commit.gpgSign=false",
        "commit", "--no-verify", "--no-gpg-sign", "-m", message,
      );
    }
    const mirror = await this.existingMirror(worktreePath);
    if (mirror) await this.publish(mirror, await this.sessionBranch(worktreePath));
    return this.git(worktreePath, "rev-parse", "HEAD");
  }

  async rollback(worktreePath: string, baselineCommit: string): Promise<void> {
    await this.assertManagedPath(worktreePath);
    await this.git(worktreePath, "reset", "--hard", baselineCommit);
    await this.git(worktreePath, "clean", "-fdx");
  }

  /**
   * Removes the worktree and its mirror. Commits past the baseline are published first, so they
   * survive in the main repository; a branch there that never moved past the baseline is deleted.
   * A legacy worktree on the shared repository metadata keeps its branch, as before mirrors.
   */
  async remove(worktreePath: string): Promise<void> {
    await this.assertManagedPath(worktreePath);
    const mirror = await this.existingMirror(worktreePath);
    if (!mirror) {
      await this.git(this.repoPath, "worktree", "remove", "--force", path.resolve(worktreePath));
      return;
    }
    const branch = await this.sessionBranch(worktreePath);
    const baseline = await this.git(mirror, "config", "--get", BASELINE_KEY);
    if (!await this.isAncestor(mirror, `refs/heads/${branch}`, baseline)) await this.publish(mirror, branch);
    await this.git(mirror, "worktree", "remove", "--force", path.resolve(worktreePath));
    await rm(mirror, { recursive: true, force: true });
    const published = await this.git(this.repoPath, "rev-parse", "--verify", "--quiet", `refs/heads/${branch}`).catch(() => "");
    if (published && await this.isAncestor(this.repoPath, published, baseline)) {
      await this.git(this.repoPath, "branch", "-D", "--", branch);
    }
  }

  /** The bare mirror backing a managed worktree; derived from the host-side layout, never from agent-writable files. */
  private mirrorFor(worktreePath: string): string {
    return path.join(this.worktreeRoot, MIRROR_DIR, `${path.basename(path.resolve(worktreePath))}.git`);
  }

  private async existingMirror(worktreePath: string): Promise<string | undefined> {
    const mirror = this.mirrorFor(worktreePath);
    try {
      if (!(await stat(mirror)).isDirectory()) return undefined;
    } catch {
      return undefined;
    }
    const common = path.resolve(worktreePath, await this.git(worktreePath, "rev-parse", "--git-common-dir"));
    if (await realpath(common) !== await realpath(mirror)) throw new WorktreeError("Worktree is not backed by its session mirror");
    return mirror;
  }

  private async sessionBranch(worktreePath: string): Promise<string> {
    const branch = await this.git(worktreePath, "symbolic-ref", "--short", "HEAD");
    if (!branch.startsWith("pi-agent/")) throw new WorktreeError("Worktree is not on its session branch");
    return branch;
  }

  /** Fast-forwards the main repository's copy of the session branch from the mirror; never forces. */
  private async publish(mirror: string, branch: string): Promise<void> {
    await this.git(this.repoPath, "-c", "core.hooksPath=/dev/null", "fetch", "--quiet", "--no-tags", "--no-write-fetch-head",
      "--recurse-submodules=no", mirror, `refs/heads/${branch}:refs/heads/${branch}`);
  }

  private async isAncestor(cwd: string, commit: string, of: string): Promise<boolean> {
    try {
      await this.git(cwd, "merge-base", "--is-ancestor", commit, of);
      return true;
    } catch {
      return false;
    }
  }

  private async assertManagedPath(candidate: string): Promise<void> {
    const resolved = await realpath(candidate);
    const root = await realpath(this.worktreeRoot);
    if (resolved === root || !resolved.startsWith(`${root}${path.sep}`)) {
      throw new WorktreeError("Refusing to operate outside the managed worktree root");
    }
  }

  private parseStatusPaths(status: string): string[] {
    const entries = status.split("\0").filter(Boolean);
    const paths: string[] = [];
    for (let index = 0; index < entries.length; index += 1) {
      const entry = entries[index];
      const statusCode = entry.slice(0, 2);
      const file = entry.slice(3);
      if (file) paths.push(file);
      if (statusCode.includes("R") || statusCode.includes("C")) index += 1;
    }
    return [...new Set(paths)];
  }

  private async git(cwd: string, ...args: string[]): Promise<string> {
    return (await this.gitRaw(cwd, ...args)).trim();
  }

  private async gitRaw(cwd: string, ...args: string[]): Promise<string> {
    const result = await exec("git", args, {
      cwd,
      timeout: 30_000,
      maxBuffer: 4 * 1024 * 1024,
      encoding: "utf8",
      env: hardenedGitEnv(),
    });
    return result.stdout;
  }
}

/** Container mount for a session's Git metadata: its mirror, or the shared metadata a legacy session's worktree points at. */
export function writerGitDir(session: { gitDir?: string }, sharedGitDir: string): string {
  return session.gitDir ?? sharedGitDir;
}

/**
 * Git never reads the host's global or system config here: agent-controlled `.gitattributes` could
 * otherwise invoke a filter, diff driver or textconv the host user configured. No credential prompts.
 */
export function hardenedGitEnv(): NodeJS.ProcessEnv {
  return {
    PATH: process.env.PATH,
    GIT_CONFIG_GLOBAL: "/dev/null",
    GIT_CONFIG_SYSTEM: "/dev/null",
    GIT_CONFIG_NOSYSTEM: "1",
    GIT_TERMINAL_PROMPT: "0",
    GIT_AUTHOR_NAME: "Pi Coding Agent",
    GIT_AUTHOR_EMAIL: "pi-agent@localhost",
    GIT_COMMITTER_NAME: "Pi Coding Agent",
    GIT_COMMITTER_EMAIL: "pi-agent@localhost",
  };
}

function isProtectedChangedPath(file: string): boolean {
  const normalized = file.split(path.sep).join("/");
  const basename = path.posix.basename(normalized);
  return normalized === ".git" || normalized.startsWith(".git/") ||
    /^\.env(?:\..+)?$/i.test(basename) ||
    /^(?:id_rsa|id_ed25519|credentials(?:\.json)?|service-account\.json|\.npmrc|\.pypirc|\.netrc)$/i.test(basename) ||
    /\.(?:pem|key|p12|pfx)$/i.test(basename);
}
