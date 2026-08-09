import { execFile } from "node:child_process";
import { mkdir, realpath, stat } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

const exec = promisify(execFile);

export class WorktreeError extends Error {}
export class DiffLimitError extends WorktreeError {}

export interface WorktreeInfo {
  path: string;
  branch: string;
  baselineCommit: string;
}

export interface DiffLimits {
  maxFiles: number;
  maxBytes: number;
  maxSingleFileBytes: number;
}

export interface DiffSummary {
  files: string[];
  bytes: number;
  stat: string;
}

export class WorktreeManager {
  private readonly repoPath: string;
  private readonly worktreeRoot: string;
  private readonly baseRef: string;

  constructor(options: { repoPath: string; worktreeRoot: string; baseRef: string }) {
    this.repoPath = path.resolve(options.repoPath);
    this.worktreeRoot = path.resolve(options.worktreeRoot);
    this.baseRef = options.baseRef;
  }

  async create(runId: string, startCommit?: string): Promise<WorktreeInfo> {
    await mkdir(this.worktreeRoot, { recursive: true });
    const safeId = runId.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 64) || "run";
    const date = new Date().toISOString().slice(0, 10).replaceAll("-", "");
    const branch = `pi-agent/${date}-${safeId}`;
    const worktreePath = path.join(this.worktreeRoot, `${date}-${safeId}`);
    const baselineCommit = await this.git(this.repoPath, "rev-parse", "--verify", `${startCommit ?? this.baseRef}^{commit}`);
    try {
      await this.git(this.repoPath, "worktree", "add", "-b", branch, worktreePath, baselineCommit);
    } catch (error) {
      throw new WorktreeError("Unable to create isolated Git worktree", { cause: error });
    }
    return { path: await realpath(worktreePath), branch, baselineCommit };
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
      throw new DiffLimitError(`Protected file modification detected: ${protectedFile}`);
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
    return this.git(worktreePath, "rev-parse", "HEAD");
  }

  async rollback(worktreePath: string, baselineCommit: string): Promise<void> {
    await this.assertManagedPath(worktreePath);
    await this.git(worktreePath, "reset", "--hard", baselineCommit);
    await this.git(worktreePath, "clean", "-fdx");
  }

  async remove(worktreePath: string): Promise<void> {
    const resolved = path.resolve(worktreePath);
    await this.assertManagedRemovalPath(resolved);
    try {
      await this.git(this.repoPath, "worktree", "remove", "--force", resolved);
    } catch (error) {
      try {
        await stat(resolved);
      } catch (statError) {
        if (isMissingFileError(statError)) {
          await this.git(this.repoPath, "worktree", "prune", "--expire=now");
          return;
        }
      }
      throw new WorktreeError("Unable to remove isolated Git worktree", { cause: error });
    }
  }

  private async assertManagedPath(candidate: string): Promise<void> {
    const resolved = await realpath(candidate);
    const root = await realpath(this.worktreeRoot);
    if (resolved === root || !resolved.startsWith(`${root}${path.sep}`)) {
      throw new WorktreeError("Refusing to operate outside the managed worktree root");
    }
  }

  private async assertManagedRemovalPath(candidate: string): Promise<void> {
    const root = await realpath(this.worktreeRoot);
    const resolved = path.resolve(candidate);
    if (resolved === root || !resolved.startsWith(`${root}${path.sep}`)) {
      throw new WorktreeError("Refusing to remove outside the managed worktree root");
    }
    try {
      const actual = await realpath(resolved);
      if (actual === root || !actual.startsWith(`${root}${path.sep}`)) {
        throw new WorktreeError("Refusing to remove outside the managed worktree root");
      }
    } catch (error) {
      if (!isMissingFileError(error)) throw error;
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
      maxBuffer: 4 * 1024 * 1024,
      encoding: "utf8",
      env: {
        PATH: process.env.PATH,
        GIT_CONFIG_GLOBAL: "/dev/null",
        GIT_CONFIG_SYSTEM: "/dev/null",
        GIT_TERMINAL_PROMPT: "0",
        GIT_AUTHOR_NAME: "Pi Coding Agent",
        GIT_AUTHOR_EMAIL: "pi-agent@localhost",
        GIT_COMMITTER_NAME: "Pi Coding Agent",
        GIT_COMMITTER_EMAIL: "pi-agent@localhost",
      },
    });
    return result.stdout;
  }
}

function isMissingFileError(error: unknown): boolean {
  return typeof error === "object" && error !== null &&
    "code" in error && (error as { code?: string }).code === "ENOENT";
}

function isProtectedChangedPath(file: string): boolean {
  const normalized = file.split(path.sep).join("/");
  const basename = path.posix.basename(normalized);
  return normalized === ".git" || normalized.startsWith(".git/") ||
    /^\.env(?:\..+)?$/i.test(basename) ||
    /^(?:id_rsa|id_ed25519|credentials(?:\.json)?|service-account\.json|\.npmrc|\.pypirc|\.netrc)$/i.test(basename) ||
    /\.(?:pem|key|p12|pfx)$/i.test(basename);
}
