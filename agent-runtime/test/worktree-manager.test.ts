import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { test } from "node:test";

import { DiffLimitError, WorktreeManager } from "../src/worktree-manager.js";

const exec = promisify(execFile);

async function git(cwd: string, ...args: string[]): Promise<string> {
  return (await exec("git", args, { cwd })).stdout.trim();
}

async function fixture() {
  const repo = await mkdtemp(path.join(tmpdir(), "pi-source-"));
  const worktrees = await mkdtemp(path.join(tmpdir(), "pi-worktrees-"));
  await git(repo, "init", "-b", "main");
  await git(repo, "config", "user.name", "Pi Test");
  await git(repo, "config", "user.email", "pi@example.invalid");
  await writeFile(path.join(repo, "tracked.txt"), "base\n");
  await git(repo, "add", "tracked.txt");
  await git(repo, "commit", "-m", "base");
  return { repo, worktrees };
}

test("creates an isolated branch/worktree without touching a dirty checkout", async () => {
  const { repo, worktrees } = await fixture();
  await writeFile(path.join(repo, "tracked.txt"), "dirty current checkout\n");
  const manager = new WorktreeManager({ repoPath: repo, worktreeRoot: worktrees, baseRef: "HEAD" });

  const created = await manager.create("run:unsafe/id");

  assert.match(created.branch, /^pi-agent\/\d{8}-run-unsafe-id$/);
  assert.equal(await readFile(path.join(created.path, "tracked.txt"), "utf8"), "base\n");
  assert.equal(await readFile(path.join(repo, "tracked.txt"), "utf8"), "dirty current checkout\n");
});

test("validates diff limits and commits with hooks and signing disabled", async () => {
  const { repo, worktrees } = await fixture();
  const manager = new WorktreeManager({ repoPath: repo, worktreeRoot: worktrees, baseRef: "HEAD" });
  const created = await manager.create("run-1");
  await writeFile(path.join(created.path, "tracked.txt"), "changed\n");
  await writeFile(path.join(created.path, "new.txt"), "new\n");

  const summary = await manager.inspectDiff(created.path, {
    maxFiles: 50,
    maxBytes: 1024 * 1024,
    maxSingleFileBytes: 256 * 1024,
  });
  const commit = await manager.commit(created.path, "Pi agent: run-1");

  assert.deepEqual(summary.files.sort(), ["new.txt", "tracked.txt"]);
  assert.match(summary.stat, /2 files changed/);
  assert.match(commit, /^[0-9a-f]{40}$/);
  assert.equal(await git(created.path, "status", "--porcelain"), "");
  assert.equal(await git(repo, "rev-parse", "HEAD"), created.baselineCommit);
});

test("rejects oversized diffs and rollback restores the prompt baseline", async () => {
  const { repo, worktrees } = await fixture();
  const manager = new WorktreeManager({ repoPath: repo, worktreeRoot: worktrees, baseRef: "HEAD" });
  const created = await manager.create("run-2");
  await writeFile(path.join(created.path, "tracked.txt"), "first\n");
  const firstCommit = await manager.commit(created.path, "first");
  await writeFile(path.join(created.path, "tracked.txt"), "x".repeat(20));
  await writeFile(path.join(created.path, "untracked.txt"), "temporary");

  await assert.rejects(
    manager.inspectDiff(created.path, { maxFiles: 50, maxBytes: 10, maxSingleFileBytes: 10 }),
    DiffLimitError,
  );
  await manager.rollback(created.path, firstCommit);

  assert.equal(await readFile(path.join(created.path, "tracked.txt"), "utf8"), "first\n");
  assert.equal(await git(created.path, "status", "--porcelain"), "");
});

test("ignored protected files can never pass final diff validation", async () => {
  const { repo, worktrees } = await fixture();
  await writeFile(path.join(repo, ".gitignore"), ".env*\n");
  await git(repo, "add", ".gitignore");
  await git(repo, "commit", "-m", "ignore env");
  const manager = new WorktreeManager({ repoPath: repo, worktreeRoot: worktrees, baseRef: "HEAD" });
  const created = await manager.create("run-env");
  await writeFile(path.join(created.path, ".env.local"), "SECRET=value\n");

  await assert.rejects(
    manager.inspectDiff(created.path, { maxFiles: 50, maxBytes: 1024, maxSingleFileBytes: 1024 }),
    /Protected file modification/,
  );
  await manager.rollback(created.path, created.baselineCommit);
  await assert.rejects(readFile(path.join(created.path, ".env.local"), "utf8"));
});

test("fork worktrees start from the selected checkpoint commit instead of current HEAD", async () => {
  const { repo, worktrees } = await fixture();
  const checkpoint = await git(repo, "rev-parse", "HEAD");
  await writeFile(path.join(repo, "tracked.txt"), "newer parent\n");
  await git(repo, "add", "tracked.txt");
  await git(repo, "commit", "-m", "newer parent");
  const manager = new WorktreeManager({ repoPath: repo, worktreeRoot: worktrees, baseRef: "HEAD" });

  const child = await manager.create("fork-child", checkpoint);

  assert.equal(child.baselineCommit, checkpoint);
  assert.equal(await readFile(path.join(child.path, "tracked.txt"), "utf8"), "base\n");
});

test("worktree removal is idempotent for retryable cleanup tombstones", async () => {
  const { repo, worktrees } = await fixture();
  const manager = new WorktreeManager({ repoPath: repo, worktreeRoot: worktrees, baseRef: "HEAD" });
  const created = await manager.create("cleanup-retry");

  await manager.remove(created.path);
  await manager.remove(created.path);

  assert.doesNotMatch(await git(repo, "worktree", "list", "--porcelain"), /cleanup-retry/);
});
