import assert from "node:assert/strict";
import { mkdtemp, readFile, realpath, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { test } from "node:test";

import { DiffLimitError, WorktreeManager, writerGitDir } from "../src/worktree-manager.js";

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

test("each session's worktree sees only its own branch and the baseline history", async () => {
  const { repo, worktrees } = await fixture();
  await git(repo, "checkout", "-q", "-b", "unrelated");
  await writeFile(path.join(repo, "unrelated.txt"), "not for agents\n");
  await git(repo, "add", "unrelated.txt");
  await git(repo, "commit", "-q", "-m", "unrelated");
  const unrelated = await git(repo, "rev-parse", "HEAD");
  await git(repo, "checkout", "-q", "main");
  await git(repo, "remote", "add", "origin", "https://example.invalid/secret.git");
  const manager = new WorktreeManager({ repoPath: repo, worktreeRoot: worktrees, baseRef: "main" });
  const a = await manager.create("session-a");
  const b = await manager.create("session-b");
  await writeFile(path.join(b.path, "b.txt"), "only on B\n");
  const onlyOnB = await manager.commit(b.path, "B work");

  assert.deepEqual((await git(a.path, "for-each-ref", "--format=%(refname)")).split("\n"), [`refs/heads/${a.branch}`]);
  await assert.rejects(git(a.path, "cat-file", "-e", `${onlyOnB}^{commit}`));
  await assert.rejects(git(a.path, "cat-file", "-e", `${unrelated}^{commit}`));
  await assert.rejects(git(a.gitDir, "config", "--get", "remote.origin.url"));
  assert.doesNotMatch(await git(a.path, "config", "--list", "--local"), /^remote\./m);
  assert.equal(path.resolve(a.path, await git(a.path, "rev-parse", "--git-common-dir")), a.gitDir);
  assert.ok(!a.gitDir.startsWith(path.join(await realpath(repo), ".git")), "the mirror lives outside the host repository");
});

test("delivery publishes the session's branch and commit to the main repository", async () => {
  const { repo, worktrees } = await fixture();
  const manager = new WorktreeManager({ repoPath: repo, worktreeRoot: worktrees, baseRef: "HEAD" });
  const created = await manager.create("deliver");
  await assert.rejects(git(repo, "rev-parse", "--verify", `refs/heads/${created.branch}`), "an undelivered branch stays out of the main repository");

  await writeFile(path.join(created.path, "tracked.txt"), "first delivery\n");
  const first = await manager.commit(created.path, "first");
  assert.equal(await git(repo, "rev-parse", created.branch), first);

  await writeFile(path.join(created.path, "tracked.txt"), "second delivery\n");
  const second = await manager.commit(created.path, "second");
  assert.equal(await git(repo, "rev-parse", created.branch), second);
  assert.equal(await git(repo, "rev-parse", "HEAD"), created.baselineCommit);
});

test("removal deletes the mirror and an unmoved branch but keeps delivered commits", async () => {
  const { repo, worktrees } = await fixture();
  const manager = new WorktreeManager({ repoPath: repo, worktreeRoot: worktrees, baseRef: "HEAD" });
  const idle = await manager.create("idle");
  const empty = await manager.create("empty");
  const delivered = await manager.create("delivered");
  assert.equal(await manager.commit(empty.path, "nothing changed"), empty.baselineCommit);
  await writeFile(path.join(delivered.path, "tracked.txt"), "kept\n");
  const commit = await manager.commit(delivered.path, "kept");

  for (const created of [idle, empty, delivered]) {
    await manager.remove(created.path);
    await assert.rejects(stat(created.path));
    await assert.rejects(stat(created.gitDir));
  }
  await assert.rejects(git(repo, "rev-parse", "--verify", `refs/heads/${idle.branch}`));
  await assert.rejects(git(repo, "rev-parse", "--verify", `refs/heads/${empty.branch}`));
  assert.equal(await git(repo, "rev-parse", delivered.branch), commit);
  assert.equal(await git(repo, "show", `${delivered.branch}:tracked.txt`), "kept");
  assert.equal(await git(repo, "worktree", "list", "--porcelain").then((list) => list.includes(delivered.path)), false);
});

test("a legacy session on the shared repository metadata still commits, rolls back and removes", async () => {
  const { repo, worktrees } = await fixture();
  const manager = new WorktreeManager({ repoPath: repo, worktreeRoot: worktrees, baseRef: "HEAD" });
  const baseline = await git(repo, "rev-parse", "HEAD");
  const legacyPath = path.join(worktrees, "20260101-legacy");
  await git(repo, "worktree", "add", "-q", "-b", "pi-agent/20260101-legacy", legacyPath, baseline);
  const shared = path.resolve(repo, await git(repo, "rev-parse", "--git-common-dir"));

  await writeFile(path.join(legacyPath, "tracked.txt"), "legacy change\n");
  const commit = await manager.commit(legacyPath, "legacy");
  assert.equal(await git(repo, "rev-parse", "pi-agent/20260101-legacy"), commit);
  await writeFile(path.join(legacyPath, "tracked.txt"), "discarded\n");
  await manager.rollback(legacyPath, commit);
  assert.equal(await readFile(path.join(legacyPath, "tracked.txt"), "utf8"), "legacy change\n");
  await manager.remove(legacyPath);
  await assert.rejects(stat(legacyPath));
  assert.equal(await git(repo, "rev-parse", "pi-agent/20260101-legacy"), commit);

  assert.equal(writerGitDir({}, shared), shared);
  assert.equal(writerGitDir({ gitDir: "/mirrors/s.git" }, shared), "/mirrors/s.git");
});
