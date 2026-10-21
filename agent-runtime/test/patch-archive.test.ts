import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm, stat, writeFile, access } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import { promisify } from "node:util";

import { applyArchivedPatch, archivePatch, PatchApplyError, readArchivedPatch, undeliveredPatchPath } from "../src/patch-archive.js";

const exec = promisify(execFile);

async function repo(t: { after(fn: () => Promise<void>): void }): Promise<{ root: string; worktree: string; baseline: string }> {
  const root = await mkdtemp(path.join(tmpdir(), "patch-archive-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const worktree = path.join(root, "repo");
  await exec("git", ["init", "-q", "-b", "main", worktree]);
  await writeFile(path.join(worktree, "a.txt"), "one\n");
  await exec("git", ["add", "."], { cwd: worktree });
  await exec("git", ["-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q", "-m", "base"], { cwd: worktree });
  const baseline = (await exec("git", ["rev-parse", "HEAD"], { cwd: worktree })).stdout.trim();
  return { root, worktree, baseline };
}

test("archives tracked, untracked and binary changes into a read-only patch", async t => {
  const { root, worktree, baseline } = await repo(t);
  await writeFile(path.join(worktree, "a.txt"), "two\n");
  await writeFile(path.join(worktree, "new.txt"), "brand new\n");
  await writeFile(path.join(worktree, "blob.bin"), Buffer.from([0, 1, 2, 255, 0, 7]));
  const file = undeliveredPatchPath(root, "run/1");
  const archived = await archivePatch({ worktree, baseline, file });
  assert.ok(archived);
  assert.equal(archived.truncated, false);
  assert.match(path.relative(root, file), /^artifacts[\\/]run_1[\\/]undelivered-.*\.patch$/);
  const patch = await readFile(file, "utf8");
  assert.match(patch, /\+two/);
  assert.match(patch, /new\.txt/);
  assert.match(patch, /GIT binary patch/);
  assert.equal((await stat(file)).mode & 0o777, 0o400);
  // The archived patch reapplies cleanly on the baseline.
  await exec("git", ["reset", "-q", "--hard", baseline], { cwd: worktree });
  await exec("git", ["clean", "-qfd"], { cwd: worktree });
  await exec("git", ["apply", "--binary", file], { cwd: worktree });
  assert.equal(await readFile(path.join(worktree, "new.txt"), "utf8"), "brand new\n");
});

test("an unchanged worktree writes no patch", async t => {
  const { root, worktree, baseline } = await repo(t);
  const file = undeliveredPatchPath(root, "clean");
  assert.equal(await archivePatch({ worktree, baseline, file }), undefined);
  await assert.rejects(access(file));
});

test("a planted secret is redacted from the archived patch", async t => {
  const { root, worktree, baseline } = await repo(t);
  await writeFile(path.join(worktree, ".env.local"), "OPENAI_API_KEY=sk-planted-0123456789abcdef\n");
  await writeFile(path.join(worktree, "config.txt"), "token in prose: very-secret-runtime-token\n");
  const file = path.join(root, "out.patch");
  await archivePatch({ worktree, baseline, file, secrets: ["very-secret-runtime-token"] });
  const patch = await readFile(file, "utf8");
  assert.doesNotMatch(patch, /sk-planted-0123456789abcdef/);
  assert.doesNotMatch(patch, /very-secret-runtime-token/);
  assert.match(patch, /\[REDACTED\]/);
});

test("a patch over the cap is truncated and marked", async t => {
  const { root, worktree, baseline } = await repo(t);
  await writeFile(path.join(worktree, "big.txt"), "x".repeat(200_000) + "\n");
  const file = path.join(root, "big.patch");
  const archived = await archivePatch({ worktree, baseline, file, maxBytes: 64 * 1024 });
  assert.ok(archived?.truncated);
  const patch = await readFile(file, "utf8");
  assert.ok(patch.length < 64 * 1024 + 200);
  assert.match(patch, /patch truncated at 65536 bytes/);
});

test("rejects a baseline that is not a commit id", async t => {
  const { root, worktree } = await repo(t);
  await assert.rejects(archivePatch({ worktree, baseline: "--output=/tmp/x", file: path.join(root, "x.patch") }), /Invalid baseline/);
});

test("reading an archived patch redacts it again and reports a missing file as undefined", async t => {
  const { root } = await repo(t);
  const file = path.join(root, "kept.patch");
  await writeFile(file, "+OPENAI_API_KEY=sk-planted-0123456789abcdef\n+runtime-token-xyz\n");
  const read = await readArchivedPatch(file, { secrets: ["runtime-token-xyz"] });
  assert.ok(read);
  assert.equal(read.truncated, false);
  const text = read.data.toString("utf8");
  assert.doesNotMatch(text, /sk-planted-0123456789abcdef|runtime-token-xyz/);
  assert.match(text, /\[REDACTED\]/);
  assert.equal(await readArchivedPatch(path.join(root, "gone.patch")), undefined);
});

test("a patch over the read cap is cut there and marked as truncated", async t => {
  const { root } = await repo(t);
  const file = path.join(root, "big.patch");
  await writeFile(file, "+" + "y".repeat(10_000) + "\n");
  const read = await readArchivedPatch(file, { maxBytes: 1024 });
  assert.ok(read?.truncated);
  assert.equal(read.bytes, 10_002);
  const text = read.data.toString("utf8");
  assert.ok(text.startsWith("+" + "y".repeat(1023)));
  assert.match(text, /\n# \[pi-agent\] patch truncated at 1024 of 10002 bytes for download\n$/);
});

test("an archived patch reapplies onto a clean worktree as uncommitted changes", async t => {
  const { root, worktree, baseline } = await repo(t);
  await writeFile(path.join(worktree, "a.txt"), "two\n");
  await writeFile(path.join(worktree, "new.txt"), "brand new\n");
  const file = path.join(root, "kept.patch");
  assert.ok(await archivePatch({ worktree, baseline, file }));
  await exec("git", ["reset", "-q", "--hard", baseline], { cwd: worktree });
  await exec("git", ["clean", "-qfd"], { cwd: worktree });
  await applyArchivedPatch(worktree, file);
  assert.equal(await readFile(path.join(worktree, "a.txt"), "utf8"), "two\n");
  assert.equal(await readFile(path.join(worktree, "new.txt"), "utf8"), "brand new\n");
  assert.equal((await exec("git", ["rev-parse", "HEAD"], { cwd: worktree })).stdout.trim(), baseline);
});

test("applying refuses a dirty worktree, a conflicting patch and a truncated archive without changing files", async t => {
  const { root, worktree, baseline } = await repo(t);
  await writeFile(path.join(worktree, "a.txt"), "two\n");
  const file = path.join(root, "kept.patch");
  assert.ok(await archivePatch({ worktree, baseline, file }));
  // Still dirty with the patch's own change.
  await assert.rejects(applyArchivedPatch(worktree, file), (error: unknown) => error instanceof PatchApplyError && error.code === "worktree_not_clean");
  await exec("git", ["reset", "-q", "--hard", baseline], { cwd: worktree });
  await writeFile(path.join(worktree, "a.txt"), "moved on\n");
  await exec("git", ["-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qam", "moved"], { cwd: worktree });
  await assert.rejects(applyArchivedPatch(worktree, file), (error: unknown) => error instanceof PatchApplyError && error.code === "patch_conflict");
  assert.equal(await readFile(path.join(worktree, "a.txt"), "utf8"), "moved on\n");
  await writeFile(path.join(worktree, "big.txt"), "z".repeat(200_000) + "\n");
  const truncated = path.join(root, "truncated.patch");
  const archived = await archivePatch({ worktree, baseline: (await exec("git", ["rev-parse", "HEAD"], { cwd: worktree })).stdout.trim(), file: truncated, maxBytes: 4096 });
  assert.ok(archived?.truncated);
  await exec("git", ["reset", "-q", "--hard"], { cwd: worktree });
  await exec("git", ["clean", "-qfd"], { cwd: worktree });
  await assert.rejects(applyArchivedPatch(worktree, truncated), (error: unknown) => error instanceof PatchApplyError && error.code === "patch_truncated");
  await assert.rejects(access(path.join(worktree, "big.txt")));
});
