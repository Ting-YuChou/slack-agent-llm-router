import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";

import { loadProfile } from "../src/verification.js";

const PROFILE = {
  id: "demo", repository: "acme/widgets", image: `sha256:${"a".repeat(64)}`, rubric: ["works"], allowedPaths: ["src/"],
  checks: [{ name: "unit", argv: ["npm", "test"], cwd: ".", timeoutMs: 1000 }],
};

async function layout(t: { after(fn: () => Promise<void>): void }) {
  const root = await mkdtemp(path.join(os.tmpdir(), "profile-location-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const repo = path.join(root, "repo"); const worktrees = path.join(root, "worktrees"); const trusted = path.join(root, "trusted");
  await Promise.all([mkdir(repo), mkdir(worktrees), mkdir(trusted)]);
  return { root, repo, worktrees, trusted };
}

test("loadProfile accepts a profile outside the repository and worktrees and checksums its bytes", async t => {
  const { repo, worktrees, trusted } = await layout(t);
  const file = path.join(trusted, "profile.json");
  await writeFile(file, JSON.stringify(PROFILE));
  const loaded = await loadProfile(file, repo, worktrees);
  assert.equal(loaded.profile.id, "demo");
  assert.match(loaded.checksum, /^[a-f0-9]{64}$/);
});

for (const [where, place] of [
  ["at the repository root", (l: Awaited<ReturnType<typeof layout>>) => path.join(l.repo, "profile.json")],
  ["nested in the repository", (l: Awaited<ReturnType<typeof layout>>) => path.join(l.repo, "config", "verification", "profile.json")],
  ["in a repository directory whose name starts with ..", (l: Awaited<ReturnType<typeof layout>>) => path.join(l.repo, "..trusted", "profile.json")],
  ["named ..profile.json in the repository", (l: Awaited<ReturnType<typeof layout>>) => path.join(l.repo, "..profile.json")],
  ["inside a worktree", (l: Awaited<ReturnType<typeof layout>>) => path.join(l.worktrees, "run-1", "profile.json")],
] as const) {
  test(`loadProfile rejects a profile ${where}`, async t => {
    const l = await layout(t);
    const file = place(l);
    await mkdir(path.dirname(file), { recursive: true });
    await writeFile(file, JSON.stringify(PROFILE));
    await assert.rejects(loadProfile(file, l.repo, l.worktrees), /must live outside the repository\/worktrees/);
  });
}

test("loadProfile resolves symlinks, so an outside link to a repository file is rejected", async t => {
  const { repo, worktrees, trusted } = await layout(t);
  const inside = path.join(repo, "profile.json");
  await writeFile(inside, JSON.stringify(PROFILE));
  const link = path.join(trusted, "profile.json");
  await symlink(inside, link);
  await assert.rejects(loadProfile(link, repo, worktrees), /must live outside the repository\/worktrees/);
});

test("loadProfile resolves a symlinked repository path before comparing", async t => {
  const { root, repo, worktrees } = await layout(t);
  const file = path.join(repo, "profile.json");
  await writeFile(file, JSON.stringify(PROFILE));
  const repoLink = path.join(root, "repo-link");
  await symlink(repo, repoLink);
  await assert.rejects(loadProfile(file, repoLink, worktrees), /must live outside the repository\/worktrees/);
});
