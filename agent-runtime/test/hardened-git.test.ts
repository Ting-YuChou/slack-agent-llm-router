import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

import { archivePatch } from "../src/patch-archive.js";
import { CandidateStore } from "../src/verification.js";
import { hardenedGitEnv } from "../src/worktree-manager.js";

/** A repo whose tree asks for a filter and diff driver that only the host's global git config defines. */
async function hostileRepo() {
  const root = await mkdtemp(path.join(tmpdir(), "hardened-git-"));
  const repo = path.join(root, "repo"); const marker = path.join(root, "pwned");
  const globalConfig = path.join(root, "global.gitconfig");
  await writeFile(globalConfig, `[filter "evil"]\n\tclean = sh -c 'touch ${marker}; cat'\n[diff "evil"]\n\ttextconv = sh -c 'touch ${marker}; cat \\"$1\\"' --\n`);
  const git = (...args: string[]) => execFileSync("git", args, { cwd: repo, env: { ...hardenedGitEnv() } });
  execFileSync("git", ["init", "-q", "-b", "main", repo]);
  await writeFile(path.join(repo, "a.txt"), "one\n");
  git("add", "."); git("commit", "-q", "-m", "baseline");
  const baseline = git("rev-parse", "HEAD").toString().trim();
  await writeFile(path.join(repo, ".gitattributes"), "* filter=evil diff=evil\n");
  await writeFile(path.join(repo, "a.txt"), "two\n");
  return { root, repo, marker, globalConfig, baseline };
}

async function withGlobalConfig<T>(file: string, work: () => Promise<T>): Promise<T> {
  const previous = process.env.GIT_CONFIG_GLOBAL;
  process.env.GIT_CONFIG_GLOBAL = file;
  try { return await work(); } finally { if (previous === undefined) delete process.env.GIT_CONFIG_GLOBAL; else process.env.GIT_CONFIG_GLOBAL = previous; }
}

test("the hardened git env ignores global and system config and never prompts", () => {
  const env = hardenedGitEnv();
  assert.equal(env.GIT_CONFIG_GLOBAL, "/dev/null");
  assert.equal(env.GIT_CONFIG_SYSTEM, "/dev/null");
  assert.equal(env.GIT_TERMINAL_PROMPT, "0");
});

test("snapshotting a candidate never runs a filter or diff driver from the host's git config", async () => {
  const h = await hostileRepo();
  try {
    await withGlobalConfig(h.globalConfig, () => new CandidateStore(path.join(h.root, "candidates")).create(h.repo, h.baseline, "run-1"));
    assert.equal(existsSync(h.marker), false, "a host-configured filter ran on agent-controlled .gitattributes");
  } finally { await rm(h.root, { recursive: true, force: true }); }
});

test("archiving an undelivered patch never runs a filter or diff driver from the host's git config", async () => {
  const h = await hostileRepo();
  try {
    await withGlobalConfig(h.globalConfig, () => archivePatch({ worktree: h.repo, baseline: h.baseline, file: path.join(h.root, "out", "p.patch") }));
    assert.equal(existsSync(h.marker), false, "a host-configured filter ran on agent-controlled .gitattributes");
  } finally { await rm(h.root, { recursive: true, force: true }); }
});
