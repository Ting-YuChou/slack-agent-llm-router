import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { lstatSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, symlinkSync, writeFileSync } from "node:fs";
import { rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { after, test } from "node:test";

import { applyFixers, dockerFixerRunner, type RunFixer } from "../src/fixers.js";
import { validateProfile } from "../src/verification.js";

const roots: string[] = [];
after(async () => { for (const root of roots) await rm(root, { recursive: true, force: true }); });

/** Baseline has messy a.py and b.py; the candidate changes only a.py (still messy). */
function fixture() {
  const root = mkdtempSync(path.join(os.tmpdir(), "fixers-"));
  roots.push(root);
  const repo = path.join(root, "repo");
  mkdirSync(path.join(repo, "src"), { recursive: true });
  const git = (...args: string[]) => execFileSync("git", args, { cwd: repo, encoding: "utf8" }).trim();
  git("init", "-q"); git("config", "user.email", "t@example.com"); git("config", "user.name", "t");
  writeFileSync(path.join(repo, "src", "a.py"), "x=1\n");
  writeFileSync(path.join(repo, "src", "b.py"), "y=2\n");
  git("add", "."); git("commit", "-qm", "base");
  const baseline = git("rev-parse", "HEAD");
  writeFileSync(path.join(repo, "src", "a.py"), "x=1\nz=3\n");
  return { root, repo, baseline, scratch: path.join(root, "scratch") };
}

/** Formats every .py file under the workspace and drops in a new file, like an over-eager formatter. */
const formatAll: RunFixer = async (_fixer, workspace) => {
  const dir = path.join(workspace, "src");
  for (const name of readdirSync(dir)) {
    if (name.endsWith(".py")) writeFileSync(path.join(dir, name), readFileSync(path.join(dir, name), "utf8").replace(/=/g, " = "));
  }
  writeFileSync(path.join(dir, "new.py"), "created = True\n");
  return { exit: 1, timedOut: false };
};
const fixer = { name: "black", argv: ["black", "src"], cwd: ".", timeoutMs: 30_000 };
const signal = new AbortController().signal;

test("fixes land only on files the candidate changed; new files and untouched files are dropped", async () => {
  const { repo, baseline, scratch } = fixture();
  const outcome = await applyFixers({ fixers: [fixer], worktree: repo, baseline, scratch, run: formatAll, signal });
  assert.deepEqual(outcome, { fixes: [{ name: "black", files: ["src/a.py"] }] });
  assert.equal(readFileSync(path.join(repo, "src", "a.py"), "utf8"), "x = 1\nz = 3\n");
  assert.equal(readFileSync(path.join(repo, "src", "b.py"), "utf8"), "y=2\n");
  assert.throws(() => lstatSync(path.join(repo, "src", "new.py")));
  assert.deepEqual(readdirSync(scratch), [], "the scratch workspace is removed");
});

test("a fixer that crashes or times out changes nothing and names its cause", async () => {
  for (const [result, cause] of [[{ exit: null, timedOut: false }, "fixer_failed"], [{ exit: null, timedOut: true }, "fixer_timeout"]] as const) {
    const { repo, baseline, scratch } = fixture();
    const partial: RunFixer = async (_fixer, workspace) => { writeFileSync(path.join(workspace, "src", "a.py"), "half"); return result; };
    const outcome = await applyFixers({ fixers: [fixer, { ...fixer, name: "ruff" }], worktree: repo, baseline, scratch, run: partial, signal });
    assert.deepEqual(outcome, { fixes: [], cause });
    assert.equal(readFileSync(path.join(repo, "src", "a.py"), "utf8"), "x=1\nz=3\n");
  }
});

test("a thrown runner error (e.g. Docker unavailable) is a failed fixer, not a failed run", async () => {
  const { repo, baseline, scratch } = fixture();
  const outcome = await applyFixers({ fixers: [fixer], worktree: repo, baseline, scratch, run: async () => { throw new Error("docker gone"); }, signal });
  assert.deepEqual(outcome, { fixes: [], cause: "fixer_failed" });
});

test("nothing runs without fixers or without changes", async () => {
  const { repo, baseline, scratch } = fixture();
  let calls = 0;
  const counting: RunFixer = async () => { calls += 1; return { exit: 0, timedOut: false }; };
  assert.deepEqual(await applyFixers({ fixers: [], worktree: repo, baseline, scratch, run: counting, signal }), { fixes: [] });
  execFileSync("git", ["checkout", "--", "."], { cwd: repo });
  assert.deepEqual(await applyFixers({ fixers: [fixer], worktree: repo, baseline, scratch, run: counting, signal }), { fixes: [] });
  assert.equal(calls, 0);
});

test("a changed symlink is never followed or rewritten", async () => {
  const { root, repo, baseline, scratch } = fixture();
  const outside = path.join(root, "outside.py");
  writeFileSync(outside, "secret=1\n");
  symlinkSync(outside, path.join(repo, "src", "link.py"));
  const outcome = await applyFixers({ fixers: [fixer], worktree: repo, baseline, scratch, run: formatAll, signal });
  assert.deepEqual(outcome.fixes, [{ name: "black", files: ["src/a.py"] }]);
  assert.equal(readFileSync(outside, "utf8"), "secret=1\n");
});

test("the Docker runner mounts only the scratch workspace, with no network", async () => {
  const calls: string[][] = [];
  const run = dockerFixerRunner("sha256:" + "a".repeat(64), async ({ args }) => { calls.push(args); return { output: "", exit: 0, timedOut: false }; });
  assert.deepEqual(await run({ ...fixer, cwd: "src" }, "/tmp/ws", signal), { exit: 0, timedOut: false });
  const args = calls[0]!;
  for (const flag of ["--network", "none", "--read-only", "--cap-drop", "ALL", "--workdir", "/workspace/src"]) assert.ok(args.includes(flag), flag);
  assert.ok(args.includes("type=bind,src=/tmp/ws,dst=/workspace"));
  assert.deepEqual(args.slice(-3), ["sha256:" + "a".repeat(64), "black", "src"]);
});

test("profiles validate their fixers", () => {
  const base = { id: "p", repository: "o/r", image: "sha256:" + "0".repeat(64), rubric: ["r"], allowedPaths: ["src/"], checks: [{ name: "t", argv: ["true"], cwd: ".", timeoutMs: 1000 }] };
  assert.deepEqual(validateProfile({ ...base, fixers: [fixer] }).fixers, [fixer]);
  assert.equal(validateProfile(base).fixers, undefined);
  for (const fixers of [
    [fixer, fixer],
    [{ ...fixer, argv: [] }],
    [{ ...fixer, cwd: "../x" }],
    [{ ...fixer, argv: ["black", "{report}"] }],
    [{ ...fixer, timeoutMs: 0 }],
    Array.from({ length: 6 }, (_, i) => ({ ...fixer, name: `f${i}` })),
    "black",
  ]) assert.throws(() => validateProfile({ ...base, fixers }), /fixer/i, JSON.stringify(fixers));
});
