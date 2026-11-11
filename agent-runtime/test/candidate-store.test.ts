import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { chmod, lstat, mkdir, mkdtemp, readFile, readdir, rm, symlink, unlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test, type TestContext } from "node:test";

import { HarnessError } from "../src/error-taxonomy.js";
import { readBlobs } from "../src/git-blobs.js";
import { CandidateStore } from "../src/verification.js";
import { hardenedGitEnv } from "../src/worktree-manager.js";

type Files = Record<string, { content: string | Buffer; executable?: true }>;
// Bytes a line-oriented reader would trip over: NULs, bare newlines, and text shaped like a cat-file header.
const BINARY = Buffer.concat([Buffer.from([0, 1, 2, 10, 0, 255]), Buffer.from(`\n${"a".repeat(40)} blob 12\nnot a header\n`), Buffer.from([0, 10, 10, 13])]);
const BIG = Buffer.alloc(200 * 1024, "x\n");

/** The digest a candidate hash is documented to be; resumed workflows compare it with one stored before a restart. */
function digest(baseline: string, files: Files): string {
  const hash = createHash("sha256").update(baseline);
  for (const file of Object.keys(files).sort()) hash.update(file).update("\0").update(files[file]!.executable ? String(0o111) : "0").update("\0").update(files[file]!.content);
  return hash.digest("hex");
}

async function listFiles(directory: string, prefix = ""): Promise<string[]> {
  const found: string[] = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const name = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) found.push(...await listFiles(path.join(directory, entry.name), name)); else found.push(name);
  }
  return found.sort();
}

async function repository(t: TestContext) {
  const root = await mkdtemp(path.join(tmpdir(), "candidate-store-"));
  // Snapshots are read-only on purpose; make them removable again.
  t.after(async () => { execFileSync("chmod", ["-R", "u+rwX", root]); await rm(root, { recursive: true, force: true }); });
  const repo = path.join(root, "repo");
  execFileSync("git", ["init", "-q", "-b", "main", repo]);
  const git = (...args: string[]) => execFileSync("git", args, { cwd: repo, env: hardenedGitEnv() });
  const write = async (file: string, content: string | Buffer, mode?: number) => {
    const target = path.join(repo, file); await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, content); if (mode) await chmod(target, mode);
  };
  const commit = (message: string) => { git("add", "-f", "."); git("commit", "-q", "-m", message); return git("rev-parse", "HEAD").toString().trim(); };
  return { root, repo, store: path.join(root, "artifacts"), git, write, commit };
}

/** A committed tree, and a working tree that edits, deletes and adds files against it. */
async function fixture(t: TestContext) {
  const r = await repository(t);
  await r.write("src/a.txt", "one\n"); await r.write("src/gone.txt", "bye\n"); await r.write("bin/run.sh", "#!/bin/sh\n", 0o755);
  await r.write("data/blob.bin", BINARY); await r.write("data/empty", ""); await r.write("data/big.txt", BIG);
  await r.write(".env", "SECRET=1\n"); await r.write(".gitignore", "ignored/\n");
  await symlink("src/a.txt", path.join(r.repo, "link"));
  const baseline = r.commit("baseline");
  await r.write("src/a.txt", "two\n"); await unlink(path.join(r.repo, "src/gone.txt")); await unlink(path.join(r.repo, "link"));
  await r.write("src/new.txt", "new\n"); await r.write("bin/new.sh", "#!/bin/sh\necho\n", 0o755);
  await r.write(".env", "SECRET=2\n"); await r.write("ignored/skip.txt", "ignored\n");
  // What a snapshot holds: no protected, ignored or deleted path.
  const candidate: Files = {
    ".gitignore": { content: "ignored/\n" }, "bin/new.sh": { content: "#!/bin/sh\necho\n", executable: true }, "bin/run.sh": { content: "#!/bin/sh\n", executable: true },
    "data/big.txt": { content: BIG }, "data/blob.bin": { content: BINARY }, "data/empty": { content: "" }, "src/a.txt": { content: "two\n" }, "src/new.txt": { content: "new\n" },
  };
  return { ...r, baseline, candidate };
}

const mode = async (file: string) => (await lstat(file)).mode & 0o777;

// --- behaviour the snapshot already had; these pin it across the rewrite -------------------------

test("the candidate hash is the documented digest of the working tree", async t => {
  const f = await fixture(t);
  const snapshot = await new CandidateStore(f.store).create(f.repo, f.baseline, "run");
  assert.equal(snapshot.hash, digest(f.baseline, f.candidate));
  assert.deepEqual(await listFiles(snapshot.path), Object.keys(f.candidate).sort());
  for (const [file, entry] of Object.entries(f.candidate)) {
    assert.deepEqual(await readFile(path.join(snapshot.path, file)), Buffer.from(entry.content), file);
    assert.equal(await mode(path.join(snapshot.path, file)), entry.executable ? 0o500 : 0o400, file);
  }
});

test("files under the store root are not part of the candidate", async t => {
  const f = await fixture(t);
  const store = new CandidateStore(path.join(f.repo, "artifacts"));
  const first = await store.create(f.repo, f.baseline, "run");
  const second = await store.create(f.repo, f.baseline, "run");
  assert.equal(first.hash, digest(f.baseline, f.candidate));
  assert.equal(second.hash, first.hash);
});

test("the baseline tree holds the commit's bytes and modes", async t => {
  const f = await fixture(t);
  const snapshot = await new CandidateStore(f.store).create(f.repo, f.baseline, "run");
  const baselineDir = path.join(path.dirname(snapshot.path), "baseline");
  const entries = f.git("ls-tree", "-r", "-z", f.baseline).toString().split("\0").filter(Boolean).map(entry => ({ mode: entry.split(" ")[0]!, file: entry.slice(entry.indexOf("\t") + 1) }));
  const expected = entries.filter(entry => entry.file !== ".env");
  assert.deepEqual(await listFiles(baselineDir), expected.map(entry => entry.file).sort(), "protected paths never reach the baseline tree");
  assert.ok(expected.some(entry => entry.mode === "120000"), "the fixture commits a symlink");
  for (const entry of expected) {
    const copy = path.join(baselineDir, entry.file);
    assert.ok((await lstat(copy)).isFile(), `${entry.file} is a regular file`);
    assert.deepEqual(await readFile(copy), f.git("show", `${f.baseline}:${entry.file}`), entry.file);
    assert.equal(await mode(copy), entry.mode === "100755" ? 0o500 : 0o400, entry.file);
  }
  assert.deepEqual(await readFile(path.join(baselineDir, "src/a.txt")), Buffer.from("one\n"), "the candidate's edit is not in the baseline tree");
  assert.deepEqual(await readFile(path.join(baselineDir, "link")), Buffer.from("src/a.txt"), "a committed symlink becomes a file holding its target");
});

test("a baseline gitlink is refused", async t => {
  const r = await repository(t);
  await r.write("src/a.txt", "one\n");
  r.commit("files");
  r.git("update-index", "--add", "--cacheinfo", `160000,${"1".repeat(40)},vendor/sub`);
  r.git("commit", "-q", "-m", "gitlink");
  const baseline = r.git("rev-parse", "HEAD").toString().trim();
  await assert.rejects(new CandidateStore(r.store).create(r.repo, baseline, "run"), /Baseline contains an unsupported entry/);
});

test("a baseline file over 2 MiB is refused", async t => {
  const r = await repository(t);
  await r.write("src/a.txt", "one\n"); await r.write("data/huge.bin", Buffer.alloc(2 * 1024 * 1024 + 1, 1));
  const baseline = r.commit("baseline");
  await unlink(path.join(r.repo, "data/huge.bin"));
  // The binary diff of such a file already exceeds git's output limit, so the snapshot fails before the baseline copy.
  await assert.rejects(new CandidateStore(r.store).create(r.repo, baseline, "run"));
  // The baseline reader has the same limit of its own, and reports it before buffering the blob.
  const oid = r.git("rev-parse", `${baseline}:data/huge.bin`).toString().trim();
  await assert.rejects(async () => { for await (const _blob of readBlobs(r.repo, [oid], 2 * 1024 * 1024)) { /* never yielded */ } }, /is larger than 2097152 bytes/);
  const small = r.git("rev-parse", `${baseline}:src/a.txt`).toString().trim();
  const read: string[] = [];
  for await (const blob of readBlobs(r.repo, [small, small], 2 * 1024 * 1024)) read.push(blob.data.toString());
  assert.deepEqual(read, ["one\n", "one\n"]);
  await assert.rejects(async () => { for await (const _blob of readBlobs(r.repo, ["0".repeat(40)], 1024)) { /* missing */ } }, /is not a readable blob/);
});

test("a snapshot without a baseline tree keeps an empty baseline directory for the mount", async t => {
  const f = await fixture(t);
  const snapshot = await new CandidateStore(f.store).create(f.repo, f.baseline, "run", { baselineTree: false });
  assert.equal(snapshot.hash, digest(f.baseline, f.candidate));
  assert.deepEqual(await readdir(path.join(path.dirname(snapshot.path), "baseline")), []);
});

test("a snapshot refuses a symlink and a file over 2 MiB in the working tree", async t => {
  const linked = await fixture(t);
  await symlink("src/a.txt", path.join(linked.repo, "src/link"));
  await assert.rejects(new CandidateStore(linked.store).create(linked.repo, linked.baseline, "run"), /unsupported file/);
  // Committed and untouched, so the diff stays small and the size limit is what refuses it.
  const large = await repository(t);
  await large.write("src/a.txt", "one\n"); await large.write("src/huge.bin", Buffer.alloc(2 * 1024 * 1024 + 1, 1));
  const baseline = large.commit("baseline");
  const error = await new CandidateStore(large.store).create(large.repo, baseline, "run").catch(caught => caught);
  assert.ok(error instanceof HarnessError, String(error));
  assert.equal(error.causeCode, "snapshot_too_large");
});

// --- reading the baseline in one process, and hashing without a snapshot -------------------------

test("hash() gives the snapshot's hash without touching the working tree or the store", async t => {
  const f = await fixture(t);
  const store = new CandidateStore(f.store);
  const status = () => f.git("status", "--porcelain", "--untracked-files=all").toString();
  const before = status();
  const hashed = await store.hash(f.repo, f.baseline);
  assert.equal(hashed, digest(f.baseline, f.candidate));
  assert.equal(existsSync(f.store), false, "hashing creates no store directory");
  assert.equal(status(), before, "hashing leaves the index alone");
  assert.equal((await store.create(f.repo, f.baseline, "run")).hash, hashed, "a snapshot taken afterwards agrees");
});

test("assertUnchanged compares hashes without taking another snapshot", async t => {
  const f = await fixture(t);
  const store = new CandidateStore(f.store);
  const verified = await store.create(f.repo, f.baseline, "run");
  const snapshots = () => readdir(path.join(f.store, "run"));
  const before = await snapshots();
  await store.assertUnchanged(verified, f.repo, f.baseline, "run");
  assert.deepEqual(await snapshots(), before);
  await f.write("src/new.txt", "changed after verification\n");
  const error = await store.assertUnchanged(verified, f.repo, f.baseline, "run").catch(caught => caught);
  assert.ok(error instanceof HarnessError, String(error));
  assert.equal(error.causeCode, "candidate_changed");
  assert.deepEqual(await snapshots(), before);
});

test("hash() refuses what a snapshot refuses", async t => {
  const linked = await fixture(t);
  await symlink("src/a.txt", path.join(linked.repo, "src/link"));
  await assert.rejects(new CandidateStore(linked.store).hash(linked.repo, linked.baseline), /unsupported file/);
  const large = await fixture(t);
  await large.write("src/huge.bin", Buffer.alloc(2 * 1024 * 1024 + 1, 1));
  const error = await new CandidateStore(large.store).hash(large.repo, large.baseline).catch(caught => caught);
  assert.ok(error instanceof HarnessError, String(error));
  assert.equal(error.causeCode, "snapshot_too_large");
  await assert.rejects(new CandidateStore(large.store).hash(large.repo, "HEAD"), /Invalid baseline/);
});

test("the number of git processes does not grow with the baseline tree", async t => {
  const real = execFileSync("sh", ["-c", "command -v git"]).toString().trim();
  const processes = async (files: number) => {
    const r = await repository(t);
    for (let index = 0; index < files; index++) await r.write(`src/f${index}.txt`, `${index}\n`);
    const baseline = r.commit("baseline");
    await r.write("src/f0.txt", "changed\n");
    // A git on PATH that counts its own runs; hardenedGitEnv keeps PATH.
    const shim = path.join(r.root, "shim"); const log = path.join(r.root, "git.log");
    await mkdir(shim); await writeFile(path.join(shim, "git"), `#!/bin/sh\necho run >> '${log}'\nexec '${real}' "$@"\n`, { mode: 0o755 });
    const previous = process.env.PATH; process.env.PATH = `${shim}${path.delimiter}${previous}`;
    try { await new CandidateStore(r.store).create(r.repo, baseline, "run"); } finally { process.env.PATH = previous; }
    return (await readFile(log, "utf8")).split("\n").filter(Boolean).length;
  };
  assert.equal(await processes(30), await processes(3));
});

test("hash() reads the working tree as git lists it; a snapshot taken first settles a file replaced by a directory", async t => {
  const f = await fixture(t);
  const store = new CandidateStore(f.store);
  // A tracked file becomes a directory: until `create` refreshes the index, git still lists the old path.
  await unlink(path.join(f.repo, "src/a.txt")); await f.write("src/a.txt/inner.txt", "inner\n");
  await assert.rejects(store.hash(f.repo, f.baseline), /unsupported file/);
  const snapshot = await store.create(f.repo, f.baseline, "run");
  assert.equal(await store.hash(f.repo, f.baseline), snapshot.hash);
  await store.assertUnchanged(snapshot, f.repo, f.baseline, "run");
});
