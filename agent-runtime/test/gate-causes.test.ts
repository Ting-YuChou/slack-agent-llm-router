import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { chmod, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";

import { classifyGateError, HarnessError } from "../src/error-taxonomy.js";
import { CandidateStore, DockerVerifier, validateProfile } from "../src/verification.js";

const profile = validateProfile({ id: "causes", repository: "org/repo", image: `sha256:${"a".repeat(64)}`, rubric: ["correct"], allowedPaths: ["src/"],
  checks: [{ name: "unit", argv: ["node", "--test"], cwd: ".", timeoutMs: 5_000 }] });
/** A real candidate directory: the verifier writes per-mode output next to it before calling Docker. */
async function candidateIn(t: { after(fn: () => Promise<void>): void }) {
  const root = await mkdtemp(path.join(tmpdir(), "cause-candidate-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(path.join(root, "candidate"));
  return { hash: "h", path: path.join(root, "candidate") };
}

/** Runs `action` with PATH pointing only at `directory`, so `docker` resolves to a fake (or to nothing). */
async function withPath<T>(directory: string, action: () => Promise<T>): Promise<T> {
  const saved = process.env.PATH;
  process.env.PATH = directory;
  try { return await action(); } finally { process.env.PATH = saved; }
}

test("each gate cause is classified from HarnessError and from legacy messages", () => {
  for (const cause of ["snapshot_too_large", "candidate_changed", "docker_unavailable", "verification_stop_unconfirmed", "candidate_artifact_missing"] as const) {
    const failure = classifyGateError(new HarnessError(cause, cause));
    assert.equal(failure.code, "quality_gate_unavailable");
    assert.equal(failure.cause_code, cause);
  }
  assert.equal(classifyGateError(new Error("Candidate snapshot too large")).cause_code, "snapshot_too_large");
  assert.equal(classifyGateError(new Error("candidate_changed_after_verification")).cause_code, "candidate_changed");
  assert.equal(classifyGateError(new Error("candidate_artifact_missing")).cause_code, "candidate_artifact_missing");
  assert.equal(classifyGateError(new Error("verification_stop_unconfirmed")).cause_code, "verification_stop_unconfirmed");
  assert.equal(classifyGateError(Object.assign(new Error("spawn docker ENOENT"), { code: "ENOENT" })).cause_code, "docker_unavailable");
  assert.equal(classifyGateError(Object.assign(new Error("Command failed: docker run"), { code: 1, stderr: "Cannot connect to the Docker daemon at unix:///var/run/docker.sock" })).cause_code, "docker_unavailable");
});

test("a missing docker binary fails verification as docker_unavailable", async t => {
  const empty = await mkdtemp(path.join(tmpdir(), "no-docker-"));
  t.after(() => rm(empty, { recursive: true, force: true }));
  const target = await candidateIn(t);
  const error = await withPath(empty, () => new DockerVerifier(profile).verify(target, new AbortController().signal)).catch(caught => caught);
  assert.ok(error instanceof HarnessError, String(error));
  assert.equal(error.causeCode, "docker_unavailable");
});

test("an unreachable Docker daemon fails verification as docker_unavailable, not stop_unconfirmed", async t => {
  const bin = await mkdtemp(path.join(tmpdir(), "down-docker-"));
  t.after(() => rm(bin, { recursive: true, force: true }));
  await writeFile(path.join(bin, "docker"), "#!/bin/sh\necho 'Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?' >&2\nexit 1\n");
  await chmod(path.join(bin, "docker"), 0o755);
  const target = await candidateIn(t);
  const error = await withPath(bin, () => new DockerVerifier(profile).verify(target, new AbortController().signal)).catch(caught => caught);
  assert.ok(error instanceof HarnessError, String(error));
  assert.equal(error.causeCode, "docker_unavailable");
});

test("a container that cannot be removed is verification_stop_unconfirmed", async t => {
  const bin = await mkdtemp(path.join(tmpdir(), "stuck-docker-"));
  t.after(() => rm(bin, { recursive: true, force: true }));
  // `docker run` reports a failing check; `docker rm` cannot confirm the container is gone.
  await writeFile(path.join(bin, "docker"), "#!/bin/sh\nif [ \"$1\" = rm ]; then echo 'Error response from daemon: removal in progress' >&2; exit 1; fi\nexit 1\n");
  await chmod(path.join(bin, "docker"), 0o755);
  const target = await candidateIn(t);
  const error = await withPath(bin, () => new DockerVerifier(profile).verify(target, new AbortController().signal)).catch(caught => caught);
  assert.ok(error instanceof HarnessError, String(error));
  assert.equal(error.causeCode, "verification_stop_unconfirmed");
});

test("a candidate changed after verification is candidate_changed", async t => {
  const root = await mkdtemp(path.join(tmpdir(), "changed-candidate-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const repo = path.join(root, "repo");
  await mkdir(path.join(repo, "src"), { recursive: true });
  const git = (...args: string[]) => execFileSync("git", ["-C", repo, ...args]).toString().trim();
  git("init", "-q");
  await writeFile(path.join(repo, "src", "a.txt"), "one\n");
  git("add", "."); git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "base");
  const baseline = git("rev-parse", "HEAD");
  const store = new CandidateStore(path.join(root, "artifacts"));
  await writeFile(path.join(repo, "src", "a.txt"), "two\n");
  const verified = await store.create(repo, baseline, "run");
  await writeFile(path.join(repo, "src", "a.txt"), "three\n");
  const error = await store.assertUnchanged(verified, repo, baseline, "run").catch(caught => caught);
  assert.ok(error instanceof HarnessError);
  assert.equal(error.causeCode, "candidate_changed");
  assert.equal(error.message, "candidate_changed_after_verification");
});
