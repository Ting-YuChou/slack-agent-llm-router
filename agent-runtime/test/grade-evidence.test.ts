import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";

import { deletedPaths, verifyGradeEvidence } from "../src/grade-evidence.js";
import type { Grade } from "../src/quality-gate.js";

async function withSnapshot(run: (root: string) => Promise<void>) {
  const root = await mkdtemp(path.join(os.tmpdir(), "grade-evidence-"));
  try {
    await mkdir(path.join(root, "src"));
    await writeFile(path.join(root, "src", "a.js"), "line1\nline2\nline3\n");
    await writeFile(path.join(root, "Makefile"), "all:\n");
    await run(root);
  } finally { await rm(root, { recursive: true, force: true }); }
}

const grade = (overrides: Partial<Grade>): Grade => ({ status: "passed", candidate_hash: "h", criteria: [{ id: "correct", passed: true, evidence: "src/a.js:2 returns the fixed value" }], findings: [], ...overrides });

test("citations of existing files and lines are accepted", () => withSnapshot(async root => {
  await verifyGradeEvidence(grade({}), root);
  await verifyGradeEvidence(grade({ criteria: [{ id: "correct", passed: true, evidence: "see src/a.js lines and Makefile" }] }), root);
}));

test("a finding at a line past the end of the file is rejected", () => withSnapshot(async root => {
  await assert.rejects(verifyGradeEvidence(grade({ status: "needs_changes", findings: [{ path: "src/a.js", line: 999, severity: "error", reason: "bug" }] }), root), /src\/a\.js:999/);
}));

test("a finding in a file that does not exist is rejected", () => withSnapshot(async root => {
  await assert.rejects(verifyGradeEvidence(grade({ status: "needs_changes", findings: [{ path: "src/missing.js", line: 1, severity: "warning", reason: "x" }] }), root), /src\/missing\.js/);
}));

test("a passed criterion must cite at least one existing file", () => withSnapshot(async root => {
  await assert.rejects(verifyGradeEvidence(grade({ criteria: [{ id: "correct", passed: true, evidence: "all tests pass" }] }), root), /correct/);
  await assert.rejects(verifyGradeEvidence(grade({ criteria: [{ id: "correct", passed: true, evidence: "src/invented.js:4 is fine" }] }), root), /correct/);
}));

test("a cited line out of range rejects even when the file exists", () => withSnapshot(async root => {
  await assert.rejects(verifyGradeEvidence(grade({ criteria: [{ id: "correct", passed: true, evidence: "src/a.js:40 handles it" }] }), root), /src\/a\.js:40/);
}));

test("failed criteria need no citation", () => withSnapshot(async root => {
  await verifyGradeEvidence(grade({ status: "needs_changes", criteria: [{ id: "correct", passed: false, evidence: "not done" }] }), root);
}));

test("files the candidate deletes may be cited", () => withSnapshot(async root => {
  await verifyGradeEvidence(grade({ criteria: [{ id: "correct", passed: true, evidence: "old/legacy.js was removed as asked" }], findings: [{ path: "old/legacy.js", line: 3, severity: "warning", reason: "removed" }] }), root, ["old/legacy.js"]);
}));

test("symlinks never count as evidence", () => withSnapshot(async root => {
  await symlink("/etc/hosts", path.join(root, "src", "link.js"));
  await assert.rejects(verifyGradeEvidence(grade({ criteria: [{ id: "correct", passed: true, evidence: "src/link.js:1" }] }), root), /correct/);
}));

test("deleted paths are read from the candidate patch", () => {
  const patch = "diff --git a/old/legacy.js b/old/legacy.js\ndeleted file mode 100644\nindex 1..0\n--- a/old/legacy.js\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\ndiff --git a/src/a.js b/src/a.js\nindex 1..2 100644\n";
  assert.deepEqual(deletedPaths(patch), ["old/legacy.js"]);
  assert.deepEqual(deletedPaths(undefined), []);
});
