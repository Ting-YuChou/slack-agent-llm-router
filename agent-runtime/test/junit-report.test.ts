import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

import { parseJUnit, ReportError } from "../src/check-report.js";
import { checkArgv, readReport, validateProfile } from "../src/verification.js";

const fixture = (name: string) => readFile(new URL(`../../test/fixtures/junit/${name}`, import.meta.url));
const base = { id: "p", repository: "o/r", image: `sha256:${"0".repeat(64)}`, rubric: ["ok"], allowedPaths: ["src"] };

test("parseJUnit reads pytest xunit1 failures at the failing line", async () => {
  const failures = parseJUnit(await fixture("pytest-xunit1.xml"));
  assert.equal(failures.length, 2);
  assert.deepEqual({ ...failures[0], message: undefined }, { file: "tests/test_a.py", line: 2, code: "test_failure", message: undefined });
  assert.match(failures[0].message, /tests\.test_a::test_bad: assert 1 == 2/);
  assert.equal(failures[1].line, 8);
  assert.match(failures[1].message, /ValueError: boom/);
});

test("parseJUnit reads Node junit failures from the stack frame, relative to the repository", async () => {
  const failures = parseJUnit(await fixture("node-junit.xml"), "agent-runtime");
  assert.equal(failures.length, 1);
  assert.equal(failures[0].file, "agent-runtime/dist/test/a.test.js");
  assert.equal(failures[0].line, 2);
  assert.equal(failures[0].column, 29);
  assert.match(failures[0].message, /adds: Expected values to be strictly equal/);
});

test("parseJUnit rejects oversized reports and document type declarations", () => {
  assert.throws(() => parseJUnit(Buffer.alloc(2 * 1024 * 1024 + 1, 32)), ReportError);
  const bomb = `<?xml version="1.0"?><!DOCTYPE t [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]><testsuites><testcase name="x"><failure message="&b;"/></testcase></testsuites>`;
  assert.throws(() => parseJUnit(bomb), ReportError);
  assert.deepEqual(parseJUnit(`<testsuites><testcase name="x" file="/etc/passwd"><failure message="&unknown;"/></testcase></testsuites>`), [{ code: "test_failure", message: "x: &unknown;" }]);
});

test("profile report format and the {report} argv token", () => {
  const check = { name: "py test", argv: ["pytest", "--junitxml={report}"], cwd: ".", timeoutMs: 1000, report: { format: "junit" as const } };
  assert.deepEqual(checkArgv(validateProfile({ ...base, checks: [check] }).checks[0]), ["pytest", "--junitxml=/out/py_test.report"]);
  assert.throws(() => validateProfile({ ...base, checks: [{ ...check, report: { format: "tap" } }] }), /report/);
  assert.throws(() => validateProfile({ ...base, checks: [{ ...check, report: undefined }] }), /report/);
});

test("the example profile asks pytest and node for junit reports", () => {
  const example = JSON.parse(readFileSync(fileURLToPath(new URL("../../../config/verification/slack-router.example.json", import.meta.url)), "utf8"));
  const profile = validateProfile(example);
  for (const name of ["pytest", "node-tests"]) {
    const check = profile.checks.find(item => item.name === name)!;
    assert.equal(check.report?.format, "junit", name);
    assert.ok(checkArgv(check).some(arg => arg.includes(`/out/${name.replace(/[^\w-]/g, "_")}.report`)), name);
  }
});

test("readReport ignores symlinks and oversized files", async () => {
  const out = await mkdtemp(path.join(tmpdir(), "report-"));
  try {
    await writeFile(path.join(out, "secret"), "host file");
    await symlink(path.join(out, "secret"), path.join(out, "link.report"));
    assert.equal(await readReport(out, "link"), undefined);
    await writeFile(path.join(out, "big.report"), Buffer.alloc(2 * 1024 * 1024 + 1));
    assert.equal(await readReport(out, "big"), undefined);
    await writeFile(path.join(out, "ok.report"), "<testsuites/>");
    assert.equal((await readReport(out, "ok"))?.toString(), "<testsuites/>");
    assert.equal(await readReport(out, "missing"), undefined);
  } finally { await rm(out, { recursive: true, force: true }); }
});
