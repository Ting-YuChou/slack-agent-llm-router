import assert from "node:assert/strict";
import { test } from "node:test";

import { validateGrade } from "../src/quality-gate.js";

const HASH = "candidate";
const RUBRIC = ["correct", "scoped"];

function grade(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    status: "passed",
    candidate_hash: HASH,
    criteria: [{ id: "correct", passed: true, evidence: "tests cover the fix" }, { id: "scoped", passed: true, evidence: "only src/ changed" }],
    findings: [],
    ...overrides,
  };
}
const finding = (overrides: Record<string, unknown> = {}) => ({ path: "src/a.ts", line: 3, severity: "warning", reason: "nit", ...overrides });
const failing = [{ id: "correct", passed: false }, { id: "scoped", passed: true }];

// Every rejection must keep a message graderCause classifies as grader_invalid_submission.
const REJECTED = /^(Invalid grader|Missing grader|Inconsistent grader)/;

test("validateGrade accepts well-formed verdicts", () => {
  assert.equal(validateGrade(grade(), HASH, RUBRIC).status, "passed");
  assert.equal(validateGrade(grade({ findings: [finding()] }), HASH, RUBRIC).status, "passed");
  assert.equal(validateGrade(grade({ status: "needs_changes", criteria: failing, findings: [finding({ severity: "error" })] }), HASH, RUBRIC).status, "needs_changes");
  assert.equal(validateGrade(grade({ status: "unknown", criteria: failing }), HASH, RUBRIC).status, "unknown");
  assert.equal(validateGrade(grade({ status: "needs_changes", criteria: failing, findings: Array.from({ length: 50 }, () => finding()) }), HASH, RUBRIC).findings.length, 50);
  assert.ok(validateGrade(grade({ findings: [finding({ reason: "r".repeat(4000) })] }), HASH, RUBRIC));
});

const cases: Array<[string, unknown, RegExp]> = [
  ["status outside the enum", grade({ status: "failed" }), /^Invalid grader result/],
  ["missing status", grade({ status: undefined }), /^Invalid grader result/],
  ["criteria not an array", grade({ criteria: { correct: true } }), /^Invalid grader result/],
  ["findings not an array", grade({ findings: "none" }), /^Invalid grader result/],
  ["more than 50 findings", grade({ status: "needs_changes", criteria: failing, findings: Array.from({ length: 51 }, () => finding()) }), /^Invalid grader result/],
  ["duplicate criterion id", grade({ criteria: [{ id: "correct", passed: true, evidence: "e" }, { id: "correct", passed: true, evidence: "e" }] }), /^Missing grader criteria/],
  ["unknown criterion id", grade({ criteria: [{ id: "correct", passed: true, evidence: "e" }, { id: "fast", passed: true, evidence: "e" }] }), /^Missing grader criteria/],
  ["extra unknown criterion", grade({ criteria: [{ id: "correct", passed: true, evidence: "e" }, { id: "scoped", passed: true, evidence: "e" }, { id: "fast", passed: true, evidence: "e" }] }), /^Invalid grader result/],
  ["non-boolean passed", grade({ criteria: [{ id: "correct", passed: "true", evidence: "e" }, { id: "scoped", passed: true, evidence: "e" }] }), /^Missing grader criteria/],
  ["null criterion entry", grade({ criteria: [null, { id: "scoped", passed: true, evidence: "e" }] }), /^Invalid grader result/],
  ["non-object criterion entry", grade({ criteria: ["correct", { id: "scoped", passed: true, evidence: "e" }] }), /^Invalid grader result/],
  ["absolute finding path", grade({ findings: [finding({ path: "/etc/passwd" })] }), /^Invalid grader finding/],
  ["parent-relative finding path", grade({ findings: [finding({ path: "src/../../secret" })] }), /^Invalid grader finding/],
  ["backslash parent finding path", grade({ findings: [finding({ path: "src\\..\\..\\secret" })] }), /^Invalid grader finding/],
  ["non-string finding path", grade({ findings: [finding({ path: 7 })] }), /^Invalid grader finding/],
  ["zero finding line", grade({ findings: [finding({ line: 0 })] }), /^Invalid grader finding/],
  ["fractional finding line", grade({ findings: [finding({ line: 1.5 })] }), /^Invalid grader finding/],
  ["string finding line", grade({ findings: [finding({ line: "3" })] }), /^Invalid grader finding/],
  ["severity outside the enum", grade({ findings: [finding({ severity: "info" })] }), /^Invalid grader finding/],
  ["null finding entry", grade({ findings: [null] }), /^Invalid grader finding/],
  ["reason over 4000 chars", grade({ findings: [finding({ reason: "r".repeat(4001) })] }), /^Invalid grader finding/],
  ["evidence over 4000 chars", grade({ criteria: [{ id: "correct", passed: true, evidence: "e".repeat(4001) }, { id: "scoped", passed: true, evidence: "e" }] }), /^Invalid grader evidence/],
  ["evidence over 4000 chars on a failed criterion", grade({ status: "needs_changes", criteria: [{ id: "correct", passed: false, evidence: "e".repeat(4001) }, { id: "scoped", passed: true }] }), /^Invalid grader evidence/],
  ["non-string evidence", grade({ criteria: [{ id: "correct", passed: true, evidence: 1 }, { id: "scoped", passed: true, evidence: "e" }] }), /^Invalid grader evidence/],
  ["passed with an error-severity finding", grade({ findings: [finding({ severity: "error" })] }), /^Inconsistent grader verdict/],
  ["passed with blank evidence", grade({ criteria: [{ id: "correct", passed: true, evidence: "  " }, { id: "scoped", passed: true, evidence: "e" }] }), /^Inconsistent grader verdict/],
];

for (const [name, value, message] of cases) {
  test(`validateGrade rejects ${name}`, () => {
    assert.throws(() => validateGrade(value, HASH, RUBRIC), (error: unknown) => {
      assert.ok(error instanceof Error);
      assert.match(error.message, REJECTED);
      assert.match(error.message, message);
      return true;
    });
  });
}
