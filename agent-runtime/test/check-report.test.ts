import assert from "node:assert/strict";
import { test } from "node:test";

import { excerpt, formatCheckFeedback, parseDiagnostics } from "../src/check-report.js";
import type { VerificationResult } from "../src/quality-gate.js";

const diagnostics: Array<{ tool: string; output: string; cwd?: string; expected: Array<Record<string, unknown>> }> = [
  { tool: "tsc", cwd: "agent-runtime",
    output: "src/a.ts(12,5): error TS2322: Type 'string' is not assignable to type 'number'.\nFound 1 error.\n",
    expected: [{ file: "agent-runtime/src/a.ts", line: 12, column: 5, code: "TS2322", message: "Type 'string' is not assignable to type 'number'." }] },
  { tool: "tsc pretty", cwd: "agent-runtime",
    output: "\u001b[96msrc/b.ts\u001b[0m:\u001b[93m3\u001b[0m:\u001b[93m7\u001b[0m - \u001b[91merror\u001b[0m\u001b[90m TS2304: \u001b[0mCannot find name 'x'.\n",
    expected: [{ file: "agent-runtime/src/b.ts", line: 3, column: 7, code: "TS2304", message: "Cannot find name 'x'." }] },
  { tool: "flake8",
    output: "/workspace/check/src/a.py:12:80: E501 line too long (100 > 79 characters)\nslack/b.py:1:1: F401 'os' imported but unused\n",
    expected: [
      { file: "src/a.py", line: 12, column: 80, code: "E501", message: "line too long (100 > 79 characters)" },
      { file: "slack/b.py", line: 1, column: 1, code: "F401", message: "'os' imported but unused" },
    ] },
  { tool: "mypy",
    output: "src/a.py:7: error: Incompatible return value type (got \"str\", expected \"int\")  [return-value]\nsrc/a.py:7: note: See docs\nFound 1 error in 1 file (checked 3 source files)\n",
    expected: [{ file: "src/a.py", line: 7, code: "return-value", message: "Incompatible return value type (got \"str\", expected \"int\")" }] },
  { tool: "black",
    output: "would reformat /workspace/check/src/a.py\nerror: cannot format /workspace/check/tests/b.py: Cannot parse: 3:4: def f(:\nOh no! 1 file would be reformatted.\n",
    expected: [
      { file: "src/a.py", message: "would reformat" },
      { file: "tests/b.py", line: 3, column: 4, message: "Cannot parse: 3:4: def f(:" },
    ] },
  { tool: "compileall",
    output: "Listing 'src'...\n*** Error compiling '/workspace/check/src/broken.py'...\n  File \"/workspace/check/src/broken.py\", line 4\n    def f(:\n          ^\nSyntaxError: invalid syntax\n",
    expected: [{ file: "src/broken.py", line: 4, code: "SyntaxError", message: "invalid syntax" }] },
];

for (const { tool, output, cwd, expected } of diagnostics) {
  test(`parseDiagnostics reads ${tool} output`, () => {
    assert.deepEqual(parseDiagnostics(output, cwd).map(failure => ({ ...failure })), expected);
  });
}

test("parseDiagnostics never reports paths outside the workspace and is bounded", () => {
  assert.deepEqual(parseDiagnostics("/etc/passwd:1:1: E000 nope\n../x.py:1:1: E000 nope\n"), []);
  const many = Array.from({ length: 500 }, (_, i) => `src/a.py:${i + 1}:1: E501 long`).join("\n");
  assert.equal(parseDiagnostics(many).length, 50);
});

test("excerpt keeps the tail on line boundaries without ANSI or secrets", () => {
  const lines = Array.from({ length: 2000 }, (_, i) => `line ${i} \u001b[31mred\u001b[0m`);
  const text = excerpt(`${lines.join("\n")}\nAuthorization: Bearer abc.def.ghi\n`);
  assert.ok(Buffer.byteLength(text) <= 8 * 1024);
  assert.ok(!text.includes("\u001b"));
  assert.ok(text.startsWith("line "), "cut on a line boundary");
  assert.match(text, /line 1999 red/);
  assert.ok(!text.includes("abc.def.ghi"));
  assert.equal(excerpt("short\n"), "short");
});

test("formatCheckFeedback lists failures without host artifact paths and reports omissions", () => {
  const verification: VerificationResult = { status: "failed", checks: [
    { name: "baseline:pytest", status: "passed", exit_code: 0, artifact: "/var/lib/pi/artifacts/r/1/baseline-pytest.log" },
    { name: "candidate:flake8", status: "failed", exit_code: 1, artifact: "/var/lib/pi/artifacts/r/1/candidate-flake8.log",
      failures: [{ file: "src/a.py", line: 3, column: 1, code: "F401", message: "'os' imported but unused" }], excerpt: "src/a.py:3:1: F401 'os' imported but unused" },
    { name: "candidate:pytest", status: "unknown", exit_code: null, timed_out: true, cause: "check_timeout" },
  ] };
  const text = formatCheckFeedback(verification);
  assert.match(text, /candidate:flake8/);
  assert.match(text, /src\/a\.py:3:1 F401 'os' imported but unused/);
  assert.match(text, /timed out/);
  assert.ok(!text.includes("/var/lib/pi"));
  const huge: VerificationResult = { status: "failed", checks: Array.from({ length: 40 }, (_, i) => ({
    name: `candidate:check${i}`, status: "failed" as const, exit_code: 1,
    failures: Array.from({ length: 50 }, (_, j) => ({ file: `src/f${j}.py`, line: j + 1, message: "x".repeat(200) })), excerpt: "y".repeat(8000) })) };
  const bounded = formatCheckFeedback(huge, { maxBytes: 16 * 1024 });
  assert.ok(Buffer.byteLength(bounded) <= 16 * 1024);
  assert.match(bounded, /\d+ more omitted/);
});
