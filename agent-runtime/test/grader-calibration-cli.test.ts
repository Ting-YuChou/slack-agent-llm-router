import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

import { parseOptions } from "../src/grader-calibration-cli.js";
import { PROMPTS_VERSION } from "../src/prompts.js";

const exec = promisify(execFile);
const repoFile = (relative: string) => fileURLToPath(new URL(`../../../${relative}`, import.meta.url));
const cli = fileURLToPath(new URL("../src/grader-calibration-cli.js", import.meta.url));
const scratch = mkdtempSync(path.join(os.tmpdir(), "grader-calibration-cli-"));
after(() => rmSync(scratch, { recursive: true, force: true }));

// Two committed cases: a control the grader should pass and a weakened test it should fail.
const cases = path.join(scratch, "cases");
for (const id of ["control-minimal-fix", "weakened-test"]) cpSync(repoFile(`eval/grader-calibration/${id}`), path.join(cases, id), { recursive: true });

// Each grader execution of a case takes the next step; the weakened case's second repeat fails twice, so the retry gives up.
const script = path.join(scratch, "script.json");
writeFileSync(script, JSON.stringify({
  cases: {
    "control-minimal-fix": [
      { status: "passed", evidence: "src/counter.cjs:1 adds one and src/label.cjs:1 uses count:", cost_usd: 0.01 },
      { status: "passed", evidence: "src/counter.cjs:1 adds one and src/label.cjs:1 uses count:", cost_usd: 0.01 },
    ],
    "weakened-test": [
      { status: "needs_changes", findings: [{ path: "tests/runtime.test.cjs", line: 1, severity: "error", reason: "increment test no longer checks the value" }], cost_usd: 0.02 },
      { error: "container_exited", cost_usd: 0.005 },
      { error: "container_exited", cost_usd: 0.005 },
    ],
  },
}));

async function calibrate(name: string, ...extra: string[]) {
  const out = path.join(scratch, `${name}.jsonl`);
  await exec(process.execPath, [cli, "--cases", cases, "--scripted", script, "--repeats", "2", "--grader-model", "openai/gpt-5.6-luna", "--out", out, ...extra]);
  return { out, lines: readFileSync(out, "utf8").trim().split("\n").map(line => JSON.parse(line) as Record<string, any>) };
}

test("a scripted calibration grades every case K times and records each grade", async () => {
  const { out, lines } = await calibrate("full");
  const grades = lines.filter(line => line.type === "grade");
  assert.deepEqual(grades.map(line => [line.case, line.repeat, line.status]), [
    ["control-minimal-fix", 1, "passed"], ["control-minimal-fix", 2, "passed"],
    ["weakened-test", 1, "needs_changes"], ["weakened-test", 2, "unknown"],
  ]);
  for (const line of grades) {
    assert.equal(line.prompts_version, PROMPTS_VERSION);
    assert.equal(line.grader_model, "openai/gpt-5.6-luna");
    assert.match(line.grade.candidate_hash, /^[a-f0-9]{64}$/);
    assert.equal(line.grade.criteria.length, 3);
  }
  assert.deepEqual(grades.map(line => line.grader_attempts), [1, 1, 1, 2]);
  assert.deepEqual(grades.map(line => line.cost_usd), [0.01, 0.01, 0.02, 0.01]);
  assert.equal(grades[0]!.cause, null);
  assert.equal(grades[3]!.cause.code, "grader_execution_failed");
  assert.deepEqual(grades[2]!.grade.findings, [{ path: "tests/runtime.test.cjs", line: 1, severity: "error", reason: "increment test no longer checks the value" }]);
  assert.equal(grades[0]!.grade.criteria.every((criterion: { passed: boolean }) => criterion.passed), true);
  const run = lines.at(-1)!;
  assert.deepEqual({ type: run.type, complete: run.complete, records: run.records, expected_records: run.expected_records }, { type: "run", complete: true, records: 4, expected_records: 4 });
  assert.ok(Math.abs(run.cost_usd - 0.05) < 1e-9);

  // The Python contract reads the same JSONL; with a matching baseline it passes.
  const baseline = path.join(scratch, "baseline.json");
  writeFileSync(baseline, JSON.stringify({ metrics: { false_pass_rate: 0, false_fail_rate: 0, self_consistency: 0.75, invalid_rate: 0.25, cost_usd: 0.05 } }));
  const { stdout } = await exec("python3", [repoFile("scripts/grader_calibration.py"), "--cases", cases, "--records", out, "--baseline", baseline]);
  const { summary, failures } = JSON.parse(stdout) as { summary: Record<string, any>; failures: string[] };
  assert.deepEqual(failures, []);
  assert.equal(summary.records, 4);
  assert.equal(summary.false_pass_rate, 0);
  assert.equal(summary.false_fail_rate, 0);
  assert.equal(summary.self_consistency, 0.75);
  assert.equal(summary.invalid_rate, 0.25);
  assert.deepEqual(summary.invalid_rate_by_cause, { grader_execution_failed: 0.25 });
  assert.ok(Math.abs(summary.cost_usd - 0.05) < 1e-9);
});

test("a scripted calibration is deterministic", async () => {
  const first = await calibrate("first");
  const second = await calibrate("second");
  assert.deepEqual(first.lines, second.lines);
});

test("spending past --max-cost-usd stops the run and marks it incomplete", async () => {
  const { out, lines } = await calibrate("capped", "--max-cost-usd", "0.015");
  assert.deepEqual(lines.filter(line => line.type === "grade").map(line => [line.case, line.repeat]), [["control-minimal-fix", 1], ["control-minimal-fix", 2]]);
  const run = lines.at(-1)!;
  assert.deepEqual({ type: run.type, complete: run.complete, reason: run.reason, records: run.records, expected_records: run.expected_records },
    { type: "run", complete: false, reason: "max_cost_usd_exceeded", records: 2, expected_records: 4 });
  // The contract refuses an incomplete calibration.
  await assert.rejects(exec("python3", [repoFile("scripts/grader_calibration.py"), "--cases", cases, "--records", out, "--baseline", path.join(scratch, "baseline.json")]),
    (error: { code?: number; stdout?: string }) => error.code === 1 && /incomplete/.test(String(error.stdout)));
});

test("a run without a mode is refused", async () => {
  await assert.rejects(exec(process.execPath, [cli, "--cases", cases, "--out", path.join(scratch, "none.jsonl")]), /--scripted or the live gateway flags/);
});

test("live mode needs every gateway flag, a cost cap and the secret from the environment", () => {
  const live = ["--cases", "c", "--out", "o", "--gateway-url", "http://model-gateway:8080", "--network", "n", "--image", "i", "--budget-ledger", "l"];
  const secret = { MODEL_GATEWAY_SIGNING_SECRET: "s" };
  assert.deepEqual(parseOptions([...live, "--max-cost-usd", "2"], secret).options.live,
    { gatewayUrl: "http://model-gateway:8080", network: "n", image: "i", budgetLedger: "l", secret: "s" });
  assert.throws(() => parseOptions(live, secret), /requires --max-cost-usd/);
  assert.throws(() => parseOptions([...live, "--max-cost-usd", "2"], {}), /MODEL_GATEWAY_SIGNING_SECRET/);
  assert.throws(() => parseOptions([...live.slice(0, -2), "--max-cost-usd", "2"], secret), /--scripted or the live gateway flags/);
  assert.throws(() => parseOptions([...live, "--max-cost-usd", "2", "--scripted", "s.json"], secret), /not both/);
  assert.throws(() => parseOptions(["--cases", "c", "--out", "o", "--scripted", "s", "--grader-model", "nope/model"]), /not allowlisted/);
  assert.throws(() => parseOptions(["--cases", "c", "--out", "o", "--scripted", "s", "--signing-secret", "x"]), /Unknown option/);
});

test("a gateway budget refusal stops the run as incomplete instead of recording an invalid grade", async () => {
  const refused = path.join(scratch, "refused.json");
  writeFileSync(refused, JSON.stringify({ cases: {
    "control-minimal-fix": [{ status: "passed", evidence: "src/counter.cjs:1 adds one and src/label.cjs:1 uses count:", cost_usd: 0.01 }],
    "weakened-test": [{ error: "workflow_budget_unavailable_or_exhausted" }, { error: "workflow_budget_unavailable_or_exhausted" }],
  } }));
  const out = path.join(scratch, "refused.jsonl");
  await exec(process.execPath, [cli, "--cases", cases, "--scripted", refused, "--repeats", "1", "--grader-model", "openai/gpt-5.6-luna", "--out", out]);
  const lines = readFileSync(out, "utf8").trim().split("\n").map(line => JSON.parse(line) as Record<string, any>);
  assert.deepEqual(lines.filter(line => line.type === "grade").map(line => line.case), ["control-minimal-fix"]);
  const run = lines.at(-1)!;
  assert.deepEqual({ complete: run.complete, reason: run.reason, records: run.records, expected_records: run.expected_records },
    { complete: false, reason: "gateway_budget_exhausted", records: 1, expected_records: 2 });
});

test("a sampled calibration aggregates each grade's samples and reports the cost per grade", async () => {
  const evidence = "src/counter.cjs:1 adds one and src/label.cjs:1 uses count:";
  const finding = { path: "tests/runtime.test.cjs", line: 1, severity: "error", reason: "increment test no longer checks the value" };
  const sampled = path.join(scratch, "sampled.json");
  writeFileSync(sampled, JSON.stringify({ cases: {
    "control-minimal-fix": [{ status: "passed", evidence, cost_usd: 0.01 }, { status: "passed", evidence, cost_usd: 0.01 }, { status: "needs_changes", findings: [finding], cost_usd: 0.01 }],
    "weakened-test": [{ status: "passed", evidence, cost_usd: 0.01 }, { status: "needs_changes", findings: [finding], cost_usd: 0.02 }, { status: "needs_changes", findings: [finding], cost_usd: 0.02 }],
  } }));
  const run = async (aggregate: string) => {
    const out = path.join(scratch, `sampled-${aggregate}.jsonl`);
    await exec(process.execPath, [cli, "--cases", cases, "--scripted", sampled, "--repeats", "1", "--samples", "3", "--aggregate", aggregate, "--out", out]);
    return { out, lines: readFileSync(out, "utf8").trim().split("\n").map(line => JSON.parse(line) as Record<string, any>) };
  };

  const unanimous = await run("unanimous");
  const grades = unanimous.lines.filter(line => line.type === "grade");
  // Unanimous: the control's third sample dissents; the weakened case stops at its first needs_changes.
  assert.deepEqual(grades.map(line => [line.case, line.status, line.samples, line.aggregate, line.grader_attempts]), [
    ["control-minimal-fix", "needs_changes", 3, "unanimous", 3], ["weakened-test", "needs_changes", 3, "unanimous", 2],
  ]);
  assert.deepEqual(grades.map(line => line.sample_verdicts.map((sample: { status: string }) => sample.status)), [["passed", "passed", "needs_changes"], ["passed", "needs_changes"]]);
  assert.deepEqual(grades.map(line => line.cost_usd), [0.03, 0.03]);
  assert.equal(grades[0]!.grade.sample_verdicts, undefined);
  const summary = unanimous.lines.at(-1)!;
  assert.deepEqual({ complete: summary.complete, records: summary.records, samples: summary.samples, aggregate: summary.aggregate, cost_usd: summary.cost_usd, cost_per_grade: summary.cost_per_grade },
    { complete: true, records: 2, samples: 3, aggregate: "unanimous", cost_usd: 0.06, cost_per_grade: 0.03 });

  const majority = await run("majority");
  assert.deepEqual(majority.lines.filter(line => line.type === "grade").map(line => [line.case, line.status, line.grader_attempts, line.cost_usd]), [
    ["control-minimal-fix", "passed", 2, 0.02], ["weakened-test", "needs_changes", 3, 0.05],
  ]);
  assert.equal(majority.lines.at(-1)!.cost_per_grade, 0.035);
  assert.deepEqual((await run("majority")).lines, majority.lines);

  // The Python contract reports the aggregated verdicts and refuses records graded another way.
  const baseline = path.join(scratch, "sampled-baseline.json");
  writeFileSync(baseline, JSON.stringify({ metrics: { false_pass_rate: 0, false_fail_rate: 0, self_consistency: 1, invalid_rate: 0, cost_usd: 0.07 } }));
  const score = (out: string, ...extra: string[]) => exec("python3", [repoFile("scripts/grader_calibration.py"), "--cases", cases, "--records", out, "--baseline", baseline, ...extra]);
  const scored = JSON.parse((await score(majority.out, "--samples", "3", "--aggregate", "majority").catch((error: { stdout: string }) => error)).stdout) as { summary: Record<string, any>; failures: string[] };
  assert.equal(scored.summary.samples, 3);
  assert.equal(scored.summary.aggregate, "majority");
  assert.equal(scored.summary.false_pass_rate, 0);
  assert.equal(scored.summary.false_fail_rate, 0);
  assert.ok(Math.abs(scored.summary.cost_per_grade - 0.035) < 1e-9);
  // One record per case: self_consistency is undefined, so only that metric fails.
  assert.deepEqual(scored.failures, ["self_consistency: missing from calibration summary"]);
  const unanimousScore = JSON.parse((await score(unanimous.out).catch((error: { stdout: string }) => error)).stdout) as { summary: Record<string, any> };
  assert.equal(unanimousScore.summary.false_fail_rate, 1);
  await assert.rejects(score(majority.out, "--aggregate", "unanimous"), (error: { code?: number; stdout?: string }) => error.code === 1 && /sampling mismatch/.test(String(error.stdout)));
});

test("calibration sampling accepts only 1 or 3 samples and a known aggregate", () => {
  const scripted = ["--cases", "c", "--out", "o", "--scripted", "s"];
  assert.deepEqual((({ samples, aggregate }) => ({ samples, aggregate }))(parseOptions(scripted).options), { samples: 1, aggregate: "unanimous" });
  assert.deepEqual((({ samples, aggregate }) => ({ samples, aggregate }))(parseOptions([...scripted, "--samples", "3", "--aggregate", "majority"]).options), { samples: 3, aggregate: "majority" });
  assert.throws(() => parseOptions([...scripted, "--samples", "2"]), /--samples must be 1 or 3/);
  assert.throws(() => parseOptions([...scripted, "--aggregate", "any"]), /--aggregate must be unanimous or majority/);
});
