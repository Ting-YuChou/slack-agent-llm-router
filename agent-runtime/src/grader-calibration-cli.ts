/**
 * Grades every calibration case under `--cases` K times with the real GraderRunner
 * and writes one JSONL line per (case, repeat), then a closing `run` line.
 *
 *   --scripted <file>  offline: a scripted grader process, deterministic, no model calls
 *   live               --gateway-url, --network, --image, --budget-ledger and
 *                      MODEL_GATEWAY_SIGNING_SECRET in the environment (spends model budget)
 *
 * scripts/grader_calibration.py boots the live gateway, runs this CLI and scores its output.
 */
import { execFile } from "node:child_process";
import { cp, mkdir, mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs, promisify } from "node:util";

import { resolveAgentModel } from "./agent-model.js";
import { BudgetLedger } from "./budget-ledger.js";
import { issueGatewayToken } from "./gateway-token.js";
import { DEFAULT_GRADER_MODEL } from "./grader-config.js";
import { GraderRunner, type GraderProcessFactory } from "./grader.js";
import type { RouteDecision } from "./jev-router.js";
import type { PublicRunEvent } from "./pi-rpc.js";
import { PROMPTS_VERSION } from "./prompts.js";
import { QualityGate, type Candidate, type Grade, type VerificationResult } from "./quality-gate.js";
import { outputTokenLimitFor } from "./run-budget.js";
import { CandidateStore } from "./verification.js";
import { DEFAULT_DIFF_LIMITS } from "./worktree-manager.js";

const exec = promisify(execFile);

/** The task the calibration candidates attempt; a case may override it with `requirement`. Never the case description: that is the answer key. */
export const DEFAULT_REQUIREMENT = "Fix two independent bugs in src/counter.cjs and src/label.cjs. increment(2) must return 3 and label(2) must return count:2. Preserve both original tests.";

export interface CalibrationCase {
  id: string; expected: "pass" | "fail"; category: string; rubric: string[]; allowedPaths: string[];
  verification: VerificationResult; requirement: string; base: string; patch: string;
}

/** One scripted grader execution: a grade to submit, an execution error, or (neither) settling without a submission. */
export interface ScriptedStep {
  status?: Grade["status"]; evidence?: string; criteria?: Array<{ passed: boolean; evidence?: string }>; findings?: Grade["findings"];
  error?: string; cost_usd?: number;
}
export interface Script { cases: Record<string, ScriptedStep[]> }

export async function loadCases(root: string): Promise<CalibrationCase[]> {
  const entries = (await readdir(root, { withFileTypes: true })).filter(entry => entry.isDirectory()).map(entry => entry.name).sort();
  return Promise.all(entries.map(async id => {
    const dir = path.join(root, id);
    const raw = JSON.parse(await readFile(path.join(dir, "case.json"), "utf8")) as Partial<CalibrationCase>;
    if (!["pass", "fail"].includes(String(raw.expected)) || !Array.isArray(raw.rubric) || !Array.isArray(raw.allowedPaths) || raw.verification?.status !== "passed") throw new Error(`${id}: malformed case.json`);
    return {
      id, expected: raw.expected!, category: String(raw.category), rubric: raw.rubric, allowedPaths: raw.allowedPaths,
      verification: raw.verification, requirement: typeof raw.requirement === "string" ? raw.requirement : DEFAULT_REQUIREMENT,
      base: path.join(dir, "base"), patch: path.join(dir, "candidate.patch"),
    };
  }));
}

// Fixed identity and dates make the baseline commit, and so the candidate hash, reproducible.
const GIT_ENV = {
  GIT_AUTHOR_NAME: "calibration", GIT_AUTHOR_EMAIL: "calibration@example.invalid", GIT_AUTHOR_DATE: "2026-01-01T00:00:00Z",
  GIT_COMMITTER_NAME: "calibration", GIT_COMMITTER_EMAIL: "calibration@example.invalid", GIT_COMMITTER_DATE: "2026-01-01T00:00:00Z",
  GIT_CONFIG_NOSYSTEM: "1", GIT_CONFIG_GLOBAL: os.devNull,
};
const git = async (cwd: string, ...args: string[]) => (await exec("git", args, { cwd, env: { PATH: process.env.PATH, ...GIT_ENV } })).stdout.trim();

/** Commits `base/` in a fresh repository, applies the patch and snapshots it the way the runtime does before grading. */
export async function buildCandidate(item: CalibrationCase, work: string): Promise<Candidate> {
  const repo = path.join(work, "repo");
  await cp(item.base, repo, { recursive: true });
  await git(repo, "init", "-q");
  await git(repo, "add", "-A");
  await git(repo, "commit", "-qm", "base");
  const baseline = await git(repo, "rev-parse", "HEAD");
  await git(repo, "apply", "--whitespace=nowarn", item.patch);
  return new CandidateStore(path.join(work, "snapshots")).create(repo, baseline, item.id, { baselineTree: false });
}

/** A grader process that plays the next scripted step for its case, filling in the hash and rubric ids the harness expects. */
export function scriptedGrader(steps: ScriptedStep[]): GraderProcessFactory {
  let next = 0;
  return (options, onEvent, onRecord) => ({
    prompt: () => {
      const step = steps[next++] ?? {};
      const { candidate_hash: hash, rubric } = options.graderContext as { candidate_hash: string; rubric: string[] };
      setImmediate(() => {
        if (step.cost_usd) onEvent({ type: "usage", cost_usd: step.cost_usd });
        if (step.error) { onEvent({ type: "error", code: step.error, message: "scripted" } as PublicRunEvent); return; }
        if (step.status) {
          const criteria = rubric.map((id, index) => ({ id, ...(step.criteria?.[index] ?? { passed: step.status === "passed", evidence: step.evidence ?? "" }) }));
          const args = { status: step.status, candidate_hash: hash, criteria, findings: step.findings ?? [] };
          onRecord({ type: "tool_execution_start", toolName: "submit_grade", toolCallId: "scripted", args });
          onRecord({ type: "tool_execution_end", toolName: "submit_grade", toolCallId: "scripted", isError: false });
        }
        onEvent({ type: "settled", outcome: "completed" } as PublicRunEvent);
      });
    },
    close: async () => {},
  });
}

interface LiveGateway { gatewayUrl: string; network: string; image: string; budgetLedger: string; secret: string }
interface Options {
  cases: string; out: string; repeats: number; maxCostUsd?: number; graderModel: string; timeoutMs: number;
  script?: Script; live?: LiveGateway;
}

export function parseOptions(argv: string[], env: NodeJS.ProcessEnv = process.env): { options: Omit<Options, "script">; scripted?: string } {
  const { values } = parseArgs({ args: argv, strict: true, options: {
    cases: { type: "string" }, out: { type: "string" }, repeats: { type: "string", default: "1" }, "max-cost-usd": { type: "string" },
    "grader-model": { type: "string", default: DEFAULT_GRADER_MODEL }, "timeout-ms": { type: "string", default: "120000" }, scripted: { type: "string" },
    "gateway-url": { type: "string" }, network: { type: "string" }, image: { type: "string" }, "budget-ledger": { type: "string" },
  } });
  if (!values.cases || !values.out) throw new Error("--cases and --out are required");
  const repeats = Number(values.repeats); const timeoutMs = Number(values["timeout-ms"]);
  const maxCostUsd = values["max-cost-usd"] === undefined ? undefined : Number(values["max-cost-usd"]);
  if (!Number.isInteger(repeats) || repeats < 1) throw new Error("--repeats must be a positive integer");
  if (!Number.isInteger(timeoutMs) || timeoutMs < 1) throw new Error("--timeout-ms must be a positive integer");
  if (maxCostUsd !== undefined && !(Number.isFinite(maxCostUsd) && maxCostUsd > 0)) throw new Error("--max-cost-usd must be a positive number");
  resolveAgentModel(values["grader-model"]);
  const liveFlags = [values["gateway-url"], values.network, values.image, values["budget-ledger"]];
  const options: Omit<Options, "script"> = { cases: values.cases, out: values.out, repeats, maxCostUsd, graderModel: values["grader-model"]!, timeoutMs };
  if (values.scripted !== undefined && liveFlags.every(flag => flag === undefined)) return { options, scripted: values.scripted };
  if (values.scripted === undefined && liveFlags.every(Boolean)) {
    const secret = env.MODEL_GATEWAY_SIGNING_SECRET ?? "";
    if (!secret) throw new Error("live mode reads the gateway signing secret from MODEL_GATEWAY_SIGNING_SECRET");
    // Live runs spend real money; the cap is also the gateway budget, so it is mandatory.
    if (maxCostUsd === undefined) throw new Error("live mode requires --max-cost-usd");
    return { options: { ...options, live: { gatewayUrl: liveFlags[0]!, network: liveFlags[1]!, image: liveFlags[2]!, budgetLedger: liveFlags[3]!, secret } } };
  }
  throw new Error("choose --scripted or the live gateway flags (--gateway-url, --network, --image, --budget-ledger), not both");
}

function graderRunner(options: Options, item: CalibrationCase, stateRoot: string, route: RouteDecision, budget: string | undefined, costs: (cost: number) => void): GraderRunner {
  const live = options.live;
  return new GraderRunner({
    image: live?.image ?? "scripted", network: live?.network ?? "none", gatewayUrl: live?.gatewayUrl ?? "http://scripted.invalid", stateRoot,
    route, timeoutMs: options.timeoutMs, constraints: { allowedPaths: item.allowedPaths, diffLimits: DEFAULT_DIFF_LIMITS },
    cost: async cost => { costs(cost); },
    ...(live ? {} : { createProcess: scriptedGrader(options.script?.cases[item.id] ?? []) }),
    token: (id, selected) => {
      if (!live || !budget) throw new Error("scripted graders take no gateway token");
      const model = resolveAgentModel(selected.modelRef);
      return issueGatewayToken({
        runId: id, workflowRunId: budget, role: "grader", outputTokenLimit: outputTokenLimitFor("grader"), provider: model.provider, model: model.id,
        api: model.api, reasoningEffort: selected.effort, expiresAt: Date.now() + options.timeoutMs + 60_000,
      }, live.secret);
    },
  });
}

export async function calibrate(options: Options): Promise<{ complete: boolean; records: number }> {
  const cases = await loadCases(options.cases);
  const route: RouteDecision = { modelRef: options.graderModel, effort: "high", source: "grader" };
  const work = await mkdtemp(path.join(os.tmpdir(), "grader-calibration-"));
  // The ledger caps live spend at the gateway too, not just between gradings.
  const budget = options.live ? `grader-calibration-${path.basename(work)}` : undefined;
  if (options.live && budget) new BudgetLedger(options.live.budgetLedger).open(budget, options.maxCostUsd!);
  const lines: string[] = []; let spent = 0; let records = 0; let stopped = false;
  try {
    for (const item of cases) {
      if (stopped) break;
      const caseDir = path.join(work, item.id); await mkdir(caseDir, { recursive: true });
      const candidate = await buildCandidate(item, caseDir);
      let cost = 0;
      const runner = graderRunner(options, item, path.join(caseDir, "state"), route, budget, spent => { cost += spent; });
      const gate = new QualityGate({
        profile: { id: "grader-calibration", rubric: item.rubric, maxRepairs: 0 },
        snapshot: async () => candidate, verify: async () => item.verification,
        grade: (snapshot, verification, input) => runner.grade(snapshot, verification, input.requirement, item.rubric, input.run_id, input.signal),
      });
      for (let repeat = 1; repeat <= options.repeats; repeat++) {
        if (options.maxCostUsd !== undefined && spent > options.maxCostUsd) { stopped = true; break; }
        cost = 0;
        const result = await gate.evaluate({ run_id: `${item.id}-${repeat}`, worktree: caseDir, baseline: "", requirement: item.requirement, attempt: 1, signal: new AbortController().signal });
        const { cause, grader_attempts: attempts, ...grade } = result.grade!;
        spent += cost; records++;
        lines.push(JSON.stringify({
          type: "grade", case: item.id, repeat, expected: item.expected, category: item.category, status: grade.status,
          cause: cause ?? null, grader_attempts: attempts, prompts_version: PROMPTS_VERSION, grader_model: options.graderModel,
          cost_usd: roundUsd(cost), grade,
        }));
      }
    }
  } finally { await rm(work, { recursive: true, force: true }); }
  const expected = cases.length * options.repeats;
  const complete = records === expected;
  lines.push(JSON.stringify({
    type: "run", complete, ...(complete ? {} : { reason: "max_cost_usd_exceeded" }), records, expected_records: expected,
    repeats: options.repeats, max_cost_usd: options.maxCostUsd ?? null, cost_usd: roundUsd(spent), grader_model: options.graderModel,
    prompts_version: PROMPTS_VERSION, mode: options.live ? "live" : "scripted",
  }));
  await writeFile(options.out, lines.join("\n") + "\n");
  return { complete, records };
}

/** Float sums of cents drift (0.1 + 0.2); results are compared exactly, so keep 12 decimals. */
function roundUsd(value: number): number { return Math.round(value * 1e12) / 1e12; }

/** An incomplete run still exits 0: its `run` line says so, and the contract fails it. */
async function main(argv: string[]): Promise<void> {
  const { options, scripted } = parseOptions(argv);
  const script = scripted === undefined ? undefined : JSON.parse(await readFile(scripted, "utf8")) as Script;
  await calibrate({ ...options, script });
}

const entrypoint = process.argv[1] ? pathToFileURL(process.argv[1]).href : undefined;
if (entrypoint === import.meta.url) {
  main(process.argv.slice(2)).catch((error: unknown) => {
    process.stderr.write(`grader calibration failed: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
