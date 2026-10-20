import { execFile } from "node:child_process";
import { writeFile, unlink, access } from "node:fs/promises";
import { DEFAULT_DIFF_LIMITS, WorktreeManager, hardenedGitEnv } from "./worktree-manager.js";
import { promisify } from "node:util";
import path from "node:path";
import type { ControlStore } from "./ci-control.js";
import { BudgetLedger } from "./budget-ledger.js";
import { QualityGate, type GateInput } from "./quality-gate.js";
import { assertCandidateScope } from "./candidate-scope.js";
import { CandidateStore, DockerVerifier, type VerificationProfile } from "./verification.js";
import { GraderRunner, type GraderProcessFactory } from "./grader.js";
import { DEFAULT_GRADER_MODEL, type GraderConfig } from "./grader-config.js";
import type { QualityServices } from "./orchestrator.js";
import { redact, type AgentTelemetry } from "./telemetry.js";
import { HarnessError } from "./error-taxonomy.js";
import type { AgentTracing } from "./tracing.js";
import { WorkflowCoordinator } from './structured-workflow.js';
import { AgentExecutionRunner, type InvestigationProcessFactory } from './agent-execution-runner.js';
import { promptTelemetry } from "./prompts.js";
import { WriterChecks } from "./writer-checks.js";
import { applyFixers, dockerFixerRunner, type RunFixer } from "./fixers.js";
import type { Candidate } from "./quality-gate.js";
import type { RouteDecision } from './jev-router.js';
const exec = promisify(execFile);
/** What the gate and run_checks need from a verifier; DockerVerifier in production. */
export type QualityVerifier = Pick<DockerVerifier, "verify" | "stop">;

export function createQualityServices(options: {
  control: ControlStore; session: (id: string) => string;
  priorActive: Map<string, number>; repoPath: string; worktreeRoot: string; verificationBases: Map<string, string>;
  profile: VerificationProfile; checksum: string; root: string; image: string; network: string; gatewayUrl: string;
  token: (id: string, route: RouteDecision, role: "grader" | "investigator") => string; ledger: BudgetLedger; mapping: Map<string, string>; telemetry?: AgentTelemetry; tracing?: AgentTracing;
  phase: (run: string, phase: string, cost?: number) => Promise<void>;
  githubMcp?: { gatewayUrl: string; token(id: string, owner: string): string };
  grader?: GraderConfig;
  /** Test seams; production leaves them unset and runs Docker. */
  createVerifier?: (profile: VerificationProfile) => QualityVerifier;
  createGraderProcess?: GraderProcessFactory;
  createExecutionProcess?: InvestigationProcessFactory;
  createFixerRunner?: (image: string) => RunFixer;
}): QualityServices {
  const { profile, telemetry, tracing } = options;
  const grader = options.grader ?? { route: { modelRef: DEFAULT_GRADER_MODEL, effort: "high" as const, source: "grader" }, timeoutMs: 120_000 };
  // Keep the historical fingerprint for the default grader so in-flight workflows still resume after upgrade.
  const graderTag = grader.route.modelRef === DEFAULT_GRADER_MODEL && grader.route.effort === "high" ? "sol-high" : `grader=${grader.route.modelRef}@${grader.route.effort}`;
  const graderModelId = grader.route.modelRef.split("/").slice(1).join("/");
  const store = new CandidateStore(path.join(options.root, "artifacts"));
  const createVerifier = options.createVerifier ?? ((selected: VerificationProfile) => new DockerVerifier(selected));
  const fixerRunner = (options.createFixerRunner ?? dockerFixerRunner)(profile.image);
  const graders = new Map<string, GraderRunner>();
  const verifiers = new Map<string, QualityVerifier>();
  const artifacts = new Map<string, string>();
  const executions = new Map<string, { parent: string; role: string; attempt: number; started_at: string }>();
  const fingerprint = `${options.checksum}:${options.image}:pi-1.0.1:readonly-v1:${graderTag}:luna-max:github-direct4:${options.githubMcp?.gatewayUrl ?? "off"}`;
  async function execution(id: string, parent: string, role: string, attempt: number, phase: "start" | "end", cost?: number, queueMs = 0) {
    const session = !['implementation','discussion'].includes(role) ? id : options.session(parent);
    const statePath = path.join(options.root, ['implementation','discussion'].includes(role) ? 'sessions' : role === 'grader' ? 'graders' : 'investigations', session);
    options.control.put("execution", id, { id, parent, role, attempt, session, statePath, status: phase === "start" ? "running" : "completed", traceparent: tracing?.traceparent(id) });
    if (phase === "start") {
      if (!['implementation','discussion','grader'].includes(role)) options.ledger.executionScope(id,parent,'discovery',profile.structuredWorkflow!.discoveryCostUsd);
      options.mapping.set(id, parent);
      executions.set(id, { parent, role, attempt, started_at: new Date().toISOString() });
      tracing?.startRun(id, session, parent); telemetry?.setTraceparent(id, tracing?.traceparent(id));
    }
    const info = executions.get(id);
    if (phase === "start") options.control.put("execution", id, { id, parent, role, attempt, session, statePath, status: "running", traceparent: tracing?.traceparent(id) });
    const written = await telemetry?.record(id, session, "execution", { execution_id: id, workflow_run_id: parent, role, attempt, phase, profile_id: profile.id, profile_checksum: options.checksum, started_at: info?.started_at, duration_ms: phase === "end" && info ? Date.now() - Date.parse(info.started_at) : 0, queue_ms: queueMs, cost_usd: cost ?? null, ...promptTelemetry(role) }, `execution:${id}:${phase}`);
    if (written === false) telemetry?.markIncomplete(parent, "execution_mapping_capture_failed");
    if (phase === "end") {
      for (const reason of telemetry?.completeness(id) ?? []) telemetry?.markIncomplete(parent, `execution:${id}:${reason}`);
      // Each Pi execution reconciles its own before/after snapshots and official entries.
      const terminalWritten = await telemetry?.record(id, session, "run", { status: "completed", model: role === 'grader' ? graderModelId : role === 'planner' ? "gpt-5.6-sol" : ['implementation','discussion'].includes(role) ? 'unknown' : 'gpt-5.6-luna', reasoning_effort: role === 'grader' ? grader.route.effort : role === 'planner' ? "high" : ['implementation','discussion'].includes(role) ? 'unknown' : 'max', capture_complete: telemetry ? telemetry.completeness(id).length === 0 : null, workflow_run_id: parent, ...promptTelemetry(role) });
      if (terminalWritten === false) telemetry?.markIncomplete(parent, "execution_terminal_capture_failed");
      telemetry?.releaseRun(id);
      tracing?.endRun(id, "completed"); executions.delete(id);
    }
  }
  const runner = new AgentExecutionRunner({ ...options, stateRoot: path.join(options.root,'investigations'), execution,
    token: (id,route) => options.token(id,route,"investigator"), prompt: id => options.ledger.prompt(id),
    cost: (id,cost) => options.phase(id,'investigating',cost), createProcess: options.createExecutionProcess,
  });
  const coordinator = new WorkflowCoordinator({ store: options.control, concurrency: profile.structuredWorkflow?.concurrency ?? 2, degradeUnknown: profile.structuredWorkflow?.degradeUnknown ?? false,
    execute: request => runner.run(request),
    record: async node => {
      await options.phase(node.workflow,node.role === 'planner' ? 'planning' : 'investigating');
      const { result, ...metadata } = node;
      await telemetry?.record(node.workflow,node.workflow,'workflow_node',{ ...metadata, purpose: undefined, expected_output: undefined, workflow_run_id: node.workflow, node_id: node.id, execution_id: node.execution ?? '', duration_ms: node.duration_ms ?? 0, queue_ms: node.queue_ms ?? 0, result_ref: `control:${node.workflow}:${node.id}`, profile_checksum: options.checksum }, `node:${node.workflow}:${node.id}:${node.attempt}:${node.state}`);
      if (node.role === 'planner' && node.state === 'succeeded' && result && 'tasks' in result) await telemetry?.record(node.workflow,node.workflow,'workflow_revision',{ workflow_run_id: node.workflow, revision: node.wave, task_count: result.tasks.length },`revision:${node.workflow}:${node.wave}`);
      for (const dependency of new Set([...node.dependencies, ...(node.fixed_dependencies ?? [])])) await telemetry?.record(node.workflow,node.workflow,'workflow_dependency',{ workflow_run_id: node.workflow, node_id: node.id, depends_on: dependency },`dependency:${node.workflow}:${node.id}:${dependency}`);
    },
  });
  const assertScope = async (worktree: string, baseline: string) => {
    const files = (await exec("git", ["diff", "--name-only", "-z", baseline], { cwd: worktree, env: hardenedGitEnv() })).stdout.split("\0").filter(Boolean);
    const untracked = (await exec("git", ["ls-files", "--others", "--exclude-standard", "-z"], { cwd: worktree, env: hardenedGitEnv() })).stdout.split("\0").filter(Boolean);
    assertCandidateScope([...files, ...untracked], profile.allowedPaths);
  };
  const writerChecks = new WriterChecks({ profile,
    snapshot: async request => { await assertScope(request.worktree, request.baseline); return store.create(request.worktree, request.baseline, request.run_id, { baselineTree: false }); },
    verifier: () => createVerifier(profile),
  });
  const gate = new QualityGate({
    profile,
    snapshot: async input => {
      await assertScope(input.worktree, input.baseline);
      // Fixers edit the worktree itself, so the snapshot, assertUnchanged and the delivery digest all see the fixed diff.
      const fixed = profile.fixers?.length ? await applyFixers({ fixers: profile.fixers, worktree: input.worktree, baseline: input.baseline,
        scratch: path.join(options.root, "fixers"), run: fixerRunner, signal: input.signal }) : undefined;
      const withFixes = (candidate: Candidate): Candidate => fixed ? { ...candidate, fixes: fixed.fixes, ...(fixed.cause ? { fixer_cause: fixed.cause } : {}) } : candidate;
      const source = await store.create(input.worktree, input.baseline, input.run_id);
      const tested = options.verificationBases.get(input.run_id);
      if (!tested || tested === input.baseline) { artifacts.set(`${input.run_id}:${input.attempt}`, path.dirname(source.path)); return withFixes(source); }
      const worktrees = new WorktreeManager({ repoPath: options.repoPath, worktreeRoot: options.worktreeRoot, baseRef: tested });
      const tree = await worktrees.create(`verify-${input.run_id}-${input.attempt}`, tested);
      const patchFile = path.join(path.dirname(source.path), "merge-input.patch");
      try {
        await writeFile(patchFile, source.patch ?? "", { mode: 0o600 });
        if (source.patch) await exec("git", ["apply", "--binary", "--", patchFile], { cwd: tree.path, env: hardenedGitEnv() });
        const merged = await store.create(tree.path, tested, input.run_id);
        artifacts.set(`${input.run_id}:${input.attempt}`, path.dirname(merged.path));
        return withFixes({ ...merged, patch: source.patch, sourceHash: source.hash, testedBaseline: tested });
      } finally { await unlink(patchFile).catch(() => {}); await worktrees.remove(tree.path); }
    },
    verify: async (candidate, input) => {
      await options.phase(input.run_id, "verifying"); tracing?.lifecycle(input.run_id, "verification", "start");
      const verifier = createVerifier(profile); verifiers.set(input.run_id, verifier);
      try { return await verifier.verify(candidate, input.signal); }
      finally { await verifier.stop(); verifiers.delete(input.run_id); tracing?.lifecycle(input.run_id, "verification", "end"); }
    },
    grade: async (candidate, verification, input) => {
      const runner = new GraderRunner({ ...options, prompt: id => options.ledger.prompt(id), stateRoot: path.join(options.root, "graders"),
        token: (id, route) => options.token(id, route, "grader"), route: grader.route, timeoutMs: grader.timeoutMs,
        constraints: { allowedPaths: profile.allowedPaths, diffLimits: DEFAULT_DIFF_LIMITS },
        execution: (id, parent, phase, cost, queueMs) => execution(id, parent, "grader", input.attempt, phase, cost, queueMs),
        cost: cost => options.phase(input.run_id, "grading", cost), createProcess: options.createGraderProcess,
      });
      graders.set(input.run_id, runner);
      await options.phase(input.run_id, "grading"); tracing?.lifecycle(input.run_id, "grading", "start");
      try { return await runner.grade(candidate, verification, input.requirement, profile.rubric, input.run_id, input.signal); }
      finally { if (runner.stopped) graders.delete(input.run_id); tracing?.lifecycle(input.run_id, "grading", "end"); }
    },
    record: async (kind, payload, input) => {
      const location = artifacts.get(`${input.run_id}:${input.attempt}`);
      if (!location) throw new HarnessError("candidate_artifact_missing", "candidate_artifact_missing");
      const json = JSON.stringify(payload);
      await writeFile(path.join(location, `${kind}.json`), telemetry?.redactText(json) ?? String(redact(json)), { mode: 0o600 });
      await telemetry?.record(input.run_id, input.run_id, kind, { ...(payload as object), attempt: input.attempt, profile_id: profile.id, profile_checksum: options.checksum }, `${kind}:${input.run_id}:${input.attempt}`);
    },
  });
  return {
    structured: profile.structuredWorkflow,
    prepare: (id,mode) => { options.ledger.open(id,mode === 'structured' ? profile.structuredWorkflow!.maxCostUsd : profile.maxCostUsd); },
    discover: async input => {
      const snapshot = await store.create(input.worktree,input.baseline,input.run_id);
      return coordinator.discover({ run_id: input.run_id, owner: input.owner, requirement: input.requirement, baseline: input.baseline, snapshot, fingerprint, initial_active_ms: input.initial_active_ms },input.signal);
    },
    validateResume: async input => {
      const saved = options.control.get<{ input: { requirement: string; baseline: string; fingerprint: string; snapshot: { path: string; hash: string; artifact?: string } } }>('structured',input.run_id)?.input;
      if (!saved || saved.requirement !== input.requirement || saved.baseline !== input.baseline || saved.fingerprint !== fingerprint) throw new Error('workflow_identity_changed');
      if ((await exec('git',['rev-parse','HEAD'],{ cwd: input.worktree, env: hardenedGitEnv() })).stdout.trim() !== input.baseline) throw new Error('workflow_baseline_changed');
      await access(saved.snapshot.path); if (saved.snapshot.artifact) await access(saved.snapshot.artifact);
      const current = await store.create(input.worktree,input.baseline,input.run_id);
      if (current.hash !== saved.snapshot.hash) throw new Error('workflow_snapshot_changed');
    },
    writerStarted: id => coordinator.writerStarted(id), canResume: id => coordinator.canResume(id), resumeElapsed: id => coordinator.elapsed(id), tasks: id => coordinator.tasks(id),
    resumeCheckpoint: id => options.control.get<{ updated_at: number }>('structured',id)?.updated_at ?? 0,
    stopDiscovery: id => runner.stop(id),
    stop: async id => { options.ledger.stop(id); await runner.stop(id); await writerChecks.stop(id); await verifiers.get(id)?.stop(); await graders.get(id)?.stop(); },
    runChecks: request => writerChecks.run(request),
    maxRepairs: profile.maxRepairs,
    get writerChecks() { return { maxCalls: profile.writerChecks?.maxCalls ?? 4, checks: writerChecks.checkNames, defaultChecks: writerChecks.defaultChecks, estimateMs: (names: string[]) => writerChecks.estimateMs(names), reserveMs: writerChecks.reserveMs }; },
    open: (id, cost) => options.ledger.open(id, profile.maxCostUsd, cost),
    close: id => { telemetry?.record(id, id, "budget", options.ledger.summary(id)).catch(() => console.error(JSON.stringify({ event: "agent_budget_telemetry_failed", run_id: id }))); options.ledger.stop(id); options.verificationBases.delete(id); for (const [execution, parent] of options.mapping) if (parent === id) options.mapping.delete(execution); },
    elapsedMs: id => options.priorActive.get(id) ?? 0,
    activeTimeoutMs: profile.activeTimeoutMs, maxCostUsd: profile.maxCostUsd,
    evaluate: input => gate.evaluate(input),
    assertUnchanged: (candidate, input) => store.assertUnchanged({ ...candidate, hash: candidate.sourceHash ?? candidate.hash }, input.worktree, input.baseline, input.run_id),
    archive: async input => { await store.create(input.worktree, input.baseline, input.run_id); },
    execution: (id, parent, role, currentAttempt, phase, queueMs) => execution(id, parent, role, currentAttempt, phase, undefined, queueMs),
  };
}
