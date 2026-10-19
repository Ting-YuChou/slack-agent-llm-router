import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { after, before, test } from "node:test";
import { promisify } from "node:util";

import { BudgetLedger } from "../src/budget-ledger.js";
import { ControlStore } from "../src/ci-control.js";
import { DockerPiProcess } from "../src/container-runtime.js";
import type { GraderProcessFactory } from "../src/grader.js";
import { CodingAgentOrchestrator, type RunRecord } from "../src/orchestrator.js";
import type { PublicRunEvent } from "../src/pi-rpc.js";
import { writerContextFor, writerTools } from "../src/server.js";
import { validateProfile } from "../src/verification.js";
import { createQualityServices } from "../src/workflow-runtime.js";
import { WorktreeManager } from "../src/worktree-manager.js";
import { fakeGatewayScript, scriptedTurns, waitForContainerHealthz, type TurnScript } from "./support/fake-provider.js";
import { AGENT_TEST_IMAGE } from "./support/docker-images.js";
import { waitFor } from "./support/scenario.js";

/**
 * The whole loop with nothing scripted below the model: real Pi and policy extension in the agent image, the
 * real DockerVerifier (baseline test restore included), the real quality services and delivery approval. Only
 * the model (a fake Responses gateway) and the grader's submission are scripted.
 */
const exec = promisify(execFile);
const agentImage = AGENT_TEST_IMAGE;
const verificationImage = process.env.PI_AGENT_VERIFICATION_TEST_IMAGE;
const enabled = process.env.PI_AGENT_DOCKER_INTEGRATION === "1" && Boolean(verificationImage);
const skip = enabled ? false : "needs PI_AGENT_DOCKER_INTEGRATION=1 and PI_AGENT_VERIFICATION_TEST_IMAGE (an image digest)";

const ORACLE = "require('node:assert/strict').equal(require('node:fs').readFileSync('src/value.txt','utf8'),'fixed\\n');\n";
const write = (file: string, content: string): TurnScript => ({ kind: "function_call", name: "write", arguments: { path: file, content } });
const done: TurnScript = { kind: "text", text: "Done." };

const saved: Record<string, string | undefined> = {};
const identity = { GIT_AUTHOR_NAME: "loop", GIT_AUTHOR_EMAIL: "loop@example.invalid", GIT_COMMITTER_NAME: "loop", GIT_COMMITTER_EMAIL: "loop@example.invalid" };
before(() => { for (const [key, value] of Object.entries(identity)) { saved[key] = process.env[key]; process.env[key] = value; } });
after(() => { for (const [key, value] of Object.entries(saved)) { if (value === undefined) delete process.env[key]; else process.env[key] = value; } });

async function sh(command: string, args: string[], cwd?: string): Promise<string> {
  return (await exec(command, args, { cwd, encoding: "utf8" })).stdout.trim();
}

interface LoopResult { run: RunRecord; commits: number; files: string[]; deliveryCard: boolean }

async function runLoop(name: string, turns: TurnScript[], grade: "passed" | "unknown" = "passed"): Promise<LoopResult> {
  const suffix = Math.random().toString(16).slice(2, 10);
  const network = `pi-loop-${suffix}`;
  const gateway = `pi-loop-gateway-${suffix}`;
  const root = await mkdtemp(path.join(tmpdir(), "pi-full-loop-"));
  const repo = path.join(root, "repo");
  await mkdir(path.join(repo, "src"), { recursive: true });
  await mkdir(path.join(repo, "test"));
  await writeFile(path.join(repo, "src", "value.txt"), "broken\n");
  await writeFile(path.join(repo, "test", "oracle.cjs"), ORACLE);
  await sh("git", ["init", "-q", "-b", "main"], repo);
  await sh("git", ["add", "."], repo);
  await sh("git", ["commit", "-qm", "baseline"], repo);
  const profile = validateProfile({
    id: `full-loop-${name}`, repository: "fixture/full-loop", image: verificationImage, rubric: ["src/value.txt is fixed"],
    allowedPaths: ["src/", "test/"], baselinePaths: ["test/"], maxRepairs: 1, maxCostUsd: 1, activeTimeoutMs: 240_000,
    checks: [{ name: "oracle", argv: ["node", "test/oracle.cjs"], cwd: ".", timeoutMs: 30_000 }],
  });
  const worktrees = new WorktreeManager({ repoPath: repo, worktreeRoot: path.join(root, "worktrees"), baseRef: "main" });
  const hostUid = process.getuid?.();
  const hostGid = process.getgid?.();
  assert.ok(hostUid !== undefined && hostUid !== 0, "the full loop must run as a non-root host user");
  const createGraderProcess: GraderProcessFactory = (_options, onEvent, onRecord) => ({
    prompt: (_id, prompt) => {
      const hash = /"candidate_hash":"([^"]+)"/.exec(prompt)?.[1];
      const args = { status: grade, candidate_hash: hash, criteria: [{ id: "src/value.txt is fixed", passed: grade === "passed", evidence: grade === "passed" ? "src/value.txt:1 reads fixed" : "" }], findings: [] };
      setImmediate(() => {
        onRecord({ type: "tool_execution_start", toolName: "submit_grade", toolCallId: "g1", args });
        onRecord({ type: "tool_execution_end", toolName: "submit_grade", toolCallId: "g1", isError: false });
        onEvent({ type: "settled", outcome: "completed" } as PublicRunEvent);
      });
    },
    close: async () => {},
  });
  const quality = createQualityServices({
    control: new ControlStore(path.join(root, "control.db")), session: id => id, priorActive: new Map(),
    repoPath: repo, worktreeRoot: path.join(root, "verify-worktrees"), verificationBases: new Map(), profile, checksum: "full-loop",
    root: path.join(root, "runtime"), image: agentImage, network, gatewayUrl: "http://model-gateway:8080", token: () => "unused",
    ledger: new BudgetLedger(path.join(root, "budget.db")), mapping: new Map(), phase: async () => {}, createGraderProcess,
  });
  let orchestrator: CodingAgentOrchestrator | undefined;
  try {
    await sh("docker", ["network", "create", "--internal", network]);
    await sh("docker", ["run", "--detach", "--rm", "--name", gateway, "--network", network, "--network-alias", "model-gateway",
      "node:22.22.0-bookworm-slim", "node", "-e", fakeGatewayScript(scriptedTurns(turns))]);
    await waitForContainerHealthz(gateway);
    orchestrator = new CodingAgentOrchestrator({
      worktrees,
      startProcess: async (session, onEvent) => {
        const sessionStatePath = path.join(root, "sessions", session.id);
        await mkdir(sessionStatePath, { recursive: true });
        return new DockerPiProcess({
          name: `pi-loop-${suffix}-${session.id.slice(0, 6)}`, image: agentImage, network, worktreePath: session.worktreePath,
          gitMetadataPath: path.join(repo, ".git"), sessionStatePath, gatewayUrl: "http://model-gateway:8080", modelRef: session.modelRef,
          ...writerTools(profile, ["/opt/pi/extensions/policy.ts", "/opt/pi/extensions/model-gateway.ts", "/opt/pi/extensions/secret-env.ts"], ["read", "write", "edit", "bash", "grep", "find", "ls"]),
          pluginPaths: [], skillPaths: [], writerContext: writerContextFor(profile), writeApproval: "none",
          user: `${hostUid}:${hostGid}`,
        }, () => "fake-run-token", onEvent);
      },
      deadlineMs: 240_000,
      quality,
      deliveryApproval: { ttlMs: 120_000 },
    });
    const created = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: name, user_id: "U", prompt: "Make src/value.txt read fixed so test/oracle.cjs passes." });
    const terminal = new Set(["completed", "rejected", "cancelled", "failed", "timed_out", "interrupted"]);
    const read = () => orchestrator!.getRun(created.run_id)!;
    const waiting = await waitFor(read, run => terminal.has(run.status) || run.phase === "awaiting_delivery", run => `${name}: stuck in ${run.status}/${run.phase}`, 200_000);
    const card = [...waiting.events].reverse().find(event => event.type === "approval" && event.kind === "delivery");
    if (card) await orchestrator.decide(created.run_id, String(card.approval_id), "U", "approve");
    const run = structuredClone(await waitFor(read, current => terminal.has(current.status), current => `${name}: never finished (${current.status})`, 120_000));
    const worktree = (orchestrator as unknown as { sessions: Map<string, { worktreePath: string }> }).sessions.get(created.session_id)!.worktreePath;
    // Rollback runs after the terminal status; give it a moment before reading the tree.
    await waitFor(async () => sh("git", ["status", "--porcelain"], worktree), status => run.status === "completed" || status === "", status => `${name}: worktree still dirty: ${status}`, 30_000);
    const commits = Number(await sh("git", ["rev-list", "--count", "HEAD"], worktree)) - 1;
    const files = (await sh("git", ["ls-files"], worktree)).split("\n");
    if (commits > 0) assert.equal(await readFile(path.join(worktree, "src", "value.txt"), "utf8"), "fixed\n");
    return { run, commits, files, deliveryCard: Boolean(card) };
  } finally {
    await orchestrator?.shutdown().catch(() => undefined);
    await sh("docker", ["stop", gateway]).catch(() => "");
    await sh("docker", ["network", "rm", network]).catch(() => "");
    await rm(root, { recursive: true, force: true });
  }
}

test("full loop: a verified fix is graded, approved at delivery and committed", { skip, timeout: 300_000 }, async () => {
  const { run, commits, deliveryCard } = await runLoop("pass", [write("src/value.txt", "fixed\n"), done]);
  assert.equal(run.status, "completed", JSON.stringify(run.error));
  assert.equal(run.verification_status, "passed");
  assert.equal(run.grader_status, "passed");
  assert.ok(deliveryCard, "delivery approval shows a card before commit");
  assert.equal(commits, 1);
});

test("full loop: weakening the trusted test fails in the baseline run and never commits", { skip, timeout: 300_000 }, async () => {
  const weaken = write("test/oracle.cjs", "process.exit(0);\n");
  const { run, commits } = await runLoop("weakened", [weaken, done, weaken, done]);
  assert.equal(run.status, "failed");
  assert.equal(run.error?.code, "verification_failed");
  assert.equal(commits, 0);
});

test("full loop: a write outside allowed paths is blocked and nothing is committed", { skip, timeout: 300_000 }, async () => {
  const { run, commits, files } = await runLoop("scope", [write("docs/notes.md", "scratch\n"), done]);
  assert.equal(commits, 0, JSON.stringify(run));
  assert.ok(!files.includes("docs/notes.md"));
});

test("full loop: an unknown grade never commits", { skip, timeout: 300_000 }, async () => {
  const { run, commits } = await runLoop("unknown", [write("src/value.txt", "fixed\n"), done], "unknown");
  assert.equal(run.status, "failed");
  assert.equal(run.error?.cause_code, "insufficient_evidence");
  assert.equal(commits, 0);
});

test("full loop: a truncated model turn never commits", { skip, timeout: 300_000 }, async () => {
  const { run, commits } = await runLoop("truncated", [write("src/value.txt", "fixed\n"), { kind: "incomplete", reason: "max_output_tokens", text: "partial" }]);
  assert.equal(run.status, "failed");
  assert.equal(run.error?.code, "agent_output_truncated");
  assert.equal(commits, 0);
});
