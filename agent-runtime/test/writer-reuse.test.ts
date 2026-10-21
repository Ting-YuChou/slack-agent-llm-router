import assert from "node:assert/strict";
import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";

import { agentScheduler } from "../src/agent-scheduler.js";
import { DockerPiProcess } from "../src/container-runtime.js";
import { CodingAgentOrchestrator } from "../src/orchestrator.js";
import type { PublicRunEvent } from "../src/pi-rpc.js";
import { MCP_REUSE_MAX_AGE_MS, writerReuseOptions } from "../src/server.js";
import { validateProfile } from "../src/verification.js";
import { FAKE_DOCKER } from "./support/fake-docker.js";

const base = { id: "p", repository: "o/r", image: `sha256:${"0".repeat(64)}`, rubric: ["ok"], allowedPaths: ["src"], checks: [{ name: "t", argv: ["true"], cwd: ".", timeoutMs: 1000 }] };

test("writer reuse is off unless the profile turns it on; with MCP a paused writer is reused only while young", () => {
  assert.equal(validateProfile(base).writerReuse, false);
  assert.equal(validateProfile({ ...base, writerReuse: true }).writerReuse, true);
  for (const writerReuse of [1, "true", null, {}]) assert.throws(() => validateProfile({ ...base, writerReuse }), /Invalid writer reuse/, JSON.stringify(writerReuse));
  assert.deepEqual(writerReuseOptions(undefined, 0), {});
  assert.deepEqual(writerReuseOptions(validateProfile(base), 0), {});
  assert.deepEqual(writerReuseOptions(validateProfile({ ...base, writerReuse: true }), 0), { reuse: true });
  // MCP tokens are issued when the container starts, for its run, and revoked when the run ends; the
  // age limit makes a late repair cold-start with fresh ones before the first expire.
  assert.deepEqual(writerReuseOptions(validateProfile({ ...base, writerReuse: true }), 1), { reuse: true, reuseMaxAgeMs: MCP_REUSE_MAX_AGE_MS });
  assert.ok(MCP_REUSE_MAX_AGE_MS <= 10 * 60_000);
});


async function withFakeDocker(reuse: boolean | { reuse: boolean; reuseMaxAgeMs?: number }, body: (h: { dir: string; process: DockerPiProcess; events: PublicRunEvent[]; settled: () => Promise<void>; log: () => Promise<string[]> }) => Promise<void>) {
  const dir = await mkdtemp(path.join(os.tmpdir(), "fake-docker-reuse-"));
  await writeFile(path.join(dir, "docker"), FAKE_DOCKER);
  await chmod(path.join(dir, "docker"), 0o755);
  const previousPath = process.env.PATH;
  process.env.PATH = `${dir}:${previousPath}`;
  const events: PublicRunEvent[] = [];
  let waiters: Array<() => void> = [];
  const piProcess = new DockerPiProcess({
    name: `pi-session-reuse-${JSON.stringify(reuse).replace(/\W/g, "")}`, image: "pi-agent@sha256:abc", network: "pi-model-only", worktreePath: "/w", gitMetadataPath: "/repo/.git",
    sessionStatePath: dir, gatewayUrl: "http://model-gateway:8080", modelRef: "openai/gpt-5.6-luna",
    extensionPaths: [], pluginPaths: [], skillPaths: [], toolNames: ["read"], ...(typeof reuse === "boolean" ? { reuse } : reuse),
  }, (runId) => `token-${runId}`, (event) => {
    events.push(event);
    if (event.type === "settled" || event.type === "error") { const ready = waiters; waiters = []; for (const wake of ready) wake(); }
  });
  const settled = () => new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`no settle; events ${JSON.stringify(events)}`)), 10_000);
    waiters.push(() => { clearTimeout(timer); resolve(); });
  });
  const log = async () => (await readFile(path.join(dir, "log"), "utf8").catch(() => "")).trim().split("\n").filter(Boolean);
  try { await body({ dir, process: piProcess, events, settled, log }); }
  finally {
    await piProcess.close();
    process.env.PATH = previousPath;
    await rm(dir, { recursive: true, force: true });
  }
}

const route = { modelRef: "openai/gpt-5.6-luna", effort: "max", source: "explicit" } as const;

test("with reuse, a settled writer is paused, and its repair runs in the same container with fresh credentials", { timeout: 30_000 }, async () => {
  await withFakeDocker(true, async ({ process, events, settled, log }) => {
    let done = settled(); process.prompt("exec-1", "fix", route); await done;
    // Paused before the host sees the settle, so nothing runs while verification and grading do.
    assert.deepEqual(await log(), ["run", "credential token-exec-1", "pause"]);
    done = settled(); process.prompt("exec-2", "repair", route); await done;
    assert.deepEqual(await log(), ["run", "credential token-exec-1", "pause", "unpause", "credential token-exec-2", "pause"]);
    const startups = events.filter((event) => event.type === "startup").map((event) => (event as Extract<PublicRunEvent, { type: "startup" }>).startup);
    assert.equal(startups.length, 2);
    assert.equal(startups[0].reused, false);
    assert.equal(startups[1].reused, true);
    assert.equal(startups[1].queue_ms, 0);
    assert.equal(typeof startups[1].dispatch_to_prompt_ms, "number");
    // Every other ending stops the container; a paused one is unpaused first.
    await process.release();
    assert.deepEqual((await log()).slice(-2), ["unpause", "stop"]);
  });
});

test("cancelling a paused writer unpauses it before stopping it", { timeout: 30_000 }, async () => {
  await withFakeDocker(true, async ({ process, settled, log }) => {
    const done = settled(); process.prompt("exec-1", "fix", route); await done;
    await process.abort("exec-1");
    assert.deepEqual(await log(), ["run", "credential token-exec-1", "pause", "unpause", "stop"]);
  });
});

test("when the paused writer cannot be resumed, the repair cold-starts and says so", { timeout: 30_000 }, async () => {
  await withFakeDocker(true, async ({ dir, process, events, settled, log }) => {
    let done = settled(); process.prompt("exec-1", "fix", route); await done;
    await writeFile(path.join(dir, "fail-unpause"), "");
    done = settled(); process.prompt("exec-2", "repair", route); await done;
    assert.ok(events.some((event) => event.type === "container_reuse_failed"), JSON.stringify(events));
    const lines = await log();
    assert.deepEqual(lines.filter((line) => line === "run").length, 2);
    assert.ok(lines.indexOf("stop") > lines.indexOf("unpause"), lines.join(","));
    assert.equal(lines.at(-1), "pause");
    assert.ok(lines.includes("credential token-exec-2"));
  });
});

test("without reuse every execution gets its own container, stopped at settle", { timeout: 30_000 }, async () => {
  await withFakeDocker(false, async ({ process, settled, log }) => {
    let done = settled(); process.prompt("exec-1", "fix", route); await done;
    done = settled(); process.prompt("exec-2", "repair", route); await done;
    assert.deepEqual(await log(), ["run", "credential token-exec-1", "stop", "run", "credential token-exec-2", "stop"]);
  });
});

test("a paused writer is not reused for a different model, and discussions are never paused", { timeout: 30_000 }, async () => {
  await withFakeDocker(true, async ({ process, settled, log }) => {
    let done = settled(); process.prompt("exec-1", "fix", route); await done;
    done = settled(); process.prompt("exec-2", "fix", { modelRef: "openai/gpt-5.6-sol", effort: "high", source: "explicit" }); await done;
    assert.deepEqual(await log(), ["run", "credential token-exec-1", "pause", "unpause", "stop", "run", "credential token-exec-2", "pause"]);
    done = settled(); process.prompt("exec-3", "explain", route, "discuss"); await done;
    assert.deepEqual((await log()).slice(-3), ["run", "credential token-exec-3", "stop"]);
  });
});

/** How many agent containers could start right now (the scheduler is process-wide). */
async function freeSlots(): Promise<number> {
  const releases: Array<() => void> = [];
  for (let i = 0; i < agentScheduler.limit; i++) {
    try { releases.push(await agentScheduler.acquire(`probe-${i}`, AbortSignal.timeout(50))); } catch { break; }
  }
  for (const release of releases) release();
  return releases.length;
}

test("a kept writer whose container died on its own still frees its execution slot when released", { timeout: 30_000 }, async () => {
  await withFakeDocker(true, async ({ dir, process: piProcess, settled }) => {
    const done = settled(); piProcess.prompt("exec-1", "fix", route); await done;
    assert.equal(await freeSlots(), agentScheduler.limit - 1, "the paused writer holds its slot");
    const pid = Number(await readFile(path.join(dir, "run.pid"), "utf8"));
    process.kill(pid, "SIGTERM");
    for (let i = 0; i < 200; i++) {
      try { process.kill(pid, 0); } catch { break; }
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
    await piProcess.release();
    assert.equal(await freeSlots(), agentScheduler.limit);
  });
});

test("a paused writer older than its reuse age limit is stopped and the repair cold-starts", { timeout: 30_000 }, async () => {
  await withFakeDocker({ reuse: true, reuseMaxAgeMs: 0 }, async ({ process, settled, log }) => {
    let done = settled(); process.prompt("exec-1", "fix", route); await done;
    await new Promise((resolve) => setTimeout(resolve, 5));
    done = settled(); process.prompt("exec-2", "repair", route); await done;
    assert.deepEqual(await log(), ["run", "credential token-exec-1", "pause", "unpause", "stop", "run", "credential token-exec-2", "pause"]);
  });
});

class OrderedProcess {
  handler: (event: PublicRunEvent) => void = () => undefined;
  constructor(private readonly order: string[], private readonly releaseError?: Error) {}
  prompt(runId: string) { this.order.push(`prompt ${runId.length ? "exec" : ""}`); }
  decide() {}
  respondUi() {}
  async release() { this.order.push("release"); if (this.releaseError) throw this.releaseError; }
  async abort() { this.order.push("abort"); }
  async close() {}
  emit(event: PublicRunEvent) { this.handler(event); }
}

function ordered(results: Array<"repair" | "pass" | "fail">, releaseError?: Error) {
  const order: string[] = [];
  const process = new OrderedProcess(order, releaseError);
  let evaluations = 0;
  const quality = {
    execution: () => {}, assertUnchanged: async () => { order.push("assert_unchanged"); },
    evaluate: async () => {
      order.push("evaluate");
      const action = results[evaluations++];
      return { action, ...(action === "repair" ? { repair_prompt: "fix it" } : {}), verification: { status: action === "pass" ? "passed" : "failed", checks: [] }, candidate: { hash: "h", path: "/c" } };
    },
  };
  const orchestrator = new CodingAgentOrchestrator({
    worktrees: {
      create: async () => ({ path: "/managed/w1", branch: "pi-agent/run", baselineCommit: "a".repeat(40) }),
      inspectDiff: async () => ({ files: ["src/a.ts"], bytes: 10, stat: "1 file changed" }),
      commit: async () => { order.push("commit"); return "b".repeat(40); },
      rollback: async () => { order.push("rollback"); },
      remove: async () => undefined,
    },
    startProcess: async (_session: unknown, onEvent: (event: PublicRunEvent) => void) => { process.handler = onEvent; return process; },
    maxActiveSessions: 2, deadlineMs: 10_000, quality,
  } as any);
  return { orchestrator, process, order };
}

const tick = async () => { for (let i = 0; i < 10; i++) await new Promise((resolve) => setImmediate(resolve)); };
const SETTLED = { type: "settled", outcome: "completed" } as const;

test("a passing run releases its writer before committing; a repair keeps it", async () => {
  const { orchestrator, process, order } = ordered(["repair", "pass"]);
  const run = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "reuse-pass", user_id: "U", prompt: "fix" });
  process.emit(SETTLED); await tick();
  process.emit(SETTLED); await tick();
  assert.equal(orchestrator.getRun(run.run_id).status, "completed");
  assert.deepEqual(order, ["prompt exec", "evaluate", "prompt exec", "evaluate", "release", "assert_unchanged", "commit"]);
});

test("a failing run stops its writer before rolling back", async () => {
  const { orchestrator, process, order } = ordered(["fail"]);
  const run = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "reuse-fail", user_id: "U", prompt: "fix" });
  process.emit(SETTLED); await tick();
  assert.equal(orchestrator.getRun(run.run_id).status, "failed");
  assert.ok(order.indexOf("abort") >= 0 && order.indexOf("abort") < order.indexOf("rollback"), order.join(","));
  assert.ok(!order.includes("commit"));
});

test("a writer that cannot be stopped is never committed", async () => {
  const { orchestrator, process, order } = ordered(["pass"], new Error("docker stop failed"));
  const run = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "reuse-stuck", user_id: "U", prompt: "fix" });
  process.emit(SETTLED); await tick();
  assert.equal(orchestrator.getRun(run.run_id).status, "failed");
  assert.equal(orchestrator.getRun(run.run_id).error?.code, "container_stop_failed");
  assert.ok(!order.includes("commit"), order.join(","));
});
