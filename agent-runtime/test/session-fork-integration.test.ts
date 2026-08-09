import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import { promisify } from "node:util";

import { SessionManager } from "@earendil-works/pi-coding-agent";

import { CodingAgentOrchestrator, type AgentProcess, type ProcessCheckpoint } from "../src/orchestrator.js";
import type { PublicRunEvent } from "../src/pi-rpc.js";
import { PiSessionStore } from "../src/pi-session-store.js";
import { WorktreeManager } from "../src/worktree-manager.js";

const exec = promisify(execFile);

async function git(cwd: string, ...args: string[]) {
  return (await exec("git", args, { cwd, encoding: "utf8" })).stdout.trim();
}

function userMessage(text: string) {
  return { role: "user", content: text, timestamp: Date.now() } as const;
}

function assistantMessage(text: string) {
  return {
    role: "assistant",
    content: [{ type: "text", text }],
    provider: "openai",
    model: "gpt-5.6-luna",
    usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
    stopReason: "stop",
    timestamp: Date.now(),
  } as const;
}

class FauxPiProcess implements AgentProcess {
  private checkpoint?: ProcessCheckpoint;

  constructor(
    private readonly manager: SessionManager,
    private readonly worktreePath: string,
    private readonly onEvent: (event: PublicRunEvent) => void,
  ) {}

  prompt(_runId: string, prompt: string): void {
    const userEntryId = this.manager.appendMessage(userMessage(prompt));
    void writeFile(path.join(this.worktreePath, "shared.txt"), `${prompt}\n`).then(() => {
      const leafId = this.manager.appendMessage(assistantMessage(`completed ${prompt}`) as never);
      this.checkpoint = {
        piSessionId: this.manager.getSessionId(),
        sessionFile: path.basename(this.manager.getSessionFile()!),
        leafId,
        userEntryId,
      };
      this.onEvent({ type: "answer", text: `completed ${prompt}` });
      this.onEvent({ type: "settled" });
    });
  }

  decide() {}
  async abort() {}
  async close() {}
  async getCheckpoint() {
    if (!this.checkpoint) throw new Error("checkpoint missing");
    return this.checkpoint;
  }
  async getTree() {
    return {
      tree: this.manager.getTree() as unknown as Array<Record<string, unknown>>,
      leafId: this.manager.getLeafId(),
    };
  }
}

async function waitForRun(orchestrator: CodingAgentOrchestrator, runId: string) {
  for (let index = 0; index < 100; index += 1) {
    const run = orchestrator.getRun(runId);
    if (run.status === "completed" || run.status === "failed") return run;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error("run did not settle");
}

test("faux Pi runs fork one completed leaf into divergent Git and conversation children", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "pi-fork-integration-"));
  const repo = path.join(root, "repo");
  const worktreeRoot = path.join(root, "worktrees");
  const sessionRoot = path.join(root, "sessions");
  await mkdir(repo);
  await git(repo, "init", "-b", "main");
  await git(repo, "config", "user.name", "Pi Test");
  await git(repo, "config", "user.email", "pi@example.invalid");
  await writeFile(path.join(repo, "shared.txt"), "base\n");
  await git(repo, "add", "shared.txt");
  await git(repo, "commit", "-m", "base");
  const worktrees = new WorktreeManager({ repoPath: repo, worktreeRoot, baseRef: "HEAD" });
  const piSessions = new PiSessionStore(sessionRoot);
  const orchestrator = new CodingAgentOrchestrator({
    worktrees,
    piSessions,
    startProcess: async (session, onEvent) => {
      const directory = path.join(sessionRoot, session.id);
      await mkdir(directory, { recursive: true });
      const manager = session.piSessionFile
        ? SessionManager.open(path.join(directory, session.piSessionFile), directory, session.worktreePath)
        : SessionManager.create(session.worktreePath, directory, { id: session.piSessionId });
      return new FauxPiProcess(manager, session.worktreePath, onEvent);
    },
  });

  const first = await orchestrator.createSession({ team_id: "T", channel_id: "C", thread_ts: "parent", user_id: "U", prompt: "parent-one" });
  const firstRun = await waitForRun(orchestrator, first.run_id);
  const second = await orchestrator.prompt(first.session_id, "parent-two", "U");
  await waitForRun(orchestrator, second.run_id);
  const child = await orchestrator.fork(first.session_id, {
    user_id: "U", source_run_id: first.run_id, team_id: "T", channel_id: "C", thread_ts: "child",
  });
  const childPrompt = await orchestrator.prompt(String(child.session_id), "child-change", "U");
  const childRun = await waitForRun(orchestrator, childPrompt.run_id);
  const parentThird = await orchestrator.prompt(first.session_id, "parent-change", "U");
  const parentRun = await waitForRun(orchestrator, parentThird.run_id);

  assert.notEqual(childRun.commit, parentRun.commit);
  assert.equal(firstRun.commit, child.baseline_commit);
  const parentLookup = orchestrator.lookupSession({ team_id: "T", channel_id: "C", thread_ts: "parent" })!;
  const childLookup = orchestrator.lookupSession({ team_id: "T", channel_id: "C", thread_ts: "child" })!;
  assert.notEqual(parentLookup.branch, childLookup.branch);
  const childTree = await orchestrator.getTree(String(child.session_id), "U");
  assert.equal(childTree.lineage.parent_session_id, first.session_id);
  assert.equal(childTree.lineage.fork_source_run_id, first.run_id);
  assert.equal(await readFile(repo + "/shared.txt", "utf8"), "base\n");
});
