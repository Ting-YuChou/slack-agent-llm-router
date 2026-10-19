import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { GitHubAppTokenProvider } from "./github-app-token.js";
import { CiGitHub } from "./ci-github.js";
import { CiRepairs } from "./ci-repair.js";
import { ControlStore } from "./ci-control.js";
import { CandidateStore, DockerVerifier, type VerificationProfile } from "./verification.js";
import { WorktreeManager } from "./worktree-manager.js";
import type { CodingAgentOrchestrator } from "./orchestrator.js";
import type { AgentTracing } from "./tracing.js";
import type { AgentTelemetry } from "./telemetry.js";
import { RuntimeError } from "./orchestrator.js";
import { createCiWebhook } from "./ci-webhook.js";
import { renderPrompt } from "./prompts.js";

export async function createCiRuntime(options: { priorActive: Map<string, number>; verificationBases: Map<string, string>; profile: VerificationProfile; root: string; repoPath: string; worktreeRoot: string; orchestrator: CodingAgentOrchestrator; tracing?: AgentTracing; telemetry?: AgentTelemetry }) {
  const { profile, root, repoPath, orchestrator, telemetry } = options;
  if (!profile.slack?.team_id || !profile.slack.channel_id || !profile.slack.user_id || !profile.workflows?.length) throw new Error("CI repair requires profile Slack owner mapping and workflow allowlist");
  async function credentials(prefix: string, required: boolean) {
    const appId = process.env[`${prefix}_APP_ID`], installationId = process.env[`${prefix}_APP_INSTALLATION_ID`], file = process.env[`${prefix}_APP_PRIVATE_KEY_PATH`];
    if (!appId || !installationId || !file) { if (required) throw new Error(`${prefix} App credentials missing`); return undefined; }
    return new GitHubAppTokenProvider({ appId, installationId, repository: profile.repository.split("/")[1], permissions: prefix === "GITHUB_PUBLISHER" ? { contents: "write", pull_requests: "write" } : { contents: "read", pull_requests: "read", actions: "read" }, privateKey: await readFile(file, "utf8") });
  }
  const reader = await credentials("GITHUB", true);
  const publisher = await credentials("GITHUB_PUBLISHER", true);
  if (process.env.GITHUB_APP_ID === process.env.GITHUB_PUBLISHER_APP_ID) throw new Error("Publisher must use a separate GitHub App from read-only MCP");
  const store = new ControlStore(path.join(root, "control", "state.sqlite"), () => { void orchestrator.shutdown().catch(() => {}); });
  const github = new CiGitHub({ repository: profile.repository, workflows: profile.workflows, checkout: profile.ciCheckout, repoPath, store, token: () => reader!.getToken(), publisherToken: () => publisher!.getToken() });
  const worktrees = new WorktreeManager({ repoPath, worktreeRoot: options.worktreeRoot, baseRef: "HEAD" });
  const snapshots = new CandidateStore(path.join(root, "reproductions"));
  const verifier = new DockerVerifier(profile);
  const stages = new Map<string, string>();
  const ci = new CiRepairs({
    observe: job => {
      const tracing = options.tracing;
      tracing?.startRun(job.run_id, job.session_id);
      const old = stages.get(job.run_id);
      if (old !== job.phase) {
        if (old) tracing?.lifecycle(job.run_id, old, "end", job.status === "failed");
        if (job.status === "running" || job.status === "awaiting_approval") tracing?.lifecycle(job.run_id, job.phase, "start");
        stages.set(job.run_id, job.phase);
      }
      if (!["running", "awaiting_approval"].includes(job.status)) { tracing?.endRun(job.run_id, job.status); stages.delete(job.run_id); }
    }, store, repository: profile.repository, owner: profile.slack,
    validateMode: mode => { if (mode === 'structured' && !profile.structuredWorkflow?.enabled) throw new RuntimeError('Structured workflow requires enabled profile','workflow_setup_required',503); },
    resumeExecution: (id,owner) => orchestrator.resume(id,owner),
    investigate: input => github.investigate(input),
    reproduce: async (target, signal, workflowId) => {
      const started = performance.now();
      const bounded = AbortSignal.any([signal, AbortSignal.timeout(profile.activeTimeoutMs)]);
      await github.fetchCommit(target.tested);
      const id = `reproduce-${target.run}-${target.attempt}-${Date.now()}`;
      const tree = await worktrees.create(id, target.tested);
      try {
        const result = await verifier.verify(await snapshots.create(tree.path, target.tested, id), bounded);
        if (result.status !== "failed") return result.status;
        const reproduced = result.checks.some(check => check.status === "failed" && profile.checks.some(config => check.name === `baseline:${config.name}` && config.ciJob && target.failed_jobs?.includes(config.ciJob)));
        return reproduced ? "failed" : "unknown";
      }
      finally { await verifier.stop(); await worktrees.remove(tree.path); options.priorActive.set(workflowId, performance.now() - started); }
    },
    execute: (target, id, mode) => { options.verificationBases.set(id, target.tested); return orchestrator.createSession({ ...profile.slack!, thread_ts: `ci-repair-${id}`, task_id: id, workflow_mode: mode,
      prompt: renderPrompt('ci-repair', { repository: target.repository, pr: target.pr, head: target.head, diagnostic: target.diagnostic.slice(0, 12000) }),
    }, target.head, id, { delivery: "skip" }); },
    result: id => orchestrator.getRun(id),
    decideExecution: (id, approval, owner, decision) => orchestrator.decide(id, approval, owner, decision),
    cancel: (id, owner) => orchestrator.cancel(id, owner),
    publish: (target, commit, branch, id) => github.publish(target, commit, branch, id),
    record: async (id, payload) => { await telemetry?.record(id, id, "delivery", payload); },
  });
  ci.recover();
  let draining = false; let lastPrune = 0;
  const drain = async () => {
    if (draining) return; draining = true;
    try {
      if (Date.now() - lastPrune > 86400000) { ci.pruneDiagnostics(); lastPrune = Date.now(); }
      for (const row of store.pending()) {
        try {
          const event = row.payload as { run_id: number; attempt: number; head_sha: string; branch: string; workflow: string };
          const prior = store.get<{ id: string }>("inbox_repair", row.delivery);
          if (!prior) {
            const current = await github.investigate(String(event.run_id));
            if (current.attempt !== event.attempt || current.head !== event.head_sha || current.branch !== event.branch || current.workflow !== event.workflow) { store.finish(row.delivery); continue; }
            const job = await ci.create(String(event.run_id), profile.slack!.user_id, createHash("sha256").update(row.delivery).digest("hex"), current);
            store.put("inbox_repair", row.delivery, { id: job.run_id });
          }
          store.finish(row.delivery);
        } catch (error) {
          const reason = error instanceof Error ? error.message : "investigation_failed";
          store.put("inbox_error", row.delivery, { reason, at: new Date().toISOString() });
          if (/^(?:workflow_must_identify_one_pr|stale_fork_or_unsupported_ci_identity|no_allowlisted_failed_workflow_for_head|pr_not_found)$/.test(reason)) store.finish(row.delivery);
        }
      }
    } catch { /* durable inbox remains pending; next bounded poll retries */ }
    finally { draining = false; }
  };
  // drain catches everything itself; the outer catch only keeps the interval from leaking a rejection.
  const timer = setInterval(() => { drain().catch(() => undefined); }, 5_000); timer.unref();
  const webhook = process.env.PI_AGENT_CI_WEBHOOK_ENABLED === "true" ? createCiWebhook({ store, secret: process.env.GITHUB_WEBHOOK_SECRET ?? "", repository: profile.repository, workflows: profile.workflows }) : undefined;
  if (webhook) await new Promise<void>((resolve, reject) => { webhook.once("error", reject); webhook.listen(Number(process.env.PI_AGENT_CI_WEBHOOK_PORT ?? 3003), "127.0.0.1", resolve); });
  return { ci, close: async () => { clearInterval(timer); await new Promise<void>(resolve => { if (webhook) webhook.close(() => resolve()); else resolve(); }); store.close(); } };
}
