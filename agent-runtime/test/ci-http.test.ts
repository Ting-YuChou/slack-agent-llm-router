import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import type { Server } from "node:http";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { ControlStore } from "../src/ci-control.js";
import { CiRepairs, type RepairTarget } from "../src/ci-repair.js";
import { createCiWebhook } from "../src/ci-webhook.js";
import { createAgentHttpServer } from "../src/server.js";
import { waitFor } from "./support/wait.js";

async function listen(server: Server) {
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  assert.ok(address && typeof address === "object");
  return `http://127.0.0.1:${address.port}`;
}
async function close(server: Server) {
  await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
}

test("signed HTTP webhook persists before ACK, deduplicates replay, and refuses unavailable storage", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "webhook-http-"));
  const file = path.join(root, "control.sqlite");
  let store = new ControlStore(file);
  const server = createCiWebhook({ secret: "test-secret", repository: "org/repo", workflows: ["CI"], store });
  try {
    const base = await listen(server);
    const raw = JSON.stringify({ action: "completed", repository: { full_name: "org/repo" }, workflow_run: {
      name: "CI", id: 42, run_attempt: 1, head_sha: "a".repeat(40), head_branch: "feature", conclusion: "failure",
    } });
    const signature = "sha256=" + createHmac("sha256", "test-secret").update(raw).digest("hex");
    const send = (body: string, delivery: string) => fetch(`${base}/webhooks/github`, { method: "POST", body,
      headers: { "x-hub-signature-256": signature, "x-github-event": "workflow_run", "x-github-delivery": delivery } });
    assert.equal((await send(raw + " ", "tampered")).status, 401);
    assert.equal(store.pending().length, 0);
    assert.equal((await send(raw, "delivery-1")).status, 202);
    assert.equal((await send(raw, "delivery-1")).status, 202);
    assert.equal((await send(raw, "delivery-2")).status, 202);
    assert.equal(store.pending().length, 1);
    store.close();
    assert.equal((await send(raw, "delivery-3")).status, 503);
    store = new ControlStore(file);
    assert.deepEqual(store.pending().map(row => row.delivery), ["delivery-1"]);
    assert.deepEqual(store.pending()[0].payload, { repository: "org/repo", workflow: "CI", run_id: 42,
      attempt: 1, head_sha: "a".repeat(40), branch: "feature" });
  } finally { await close(server); store.close(); await rm(root, { recursive: true, force: true }); }
});

test("HTTP CI workflow keeps publication behind a distinct single-use approval and retains verified results after publish failure", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "ci-http-"));
  const store = new ControlStore(path.join(root, "control.sqlite"));
  const target: RepairTarget = { repository: "org/repo", pr: 7, head: "a".repeat(40), tested: "b".repeat(40),
    branch: "feature", workflow: "CI", run: 42, attempt: 1, diagnostic: "test failed" };
  let result: any = { status: "running", phase: "verifying", execution_status: "succeeded" };
  let publicationRequests = 0;
  let releaseReproduction!: () => void;
  const reproduction = new Promise<void>(resolve => { releaseReproduction = resolve; });
  const ci = new CiRepairs({ store, repository: "org/repo", owner: { team_id: "T", channel_id: "C", user_id: "owner" },
    investigate: async () => target,
    reproduce: async () => { await reproduction; return "failed"; },
    execute: async (_target, id) => ({ run_id: id, session_id: "session" }),
    result: () => result,
    publish: async () => { if (++publicationRequests === 1) throw new Error("provider timeout"); return "https://github.com/org/repo/pull/8"; },
  });
  const server = createAgentHttpServer({ token: "secret", ci, orchestrator: {} as any, health: () => ({}) });
  try {
    const base = await listen(server);
    const headers = { authorization: "Bearer secret", "content-type": "application/json" };
    const post = (route: string, body: unknown) => fetch(base + route, { method: "POST", headers, body: JSON.stringify(body) });
    const context = { team_id: "T", channel_id: "C", user_id: "owner" };
    assert.equal((await post("/v1/ci-repairs", { ...context, channel_id: "other", target: "7" })).status, 401);
    const created = await post("/v1/ci-repairs", { ...context, target: "7" });
    assert.equal(created.status, 202);
    const job = await created.json() as any;
    const status = async () => (await fetch(`${base}/v1/runs/${job.run_id}`, { headers })).json() as Promise<any>;
    const decide = (approval_id: string, user_id = "owner") => post(`/v1/runs/${job.run_id}/decisions`, { approval_id, user_id, decision: "approve" });
    assert.equal((await decide(job.approval_id, "other")).status, 401);
    assert.equal((await decide(job.approval_id)).status, 202);
    assert.equal((await status()).phase, "reproducing");
    assert.equal((await decide(job.approval_id)).status, 409);
    releaseReproduction();
    await waitFor(status, run => run.phase === "verifying", run => `workflow transition timed out; phase ${run.phase}`, 2000);
    assert.equal((await status()).approval_id, undefined);
    assert.equal(publicationRequests, 0);
    result = { status: "completed", execution_status: "succeeded", commit: "c".repeat(40), branch: "pi-agent/repair-test",
      attempts: [{ candidate_hash: "candidate" }], verification_status: "passed", grader_status: "passed" };
    const verified = await status();
    assert.equal(verified.phase, "awaiting_publication");
    assert.notEqual(verified.approval_id, job.approval_id);
    assert.equal((await decide(job.approval_id)).status, 409);
    assert.equal((await decide(verified.approval_id)).status, 202);
    await waitFor(status, run => run.publication_status === "failed", run => `workflow transition timed out; publication_status ${run.publication_status}`, 2000);
    assert.equal((await status()).commit, "c".repeat(40));
    const retry = await post(`/v1/ci-repairs/${job.run_id}/publication`, context);
    assert.equal(retry.status, 202);
    const approval = await retry.json() as any;
    assert.notEqual(approval.approval_id, verified.approval_id);
    assert.equal((await decide(approval.approval_id)).status, 202);
    await waitFor(status, run => run.publication_status === "published", run => `workflow transition timed out; publication_status ${run.publication_status}`, 2000);
    const published = await status();
    assert.equal(published.status, "completed");
    assert.equal(published.url, "https://github.com/org/repo/pull/8");
    assert.equal((await decide(approval.approval_id)).status, 409);
    assert.equal(publicationRequests, 2);
  } finally { releaseReproduction(); await close(server); store.close(); await rm(root, { recursive: true, force: true }); }
});
