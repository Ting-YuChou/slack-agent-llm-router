import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { ControlStore } from '../src/ci-control.js';
import { CiRepairs, type RepairTarget } from '../src/ci-repair.js';
const target: RepairTarget = { repository: 'org/repo', pr: 7, head: 'a'.repeat(40), tested: 'a'.repeat(40), branch: 'feature', workflow: 'CI', run: 10, attempt: 1, diagnostic: 'failed tests' };
test('CI requires owner approval before reproduction or writer and validates head before publication', async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'ci-repair-')); const store = new ControlStore(path.join(root, 'db'));
  let executions = 0, reproductions = 0, publications = 0, moved = false;
  const ci = new CiRepairs({ store, repository: 'org/repo', owner: { team_id: 't', channel_id: 'c', user_id: 'owner' },
    investigate: async () => ({ ...target, head: moved ? 'b'.repeat(40) : target.head }),
    reproduce: async () => { reproductions++; return 'failed'; },
    execute: async () => { executions++; return { run_id: 'child', session_id: 'session' }; },
    result: () => ({ status: 'completed', commit: 'c'.repeat(40), branch: 'pi-agent/x', attempts: [{ candidate_hash: 'hash' }], verification_status: 'passed', grader_status: 'passed' }),
    publish: async () => { publications++; return 'https://github.com/org/repo/pull/8'; },
  });
  try {
    const job = await ci.create('7', 'owner');
    assert.equal(executions, 0); assert.equal(reproductions, 0);
    await assert.rejects(ci.decide(job.run_id, job.approval_id!, 'other', 'approve'));
    await ci.decide(job.run_id, job.approval_id!, 'owner', 'approve');
    assert.equal(reproductions, 1); assert.equal(executions, 1);
    const completed = await ci.get(job.run_id);
    assert.equal(completed.phase, 'awaiting_publication');
    moved = true;
    await ci.decide(job.run_id, completed.approval_id!, 'owner', 'approve');
    assert.equal((await ci.get(job.run_id)).publication_status, 'stale'); assert.equal(publications, 0);
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test('CI wrapper forwards an in-run Pi approval to the exact execution instead of dropping it', async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'ci-approval-')); const store = new ControlStore(path.join(root, 'db'));
  let decided = false;
  const ci = new CiRepairs({ store, repository: 'org/repo', owner: { team_id: 't', channel_id: 'c', user_id: 'owner' },
    investigate: async () => target, reproduce: async () => 'failed', execute: async () => ({ run_id: 'child', session_id: 'session' }),
    result: () => ({ status: 'awaiting_approval', events: [{ type: 'approval', approval_id: 'tool-approval' }] }),
    decideExecution: async (id, approval, owner) => { assert.equal(id, 'child'); assert.equal(approval, 'tool-approval'); assert.equal(owner, 'owner'); decided = true; },
    publish: async () => { throw new Error('should not publish'); },
  });
  try {
    const job = await ci.create('7', 'owner'); await ci.decide(job.run_id, job.approval_id!, 'owner', 'approve');
    const waiting = await ci.get(job.run_id); assert.equal(waiting.status, 'awaiting_approval');
    await ci.decide(job.run_id, 'tool-approval', 'owner', 'approve'); assert.equal(decided, true);
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test('CI timing includes publication, excludes both owner waits, and freezes after completion', async t => {
  t.mock.timers.enable({ apis: ['Date'], now: 1_000_000 });
  const root = mkdtempSync(path.join(os.tmpdir(), 'ci-timing-'));
  const store = new ControlStore(path.join(root, 'db'));
  const ci = new CiRepairs({ store, repository: 'org/repo', owner: { team_id: 't', channel_id: 'c', user_id: 'owner' },
    investigate: async () => target,
    reproduce: async () => { t.mock.timers.tick(100); return 'failed'; },
    execute: async () => ({ run_id: 'child', session_id: 'session' }),
    result: () => ({ status: 'completed', commit: 'c'.repeat(40), branch: 'pi-agent/x', attempts: [{ candidate_hash: 'hash' }],
      verification_status: 'passed', grader_status: 'passed', timing: { wall_ms: 300, active_ms: 300, human_wait_ms: 0 } }),
    publish: async () => { t.mock.timers.tick(50); return 'https://github.com/org/repo/pull/8'; },
  });
  try {
    const created = await ci.create('7', 'owner');
    t.mock.timers.tick(1000);
    await ci.decide(created.run_id, created.approval_id!, 'owner', 'approve');
    t.mock.timers.tick(200);
    const verified = await ci.get(created.run_id);
    t.mock.timers.tick(2000);
    const waiting = await ci.get(created.run_id);
    assert.deepEqual(waiting.timing, { wall_ms: 3300, active_ms: 300, human_wait_ms: 3000 });
    await ci.decide(created.run_id, verified.approval_id!, 'owner', 'approve');
    assert.deepEqual((await ci.get(created.run_id)).timing, { wall_ms: 3350, active_ms: 350, human_wait_ms: 3000 });
    t.mock.timers.tick(10_000);
    assert.deepEqual((await ci.get(created.run_id)).timing, { wall_ms: 3350, active_ms: 350, human_wait_ms: 3000 });
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test('late approval expiry observation is waiting time rather than executed work', async t => {
  t.mock.timers.enable({ apis: ['Date'], now: 1_000_000 });
  const root = mkdtempSync(path.join(os.tmpdir(), 'ci-expiry-'));
  const store = new ControlStore(path.join(root, 'db'));
  const ci = new CiRepairs({ store, repository: 'org/repo', owner: { team_id: 't', channel_id: 'c', user_id: 'owner' },
    investigate: async () => target, reproduce: async () => { throw new Error('must not execute'); },
    execute: async () => { throw new Error('must not execute'); }, result: () => ({ status: 'running' }),
    publish: async () => { throw new Error('must not execute'); },
  });
  try {
    const job = await ci.create('7', 'owner');
    t.mock.timers.tick(400_000);
    const expired = await ci.get(job.run_id);
    assert.equal(expired.status, 'timed_out');
    assert.deepEqual(expired.timing, { wall_ms: 400_000, active_ms: 0, human_wait_ms: 400_000 });
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});
