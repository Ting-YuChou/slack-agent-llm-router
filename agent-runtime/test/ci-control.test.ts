import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { ControlStore, verifyWebhook, parseFailureEvent } from '../src/ci-control.js';
import { createHmac } from 'node:crypto';
test('webhook authentication uses exact bytes and rejects malformed signatures', () => {
  const body = Buffer.from('{"x":1}');
  const signature = 'sha256=' + createHmac('sha256', 'secret').update(body).digest('hex');
  assert.equal(verifyWebhook(body, signature, 'secret'), true);
  assert.equal(verifyWebhook(Buffer.from('{ "x":1}'), signature, 'secret'), false);
  assert.equal(verifyWebhook(body, 'sha256=abc', 'secret'), false);
});
test('inbox deduplicates delivery and workflow identity durably; approvals are bound and consumed once', () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'ci-control-'));
  try {
    let store = new ControlStore(path.join(root, 'control.sqlite'));
    assert.equal(store.enqueue('a', 'repo:run:1:sha', { x: 1 }), true);
    assert.equal(store.enqueue('b', 'repo:run:1:sha', {}), false);
    assert.equal(store.enqueue('a', 'other', {}), false);
    const id = store.approval('run', 'owner', 'publish', 'hash', 'org/repo:branch', 1000);
    assert.throws(() => store.consume(id, 'run', 'attacker', 'publish', 'hash', 'org/repo:branch', 1001));
    store.consume(id, 'run', 'owner', 'publish', 'hash', 'org/repo:branch', 1001);
    assert.throws(() => store.consume(id, 'run', 'owner', 'publish', 'hash', 'org/repo:branch', 1002));
    store.close(); store = new ControlStore(path.join(root, 'control.sqlite'));
    assert.equal(store.pending().length, 1); store.close();
  } finally { rmSync(root, { recursive: true, force: true }); }
});
test('only allowlisted failures are admitted and repair branches cannot recurse', () => {
  const event = { action: 'completed', repository: { full_name: 'org/repo' }, workflow_run: { id: 12, run_attempt: 2, head_sha: 'a'.repeat(40), head_branch: 'main', name: 'Tests', conclusion: 'failure' } };
  assert.ok(parseFailureEvent(event, 'org/repo', ['Tests']));
  assert.equal(parseFailureEvent(event, 'org/other', ['Tests']), null);
  assert.equal(parseFailureEvent({ ...event, workflow_run: { ...event.workflow_run, head_branch: 'pi-agent/repair-123' } }, 'org/repo', ['Tests']), null);
});

test('SQL control failures stop new mutations and trigger shutdown once', () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'control-failure-')); let stopped = 0;
  try {
    const store = new ControlStore(path.join(root, 'db'), () => { stopped++; });
    store.close();
    assert.throws(() => store.put('repair', 'id', {}));
    assert.equal(store.healthy, false);
    assert.throws(() => store.put('repair', 'id2', {}));
    assert.equal(stopped, 1);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
