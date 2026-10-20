import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { ControlStore, INBOX_MAX_ATTEMPTS, inboxRetryDelayMs, verifyWebhook, parseFailureEvent } from '../src/ci-control.js';
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

test('inbox retries back off exponentially from 5 s to 5 min and the row goes dead after 8 failed attempts', () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'ci-inbox-backoff-'));
  try {
    assert.equal(INBOX_MAX_ATTEMPTS, 8);
    assert.deepEqual([1, 2, 3, 4, 5, 6, 7, 8].map(inboxRetryDelayMs), [5_000, 10_000, 20_000, 40_000, 80_000, 160_000, 300_000, 300_000]);
    let store = new ControlStore(path.join(root, 'control.sqlite'));
    store.enqueue('d', 'repo:run:1:sha', { x: 1 });
    let now = 1_000_000;
    for (let attempt = 1; attempt < INBOX_MAX_ATTEMPTS; attempt++) {
      assert.deepEqual(store.pending(now).map(row => row.delivery), ['d'], `attempt ${attempt} is due`);
      assert.deepEqual(store.fail('d', now), { attempts: attempt, status: 'pending' });
      const delay = inboxRetryDelayMs(attempt);
      assert.equal(store.pending(now + delay - 1).length, 0, `attempt ${attempt} waits ${delay} ms`);
      now += delay;
    }
    store.close(); store = new ControlStore(path.join(root, 'control.sqlite'));
    assert.deepEqual(store.pending(now).map(row => row.delivery), ['d'], 'attempts and schedule survive a restart');
    assert.deepEqual(store.fail('d', now), { attempts: INBOX_MAX_ATTEMPTS, status: 'dead' });
    assert.equal(store.pending(Number.MAX_SAFE_INTEGER).length, 0);
    store.close();
    const db = new DatabaseSync(path.join(root, 'control.sqlite'));
    assert.deepEqual({ ...db.prepare("SELECT status,attempts FROM inbox WHERE delivery='d'").get() }, { status: 'dead', attempts: INBOX_MAX_ATTEMPTS });
    db.close();
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('opening a pre-backoff control store migrates the inbox without losing pending rows', () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'ci-inbox-migrate-')); const file = path.join(root, 'control.sqlite');
  try {
    const legacy = new DatabaseSync(file);
    legacy.exec(`CREATE TABLE inbox(delivery TEXT PRIMARY KEY, identity TEXT UNIQUE NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending');
      INSERT INTO inbox(delivery,identity,payload) VALUES('old','repo:run:1:sha','{"x":1}');
      INSERT INTO inbox(delivery,identity,payload,status) VALUES('finished','repo:run:2:sha','{}','done');`);
    legacy.close();
    let store = new ControlStore(file);
    assert.deepEqual(store.pending(0), [{ delivery: 'old', payload: { x: 1 } }]);
    assert.deepEqual(store.fail('old', 0), { attempts: 1, status: 'pending' });
    store.close(); store = new ControlStore(file);
    assert.equal(store.pending(inboxRetryDelayMs(1)).length, 1, 'reopening a migrated store is idempotent');
    store.close();
  } finally { rmSync(root, { recursive: true, force: true }); }
});
