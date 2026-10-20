import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, utimes, access, rm, writeFile, chmod } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pruneEvidence } from '../src/evidence-retention.js';
test('retention only removes expired evidence and preserves active Docker mounts', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'evidence-'));
  try {
    for (const name of ['old', 'active', 'new']) await mkdir(path.join(root, 'artifacts', name), { recursive: true });
    for (const name of ['old', 'active']) await utimes(path.join(root, 'artifacts', name), new Date(0), new Date(0));
    await pruneEvidence(root, [path.join(root, 'artifacts', 'active', 'candidate')], Date.now() - 30 * 86400000);
    await assert.rejects(access(path.join(root, 'artifacts', 'old')));
    await access(path.join(root, 'artifacts', 'active')); await access(path.join(root, 'artifacts', 'new'));
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('retention preserves an old session directory when its context was recently appended', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'session-evidence-'));
  try {
    const session = path.join(root, 'sessions', 'recent-context');
    const expired = path.join(root, 'sessions', 'expired-context');
    await mkdir(session, { recursive: true }); await mkdir(expired, { recursive: true });
    await writeFile(path.join(session, 'session.jsonl'), '{"type":"message"}\n');
    await writeFile(path.join(expired, 'session.jsonl'), '{}\n');
    await utimes(path.join(expired, 'session.jsonl'), new Date(0), new Date(0));
    for (const directory of [session, expired]) await utimes(directory, new Date(0), new Date(0));
    await pruneEvidence(root, [], Date.now() - 30 * 86400000);
    await access(path.join(session, 'session.jsonl'));
    await assert.rejects(access(expired));
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('retention prunes expired read-only undelivered patches', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'patch-evidence-'));
  try {
    const expired = path.join(root, 'artifacts', 'old-run');
    const recent = path.join(root, 'artifacts', 'new-run');
    for (const directory of [expired, recent]) {
      await mkdir(directory, { recursive: true, mode: 0o700 });
      await writeFile(path.join(directory, 'undelivered-2026.patch'), 'diff --git a/x b/x\n', { mode: 0o400 });
    }
    await utimes(path.join(expired, 'undelivered-2026.patch'), new Date(0), new Date(0));
    await utimes(expired, new Date(0), new Date(0));
    await pruneEvidence(root, [], Date.now() - 30 * 86400000);
    await assert.rejects(access(expired));
    await access(path.join(recent, 'undelivered-2026.patch'));
  } finally { await chmod(root, 0o700).catch(() => {}); await rm(root, { recursive: true, force: true }); }
});

test('a broken symlink in old evidence neither stops the pass nor keeps the directory', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'symlink-evidence-'));
  try {
    const first = path.join(root, 'artifacts', 'a-with-link');
    const second = path.join(root, 'artifacts', 'b-plain');
    await mkdir(first, { recursive: true }); await mkdir(second, { recursive: true });
    const { symlink, lutimes } = await import('node:fs/promises');
    await symlink(path.join(root, 'does-not-exist'), path.join(first, 'dangling'));
    await lutimes(path.join(first, 'dangling'), new Date(0), new Date(0));
    for (const directory of [first, second]) await utimes(directory, new Date(0), new Date(0));
    await pruneEvidence(root, [], Date.now() - 30 * 86400000);
    await assert.rejects(access(first));
    await assert.rejects(access(second));
  } finally { await rm(root, { recursive: true, force: true }); }
});
