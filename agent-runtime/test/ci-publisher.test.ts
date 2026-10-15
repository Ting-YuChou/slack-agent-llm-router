import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { ControlStore } from '../src/ci-control.js';
import { CiGitHub } from '../src/ci-github.js';
test('publication recovers a created PR after a response timeout without creating a duplicate', async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'publish-')); const store = new ControlStore(path.join(root, 'db'));
  const sha = 'a'.repeat(40), commit = 'b'.repeat(40); let created = false, creates = 0;
  const client = new CiGitHub({ repository: 'org/repo', workflows: ['CI'], repoPath: root, store,
    token: async () => 'read-token', publisherToken: async () => 'write-token',
    fetch: async (url, init) => {
      if (String(url).endsWith('/pulls/7')) return Response.json({ state: 'open', head: { sha, ref: 'feature' } });
      if (String(url).includes('/git/ref/')) return Response.json({ object: { sha: commit } });
      if (init?.method === 'POST') { creates++; created = true; throw new Error('response_timeout'); }
      return Response.json(created ? [{ html_url: 'https://github.com/org/repo/pull/8', head: { ref: 'pi-agent/repair-job' }, base: { ref: 'feature' } }] : []);
    },
  });
  const target = { repository: 'org/repo', pr: 7, head: sha, tested: sha, branch: 'feature', workflow: 'CI', run: 1, attempt: 1, diagnostic: '' };
  try {
    await assert.rejects(client.publish(target, commit, 'local', 'job'));
    assert.equal(await client.publish(target, commit, 'local', 'job'), 'https://github.com/org/repo/pull/8');
    assert.equal(creates, 1);
    await assert.rejects(client.publish(target, sha, 'local', 'job'));
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});
