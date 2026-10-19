import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import os from 'node:os';
import path from 'node:path';
import { CandidateStore, DockerVerifier, validateProfile } from '../src/verification.js';

test('real Docker verifier restores trusted tests and isolates credentials and snapshot writes', { skip: !process.env.PI_AGENT_VERIFICATION_TEST_IMAGE }, async () => {
  const root = await mkdtemp(path.join(os.tmpdir(),'docker-verifier-'));
  const repo = path.join(root,'repo'); await mkdir(path.join(repo,'src'),{ recursive: true }); await mkdir(path.join(repo,'tests'));
  const git = (...args: string[]) => execFileSync('git',['-C',repo,...args]).toString().trim();
  try {
    git('init','-q'); await writeFile(path.join(repo,'src/value.txt'),'original');
    await writeFile(path.join(repo,'tests/oracle.cjs'), "require('node:assert/strict').equal(require('node:fs').readFileSync('src/value.txt','utf8'),'fixed');\n");
    git('add','.'); git('-c','user.name=Test','-c','user.email=t@example.com','commit','-qm','baseline'); const baseline = git('rev-parse','HEAD');
    await writeFile(path.join(repo,'src/value.txt'),'broken'); await writeFile(path.join(repo,'tests/oracle.cjs'),'process.exit(0);\n');
    const profile = validateProfile({ id: 'docker-test',repository: 'org/repo',image: process.env.PI_AGENT_VERIFICATION_TEST_IMAGE,rubric: ['correct'],allowedPaths: ['src/','tests/'],checks: [
      { name: 'oracle',argv: ['node','--test','tests/oracle.cjs'],cwd: '.',timeoutMs: 30_000 },
      { name: 'isolation',argv: ['node','-e',"const a=require('node:assert/strict');a.equal(process.env.OPENAI_API_KEY,undefined);a.equal(process.env.GITHUB_APP_PRIVATE_KEY_PATH,undefined);a.equal(require('node:child_process').execFileSync('redis-cli',['ping']).toString().trim(),'PONG');a.throws(()=>require('node:fs').writeFileSync('/candidate/src/value.txt','escape'));"],cwd: '.',timeoutMs: 30_000 },
    ] });
    const snapshots = new CandidateStore(path.join(root,'artifacts'));
    const verifier = new DockerVerifier(profile);
    try {
      const broken = await verifier.verify(await snapshots.create(repo,baseline,'broken'),new AbortController().signal);
      assert.equal(broken.status,'failed');
      assert.equal(broken.checks.find(check => check.name === 'baseline:oracle')?.status,'failed');
      assert.equal(broken.checks.find(check => check.name === 'candidate:oracle')?.status,'passed');
      assert.ok(broken.checks.filter(check => check.name.endsWith(':isolation')).every(check => check.status === 'passed'));
      await writeFile(path.join(repo,'src/value.txt'),'fixed');
      const fixed = await verifier.verify(await snapshots.create(repo,baseline,'fixed'),new AbortController().signal);
      assert.equal(fixed.status,'passed'); assert.equal(fixed.checks.length,4);
      assert.equal(fixed.checks.find(check => check.name === 'baseline:oracle')?.passed,1);
      assert.equal(fixed.checks.find(check => check.name === 'baseline:oracle')?.failed,0);
    } finally { await verifier.stop(); }
  } finally { await rm(root,{ recursive: true,force: true }); }
});
