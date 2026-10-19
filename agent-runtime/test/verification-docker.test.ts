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

test('real Docker verifier reads junit reports from /out into structured failures', { skip: !process.env.PI_AGENT_VERIFICATION_TEST_IMAGE }, async () => {
  const root = await mkdtemp(path.join(os.tmpdir(),'docker-junit-'));
  const repo = path.join(root,'repo'); await mkdir(path.join(repo,'tests'),{ recursive: true });
  const git = (...args: string[]) => execFileSync('git',['-C',repo,...args]).toString().trim();
  try {
    git('init','-q'); await writeFile(path.join(repo,'tests/test_value.py'),'def test_value():\n    assert 1 == 2\n');
    git('add','.'); git('-c','user.name=Test','-c','user.email=t@example.com','commit','-qm','baseline'); const baseline = git('rev-parse','HEAD');
    const profile = validateProfile({ id: 'docker-junit',repository: 'org/repo',image: process.env.PI_AGENT_VERIFICATION_TEST_IMAGE,rubric: ['correct'],allowedPaths: ['tests/'],checks: [
      { name: 'pytest',argv: ['python','-m','pytest','tests','-q','-p','no:cacheprovider','-o','junit_family=xunit1','--junitxml={report}'],cwd: '.',timeoutMs: 60_000,report: { format: 'junit' } },
    ] });
    const verifier = new DockerVerifier(profile);
    try {
      const result = await verifier.verify(await new CandidateStore(path.join(root,'artifacts')).create(repo,baseline,'junit'),new AbortController().signal);
      const check = result.checks.find(item => item.name === 'candidate:pytest');
      assert.equal(check?.status,'failed');
      assert.equal(check?.failures?.[0]?.file,'tests/test_value.py');
      assert.equal(check?.failures?.[0]?.line,2);
    } finally { await verifier.stop(); }
  } finally { await rm(root,{ recursive: true,force: true }); }
});

test('real Docker per-mode runner gives every check a fresh workspace, HOME, Redis and its own timeout', { skip: !process.env.PI_AGENT_VERIFICATION_TEST_IMAGE }, async () => {
  const root = await mkdtemp(path.join(os.tmpdir(),'docker-per-mode-'));
  const repo = path.join(root,'repo'); await mkdir(path.join(repo,'src'),{ recursive: true });
  const git = (...args: string[]) => execFileSync('git',['-C',repo,...args]).toString().trim();
  try {
    git('init','-q'); await writeFile(path.join(repo,'src/value.txt'),'v');
    git('add','.'); git('-c','user.name=Test','-c','user.email=t@example.com','commit','-qm','baseline'); const baseline = git('rev-parse','HEAD');
    const dirty = "const fs=require('node:fs'),cp=require('node:child_process');fs.writeFileSync('leak.txt','x');fs.writeFileSync(process.env.HOME+'/home-leak','x');cp.execFileSync('redis-cli',['set','leak','x']);cp.spawn('sleep',['300'],{detached:true,stdio:'ignore'}).unref();";
    const fresh = "const a=require('node:assert/strict'),fs=require('node:fs'),cp=require('node:child_process');a.ok(!fs.existsSync('leak.txt'));a.deepEqual(fs.readdirSync(process.env.HOME),[]);a.equal(cp.execFileSync('redis-cli',['get','leak']).toString().trim(),'');a.equal(process.env.PYTHONNOUSERSITE,'1');const sleeping=fs.readdirSync('/proc').filter(p=>/^\\d+$/.test(p)).filter(p=>{try{return fs.readFileSync('/proc/'+p+'/cmdline','utf8').startsWith('sleep')}catch{return false}});a.deepEqual(sleeping,[]);console.log('fresh ok');";
    const profile = validateProfile({ id: 'docker-per-mode',repository: 'org/repo',image: process.env.PI_AGENT_VERIFICATION_TEST_IMAGE,rubric: ['correct'],allowedPaths: ['src/'],checks: [
      { name: 'dirty',argv: ['node','-e',dirty],cwd: '.',timeoutMs: 30_000 },
      { name: 'fresh',argv: ['node','-e',fresh],cwd: '.',timeoutMs: 30_000 },
      { name: 'hangs',argv: ['node','-e','setInterval(()=>{},1000)'],cwd: '.',timeoutMs: 2_000 },
    ] });
    assert.equal(profile.isolation,'per_mode');
    const verifier = new DockerVerifier(profile);
    try {
      const result = await verifier.verify(await new CandidateStore(path.join(root,'artifacts')).create(repo,baseline,'per-mode'),new AbortController().signal,['candidate']);
      const byName = Object.fromEntries(result.checks.map(check => [check.name,check]));
      assert.equal(byName['candidate:dirty']?.status,'passed',byName['candidate:dirty']?.excerpt);
      assert.equal(byName['candidate:fresh']?.status,'passed',byName['candidate:fresh']?.excerpt);
      assert.equal(byName['candidate:hangs']?.status,'unknown');
      assert.equal(byName['candidate:hangs']?.timed_out,true);
      assert.equal(result.status,'unknown');
    } finally { await verifier.stop(); }
  } finally { await rm(root,{ recursive: true,force: true }); }
});
