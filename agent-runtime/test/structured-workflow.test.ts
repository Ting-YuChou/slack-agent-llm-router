import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { ControlStore } from '../src/ci-control.js';
import * as structuredWorkflowModule from "../src/structured-workflow.js";
import * as agentSchedulerModule from "../src/agent-scheduler.js";

const task = (id: string, dependencies: string[] = []) => ({ id, role: 'code_explorer', purpose: 'Find counter implementation', paths: ['src/'], dependencies, expected_output: 'Located code and evidence' });
test('trusted profile enables separate structured limits without changing single mode', async () => {
  const { validateProfile } = await import('../src/verification.js');
  const profile = { id: 'p', repository: 'o/r', image: 'sha256:' + 'a'.repeat(64), rubric: ['correct'], checks: [{ name: 'test', argv: ['pytest'], cwd: '.', timeoutMs: 1000 }], allowedPaths: ['src/'], structuredWorkflow: { enabled: true } };
  const parsed: any = validateProfile(profile);
  assert.equal(parsed.maxCostUsd, 1); assert.equal(parsed.activeTimeoutMs, 900_000);
  assert.equal(parsed.structuredWorkflow.maxCostUsd, 5); assert.equal(parsed.structuredWorkflow.activeTimeoutMs, 1_800_000);
  assert.throws(() => validateProfile({ ...profile, structuredWorkflow: { enabled: true, concurrency: 10 } }));
});

test('structured plans reject cycles, privilege escalation, escaping paths and too many tasks', async () => {
  const m = structuredWorkflowModule;
  assert.ok(m, 'structured workflow implementation exists');
  assert.equal(m.validatePlan({ tasks: [task('a')] }, []).tasks.length, 1);
  assert.throws(() => m.validatePlan({ tasks: [] }, [], true));
  for (const tasks of [[task('a', ['b']), task('b', ['a'])], [{ ...task('a'), role: 'writer' }], [{ ...task('a'), paths: ['../secret'] }], Array.from({ length: 9 }, (_, i) => task('t' + i))]) {
    assert.throws(() => m.validatePlan({ tasks }, []));
  }
});

test('snapshot readers allow absolute paths inside their mount and reject outside paths and broader scopes', async () => {
  const policy: any = await import('../src/policy.js'); assert.equal(typeof policy.validateSnapshotRead,'function');
  const { mkdirSync,writeFileSync } = await import('node:fs');
  const root = mkdtempSync(path.join(os.tmpdir(),'snapshot-read-'));
  try {
    mkdirSync(path.join(root,'src')); mkdirSync(path.join(root,'tests')); writeFileSync(path.join(root,'src/a.ts'),'export const a=1');
    assert.equal((await policy.validateSnapshotRead(root,'/fresh-session',path.join(root,'src/a.ts'),['src/'])).allowed,true);
    assert.equal((await policy.validateSnapshotRead(root,'/fresh-session',root,['.'])).allowed,true);
    assert.equal((await policy.validateSnapshotRead(root,'/fresh-session','/etc/passwd',['.'])).allowed,false);
    assert.equal((await policy.validateSnapshotRead(root,'/fresh-session',path.join(root,'tests'),['src/'])).allowed,false);
    assert.equal((await policy.validateSnapshotRead(root,'/fresh-session',undefined,['.'])).allowed,false);
  } finally { rmSync(root,{ recursive: true,force: true }); }
});

test('planner tool schema advertises the actual allowed roles instead of making the model guess', async () => {
  const { pathToFileURL } = await import('node:url');
  const before = process.env.PI_AGENT_WORKFLOW_JSON;
  try {
    process.env.PI_AGENT_WORKFLOW_JSON = JSON.stringify({ role: 'planner',wave: 1,paths: ['.'],previousTasks: [] });
    const module = await import(pathToFileURL(path.resolve(import.meta.dirname,'../../extensions/workflow-agent.ts')).href);
    let tool: any;
    module.default({ on: () => {},registerTool: (registered: any) => { tool = registered; } });
    const schema = tool.parameters.properties.tasks.items.properties.role;
    assert.deepEqual(schema.anyOf?.map((choice: any) => choice.const),['code_explorer','ci_investigator','test_analyst']);
  } finally { if (before === undefined) delete process.env.PI_AGENT_WORKFLOW_JSON; else process.env.PI_AGENT_WORKFLOW_JSON = before; }
});

test('coordinator persists two waves, enforces dependencies and reuses completed results on resume', async () => {
  const m = structuredWorkflowModule;
  assert.ok(m, 'structured workflow implementation exists');
  const root = mkdtempSync(path.join(os.tmpdir(), 'structured-')); const store = new ControlStore(path.join(root, 'db'));
  const input = { run_id: 'r', owner: 'owner', requirement: 'fix counter', baseline: 'a'.repeat(40), snapshot: { hash: 'h', path: root }, fingerprint: 'profile-tools-models' };
  let plans = 0; const calls: string[] = [];
  const coordinator = new m.WorkflowCoordinator({ store, concurrency: 2,
    execute: async (request: any) => {
      calls.push(request.node.id);
      if (request.node.role === 'planner') return { tasks: ++plans === 1 ? [task('a'), task('b', ['a'])] : [task('c', ['b'])], summary: 'Investigated counter' };
      return { status: 'succeeded', findings: [{ path: 'src/counter.ts', line: 1, reason: 'Counter reset fails' }], evidence: ['src/counter.ts:1'], unresolved: [], suggested_tasks: [] };
    },
  });
  try {
    const result = await coordinator.discover(input, new AbortController().signal);
    assert.match(result, /Counter reset fails/);
    assert.equal(plans, 2); assert.ok(calls.indexOf('a') < calls.indexOf('b')); assert.ok(calls.indexOf('b') < calls.indexOf('c'));
    assert.equal(coordinator.tasks('r').filter((n: any) => n.role !== 'planner').length, 3);
    assert.deepEqual(coordinator.tasks('r').find((n: any) => n.id === 'c')?.fixed_dependencies,['planner-2']);
    assert.deepEqual(coordinator.tasks('r').find((n: any) => n.id === 'planner-2')?.fixed_dependencies,['a','b']);
    const before = calls.length;
    await coordinator.discover(input, new AbortController().signal);
    assert.equal(calls.length, before);
    await assert.rejects(coordinator.discover({ ...input, baseline: 'b'.repeat(40) }, new AbortController().signal), /identity/);
  } finally { store.close(); rmSync(root, { recursive: true, force: true }); }
});

test('resource scheduler limits active containers and removes cancelled queued work', async () => {
  const m = agentSchedulerModule;
  assert.ok(m, 'scheduler implementation exists');
  const scheduler = new m.AgentScheduler(1);
  const release = await scheduler.acquire('a', new AbortController().signal);
  const abort = new AbortController(); const queued = scheduler.acquire('b', abort.signal);
  abort.abort(); await assert.rejects(queued);
  let entered = false;
  const next = scheduler.acquire('c', new AbortController().signal).then((done: () => void) => { entered = true; done(); });
  await new Promise(resolve => setImmediate(resolve)); assert.equal(entered, false);
  release(); await next; assert.equal(entered, true);
});

test('node fencing rejects simultaneous claims and stale terminal results', () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'node-fence-')); const store = new ControlStore(path.join(root,'db'));
  try {
    store.saveWorkflowNode('w','n',{ state: 'pending' });
    store.claimWorkflowNode('w','n','e1',{ state: 'running' });
    assert.throws(() => store.claimWorkflowNode('w','n','e2',{ state: 'running' }));
    assert.throws(() => store.finishWorkflowNode('w','n','wrong',{ state: 'succeeded' }));
    store.finishWorkflowNode('w','n','e1',{ state: 'succeeded' });
    assert.throws(() => store.finishWorkflowNode('w','n','e1',{ state: 'unknown' }));
  } finally { store.close(); rmSync(root,{ recursive: true, force: true }); }
});

test('investigation containers have fresh context, fixed role models and no mutation tools', async () => {
  const { investigationDockerOptions } = await import('../src/agent-execution-runner.js');
  const { buildAgentDockerArgs } = await import('../src/container-runtime.js');
  for (const role of ['planner','test_analyst'] as const) {
    const options = investigationDockerOptions({ input: { run_id: 'r', owner: 'u', requirement: 'fix', baseline: 'a'.repeat(40), snapshot: { hash: 'h', path: '/snapshot' }, fingerprint: 'f' }, node: { ...task('node'), role, workflow: 'r', wave: 1, state: 'running', attempt: 1, execution: 'unique-execution', queued_at: 0 }, results: [], signal: new AbortController().signal }, { image: 'pi', network: 'models', gatewayUrl: 'http://models', statePath: '/fresh' });
    assert.equal(options.modelRef,role === 'planner' ? 'openai/gpt-5.6-sol' : 'openai/gpt-5.6-luna');
    assert.equal(options.reasoningEffort,role === 'planner' ? 'high' : 'max');
    const args = buildAgentDockerArgs({ ...options,gatewayToken: 'token' });
    assert.ok(args.includes('--no-context-files')); assert.ok(!args.includes('--continue'));
    assert.ok(args.includes('type=bind,src=/snapshot,dst=/snapshot,readonly'));
    assert.deepEqual(options.pluginPaths,[]); assert.deepEqual(options.skillPaths,[]);
    assert.ok(!options.toolNames.some(tool => ['bash','edit','write','codemode'].includes(tool)));
  }
});

test('read-only resume reuses finished siblings and retries interrupted work with a new execution ID', async () => {
  const { WorkflowCoordinator } = await import('../src/structured-workflow.js');
  const root = mkdtempSync(path.join(os.tmpdir(),'resume-discovery-')); const store = new ControlStore(path.join(root,'db'));
  const input = { run_id: 'r', owner: 'u', requirement: 'fix', baseline: 'a'.repeat(40), snapshot: { hash: 'h', path: root }, fingerprint: 'f' };
  const controller = new AbortController(); let readers = 0; const interrupted: string[] = [];
  const result = { status: 'succeeded', findings: [], evidence: ['src/counter.ts:1'], unresolved: [], suggested_tasks: [] };
  try {
    const first = new WorkflowCoordinator({ store, concurrency: 1, execute: async request => {
      if (request.node.role === 'planner') return { tasks: [task('a'),task('b',['a'])] };
      if (request.node.id === 'a') { readers++; return result; }
      interrupted.push(request.node.execution!); controller.abort(); throw new Error('stopped');
    } });
    await assert.rejects(first.discover(input,controller.signal));
    store.interruptWorkflowNodes();
    const resumed = new WorkflowCoordinator({ store, concurrency: 2, execute: async request => {
      if (request.node.role === 'planner') return { tasks: [] };
      if (request.node.id === 'a') readers++;
      assert.notEqual(request.node.execution,interrupted[0]); return result;
    } });
    await resumed.discover(input,new AbortController().signal);
    assert.equal(readers,1); assert.equal(resumed.tasks('r').find(node => node.id === 'b')!.attempt,2);
    resumed.writerStarted('r'); assert.equal(resumed.canResume('r'),false);
    await assert.rejects(resumed.discover(input,new AbortController().signal),/writer_started/);
  } finally { store.close(); rmSync(root,{ recursive: true,force: true }); }
});

test('malformed required findings are retried once and never become a successful workflow', async () => {
  const { WorkflowCoordinator } = await import('../src/structured-workflow.js');
  const root = mkdtempSync(path.join(os.tmpdir(),'bad-investigation-')); const store = new ControlStore(path.join(root,'db')); let tries = 0;
  try {
    const coordinator = new WorkflowCoordinator({ store, concurrency: 2, execute: async request => request.node.role === 'planner' ? { tasks: [task('a')] } : (++tries, { status: 'succeeded', findings: [], evidence: [], unresolved: [], suggested_tasks: [] }) });
    await assert.rejects(coordinator.discover({ run_id: 'r',owner: 'u',requirement: 'fix',baseline: 'a'.repeat(40),snapshot: { hash: 'h',path: root },fingerprint: 'f' },new AbortController().signal),/evidence/);
    assert.equal(tries,2); assert.equal(coordinator.tasks('r').find(node => node.id === 'a')!.state,'unknown');
  } finally { store.close(); rmSync(root,{ recursive: true,force: true }); }
});
