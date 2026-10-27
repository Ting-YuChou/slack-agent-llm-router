import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdir, mkdtemp, writeFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

// Read-only roles run with cwd outside the snapshot (container-runtime mounts the session state here).
const SESSION_CWD = '/var/lib/pi-session';

async function loadExtension(name: string, env: Record<string, string>) {
  const previous = Object.fromEntries(Object.keys(env).map(key => [key, process.env[key]]));
  Object.assign(process.env, env);
  try {
    const extension = await import(pathToFileURL(path.resolve(import.meta.dirname, `../../extensions/${name}.ts`)).href);
    const handlers = new Map<string, (...args: any[]) => any>();
    extension.default({ on: (event: string, handler: (...args: any[]) => any) => handlers.set(event, handler), registerTool: () => undefined });
    const ctx = { cwd: SESSION_CWD, abort: () => undefined };
    return (toolName: string, input: Record<string, unknown>) => handlers.get('tool_call')!({ toolName, input }, ctx);
  } finally {
    for (const [key, value] of Object.entries(previous)) if (value === undefined) delete process.env[key]; else process.env[key] = value;
  }
}

async function withSnapshot(run: (root: string) => Promise<void>) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'readonly-ext-'));
  try {
    await mkdir(path.join(root, 'src'));
    await writeFile(path.join(root, 'src', 'a.ts'), 'export const a = 1');
    await writeFile(path.join(root, 'README.md'), 'readme');
    await run(root);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

test('grader path-less ls/find/grep default to the snapshot root instead of the session cwd', async () => withSnapshot(async root => {
  const call = await loadExtension('grader', { PI_AGENT_WORKTREE: root, PI_AGENT_GRADER_JSON: JSON.stringify({ candidate_hash: 'h', rubric: [] }) });
  for (const tool of ['ls', 'find', 'grep']) {
    const input: Record<string, unknown> = { pattern: 'a' };
    assert.equal(await call(tool, input), undefined, tool);
    assert.equal(input.path, root);
  }
  assert.equal(await call('read', { path: path.join(root, 'src', 'a.ts') }), undefined);
  assert.equal((await call('read', { path: '/etc/passwd' })).block, true);
  assert.equal((await call('ls', { path: '../escape' })).block, true);
  assert.equal((await call('bash', { command: 'ls' })).block, true);
}));

test('workflow agent path-less reads default to the task scope instead of the session cwd', async () => withSnapshot(async root => {
  const scoped = await loadExtension('workflow-agent', { PI_AGENT_WORKTREE: root, PI_AGENT_WORKFLOW_JSON: JSON.stringify({ role: 'code_explorer', paths: ['src'] }) });
  const input: Record<string, unknown> = { pattern: 'a' };
  assert.equal(await scoped('grep', input), undefined);
  assert.equal(input.path, path.join(root, 'src'));
  assert.equal((await scoped('read', { path: path.join(root, 'README.md') })).block, true);

  const whole = await loadExtension('workflow-agent', { PI_AGENT_WORKTREE: root, PI_AGENT_WORKFLOW_JSON: JSON.stringify({ role: 'planner', paths: ['.'] }) });
  for (const tool of ['ls', 'find']) {
    const event: Record<string, unknown> = {};
    assert.equal(await whole(tool, event), undefined, tool);
    assert.equal(event.path, root);
  }
  assert.equal((await whole('write', { path: path.join(root, 'src', 'a.ts') })).block, true);
}));

test('grader submit_grade rejects evidence the snapshot does not contain', async () => withSnapshot(async root => {
  const tools = new Map<string, any>();
  const previous = { w: process.env.PI_AGENT_WORKTREE, g: process.env.PI_AGENT_GRADER_JSON };
  process.env.PI_AGENT_WORKTREE = root;
  process.env.PI_AGENT_GRADER_JSON = JSON.stringify({ candidate_hash: 'h', rubric: ['correct'] });
  try {
    const extension = await import(pathToFileURL(path.resolve(import.meta.dirname, '../../extensions/grader.ts')).href);
    extension.default({ on: () => undefined, registerTool: (tool: any) => tools.set(tool.name, tool) });
  } finally {
    if (previous.w === undefined) delete process.env.PI_AGENT_WORKTREE; else process.env.PI_AGENT_WORKTREE = previous.w;
    if (previous.g === undefined) delete process.env.PI_AGENT_GRADER_JSON; else process.env.PI_AGENT_GRADER_JSON = previous.g;
  }
  const submit = tools.get('submit_grade');
  const fabricated = { status: 'passed', candidate_hash: 'h', criteria: [{ id: 'correct', passed: true, evidence: 'src/a.ts:99 is correct' }], findings: [] };
  await assert.rejects(submit.execute('t1', fabricated), /src\/a\.ts:99/);
  const grounded = { ...fabricated, criteria: [{ id: 'correct', passed: true, evidence: 'src/a.ts:1 exports a' }] };
  const result = await submit.execute('t2', grounded);
  assert.equal(result.details.status, 'passed');
}));

test('read-only roles resolve relative paths against the snapshot, not the session cwd', async () => withSnapshot(async root => {
  const workflow = await loadExtension('workflow-agent', { PI_AGENT_WORKTREE: root, PI_AGENT_WORKFLOW_JSON: JSON.stringify({ role: 'code_explorer', paths: ['src'] }) });
  const read: Record<string, unknown> = { path: 'src/a.ts' };
  assert.equal(await workflow('read', read), undefined);
  assert.equal(read.path, path.join(root, 'src', 'a.ts'));
  assert.equal((await workflow('read', { path: 'README.md' })).block, true);
  assert.equal((await workflow('read', { path: '../escape' })).block, true);

  const grader = await loadExtension('grader', { PI_AGENT_WORKTREE: root, PI_AGENT_GRADER_JSON: JSON.stringify({ candidate_hash: 'h', rubric: [] }) });
  const graded: Record<string, unknown> = { path: './src/a.ts' };
  assert.equal(await grader('read', graded), undefined);
  assert.equal(graded.path, path.join(root, 'src', 'a.ts'));
}));

test('workflow agent with several task paths defaults path-less search to the first task path', async () => withSnapshot(async root => {
  const call = await loadExtension('workflow-agent', { PI_AGENT_WORKTREE: root, PI_AGENT_WORKFLOW_JSON: JSON.stringify({ role: 'code_explorer', paths: ['src', 'README.md'] }) });
  const input: Record<string, unknown> = { pattern: 'a' };
  assert.equal(await call('grep', input), undefined);
  assert.equal(input.path, path.join(root, 'src'));
}));

test('workflow agent keeps the submit tool reachable after the read budget is spent', async () => withSnapshot(async root => {
  let aborted = 0;
  const previous = { w: process.env.PI_AGENT_WORKTREE, j: process.env.PI_AGENT_WORKFLOW_JSON };
  process.env.PI_AGENT_WORKTREE = root;
  process.env.PI_AGENT_WORKFLOW_JSON = JSON.stringify({ role: 'code_explorer', paths: ['src'] });
  const handlers = new Map<string, (...args: any[]) => any>();
  try {
    const extension = await import(pathToFileURL(path.resolve(import.meta.dirname, '../../extensions/workflow-agent.ts')).href);
    extension.default({ on: (event: string, handler: (...args: any[]) => any) => handlers.set(event, handler), registerTool: () => undefined });
  } finally {
    if (previous.w === undefined) delete process.env.PI_AGENT_WORKTREE; else process.env.PI_AGENT_WORKTREE = previous.w;
    if (previous.j === undefined) delete process.env.PI_AGENT_WORKFLOW_JSON; else process.env.PI_AGENT_WORKFLOW_JSON = previous.j;
  }
  const ctx = { cwd: SESSION_CWD, abort: () => { aborted++; } };
  const call = (toolName: string, input: Record<string, unknown>) => handlers.get('tool_call')!({ toolName, input }, ctx);
  // Blocked calls count toward the budget too.
  for (let i = 0; i < 20; i++) await call('read', { path: i % 2 ? 'src/a.ts' : 'README.md' });
  const over = await call('read', { path: 'src/a.ts' });
  assert.equal(over.block, true);
  assert.match(over.reason, /submit_investigation/);
  assert.equal(aborted, 0);
  assert.equal(await call('submit_investigation', {}), undefined);
  for (let i = 0; i < 5; i++) await call('grep', { pattern: 'x' });
  assert.equal(aborted, 1);
}));
