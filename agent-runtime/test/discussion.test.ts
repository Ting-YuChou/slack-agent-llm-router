import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, writeFile, symlink, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

test('discussion extension denies mutation, protected reads and symlink escapes without approval', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(),'discussion-'));
  const previous = process.env.PI_AGENT_WORKTREE;
  try {
    await writeFile(path.join(root,'code.ts'),'export const ok = true');
    await writeFile(path.join(root,'.env'),'SECRET=never-read');
    await symlink(os.tmpdir(),path.join(root,'outside'));
    process.env.PI_AGENT_WORKTREE = root;
    const extension = await import(pathToFileURL(path.resolve(import.meta.dirname,'../../extensions/discussion.ts')).href);
    const handlers = new Map<string, (...args: any[]) => any>(); let aborted = 0;
    extension.default({ on: (name: string,handler: (...args: any[]) => any) => handlers.set(name,handler) });
    const ctx = { cwd: '/var/lib/pi-session',abort: () => { aborted++; }, ui: { confirm: () => { throw new Error('discussion cannot request write approval'); } } };
    const call = (toolName: string, file?: unknown) => handlers.get('tool_call')!({ toolName,input: file === undefined ? {} : { path: file } },ctx);
    assert.equal(await call('read',path.join(root,'code.ts')),undefined);
    for (const tool of ['write','edit','bash','codemode','mcp__github__create_pull_request','submit_grade']) assert.equal((await call(tool)).block,true);
    for (const file of [path.join(root,'.env'),path.join(root,'outside'),'/etc/passwd','../escape']) assert.equal((await call('read',file)).block,true);
    const event = { toolName: 'ls',input: {} as Record<string,unknown> };
    assert.equal(await handlers.get('tool_call')!(event,ctx),undefined); assert.equal(event.input.path,root);
    handlers.get('input')!();
    for (let i = 0; i < 40; i++) assert.equal(await call('read',path.join(root,'code.ts')),undefined);
    assert.equal((await call('read',path.join(root,'code.ts'))).block,true); assert.equal(aborted,1);
  } finally {
    if (previous === undefined) delete process.env.PI_AGENT_WORKTREE; else process.env.PI_AGENT_WORKTREE = previous;
    await rm(root,{ recursive: true,force: true });
  }
});

test('discussion execution uses the regular session ledger rather than discovery budget', async () => {
  const { createQualityServices } = await import('../src/workflow-runtime.js');
  const mappings = new Map<string,string>(), records: any[] = [];
  const services = createQualityServices({ root: '/runtime',profile: { id: 'legacy-single-profile' },checksum: 'profile',
    session: () => 'S',mapping: mappings,control: { put: (...args: unknown[]) => records.push(args) },
    ledger: { executionScope: () => { throw new Error('discussion must not require structured discovery budget'); } },
  } as any);
  await services.execution('execution','parent','discussion',1,'start');
  assert.equal(mappings.get('execution'),'parent');
  assert.equal(records.at(-1)[2].statePath,path.join('/runtime','sessions','S'));
  assert.equal(records.at(-1)[2].role,'discussion');
  await services.execution('execution','parent','discussion',1,'end');
});
