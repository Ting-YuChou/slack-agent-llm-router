import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createAgentHttpServer } from '../src/server.js';
import { BudgetLedger } from '../src/budget-ledger.js';

test('gateway budget RPC reserves on the sole host ledger and rejects callers without its secret', async () => {
  const m = await import('../src/budget-rpc.js' as string).catch(() => null); assert.ok(m,'budget RPC client exists');
  const root = mkdtempSync(path.join(os.tmpdir(),'budget-rpc-')); const ledger = new BudgetLedger(path.join(root,'db'));
  let healthy = true;
  const server = createAgentHttpServer({ token: 'runtime',orchestrator: {} as any,health: () => ({}),acceptRuns: () => healthy,budgetLedger: ledger,budgetSecret: 'gateway' } as any);
  try {
    await new Promise<void>(resolve => server.listen(0,'127.0.0.1',resolve)); const address = server.address() as any; const base = `http://127.0.0.1:${address.port}`;
    const rpc = new m.BudgetRpcClient(base,'gateway'); ledger.open('w',5); ledger.executionScope('reader','w','discovery',1.5);
    await rpc.reserve('w','r',1,'reader'); assert.equal(ledger.summary('w').unknown_requests,1);
    await assert.rejects(rpc.reserve('w','r2',1,'reader'));
    await rpc.settle('r',.2); assert.equal(ledger.summary('w').token_reported_cost_estimate_usd,.2);
    ledger.prompt('reader',1234); assert.equal(await rpc.firstModel('reader'),1234); assert.equal(await rpc.firstModel('reader'),undefined);
    const wrong = new m.BudgetRpcClient(base,'runtime'); await assert.rejects(wrong.reserve('w','attacker',.1));
    healthy = false; await assert.rejects(rpc.reserve('w','unhealthy',.1));
    ledger.stop('w'); await assert.rejects(rpc.reserve('w','late',.1));
  } finally { await new Promise<void>(resolve => server.close(() => resolve())); ledger.close(); rmSync(root,{ recursive: true,force: true }); }
});
