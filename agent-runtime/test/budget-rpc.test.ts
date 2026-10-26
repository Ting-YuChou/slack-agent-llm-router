import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createAgentHttpServer } from '../src/server.js';
import { BudgetLedger } from '../src/budget-ledger.js';
import * as budgetRpcModule from "../src/budget-rpc.js";

test('gateway budget RPC reserves on the sole host ledger and rejects callers without its secret', async () => {
  const m = budgetRpcModule; assert.ok(m,'budget RPC client exists');
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

test('the budget RPC bearer is derived from, and never equal to, the gateway signing secret', async () => {
  const master = 'gateway-signing-secret';
  const derived = budgetRpcModule.deriveBudgetRpcSecret(master);
  assert.notEqual(derived, master);
  assert.equal(derived, budgetRpcModule.deriveBudgetRpcSecret(master), 'deterministic on both sides');
  assert.equal(Buffer.from(derived, 'base64url').length, 32);
  assert.throws(() => budgetRpcModule.deriveBudgetRpcSecret(''), /signing secret/);
});

test('the gateway sends the derived bearer and the runtime refuses the raw signing secret', async () => {
  const { gatewayLedgerFromEnv } = await import('../src/model-gateway.js');
  const master = 'gateway-signing-secret';
  const root = mkdtempSync(path.join(os.tmpdir(),'budget-rpc-')); const ledger = new BudgetLedger(path.join(root,'db'));
  const server = createAgentHttpServer({ token: 'runtime',orchestrator: {} as any,health: () => ({}),acceptRuns: () => true,budgetLedger: ledger,budgetSecret: budgetRpcModule.deriveBudgetRpcSecret(master) } as any);
  try {
    await new Promise<void>(resolve => server.listen(0,'127.0.0.1',resolve)); const address = server.address() as any; const base = `http://127.0.0.1:${address.port}`;
    ledger.open('w',5);
    const gateway = gatewayLedgerFromEnv({ PI_AGENT_BUDGET_RPC_URL: base }, master);
    await gateway.reserve('w','ok',.1);
    await assert.rejects(new budgetRpcModule.BudgetRpcClient(base, master).reserve('w','raw',.1), 'a leaked signing secret is not a budget credential');
  } finally { await new Promise<void>(resolve => server.close(() => resolve())); ledger.close(); rmSync(root,{ recursive: true,force: true }); }
});

test('the budget RPC tells a budget that refused from one it could not ask', async () => {
  const { budgetRefusal, BudgetRpcClient } = budgetRpcModule;
  const root = mkdtempSync(path.join(os.tmpdir(),'budget-rpc-')); const ledger = new BudgetLedger(path.join(root,'db'));
  let healthy = true;
  // The host store itself failing (a disk error, say) is not an exhausted budget.
  const store = { reserve: (workflow: string, id: string, amount: number, execution?: string) => { if (workflow === 'broken') throw new Error('disk I/O error'); ledger.reserve(workflow,id,amount,execution); } };
  const server = createAgentHttpServer({ token: 'runtime',orchestrator: {} as any,health: () => ({}),acceptRuns: () => healthy,budgetLedger: store,budgetSecret: 'gateway' } as any);
  const refusal = (attempt: Promise<void>) => attempt.then(() => assert.fail('the reservation was admitted'), budgetRefusal);
  try {
    await new Promise<void>(resolve => server.listen(0,'127.0.0.1',resolve)); const address = server.address() as any; const base = `http://127.0.0.1:${address.port}`;
    const rpc = new BudgetRpcClient(base,'gateway'); ledger.open('w',1); ledger.open('stopped',1); ledger.stop('stopped');
    assert.equal(await refusal(rpc.reserve('w','over',2)), 'exhausted', 'over the limit');
    assert.equal(await refusal(rpc.reserve('stopped','late',.1)), 'exhausted', 'a stopped budget');
    assert.equal(await refusal(rpc.reserve('never-opened','x',.1)), 'exhausted', 'a budget never opened');
    assert.equal(await refusal(rpc.reserve('broken','x',.1)), 'unavailable', 'the host store failed');
    assert.equal(await refusal(new BudgetRpcClient(base,'runtime').reserve('w','x',.1)), 'unavailable', 'a refused bearer');
    healthy = false;
    assert.equal(await refusal(rpc.reserve('w','x',.1)), 'unavailable', 'an unhealthy runtime');
    assert.equal(await refusal(new BudgetRpcClient('http://127.0.0.1:1','gateway').reserve('w','x',.1)), 'unavailable', 'an unreachable host');
    assert.equal(budgetRefusal(new Error('budget_required')), 'exhausted', 'a token bound to no budget has nothing to spend');
    assert.equal(budgetRefusal(new Error('workflow_budget_setup_missing')), 'unavailable', 'a model the gateway cannot price');
  } finally { await new Promise<void>(resolve => server.close(() => resolve())); ledger.close(); rmSync(root,{ recursive: true,force: true }); }
});

test('the gateway reports a truncated response to the host with its execution and a known role only', async () => {
  const root = mkdtempSync(path.join(os.tmpdir(),'budget-rpc-')); const ledger = new BudgetLedger(path.join(root,'db'));
  const reports: Array<[string, string | undefined]> = [];
  const server = createAgentHttpServer({ token: 'runtime',orchestrator: {} as any,health: () => ({}),budgetLedger: ledger,budgetSecret: 'gateway',truncated: (execution: string, role?: string) => { reports.push([execution, role]); } } as any);
  try {
    await new Promise<void>(resolve => server.listen(0,'127.0.0.1',resolve)); const base = `http://127.0.0.1:${(server.address() as any).port}`;
    const rpc = new budgetRpcModule.BudgetRpcClient(base,'gateway');
    await rpc.truncated('exec-1','writer');
    await rpc.truncated('exec-2');
    await rpc.truncated('exec-3','admin' as any);
    assert.deepEqual(reports, [['exec-1','writer'],['exec-2',undefined],['exec-3',undefined]]);
    await assert.rejects(new budgetRpcModule.BudgetRpcClient(base,'runtime').truncated('exec-4','writer'));
    assert.equal(reports.length, 3, 'a caller without the budget bearer records nothing');
  } finally { await new Promise<void>(resolve => server.close(() => resolve())); ledger.close(); rmSync(root,{ recursive: true,force: true }); }
});
