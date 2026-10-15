import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { BudgetLedger } from '../src/budget-ledger.js';
test('discovery reservations share a durable stage cap without resetting the parent budget', () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'discovery-budget-')); const ledger = new BudgetLedger(path.join(root, 'db'));
  try {
    ledger.open('parent', 5);
    const stage = ledger as any; stage.executionScope('reader', 'parent', 'discovery', 1.5);
    stage.reserve('parent', 'r1', 1.2, 'reader');
    assert.throws(() => stage.reserve('parent', 'r2', .4, 'reader'));
    stage.settle('r1', null); assert.throws(() => stage.reserve('parent', 'r2', .4, 'reader'));
    stage.reserve('parent', 'writer', 3);
    ledger.open('parent', 100); assert.throws(() => stage.reserve('parent', 'r3', 1));
  } finally { ledger.close(); rmSync(root, { recursive: true, force: true }); }
});
test('durable gateway reservations serialize admission and unknown usage stays charged', () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'budget-'));
  try {
    const ledger = new BudgetLedger(path.join(root, 'db'));
    ledger.open('workflow', 1, .1);
    ledger.reserve('workflow', 'request1', .6);
    assert.throws(() => ledger.reserve('workflow', 'request2', .4));
    ledger.settle('request1', null);
    assert.throws(() => ledger.reserve('workflow', 'request2', .4));
    ledger.settle('request1', .2);
    ledger.reserve('workflow', 'request2', .6);
    assert.throws(() => ledger.reserve('missing', 'request3', .01));
    ledger.close();
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('gateway limits output, settles reported tokens, and refuses missing price configuration', async () => {
  const { admitBudget } = await import('../src/gateway-budget.js');
  const root = mkdtempSync(path.join(os.tmpdir(), 'admission-'));
  const ledger = new BudgetLedger(path.join(root, 'db'));
  try {
    ledger.open('workflow', 1);
    await assert.rejects(admitBudget(ledger, {}, 'workflow', 'openai/model', Buffer.from('{}')));
    const admitted = await admitBudget(ledger, { 'openai/model': { input: 1, output: 10 } }, 'workflow', 'openai/model', Buffer.from('{"max_output_tokens":100000}'));
    assert.equal(JSON.parse(admitted.body.toString()).max_output_tokens, 4096);
    admitted.data(Buffer.from('data: {"response":{"usage":{"input_tokens":100,"output_tokens":10}}}\n\n'));
    await admitted.finish(); await admitted.finish();
    ledger.reserve('workflow', 'next', .99);
  } finally { ledger.close(); rmSync(root, { recursive: true, force: true }); }
});
