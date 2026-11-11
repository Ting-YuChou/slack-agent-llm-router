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

test('reservation totals are found through an index, also in a ledger file created before the indexes', async () => {
  const { DatabaseSync } = await import('node:sqlite');
  const root = mkdtempSync(path.join(os.tmpdir(), 'budget-index-')); const file = path.join(root, 'db');
  try {
    // A ledger as an earlier release left it: the tables and their rows, no indexes.
    const old = new DatabaseSync(file);
    old.exec(`CREATE TABLE budgets(id TEXT PRIMARY KEY, limit_usd REAL NOT NULL, initial_usd REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1);
      CREATE TABLE reservations(id TEXT PRIMARY KEY, workflow TEXT NOT NULL, reserved REAL NOT NULL, actual REAL);
      CREATE TABLE prompt_times(execution TEXT PRIMARY KEY, sent INTEGER NOT NULL, first_model INTEGER);
      CREATE TABLE execution_scopes(execution TEXT PRIMARY KEY,workflow TEXT NOT NULL,stage TEXT NOT NULL,cap REAL NOT NULL);
      CREATE TABLE reservation_scopes(id TEXT PRIMARY KEY,workflow TEXT NOT NULL,stage TEXT NOT NULL);
      INSERT INTO budgets(id,limit_usd,initial_usd) VALUES('parent',5,0);
      INSERT INTO reservations(id,workflow,reserved,actual) VALUES('r0','parent',1,0.5);`);
    old.close();
    // The gateway and the host may both open the same file.
    const ledger = new BudgetLedger(file); const second = new BudgetLedger(file);
    const db = new DatabaseSync(file);
    try {
      assert.equal(ledger.summary('parent').estimated_usd, 0.5, 'existing rows keep their totals');
      ledger.reserve('parent', 'r1', 1); assert.equal(second.summary('parent').estimated_usd, 1.5);
      const plan = (sql: string) => db.prepare(`EXPLAIN QUERY PLAN ${sql}`).all().map(row => String(row.detail)).join('\n');
      const total = plan("SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) total FROM reservations WHERE workflow='parent'");
      assert.match(total, /USING INDEX reservations_workflow/); assert.doesNotMatch(total, /SCAN/);
      const stage = plan("SELECT COALESCE(SUM(COALESCE(r.actual,r.reserved)),0) total FROM reservations r JOIN reservation_scopes s ON r.id=s.id WHERE s.workflow='parent' AND s.stage='discovery'");
      assert.match(stage, /INDEX reservation_scopes_stage/); assert.doesNotMatch(stage, /SCAN/);
    } finally { db.close(); ledger.close(); second.close(); }
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('a second prompt for the same execution never re-arms first-model', () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'first-model-')); const ledger = new BudgetLedger(path.join(root, 'db'));
  try {
    assert.equal(ledger.firstModel('reader', 50), undefined, 'no prompt yet');
    ledger.prompt('reader', 100);
    assert.equal(ledger.firstModel('reader', 200), 100);
    ledger.prompt('reader', 300);
    assert.equal(ledger.firstModel('reader', 400), undefined, 'the gateway may stop asking once it has an answer');
  } finally { ledger.close(); rmSync(root, { recursive: true, force: true }); }
});
