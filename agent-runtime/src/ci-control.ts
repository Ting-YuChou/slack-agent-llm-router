import { createHmac, timingSafeEqual, randomUUID } from 'node:crypto';
import { mkdirSync, chmodSync } from 'node:fs';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';

export function verifyWebhook(raw: Buffer, signature: string | undefined, secret: string): boolean {
  if (!secret || !signature || !/^sha256=[a-f0-9]{64}$/.test(signature)) return false;
  return timingSafeEqual(Buffer.from(signature.slice(7), 'hex'), createHmac('sha256', secret).update(raw).digest());
}
export function parseFailureEvent(event: any, repository: string, workflows: string[]) {
  const run = event?.workflow_run;
  if (event?.action !== 'completed' || run?.conclusion !== 'failure' || event?.repository?.full_name !== repository || !workflows.includes(run?.name) || typeof run.head_branch !== 'string' || run.head_branch.startsWith('pi-agent/') || !/^[a-f0-9]{40}$/.test(run.head_sha) || !Number.isSafeInteger(run.id) || !Number.isSafeInteger(run.run_attempt)) return null;
  return { repository, workflow: run.name as string, run_id: run.id as number, attempt: run.run_attempt as number, head_sha: run.head_sha as string, branch: run.head_branch as string };
}
/** Execution control never shares telemetry's availability or retention policy. */
export class ControlStore {
  healthy = true;
  private readonly db: DatabaseSync;
  constructor(file: string, private readonly onFailure?: () => void) {
    mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
    this.db = new DatabaseSync(file); chmodSync(file, 0o600);
    this.db.exec(`PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA busy_timeout=5000;
      CREATE TABLE IF NOT EXISTS inbox(delivery TEXT PRIMARY KEY, identity TEXT UNIQUE NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending');
      CREATE TABLE IF NOT EXISTS approvals(id TEXT PRIMARY KEY, run TEXT NOT NULL, owner TEXT NOT NULL, kind TEXT NOT NULL, hash TEXT NOT NULL, target TEXT NOT NULL, expires INTEGER NOT NULL, consumed INTEGER NOT NULL DEFAULT 0);
      CREATE TABLE IF NOT EXISTS records(kind TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(kind,id));`);
    this.db.exec(`CREATE TABLE IF NOT EXISTS workflow_nodes(workflow TEXT NOT NULL,id TEXT NOT NULL,execution TEXT,payload TEXT NOT NULL,PRIMARY KEY(workflow,id));
      CREATE TABLE IF NOT EXISTS workflow_dependencies(workflow TEXT NOT NULL,node TEXT NOT NULL,dependency TEXT NOT NULL,PRIMARY KEY(workflow,node,dependency));
      CREATE TABLE IF NOT EXISTS coordinator_owner(id INTEGER PRIMARY KEY CHECK(id=1),pid INTEGER NOT NULL,token TEXT NOT NULL);`);
    this.db.exec('CREATE TABLE IF NOT EXISTS workflow_attempts(execution TEXT PRIMARY KEY,workflow TEXT NOT NULL,node TEXT NOT NULL,payload TEXT NOT NULL)');
  }
  private checked<T>(operation: () => T): T {
    if (!this.healthy) throw new Error("control_store_unavailable");
    try { return operation(); }
    catch (error) { this.healthy = false; this.onFailure?.(); throw error; }
  }
  enqueue(delivery: string, identity: string, payload: unknown) {
    if (!delivery || delivery.length > 200) throw new Error('invalid_delivery_id');
    return this.checked(() => this.db.prepare('INSERT OR IGNORE INTO inbox(delivery,identity,payload) VALUES(?,?,?)').run(delivery, identity, JSON.stringify(payload))).changes === 1;
  }
  pending(): Array<{ delivery: string; payload: unknown }> {
    return this.checked(() => this.db.prepare("SELECT delivery,payload FROM inbox WHERE status='pending' LIMIT 100").all()).map(row => ({ delivery: String(row.delivery), payload: JSON.parse(String(row.payload)) }));
  }
  finish(delivery: string) { this.checked(() => this.db.prepare("UPDATE inbox SET status='done' WHERE delivery=?").run(delivery)); }
  approval(run: string, owner: string, kind: string, hash: string, target: string, now = Date.now()) {
    const id = randomUUID(); this.checked(() => this.db.prepare('INSERT INTO approvals(id,run,owner,kind,hash,target,expires) VALUES(?,?,?,?,?,?,?)').run(id, run, owner, kind, hash, target, now + 300_000)); return id;
  }
  consume(id: string, run: string, owner: string, kind: string, hash: string, target: string, now = Date.now()) {
    const result = this.checked(() => this.db.prepare('UPDATE approvals SET consumed=1 WHERE id=? AND run=? AND owner=? AND kind=? AND hash=? AND target=? AND expires>? AND consumed=0').run(id, run, owner, kind, hash, target, now));
    if (result.changes !== 1) throw new Error('approval_invalid_expired_or_consumed');
  }
  put(kind: string, id: string, value: unknown) {
    this.checked(() => this.db.prepare('INSERT INTO records(kind,id,payload) VALUES(?,?,?) ON CONFLICT(kind,id) DO UPDATE SET payload=excluded.payload').run(kind, id, JSON.stringify(value)));
  }
  get<T>(kind: string, id: string): T | undefined {
    const row = this.checked(() => this.db.prepare('SELECT payload FROM records WHERE kind=? AND id=?').get(kind,id));
    return row ? JSON.parse(String(row.payload)) as T : undefined;
  }
  all<T>(kind: string): T[] { return this.checked(() => this.db.prepare('SELECT payload FROM records WHERE kind=?').all(kind)).map(row => JSON.parse(String(row.payload)) as T); }
  workflowNodes<T>(workflow: string): T[] { return this.checked(() => this.db.prepare('SELECT payload FROM workflow_nodes WHERE workflow=? ORDER BY rowid').all(workflow)).map(row => JSON.parse(String(row.payload)) as T); }
  saveWorkflowNode(workflow: string, id: string, value: unknown) {
    this.checked(() => this.db.prepare('INSERT OR IGNORE INTO workflow_nodes(workflow,id,payload) VALUES(?,?,?)').run(workflow,id,JSON.stringify(value)));
  }
  saveDependencies(workflow: string, node: string, dependencies: string[]) {
    this.checked(() => { for (const dependency of dependencies) this.db.prepare('INSERT OR IGNORE INTO workflow_dependencies VALUES(?,?,?)').run(workflow,node,dependency); });
  }
  claimWorkflowNode(workflow: string, id: string, execution: string, value: unknown) {
    const updated = this.checked(() => this.db.prepare('UPDATE workflow_nodes SET execution=?,payload=? WHERE workflow=? AND id=? AND execution IS NULL').run(execution,JSON.stringify(value),workflow,id));
    if (updated.changes !== 1) throw new Error('workflow_node_already_claimed');
    this.checked(() => this.db.prepare('INSERT INTO workflow_attempts VALUES(?,?,?,?)').run(execution,workflow,id,JSON.stringify(value)));
  }
  finishWorkflowNode(workflow: string, id: string, execution: string, value: unknown) {
    const updated = this.checked(() => this.db.prepare('UPDATE workflow_nodes SET execution=NULL,payload=? WHERE workflow=? AND id=? AND execution=?').run(JSON.stringify(value),workflow,id,execution));
    if (updated.changes !== 1) throw new Error('stale_workflow_execution');
    this.checked(() => this.db.prepare('UPDATE workflow_attempts SET payload=? WHERE execution=?').run(JSON.stringify(value),execution));
  }
  interruptWorkflowNodes() {
    this.checked(() => this.db.exec(`UPDATE workflow_nodes SET execution=NULL,payload=json_set(payload,'$.state','interrupted') WHERE execution IS NOT NULL`));
  }
  pruneWorkflowContent(before: number) {
    this.checked(() => {
      this.db.prepare("UPDATE records SET payload=json_set(json_remove(payload,'$.input'),'$.expired',json('true')) WHERE kind='structured' AND json_extract(payload,'$.updated_at')<?").run(before);
      this.db.prepare("UPDATE workflow_nodes SET payload=json_remove(payload,'$.result','$.purpose','$.expected_output') WHERE execution IS NULL AND json_extract(payload,'$.queued_at')<?").run(before);
      this.db.prepare("UPDATE workflow_attempts SET payload=json_remove(payload,'$.result','$.purpose','$.expected_output') WHERE json_extract(payload,'$.queued_at')<?").run(before);
    });
  }
  acquireCoordinator() {
    const token = randomUUID(); this.db.exec('BEGIN IMMEDIATE');
    try {
      const owner = this.db.prepare('SELECT pid FROM coordinator_owner WHERE id=1').get();
      if (owner) {
        let alive = true; try { process.kill(Number(owner.pid), 0); } catch (error) { alive = (error as NodeJS.ErrnoException).code !== 'ESRCH'; }
        if (alive) throw new Error('Another workflow coordinator owns this control store');
      }
      this.db.prepare('INSERT OR REPLACE INTO coordinator_owner VALUES(1,?,?)').run(process.pid,token); this.db.exec('COMMIT');
    } catch (error) { this.db.exec('ROLLBACK'); throw error; }
    return () => { this.checked(() => this.db.prepare('DELETE FROM coordinator_owner WHERE token=?').run(token)); };
  }
  close() { this.db.close(); }
}
