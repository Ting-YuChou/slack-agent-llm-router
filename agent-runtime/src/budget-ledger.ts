import { DatabaseSync } from "node:sqlite";
import { mkdirSync, chmodSync } from "node:fs";
import path from "node:path";
/** Shared only with the model gateway, never mounted in a Pi container. */
export class BudgetLedger {
  private readonly db: DatabaseSync;
  constructor(file: string) {
    mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 }); this.db = new DatabaseSync(file); chmodSync(file, 0o600);
    this.db.exec(`PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA busy_timeout=5000;
      CREATE TABLE IF NOT EXISTS budgets(id TEXT PRIMARY KEY, limit_usd REAL NOT NULL, initial_usd REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1);
      CREATE TABLE IF NOT EXISTS reservations(id TEXT PRIMARY KEY, workflow TEXT NOT NULL, reserved REAL NOT NULL, actual REAL);
      CREATE TABLE IF NOT EXISTS prompt_times(execution TEXT PRIMARY KEY, sent INTEGER NOT NULL, first_model INTEGER);`);
    this.db.exec(`CREATE TABLE IF NOT EXISTS execution_scopes(execution TEXT PRIMARY KEY,workflow TEXT NOT NULL,stage TEXT NOT NULL,cap REAL NOT NULL);
      CREATE TABLE IF NOT EXISTS reservation_scopes(id TEXT PRIMARY KEY,workflow TEXT NOT NULL,stage TEXT NOT NULL);`);
  }
  open(id: string, limit: number, initial = 0) {
    if (!Number.isFinite(limit) || limit <= 0 || !Number.isFinite(initial) || initial < 0) throw new Error("Invalid budget");
    this.db.prepare("INSERT OR IGNORE INTO budgets(id,limit_usd,initial_usd) VALUES(?,?,?)").run(id,limit,initial);
  }
  stop(id: string) { this.db.prepare("UPDATE budgets SET active=0 WHERE id=?").run(id); }
  /** An unknown budget is not active: a token whose budget was never opened has nothing to spend. */
  active(id: string): boolean { return (this.db.prepare("SELECT active FROM budgets WHERE id=?").get(id) as { active?: number } | undefined)?.active === 1; }
  executionScope(execution: string, workflow: string, stage: string, cap: number) {
    if (!Number.isFinite(cap) || cap <= 0) throw new Error('Invalid stage cap');
    this.db.prepare('INSERT OR IGNORE INTO execution_scopes VALUES(?,?,?,?)').run(execution,workflow,stage,cap);
  }
  reserve(workflow: string, id: string, amount: number, execution?: string) {
    if (!Number.isFinite(amount) || amount < 0) throw new Error("Invalid reservation");
    this.db.exec("BEGIN IMMEDIATE");
    try {
      const budget = this.db.prepare("SELECT limit_usd,initial_usd,active FROM budgets WHERE id=?").get(workflow);
      const total = this.db.prepare("SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) total FROM reservations WHERE workflow=?").get(workflow)!;
      if (!budget || !budget.active || Number(budget.initial_usd) + Number(total.total) + amount > Number(budget.limit_usd)) throw new Error("workflow_budget_exhausted");
      const scope = execution ? this.db.prepare('SELECT * FROM execution_scopes WHERE execution=? AND workflow=?').get(execution,workflow) : undefined;
      if (scope) {
        const used = this.db.prepare('SELECT COALESCE(SUM(COALESCE(r.actual,r.reserved)),0) total FROM reservations r JOIN reservation_scopes s ON r.id=s.id WHERE s.workflow=? AND s.stage=?').get(workflow,scope.stage)!;
        if (Number(used.total) + amount > Number(scope.cap)) throw new Error('discovery_budget_exhausted');
        this.db.prepare('INSERT INTO reservation_scopes VALUES(?,?,?)').run(id,workflow,scope.stage);
      }
      this.db.prepare("INSERT INTO reservations(id,workflow,reserved) VALUES(?,?,?)").run(id,workflow,amount); this.db.exec("COMMIT");
    } catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }
  settle(id: string, amount: number | null) {
    if (amount === null) return;
    if (!Number.isFinite(amount) || amount < 0) throw new Error("Invalid usage");
    this.db.prepare("UPDATE reservations SET actual=? WHERE id=? AND actual IS NULL").run(amount,id);
  }
  summary(workflow: string) {
    const row = this.db.prepare("SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) AS estimated_usd, COALESCE(SUM(actual),0) AS token_reported_cost_estimate_usd, COUNT(*) FILTER(WHERE actual IS NULL) AS unknown_requests FROM reservations WHERE workflow=?").get(workflow)!;
    const budget = this.db.prepare("SELECT initial_usd,limit_usd FROM budgets WHERE id=?").get(workflow);
    return { estimated_usd: Number(row.estimated_usd), token_reported_cost_estimate_usd: Number(row.token_reported_cost_estimate_usd), unknown_requests: Number(row.unknown_requests), initial_cost_usd: budget ? Number(budget.initial_usd) : undefined, limit_usd: budget ? Number(budget.limit_usd) : undefined, estimated_overshoot_usd: Math.max(0, Number(row.estimated_usd) + Number(budget?.initial_usd ?? 0) - Number(budget?.limit_usd ?? 0)) };
  }
  prompt(execution: string, now = Date.now()) { this.db.prepare("INSERT OR IGNORE INTO prompt_times(execution,sent) VALUES(?,?)").run(execution, now); }
  firstModel(execution: string, now = Date.now()): number | undefined {
    const row = this.db.prepare("UPDATE prompt_times SET first_model=? WHERE execution=? AND first_model IS NULL RETURNING sent").get(now,execution);
    return row ? Number(row.sent) : undefined;
  }
  close() { this.db.close(); }
}
