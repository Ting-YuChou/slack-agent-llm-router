import { chmodSync, mkdirSync } from "node:fs";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";

import type { ApprovalPersistence, PersistedApproval } from "./policy.js";

/** Used or expired rows are kept this long after the fact, then pruned. */
export const LEDGER_RETENTION_MS = 24 * 60 * 60_000;

/**
 * Durable delivery approvals under the runtime state root. Unlike ControlStore it exists with or without a
 * trusted profile, so a pending delivery survives a restart bound to the patch hash its card showed.
 * Tool approvals are not recorded: they answer a live Pi dialog that no restart can resume.
 */
export class ApprovalLedger implements ApprovalPersistence {
  private readonly db: DatabaseSync;

  constructor(file: string, private readonly now: () => number = Date.now) {
    mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
    this.db = new DatabaseSync(file); chmodSync(file, 0o600);
    this.db.exec(`PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA busy_timeout=5000;
      CREATE TABLE IF NOT EXISTS delivery_approvals(id TEXT PRIMARY KEY, run TEXT NOT NULL, tool_call TEXT NOT NULL, owner TEXT NOT NULL, action TEXT NOT NULL,
        patch_hash TEXT NOT NULL, created INTEGER NOT NULL, expires INTEGER NOT NULL, used_at INTEGER);
      CREATE INDEX IF NOT EXISTS delivery_approvals_run ON delivery_approvals(run);`);
  }

  record(approval: PersistedApproval): void {
    if (approval.used) throw new Error("A used approval cannot be recorded");
    this.db.prepare("INSERT INTO delivery_approvals(id,run,tool_call,owner,action,patch_hash,created,expires) VALUES(?,?,?,?,?,?,?,?)")
      .run(approval.id, approval.runId, approval.toolCallId, approval.userId, approval.action, approval.patchHash, this.now(), approval.expiresAt);
  }

  markUsed(id: string): void {
    this.db.prepare("UPDATE delivery_approvals SET used_at=? WHERE id=? AND used_at IS NULL").run(this.now(), id);
  }

  /** The run's newest unused delivery approval, expired or not; the caller decides what an expired one means. */
  pendingDelivery(runId: string): PersistedApproval | undefined {
    const row = this.db.prepare("SELECT * FROM delivery_approvals WHERE run=? AND used_at IS NULL ORDER BY created DESC, rowid DESC LIMIT 1").get(runId);
    if (!row) return undefined;
    return {
      id: String(row.id), runId: String(row.run), toolCallId: String(row.tool_call), userId: String(row.owner), action: String(row.action),
      patchHash: String(row.patch_hash), expiresAt: Number(row.expires), used: false,
    };
  }

  /** Deletes rows used, or expired, more than LEDGER_RETENTION_MS ago and returns how many went. */
  prune(): number {
    const cutoff = this.now() - LEDGER_RETENTION_MS;
    return Number(this.db.prepare("DELETE FROM delivery_approvals WHERE expires<? OR used_at<?").run(cutoff, cutoff).changes);
  }

  close(): void {
    if (this.db.isOpen) this.db.close();
  }
}
