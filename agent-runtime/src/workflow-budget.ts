/** Unknown billed usage remains reserved; only an explicit report releases it. */
export class WorkflowBudget {
  private readonly started: number;
  private pausedAt?: number;
  private pausedMs = 0;
  private reservations = new Map<string, number>();
  private reported = 0;
  constructor(readonly limitUsd = 1, readonly limitMs = 900_000, private readonly now = Date.now) {
    if (!Number.isFinite(limitUsd) || limitUsd <= 0 || !Number.isFinite(limitMs) || limitMs <= 0) throw new Error("Invalid workflow budget");
    this.started = now();
  }
  activeMs() { return (this.pausedAt ?? this.now()) - this.started - this.pausedMs; }
  remainingMs() { return Math.max(0, this.limitMs - this.activeMs()); }
  pause() { this.pausedAt ??= this.now(); }
  resume() { if (this.pausedAt !== undefined) { this.pausedMs += this.now() - this.pausedAt; this.pausedAt = undefined; } }
  get estimatedUsd() { return this.reported + [...this.reservations.values()].reduce((a, b) => a + b, 0); }
  check() { if (this.activeMs() >= this.limitMs || this.estimatedUsd > this.limitUsd) throw new Error("workflow_budget_exhausted"); }
  reserve(id: string, usd: number) {
    this.check();
    if (!Number.isFinite(usd) || usd < 0 || this.reservations.has(id) || this.estimatedUsd + usd > this.limitUsd) throw new Error("workflow_budget_exhausted");
    this.reservations.set(id, usd);
  }
  settle(id: string, usd: number | null) {
    if (!this.reservations.has(id)) throw new Error("Unknown budget reservation");
    if (usd === null) return;
    if (!Number.isFinite(usd) || usd < 0) throw new Error("Invalid reported usage");
    this.reservations.delete(id); this.reported += usd;
  }
  report(usd: number) { if (!Number.isFinite(usd) || usd < 0) throw new Error("Invalid reported usage"); this.reported += usd; }
  snapshot() { return { reported_usd: this.reported, estimated_usd: this.estimatedUsd, overshoot_usd: Math.max(0, this.reported - this.limitUsd), active_ms: this.activeMs() }; }
}
