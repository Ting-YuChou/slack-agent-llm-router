import { mkdir, open, readFile, rename } from "node:fs/promises";
import path from "node:path";

/** The same Idempotency-Key was sent with a different request. */
export class IdempotencyKeyReusedError extends Error {
  constructor() { super("Idempotency-Key was already used for a different request"); this.name = "IdempotencyKeyReusedError"; }
}

interface Entry { fingerprint: string; result: unknown; expires_at: number }

/**
 * Remembers a request's successful result by its Idempotency-Key for `ttlMs`, so a client retry
 * (Slack redelivery, a timeout the runtime survived) gets the first answer instead of a second run.
 * Failures are never remembered. With `file`, results survive a runtime restart.
 */
export class IdempotencyCache {
  private readonly entries = new Map<string, Entry>();
  private readonly inflight = new Map<string, { fingerprint: string; promise: Promise<unknown> }>();
  private readonly ttlMs: number;
  private readonly max: number;
  private readonly now: () => number;
  private writes: Promise<void> = Promise.resolve();

  constructor(private readonly options: { file?: string; ttlMs?: number; max?: number; now?: () => number } = {}) {
    this.ttlMs = options.ttlMs ?? 24 * 60 * 60 * 1000;
    this.max = options.max ?? 1000;
    this.now = options.now ?? Date.now;
  }

  get size(): number { return this.entries.size; }

  async load(): Promise<void> {
    if (!this.options.file) return;
    let saved: unknown;
    try { saved = JSON.parse(await readFile(this.options.file, "utf8")); }
    catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return; throw error; }
    if (!Array.isArray(saved)) return;
    for (const item of saved) {
      if (!Array.isArray(item) || typeof item[0] !== "string" || !item[1] || typeof item[1].fingerprint !== "string" || typeof item[1].expires_at !== "number") continue;
      if (item[1].expires_at > this.now()) this.entries.set(item[0], item[1] as Entry);
    }
    this.trim();
  }

  async run<T>(key: string, fingerprint: string, work: () => Promise<T>): Promise<T> {
    const saved = this.entries.get(key);
    if (saved && saved.expires_at <= this.now()) this.entries.delete(key);
    else if (saved) {
      if (saved.fingerprint !== fingerprint) throw new IdempotencyKeyReusedError();
      return saved.result as T;
    }
    const running = this.inflight.get(key);
    if (running) {
      if (running.fingerprint !== fingerprint) throw new IdempotencyKeyReusedError();
      return running.promise as Promise<T>;
    }
    const promise = work();
    this.inflight.set(key, { fingerprint, promise });
    try {
      const result = await promise;
      this.entries.set(key, { fingerprint, result, expires_at: this.now() + this.ttlMs });
      this.trim();
      await this.persist();
      return result;
    } finally {
      this.inflight.delete(key);
    }
  }

  /** Drops expired entries, then the oldest beyond `max` (Map keeps insertion order). */
  private trim(): void {
    const now = this.now();
    for (const [key, entry] of this.entries) if (entry.expires_at <= now) this.entries.delete(key);
    for (const key of this.entries.keys()) { if (this.entries.size <= this.max) break; this.entries.delete(key); }
  }

  /** Best effort: losing the file only means a retry after a restart is not deduplicated. */
  private persist(): Promise<void> {
    const file = this.options.file;
    if (!file) return Promise.resolve();
    const body = JSON.stringify([...this.entries]);
    this.writes = this.writes.then(async () => {
      await mkdir(path.dirname(file), { recursive: true });
      const temp = `${file}.${process.pid}.tmp`;
      const handle = await open(temp, "w", 0o600);
      try { await handle.writeFile(body); await handle.sync(); } finally { await handle.close(); }
      await rename(temp, file);
    }).catch(() => { console.error(JSON.stringify({ event: "agent_idempotency_persist_failed" })); });
    return this.writes;
  }
}
