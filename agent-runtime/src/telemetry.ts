import { createHash, randomUUID } from "node:crypto";
import { mkdirSync, chmodSync } from "node:fs";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import { Worker } from "node:worker_threads";
export interface AgentEnvelope {
  schema_version: 1;
  pi_version: string;
  event_id: string;
  run_id: string;
  session_id: string;
  sequence: number;
  version: string;
  collected_at: string;
  source_time: string | null;
  kind: string;
  trace_id?: string;
  pi_session_id?: string;
  payload: unknown;
  topic: "agent.events.v1" | "agent.content.v1";
}
export const CHUNK_BYTES = 256 * 1024;
const KAFKA_PRODUCE_BUDGET = 768 * 1024;
function redactSignedCredentials(text: string): string {
  // Recovery runs after a restart, when the registry of previously issued tokens is gone.
  return text.replace(/eyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{43}(?![A-Za-z0-9_-])/g, token => {
    const payload = token.split(".")[0];
    if (payload.length > 65536) return token;
    try {
      const claims = JSON.parse(Buffer.from(payload, "base64url").toString("utf8"));
      const gateway = typeof claims.runId === "string" && typeof claims.expiresAt === "number";
      const mcp = typeof claims.run === "string" && typeof claims.session === "string"
        && claims.mode === "github_read_only" && typeof claims.expiry === "number";
      return gateway || mcp ? "[REDACTED]" : token;
    } catch { return token; }
  });
}
const credentialSuffix = "(?:AUTHORIZATION|SECRET[_-]?ACCESS[_-]?KEY|SECRET[_-]?KEY|PRIVATE[_-]?KEY|ACCESS[_-]?KEY|API[_-]?KEY|ACCESS[_-]?TOKEN|REFRESH[_-]?TOKEN|GATEWAY[_-]?TOKEN|SIGNING[_-]?SECRET|PASSWORD|PASSPHRASE|SECRET|TOKEN)";
const credentialName = `[A-Za-z0-9_-]*${credentialSuffix}`;
const credentialKeyPattern = new RegExp(`(?:^|[_-])${credentialSuffix}$`, "i");
export function isCredentialKey(key: string): boolean { return credentialKeyPattern.test(key); }
function redactPrivateKeys(text: string): string {
  return text.replace(/-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----[\s\S]*?(?:-----END (?:[A-Z0-9]+ )?PRIVATE KEY-----|$)/g, "[REDACTED]");
}
function redactAssignments(text: string, completeOnly = false): string {
  // Opaque tool output may contain escaped quotes or multiword dotenv values.
  // Mask the remainder of the credential's line rather than guessing its grammar.
  const boundary = completeOnly ? "(?=\\r|\\n)" : "(?=\\r|\\n|$)";
  return text.replace(new RegExp(`(\\b${credentialName}["']?[\\t ]*[:=][\\t ]*)(?![\\t ]*\\[REDACTED\\](?=\\r|\\n|$))[^\\r\\n]+${boundary}`, "gi"), "$1[REDACTED]");
}
export function redact(value: unknown, secrets: string[] = []): unknown {
  if (typeof value === "string") {
    let result = value;
    for (const secret of secrets.filter(Boolean).sort((a, b) => b.length - a.length))
      result = result.split(secret).join("[REDACTED]");
    return redactPrivateKeys(redactAssignments(redactSignedCredentials(result.replace(/Bearer\s+[A-Za-z0-9._~+\/-]+/gi, "Bearer [REDACTED]"))));
  }
  if (Array.isArray(value))
    return value.map(v => redact(v, secrets));
  if (value && typeof value === "object")
    return Object.fromEntries(Object.entries(value).map(([key, v]) => [key,
      isCredentialKey(key) ? "[REDACTED]" : redact(v, secrets)]));
  return value;
}
export function redactBytes(value: Buffer, secrets: string[]): Buffer {
  // Latin-1 is a reversible byte mapping; invalid UTF-8 and NUL bytes are retained.
  let text = value.toString("latin1");
  for (const secret of secrets.filter(Boolean).sort((a, b) => b.length - a.length))
    text = text.split(Buffer.from(secret).toString("latin1")).join("[REDACTED]");
  return Buffer.from(redactPrivateKeys(redactAssignments(text.replace(/Bearer\s+[A-Za-z0-9._~+\/-]+/gi, "Bearer [REDACTED]"))), "latin1");
}
export class BinaryRedactor {
  private pending = "";
  private inBearer = false;
  private inAssignment = false;
  private inPrivateKey = false;
  private readonly keep: number;
  private readonly secrets: string[];
  constructor(secrets: string[]) {
    this.secrets = secrets.filter(Boolean).map(s => Buffer.from(s).toString("latin1")).sort((a, b) => b.length - a.length);
    this.keep = Math.max(256, ...this.secrets.map(s => s.length));
  }
  push(chunk: Buffer): Buffer {
    let input = chunk.toString("latin1");
    if (this.inPrivateKey) {
      const text = this.pending + input;
      const end = /-----END (?:[A-Z0-9]+ )?PRIVATE KEY-----/.exec(text);
      if (!end) { this.pending = text.slice(-128); return Buffer.alloc(0); }
      input = text.slice(end.index + end[0].length);
      this.pending = ""; this.inPrivateKey = false;
    }
    if (this.inAssignment) {
      const delimiter = input.search(/[\r\n]/);
      if (delimiter < 0) return Buffer.alloc(0);
      input = input.slice(delimiter);
      this.inAssignment = false;
    }
    if (this.inBearer) {
      const prefix = /^[A-Za-z0-9._~+\/-]*/.exec(input)![0].length;
      if (prefix === input.length)
        return Buffer.alloc(0);
      input = input.slice(prefix);
      this.inBearer = false;
    }
    let text = this.pending + input;
    for (const secret of this.secrets)
      text = text.split(secret).join("[REDACTED]");
    let begin: RegExpExecArray | null;
    while ((begin = /-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----/.exec(text))) {
      const after = text.slice(begin.index + begin[0].length);
      const end = /-----END (?:[A-Z0-9]+ )?PRIVATE KEY-----/.exec(after);
      if (!end) {
        this.inPrivateKey = true; this.pending = after.slice(-128);
        return redactBytes(Buffer.from(text.slice(0, begin.index) + "[REDACTED]", "latin1"), []);
      }
      text = text.slice(0, begin.index) + "[REDACTED]" + after.slice(end.index + end[0].length);
    }
    text = redactAssignments(text, true);
    const assignment = new RegExp(`\\b${credentialName}["']?[\\t ]*[:=][\\t ]*(?![\\t ]*\\[REDACTED\\](?=\\r|\\n|$))[^\\r\\n]+$`, "i").exec(text);
    if (assignment && assignment[0].length > this.keep) {
      const prefix = new RegExp(`^${credentialName}["']?[\\t ]*[:=][\\t ]*`, "i").exec(assignment[0])![0];
      this.inAssignment = true;
      this.pending = "";
      return Buffer.from(text.slice(0, assignment.index) + prefix + "[REDACTED]", "latin1");
    }
    // Do not prematurely redact a token whose final bytes have not arrived.
    text = text.replace(/Bearer\s+[A-Za-z0-9._~+\/-]+(?=[^A-Za-z0-9._~+\/-])/gi, "Bearer [REDACTED]");
    const trailing = /Bearer\s+[A-Za-z0-9._~+\/-]+$/i.exec(text);
    if (trailing && trailing[0].length > this.keep) {
      this.pending = "";
      this.inBearer = true;
      return Buffer.from(text.slice(0, trailing.index) + "Bearer [REDACTED]", "latin1");
    }
    const cut = Math.max(0, Math.min(text.length - this.keep, Math.min(trailing?.index ?? text.length, assignment?.index ?? text.length)));
    this.pending = text.slice(cut);
    return Buffer.from(text.slice(0, cut), "latin1");
  }
  finish(): Buffer {
    const result = this.inPrivateKey ? Buffer.alloc(0) : redactBytes(Buffer.from(this.pending, "latin1"), this.secrets.map(s => Buffer.from(s, "latin1").toString("utf8")));
    this.pending = "";
    this.inBearer = false;
    this.inAssignment = false;
    this.inPrivateKey = false;
    return result;
  }
}
export function contentRecords(base: Partial<AgentEnvelope>, name: string, content: string | Buffer): AgentEnvelope[] {
  const bytes = Buffer.isBuffer(content) ? content : Buffer.from(content);
  const contentId = createHash("sha256").update(`${base.event_id}:${name}`).digest("hex");
  const checksum = createHash("sha256").update(bytes).digest("hex");
  const count = Math.max(1, Math.ceil(bytes.length / CHUNK_BYTES));
  return Array.from({ length: count }, (_, i) => ({
    schema_version: 1, pi_version: "1.0.1", sequence: 0, version: (BigInt(Date.now()) * 1000n).toString(), collected_at: new Date().toISOString(), source_time: null,
    run_id: "", session_id: "", ...base, event_id: `${contentId}:${i}`, kind: "content", topic: "agent.content.v1",
    payload: {
      content_id: contentId, name, chunk_index: i, chunk_count: count, bytes: bytes.length, sha256: checksum,
      data: bytes.subarray(i * CHUNK_BYTES, (i + 1) * CHUNK_BYTES).toString("base64")
    },
  }));
}
/** Only the writer worker instantiates this in production. Tests use real SQLite. */
export class TelemetryStore {
  private readonly db: DatabaseSync;
  constructor(file: string, private readonly maxBytes = 4 * 1024 ** 3) {
    mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
    this.db = new DatabaseSync(file);
    chmodSync(file, 0o600);
    this.db.exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA busy_timeout=5000;");
    this.db.exec("CREATE TABLE IF NOT EXISTS feedback_receipts (id TEXT PRIMARY KEY, created INTEGER NOT NULL)");
    this.db.exec("CREATE TABLE IF NOT EXISTS outbox (id TEXT PRIMARY KEY, topic TEXT NOT NULL, run_id TEXT NOT NULL, body TEXT NOT NULL, bytes INTEGER NOT NULL, created INTEGER NOT NULL)");
    // Maintain quota totals in the same transaction; backlog size must not make each append scan every row.
    this.db.exec(`CREATE TABLE IF NOT EXISTS outbox_totals (id INTEGER PRIMARY KEY, depth INTEGER NOT NULL, bytes INTEGER NOT NULL);
      INSERT OR REPLACE INTO outbox_totals SELECT 1, count(*), coalesce(sum(bytes),0) FROM outbox;
      CREATE TRIGGER IF NOT EXISTS outbox_insert_total AFTER INSERT ON outbox BEGIN
        UPDATE outbox_totals SET depth=depth+1, bytes=bytes+NEW.bytes WHERE id=1;
      END;
      CREATE TRIGGER IF NOT EXISTS outbox_delete_total AFTER DELETE ON outbox BEGIN
        UPDATE outbox_totals SET depth=depth-1, bytes=bytes-OLD.bytes WHERE id=1;
      END;`);
  }
  append(record: AgentEnvelope): void { this.appendMany([record]); }
  appendMany(records: AgentEnvelope[]): void {
    this.db.exec("BEGIN IMMEDIATE");
    try {
      this.db.prepare("DELETE FROM feedback_receipts WHERE created < ? AND id NOT IN (SELECT id FROM outbox)").run(Date.now() - 90 * 86400000);
      const insert = this.db.prepare("INSERT OR IGNORE INTO outbox VALUES (?, ?, ?, ?, ?, ?)");
      for (const record of records) {
        if (this.db.prepare("SELECT id FROM outbox WHERE id=?").get(record.event_id))
          continue;
        if (record.kind === "feedback" && this.db.prepare("SELECT id FROM feedback_receipts WHERE id=?").get(record.event_id))
          continue;
        const body = JSON.stringify(record);
        if (this.health().bytes + Buffer.byteLength(body) > this.maxBytes)
          throw new Error("outbox_capacity_exceeded");
        insert.run(record.event_id, record.topic, record.run_id, body, Buffer.byteLength(body), Date.now());
        if (record.kind === "feedback")
          this.db.prepare("INSERT INTO feedback_receipts VALUES (?,?)").run(record.event_id, Date.now());
      }
      this.db.exec("COMMIT");
    }
    catch (error) {
      this.db.exec("ROLLBACK");
      throw error;
    }
  }
  pending(limit = 100): AgentEnvelope[] {
    return (this.db.prepare("SELECT body FROM outbox ORDER BY rowid LIMIT ?").all(limit) as Array<{
      body: string;
    }>).map(r => JSON.parse(r.body));
  }
  ack(ids: string[]): void {
    this.db.exec("BEGIN IMMEDIATE");
    try {
      const remove = this.db.prepare("DELETE FROM outbox WHERE id=?");
      for (const id of ids)
        remove.run(id);
      this.db.exec("COMMIT");
    }
    catch (error) {
      this.db.exec("ROLLBACK");
      throw error;
    }
  }
  health(): {
    depth: number;
    bytes: number;
    oldest_age_ms: number;
    capacity_bytes: number;
  } {
    const row = this.db.prepare("SELECT depth, bytes, (SELECT created FROM outbox ORDER BY rowid LIMIT 1) AS oldest FROM outbox_totals WHERE id=1").get() as {
      depth: number;
      bytes: number;
      oldest: number | null;
    };
    return { depth: row.depth, bytes: row.bytes, oldest_age_ms: row.oldest === null ? 0 : Date.now() - row.oldest, capacity_bytes: this.maxBytes };
  }
  close(): void { this.db.close(); }
}
export async function publishOutbox(store: TelemetryStore, producer: {
  send: (request: {
    topic: string;
    acks: -1;
    messages: Array<{
      key: string;
      value: string;
    }>;
  }) => Promise<unknown>;
}): Promise<void> {
  const records = store.pending();
  for (const topic of ["agent.content.v1", "agent.events.v1"] as const) {
    let messages: Array<{ key: string; value: string }> = [];
    let ids: string[] = [];
    let bytes = 0;
    const flush = async () => {
      if (!messages.length) return;
      await producer.send({ topic, acks: -1, messages });
      store.ack(ids);
      messages = []; ids = []; bytes = 0;
    };
    for (const record of records.filter(r => r.topic === topic)) {
      const value = JSON.stringify(record);
      const size = Buffer.byteLength(value);
      if (messages.length && bytes + size > KAFKA_PRODUCE_BUDGET)
        await flush();
      messages.push({ key: record.run_id, value });
      ids.push(record.event_id);
      bytes += size;
    }
    await flush();
  }
}
export class AgentTelemetry {
  private readonly worker: Worker;
  private readonly pending = new Map<string, {
    resolve: () => void;
    reject: (error: Error) => void;
  }>();
  private readonly runContext = new Map<string, {
    trace_id?: string;
    pi_session_id?: string;
  }>();
  private readonly issuedSecrets = new Map<string, string[]>();
  private readonly versions = new Map<string, bigint>();
  private feedbackVersion = 0n;
  private captureFailureReports = 0;
  private readonly sequences = new Map<string, number>();
  private readonly incomplete = new Map<string, Set<string>>();
  private healthState: Record<string, unknown> = { status: "starting" };
  constructor(options: {
    file: string;
    brokers: string[];
    secrets?: string[];
    maxBytes?: number;
  }, private readonly secrets = options.secrets ?? []) {
    this.worker = new Worker(new URL("./telemetry-worker.js", import.meta.url), { workerData: options });
    this.worker.on("message", message => {
      if (message.health)
        this.healthState = message.health;
      const pending = this.pending.get(message.id);
      if (pending) {
        this.pending.delete(message.id);
        message.error ? pending.reject(new Error(message.error)) : pending.resolve();
      }
    });
    const failed = () => {
      this.healthState = { status: "failed" };
      for (const pending of this.pending.values())
        pending.reject(new Error("telemetry_writer_failed"));
      this.pending.clear();
    };
    this.worker.on("error", failed);
    this.worker.on("exit", failed);
  }
  setTraceparent(runId: string, parent?: string): void {
    if (parent)
      this.runContext.set(runId, { ...this.runContext.get(runId), trace_id: parent.split("-")[1] });
  }
  setPiSessionId(runId: string, id: string): void { this.runContext.set(runId, { ...this.runContext.get(runId), pi_session_id: id }); }
  addSecret(secret: string, runId?: string): void {
    if (!secret) return;
    if (!runId) { if (!this.secrets.includes(secret)) this.secrets.push(secret); return; }
    const values = this.issuedSecrets.get(runId) ?? [];
    if (!values.includes(secret)) values.push(secret);
    this.issuedSecrets.set(runId, values);
  }
  private secretsFor(runId?: string): string[] { return [...this.secrets, ...(runId ? this.issuedSecrets.get(runId) ?? [] : [])]; }
  releaseRun(runId: string): void {
    if (this.incomplete.get(runId)?.size) this.captureFailureReports = (this.captureFailureReports ?? 0) + 1;
    this.issuedSecrets.delete(runId); this.runContext.delete(runId);
    this.sequences.delete(runId); this.versions.delete(runId); this.incomplete.delete(runId);
  }
  binaryRedactor(runId?: string): BinaryRedactor { return new BinaryRedactor(this.secretsFor(runId)); }
  redactBuffer(value: Buffer): Buffer { return redactBytes(value, this.secrets); }
  redactText(value: string): string { return redact(value, this.secrets) as string; }
  secretOverlap(): number { return Math.max(256, ...this.secrets.map(s => Buffer.byteLength(s))); }
  markIncomplete(runId: string, reason: string): void {
    const reasons = this.incomplete.get(runId) ?? new Set<string>();
    if (reasons.has(reason))
      return;
    reasons.add(reason);
    this.incomplete.set(runId, reasons);
    console.error(JSON.stringify({ event: "agent_capture_incomplete", run_id: runId, reason }));
  }
  completeness(runId: string): string[] { return [...(this.incomplete.get(runId) ?? [])]; }
  health(): Record<string, unknown> { return { ...this.healthState, incomplete_runs: this.incomplete.size, capture_failure_reports: this.captureFailureReports ?? 0 }; }
  async record(runId: string, sessionId: string, kind: string, payload: unknown, stableId?: string): Promise<boolean> {
    try {
      const now = BigInt(Date.now()) * 1000n;
      const feedback = kind === "feedback";
      const previous = feedback ? this.feedbackVersion ?? 0n : this.versions.get(runId) ?? 0n;
      const version = now > previous ? now : previous + 1n;
      // Feedback has a timestamp-based sequence after the captured run's counter.
      // One scalar suffices for ordering without restoring historical per-run maps.
      const sequence = feedback ? Number(version) : (this.sequences.get(runId) ?? 0) + 1;
      if (feedback) this.feedbackVersion = version;
      else { this.sequences.set(runId, sequence); this.versions.set(runId, version); }
      const envelope: AgentEnvelope = {
        schema_version: 1, pi_version: "1.0.1", event_id: stableId ?? randomUUID(), run_id: runId, session_id: sessionId,
        sequence, version: version.toString(), collected_at: new Date().toISOString(), source_time: sourceTime(payload), ...this.runContext.get(runId), kind, payload: redact(payload, this.secretsFor(runId)), topic: "agent.events.v1"
      };
      const records = [envelope];
      const encoded = JSON.stringify(envelope.payload);
      if (Buffer.byteLength(encoded) > CHUNK_BYTES) {
        const chunks = contentRecords(envelope, "payload", encoded);
        envelope.payload = {
          content_ref: (chunks[0].payload as {
            content_id: string;
          }).content_id
        };
        records.unshift(...chunks);
      }
      await this.send({ op: "append", records });
      return true;
    }
    catch {
      this.markIncomplete(runId, "outbox_write_failed");
      return false;
    }
  }
  private send(message: Record<string, unknown>): Promise<void> {
    const id = randomUUID();
    return new Promise((resolve, reject) => {
      if (this.pending.size >= 256) {
        reject(new Error("telemetry_queue_full"));
        return;
      }
      if (this.healthState.status === "failed") {
        reject(new Error("telemetry_writer_failed"));
        return;
      }
      const timeout = setTimeout(() => { this.pending.delete(id); reject(new Error("telemetry_writer_timeout")); }, 5000);
      this.pending.set(id, { resolve: () => { clearTimeout(timeout); resolve(); }, reject: error => { clearTimeout(timeout); reject(error); } });
      this.worker.postMessage({ ...message, id });
    });
  }
  async close(): Promise<void> {
    try {
      await this.send({ op: "close" });
    }
    finally {
      await this.worker.terminate();
    }
  }
}
function sourceTime(payload: unknown): string | null {
  const p = payload as {
    entry?: {
      timestamp?: unknown;
    };
    message?: {
      timestamp?: unknown;
    };
    timestamp?: unknown;
  } | null;
  const value = p?.entry?.timestamp ?? p?.message?.timestamp ?? p?.timestamp;
  if (typeof value !== "string" && typeof value !== "number")
    return null;
  const date = new Date(value);
  return Number.isFinite(date.getTime()) ? date.toISOString() : null;
}
