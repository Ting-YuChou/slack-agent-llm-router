import { spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { mkdir, open, opendir, rename, unlink } from "node:fs/promises";
import { constants } from "node:fs";
import path from "node:path";
import type { PiRpcBridge } from "./pi-rpc.js";
import type { AgentTelemetry } from "./telemetry.js";
function record(value: unknown): Record<string, any> { return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, any> : {}; }
export function newSessionEntries(entries: unknown[], baseline: Set<string>): Record<string, any>[] {
  const seen = new Set(baseline);
  return entries.map(record).filter(entry => typeof entry.id === "string" && !seen.has(entry.id) && Boolean(seen.add(entry.id)));
}
export function isPiBashPath(value: unknown): value is string { return typeof value === "string" && /^\/tmp\/pi-bash-[a-f0-9]{16}\.log$/.test(value); }
// Open with O_NOFOLLOW, then validate the open descriptor. Never copy a model-supplied arbitrary file.
export const bashReadScript = `const fs=require('node:fs'); const p=process.argv[1];
if(!/^\\/tmp\\/pi-bash-[a-f0-9]{16}\\.log$/.test(p)) process.exit(2);
const fd=fs.openSync(p,fs.constants.O_RDONLY|fs.constants.O_NOFOLLOW|fs.constants.O_NONBLOCK); const s=fs.fstatSync(fd);
if(!s.isFile()||s.size>536870912) process.exit(3);
fs.createReadStream(null,{fd,autoClose:true,highWaterMark:131072}).pipe(process.stdout);`;
async function openRegular(file: string) {
  const handle = await open(file, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
  try { if (!(await handle.stat()).isFile()) throw new Error("not a regular file"); return handle; }
  catch (error) { await handle.close(); throw error; }
}
async function replaceCursor(file: string, content: string): Promise<void> {
  const temporary = `${file}.${randomUUID()}.tmp`;
  const handle = await open(temporary, "wx", 0o600);
  try { await handle.writeFile(content); await handle.sync(); await handle.close(); await rename(temporary, file); }
  finally { await handle.close(); await unlink(temporary).catch(() => {}); }
}
// Bound per-line allocation and total scanning; never buffer an untrusted session file.
async function* sessionLines(file: string, budget: {bytes: number; started: number}): AsyncGenerator<string> {
  const handle = await openRegular(file);
  let parts: Buffer[] = [];
  let length = 0;
  try {
    for await (const chunk of handle.createReadStream({highWaterMark: 65536, autoClose: false})) {
      budget.bytes += chunk.length;
      if (budget.bytes > 512 * 1024 * 1024 || Date.now() - budget.started > 10000) throw new Error("session recovery limit");
      let offset = 0;
      while (offset < chunk.length) {
        const newline = chunk.indexOf(10, offset);
        const end = newline < 0 ? chunk.length : newline;
        parts.push(chunk.subarray(offset, end));
        length += end - offset;
        if (length > 32 * 1024 * 1024) throw new Error("session line limit");
        if (newline < 0) break;
        yield Buffer.concat(parts, length).toString("utf8");
        parts = []; length = 0; offset = newline + 1;
      }
    }
    if (length) yield Buffer.concat(parts, length).toString("utf8");
  } finally { await handle.close(); }
}
export class PiRunCapture {
  private baseline = new Set<string>();
  private baselineKnown = false;
  private piSessionId = "unknown";
  private readonly commands = new Map<string, string>();
  private readonly bashFiles = new Map<string, string>();
  private readonly writes = new Set<Promise<void>>();
  private stopped = false;
  private get cursorPath(): string { return path.join(`${this.sessionPath}.capture`, `${this.runId}.json`); }
  constructor(private readonly telemetry: AgentTelemetry, readonly runId: string, readonly sessionId: string, private readonly sessionPath: string, private readonly container: string) { }
  private save(kind: string, payload: unknown, stableId?: string): Promise<void> {
    const write = this.telemetry.record(this.runId, this.sessionId, kind, payload, stableId).then(() => { }).catch(() => { this.telemetry.markIncomplete(this.runId, "capture_write_failed"); });
    this.writes.add(write);
    void write.finally(() => this.writes.delete(write));
    return write;
  }
  onCommand(value: Record<string, unknown>): void { if (this.stopped) return; void this.save("rpc_command", value); }
  onRecord(value: Record<string, unknown>): void {
    if (this.stopped) return;
    void this.save("rpc", value);
    if (value.type === "tool_execution_start" && typeof value.toolCallId === "string" && typeof record(value.args).command === "string")
      this.commands.set(value.toolCallId, record(value.args).command);
    const walk = (v: unknown, depth = 0): void => {
      if (depth > 128) { this.telemetry.markIncomplete(this.runId, "rpc_nesting_limit"); return; }
      if (!v || typeof v !== "object")
        return;
      for (const [key, child] of Object.entries(v)) {
        if ((key === "fullOutputPath" || key === "full_output_path") && isPiBashPath(child))
          this.bashFiles.set(child, typeof value.toolCallId === "string" ? value.toolCallId : this.bashFiles.get(child) ?? "unknown");
        else if (typeof child === "object")
          walk(child, depth + 1);
      }
    };
    walk(value);
  }
  async beforePrompt(bridge: PiRpcBridge): Promise<void> {
    try {
      const snapshot = await bridge.snapshot();
      if (this.stopped) return;
      this.piSessionId = record(snapshot.state).sessionId ?? "unknown";
      this.telemetry.setPiSessionId(this.runId, this.piSessionId);
      this.baseline = new Set((record(snapshot.entries).entries ?? []).map((e: any) => e.id));
      this.baselineKnown = true;
      await mkdir(path.dirname(this.cursorPath), {recursive: true, mode: 0o700});
      await replaceCursor(this.cursorPath, JSON.stringify({
        run_id: this.runId, session_id: this.sessionId,
        pi_session_id: this.piSessionId, baseline: [...this.baseline]
      }));
      if (this.stopped) { await unlink(this.cursorPath).catch(() => {}); return; }
      await this.save("session_stats", { phase: "start", pi_session_id: this.piSessionId, state: snapshot.state, stats: snapshot.stats });
    }
    catch {
      this.telemetry.markIncomplete(this.runId, "baseline_snapshot_failed");
    }
  }
  async finalize(bridge: PiRpcBridge): Promise<void> {
    try {
      const snapshot = await bridge.snapshot();
      if (this.stopped) return;
      await this.save("session_stats", { phase: "end", pi_session_id: this.piSessionId, state: snapshot.state, stats: snapshot.stats });
      await this.entries(record(snapshot.entries).entries ?? [], false);
      for (const [file, toolId] of this.bashFiles) {
        if (this.stopped)
          break;
        await this.captureBash(file, toolId);
      }
    }
    catch {
      if (!this.stopped) this.telemetry.markIncomplete(this.runId, "final_snapshot_failed");
    }
    await Promise.all([...this.writes]);
    if (this.stopped) return;
    await this.save("capture_status", { complete: this.telemetry.completeness(this.runId).length === 0, reasons: this.telemetry.completeness(this.runId) });
  }
  cancel(): void { this.stopped = true; }
  async recover(): Promise<void> {
    this.telemetry.markIncomplete(this.runId, "container_stopped_before_final_snapshot");
    try {
      const handle = await openRegular(this.cursorPath);
      let cursor: any;
      try {
        if ((await handle.stat()).size > 16 * 1024 * 1024) throw new Error("cursor limit");
        const data = Buffer.alloc(16 * 1024 * 1024 + 1);
        let length = 0;
        while (length < data.length) {
          const {bytesRead} = await handle.read(data, length, data.length - length, length);
          if (!bytesRead) break;
          length += bytesRead;
        }
        if (length === data.length) throw new Error("cursor limit");
        cursor = JSON.parse(data.subarray(0, length).toString("utf8"));
      } finally { await handle.close(); }
      if (cursor.run_id !== this.runId || cursor.session_id !== this.sessionId || typeof cursor.pi_session_id !== "string" || !Array.isArray(cursor.baseline) || !cursor.baseline.every((id: unknown) => typeof id === "string"))
        throw new Error("cursor mismatch");
      this.baseline = new Set(cursor.baseline);
      this.baselineKnown = true;
      this.piSessionId = cursor.pi_session_id;
      const started = Date.now();
      const directory = await opendir(this.sessionPath);
      const budget = {bytes: 0, started};
      let scanned = 0;
      for await (const item of directory) {
        if (++scanned > 128 || Date.now() - started > 10000) throw new Error("session scan limit");
        const name = item.name;
        if (!name.endsWith(".jsonl")) continue;
        try {
          let header = true;
          for await (const line of sessionLines(path.join(this.sessionPath, name), budget)) {
            if (Date.now() - started > 10000) throw new Error("session scan limit");
            if (header) {
              const metadata = record(JSON.parse(line));
              if (metadata.type !== "session" || metadata.id !== this.piSessionId) break;
              header = false;
            } else if (line.trim()) {
              await this.entries([JSON.parse(line)], true);
            }
          }
        } catch { this.telemetry.markIncomplete(this.runId, "session_recovery_failed"); }
      }
    }
    catch {
      this.telemetry.markIncomplete(this.runId, "session_recovery_failed");
    }
    await this.save("capture_status", { complete: false, reasons: this.telemetry.completeness(this.runId) });
  }
  private async entries(entries: unknown[], recovered: boolean): Promise<void> {
    if (!recovered && this.stopped) return;
    if (!this.baselineKnown) {
      await this.save("unassigned_entries", { entries });
      return;
    }
    for (const entry of entries.map(record))
      for (const block of record(entry.message).content ?? [])
        if (block.type === "toolCall" && typeof record(block.arguments).command === "string")
          this.commands.set(block.id, block.arguments.command);
    for (const entry of newSessionEntries(entries, this.baseline)) {
      if (!recovered && this.stopped) return;
      await this.save("session_entry", { pi_session_id: this.piSessionId, recovered, entry, command: this.commands.get(record(entry.message).toolCallId) }, `${this.sessionId}:${this.piSessionId}:${entry.id}`);
    }
  }
  private async captureBash(file: string, toolId: string): Promise<void> {
    const child = spawn("docker", ["exec", this.container, "node", "-e", bashReadScript, file], { env: { PATH: process.env.PATH } });
    let index = 0;
    let bytes = 0;
    const hash = createHash("sha256");
    const redactor = this.telemetry.binaryRedactor(this.runId);
    const emit = async (data: Buffer) => {
      if (!data.length || this.stopped)
        return;
      bytes += data.length;
      hash.update(data);
      await this.save("bash_output_chunk", { tool_call_id: toolId, path: file, chunk_index: index++, data: data.toString("base64") });
    };
    const timeout = setTimeout(() => child.kill("SIGKILL"), 10000);
    const exit = new Promise<number | null>((resolve, reject) => { child.once("error", reject); child.once("exit", resolve); });
    child.stderr.resume();
    try {
      for await (const chunk of child.stdout) {
        if (this.stopped) {
          child.kill("SIGKILL");
          break;
        }
        await emit(redactor.push(chunk));
      }
      await emit(redactor.finish());
      const code = await exit;
      if (code !== 0 || this.stopped)
        throw new Error("bash archive incomplete");
      await this.save("bash_output_manifest", { tool_call_id: toolId, path: file, command: this.commands.get(toolId), chunks: index, bytes, sha256: hash.digest("hex"), encoding: "binary", redacted: true });
    }
    catch {
      this.telemetry.markIncomplete(this.runId, "bash_output_recovery_failed");
    }
    finally {
      clearTimeout(timeout);
      child.kill("SIGKILL");
    }
  }
}
