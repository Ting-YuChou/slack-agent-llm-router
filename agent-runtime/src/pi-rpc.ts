import { StringDecoder } from "node:string_decoder";

import { redact } from "./telemetry.js";

export type PublicRunEvent =
  | { type: "turn" }
  | { type: "usage"; cost_usd: number }
  | { type: "tool"; phase: "start" | "end"; tool_call_id: string; tool: string; input?: Record<string, unknown>; is_error?: boolean; result_bytes?: number }
  | { type: "approval"; approval_id: string; title: string; detail: string; timeout_ms?: number }
  | { type: "answer_delta"; text: string }
  | { type: "answer"; text: string }
  | { type: "settled"; outcome: RunOutcome; limit?: LimitKind; stop_reason?: string; error_message?: string }
  | { type: "error"; code: string; message: string };

/** How a Pi prompt ended; only `completed` may be delivered. */
export type RunOutcome = "completed" | "truncated" | "limit_reached" | "aborted" | "error";
export type LimitKind = "turns" | "tool_calls";

export class RpcProtocolError extends Error {}

export class RpcJsonlDecoder {
  private buffer = "";
  private readonly utf8 = new StringDecoder("utf8");

  push(chunk: Buffer | string): Array<Record<string, unknown>> {
    this.buffer += typeof chunk === "string" ? chunk : this.utf8.write(chunk);
    const records: Array<Record<string, unknown>> = [];
    while (true) {
      const index = this.buffer.indexOf("\n");
      if (index < 0) break;
      let line = this.buffer.slice(0, index);
      this.buffer = this.buffer.slice(index + 1);
      if (line.endsWith("\r")) line = line.slice(0, -1);
      if (!line) continue;
      let parsed: unknown;
      try {
        parsed = JSON.parse(line);
      } catch {
        throw new RpcProtocolError("Pi RPC emitted invalid JSONL");
      }
      if (!isRecord(parsed)) throw new RpcProtocolError("Pi RPC record must be an object");
      records.push(parsed);
    }
    return records;
  }
}

export function sanitizeRpcEvent(record: Record<string, unknown>): PublicRunEvent | null {
  const type = record.type;
  if (type === "turn_start") return { type: "turn" };
  if (type === "message_update") {
    const delta = isRecord(record.assistantMessageEvent) ? record.assistantMessageEvent : {};
    const deltaType = delta.type;
    if (typeof deltaType === "string" && deltaType.startsWith("thinking")) return null;
    if (deltaType !== "text_delta" || typeof delta.delta !== "string") return null;
    return { type: "answer_delta", text: delta.delta };
  }
  if (type === "message_end") {
    const message = isRecord(record.message) ? record.message : {};
    if (message.role !== "assistant" || !Array.isArray(message.content)) return null;
    const text = message.content
      .filter((block): block is Record<string, unknown> => isRecord(block) && block.type === "text")
      .map((block) => typeof block.text === "string" ? block.text : "")
      .join("\n")
      .trim();
    return text ? { type: "answer", text } : null;
  }
  if (type === "tool_execution_start") {
    const tool = safeString(record.toolName);
    return {
      type: "tool",
      phase: "start",
      tool_call_id: safeString(record.toolCallId),
      tool,
      ...(tool.startsWith("mcp__") ? {} : { input: sanitizeToolInput(record.args) }),
    };
  }
  if (type === "tool_execution_end") {
    return {
      type: "tool",
      phase: "end",
      tool_call_id: safeString(record.toolCallId),
      tool: safeString(record.toolName),
      is_error: record.isError === true,
      result_bytes: byteSize(record.result),
    };
  }
  if (type === "extension_ui_request" && record.method === "confirm") {
    return {
      type: "approval",
      approval_id: safeString(record.id),
      title: safeString(record.title).slice(0, 300),
      detail: safeString(record.message).slice(0, 4_000),
      ...(typeof record.timeout === "number" ? { timeout_ms: record.timeout } : {}),
    };
  }
  if (type === "agent_settled") return { type: "settled", outcome: "completed" };
  if (type === "extension_error") {
    return { type: "error", code: "extension_error", message: "A trusted extension failed" };
  }
  return null;
}

function byteSize(value: unknown): number {
  try { return Buffer.byteLength(JSON.stringify(value ?? null)); } catch { return 0; }
}

function sanitizeToolInput(value: unknown): Record<string, unknown> | undefined {
  if (!isRecord(value)) return undefined;
  const safe: Record<string, unknown> = {};
  for (const key of ["path", "query", "pattern", "command"]) {
    if (typeof value[key] === "string") safe[key] = value[key].slice(0, 4_000);
  }
  if (Array.isArray(value.edits)) safe.edit_count = value.edits.length;
  if (typeof value.content === "string") safe.content_bytes = Buffer.byteLength(value.content);
  return safe;
}

export class PiRpcBridge {
  private readonly decoder = new RpcJsonlDecoder();
  private readonly writeLine: (line: string) => void;
  private readonly onEvent: (event: PublicRunEvent) => void;
  private readonly onCommand?: (record: Record<string, unknown>) => void;
  private readonly onRecord?: (record: Record<string, unknown>) => void;
  private readonly pending = new Map<string, { command: string; resolve: (response: Record<string, unknown>) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> }>();
  private sequence = 0;
  private ready = false;
  private promptId?: string;
  private lastAssistant?: { stopReason: string; errorMessage?: string };
  private limit?: LimitKind;
  private readonly onReady?: () => void;

  constructor(options: { writeLine: (line: string) => void; onEvent: (event: PublicRunEvent) => void; onRecord?: (record: Record<string, unknown>) => void; onCommand?: (record: Record<string, unknown>) => void; onReady?: () => void }) {
    this.writeLine = options.writeLine;
    this.onEvent = options.onEvent;
    this.onRecord = options.onRecord;
    this.onCommand = options.onCommand;
    this.onReady = options.onReady;
  }

  feed(chunk: Buffer | string): void {
    for (const record of this.decoder.push(chunk)) {
      try { this.onRecord?.(record); } catch { /* telemetry must not stop agent execution */ }
      if (record.type === "message_end" && isRecord(record.message)) {
        const usage = isRecord(record.message.usage) ? record.message.usage : {};
        const cost = isRecord(usage.cost) ? usage.cost : {};
        if (record.message.role === "assistant" && typeof cost.total === "number" && Number.isFinite(cost.total) && cost.total >= 0) {
          this.onEvent({ type: "usage", cost_usd: cost.total });
        }
      }
      if (record.type === "message_end" && isRecord(record.message) && record.message.role === "assistant") {
        const message = record.message;
        this.lastAssistant = { stopReason: safeString(message.stopReason) || "unknown", errorMessage: typeof message.errorMessage === "string" ? message.errorMessage : undefined };
      }
      if (record.type === "entry_appended" && isRecord(record.entry) && record.entry.type === "custom" && record.entry.customType === "pi_agent_limit") {
        const kind = isRecord(record.entry.data) ? record.entry.data.kind : undefined;
        if (kind === "turns" || kind === "tool_calls") this.limit = kind;
      }
      if (record.type === "agent_settled") {
        this.onEvent(this.settledEvent());
        continue;
      }
      if (record.type === "response" && record.command === "prompt" && record.id === this.promptId && record.success === false) {
        this.onEvent({ type: "error", code: "prompt_rejected", message: "Pi rejected the prompt before it started" });
        continue;
      }
      if (record.type === "response" && typeof record.id === "string") {
        const pending = this.pending.get(record.id);
        if (pending) {
          clearTimeout(pending.timer);
          this.pending.delete(record.id);
          if (record.command !== pending.command || record.success !== true) pending.reject(new RpcProtocolError(`Pi RPC ${pending.command} failed`));
          else {
            if (!this.ready) { this.ready = true; this.onReady?.(); }
            pending.resolve(record);
          }
          continue;
        }
      }
      const event = sanitizeRpcEvent(record);
      if (event) this.onEvent(event);
    }
  }

  prompt(id: string, message: string): void {
    this.promptId = id;
    this.lastAssistant = undefined;
    this.limit = undefined;
    this.send({ id, type: "prompt", message });
  }

  /** Pi emits agent_settled unconditionally; classify how the prompt actually ended. */
  private settledEvent(): Extract<PublicRunEvent, { type: "settled" }> {
    const last = this.lastAssistant;
    const stop_reason = last?.stopReason ?? "no_response";
    const error_message = last?.errorMessage ? String(redact(last.errorMessage)).slice(0, 500) : undefined;
    if (this.limit) return { type: "settled", outcome: "limit_reached", limit: this.limit, stop_reason };
    const outcome: RunOutcome = stop_reason === "stop" ? "completed" : stop_reason === "length" ? "truncated" : stop_reason === "aborted" ? "aborted" : "error";
    return { type: "settled", outcome, stop_reason, ...(error_message ? { error_message } : {}) };
  }

  async configureAndPrompt(id: string, message: string, provider: string, model: string, effort: string, beforePrompt?: () => Promise<void>): Promise<void> {
    await this.command("set_model", { provider, modelId: model });
    await this.command("set_thinking_level", { level: effort });
    const state = await this.command("get_state", {});
    const data = isRecord(state.data) ? state.data : {};
    const activeModel = isRecord(data.model) ? data.model : {};
    if (activeModel.provider !== provider || activeModel.id !== model || data.thinkingLevel !== effort) {
      throw new RpcProtocolError("Pi state mismatch after model configuration");
    }
    await beforePrompt?.();
    this.prompt(id, message);
  }

  async snapshot(): Promise<{ state: unknown; stats: unknown; entries: unknown }> {
    const [state, stats, entries] = await Promise.all([
      this.command("get_state", {}), this.command("get_session_stats", {}), this.command("get_entries", {}),
    ]);
    return { state: state.data, stats: stats.data, entries: entries.data };
  }

  failPending(): void {
    for (const [id, pending] of this.pending) {
      clearTimeout(pending.timer);
      pending.reject(new RpcProtocolError(`Pi RPC ${pending.command} interrupted`));
      this.pending.delete(id);
    }
  }

  private command(type: string, fields: Record<string, unknown>): Promise<Record<string, unknown>> {
    const id = `route-${++this.sequence}`;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new RpcProtocolError(`Pi RPC ${type} timed out`));
      }, 5_000);
      this.pending.set(id, { command: type, resolve, reject, timer });
      this.send({ id, type, ...fields });
    });
  }

  respondToUi(id: string, approved: boolean): void {
    this.send({ type: "extension_ui_response", id, confirmed: approved });
  }

  abort(id: string): void {
    this.send({ id, type: "abort" });
  }

  private send(record: Record<string, unknown>): void {
    try { this.onCommand?.(record); } catch { /* monitoring must not interrupt RPC */ }
    this.writeLine(`${JSON.stringify(record)}\n`);
  }
}

function safeString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
