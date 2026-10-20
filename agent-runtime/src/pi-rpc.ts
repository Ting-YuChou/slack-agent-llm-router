import { StringDecoder } from "node:string_decoder";

import { parsePolicyDecision, POLICY_ENTRY, type PolicyDecision } from "./policy-decision.js";
import { BUDGET_STATUS_DIALOG } from "./budget-notice.js";
import { redact } from "./telemetry.js";

export type PublicRunEvent =
  | { type: "turn" }
  | { type: "usage"; cost_usd: number }
  | { type: "tool"; phase: "start" | "end"; tool_call_id: string; tool: string; input?: Record<string, unknown>; is_error?: boolean; result_bytes?: number }
  | { type: "approval"; approval_id: string; title: string; detail: string; timeout_ms?: number }
  | { type: "checks_request"; request_id: string; tool_call_id: string; checks?: string[] }
  | { type: "budget_request"; request_id: string }
  | { type: "answer_delta"; text: string }
  | { type: "answer"; text: string }
  | { type: "settled"; outcome: RunOutcome; limit?: LimitKind; stop_reason?: string; error_message?: string }
  | { type: "error"; code: string; message: string }
  | ({ type: "policy_decision" } & PolicyDecision);

/** How a Pi prompt ended; only `completed` may be delivered. */
export type RunOutcome = "completed" | "truncated" | "limit_reached" | "aborted" | "error";
export type LimitKind = "turns" | "tool_calls";

export class RpcProtocolError extends Error {}

/** The writer's run_checks tool and the extension UI dialog it asks the host through. */
export const RUN_CHECKS_TOOL = "run_checks";
export const RUN_CHECKS_DIALOG = "pi-agent/run_checks";
const DIALOG_METHODS = new Set(["select", "confirm", "input", "editor"]);
export type UiResponse = { value: string } | { cancelled: true };

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
  // The bridge classifies settles from the last assistant message; on its own a settle proves nothing.
  if (type === "agent_settled") return { type: "settled", outcome: "error", stop_reason: "no_response" };
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
  /** run_checks tool calls currently executing, and those that already asked the host once. */
  private readonly checkCalls = new Set<string>();
  private readonly checkRequests = new Set<string>();
  /** Whether this turn's one budget_status dialog was already passed to the host. */
  private budgetAsked = false;

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
      if (record.type === "entry_appended" && isRecord(record.entry) && record.entry.type === "custom" && record.entry.customType === POLICY_ENTRY) {
        const decision = parsePolicyDecision(record.entry.data);
        if (decision) this.onEvent({ type: "policy_decision", ...decision });
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
      if (record.type === "turn_start") this.budgetAsked = false;
      if (record.type === "tool_execution_start" && record.toolName === RUN_CHECKS_TOOL && typeof record.toolCallId === "string") this.checkCalls.add(record.toolCallId);
      if (record.type === "tool_execution_end" && typeof record.toolCallId === "string") this.checkCalls.delete(record.toolCallId);
      if (record.type === "extension_ui_request" && typeof record.method === "string" && DIALOG_METHODS.has(record.method) && record.method !== "confirm") {
        this.dialog(record);
        continue;
      }
      const event = sanitizeRpcEvent(record);
      if (event) this.onEvent(event);
    }
  }

  /**
   * The host answers only run_checks, for a run_checks tool call that is executing and has not asked
   * before, and budget_status, once per turn; any other dialog is cancelled so Pi never waits on it.
   */
  private dialog(record: Record<string, unknown>): void {
    const id = safeString(record.id);
    if (id && record.method === "input" && record.title === BUDGET_STATUS_DIALOG) {
      if (this.budgetAsked) { this.respondUi(id, { cancelled: true }); return; }
      this.budgetAsked = true;
      this.onEvent({ type: "budget_request", request_id: id });
      return;
    }
    const request = record.method === "input" && record.title === RUN_CHECKS_DIALOG ? parseChecksRequest(record.placeholder) : undefined;
    if (!id || !request || !this.checkCalls.has(request.tool_call_id) || this.checkRequests.has(request.tool_call_id)) {
      if (id) this.respondUi(id, { cancelled: true });
      return;
    }
    this.checkRequests.add(request.tool_call_id);
    this.onEvent({ type: "checks_request", request_id: id, ...request });
  }

  respondUi(id: string, response: UiResponse): void {
    this.send({ type: "extension_ui_response", id, ...response });
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

function parseChecksRequest(raw: unknown): { tool_call_id: string; checks?: string[] } | undefined {
  if (typeof raw !== "string" || raw.length > 8192) return undefined;
  let value: unknown;
  try { value = JSON.parse(raw); } catch { return undefined; }
  if (!isRecord(value) || typeof value.tool_call_id !== "string" || !value.tool_call_id || value.tool_call_id.length > 256) return undefined;
  const checks = value.checks;
  if (checks !== undefined && (!Array.isArray(checks) || checks.length > 20 || checks.some(name => typeof name !== "string" || !name || name.length > 80))) return undefined;
  return { tool_call_id: value.tool_call_id, ...(checks?.length ? { checks: checks as string[] } : {}) };
}

function safeString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
