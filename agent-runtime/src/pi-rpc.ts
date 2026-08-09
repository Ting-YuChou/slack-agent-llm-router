export type PublicRunEvent =
  | { type: "turn" }
  | { type: "tool"; phase: "start" | "end"; tool_call_id: string; tool: string; input?: Record<string, unknown>; is_error?: boolean }
  | { type: "approval"; approval_id: string; title: string; detail: string; timeout_ms?: number }
  | { type: "answer"; text: string }
  | { type: "compaction"; phase: "start" | "end"; reason?: string; aborted?: boolean; will_retry?: boolean; error?: string }
  | { type: "settled" }
  | { type: "error"; code: string; message: string };

export class RpcProtocolError extends Error {}
export class RpcCommandError extends Error {
  constructor(public readonly code = "rpc_command_failed") {
    super("Pi RPC command failed");
  }
}

export interface RpcSessionState {
  sessionId: string;
  sessionFile?: string;
  autoCompactionEnabled: boolean;
  messageCount: number;
  pendingMessageCount: number;
  isStreaming: boolean;
  isCompacting: boolean;
  [key: string]: unknown;
}

export interface RpcSessionStats {
  sessionId: string;
  userMessages: number;
  assistantMessages: number;
  toolCalls: number;
  toolResults: number;
  totalMessages: number;
  tokens: Record<string, number>;
  cost: number;
  contextUsage?: Record<string, number | null>;
  [key: string]: unknown;
}

export interface RpcEntrySnapshot {
  entries: Array<Record<string, unknown>>;
  leafId: string | null;
}

export interface RpcTreeSnapshot {
  tree: Array<Record<string, unknown>>;
  leafId: string | null;
}

export class RpcJsonlDecoder {
  private buffer = "";

  push(chunk: Buffer | string): Array<Record<string, unknown>> {
    this.buffer += typeof chunk === "string" ? chunk : chunk.toString("utf8");
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
    return { type: "answer", text: delta.delta };
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
    return {
      type: "tool",
      phase: "start",
      tool_call_id: safeString(record.toolCallId),
      tool: safeString(record.toolName),
      input: sanitizeToolInput(record.args),
    };
  }
  if (type === "tool_execution_end") {
    return {
      type: "tool",
      phase: "end",
      tool_call_id: safeString(record.toolCallId),
      tool: safeString(record.toolName),
      is_error: record.isError === true,
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
  if (type === "compaction_start") {
    return {
      type: "compaction",
      phase: "start",
      ...(typeof record.reason === "string" ? { reason: record.reason } : {}),
    };
  }
  if (type === "compaction_end") {
    return {
      type: "compaction",
      phase: "end",
      ...(typeof record.reason === "string" ? { reason: record.reason } : {}),
      aborted: record.aborted === true,
      will_retry: record.willRetry === true,
      ...(typeof record.errorMessage === "string" ? { error: "Compaction failed" } : {}),
    };
  }
  if (type === "agent_settled") return { type: "settled" };
  if (type === "extension_error") {
    return { type: "error", code: "extension_error", message: "A trusted extension failed" };
  }
  return null;
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
  private readonly pending = new Map<string, {
    command: string;
    resolve: (value: unknown) => void;
    reject: (error: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  }>();
  private readonly ignoredResponseIds = new Set<string>();
  private nextRequestId = 1;

  constructor(options: { writeLine: (line: string) => void; onEvent: (event: PublicRunEvent) => void }) {
    this.writeLine = options.writeLine;
    this.onEvent = options.onEvent;
  }

  feed(chunk: Buffer | string): void {
    for (const record of this.decoder.push(chunk)) {
      if (record.type === "response") {
        this.handleResponse(record);
        continue;
      }
      const event = sanitizeRpcEvent(record);
      if (event) this.onEvent(event);
    }
  }

  prompt(id: string, message: string): void {
    this.ignoredResponseIds.add(id);
    this.send({ id, type: "prompt", message });
  }

  respondToUi(id: string, approved: boolean): void {
    this.send({ type: "extension_ui_response", id, confirmed: approved });
  }

  abort(id: string): void {
    this.ignoredResponseIds.add(id);
    this.send({ id, type: "abort" });
  }

  getState(timeoutMs = 5_000): Promise<RpcSessionState> {
    return this.request("get_state", {}, timeoutMs).then(validateSessionState);
  }

  getSessionStats(timeoutMs = 5_000): Promise<RpcSessionStats> {
    return this.request("get_session_stats", {}, timeoutMs).then(validateSessionStats);
  }

  getEntries(timeoutMs = 5_000): Promise<RpcEntrySnapshot> {
    return this.request("get_entries", {}, timeoutMs).then(validateEntrySnapshot);
  }

  getTree(timeoutMs = 5_000): Promise<RpcTreeSnapshot> {
    return this.request("get_tree", {}, timeoutMs).then(validateTreeSnapshot);
  }

  compact(customInstructions?: string, timeoutMs = 5 * 60_000): Promise<Record<string, unknown>> {
    return this.request("compact", customInstructions ? { customInstructions } : {}, timeoutMs)
      .then((data) => asRecord(data));
  }

  setAutoCompaction(enabled: boolean, timeoutMs = 5_000): Promise<void> {
    return this.request("set_auto_compaction", { enabled }, timeoutMs).then(() => undefined);
  }

  private request(command: string, fields: Record<string, unknown>, timeoutMs: number): Promise<unknown> {
    const id = `runtime-${this.nextRequestId++}`;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new RpcProtocolError(`Pi RPC ${command} timed out`));
      }, timeoutMs);
      this.pending.set(id, { command, resolve, reject, timer });
      this.send({ id, type: command, ...fields });
    });
  }

  private handleResponse(record: Record<string, unknown>): void {
    const id = typeof record.id === "string" ? record.id : "";
    if (this.ignoredResponseIds.delete(id)) return;
    const request = this.pending.get(id);
    if (!request) throw new RpcProtocolError("Pi RPC response has an unknown request id");
    this.pending.delete(id);
    clearTimeout(request.timer);
    if (record.command !== request.command) {
      request.reject(new RpcProtocolError("Pi RPC response command did not match its request"));
      return;
    }
    if (record.success !== true) {
      request.reject(new RpcCommandError());
      return;
    }
    request.resolve(record.data);
  }

  private send(record: Record<string, unknown>): void {
    this.writeLine(`${JSON.stringify(record)}\n`);
  }
}

function safeString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asRecord(value: unknown): Record<string, unknown> {
  if (!isRecord(value)) throw new RpcProtocolError("Pi RPC response data must be an object");
  return value;
}

function validateSessionState(value: unknown): RpcSessionState {
  const data = asRecord(value);
  requireString(data.sessionId, "sessionId");
  requireOptionalString(data.sessionFile, "sessionFile");
  requireBoolean(data.autoCompactionEnabled, "autoCompactionEnabled");
  requireCount(data.messageCount, "messageCount");
  requireCount(data.pendingMessageCount, "pendingMessageCount");
  requireBoolean(data.isStreaming, "isStreaming");
  requireBoolean(data.isCompacting, "isCompacting");
  return data as RpcSessionState;
}

function validateSessionStats(value: unknown): RpcSessionStats {
  const data = asRecord(value);
  requireString(data.sessionId, "sessionId");
  for (const field of [
    "userMessages",
    "assistantMessages",
    "toolCalls",
    "toolResults",
    "totalMessages",
  ]) {
    requireCount(data[field], field);
  }
  const tokens = asRecord(data.tokens);
  for (const key of ["input", "output", "cacheRead", "cacheWrite", "total"]) {
    requireCount(tokens[key], `tokens.${key}`);
  }
  requireFiniteNumber(data.cost, "cost");
  if (data.contextUsage !== undefined) {
    const usage = asRecord(data.contextUsage);
    for (const [key, amount] of Object.entries(usage)) {
      if (amount !== null) requireFiniteNumber(amount, `contextUsage.${key}`);
    }
  }
  return data as RpcSessionStats;
}

function validateEntrySnapshot(value: unknown): RpcEntrySnapshot {
  const data = asRecord(value);
  if (!Array.isArray(data.entries) || data.entries.length > 100_000) {
    throw new RpcProtocolError("Pi RPC entries must be a bounded array");
  }
  for (const entry of data.entries) validateEntry(entry);
  requireNullableString(data.leafId, "leafId");
  return data as unknown as RpcEntrySnapshot;
}

function validateTreeSnapshot(value: unknown): RpcTreeSnapshot {
  const data = asRecord(value);
  if (!Array.isArray(data.tree) || data.tree.length > 100_000) {
    throw new RpcProtocolError("Pi RPC tree must be a bounded array");
  }
  let nodes = 0;
  const visit = (value: unknown, depth: number) => {
    if (depth > 1_000 || ++nodes > 100_000) {
      throw new RpcProtocolError("Pi RPC tree exceeds its safety limit");
    }
    const node = asRecord(value);
    validateEntry(node.entry);
    if (!Array.isArray(node.children)) {
      throw new RpcProtocolError("Pi RPC tree children must be an array");
    }
    for (const child of node.children) visit(child, depth + 1);
  };
  for (const root of data.tree) visit(root, 0);
  requireNullableString(data.leafId, "leafId");
  return data as unknown as RpcTreeSnapshot;
}

function validateEntry(value: unknown): Record<string, unknown> {
  const entry = asRecord(value);
  requireString(entry.id, "entry.id");
  requireString(entry.type, "entry.type");
  requireNullableString(entry.parentId, "entry.parentId");
  return entry;
}

function requireString(value: unknown, field: string): asserts value is string {
  if (typeof value !== "string" || !value) {
    throw new RpcProtocolError(`Pi RPC ${field} must be a non-empty string`);
  }
}

function requireOptionalString(value: unknown, field: string): void {
  if (value !== undefined) requireString(value, field);
}

function requireNullableString(value: unknown, field: string): void {
  if (value !== null) requireString(value, field);
}

function requireBoolean(value: unknown, field: string): asserts value is boolean {
  if (typeof value !== "boolean") {
    throw new RpcProtocolError(`Pi RPC ${field} must be a boolean`);
  }
}

function requireCount(value: unknown, field: string): void {
  requireFiniteNumber(value, field);
  if ((value as number) < 0 || !Number.isInteger(value)) {
    throw new RpcProtocolError(`Pi RPC ${field} must be a non-negative integer`);
  }
}

function requireFiniteNumber(value: unknown, field: string): asserts value is number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new RpcProtocolError(`Pi RPC ${field} must be a finite number`);
  }
}
