import { ROOT_CONTEXT, trace, SpanStatusCode, type Span, type Tracer } from "@opentelemetry/api";
import { BasicTracerProvider, BatchSpanProcessor, AlwaysOnSampler, type SpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { LoggerProvider, BatchLogRecordProcessor } from "@opentelemetry/sdk-logs";
import { OTLPLogExporter } from "@opentelemetry/exporter-logs-otlp-http";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { POLICY_ENTRY, parsePolicyDecision } from "./policy-decision.js";
import type { RouteDecision } from "./jev-router.js";
export function validTraceparent(value: unknown): value is string {
  return typeof value === "string" && /^00-(?!0{32})[a-f0-9]{32}-(?!0{16})[a-f0-9]{16}-0[01]$/.test(value);
}
/** Harness decisions that explain a run on its trace: the phase it entered, a repair it started, a limit it hit. */
export type TraceSignal =
  | { type: "phase"; phase: string }
  | { type: "repair"; attempt: number; reason: string }
  | { type: "limit"; kind: string; code: string; action: "fail" | "repair" };

/** What the run span says about how the run was routed and how its writer ended. */
export interface RunDescription { model?: string; reasoning_effort?: string; routing?: RouteDecision; execution_outcome?: string }

const FAILED = ["failed", "timed_out", "interrupted"];

export class AgentTracing {
  static fromEnvironment(serviceName = "pi-agent-runtime"): AgentTracing | undefined {
    if (process.env.PI_AGENT_OTEL_ENABLED !== "true")
      return undefined;
    try {
      return new AgentTracing(undefined, serviceName);
    }
    catch {
      console.error(JSON.stringify({ event: "agent_otel_initialization_failed", service: serviceName }));
      return undefined;
    }
  }
  private readonly provider: BasicTracerProvider;
  private readonly tracer: Tracer;
  private readonly logs?: LoggerProvider;
  private readonly runs = new Map<string, {
    root: Span;
    children: Map<string, Span>;
    /** The open phase and repair spans; each ends when the next one starts or with the run. */
    phase?: Span;
    repair?: Span;
    repairs?: number;
  }>();
  constructor(processors?: SpanProcessor[], serviceName = "pi-agent-runtime") {
    this.provider = new BasicTracerProvider({
      sampler: new AlwaysOnSampler(), spanProcessors: processors ?? [new BatchSpanProcessor(new OTLPTraceExporter())],
      resource: resourceFromAttributes({ "service.name": serviceName, "pi.version": "1.0.1" })
    });
    if (!processors)
      this.logs = new LoggerProvider({ resource: resourceFromAttributes({ "service.name": serviceName }), processors: [new BatchLogRecordProcessor({ exporter: new OTLPLogExporter() })] });
    this.tracer = this.provider.getTracer("pi-agent-observability", "1.0.0");
  }
  startRun(runId: string, sessionId: string, parentRunId?: string, persistedParent?: string): void {
    if (this.runs.has(runId))
      return;
    const parent = parentRunId ? this.runs.get(parentRunId)?.root : undefined;
    let context = parent ? trace.setSpan(ROOT_CONTEXT,parent) : ROOT_CONTEXT;
    if (!parent && validTraceparent(persistedParent)) {
      const [,traceId,spanId,flags] = persistedParent.split('-');
      context = trace.setSpanContext(context,{ traceId,spanId,traceFlags: parseInt(flags,16),isRemote: true });
    }
    this.runs.set(runId, { root: this.tracer.startSpan("agent.run", { attributes: { "agent.run_id": runId, "agent.session_id": sessionId, ...(parentRunId ? { "agent.workflow_run_id": parentRunId } : {}) } }, context), children: new Map() });
  }
  describeRun(runId: string, run: RunDescription): void {
    const root = this.runs.get(runId)?.root;
    if (!root)
      return;
    if (run.model)
      root.setAttribute("gen_ai.request.model", run.model);
    if (run.reasoning_effort)
      root.setAttribute("gen_ai.request.reasoning_effort", run.reasoning_effort);
    const route = run.routing;
    if (route) {
      root.setAttribute("agent.route.source", route.source);
      if (route.label)
        root.setAttribute("agent.route.label", route.label);
      if (route.probability !== undefined)
        root.setAttribute("agent.route.probability", route.probability);
    }
    if (run.execution_outcome)
      root.setAttribute("agent.outcome", run.execution_outcome);
  }
  /** Phases and repair attempts become child spans of the run; limit hits become events on it. */
  signal(runId: string, signal: TraceSignal): void {
    const run = this.runs.get(runId);
    if (!run)
      return;
    const child = (name: string, attributes: Record<string, string | number>) =>
      this.tracer.startSpan(name, { attributes: { "agent.run_id": runId, ...attributes } }, trace.setSpan(ROOT_CONTEXT, run.root));
    if (signal.type === "phase") {
      run.phase?.end();
      run.phase = child("agent.phase", { "agent.phase": signal.phase });
    }
    else if (signal.type === "repair") {
      run.repair?.end();
      run.repair = child("agent.repair", { "agent.repair.attempt": signal.attempt, "agent.repair.reason": signal.reason });
      run.repairs = (run.repairs ?? 0) + 1;
      run.root.setAttributes({ "agent.repair.reason": signal.reason, "agent.repair.count": run.repairs });
    }
    else
      run.root.addEvent("agent.limit", { "agent.limit.kind": signal.kind, "agent.limit.code": signal.code, "agent.limit.action": signal.action });
  }
  traceparent(runId: string): string | undefined {
    const span = this.runs.get(runId)?.root.spanContext();
    return span ? `00-${span.traceId}-${span.spanId}-${span.traceFlags === 1 ? "01" : "00"}` : undefined;
  }
  lifecycle(runId: string, name: string, phase: "start" | "end", failed = false): void {
    const run = this.runs.get(runId);
    if (!run)
      return;
    if (phase === "start") {
      const old = run.children.get(name);
      if (old) {
        old.setAttribute("capture.incomplete", true);
        old.end();
      }
      run.children.set(name, this.tracer.startSpan(`agent.${name}`, {attributes: {"agent.run_id": runId}}, trace.setSpan(ROOT_CONTEXT, run.root)));
    }
    else {
      const span = run.children.get(name);
      if (span) {
        if (failed)
          span.setStatus({ code: SpanStatusCode.ERROR });
        span.end();
        run.children.delete(name);
      }
    }
  }
  rpc(runId: string, record: Record<string, any>): void {
    const run = this.runs.get(runId);
    if (!run)
      return;
    const type = record.type as string;
    let key: string | undefined;
    let name: string | undefined;
    let start = false;
    let end = false;
    if (type === "entry_appended") {
      this.policyEvent(run.root, record.entry);
      return;
    }
    if (type === "tool_execution_start" || type === "tool_execution_end") {
      key = `tool:${record.toolCallId}`;
      name = "agent.tool";
      start = type.endsWith("start");
      end = !start;
    }
    else if (type === "turn_start" || type === "turn_end") {
      key = "turn";
      name = "agent.turn";
      start = type.endsWith("start");
      end = !start;
    }
    else if (type === "extension_ui_request" && record.method === "confirm") {
      key = `approval:${record.id}`;
      name = "agent.approval_wait";
      start = true;
    }
    else if (type === "auto_compaction_start" || type === "auto_compaction_end" || type === "auto_retry_start" || type === "auto_retry_end") {
      key = type.replace(/_(start|end)$/, "");
      name = `agent.${key}`;
      start = type.endsWith("start");
      end = !start;
    }
    if (!key || !name)
      return;
    if (start) {
      const previous = run.children.get(key);
      if (previous) {
        previous.setAttribute("capture.incomplete", true);
        previous.end();
      }
      const attributes: Record<string, string> = {"agent.run_id": runId};
      if (record.toolCallId)
        attributes['tool.call_id'] = record.toolCallId;
      if (record.toolName)
        attributes['tool.name'] = record.toolName;
      run.children.set(key, this.tracer.startSpan(name, { attributes }, trace.setSpan(ROOT_CONTEXT, run.root)));
    }
    if (end) {
      const span = run.children.get(key);
      if (span) {
        if (record.isError || record.error)
          span.setStatus({ code: SpanStatusCode.ERROR });
        span.end();
        run.children.delete(key);
      }
    }
  }
  /** A policy extension's decision entry, checked like the host checks it before counting. */
  private policyEvent(root: Span, entry: unknown): void {
    if (!entry || typeof entry !== "object")
      return;
    const { type, customType, data } = entry as Record<string, unknown>;
    if (type !== "custom" || customType !== POLICY_ENTRY)
      return;
    const decision = parsePolicyDecision(data);
    if (!decision)
      return;
    const attributes: Record<string, string | number | boolean> = {};
    for (const [key, value] of Object.entries(decision))
      attributes[`agent.policy.${key}`] = value;
    root.addEvent("agent.policy_decision", attributes);
  }
  approval(runId: string, id: string, approved: boolean): void {
    const children = this.runs.get(runId)?.children;
    const span = children?.get(`approval:${id}`);
    span?.setAttribute("approval.approved", approved);
    span?.end();
    children?.delete(`approval:${id}`);
  }
  firstModel(runId: string, parent: string | undefined, sent: number): void {
    if (!validTraceparent(parent)) return;
    const [, traceId, spanId, flags] = parent.split("-");
    const context = trace.setSpanContext(ROOT_CONTEXT, { traceId, spanId, traceFlags: parseInt(flags,16), isRemote: true });
    this.tracer.startSpan("agent.prompt_to_first_model_request", { startTime: new Date(sent), attributes: { "agent.run_id": runId } }, context).end();
  }
  gateway(runId: string, parent: string | undefined, provider: string, model: string): Span {
    let context = ROOT_CONTEXT;
    if (validTraceparent(parent)) {
      const [, traceId, spanId, flags] = parent.split("-");
      context = trace.setSpanContext(context, { traceId, spanId, traceFlags: parseInt(flags, 16), isRemote: true });
    }
    return this.tracer.startSpan("gen_ai.request", { attributes: { "agent.run_id": runId, "gen_ai.provider.name": provider, "gen_ai.request.model": model } }, context);
  }
  /** Marks a run's span failed (a workflow attempt's own span, keyed by its execution) with what explains it. */
  fail(runId: string, attributes: Record<string, string | number>): void {
    const root = this.runs.get(runId)?.root;
    root?.setAttributes(attributes);
    root?.setStatus({ code: SpanStatusCode.ERROR });
  }
  /** `error` is the run's public failure: its code and, when the harness knows it, the cause behind it. */
  endRun(runId: string, status: string, error?: { code: string; cause_code?: string }): void {
    const run = this.runs.get(runId);
    if (!run)
      return;
    for (const span of run.children.values()) {
      span.setAttribute("capture.incomplete", true);
      span.end();
    }
    const failed = FAILED.includes(status);
    for (const span of [run.phase, run.repair]) {
      if (failed)
        span?.setStatus({ code: SpanStatusCode.ERROR });
      span?.end();
    }
    run.root.setAttribute("agent.status", status);
    if (error)
      run.root.setAttributes({ "agent.error.code": error.code, ...(error.cause_code ? { "agent.error.cause_code": error.cause_code } : {}) });
    if (failed)
      run.root.setStatus({ code: SpanStatusCode.ERROR });
    run.root.end();
    this.runs.delete(runId);
  }
  log(runId: string, event: string, attributes: Record<string, string | number | boolean> = {}): void {
    const root = this.runs.get(runId)?.root;
    this.logs?.getLogger("pi-agent-observability").emit({
      body: event, severityText: "INFO", attributes: { "agent.run_id": runId, ...attributes },
      context: root ? trace.setSpan(ROOT_CONTEXT, root) : ROOT_CONTEXT
    });
  }
  async flush(): Promise<void> { await Promise.all([this.provider.forceFlush(), this.logs?.forceFlush()]); }
  async close(): Promise<void> { await Promise.all([this.provider.shutdown(), this.logs?.shutdown()]); }
}
