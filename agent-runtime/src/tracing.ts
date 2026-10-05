import { ROOT_CONTEXT, trace, SpanStatusCode, type Span, type Tracer } from "@opentelemetry/api";
import { BasicTracerProvider, BatchSpanProcessor, AlwaysOnSampler, type SpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { LoggerProvider, BatchLogRecordProcessor } from "@opentelemetry/sdk-logs";
import { OTLPLogExporter } from "@opentelemetry/exporter-logs-otlp-http";
import { resourceFromAttributes } from "@opentelemetry/resources";
export function validTraceparent(value: unknown): value is string {
  return typeof value === "string" && /^00-(?!0{32})[a-f0-9]{32}-(?!0{16})[a-f0-9]{16}-0[01]$/.test(value);
}
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
  startRun(runId: string, sessionId: string): void {
    if (this.runs.has(runId))
      return;
    this.runs.set(runId, { root: this.tracer.startSpan("agent.run", { attributes: { "agent.run_id": runId, "agent.session_id": sessionId } }, ROOT_CONTEXT), children: new Map() });
  }
  describeRun(runId: string, model?: string, effort?: string): void {
    const root = this.runs.get(runId)?.root;
    if (model)
      root?.setAttribute("gen_ai.request.model", model);
    if (effort)
      root?.setAttribute("gen_ai.request.reasoning_effort", effort);
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
  approval(runId: string, id: string, approved: boolean): void {
    const children = this.runs.get(runId)?.children;
    const span = children?.get(`approval:${id}`);
    span?.setAttribute("approval.approved", approved);
    span?.end();
    children?.delete(`approval:${id}`);
  }
  gateway(runId: string, parent: string | undefined, provider: string, model: string): Span {
    let context = ROOT_CONTEXT;
    if (validTraceparent(parent)) {
      const [, traceId, spanId, flags] = parent.split("-");
      context = trace.setSpanContext(context, { traceId, spanId, traceFlags: parseInt(flags, 16), isRemote: true });
    }
    return this.tracer.startSpan("gen_ai.request", { attributes: { "agent.run_id": runId, "gen_ai.provider.name": provider, "gen_ai.request.model": model } }, context);
  }
  endRun(runId: string, status: string): void {
    const run = this.runs.get(runId);
    if (!run)
      return;
    for (const span of run.children.values()) {
      span.setAttribute("capture.incomplete", true);
      span.end();
    }
    run.root.setAttribute("agent.status", status);
    if (["failed", "timed_out", "interrupted"].includes(status))
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
