import { execFile } from "node:child_process";
import { appendFileSync } from "node:fs";
import http from "node:http";
import type { AddressInfo } from "node:net";
import { promisify } from "node:util";

import { waitFor } from "./scenario.js";

const exec = promisify(execFile);

/** Checks a request must pass before its turn is served; a miss answers 400 and does not consume the turn. */
export interface RequestExpectation { model?: string; effort?: string; includes?: string[]; excludes?: string[] }
export interface TurnUsage { input_tokens: number; output_tokens: number }
interface TurnBase { expect?: RequestExpectation; usage?: TurnUsage }
export type TurnScript = TurnBase & (
  | { kind: "function_call"; name: string; arguments: string | Record<string, unknown> }
  | { kind: "text"; text: string }
  | { kind: "incomplete"; reason: "max_output_tokens"; text?: string }
  | { kind: "anthropic_text"; text: string }
);
type SseEvent = { type: string } & Record<string, unknown>;
/** One provider call: the SSE events it streams, the wire protocol, and what the request must look like. */
export interface ScriptedTurn { protocol: "openai" | "anthropic"; events: SseEvent[]; expect?: RequestExpectation }
export interface RecordedRequest { method: string; url: string; headers: Record<string, unknown>; body: unknown }
export interface FakeProvider { url: string; port: number; readonly calls: number; readonly requests: RecordedRequest[]; close(): Promise<void> }
interface ProviderConfig { turns: ScriptedTurn[]; recordPath?: string }
interface ProviderState { calls: number; requests: RecordedRequest[] }

/** Builds the SSE payloads for each scripted call; ids are numbered by turn (`fc_1`/`call_1`, `msg_2`, `resp_2`). */
export function scriptedTurns(turns: TurnScript[]): ScriptedTurn[] {
  return turns.map((turn, index) => {
    const n = index + 1;
    const input = turn.usage?.input_tokens ?? 1, output = turn.usage?.output_tokens ?? 1;
    const expect = turn.expect ? { expect: turn.expect } : {};
    if (turn.kind === "anthropic_text") {
      return { protocol: "anthropic", ...expect, events: [
        { type: "message_start", message: { id: `msg_${n}`, type: "message", role: "assistant", content: [], model: "claude-sonnet-4-6", stop_reason: null, usage: { input_tokens: input, output_tokens: output } } },
        { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
        { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: turn.text } },
        { type: "content_block_stop", index: 0 },
        { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: output } },
        { type: "message_stop" },
      ] };
    }
    const usage = { input_tokens: input, output_tokens: output, total_tokens: input + output, input_tokens_details: { cached_tokens: 0 } };
    const id = `resp_${n}`;
    if (turn.kind === "function_call") {
      const args = typeof turn.arguments === "string" ? turn.arguments : JSON.stringify(turn.arguments);
      const item = { type: "function_call", id: `fc_${n}`, call_id: `call_${n}`, name: turn.name, arguments: args, status: "completed" };
      return { protocol: "openai", ...expect, events: [
        { type: "response.created", response: { id } },
        { type: "response.output_item.added", output_index: 0, item },
        { type: "response.function_call_arguments.done", output_index: 0, arguments: args },
        { type: "response.output_item.done", output_index: 0, item },
        { type: "response.completed", response: { id, status: "completed", output: [item], usage } },
      ] };
    }
    const incomplete = turn.kind === "incomplete";
    const text = turn.text ?? "";
    const status = incomplete ? "incomplete" : "completed";
    const item = { type: "message", id: `msg_${n}`, role: "assistant", status, content: [{ type: "output_text", text, annotations: [] }], phase: "final_answer" };
    return { protocol: "openai", ...expect, events: [
      { type: "response.created", response: { id } },
      { type: "response.output_item.added", output_index: 0, item },
      { type: "response.output_text.delta", output_index: 0, content_index: 0, delta: text },
      { type: "response.output_item.done", output_index: 0, item },
      incomplete
        ? { type: "response.incomplete", response: { id, status, incomplete_details: { reason: turn.reason }, output: [item], usage } }
        : { type: "response.completed", response: { id, status, output: [item], usage } },
    ] };
  });
}

/**
 * The provider itself. It must stay self-contained (no imports or outer bindings) because
 * `fakeGatewayScript` ships its source to `node -e`; everything it needs arrives as arguments.
 */
function createProvider(httpModule: Pick<typeof http, "createServer">, append: (file: string, data: string) => void, config: ProviderConfig): { server: http.Server; state: ProviderState } {
  const state: ProviderState = { calls: 0, requests: [] };
  const reject = (res: http.ServerResponse, status: number, message: string) => { res.writeHead(status, { "content-type": "text/plain" }); res.end(message); };
  const server = httpModule.createServer((req, res) => {
    let raw = "";
    req.on("data", chunk => { raw += chunk; });
    req.on("end", () => {
      const url = req.url ?? "/";
      if (req.method === "GET" && url === "/healthz") { res.writeHead(200, { "content-type": "text/plain" }); res.end("ok"); return; }
      let body: Record<string, unknown> = {};
      try { body = raw ? JSON.parse(raw) : {}; } catch { reject(res, 400, "request body is not JSON"); return; }
      const record = { method: req.method ?? "", url, headers: req.headers as Record<string, unknown>, body };
      state.requests.push(record);
      if (config.recordPath) append(config.recordPath, `${JSON.stringify(record)}\n`);
      const turn = config.turns[state.calls];
      if (!turn) { reject(res, 500, `no scripted turn ${state.calls + 1}`); return; }
      const anthropicRoute = url.startsWith("/anthropic/");
      if (anthropicRoute !== (turn.protocol === "anthropic")) { reject(res, 400, `turn ${state.calls + 1} expects a ${turn.protocol} request, got ${url}`); return; }
      const expect = turn.expect ?? {};
      if (expect.model !== undefined && body.model !== expect.model) { reject(res, 400, `unexpected model ${String(body.model)}, expected ${expect.model}`); return; }
      const effort = (body.reasoning as { effort?: unknown } | undefined)?.effort;
      if (expect.effort !== undefined && effort !== expect.effort) { reject(res, 400, `unexpected reasoning effort ${String(effort)}, expected ${expect.effort}`); return; }
      for (const needle of expect.includes ?? []) if (!raw.includes(needle)) { reject(res, 400, `request is missing ${needle}`); return; }
      for (const needle of expect.excludes ?? []) if (raw.includes(needle)) { reject(res, 400, `request must not contain ${needle}`); return; }
      state.calls++;
      res.writeHead(200, { "content-type": "text/event-stream" });
      if (turn.protocol === "anthropic") {
        for (const event of turn.events) res.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
        res.end();
      } else {
        for (const event of turn.events) res.write(`data: ${JSON.stringify(event)}\n\n`);
        res.end("data: [DONE]\n\n");
      }
    });
  });
  return { server, state };
}

/** Starts the provider in this process on an ephemeral 127.0.0.1 port. */
export async function startFakeProvider(turns: ScriptedTurn[], options: { recordPath?: string } = {}): Promise<FakeProvider> {
  const { server, state } = createProvider(http, appendFileSync, { turns, ...options });
  await new Promise<void>((resolve, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", resolve); });
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}`, port,
    get calls() { return state.calls; },
    get requests() { return state.requests; },
    close: () => new Promise<void>((resolve, reject) => server.close(error => (error ? reject(error) : resolve()))),
  };
}

/**
 * The same provider as a self-contained script for `node -e` (inside a node container),
 * listening on 0.0.0.0:`port` (default 8080) and logging `fake-provider listening on <port>`.
 */
export function fakeGatewayScript(turns: ScriptedTurn[], options: { port?: number; recordPath?: string } = {}): string {
  const config: ProviderConfig = { turns, ...(options.recordPath ? { recordPath: options.recordPath } : {}) };
  return [
    `const createProvider = ${createProvider.toString()};`,
    `const { server } = createProvider(require("node:http"), require("node:fs").appendFileSync, ${JSON.stringify(config)});`,
    `server.listen(${options.port ?? 8080}, "0.0.0.0", () => console.log("fake-provider listening on " + server.address().port));`,
  ].join("\n");
}

/** Polls `/healthz` from inside `container` (its network is internal, so the host cannot reach it) until it answers. */
export async function waitForContainerHealthz(container: string, port = 8080, timeoutMs = 30_000): Promise<void> {
  const probe = `fetch("http://127.0.0.1:${port}/healthz").then(r => process.exit(r.ok ? 0 : 1), () => process.exit(1))`;
  const healthy = () => exec("docker", ["exec", container, "node", "-e", probe]).then(() => true, () => false);
  await waitFor(healthy, ok => ok, () => `${container} did not answer /healthz on port ${port} within ${timeoutMs} ms`, timeoutMs);
}
