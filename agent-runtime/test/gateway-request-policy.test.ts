import assert from "node:assert/strict";
import { test } from "node:test";

import { allowedAnthropicBeta, sanitizeAgentRequest } from "../src/gateway-request-policy.js";

/** Shapes Pi 1.0.1 actually sends (pi-ai buildParams for each API). */
const piResponses = () => ({
  model: "gpt-5.6-luna", input: [{ role: "developer", content: [{ type: "input_text", text: "sys" }] }], stream: true, store: false,
  prompt_cache_key: "session-1", max_output_tokens: 128000, reasoning: { effort: "max", summary: "auto" }, include: ["reasoning.encrypted_content"],
  tools: [
    { type: "function", name: "read", description: "Read", parameters: { type: "object" }, strict: false },
    { type: "custom", name: "codemode", description: "Run code", format: { type: "grammar", syntax: "lark", definition: "start: /.+/" } },
  ],
});
const piAnthropic = () => ({
  model: "claude-sonnet-4-6", messages: [], max_tokens: 64000, stream: true, system: [{ type: "text", text: "sys", cache_control: { type: "ephemeral" } }],
  tools: [{ name: "read", description: "Read", eager_input_streaming: true, input_schema: { type: "object" } }],
  thinking: { type: "adaptive", display: "summarized" }, output_config: { effort: "max" },
});
const piCompletions = () => ({
  model: "deepseek-v4-pro", messages: [], stream: true, stream_options: { include_usage: true }, max_tokens: 32000,
  tools: [{ type: "function", function: { name: "read", description: "Read", parameters: { type: "object" }, strict: false } }],
  thinking: { type: "enabled" }, reasoning_effort: "max",
});

test("requests shaped like Pi's are forwarded unchanged", () => {
  for (const [api, body] of [["openai-responses", piResponses()], ["anthropic-messages", piAnthropic()], ["openai-completions", piCompletions()]] as const) {
    const result = sanitizeAgentRequest(api, structuredClone(body));
    assert.equal(result.ok, true, `${api}: ${JSON.stringify(result)}`);
    if (result.ok) assert.deepEqual(result.body, body);
  }
});

test("provider-hosted tools and server-side state are rejected with the offending field", () => {
  const cases: Array<[Parameters<typeof sanitizeAgentRequest>[0], (body: any) => void, string]> = [
    ...["web_search", "file_search", "code_interpreter", "mcp", "image_generation", "computer_use", "shell"].map((type) =>
      ["openai-responses", (b: any) => b.tools.push({ type, server_url: "https://evil.example" }), "tools[2].type"] as [any, any, string]),
    ["openai-responses", (b) => { b.store = true; }, "store"],
    ["openai-responses", (b) => { b.previous_response_id = "resp_1"; }, "previous_response_id"],
    ["openai-responses", (b) => { b.background = true; }, "background"],
    ["openai-responses", (b) => { b.conversation = "conv_1"; }, "conversation"],
    ["openai-responses", (b) => { b.prompt = { id: "pmpt_1" }; }, "prompt"],
    ["openai-responses", (b) => { b.metadata = { a: "b" }; }, "metadata"],
    ["openai-responses", (b) => { b.service_tier = "priority"; }, "service_tier"],
    ["openai-responses", (b) => { b.include.push("file_search_call.results"); }, "include"],
    ["openai-responses", (b) => { b.unexpected = 1; }, "unexpected"],
    ["anthropic-messages", (b) => { b.mcp_servers = [{ url: "https://evil.example" }]; }, "mcp_servers"],
    ["anthropic-messages", (b) => { b.container = "c1"; }, "container"],
    ["anthropic-messages", (b) => b.tools.push({ type: "web_fetch_20250910", name: "web_fetch" }), "tools[1].type"],
    ["openai-completions", (b) => b.tools.push({ type: "web_search" }), "tools[1].type"],
    ["openai-completions", (b) => { b.web_search_options = {}; }, "web_search_options"],
  ];
  for (const [api, mutate, field] of cases) {
    const body: any = api === "openai-responses" ? piResponses() : api === "anthropic-messages" ? piAnthropic() : piCompletions();
    mutate(body);
    assert.deepEqual(sanitizeAgentRequest(api, body), { ok: false, field }, `${api} ${field}`);
  }
});

test("URL-sourced message content is rejected so the provider never fetches on the agent's behalf", () => {
  const url = "https://evil.example/exfil?secret=1";
  const cases: Array<[Parameters<typeof sanitizeAgentRequest>[0], (body: any) => void, string]> = [
    ["openai-responses", (b) => b.input.push({ role: "user", content: [{ type: "input_image", image_url: url }] }), "input[1].content[0].image_url"],
    ["openai-responses", (b) => b.input.push({ role: "user", content: [{ type: "input_image", image_url: "http://evil.example/a.png" }] }), "input[1].content[0].image_url"],
    ["openai-responses", (b) => b.input.push({ role: "user", content: [{ type: "input_file", file_url: url }] }), "input[1].content[0].file_url"],
    ["openai-responses", (b) => b.input.push({ type: "function_call_output", call_id: "c1", output: [{ type: "input_image", image_url: url }] }), "input[1].output[0].image_url"],
    ["anthropic-messages", (b) => b.messages.push({ role: "user", content: [{ type: "image", source: { type: "url", url } }] }), "messages[0].content[0].source"],
    ["anthropic-messages", (b) => b.messages.push({ role: "user", content: [{ type: "document", source: { type: "url", url } }] }), "messages[0].content[0].source"],
    ["anthropic-messages", (b) => b.messages.push({ role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: [{ type: "image", source: { type: "url", url } }] }] }), "messages[0].content[0].content[0].source"],
    ["openai-completions", (b) => b.messages.push({ role: "user", content: [{ type: "image_url", image_url: { url } }] }), "messages[0].content[0].image_url"],
    ["openai-completions", (b) => b.messages.push({ role: "user", content: [{ type: "image_url", image_url: { url: "http://evil.example/a.png" } }] }), "messages[0].content[0].image_url"],
  ];
  for (const [api, mutate, field] of cases) {
    const body: any = api === "openai-responses" ? piResponses() : api === "anthropic-messages" ? piAnthropic() : piCompletions();
    mutate(body);
    assert.deepEqual(sanitizeAgentRequest(api, body), { ok: false, field }, `${api} ${field}`);
  }
});

test("inline base64 and data: URL content still passes", () => {
  const data = "data:image/png;base64,iVBORw0KGgo=";
  const responses: any = piResponses();
  responses.input.push(
    { role: "user", content: [{ type: "input_text", text: "look" }, { type: "input_image", image_url: data, detail: "auto" }, { type: "input_file", filename: "a.pdf", file_data: "data:application/pdf;base64,JVBERi0=" }] },
    { type: "function_call_output", call_id: "c1", output: [{ type: "input_image", image_url: data }] },
  );
  const anthropic: any = piAnthropic();
  anthropic.messages.push({ role: "user", content: [
    { type: "image", source: { type: "base64", media_type: "image/png", data: "iVBORw0KGgo=" } },
    { type: "document", source: { type: "base64", media_type: "application/pdf", data: "JVBERi0=" } },
    { type: "tool_result", tool_use_id: "t1", content: [{ type: "image", source: { type: "base64", media_type: "image/png", data: "iVBORw0KGgo=" } }] },
  ] });
  const completions: any = piCompletions();
  completions.messages.push({ role: "user", content: [{ type: "text", text: "look" }, { type: "image_url", image_url: { url: data } }] });
  for (const [api, body] of [["openai-responses", responses], ["anthropic-messages", anthropic], ["openai-completions", completions]] as const) {
    const result = sanitizeAgentRequest(api, structuredClone(body));
    assert.equal(result.ok, true, `${api}: ${JSON.stringify(result)}`);
  }
});

test("responses requests are never stored server-side", () => {
  const body: any = piResponses();
  delete body.store;
  const result = sanitizeAgentRequest("openai-responses", body);
  assert.ok(result.ok);
  if (result.ok) assert.equal(result.body.store, false);
});

test("only anthropic-beta values Pi can send are forwarded", () => {
  assert.equal(allowedAnthropicBeta(undefined), undefined);
  assert.equal(allowedAnthropicBeta("fine-grained-tool-streaming-2025-05-14, interleaved-thinking-2025-05-14"), "fine-grained-tool-streaming-2025-05-14,interleaved-thinking-2025-05-14");
  assert.equal(allowedAnthropicBeta("mcp-client-2025-04-04,interleaved-thinking-2025-05-14"), "interleaved-thinking-2025-05-14");
  assert.equal(allowedAnthropicBeta("mcp-client-2025-04-04"), undefined);
});
