import type { AgentModelSpec } from "./agent-model.js";

export type RequestPolicyResult = { ok: true; body: Record<string, unknown> } | { ok: false; field: string };

/**
 * Top-level fields Pi 1.0.1 sends per API (pi-ai buildParams), plus inert sampling knobs.
 * Anything else — hosted tools, server-side state, background jobs — could reach the network
 * from the provider's side, bypassing the container's internal-only network, so it is refused.
 */
const ALLOWED_FIELDS: Record<AgentModelSpec["api"], ReadonlySet<string>> = {
  "openai-responses": new Set(["model", "input", "instructions", "stream", "store", "prompt_cache_key", "prompt_cache_retention", "max_output_tokens", "reasoning", "include", "tools", "tool_choice", "parallel_tool_calls", "temperature", "top_p", "text", "truncation"]),
  "anthropic-messages": new Set(["model", "messages", "max_tokens", "stream", "system", "tools", "tool_choice", "thinking", "output_config", "temperature", "top_p", "top_k", "stop_sequences"]),
  "openai-completions": new Set(["model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens", "tools", "tool_choice", "thinking", "reasoning_effort", "temperature", "top_p", "stop"]),
};
const ALLOWED_INCLUDES = new Set(["reasoning.encrypted_content"]);
/** Every beta pi-ai's Anthropic provider can emit (anthropic-messages.js constants). */
const ALLOWED_ANTHROPIC_BETAS = new Set([
  "fine-grained-tool-streaming-2025-05-14", "interleaved-thinking-2025-05-14", "server-side-fallback-2026-07-01",
  "mid-conversation-output-config-2026-07-01", "thinking-binding-controls-2026-08-01", "inline-tools-2026-09-15",
]);

export function sanitizeAgentRequest(api: AgentModelSpec["api"], body: Record<string, unknown>): RequestPolicyResult {
  for (const key of Object.keys(body)) if (!ALLOWED_FIELDS[api].has(key)) return { ok: false, field: key };
  if (body.tools !== undefined) {
    if (!Array.isArray(body.tools)) return { ok: false, field: "tools" };
    for (const [index, tool] of body.tools.entries()) {
      if (!isRecord(tool) || !toolTypeAllowed(api, tool.type)) return { ok: false, field: `tools[${index}].type` };
    }
  }
  const content = api === "openai-responses" ? "input" : "messages";
  const remote = remoteSource(body[content], content);
  if (remote) return { ok: false, field: remote };
  if (api !== "openai-responses") return { ok: true, body };
  if (body.store !== undefined && body.store !== false) return { ok: false, field: "store" };
  if (body.include !== undefined && (!Array.isArray(body.include) || body.include.some((item) => !ALLOWED_INCLUDES.has(item)))) return { ok: false, field: "include" };
  return { ok: true, body: { ...body, store: false } };
}

/**
 * Providers fetch URL-sourced images and files themselves (Responses input_image/input_file,
 * Anthropic image/document `source.type: "url"`, chat-completions image_url), so a URL in message
 * content is server-side egress. Only inline base64 / data: content passes; returns the offending path.
 */
function remoteSource(value: unknown, path: string): string | undefined {
  if (Array.isArray(value)) {
    for (const [index, item] of value.entries()) {
      const found = remoteSource(item, `${path}[${index}]`);
      if (found) return found;
    }
    return undefined;
  }
  if (!isRecord(value)) return undefined;
  for (const [key, child] of Object.entries(value)) {
    const at = `${path}.${key}`;
    if ((key === "image_url" || key === "file_url") && !isDataUrl(isRecord(child) ? child.url : child)) return at;
    if (key === "source" && isRecord(child) && child.type !== "base64") return at;
    const found = remoteSource(child, at);
    if (found) return found;
  }
  return undefined;
}

function isDataUrl(value: unknown): boolean {
  return typeof value === "string" && value.slice(0, 5).toLowerCase() === "data:";
}

function toolTypeAllowed(api: AgentModelSpec["api"], type: unknown): boolean {
  if (api === "openai-responses") return type === "function" || type === "custom";
  if (api === "anthropic-messages") return type === undefined || type === "custom";
  return type === "function";
}

/** Keeps only betas Pi itself can request; others (mcp-client, web-fetch, …) enable server-side egress. */
export function allowedAnthropicBeta(value: string | undefined): string | undefined {
  const kept = (value ?? "").split(",").map((item) => item.trim()).filter((item) => ALLOWED_ANTHROPIC_BETAS.has(item));
  return kept.length ? kept.join(",") : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
