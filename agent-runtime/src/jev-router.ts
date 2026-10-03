import type { AgentReasoningEffort } from "./agent-model.js";

export type JevMode = "off" | "shadow" | "on";
export interface RouteDecision {
  modelRef: string;
  effort: AgentReasoningEffort;
  source: string;
  label?: string;
  probability?: number;
  recommendedModelRef?: string;
  recommendedEffort?: AgentReasoningEffort;
  jevLatencyMs?: number;
  jevCostUsd?: number;
  jevRequestId?: string;
  reason?: string;
}

export class JevRouter {
  constructor(private readonly options: { mode: JevMode; apiKey?: string; fetchFn?: typeof fetch; timeoutMs?: number }) {
    if (options.mode !== "off" && !options.apiKey) throw new Error("OPENROUTER_API_KEY is required when Jev is enabled");
  }

  async route(routingText?: string): Promise<RouteDecision> {
    const fallback: RouteDecision = { modelRef: "openai/gpt-5.6-luna", effort: "max", source: "fallback" };
    if (this.options.mode === "off") return { ...fallback, source: "off" };
    const text = routingText?.trim();
    if (!text || text.length > 4_000) return { ...fallback, reason: "missing_or_oversized_text" };
    const started = performance.now();
    try {
      const response = await (this.options.fetchFn ?? fetch)("https://openrouter.ai/api/alpha/decisions", {
        method: "POST",
        headers: { authorization: `Bearer ${this.options.apiKey}`, "content-type": "application/json" },
        body: JSON.stringify({
          model: "typesafe/jev-1.13",
          state: text,
          questions: { task: {
            type: "choice",
            instructions: "Classify the coding agent's current task by the work it requests. Use unclear if scope is uncertain.",
            criteria: {
              read_only: "Answer, inspect, explain, or review without modifying files.",
              small_patch: "One clearly scoped local fix with a known acceptance check, usually one file.",
              complex: "Multi-file design, refactor, migration, or difficult debugging needing stronger reasoning.",
              unclear: "The requested work or scope is ambiguous, or the task mixes categories.",
            },
          } },
        }),
        signal: AbortSignal.timeout(this.options.timeoutMs ?? 1_500),
      });
      if (!response.ok) throw new Error("jev_http_error");
      const body: unknown = await response.json();
      if (!isRecord(body) || !isRecord(body.answers) || !isRecord(body.answers.task)) throw new Error("jev_invalid_response");
      const answer = body.answers.task;
      const label = answer.choice;
      const probabilities = answer.probabilities;
      if (answer.type !== "choice" || typeof label !== "string" || !isRecord(probabilities) ||
          !["read_only", "small_patch", "complex", "unclear"].includes(label) ||
          typeof probabilities[label] !== "number" || probabilities[label] < 0 || probabilities[label] > 1) {
        throw new Error("jev_invalid_response");
      }
      const probability = probabilities[label];
      const recommended = probability >= 0.9 ? routeFor(label) : fallback;
      const metadata: RouteDecision = {
        ...(this.options.mode === "on" ? recommended : fallback),
        source: this.options.mode === "shadow" ? "shadow" : probability >= 0.9 && label !== "unclear" ? "jev" : "fallback",
        label,
        probability,
        recommendedModelRef: recommended.modelRef,
        recommendedEffort: recommended.effort,
        jevLatencyMs: Math.round(performance.now() - started),
        ...(typeof body.id === "string" ? { jevRequestId: body.id } : {}),
        ...(isRecord(body.usage) && typeof body.usage.cost === "number" ? { jevCostUsd: body.usage.cost } : {}),
      };
      return metadata;
    } catch (error) {
      return { ...fallback, jevLatencyMs: Math.round(performance.now() - started), reason: error instanceof Error && error.name === "TimeoutError" ? "timeout" : "unavailable_or_invalid" };
    }
  }
}

function routeFor(label: string): RouteDecision {
  if (label === "read_only") return { modelRef: "openai/gpt-5.6-luna", effort: "low", source: "jev" };
  if (label === "small_patch") return { modelRef: "openai/gpt-5.6-luna", effort: "medium", source: "jev" };
  if (label === "complex") return { modelRef: "openai/gpt-5.6-sol", effort: "high", source: "jev" };
  return { modelRef: "openai/gpt-5.6-luna", effort: "max", source: "fallback" };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
