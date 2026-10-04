export type AgentProvider = "openai" | "anthropic" | "opencode-go";
export type AgentModelApi = "openai-responses" | "anthropic-messages" | "openai-completions";
export type AgentCredentialEnv = "OPENAI_API_KEY" | "ANTHROPIC_API_KEY" | "OPENCODE_API_KEY";
export type AgentReasoningEffort = "low" | "medium" | "high" | "max";

export interface AgentModelSpec {
  ref: string;
  provider: AgentProvider;
  id: string;
  name: string;
  api: AgentModelApi;
  credentialEnv: AgentCredentialEnv;
  gatewayPath: string;
  upstreamUrl: string;
  reasoningEffort: "max";
  input: Array<"text" | "image">;
  cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
  contextWindow: number;
  maxTokens: number;
  thinkingLevelMap: Record<string, string | null>;
  compat?: Record<string, unknown>;
}

const MODELS: readonly AgentModelSpec[] = [
  {
    ref: "openai/gpt-5.6-luna",
    provider: "openai",
    id: "gpt-5.6-luna",
    name: "GPT-5.6 Luna",
    api: "openai-responses",
    credentialEnv: "OPENAI_API_KEY",
    gatewayPath: "/openai/v1/responses",
    upstreamUrl: "https://api.openai.com/v1/responses",
    reasoningEffort: "max",
    input: ["text", "image"],
    cost: { input: 0.2, output: 1.2, cacheRead: 0.02, cacheWrite: 0.25 },
    contextWindow: 1_050_000,
    maxTokens: 128_000,
    thinkingLevelMap: {
      off: "none", minimal: null, low: "low", medium: "medium",
      high: "high", xhigh: "xhigh", max: "max",
    },
    compat: { supportsStrictMode: true, supportsOpenAIGrammarTools: true },
  },
  {
    ref: "openai/gpt-5.6-sol",
    provider: "openai",
    id: "gpt-5.6-sol",
    name: "GPT-5.6 Sol",
    api: "openai-responses",
    credentialEnv: "OPENAI_API_KEY",
    gatewayPath: "/openai/v1/responses",
    upstreamUrl: "https://api.openai.com/v1/responses",
    reasoningEffort: "max",
    input: ["text", "image"],
    cost: { input: 4, output: 20, cacheRead: 0.4, cacheWrite: 5 },
    contextWindow: 1_050_000,
    maxTokens: 128_000,
    thinkingLevelMap: {
      off: "none", minimal: null, low: "low", medium: "medium",
      high: "high", xhigh: "xhigh", max: "max",
    },
    compat: { supportsStrictMode: true, supportsOpenAIGrammarTools: true },
  },
  {
    ref: "anthropic/claude-sonnet-4-6",
    provider: "anthropic",
    id: "claude-sonnet-4-6",
    name: "Claude Sonnet 4.6",
    api: "anthropic-messages",
    credentialEnv: "ANTHROPIC_API_KEY",
    gatewayPath: "/anthropic/v1/messages",
    upstreamUrl: "https://api.anthropic.com/v1/messages",
    reasoningEffort: "max",
    input: ["text", "image"],
    cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
    contextWindow: 1_000_000,
    maxTokens: 128_000,
    thinkingLevelMap: { max: "max" },
    compat: { forceAdaptiveThinking: true, supportsStrictTools: true },
  },
  {
    ref: "opencode-go/deepseek-v4-pro",
    provider: "opencode-go",
    id: "deepseek-v4-pro",
    name: "DeepSeek V4 Pro",
    api: "openai-completions",
    credentialEnv: "OPENCODE_API_KEY",
    gatewayPath: "/opencode-go/v1/chat/completions",
    upstreamUrl: "https://opencode.ai/zen/go/v1/chat/completions",
    reasoningEffort: "max",
    input: ["text"],
    cost: { input: 0.435, output: 0.87, cacheRead: 0.003625, cacheWrite: 0 },
    contextWindow: 1_000_000,
    maxTokens: 384_000,
    thinkingLevelMap: { minimal: null, low: null, medium: null, high: "high", max: "max" },
    compat: {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: true,
      maxTokensField: "max_tokens",
      requiresReasoningContentOnAssistantMessages: true,
      thinkingFormat: "deepseek",
    },
  },
] as const;

export const DEFAULT_AGENT_MODEL_REF = "openai/gpt-5.6-luna" as const;
export const JEV_CLASSIFIER_MODEL = {
  provider: "openrouter",
  id: "typesafe/jev-1.13",
  api: "typesafe-system-one",
  credentialEnv: "OPENROUTER_API_KEY",
  gatewayPath: "/openrouter/api/v1/systemone",
  upstreamUrl: "https://openrouter.ai/api/v1/systemone",
} as const;

export function listAgentModels(): AgentModelSpec[] {
  return MODELS.map((model) => ({ ...model }));
}

export function resolveAgentModel(ref?: string): AgentModelSpec {
  const requested = ref || DEFAULT_AGENT_MODEL_REF;
  const model = MODELS.find((candidate) => candidate.ref === requested);
  if (!model) throw new Error(`Agent model is not allowlisted: ${requested}`);
  return { ...model };
}

export function supportsAgentEffort(model: AgentModelSpec, effort: string): effort is AgentReasoningEffort {
  return ["low", "medium", "high", "max"].includes(effort) && model.thinkingLevelMap[effort] === effort;
}

export function gatewayProviderConfig(baseUrl: string, ref: string): {
  provider: AgentProvider;
  api: AgentModelApi;
  apiKey: `$${AgentCredentialEnv}`;
  baseUrl: string;
} {
  const model = resolveAgentModel(ref);
  const root = baseUrl.replace(/\/$/, "");
  const suffix = model.provider === "anthropic" ? "/anthropic" : `/${model.provider}/v1`;
  return {
    provider: model.provider,
    api: model.api,
    apiKey: `$${model.credentialEnv}`,
    baseUrl: `${root}${suffix}`,
  };
}

export function buildGatewayProviderRegistrations(baseUrl: string): Array<{
  provider: AgentProvider;
  api: AgentModelApi;
  apiKey: `$${AgentCredentialEnv}`;
  baseUrl: string;
  models: AgentModelSpec[];
}> {
  const grouped = new Map<AgentProvider, AgentModelSpec[]>();
  for (const model of listAgentModels()) {
    const models = grouped.get(model.provider) ?? [];
    models.push(model);
    grouped.set(model.provider, models);
  }
  return [...grouped.values()].map((models) => ({
    ...gatewayProviderConfig(baseUrl, models[0].ref),
    models,
  }));
}

export function buildClassifierGatewayProviderRegistration(baseUrl: string): {
  provider: "openrouter";
  apiKey: "$OPENROUTER_API_KEY";
  baseUrl: string;
} {
  return {
    provider: "openrouter",
    apiKey: "$OPENROUTER_API_KEY",
    baseUrl: `${baseUrl.replace(/\/$/, "")}/openrouter/api/v1`,
  };
}

// Backward-compatible aliases for the current default model.
const defaultModel = resolveAgentModel(DEFAULT_AGENT_MODEL_REF);
export const AGENT_MODEL_ID = defaultModel.id;
export const AGENT_REASONING_EFFORT = defaultModel.reasoningEffort;
export const AGENT_MODEL_CONFIG = {
  id: defaultModel.id,
  name: defaultModel.name,
  reasoning: true,
  thinkingLevelMap: defaultModel.thinkingLevelMap,
  input: defaultModel.input,
  cost: defaultModel.cost,
  contextWindow: defaultModel.contextWindow,
  maxTokens: defaultModel.maxTokens,
  compat: defaultModel.compat,
};
