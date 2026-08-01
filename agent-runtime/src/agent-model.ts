export const AGENT_MODEL_ID = "gpt-5.6-luna" as const;
export const AGENT_REASONING_EFFORT = "max" as const;

export const AGENT_MODEL_CONFIG = {
  id: AGENT_MODEL_ID,
  name: "GPT-5.6 Luna",
  reasoning: true,
  thinkingLevelMap: {
    off: "none",
    minimal: null,
    low: "low",
    medium: "medium",
    high: "high",
    xhigh: "xhigh",
    max: "max",
  },
  input: ["text", "image"] as Array<"text" | "image">,
  cost: { input: 1, output: 6, cacheRead: 0.1, cacheWrite: 1.25 },
  contextWindow: 1_050_000,
  maxTokens: 128_000,
  compat: {
    supportsStrictMode: true,
    supportsOpenAIGrammarTools: true,
  },
};
