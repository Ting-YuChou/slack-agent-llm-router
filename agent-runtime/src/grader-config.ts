import { supportsAgentEffort, type AgentModelSpec } from "./agent-model.js";
import type { RouteDecision } from "./jev-router.js";

/** One source of truth for the grader's model, effort and time limit. */
export interface GraderConfig { route: RouteDecision; timeoutMs: number }

export const DEFAULT_GRADER_MODEL = "openai/gpt-5.6-sol";

const MIN_TIMEOUT_MS = 30_000;
const MAX_TIMEOUT_MS = 30 * 60_000;

/** Reads PI_AGENT_GRADER_MODEL / _EFFORT / _TIMEOUT_MS and refuses a grader the runtime cannot reach. */
export function parseGraderConfig(env: NodeJS.ProcessEnv, configured: AgentModelSpec[]): GraderConfig {
  const modelRef = env.PI_AGENT_GRADER_MODEL?.trim() || DEFAULT_GRADER_MODEL;
  const model = configured.find(candidate => candidate.ref === modelRef);
  if (!model) throw new Error(`PI_AGENT_GRADER_MODEL ${modelRef} is not an allowlisted model with a configured provider`);
  const effort = env.PI_AGENT_GRADER_EFFORT?.trim() || "high";
  if (!supportsAgentEffort(model, effort)) throw new Error(`PI_AGENT_GRADER_EFFORT ${effort} is not supported by ${modelRef}`);
  const rawTimeout = env.PI_AGENT_GRADER_TIMEOUT_MS?.trim();
  const timeoutMs = rawTimeout === undefined || rawTimeout === "" ? 120_000 : Number(rawTimeout);
  if (!Number.isInteger(timeoutMs) || timeoutMs < MIN_TIMEOUT_MS || timeoutMs > MAX_TIMEOUT_MS) {
    throw new Error(`PI_AGENT_GRADER_TIMEOUT_MS must be an integer between ${MIN_TIMEOUT_MS} and ${MAX_TIMEOUT_MS}`);
  }
  return { route: { modelRef, effort, source: "grader" }, timeoutMs };
}
