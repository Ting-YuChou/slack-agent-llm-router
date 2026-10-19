import type { RouteDecision } from "./jev-router.js";

/** Turn and tool-call ceilings the writer's policy extension enforces inside the container. */
export interface AgentLimits { maxTurns: number; maxToolCalls: number }
export type LimitedRoute = "read_only" | "small_patch" | "complex";
export interface AgentLimitConfig { base: AgentLimits; routes: Record<LimitedRoute, AgentLimits> }

const LIMITED_ROUTES: readonly LimitedRoute[] = ["read_only", "small_patch", "complex"];
const LIMIT_FIELDS: ReadonlyArray<keyof AgentLimits> = ["maxTurns", "maxToolCalls"];

export const DEFAULT_ROUTE_LIMITS: Readonly<Record<LimitedRoute, AgentLimits>> = Object.freeze({
  read_only: { maxTurns: 12, maxToolCalls: 24 },
  small_patch: { maxTurns: 20, maxToolCalls: 40 },
  complex: { maxTurns: 30, maxToolCalls: 60 },
});

function validLimit(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 1 && value <= 1000;
}

export function parseAgentLimits(env: NodeJS.ProcessEnv): AgentLimits {
  const read = (key: string, fallback: number) => {
    if (env[key] === undefined || env[key] === "") return fallback;
    const value = Number(env[key]);
    if (!validLimit(value)) throw new Error(`${key} must be an integer between 1 and 1000`);
    return value;
  };
  return { maxTurns: read("PI_AGENT_MAX_TURNS", 20), maxToolCalls: read("PI_AGENT_MAX_TOOL_CALLS", 40) };
}

/** `PI_AGENT_ROUTE_LIMITS_JSON` overrides per route and per field, e.g. `{"complex":{"maxToolCalls":80}}`. */
export function parseRouteLimits(env: NodeJS.ProcessEnv): Record<LimitedRoute, AgentLimits> {
  const routes = Object.fromEntries(LIMITED_ROUTES.map((route) => [route, { ...DEFAULT_ROUTE_LIMITS[route] }])) as Record<LimitedRoute, AgentLimits>;
  const raw = env.PI_AGENT_ROUTE_LIMITS_JSON;
  if (raw === undefined || raw === "") return routes;
  const invalid = (detail: string) => new Error(`PI_AGENT_ROUTE_LIMITS_JSON ${detail}`);
  let parsed: unknown;
  try { parsed = JSON.parse(raw); } catch { throw invalid("must be valid JSON"); }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) throw invalid("must be an object keyed by route");
  for (const [route, override] of Object.entries(parsed)) {
    if (!LIMITED_ROUTES.includes(route as LimitedRoute)) throw invalid(`has unknown route ${route}; expected ${LIMITED_ROUTES.join(", ")}`);
    if (typeof override !== "object" || override === null || Array.isArray(override)) throw invalid(`${route} must be an object`);
    for (const [field, value] of Object.entries(override)) {
      if (!LIMIT_FIELDS.includes(field as keyof AgentLimits)) throw invalid(`${route} has unknown field ${field}`);
      if (!validLimit(value)) throw invalid(`${route}.${field} must be an integer between 1 and 1000`);
      routes[route as LimitedRoute][field as keyof AgentLimits] = value;
    }
  }
  return routes;
}

export function parseAgentLimitConfig(env: NodeJS.ProcessEnv): AgentLimitConfig {
  return { base: parseAgentLimits(env), routes: parseRouteLimits(env) };
}

/** Only a confident Jev decision scales the budget; fallback, shadow and explicit routes keep the base limits. */
export function limitsForRoute(route: RouteDecision | undefined, config: AgentLimitConfig): AgentLimits {
  const label = route?.source === "jev" ? route.label : undefined;
  const limits = label && LIMITED_ROUTES.includes(label as LimitedRoute) ? config.routes[label as LimitedRoute] : config.base;
  return { ...limits };
}
