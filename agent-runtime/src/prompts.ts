import { PROMPTS, PROMPTS_VERSION } from "./prompts.generated.js";

export { PROMPTS_VERSION };
export type PromptId = keyof typeof PROMPTS;
export type PromptVars = Record<string, string | number>;
export const PROMPT_IDS = Object.keys(PROMPTS) as PromptId[];

/** Stable telemetry reference for one prompt: its id and the first 12 hex digits of its body's sha256. */
export function promptRef(id: PromptId): string {
  return `${id}@${prompt(id).hash}`;
}

/** Fills every {{name}} exactly once per occurrence; values are inserted verbatim and never re-expanded. */
export function renderPrompt(id: PromptId, vars: PromptVars = {}): string {
  const { body, vars: names } = prompt(id);
  const missing = names.filter(name => !Object.hasOwn(vars, name));
  if (missing.length) throw new Error(`Prompt ${id} is missing variables: ${missing.join(", ")}`);
  const unused = Object.keys(vars).filter(name => !(names as readonly string[]).includes(name));
  if (unused.length) throw new Error(`Prompt ${id} got unused variables: ${unused.join(", ")}`);
  return body.replace(/\{\{([a-z][a-z0-9_]*)\}\}/g, (_match, name: string) => String(vars[name]));
}

function prompt(id: PromptId): { body: string; vars: readonly string[]; hash: string } {
  if (!Object.hasOwn(PROMPTS, id)) throw new Error(`Unknown prompt: ${String(id)}`);
  return PROMPTS[id];
}

const WRITER_PROMPTS: readonly PromptId[] = ["policy-scope", "checks", "jev-classifier", "budget-notice", "budget-final", "repair", "scope-repair", "diff-limit-repair", "ci-repair", "investigation-results"];
export const ROLE_PROMPTS: Readonly<Record<string, readonly PromptId[]>> = {
  implementation: WRITER_PROMPTS, discussion: ["discussion"], grader: ["grader", "grader-scope"], planner: ["planner"],
  code_explorer: ["investigator"], ci_investigator: ["investigator"], test_analyst: ["investigator"],
};

/** Telemetry fields naming the prompt set and the prompts an execution role is driven by. */
export function promptTelemetry(role: string): { prompts_version: string; prompts: string[] } {
  return { prompts_version: PROMPTS_VERSION, prompts: (Object.hasOwn(ROLE_PROMPTS, role) ? ROLE_PROMPTS[role]! : []).map(promptRef) };
}
