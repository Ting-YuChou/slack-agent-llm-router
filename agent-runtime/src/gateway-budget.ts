import { randomUUID } from "node:crypto";
import type { BudgetAdmissionLedger } from './budget-rpc.js';
import { priceFor } from "./run-budget.js";
export interface ModelPrice { input: number; output: number }
/** Without a role-bound limit (classifier, legacy tokens) output stays at the historical conservative cap. */
const DEFAULT_OUTPUT_LIMIT = 4096;

export async function admitBudget(ledger: BudgetAdmissionLedger | undefined, prices: Record<string, ModelPrice>, workflow: string | undefined, model: string, raw: Buffer, execution?: string, outputLimit = DEFAULT_OUTPUT_LIMIT) {
  if (!workflow) throw new Error("budget_required");
  const price = priceFor(model, prices);
  if (!ledger || !price || !Number.isFinite(price.input) || price.input <= 0 || !Number.isFinite(price.output) || price.output < 0) throw new Error("workflow_budget_setup_missing");
  const body = JSON.parse(raw.toString());
  let effective = outputLimit;
  if (model.startsWith("openai/")) body.max_output_tokens = effective = Math.min(outputLimit, Number(body.max_output_tokens) || outputLimit);
  else if (!model.startsWith("openrouter/typesafe/jev")) body.max_tokens = effective = Math.min(outputLimit, Number(body.max_tokens) || outputLimit);
  const encoded = Buffer.from(JSON.stringify(body)); const id = randomUUID();
  // UTF-8 byte count overestimates input tokens; cache discounts are intentionally ignored.
  await ledger.reserve(workflow, id, (encoded.length * price.input + effective * price.output) / 1_000_000, execution);
  let pending = "", input: number | undefined, output: number | undefined, truncated = false;
  const decoder = new TextDecoder();
  function usage(value: any) {
    // OpenAI Responses: incomplete_details; Anthropic: stop_reason; chat completions: finish_reason.
    if (value?.response?.incomplete_details?.reason === "max_output_tokens" || value?.delta?.stop_reason === "max_tokens" || value?.stop_reason === "max_tokens" || value?.choices?.[0]?.finish_reason === "length") truncated = true;
    const candidate = value?.response?.usage ?? value?.message?.usage ?? value?.usage;
    if (!candidate) return;
    const i = candidate.input_tokens ?? candidate.prompt_tokens, o = candidate.output_tokens ?? candidate.completion_tokens;
    if (Number.isFinite(i) && i >= 0) input = i + (Number(candidate.cache_creation_input_tokens) || 0) + (Number(candidate.cache_read_input_tokens) || 0);
    if (Number.isFinite(o) && o >= 0) output = o;
  }
  return {
    body: encoded,
    data: (chunk: Buffer) => {
      pending += decoder.decode(chunk, { stream: true });
      const lines = pending.split('\n'); pending = lines.pop()!;
      for (const line of lines) { try { usage(JSON.parse(line.startsWith('data:') ? line.slice(5) : line)); } catch { /* partial/non-JSON SSE metadata */ } }
      if (pending.length > 1024 * 1024) pending = "";
    },
    finish: async () => {
      try { usage(JSON.parse(pending)); } catch { /* unknown usage keeps its full reservation */ }
      try { await ledger.settle(id, input === undefined || output === undefined ? null : (input * price.input + output * price.output) / 1_000_000); }
      catch { /* Unacknowledged settlement retains the original reservation. */ }
      return { truncated };
    },
  };
}
