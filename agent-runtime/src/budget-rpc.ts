import { hkdfSync } from "node:crypto";
import type { GatewayRole } from "./gateway-token.js";

/** The budget RPC bearer: its own key, so a leaked bearer cannot sign gateway tokens. */
export function deriveBudgetRpcSecret(signingSecret: string): string {
  if (!signingSecret) throw new Error("Model gateway signing secret is required");
  return Buffer.from(hkdfSync("sha256", signingSecret, "slack-pi-budget-rpc-v1", "budget-rpc", 32)).toString("base64url");
}

/** Why budget admission refused: the budget said no (spent, stopped or never opened), or it could not be asked or could not price the request. */
export type BudgetRefusal = "exhausted" | "unavailable";
const EXHAUSTED = new Set(["budget_required", "workflow_budget_exhausted", "discovery_budget_exhausted"]);
export function budgetRefusal(error: unknown): BudgetRefusal {
  return error instanceof Error && EXHAUSTED.has(error.message) ? "exhausted" : "unavailable";
}

export interface BudgetAdmissionLedger {
  reserve(workflow: string, id: string, amount: number, execution?: string): void | Promise<void>;
  settle(id: string, amount: number | null): void | Promise<void>;
  firstModel(execution: string, now?: number): number | undefined | Promise<number | undefined>;
  /** Whether the execution's run is still active; a gateway refuses the tokens of ended runs. */
  active?(execution: string): boolean | Promise<boolean>;
  /** A response of this execution was cut off at its output-token cap; the host records it on the execution's run. */
  truncated?(execution: string, role?: GatewayRole): void | Promise<void>;
}

/** SQLite has one owner on the host. Docker gateways never open its WAL files. */
export class BudgetRpcClient implements BudgetAdmissionLedger {
  constructor(private readonly base: string, private readonly secret: string) {
    const url = new URL(base);
    if (url.protocol !== 'http:' || !['127.0.0.1','localhost','host.docker.internal'].includes(url.hostname) || url.username || url.password || url.search || url.hash || !secret) throw new Error('Invalid budget RPC endpoint');
  }
  private async post(action: string, body: object): Promise<unknown> {
    const response = await fetch(`${this.base.replace(/\/$/,'')}/internal/budget/${action}`,{ method: 'POST',headers: { authorization: `Bearer ${this.secret}`,'content-type': 'application/json' },body: JSON.stringify(body),signal: AbortSignal.timeout(2000) });
    if (!response.ok) {
      // Only the host's 429 names an exhausted budget; a refused bearer, an unhealthy runtime or a failing store left it unasked.
      const refusal = response.status === 429 ? (await response.json().catch(() => undefined) as { error?: { cause?: unknown } } | undefined)?.error?.cause : undefined;
      throw new Error(refusal === 'exhausted' ? 'workflow_budget_exhausted' : 'workflow_budget_unavailable');
    }
    return (await response.json() as { result?: unknown }).result;
  }
  async reserve(workflow: string, id: string, amount: number, execution?: string) { await this.post('reserve',{ workflow,id,amount,execution }); }
  async settle(id: string, amount: number | null) { await this.post('settle',{ id,amount }); }
  async active(execution: string): Promise<boolean> { return await this.post('active',{ execution }) === true; }
  async truncated(execution: string, role?: GatewayRole) { await this.post('truncated',{ execution,role }); }
  async firstModel(execution: string, now = Date.now()) { const value = await this.post('first-model',{ execution,now }); return typeof value === 'number' ? value : undefined; }
}
