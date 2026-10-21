import { hkdfSync } from "node:crypto";

/** The budget RPC bearer: its own key, so a leaked bearer cannot sign gateway tokens. */
export function deriveBudgetRpcSecret(signingSecret: string): string {
  if (!signingSecret) throw new Error("Model gateway signing secret is required");
  return Buffer.from(hkdfSync("sha256", signingSecret, "slack-pi-budget-rpc-v1", "budget-rpc", 32)).toString("base64url");
}

export interface BudgetAdmissionLedger {
  reserve(workflow: string, id: string, amount: number, execution?: string): void | Promise<void>;
  settle(id: string, amount: number | null): void | Promise<void>;
  firstModel(execution: string, now?: number): number | undefined | Promise<number | undefined>;
  /** Whether the execution's run is still active; a gateway refuses the tokens of ended runs. */
  active?(execution: string): boolean | Promise<boolean>;
}

/** SQLite has one owner on the host. Docker gateways never open its WAL files. */
export class BudgetRpcClient implements BudgetAdmissionLedger {
  constructor(private readonly base: string, private readonly secret: string) {
    const url = new URL(base);
    if (url.protocol !== 'http:' || !['127.0.0.1','localhost','host.docker.internal'].includes(url.hostname) || url.username || url.password || url.search || url.hash || !secret) throw new Error('Invalid budget RPC endpoint');
  }
  private async post(action: string, body: object): Promise<unknown> {
    const response = await fetch(`${this.base.replace(/\/$/,'')}/internal/budget/${action}`,{ method: 'POST',headers: { authorization: `Bearer ${this.secret}`,'content-type': 'application/json' },body: JSON.stringify(body),signal: AbortSignal.timeout(2000) });
    if (!response.ok) throw new Error('workflow_budget_unavailable_or_exhausted');
    return (await response.json() as { result?: unknown }).result;
  }
  async reserve(workflow: string, id: string, amount: number, execution?: string) { await this.post('reserve',{ workflow,id,amount,execution }); }
  async settle(id: string, amount: number | null) { await this.post('settle',{ id,amount }); }
  async active(execution: string): Promise<boolean> { return await this.post('active',{ execution }) === true; }
  async firstModel(execution: string, now = Date.now()) { const value = await this.post('first-model',{ execution,now }); return typeof value === 'number' ? value : undefined; }
}
