import { hkdfSync } from "node:crypto";

/**
 * The MCP gateways' bearer for the host's execution-status endpoint, derived from each server's
 * signing secret: it can only ask whether a run is active, never sign an MCP token.
 */
export function deriveMcpRevocationSecret(serverSecret: string): string {
  if (!serverSecret) throw new Error("MCP gateway signing secret is required");
  return Buffer.from(hkdfSync("sha256", serverSecret, "slack-pi-mcp-revocation-v1", "execution-status", 32)).toString("base64url");
}

/** Asks the host whether an execution's run is still active (MCP gateways). */
export class ExecutionStatusClient {
  constructor(private readonly base: string, private readonly secret: string) {
    const url = new URL(base);
    if (url.protocol !== "http:" || !["127.0.0.1", "localhost", "host.docker.internal"].includes(url.hostname) || url.username || url.password || url.search || url.hash || !secret) throw new Error("Invalid execution status endpoint");
  }

  async active(execution: string): Promise<boolean> {
    const response = await fetch(`${this.base.replace(/\/$/, "")}/internal/mcp/active`, {
      method: "POST", headers: { authorization: `Bearer ${this.secret}`, "content-type": "application/json" },
      body: JSON.stringify({ execution }), signal: AbortSignal.timeout(2000),
    });
    if (!response.ok) throw new Error("execution_status_unavailable");
    return (await response.json() as { result?: unknown }).result === true;
  }
}

/** The client for an MCP gateway whose host is at PI_AGENT_REVOCATION_RPC_URL; none when it is unset. */
export function executionStatusFromEnv(env: NodeJS.ProcessEnv, serverSecret: string): ExecutionStatusClient | undefined {
  const url = env.PI_AGENT_REVOCATION_RPC_URL;
  return url ? new ExecutionStatusClient(url, deriveMcpRevocationSecret(serverSecret)) : undefined;
}
