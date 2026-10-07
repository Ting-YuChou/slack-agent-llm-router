import { createHmac, hkdfSync, timingSafeEqual } from "node:crypto";
import type { McpServerId } from "./mcp-config.js";

export interface McpClaims {
  version: 2;
  run: string;
  session: string;
  slackUser: string;
  server: McpServerId;
  mode: "read_only";
  scope: Record<string, string>;
  tools: string[];
  maxCalls: number;
  expiry: number;
}

export function deriveMcpServerSecret(rootSecret: string, server: McpServerId): string {
  if (!rootSecret) throw new Error("MCP gateway signing secret is required");
  return Buffer.from(hkdfSync("sha256", rootSecret, "slack-pi-mcp-v2", server, 32)).toString("base64url");
}

export function issueMcpToken(claims: McpClaims, secret: string): string {
  if (!secret) throw new Error("MCP gateway signing secret is required");
  const payload = Buffer.from(JSON.stringify(claims)).toString("base64url");
  const signature = createHmac("sha256", secret).update(payload).digest("base64url");
  return `${payload}.${signature}`;
}

export function verifyMcpToken(token: string, secret: string, expectedServer: McpServerId, now = Date.now()): McpClaims | null {
  const [payload, signature, extra] = token.split(".");
  if (!payload || !signature || extra || !secret) return null;
  const expected = createHmac("sha256", secret).update(payload).digest();
  let provided: Buffer;
  try { provided = Buffer.from(signature, "base64url"); } catch { return null; }
  if (provided.length !== expected.length || !timingSafeEqual(provided, expected)) return null;
  let value: unknown;
  try { value = JSON.parse(Buffer.from(payload, "base64url").toString("utf8")); } catch { return null; }
  if (!isRecord(value) || value.version !== 2 || value.server !== expectedServer || value.mode !== "read_only" ||
      typeof value.run !== "string" || !value.run || typeof value.session !== "string" || !value.session ||
      typeof value.slackUser !== "string" || !value.slackUser || typeof value.expiry !== "number" || value.expiry < now ||
      !Number.isInteger(value.maxCalls) || (value.maxCalls as number) <= 0 ||
      !Array.isArray(value.tools) || value.tools.length === 0 || value.tools.some((tool) => typeof tool !== "string" || !tool) ||
      new Set(value.tools).size !== value.tools.length || !validScope(value.server as McpServerId, value.scope)) return null;
  return value as unknown as McpClaims;
}

function validScope(server: McpServerId, scope: unknown): scope is Record<string, string> {
  if (!isRecord(scope) || Object.values(scope).some((value) => typeof value !== "string")) return false;
  if (server === "github") return typeof scope.repository === "string" && /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(scope.repository);
  if (server === "clickhouse") return scope.database === "agent_mcp";
  if (server === "context7") return scope.data === "public_docs";
  return server === "codegraph" && scope.workspace === "run_mirror";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
