import { createHmac, timingSafeEqual } from "node:crypto";

export type McpMode = "off" | "github_read_only";

export interface McpClaims {
  run: string;
  session: string;
  slackUser: string;
  repository: string;
  server: "github";
  tools: string[];
  mode: "github_read_only";
  expiry: number;
}

export function issueMcpToken(claims: McpClaims, secret: string): string {
  if (!secret) throw new Error("MCP gateway signing secret is required");
  const payload = Buffer.from(JSON.stringify(claims)).toString("base64url");
  const signature = createHmac("sha256", secret).update(payload).digest("base64url");
  return `${payload}.${signature}`;
}

export function verifyMcpToken(token: string, secret: string, now = Date.now()): McpClaims | null {
  const [payload, signature, extra] = token.split(".");
  if (!payload || !signature || extra || !secret) return null;
  const expected = createHmac("sha256", secret).update(payload).digest();
  let provided: Buffer;
  try {
    provided = Buffer.from(signature, "base64url");
  } catch {
    return null;
  }
  if (provided.length !== expected.length || !timingSafeEqual(provided, expected)) return null;
  let value: unknown;
  try {
    value = JSON.parse(Buffer.from(payload, "base64url").toString("utf8"));
  } catch {
    return null;
  }
  if (!isRecord(value) ||
    typeof value.run !== "string" || !value.run ||
    typeof value.session !== "string" || !value.session ||
    typeof value.slackUser !== "string" || !value.slackUser ||
    typeof value.repository !== "string" || !validRepository(value.repository) ||
    value.server !== "github" || value.mode !== "github_read_only" ||
    typeof value.expiry !== "number" || value.expiry < now ||
    !Array.isArray(value.tools) || value.tools.length === 0 ||
    value.tools.some((tool) => typeof tool !== "string" || !tool) ||
    new Set(value.tools).size !== value.tools.length
  ) return null;
  return {
    run: value.run,
    session: value.session,
    slackUser: value.slackUser,
    repository: value.repository,
    server: "github",
    tools: [...value.tools] as string[],
    mode: "github_read_only",
    expiry: value.expiry,
  };
}

function validRepository(value: string): boolean {
  return /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
