import { readFileSync } from "node:fs";
import { GitHubAppTokenProvider } from "./github-app-token.js";
import { allServerTools, MCP_SERVERS, type McpServerId } from "./mcp-config.js";
import { createMcpGatewayServer } from "./mcp-gateway.js";
import { authorizeClickHouseTool, authorizeContext7Tool, authorizeGitHubTool, sanitizeClickHouseResponse, validateContext7Response } from "./mcp-policy.js";

const serverId = parseServer(process.env.MCP_SERVER_ID);
const signingSecret = process.env.MCP_GATEWAY_SERVER_SIGNING_SECRET ?? "";
if (!signingSecret) throw new Error("MCP_GATEWAY_SERVER_SIGNING_SECRET is required");
const upstreamUrl = process.env.MCP_UPSTREAM_URL ?? defaultUpstream(serverId);
const host = process.env.MCP_GATEWAY_HOST ?? "0.0.0.0";
const port = positiveInteger(process.env.MCP_GATEWAY_PORT, 8090);
const githubToken = serverId === "github" ? new GitHubAppTokenProvider({
  appId: process.env.GITHUB_APP_ID ?? "", installationId: process.env.GITHUB_APP_INSTALLATION_ID ?? "",
  privateKey: readPrivateKey(process.env.GITHUB_APP_PRIVATE_KEY_PATH),
}) : undefined;

const upstreamHeaders = async (): Promise<Record<string, string>> => {
  if (serverId === "github") return {
    Authorization: `Bearer ${await githubToken!.getToken()}`,
    "X-MCP-Toolsets": "repos,issues,pull_requests,actions,code_security,dependabot,secret_protection",
    "X-MCP-Tools": allServerTools("github").join(","), "X-MCP-Readonly": "true", "X-MCP-Lockdown": "true",
  };
  if (serverId === "context7") return { Authorization: `Bearer ${required("CONTEXT7_API_KEY")}` };
  return { Authorization: `Bearer ${required("CLICKHOUSE_MCP_BEARER_TOKEN")}` };
};

const gateway = createMcpGatewayServer({
  server: serverId, allowedTools: allServerTools(serverId), signingSecret, upstreamUrl,
  timeoutMs: positiveInteger(process.env.MCP_GATEWAY_TIMEOUT_MS, serverId === "github" ? 30_000 : 10_000),
  maxRequestBytes: positiveInteger(process.env.MCP_GATEWAY_MAX_REQUEST_BYTES, serverId === "context7" || serverId === "clickhouse" ? 16_384 : 256_000),
  maxResultBytes: positiveInteger(process.env.MCP_GATEWAY_MAX_RESULT_BYTES, MCP_SERVERS[serverId].maxResultBytes),
  upstreamHeaders,
  authorizeTool: serverId === "github" ? authorizeGitHubTool : serverId === "clickhouse" ? authorizeClickHouseTool : authorizeContext7Tool,
  transformResponse: serverId === "clickhouse" ? sanitizeClickHouseResponse : serverId === "context7" ? validateContext7Response : undefined,
  healthCheck: async () => {
    try {
      const response = await fetch(upstreamUrl, { method: "GET", headers: await upstreamHeaders(), signal: AbortSignal.timeout(1_500) });
      await response.body?.cancel(); return response.status < 500;
    } catch { return false; }
  },
  onAudit: (event) => console.log(JSON.stringify({ event: "mcp_call", ...event })),
});
gateway.listen(port, host, () => console.log(JSON.stringify({ event: "mcp_gateway_started", host, port, server: serverId })));
for (const signal of ["SIGINT", "SIGTERM"] as const) process.on(signal, () => gateway.close(() => process.exit(0)));

function parseServer(value: string | undefined): Exclude<McpServerId, "codegraph"> {
  if (value === "github" || value === "clickhouse" || value === "context7") return value;
  throw new Error("MCP_SERVER_ID must be github, clickhouse, or context7");
}
function defaultUpstream(server: Exclude<McpServerId, "codegraph">): string {
  return server === "github" ? "http://github-mcp:8082/mcp" : server === "clickhouse" ? "http://mcp-clickhouse:8000/mcp" : "https://mcp.context7.com/mcp";
}
function required(name: string): string { const value = process.env[name]; if (!value) throw new Error(`${name} is required`); return value; }
function positiveInteger(value: string | undefined, fallback: number): number { const parsed = Number(value); return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback; }
function readPrivateKey(filePath: string | undefined): string {
  if (!filePath) return "";
  try { return readFileSync(filePath, "utf8"); } catch { throw new Error("GitHub App private key could not be read"); }
}
