import { readFileSync } from "node:fs";

import { GitHubAppTokenProvider } from "./github-app-token.js";
import { GITHUB_READ_ONLY_TOOLS } from "./mcp-config.js";
import { createMcpGatewayServer } from "./mcp-gateway.js";

const signingSecret = process.env.MCP_GATEWAY_SIGNING_SECRET ?? "";
const upstreamUrl = process.env.GITHUB_MCP_UPSTREAM_URL ?? "http://github-mcp:8082/mcp";
const host = process.env.MCP_GATEWAY_HOST ?? "0.0.0.0";
const port = positiveInteger(process.env.MCP_GATEWAY_PORT, 8090);
if (!signingSecret) throw new Error("MCP_GATEWAY_SIGNING_SECRET is required");
const privateKey = readPrivateKey(process.env.GITHUB_APP_PRIVATE_KEY_PATH);
const githubToken = new GitHubAppTokenProvider({
  appId: process.env.GITHUB_APP_ID ?? "",
  installationId: process.env.GITHUB_APP_INSTALLATION_ID ?? "",
  privateKey,
});
const githubUpstreamHeaders = async () => ({
  Authorization: `Bearer ${await githubToken.getToken()}`,
  "X-MCP-Toolsets": "repos,issues,pull_requests",
  "X-MCP-Tools": GITHUB_READ_ONLY_TOOLS.join(","),
  "X-MCP-Readonly": "true",
  "X-MCP-Lockdown": "true",
});

const server = createMcpGatewayServer({
  signingSecret,
  upstreamUrl,
  timeoutMs: positiveInteger(process.env.MCP_GATEWAY_TIMEOUT_MS, 30_000),
  maxResultBytes: positiveInteger(process.env.MCP_GATEWAY_MAX_RESULT_BYTES, 1_048_576),
  upstreamHeaders: githubUpstreamHeaders,
  healthCheck: async () => {
    try {
      const response = await fetch(upstreamUrl, {
        method: "GET",
        headers: await githubUpstreamHeaders(),
        signal: AbortSignal.timeout(1_500),
      });
      await response.body?.cancel();
      return true;
    } catch {
      return false;
    }
  },
  onAudit: (event) => console.log(JSON.stringify({ event: "mcp_call", ...event })),
});
server.listen(port, host, () => console.log(JSON.stringify({ event: "mcp_gateway_started", host, port, server: "github" })));

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.on(signal, () => server.close(() => process.exit(0)));
}

function positiveInteger(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function readPrivateKey(filePath: string | undefined): string {
  if (!filePath) return "";
  try { return readFileSync(filePath, "utf8"); } catch { throw new Error("GitHub App private key could not be read"); }
}
