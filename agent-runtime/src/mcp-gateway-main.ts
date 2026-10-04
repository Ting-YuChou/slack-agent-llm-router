import { createMcpGatewayServer } from "./mcp-gateway.js";

const signingSecret = process.env.MCP_GATEWAY_SIGNING_SECRET ?? "";
const upstreamUrl = process.env.GITHUB_MCP_UPSTREAM_URL ?? "http://127.0.0.1:8091/mcp";
const host = process.env.MCP_GATEWAY_HOST ?? "0.0.0.0";
const port = positiveInteger(process.env.MCP_GATEWAY_PORT, 8090);
if (!signingSecret) throw new Error("MCP_GATEWAY_SIGNING_SECRET is required");

const server = createMcpGatewayServer({
  signingSecret,
  upstreamUrl,
  timeoutMs: positiveInteger(process.env.MCP_GATEWAY_TIMEOUT_MS, 30_000),
  maxResultBytes: positiveInteger(process.env.MCP_GATEWAY_MAX_RESULT_BYTES, 1_048_576),
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
