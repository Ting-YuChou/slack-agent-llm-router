import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { buildMcpRegistrations, parseMcpMode } from "../dist/src/mcp-config.js";

export default function mcpBootstrapExtension(pi: ExtensionAPI) {
  const mode = parseMcpMode(process.env.PI_AGENT_MCP_MODE);
  if (mode === "off") throw new Error("Trusted MCP bootstrap requires read-only mode");
  let servers: unknown;
  try { servers = JSON.parse(process.env.PI_AGENT_MCP_SERVERS_JSON ?? ""); }
  catch { throw new Error("PI_AGENT_MCP_SERVERS_JSON must be valid JSON"); }
  if (!Array.isArray(servers) || servers.length === 0) throw new Error("PI_AGENT_MCP_SERVERS_JSON must contain servers");
  for (const [server, registration] of Object.entries(buildMcpRegistrations(servers as any))) {
    pi.registerMcpServer(server, registration);
  }
}
