import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

import { buildMcpRegistration } from "../dist/src/mcp-config.js";

export default function mcpBootstrapExtension(pi: ExtensionAPI) {
  const mode = process.env.PI_AGENT_MCP_MODE;
  if (mode !== "github_read_only") throw new Error("Trusted MCP bootstrap requires github_read_only mode");
  let tools: unknown;
  try {
    tools = JSON.parse(process.env.PI_AGENT_MCP_TOOLS_JSON ?? "");
  } catch {
    throw new Error("PI_AGENT_MCP_TOOLS_JSON must be valid JSON");
  }
  if (!Array.isArray(tools) || tools.some((tool) => typeof tool !== "string")) {
    throw new Error("PI_AGENT_MCP_TOOLS_JSON must contain tool names");
  }
  pi.registerMcpServer("github", buildMcpRegistration({
    mode,
    gatewayUrl: process.env.PI_AGENT_MCP_GATEWAY_URL ?? "",
    token: process.env.PI_AGENT_MCP_TOKEN ?? "",
    tools,
  }));
}
