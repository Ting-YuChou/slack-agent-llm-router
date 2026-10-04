import type { McpMode } from "./mcp-token.js";

export interface McpRunConfig {
  mode: "github_read_only";
  gatewayUrl: string;
  token: string;
  tools: string[];
}

export interface McpRegistration {
  url: string;
  headers: { Authorization: string };
  exposure: "hidden";
  toolExposure: Record<string, "direct">;
  timeout: number;
}

export function parseMcpMode(value: string | undefined): McpMode {
  if (value === undefined || value === "" || value === "off") return "off";
  if (value === "github_read_only") return value;
  if (value === "approved_write") throw new Error("PI_AGENT_MCP_MODE=approved_write is not enabled in this release");
  throw new Error("PI_AGENT_MCP_MODE must be off, github_read_only, or approved_write");
}

export function buildMcpRegistration(config: McpRunConfig): McpRegistration {
  let url: URL;
  try {
    url = new URL(config.gatewayUrl);
  } catch {
    throw new Error("MCP gateway URL is invalid");
  }
  if (url.protocol !== "http:" || url.hostname !== "mcp-gateway" || url.port !== "8090" ||
      url.pathname !== "/mcp" || url.search || url.hash || url.username || url.password) {
    throw new Error("MCP gateway URL must be http://mcp-gateway:8090/mcp");
  }
  if (!config.token) throw new Error("MCP gateway token is required");
  if (config.mode !== "github_read_only" || config.tools.length === 0 || new Set(config.tools).size !== config.tools.length) {
    throw new Error("MCP gateway tool allowlist is invalid");
  }
  return {
    url: url.toString(),
    headers: { Authorization: `Bearer ${config.token}` },
    exposure: "hidden",
    toolExposure: Object.fromEntries(config.tools.map((tool) => [tool, "direct"])),
    timeout: 30_000,
  };
}
