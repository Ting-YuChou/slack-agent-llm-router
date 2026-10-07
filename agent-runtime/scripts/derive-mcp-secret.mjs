import { hkdfSync } from "node:crypto";

const server = process.argv[2];
if (!["github", "clickhouse", "context7"].includes(server)) {
  console.error("Usage: node scripts/derive-mcp-secret.mjs <github|clickhouse|context7>");
  process.exit(2);
}
const root = process.env.MCP_GATEWAY_SIGNING_SECRET;
if (!root) {
  console.error("MCP_GATEWAY_SIGNING_SECRET is required");
  process.exit(2);
}
process.stdout.write(Buffer.from(hkdfSync("sha256", root, "slack-pi-mcp-v2", server, 32)).toString("base64url"));
