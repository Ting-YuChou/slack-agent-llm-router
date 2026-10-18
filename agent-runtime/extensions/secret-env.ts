import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { createBashTool } from "@earendil-works/pi-coding-agent";

const SECRET_NAME = /(?:KEY|TOKEN|SECRET|PASSWORD|PASSPHRASE|CREDENTIALS?)$/i;
const SECRET_NAMES = new Set(["PI_AGENT_MCP_SERVERS_JSON"]);

/** Remove gateway, provider and MCP credentials before any agent-authored command runs. */
export function scrubSecretEnv(env: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  return Object.fromEntries(Object.entries(env).filter(([name]) => !SECRET_NAMES.has(name) && !SECRET_NAME.test(name)));
}

/**
 * Replaces the built-in bash tool (extension tools override built-ins by name) with one whose
 * children never inherit the run's credentials. Policy hooks still apply to it by tool name.
 */
export default function secretEnvExtension(pi: ExtensionAPI) {
  const bash = createBashTool(process.cwd(), {
    spawnHook: ({ command, cwd, env }) => ({ command, cwd, env: scrubSecretEnv(env) }),
  });
  pi.registerTool({
    ...bash,
    execute: async (id, params, signal, onUpdate) => bash.execute(id, params, signal, onUpdate),
  });
}
