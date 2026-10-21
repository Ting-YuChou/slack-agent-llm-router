/**
 * The gateway extension's per-prompt credential query. The host answers it once per prompt with the
 * tokens it issued for that execution, so a token never sits in the container's environment.
 */
export const CREDENTIAL_DIALOG = "pi-agent/credential";

/**
 * What the container's environment carries instead of a token. Pi treats a provider as configured
 * only while its key variable is set, so the variable must exist before the first prompt.
 */
export const CREDENTIAL_PLACEHOLDER = "pi-gateway-credential-pending";

export const CREDENTIAL_NAMES = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENCODE_API_KEY", "OPENROUTER_API_KEY"] as const;
export type CredentialName = typeof CREDENTIAL_NAMES[number];
export type RunCredentials = Partial<Record<CredentialName, string>>;

const MAX_CREDENTIAL_LENGTH = 8192;

/** The host's reply as credentials to install; anything malformed installs nothing. */
export function parseCredentials(reply: string | undefined): RunCredentials | undefined {
  if (typeof reply !== "string" || reply.length > 4 * MAX_CREDENTIAL_LENGTH) return undefined;
  let parsed: unknown;
  try { parsed = JSON.parse(reply); } catch { return undefined; }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return undefined;
  const entries = Object.entries(parsed);
  if (!entries.length) return undefined;
  const credentials: RunCredentials = {};
  for (const [name, value] of entries) {
    if (!(CREDENTIAL_NAMES as readonly string[]).includes(name)) return undefined;
    if (typeof value !== "string" || !value || value.length > MAX_CREDENTIAL_LENGTH) return undefined;
    credentials[name as CredentialName] = value;
  }
  return credentials;
}
